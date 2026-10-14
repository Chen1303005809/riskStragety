import { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import {
  decodeArtifactSource,
  fetchArtifact,
  fetchConversationForChat,
  fetchModels,
  fetchMonitorAccounts,
  fetchMonitorConversation,
  fetchMonitorConversations,
  parsePythonFile,
  streamChat,
} from './api';
import type {
  AccountSummary,
  Artifact,
  ArtifactRecord,
  ChatMessage,
  ConversationDetail,
  ConversationSummary,
  ModelOptions,
  MonitorConversationsPage,
  Page,
} from './types';

function createUuid(): string {
  const cryptoApi = globalThis.crypto;
  if (typeof cryptoApi.randomUUID === 'function') {
    return cryptoApi.randomUUID();
  }

  // randomUUID() is unavailable in insecure contexts such as HTTP on a LAN IP.
  // getRandomValues() remains available there, so use it to build an RFC 4122 v4 UUID.
  const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function RichText({ content }: { content: string }) {
  const parts = content.split(/(\x60{3}[\s\S]*?\x60{3})/g);
  return (
    <div className="rich-text">
      {parts.map((part, index) => {
        if (part.startsWith('\x60\x60\x60') && part.endsWith('\x60\x60\x60')) {
          const newline = part.indexOf('\n');
          const language = newline > 0 ? part.slice(3, newline).trim() : '';
          const source = newline > 0 ? part.slice(newline + 1, -3) : part.slice(3, -3);
          return (
            <div className="code-block" key={index}>
              <div className="code-caption">{language || '代码'}</div>
              <pre><code>{source}</code></pre>
            </div>
          );
        }
        return part ? <p key={index}>{part}</p> : null;
      })}
    </div>
  );
}

function activityText(phase: string, tool?: string): string {
  if (phase === 'started') return '正在理解需求…';
  if (phase === 'tool_finished') return '正在继续生成…';
  if (phase === 'tool_started') {
    if (tool === 'lookup_exchange_by_contract_code') return '正在查询交易所…';
    if (tool === 'list_interface_files') return '正在查看接口文件…';
    if (tool === 'read_interface_file') return '正在阅读接口源码…';
    if (tool === 'write_generated_file') return '正在保存策略文件…';
    return '正在使用工具…';
  }
  return '正在生成回复…';
}

function makeMessage(role: ChatMessage['role'], content: string, status: ChatMessage['status']): ChatMessage {
  return { id: createUuid(), role, content, status, artifacts: [] };
}

interface MonitorDashboardProps {
  apiKey: string;
  onApiKey: (key: string) => void;
  onContinue: (accountId: string, conversationId: string) => void;
  onPreview: (artifact: Artifact, accountId: string) => void;
  onDownload: (artifact: Artifact, accountId: string) => void;
  artifactBusy: string;
  notice: string;
}

function displayTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false });
}

function statusLabel(status: ConversationSummary['status']): string {
  if (status === 'active') return '进行中';
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  if (status === 'interrupted') return '已中断';
  return '未开始';
}

function MonitorDashboard({
  apiKey,
  onApiKey,
  onContinue,
  onPreview,
  onDownload,
  artifactBusy,
  notice,
}: MonitorDashboardProps) {
  const [keyDraft, setKeyDraft] = useState('');
  const [checkingKey, setCheckingKey] = useState(false);
  const [error, setError] = useState('');
  const [accounts, setAccounts] = useState<Page<AccountSummary> | null>(null);
  const [accountOffset, setAccountOffset] = useState(0);
  const [selectedAccountId, setSelectedAccountId] = useState('');
  const [conversations, setConversations] = useState<MonitorConversationsPage | null>(null);
  const [conversationOffset, setConversationOffset] = useState(0);
  const [unassignedOffset, setUnassignedOffset] = useState(0);
  const [selectedConversationId, setSelectedConversationId] = useState('');
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [messageOffset, setMessageOffset] = useState(0);
  const [artifactOffset, setArtifactOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshTick, setRefreshTick] = useState(0);

  async function connectMonitor(event: FormEvent) {
    event.preventDefault();
    const key = keyDraft.trim();
    if (!key) {
      setError('请输入 API Key');
      return;
    }
    setCheckingKey(true);
    setError('');
    try {
      await fetchMonitorAccounts(key, 0, 1);
      onApiKey(key);
      setKeyDraft('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '无法读取监控数据');
    } finally {
      setCheckingKey(false);
    }
  }

  useEffect(() => {
    if (!apiKey) return undefined;
    let current = true;

    const load = async (showProgress: boolean) => {
      if (document.visibilityState === 'hidden') return;
      if (showProgress) setLoading(true);
      setRefreshing(true);
      setError('');
      try {
        const accountPage = await fetchMonitorAccounts(apiKey, accountOffset, 50);
        if (!current) return;
        setAccounts(accountPage);
        const accountExists = accountPage.items.some((item) => item.account_id === selectedAccountId);
        const nextAccountId = accountExists ? selectedAccountId : (accountPage.items[0]?.account_id || '');
        if (nextAccountId !== selectedAccountId) {
          setSelectedAccountId(nextAccountId);
          setSelectedConversationId('');
          setDetail(null);
          setConversationOffset(0);
          return;
        }
        if (!nextAccountId) {
          setConversations(null);
          setSelectedConversationId('');
          setDetail(null);
          return;
        }

        const conversationPage = await fetchMonitorConversations(apiKey, nextAccountId, {
          offset: conversationOffset,
          limit: 50,
          unassignedOffset,
          unassignedLimit: 20,
        });
        if (!current) return;
        setConversations(conversationPage);
        const conversationExists = conversationPage.items.some(
          (item) => item.conversation_id === selectedConversationId,
        );
        const nextConversationId = conversationExists
          ? selectedConversationId
          : (conversationPage.items[0]?.conversation_id || '');
        if (nextConversationId !== selectedConversationId) {
          setSelectedConversationId(nextConversationId);
          setDetail(null);
          setMessageOffset(0);
          setArtifactOffset(0);
          return;
        }
        if (!nextConversationId) {
          setDetail(null);
          return;
        }

        const conversationDetail = await fetchMonitorConversation(apiKey, nextAccountId, nextConversationId, {
          messageOffset,
          messageLimit: 100,
          artifactOffset,
          artifactLimit: 20,
        });
        if (current) setDetail(conversationDetail);
      } catch (reason) {
        if (current) setError(reason instanceof Error ? reason.message : '读取监控数据失败');
      } finally {
        if (current) {
          setLoading(false);
          setRefreshing(false);
        }
      }
    };

    void load(true);
    const interval = window.setInterval(() => {
      if (document.visibilityState !== 'hidden') void load(false);
    }, 15_000);
    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') void load(false);
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      current = false;
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [
    apiKey,
    accountOffset,
    selectedAccountId,
    conversationOffset,
    unassignedOffset,
    selectedConversationId,
    messageOffset,
    artifactOffset,
    refreshTick,
  ]);

  function selectAccount(accountId: string) {
    setSelectedAccountId(accountId);
    setSelectedConversationId('');
    setDetail(null);
    setConversationOffset(0);
    setUnassignedOffset(0);
    setMessageOffset(0);
    setArtifactOffset(0);
  }

  function selectConversation(conversationId: string) {
    setSelectedConversationId(conversationId);
    setDetail(null);
    setMessageOffset(0);
    setArtifactOffset(0);
  }

  function fileActions(file: ArtifactRecord, ownerAccountId: string) {
    return (
      <div className="monitor-file-row" key={file.artifact_id}>
        <div className="monitor-file-mark">.py</div>
        <div className="monitor-file-copy">
          <strong>{file.filename}</strong>
          <small>{displayTime(file.created_at)}</small>
        </div>
        <button type="button" onClick={() => onPreview(file, ownerAccountId)} disabled={artifactBusy === file.artifact_id}>预览</button>
        <button type="button" onClick={() => onDownload(file, ownerAccountId)} disabled={artifactBusy === file.artifact_id}>下载</button>
      </div>
    );
  }

  if (!apiKey) {
    return (
      <main className="main-area monitor-area">
        <div className="monitor-heading">
          <div><span className="eyebrow">MONITOR / OVERVIEW</span><h1>账号运行监控</h1><p>使用服务 API Key 浏览全部账号的会话与生成文件。</p></div>
          <span className="monitor-state"><span className="monitor-state-dot" />等待连接</span>
        </div>
        <section className="connection-card monitor-connection-card">
          <div className="connection-icon">◉</div>
          <span className="eyebrow">连接监控</span>
          <h2>输入 API Key 读取数据</h2>
          <p>API Key 只保留在当前页面内存。持有共享 Key 的调用方可以查看所有账号的监控信息。</p>
          <form onSubmit={connectMonitor} className="connection-form">
            <input type="password" value={keyDraft} onChange={(event) => setKeyDraft(event.target.value)} placeholder="输入 AGENT_API_KEY" autoComplete="off" aria-label="监控 API Key" required />
            <button type="submit" disabled={checkingKey}>{checkingKey ? '连接中…' : '连接并读取监控'}</button>
          </form>
          {error && <p className="inline-error" role="alert">{error}</p>}
        </section>
      </main>
    );
  }

  const accountItems = accounts?.items || [];
  const conversationItems = conversations?.items || [];
  const selectedConversation = conversationItems.find((item) => item.conversation_id === selectedConversationId);

  return (
    <main className="main-area monitor-area">
      <div className="monitor-heading">
        <div>
          <span className="eyebrow">MONITOR / OVERVIEW</span>
          <h1>账号运行监控</h1>
          <p>集中查看账号对话、会话活动和生成文件。页面可见时每 15 秒自动刷新。</p>
        </div>
        <button className="monitor-refresh" type="button" onClick={() => setRefreshTick((value) => value + 1)} disabled={refreshing}>
          <span className={refreshing ? 'refresh-glyph spinning' : 'refresh-glyph'} aria-hidden="true">↻</span>
          {refreshing ? '刷新中…' : '立即刷新'}
        </button>
      </div>

      <section className="monitor-board" aria-label="账号与会话监控">
        <aside className="monitor-index">
          <section className="monitor-card monitor-list-card" aria-labelledby="monitor-accounts-title">
            <div className="monitor-card-heading">
              <div><span className="eyebrow">ACCOUNTS</span><h2 id="monitor-accounts-title">账号 <small>{accounts?.total ?? '—'}</small></h2></div>
              <span className="monitor-state-tag">更新时间优先</span>
            </div>
            {loading && !accounts ? <div className="monitor-empty compact"><strong>正在读取账号…</strong></div> : accountItems.length ? (
              <div className="monitor-list">
                {accountItems.map((account) => (
                  <button className={account.account_id === selectedAccountId ? 'monitor-list-item selected' : 'monitor-list-item'} type="button" key={account.account_id} onClick={() => selectAccount(account.account_id)} aria-pressed={account.account_id === selectedAccountId}>
                    <strong title={account.account_id}>{account.account_id}</strong>
                    <span>{account.conversation_count} 个会话 · {account.artifact_count} 个文件</span>
                    <small>更新于 {displayTime(account.updated_at)}</small>
                  </button>
                ))}
                <div className="monitor-pagination">
                  <button type="button" onClick={() => setAccountOffset((offset) => Math.max(0, offset - 50))} disabled={accountOffset === 0}>上一页</button>
                  <span>{accountOffset + 1}–{Math.min(accountOffset + accountItems.length, accounts?.total || 0)} / {accounts?.total || 0}</span>
                  <button type="button" onClick={() => setAccountOffset((offset) => offset + 50)} disabled={accountOffset + 50 >= (accounts?.total || 0)}>下一页</button>
                </div>
              </div>
            ) : <div className="monitor-empty compact"><span className="monitor-empty-icon" aria-hidden="true">◌</span><strong>暂无账号数据</strong><p>首次成功聊天后，账号会显示在这里。</p></div>}
          </section>

          <section className="monitor-card monitor-list-card" aria-labelledby="monitor-sessions-title">
            <div className="monitor-card-heading">
              <div><span className="eyebrow">SESSIONS</span><h2 id="monitor-sessions-title">会话 <small>{conversations?.total ?? '—'}</small></h2></div>
              <span className="monitor-state-tag" title={selectedAccountId}>{selectedAccountId || '选择账号'}</span>
            </div>
            {selectedAccountId && conversations ? (
              <div className="monitor-list">
                {conversationItems.map((conversation) => (
                  <button className={conversation.conversation_id === selectedConversationId ? 'monitor-list-item selected' : 'monitor-list-item'} type="button" key={conversation.conversation_id} onClick={() => selectConversation(conversation.conversation_id)} aria-pressed={conversation.conversation_id === selectedConversationId}>
                    <strong>{conversation.title || '未命名会话'}</strong>
                    <span>{statusLabel(conversation.status)} · {conversation.message_count} 条消息 · {conversation.artifact_count} 个文件</span>
                    <small>{displayTime(conversation.updated_at)}</small>
                  </button>
                ))}
                {conversationItems.length > 0 && <div className="monitor-pagination">
                  <button type="button" onClick={() => setConversationOffset((offset) => Math.max(0, offset - 50))} disabled={conversationOffset === 0}>上一页</button>
                  <span>{conversationOffset + 1}–{Math.min(conversationOffset + conversationItems.length, conversations.total)} / {conversations.total}</span>
                  <button type="button" onClick={() => setConversationOffset((offset) => offset + 50)} disabled={conversationOffset + 50 >= conversations.total}>下一页</button>
                </div>}
                <div className="monitor-unassigned">
                  <div className="monitor-subheading"><strong>未关联会话文件</strong><span>{conversations.unassigned_total}</span></div>
                  {conversations.unassigned_artifacts.length ? conversations.unassigned_artifacts.map((file) => fileActions(file, selectedAccountId)) : <p className="monitor-muted">没有未关联文件</p>}
                  {conversations.unassigned_total > conversations.unassigned_limit && <div className="monitor-pagination compact-pagination">
                    <button type="button" onClick={() => setUnassignedOffset((offset) => Math.max(0, offset - conversations.unassigned_limit))} disabled={unassignedOffset === 0}>上一页文件</button>
                    <button type="button" onClick={() => setUnassignedOffset((offset) => offset + conversations.unassigned_limit)} disabled={unassignedOffset + conversations.unassigned_limit >= conversations.unassigned_total}>更多文件</button>
                  </div>}
                </div>
              </div>
            ) : loading ? <div className="monitor-empty compact"><strong>正在读取会话…</strong></div> : <div className="monitor-empty compact"><span className="monitor-empty-icon" aria-hidden="true">▤</span><strong>{selectedAccountId ? '暂无会话数据' : '暂无账号数据'}</strong><p>{selectedAccountId ? '这个账号还没有聊天记录。' : '首次聊天后即可浏览会话。'}</p></div>}
          </section>
        </aside>

        <div className="monitor-content">
          <section className="monitor-card monitor-detail-card" aria-labelledby="monitor-conversation-title">
            <div className="monitor-card-heading">
              <div><span className="eyebrow">CONVERSATION</span><h2 id="monitor-conversation-title">对话详情</h2></div>
              {selectedConversation && <button className="continue-button" type="button" onClick={() => onContinue(selectedAccountId, selectedConversation.conversation_id)}>在策略对话中继续</button>}
            </div>
            {detail ? <>
              <div className="monitor-detail-meta"><strong>{detail.conversation.title || '未命名会话'}</strong><span>{statusLabel(detail.conversation.status)} · {detail.message_total} 条消息</span></div>
              <div className="monitor-timeline">
                {detail.messages.map((message) => (
                  <article className={'monitor-message monitor-message-' + message.role} key={message.id}>
                    <div className="monitor-message-head"><strong>{message.role === 'user' ? '用户' : '策略 Agent'}</strong><span>{displayTime(message.created_at)}</span><em>{message.status === 'complete' ? '完成' : message.status === 'pending' ? '生成中' : message.status === 'failed' ? '失败' : '已中断'}</em></div>
                    <pre>{message.content || (message.status === 'pending' ? '正在生成…' : '（无回复文本）')}</pre>
                  </article>
                ))}
                {!detail.messages.length && <div className="monitor-empty compact"><strong>暂无消息</strong></div>}
              </div>
              <div className="monitor-pagination detail-pagination">
                <button type="button" onClick={() => setMessageOffset((offset) => Math.max(0, offset - 100))} disabled={messageOffset === 0}>较早消息</button>
                <span>{Math.min(messageOffset + 1, detail.message_total)}–{Math.min(messageOffset + detail.messages.length, detail.message_total)} / {detail.message_total}</span>
                <button type="button" onClick={() => setMessageOffset((offset) => offset + 100)} disabled={messageOffset + detail.message_limit >= detail.message_total}>较新消息</button>
              </div>
            </> : <div className="monitor-empty"><span className="monitor-empty-icon large" aria-hidden="true">☷</span><strong>{selectedConversation ? '正在读取对话…' : '选择一个会话'}</strong><p>{selectedConversation ? '消息会按时间顺序显示在这里。' : '选择账号和会话后可查看完整时间线。'}</p></div>}
          </section>

          <section className="monitor-card monitor-detail-card" aria-labelledby="monitor-files-title">
            <div className="monitor-card-heading">
              <div><span className="eyebrow">GENERATED FILES</span><h2 id="monitor-files-title">生成文件 <small>{detail?.artifact_total ?? selectedConversation?.artifact_count ?? 0}</small></h2></div>
              <span className="monitor-state-tag">{selectedConversation ? '当前会话' : '会话文件'}</span>
            </div>
            {detail ? <div className="monitor-files-list">
              {detail.artifacts.map((file) => fileActions(file, selectedAccountId))}
              {!detail.artifacts.length && <div className="monitor-empty"><span className="monitor-empty-icon large" aria-hidden="true">▧</span><strong>暂无文件记录</strong><p>这个会话没有生成文件。</p></div>}
              {detail.artifact_total > detail.artifact_limit && <div className="monitor-pagination">
                <button type="button" onClick={() => setArtifactOffset((offset) => Math.max(0, offset - 20))} disabled={artifactOffset === 0}>上一页</button>
                <span>{artifactOffset + 1}–{Math.min(artifactOffset + detail.artifacts.length, detail.artifact_total)} / {detail.artifact_total}</span>
                <button type="button" onClick={() => setArtifactOffset((offset) => offset + 20)} disabled={artifactOffset + detail.artifact_limit >= detail.artifact_total}>下一页</button>
              </div>}
            </div> : <div className="monitor-empty"><span className="monitor-empty-icon large" aria-hidden="true">▧</span><strong>{selectedConversation ? '正在读取文件…' : '暂无文件记录'}</strong><p>选择会话后可预览和下载关联文件。</p></div>}
          </section>
        </div>
      </section>
      {error && <div className="monitor-error" role="alert"><span>{error}</span><button type="button" onClick={() => setRefreshTick((value) => value + 1)}>重试</button></div>}
      {notice && <div className="monitor-error" role="alert">{notice}</div>}
      {loading && <p className="monitor-footnote">正在同步监控数据…</p>}
    </main>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState<'monitor' | 'chat'>('monitor');
  const [apiKey, setApiKey] = useState('');
  const [keyDraft, setKeyDraft] = useState('');
  const [accountId, setAccountId] = useState('');
  const [accountIdDraft, setAccountIdDraft] = useState('');
  const [conversationId, setConversationId] = useState(() => createUuid());
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState('');
  const [modelOptions, setModelOptions] = useState<ModelOptions>({
    temperature: 0.3,
    top_p: 0.8,
    num_ctx: 16_384,
  });
  const [think, setThink] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const messagesRef = useRef<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [activity, setActivity] = useState('');
  const [notice, setNotice] = useState('');
  const [preview, setPreview] = useState<{ artifact: Artifact; source: string; accountId: string } | null>(null);
  const [artifactBusy, setArtifactBusy] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const timelineEndRef = useRef<HTMLDivElement | null>(null);

  function commitMessages(next: ChatMessage[]) {
    messagesRef.current = next;
    setMessages(next);
  }

  function updateAssistant(id: string, change: Partial<ChatMessage>) {
    commitMessages(messagesRef.current.map((item) => item.id === id ? { ...item, ...change } : item));
  }

  function updateModelOption(key: keyof ModelOptions, raw: string) {
    if (!raw.trim()) return;
    const value = Number(raw);
    const valid = Number.isFinite(value) && (
      key === 'temperature' ? value >= 0 && value <= 2
        : key === 'top_p' ? value >= 0 && value <= 1
          : Number.isInteger(value) && value >= 512 && value <= 131_072
    );
    if (valid) setModelOptions((current) => ({ ...current, [key]: value }));
  }

  useEffect(() => {
    timelineEndRef.current?.scrollIntoView({ behavior: busy ? 'instant' : 'smooth', block: 'end' });
  }, [messages, busy]);

  async function connect(event: FormEvent) {
    event.preventDefault();
    const key = keyDraft.trim() || apiKey;
    const account = accountIdDraft.trim();
    if (!key) {
      setConnectionError('请输入 API Key');
      return;
    }
    if (!account) {
      setConnectionError('请输入账号 ID');
      return;
    }
    setConnecting(true);
    setConnectionError('');
    try {
      const { models: names, defaultModel } = await fetchModels(key);
      if (!names.length) throw new Error('Ollama 中没有已安装的模型');
      setApiKey(key);
      setKeyDraft('');
      setAccountId(account);
      setAccountIdDraft('');
      setModels(names);
      setModel(names.includes(defaultModel) ? defaultModel : names[0]);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : '连接失败');
    } finally {
      setConnecting(false);
    }
  }

  function disconnect() {
    abortRef.current?.abort();
    setApiKey('');
    setAccountId('');
    setAccountIdDraft('');
    setModels([]);
    setModel('');
    setConversationId(createUuid());
    setPreview(null);
    commitMessages([]);
    setNotice('');
  }

  function newConversation() {
    if (busy) return;
    commitMessages([]);
    setConversationId(createUuid());
    setPreview(null);
    setNotice('');
  }

  async function send(event?: FormEvent) {
    event?.preventDefault();
    const message = draft.trim();
    if (!apiKey || !model || !message || busy) return;
    const user = makeMessage('user', message, 'complete');
    const assistant = makeMessage('assistant', '', 'streaming');
    commitMessages([...messagesRef.current, user, assistant]);
    setDraft('');
    setBusy(true);
    setNotice('');
    setActivity('正在理解需求…');
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(apiKey, {
        account_id: accountId,
        conversation_id: conversationId,
        message,
        model,
        options: modelOptions,
        think,
      }, {
        onStatus: (phase, tool) => setActivity(activityText(phase, tool)),
        onDelta: (text) => {
          const current = messagesRef.current.find((item) => item.id === assistant.id);
          updateAssistant(assistant.id, { content: (current?.content || '') + text });
        },
        onArtifact: (artifact) => {
          const current = messagesRef.current.find((item) => item.id === assistant.id);
          const existing = current?.artifacts || [];
          if (!existing.some((item) => item.artifact_id === artifact.artifact_id)) {
            updateAssistant(assistant.id, { artifacts: [...existing, artifact] });
          }
        },
        onDone: (done) => {
          updateAssistant(assistant.id, {
            content: done.content,
            status: 'complete',
            artifacts: done.artifacts,
          });
        },
      }, controller.signal);
    } catch (error) {
      const stopped = controller.signal.aborted;
      const previousContent = messagesRef.current.find((item) => item.id === assistant.id)?.content || '';
      const failure = error instanceof Error ? error.message : '生成失败，请重试。';
      updateAssistant(assistant.id, {
        status: stopped ? 'stopped' : 'error',
        content: previousContent || (stopped ? '已停止本轮生成。' : '生成失败：' + failure),
      });
      if (!stopped) setNotice('生成失败：' + failure);
    } finally {
      abortRef.current = null;
      setBusy(false);
      setActivity('');
    }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send();
    }
  }

  async function openPreview(artifact: Artifact, ownerAccountId = accountId) {
    setArtifactBusy(artifact.artifact_id);
    setNotice('');
    try {
      const blob = await fetchArtifact(apiKey, artifact.artifact_id, ownerAccountId);
      setPreview({ artifact, source: decodeArtifactSource(await blob.arrayBuffer()), accountId: ownerAccountId });
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '无法读取生成文件');
    } finally {
      setArtifactBusy('');
    }
  }

  async function download(artifact: Artifact, ownerAccountId = accountId) {
    setArtifactBusy(artifact.artifact_id);
    setNotice('');
    try {
      const blob = await fetchArtifact(apiKey, artifact.artifact_id, ownerAccountId);
      const file = parsePythonFile(await blob.arrayBuffer(), artifact.filename);
      const url = URL.createObjectURL(file);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = file.name;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '下载失败');
    } finally {
      setArtifactBusy('');
    }
  }

  async function continueConversation(ownerAccountId: string, selectedConversationId: string) {
    setNotice('');
    try {
      let availableModels = models;
      let selectedModel = model;
      if (!availableModels.length) {
        const result = await fetchModels(apiKey);
        availableModels = result.models;
        if (!availableModels.length) throw new Error('Ollama 中没有已安装的模型');
        selectedModel = availableModels.includes(result.defaultModel) ? result.defaultModel : availableModels[0];
      }
      const restored = await fetchConversationForChat(apiKey, ownerAccountId, selectedConversationId);
      const restoredMessages: ChatMessage[] = restored.messages.map((item) => ({
        id: String(item.id),
        role: item.role,
        content: item.content,
        status: item.status === 'complete' ? 'complete' : item.status === 'interrupted' || item.status === 'pending' ? 'stopped' : 'error',
        artifacts: [],
      }));
      if (restored.artifacts.length) {
        let lastAssistant = -1;
        restoredMessages.forEach((item, index) => {
          if (item.role === 'assistant') lastAssistant = index;
        });
        if (lastAssistant >= 0) {
          restoredMessages[lastAssistant] = { ...restoredMessages[lastAssistant], artifacts: restored.artifacts };
        } else {
          restoredMessages.push({
            id: createUuid(),
            role: 'assistant',
            content: '本会话的生成文件',
            status: 'complete',
            artifacts: restored.artifacts,
          });
        }
      }
      setAccountId(ownerAccountId);
      setAccountIdDraft('');
      setModels(availableModels);
      setModel(selectedModel);
      setConversationId(selectedConversationId);
      commitMessages(restoredMessages);
      setPreview(null);
      setDraft('');
      setActivePage('chat');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '无法恢复会话');
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true"><span /></div>
          <div>
            <strong>风控策略 Agent</strong>
            <small>价格预警 · 代码生成</small>
          </div>
        </div>
        <nav className="primary-nav" aria-label="主导航">
          <button
            className={activePage === 'monitor' ? 'primary-nav-item active' : 'primary-nav-item'}
            type="button"
            aria-pressed={activePage === 'monitor'}
            onClick={() => setActivePage('monitor')}
          >监控总览</button>
          <button
            className={activePage === 'chat' ? 'primary-nav-item active' : 'primary-nav-item'}
            type="button"
            aria-pressed={activePage === 'chat'}
            onClick={() => setActivePage('chat')}
          >策略对话</button>
        </nav>
        <div className="topbar-actions">
          {apiKey && accountId && models.length > 0 && activePage === 'chat' ? (
            <>
              <span className="account-context" title={'账号 ID：' + accountId}>
                <small>ACCOUNT</small><strong>{accountId}</strong>
              </span>
              <label className="model-control">
                <span>模型</span>
                <select value={model} onChange={(event) => setModel(event.target.value)} disabled={busy}>
                  {models.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
              <details className="model-settings">
                <summary>模型参数</summary>
                <div className="model-settings-panel">
                  <strong>生成参数</strong>
                  <label className="model-setting-toggle">
                    <input
                      type="checkbox"
                      checked={think}
                      onChange={(event) => setThink(event.target.checked)}
                      disabled={busy}
                    />
                    <span>启用思考模式</span>
                  </label>
                  <label>
                    <span>上下文长度</span>
                    <input
                      aria-label="上下文长度"
                      type="number"
                      min={512}
                      max={131_072}
                      step={512}
                      value={modelOptions.num_ctx}
                      onChange={(event) => updateModelOption('num_ctx', event.target.value)}
                      disabled={busy}
                    />
                  </label>
                  <label>
                    <span>Temperature</span>
                    <input
                      aria-label="Temperature"
                      type="number"
                      min={0}
                      max={2}
                      step={0.05}
                      value={modelOptions.temperature}
                      onChange={(event) => updateModelOption('temperature', event.target.value)}
                      disabled={busy}
                    />
                  </label>
                  <label>
                    <span>Top P</span>
                    <input
                      aria-label="Top P"
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={modelOptions.top_p}
                      onChange={(event) => updateModelOption('top_p', event.target.value)}
                      disabled={busy}
                    />
                  </label>
                </div>
              </details>
              <span className="connected"><span className="connected-dot" /> 已连接</span>
              <button className="quiet-button" type="button" onClick={disconnect}>退出</button>
            </>
          ) : activePage === 'chat' ? <span className="offline">等待连接</span> : null}
        </div>
      </header>

      {activePage === 'monitor' ? <MonitorDashboard
        apiKey={apiKey}
        onApiKey={setApiKey}
        onContinue={(ownerAccountId, selectedConversationId) => void continueConversation(ownerAccountId, selectedConversationId)}
        onPreview={(artifact, ownerAccountId) => void openPreview(artifact, ownerAccountId)}
        onDownload={(artifact, ownerAccountId) => void download(artifact, ownerAccountId)}
        artifactBusy={artifactBusy}
        notice={notice}
      /> : <main className="main-area">
        <div className="workspace-heading">
          <div>
            <span className="eyebrow">WORKSPACE / 01</span>
            <h1>把风控规则，写成可用策略。</h1>
            <p>描述交易所、合约、价格条件和通知方式。信息不足时，Agent 会先向你确认。</p>
          </div>
          {apiKey && accountId && (
            <button className="new-chat-button" type="button" onClick={newConversation} disabled={busy}>新对话</button>
          )}
        </div>

        {!apiKey || !accountId || !models.length ? (
          <section className="connection-card">
            <div className="connection-icon">↗</div>
            <div>
              <span className="eyebrow">连接服务</span>
              <h2>输入 API Key 开始</h2>
              <p>密钥只保留在当前页面内存；账号 ID 会用于会话和文件的数据归属。</p>
            </div>
            <form onSubmit={connect} className="connection-form">
              <input
                type="text"
                value={accountIdDraft}
                onChange={(event) => setAccountIdDraft(event.target.value)}
                placeholder="输入账号 ID"
                autoComplete="off"
                aria-label="账号 ID"
                required
              />
              {apiKey ? <span className="connection-key-note">监控 API Key 已连接</span> : <input
                type="password"
                value={keyDraft}
                onChange={(event) => setKeyDraft(event.target.value)}
                placeholder="输入 AGENT_API_KEY"
                autoComplete="off"
                aria-label="API Key"
                required
              />}
              <button type="submit" disabled={connecting}>{connecting ? '连接中…' : '连接并读取模型'}</button>
            </form>
            {connectionError && <p className="inline-error" role="alert">{connectionError}</p>}
          </section>
        ) : (
          <section className="chat-panel" aria-label="策略对话">
            <div className="conversation">
              {messages.length === 0 ? (
                <div className="empty-conversation">
                  <div className="empty-symbol" aria-hidden="true">⌁</div>
                  <h2>从一条价格规则开始</h2>
                  <p>例如说明具体合约、交易所、阈值，以及超过、跌破或多步触发的顺序。</p>
                  <div className="empty-flow">
                    <span>01 描述需求</span><i /> <span>02 确认参数</span><i /> <span>03 下载策略</span>
                  </div>
                </div>
              ) : messages.map((item) => (
                <article className={'message message-' + item.role} key={item.id}>
                  <div className="message-avatar" aria-hidden="true">{item.role === 'user' ? '你' : 'A'}</div>
                  <div className="message-body">
                    <div className="message-label">{item.role === 'user' ? '你' : '策略 Agent'}</div>
                    {item.role === 'user' ? <p className="user-text">{item.content}</p> : (
                      <>
                        {item.content ? <RichText content={item.content} /> : <p className="stream-placeholder">{activity || '正在生成…'}</p>}
                        {item.status === 'streaming' && item.content && <p className="streaming-status"><span className="pulse-dot" />{activity || '正在生成…'}</p>}
                        {item.artifacts.map((artifact) => (
                          <div className="artifact-card" key={artifact.artifact_id}>
                            <div className="artifact-icon">.py</div>
                            <div className="artifact-details"><strong>{artifact.filename}</strong><small>已生成 · 可预览与下载</small></div>
                            <button type="button" onClick={() => void openPreview(artifact)} disabled={artifactBusy === artifact.artifact_id}>预览</button>
                            <button className="artifact-download" type="button" onClick={() => void download(artifact)} disabled={artifactBusy === artifact.artifact_id}>下载</button>
                          </div>
                        ))}
                      </>
                    )}
                  </div>
                </article>
              ))}
              <div ref={timelineEndRef} />
            </div>
            {notice && <p className="notice" role="alert">{notice}</p>}
            <form className="composer" onSubmit={(event) => void send(event)}>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={onComposerKeyDown}
                placeholder="描述预警规则，或继续调整上一版策略…"
                rows={3}
                disabled={busy}
                aria-label="输入策略需求"
              />
              <div className="composer-footer">
                <span>Enter 发送 · Shift + Enter 换行</span>
                {busy ? (
                  <button className="stop-button" type="button" onClick={() => abortRef.current?.abort()}>停止</button>
                ) : (
                  <button className="send-button" type="submit" disabled={!draft.trim()}>发送需求 <span aria-hidden="true">↗</span></button>
                )}
              </div>
            </form>
          </section>
        )}
      </main>}

      <footer className="page-footer"><span>RISK STRATEGY / AGENT</span><span>接口源码只读 · 生成文件独立保存</span></footer>

      {preview && (
        <div className="preview-backdrop" role="presentation" onClick={() => setPreview(null)}>
          <section className="preview-dialog" role="dialog" aria-modal="true" aria-label={'预览 ' + preview.artifact.filename} onClick={(event) => event.stopPropagation()}>
            <div className="preview-header">
              <div><span className="eyebrow">GENERATED FILE</span><h2>{preview.artifact.filename}</h2></div>
              <div className="preview-actions">
                <button type="button" onClick={() => void download(preview.artifact, preview.accountId)}>下载文件</button>
                <button className="close-button" type="button" onClick={() => setPreview(null)} aria-label="关闭预览">×</button>
              </div>
            </div>
            <pre className="preview-code"><code>{preview.source}</code></pre>
          </section>
        </div>
      )}
    </div>
  );
}
