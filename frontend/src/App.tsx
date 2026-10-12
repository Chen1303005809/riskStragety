import { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import { decodeArtifactSource, fetchArtifact, fetchModels, parsePythonFile, streamChat } from './api';
import type { Artifact, ChatMessage, ChatTurn, ModelOptions } from './types';

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
    if (tool === 'list_interface_files') return '正在查看接口文件…';
    if (tool === 'read_interface_file') return '正在阅读接口源码…';
    if (tool === 'write_generated_file') return '正在保存策略文件…';
    return '正在使用工具…';
  }
  return '正在生成回复…';
}

function makeMessage(role: ChatMessage['role'], content: string, status: ChatMessage['status']): ChatMessage {
  return { id: crypto.randomUUID(), role, content, status, artifacts: [] };
}

export default function App() {
  const [apiKey, setApiKey] = useState('');
  const [keyDraft, setKeyDraft] = useState('');
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
  const [preview, setPreview] = useState<{ artifact: Artifact; source: string } | null>(null);
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
    const key = keyDraft.trim();
    if (!key) {
      setConnectionError('请输入 API Key');
      return;
    }
    setConnecting(true);
    setConnectionError('');
    try {
      const { models: names, defaultModel } = await fetchModels(key);
      if (!names.length) throw new Error('Ollama 中没有已安装的模型');
      setApiKey(key);
      setKeyDraft('');
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
    setModels([]);
    setModel('');
    setPreview(null);
    commitMessages([]);
    setNotice('');
  }

  function newConversation() {
    if (busy) return;
    commitMessages([]);
    setPreview(null);
    setNotice('');
  }

  async function send(event?: FormEvent) {
    event?.preventDefault();
    const message = draft.trim();
    if (!apiKey || !model || !message || busy) return;
    const history: ChatTurn[] = messagesRef.current
      .filter((item) => item.status === 'complete' && item.content.trim())
      .map(({ role, content }) => ({ role, content }));
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
      await streamChat(apiKey, { message, history, model, options: modelOptions, think }, {
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
      updateAssistant(assistant.id, {
        status: stopped ? 'stopped' : 'error',
        content: stopped ? '已停止本轮生成。' : (
          error instanceof Error ? '生成失败：' + error.message : '生成失败，请重试。'
        ),
      });
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

  async function openPreview(artifact: Artifact) {
    setArtifactBusy(artifact.artifact_id);
    setNotice('');
    try {
      const blob = await fetchArtifact(apiKey, artifact.artifact_id);
      setPreview({ artifact, source: decodeArtifactSource(await blob.arrayBuffer()) });
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '无法读取生成文件');
    } finally {
      setArtifactBusy('');
    }
  }

  async function download(artifact: Artifact) {
    setArtifactBusy(artifact.artifact_id);
    setNotice('');
    try {
      const blob = await fetchArtifact(apiKey, artifact.artifact_id);
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
        <div className="topbar-actions">
          {apiKey ? (
            <>
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
          ) : <span className="offline">等待连接</span>}
        </div>
      </header>

      <main className="main-area">
        <div className="workspace-heading">
          <div>
            <span className="eyebrow">WORKSPACE / 01</span>
            <h1>把风控规则，写成可用策略。</h1>
            <p>描述交易所、合约、价格条件和通知方式。信息不足时，Agent 会先向你确认。</p>
          </div>
          {apiKey && messages.length > 0 && (
            <button className="new-chat-button" type="button" onClick={newConversation} disabled={busy}>新对话</button>
          )}
        </div>

        {!apiKey ? (
          <section className="connection-card">
            <div className="connection-icon">↗</div>
            <div>
              <span className="eyebrow">连接服务</span>
              <h2>输入 API Key 开始</h2>
              <p>密钥仅保留在当前页面内存，用于访问内网中的 Agent 服务。</p>
            </div>
            <form onSubmit={connect} className="connection-form">
              <input
                type="password"
                value={keyDraft}
                onChange={(event) => setKeyDraft(event.target.value)}
                placeholder="输入 AGENT_API_KEY"
                autoComplete="off"
                aria-label="API Key"
              />
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
      </main>

      <footer className="page-footer"><span>RISK STRATEGY / AGENT</span><span>接口源码只读 · 生成文件独立保存</span></footer>

      {preview && (
        <div className="preview-backdrop" role="presentation" onClick={() => setPreview(null)}>
          <section className="preview-dialog" role="dialog" aria-modal="true" aria-label={'预览 ' + preview.artifact.filename} onClick={(event) => event.stopPropagation()}>
            <div className="preview-header">
              <div><span className="eyebrow">GENERATED FILE</span><h2>{preview.artifact.filename}</h2></div>
              <div className="preview-actions">
                <button type="button" onClick={() => void download(preview.artifact)}>下载文件</button>
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
