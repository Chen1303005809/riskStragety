import type {
  AccountSummary,
  Artifact,
  ArtifactRecord,
  ChatRequest,
  ConversationDetail,
  ConversationSummary,
  DoneEvent,
  MonitorConversationsPage,
  Page,
} from './types';

interface StreamCallbacks {
  onStatus: (phase: string, tool?: string) => void;
  onDelta: (text: string) => void;
  onArtifact: (artifact: Artifact) => void;
  onDone: (done: DoneEvent) => void;
}

export class ApiError extends Error {
  constructor(message: string, readonly errCode?: string) {
    super(message);
    this.name = 'ApiError';
  }
}

function authHeaders(apiKey: string): HeadersInit {
  return {
    Authorization: 'Bearer ' + apiKey,
    'Content-Type': 'application/json',
  };
}

async function responseError(response: Response): Promise<Error> {
  let message = '';
  let errCode: string | undefined;
  try {
    const body = await response.json() as {
      detail?: unknown;
      ErrCode?: unknown;
      ErrMsg?: unknown;
    };
    message = typeof body.ErrMsg === 'string' ? body.ErrMsg : '';
    if (!message && typeof body.detail === 'string') message = body.detail;
    errCode = typeof body.ErrCode === 'string' ? body.ErrCode : undefined;
  } catch {
    message = response.statusText;
  }
  return new ApiError(message || 'HTTP ' + response.status, errCode);
}

export async function fetchModels(apiKey: string): Promise<{ models: string[]; defaultModel: string }> {
  const response = await fetch('/api/models', {
    headers: authHeaders(apiKey),
    cache: 'no-store',
  });
  if (!response.ok) throw await responseError(response);
  const data = await response.json() as { models?: unknown; default_model?: unknown };
  if (!Array.isArray(data.models)) throw new Error('模型列表格式无效');
  return {
    models: data.models.filter((name): name is string => typeof name === 'string'),
    defaultModel: typeof data.default_model === 'string' ? data.default_model : '',
  };
}

async function jsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) throw await responseError(response);
  return await response.json() as T;
}

export async function fetchMonitorAccounts(
  apiKey: string,
  offset = 0,
  limit = 50,
): Promise<Page<AccountSummary>> {
  const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  const response = await fetch('/api/monitor/accounts?' + query, {
    headers: authHeaders(apiKey),
    cache: 'no-store',
  });
  return jsonResponse<Page<AccountSummary>>(response);
}

export async function fetchMonitorConversations(
  apiKey: string,
  accountId: string,
  options: { offset?: number; limit?: number; unassignedOffset?: number; unassignedLimit?: number } = {},
): Promise<MonitorConversationsPage> {
  const query = new URLSearchParams({
    offset: String(options.offset ?? 0),
    limit: String(options.limit ?? 50),
    unassigned_offset: String(options.unassignedOffset ?? 0),
    unassigned_limit: String(options.unassignedLimit ?? 50),
  });
  const response = await fetch(
    `/api/monitor/accounts/${encodeURIComponent(accountId)}/conversations?${query}`,
    { headers: authHeaders(apiKey), cache: 'no-store' },
  );
  return jsonResponse<MonitorConversationsPage>(response);
}

export async function fetchMonitorConversation(
  apiKey: string,
  accountId: string,
  conversationId: string,
  options: { messageOffset?: number; messageLimit?: number; artifactOffset?: number; artifactLimit?: number } = {},
): Promise<ConversationDetail> {
  const query = new URLSearchParams({
    message_offset: String(options.messageOffset ?? 0),
    message_limit: String(options.messageLimit ?? 100),
    artifact_offset: String(options.artifactOffset ?? 0),
    artifact_limit: String(options.artifactLimit ?? 50),
  });
  const response = await fetch(
    `/api/monitor/accounts/${encodeURIComponent(accountId)}/conversations/${encodeURIComponent(conversationId)}?${query}`,
    { headers: authHeaders(apiKey), cache: 'no-store' },
  );
  return jsonResponse<ConversationDetail>(response);
}

export async function fetchConversationForChat(
  apiKey: string,
  accountId: string,
  conversationId: string,
): Promise<{ conversation: ConversationSummary; messages: ConversationDetail['messages']; artifacts: ArtifactRecord[] }> {
  const first = await fetchMonitorConversation(apiKey, accountId, conversationId, {
    messageOffset: 0,
    messageLimit: 200,
    artifactOffset: 0,
    artifactLimit: 100,
  });
  const messages = [...first.messages];
  const artifacts = [...first.artifacts];
  for (let offset = messages.length; offset < first.message_total; offset += first.message_limit) {
    const page = await fetchMonitorConversation(apiKey, accountId, conversationId, {
      messageOffset: offset,
      messageLimit: first.message_limit,
      artifactLimit: 1,
    });
    messages.push(...page.messages);
  }
  for (let offset = artifacts.length; offset < first.artifact_total; offset += first.artifact_limit) {
    const page = await fetchMonitorConversation(apiKey, accountId, conversationId, {
      artifactOffset: offset,
      artifactLimit: first.artifact_limit,
      messageLimit: 1,
    });
    artifacts.push(...page.artifacts);
  }
  return { conversation: first.conversation, messages, artifacts };
}

function handleFrame(frame: string, callbacks: StreamCallbacks): boolean {
  const lines = frame.split(/\r?\n/);
  const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim();
  const data = lines.filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart()).join('\n');
  if (!event || !data) return false;
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(data) as Record<string, unknown>;
  } catch {
    throw new Error('服务返回了无效的 SSE 数据');
  }
  if (event === 'status') {
    callbacks.onStatus(String(payload.phase || ''), typeof payload.tool === 'string' ? payload.tool : undefined);
  } else if (event === 'delta') {
    callbacks.onDelta(String(payload.text || ''));
  } else if (event === 'artifact') {
    callbacks.onArtifact(payload as unknown as Artifact);
  } else if (event === 'done') {
    callbacks.onDone(payload as unknown as DoneEvent);
    return true;
  } else if (event === 'error') {
    throw new ApiError(
      typeof payload.ErrMsg === 'string' && payload.ErrMsg ? payload.ErrMsg : 'Agent 执行失败',
      typeof payload.ErrCode === 'string' ? payload.ErrCode : undefined,
    );
  }
  return false;
}

export async function streamChat(
  apiKey: string,
  request: ChatRequest,
  callbacks: StreamCallbacks,
  signal: AbortSignal,
): Promise<void> {
  const requestBody = { ...request };
  delete requestBody.history;
  const response = await fetch('/api/agent/chat/stream', {
    method: 'POST',
    headers: { ...authHeaders(apiKey), Accept: 'text/event-stream' },
    body: JSON.stringify(requestBody),
    signal,
  });
  if (!response.ok) throw await responseError(response);
  if (!response.body) throw new Error('服务未返回流式内容');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = false;
  try {
    while (!completed) {
      const item = await reader.read();
      buffer += decoder.decode(item.value || new Uint8Array(), { stream: !item.done });
      let match = /\r?\n\r?\n/.exec(buffer);
      while (match) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        completed = handleFrame(frame, callbacks);
        if (completed) break;
        match = /\r?\n\r?\n/.exec(buffer);
      }
      if (item.done) break;
    }
    if (!completed) throw new Error('Agent 响应提前结束');
  } finally {
    if (!completed) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export async function fetchArtifact(apiKey: string, artifactId: string, accountId: string): Promise<Blob> {
  const response = await fetch('/api/artifacts/download', {
    method: 'POST',
    headers: authHeaders(apiKey),
    body: JSON.stringify({ artifact_id: artifactId, account_id: accountId }),
  });
  if (!response.ok) throw await responseError(response);
  return response.blob();
}

export function decodeArtifactSource(bytes: ArrayBuffer): string {
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
}

export function parsePythonFile(bytes: ArrayBuffer, filename: string): File {
  // Validate UTF-8 while preserving the original bytes in the downloaded file.
  new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
  const name = filename.toLowerCase().endsWith('.py') ? filename : `${filename}.py`;
  return new File([bytes], name, { type: 'text/x-python;charset=utf-8' });
}
