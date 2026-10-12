import type { Artifact, ChatRequest, DoneEvent } from './types';

interface StreamCallbacks {
  onStatus: (phase: string, tool?: string) => void;
  onDelta: (text: string) => void;
  onArtifact: (artifact: Artifact) => void;
  onDone: (done: DoneEvent) => void;
}

function authHeaders(apiKey: string): HeadersInit {
  return {
    Authorization: 'Bearer ' + apiKey,
    'Content-Type': 'application/json',
  };
}

async function responseError(response: Response): Promise<Error> {
  let detail = '';
  try {
    const body = await response.json() as { detail?: string };
    detail = body.detail || '';
  } catch {
    detail = response.statusText;
  }
  return new Error(detail || 'HTTP ' + response.status);
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
    throw new Error(String(payload.message || 'Agent 执行失败'));
  }
  return false;
}

export async function streamChat(
  apiKey: string,
  request: ChatRequest,
  callbacks: StreamCallbacks,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/agent/chat/stream', {
    method: 'POST',
    headers: { ...authHeaders(apiKey), Accept: 'text/event-stream' },
    body: JSON.stringify(request),
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
