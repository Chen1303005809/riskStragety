import { describe, expect, it, vi } from 'vitest';
import { streamChat } from './api';

function sseResponse(frames: string[], splitAt: number): Response {
  const raw = new TextEncoder().encode(frames.join(''));
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(raw.slice(0, splitAt));
      controller.enqueue(raw.slice(splitAt));
      controller.close();
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } });
}

describe('streamChat', () => {
  it('sends complete history and parses split SSE frames', async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([
      'event: status\ndata: {"phase":"started"}\n\n',
      'event: delta\ndata: {"text":"第一段"}\n\n',
      'event: delta\ndata: {"text":"第二段"}\n\n',
      'event: artifact\ndata: {"artifact_id":"abc","filename":"risk.py"}\n\n',
      'event: done\ndata: {"content":"第一段第二段","model":"m","artifacts":[],"usage":{}}\n\n',
    ], 31));
    vi.stubGlobal('fetch', fetchMock);
    const status = vi.fn();
    const delta = vi.fn();
    const artifact = vi.fn();
    const done = vi.fn();

    await streamChat('my-key', {
      message: '改成 3900',
      history: [
        { role: 'user', content: '原规则 3800' },
        { role: 'assistant', content: '原策略已生成' },
      ],
      model: 'm',
    }, { onStatus: status, onDelta: delta, onArtifact: artifact, onDone: done }, new AbortController().signal);

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe('POST');
    expect(init.headers).toMatchObject({ Authorization: 'Bearer my-key' });
    expect(JSON.parse(init.body as string).history).toEqual([
      { role: 'user', content: '原规则 3800' },
      { role: 'assistant', content: '原策略已生成' },
    ]);
    expect(status).toHaveBeenCalledWith('started', undefined);
    expect(delta.mock.calls.flat()).toEqual(['第一段', '第二段']);
    expect(artifact).toHaveBeenCalledWith({ artifact_id: 'abc', filename: 'risk.py' });
    expect(done).toHaveBeenCalledOnce();
  });

  it('reports interrupted streams without done', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([
      'event: delta\ndata: {"text":"partial"}\n\n',
    ], 10)));
    const callbacks = { onStatus: vi.fn(), onDelta: vi.fn(), onArtifact: vi.fn(), onDone: vi.fn() };
    await expect(streamChat('key', { message: '生成', history: [] }, callbacks, new AbortController().signal))
      .rejects.toThrow('提前结束');
  });
});
