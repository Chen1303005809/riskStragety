// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react';
import App from './App';

const artifactId = 'a'.repeat(32);

function streamReply(content: string, withArtifact = false): Response {
  const frames = [
    'event: status\ndata: {"phase":"started"}\n\n',
    'event: delta\ndata: ' + JSON.stringify({ text: content }) + '\n\n',
    ...(withArtifact ? ['event: artifact\ndata: ' + JSON.stringify({ artifact_id: artifactId, filename: 'risk.py' }) + '\n\n'] : []),
    'event: done\ndata: ' + JSON.stringify({
      content, model: 'model-b',
      artifacts: withArtifact ? [{ artifact_id: artifactId, filename: 'risk.py' }] : [],
      usage: {},
    }) + '\n\n',
  ].join('');
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(frames));
      controller.close();
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } });
}

describe('chat page', () => {
  beforeEach(() => {
    let sequence = 0;
    vi.stubGlobal('crypto', { randomUUID: () => String(++sequence) });
    Element.prototype.scrollIntoView = vi.fn();
    URL.createObjectURL = vi.fn().mockReturnValue('blob:test');
    URL.revokeObjectURL = vi.fn();
    HTMLAnchorElement.prototype.click = vi.fn();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows streaming replies, carries history, previews and downloads this turn’s file', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url === '/api/models') return Promise.resolve(Response.json({ models: ['model-a', 'model-b'], default_model: 'model-b' }));
      if (url === '/api/agent/chat/stream') {
        const number = fetchMock.mock.calls.filter(([path]) => path === url).length;
        return Promise.resolve(streamReply(number === 1 ? '请补充阈值。' : '已生成策略。', number === 2));
      }
      if (url === '/api/artifacts/download') return Promise.resolve(new Response('threshold = 3900\n'));
      throw new Error('unexpected URL: ' + url);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    fireEvent.change(screen.getByLabelText('API Key'), { target: { value: 'secret-key' } });
    fireEvent.click(screen.getByText('连接并读取模型'));
    await screen.findByText('从一条价格规则开始');

    fireEvent.change(screen.getByLabelText('输入策略需求'), { target: { value: '螺纹钢预警' } });
    fireEvent.click(screen.getByText(/发送需求/));
    await screen.findByText('请补充阈值。');

    fireEvent.change(screen.getByLabelText('输入策略需求'), { target: { value: 'SHFE rb2710 超过 3900' } });
    fireEvent.click(screen.getByText(/发送需求/));
    await screen.findByText('已生成策略。');
    await screen.findByText('risk.py');

    const chatCalls = fetchMock.mock.calls.filter(([url]) => url === '/api/agent/chat/stream');
    const request = JSON.parse((chatCalls[1][1] as RequestInit).body as string);
    expect(request.model).toBe('model-b');
    expect(request.history).toEqual([
      { role: 'user', content: '螺纹钢预警' },
      { role: 'assistant', content: '请补充阈值。' },
    ]);

    fireEvent.click(screen.getByText('预览'));
    await screen.findByText('threshold = 3900');
    fireEvent.click(screen.getByRole('button', { name: '关闭预览' }));
    fireEvent.click(screen.getByText('下载'));
    await waitFor(() => {
      expect(fetchMock.mock.calls.filter(([url]) => url === '/api/artifacts/download')).toHaveLength(2);
      expect(HTMLAnchorElement.prototype.click).toHaveBeenCalled();
    });
    const downloadCalls = fetchMock.mock.calls.filter(([url]) => url === '/api/artifacts/download');
    expect(JSON.parse((downloadCalls[1][1] as RequestInit).body as string)).toEqual({ artifact_id: artifactId });
  });
});
