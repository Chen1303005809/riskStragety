"""The SDK adapter converts Ollama chunks without requiring a running server."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent.contracts import ToolCall
from backend.adapters.ollama import OllamaModelGateway


def test_ollama_list_and_stream_conversion() -> None:
    seen = {}

    class FakeClient:
        async def list(self):
            return SimpleNamespace(models=[
                SimpleNamespace(model="model-b"),
                SimpleNamespace(model="model-a"),
                SimpleNamespace(model="model-a"),
            ])

        async def chat(self, **kwargs):
            seen.update(kwargs)

            async def chunks():
                yield SimpleNamespace(
                    message=SimpleNamespace(
                        content="正在读取", tool_calls=[SimpleNamespace(function=SimpleNamespace(
                            name="read_interface_file", arguments='{"filename":"external_head.py"}'
                        ))]
                    ),
                    done=False, prompt_eval_count=None, eval_count=None,
                )
                yield SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[]),
                    done=True, prompt_eval_count=12, eval_count=4,
                )

            return chunks()

        async def close(self):
            seen["closed"] = True

    gateway = OllamaModelGateway("http://127.0.0.1:11434")
    gateway._client = FakeClient()

    async def scenario():
        names = await gateway.list_models()
        chunks = [chunk async for chunk in gateway.stream_chat("model-a", [{"role": "user", "content": "hi"}], [])]
        await gateway.close()
        return names, chunks

    names, chunks = asyncio.run(scenario())
    assert names == ["model-a", "model-b"]
    assert seen["model"] == "model-a"
    assert seen["stream"] is True
    assert chunks[0].content == "正在读取"
    assert chunks[0].tool_calls == (ToolCall("read_interface_file", {"filename": "external_head.py"}),)
    assert chunks[1].usage == {"prompt_tokens": 12, "completion_tokens": 4}
    assert seen["closed"] is True
