"""The SDK adapter converts Ollama chunks without requiring a running server."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.contracts import (
    AssistantMessage,
    AssistantToolCall,
    ModelGateway,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from backend.adapters.ollama import OllamaModelGateway, _ollama_messages


def test_ollama_list_and_stream_conversion() -> None:
    assert issubclass(OllamaModelGateway, ModelGateway)
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
                            name="load_skill", arguments='{"skill_id":"price-alert-strategy"}'
                        ))], thinking="内部思考"
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
        chunks = [chunk async for chunk in gateway.stream_chat(
            "model-a",
            [
                SystemMessage("system prompt"),
                UserMessage("hi"),
                AssistantMessage(
                    "加载 Skill",
                    tool_calls=(
                        AssistantToolCall(
                            "internal-call-1",
                            "load_skill",
                            {"skill_id": "price-alert-strategy"},
                        ),
                    ),
                    reasoning_content="先查看源码",
                ),
                ToolResultMessage(
                    content='{"files": ["external_head.py"]}',
                    call_id="internal-call-1",
                    tool_name="load_skill",
                ),
            ],
            [],
            options={"temperature": 0.4, "top_p": 0.9, "num_ctx": 32_768},
            think=True,
        )]
        await gateway.close()
        return names, chunks

    names, chunks = asyncio.run(scenario())
    assert names == ["model-a", "model-b"]
    assert seen["model"] == "model-a"
    assert seen["stream"] is True
    assert seen["options"] == {"temperature": 0.4, "top_p": 0.9, "num_ctx": 32_768}
    assert seen["think"] is True
    assert seen["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
                "content": "加载 Skill",
            "tool_calls": [
                {
                    "function": {
                        "name": "load_skill",
                        "arguments": {"skill_id": "price-alert-strategy"},
                    }
                }
            ],
            "thinking": "先查看源码",
        },
        {
            "role": "tool",
            "content": '{"files": ["external_head.py"]}',
            "tool_name": "load_skill",
        },
    ]
    assert chunks[0].content == "正在读取"
    assert chunks[0].reasoning_content == "内部思考"
    assert chunks[0].tool_calls == (
        ToolCall("load_skill", {"skill_id": "price-alert-strategy"}),
    )
    assert chunks[1].usage == {"prompt_tokens": 12, "completion_tokens": 4}
    assert seen["closed"] is True


def test_ollama_serializer_rejects_unmatched_tool_result_ids() -> None:
    with pytest.raises(ValueError, match="没有对应的调用"):
        _ollama_messages(
            [ToolResultMessage("orphan", call_id="missing", tool_name="lookup")]
        )
