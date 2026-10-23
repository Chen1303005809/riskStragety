"""DeepSeek gateway request and streaming message conversion."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent.contracts import (
    AssistantMessage,
    AssistantToolCall,
    ModelGateway,
    ModelMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from backend.adapters.deepseek import DeepSeekModelGateway, _deepseek_messages


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *_: object) -> None:
        return None


class _FakeClient:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.seen: dict[str, Any] = {}

    def stream(self, method: str, url: str, *, json: dict[str, Any]) -> _FakeStreamContext:
        self.seen.update({"method": method, "url": url, "json": json})
        return _FakeStreamContext(_FakeResponse(self.lines))

    async def aclose(self) -> None:
        return None


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}"


def _messages() -> list[ModelMessage]:
    return [
        SystemMessage("system prompt"),
        UserMessage("请加载两个 Skill 资料"),
        AssistantMessage(
            "开始读取",
            tool_calls=(
                AssistantToolCall(
                    "call-a", "load_skill", {"skill_id": "price-alert-strategy"}
                ),
                AssistantToolCall(
                    "call-b", "load_skill", {"skill_id": "test_case_builder"}
                ),
            ),
            reasoning_content="逐个读取文件",
        ),
        ToolResultMessage(
            "读取结果 B", call_id="call-b", tool_name="load_skill"
        ),
        ToolResultMessage(
            "读取结果 A", call_id="call-a", tool_name="load_skill"
        ),
    ]


def test_deepseek_wire_messages_use_call_ids_and_openai_argument_strings() -> None:
    messages = _deepseek_messages(_messages())

    assert messages == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "请加载两个 Skill 资料"},
        {
            "role": "assistant",
            "content": "开始读取",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {
                        "name": "load_skill",
                        "arguments": '{"skill_id": "price-alert-strategy"}',
                    },
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {
                        "name": "load_skill",
                        "arguments": '{"skill_id": "test_case_builder"}',
                    },
                },
            ],
            "reasoning_content": "逐个读取文件",
        },
        {"role": "tool", "content": "读取结果 B", "tool_call_id": "call-b"},
        {"role": "tool", "content": "读取结果 A", "tool_call_id": "call-a"},
    ]


def test_deepseek_wire_serializer_rejects_unmatched_tool_result() -> None:
    with pytest.raises(ValueError, match="没有对应的调用"):
        _deepseek_messages(
            [ToolResultMessage("orphan", call_id="missing", tool_name="lookup")]
        )


def test_deepseek_wire_serializer_rejects_tool_name_mismatch() -> None:
    with pytest.raises(ValueError, match="名称与调用不匹配"):
        _deepseek_messages(
            [
                AssistantMessage(
                    "",
                    tool_calls=(
                        AssistantToolCall("call-1", "lookup", {}),
                    ),
                ),
                ToolResultMessage("result", call_id="call-1", tool_name="other"),
            ]
        )


def test_deepseek_gateway_preserves_streamed_tool_ids_and_serializes_messages() -> None:
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "model-call-a",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_",
                                        "arguments": '{"contract_',
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "model-call-b",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_exchange_by_contract_code",
                                        "arguments": '{"contract_code":"rb2710"}',
                                    },
                                },
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "已发起查询",
                            "reasoning_content": "确认合约代码",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "exchange_by_contract_code",
                                        "arguments": 'code":"rb2710"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            }
        ),
        "data: [DONE]",
    ]
    fake_client = _FakeClient(lines)
    gateway = DeepSeekModelGateway("test-key")
    gateway._client = fake_client  # type: ignore[assignment]

    async def scenario():
        chunks = [
            chunk
            async for chunk in gateway.stream_chat(
                "deepseek-flash",
                _messages(),
                [],
                options={"temperature": 0.3, "top_p": 0.8},
                think=True,
            )
        ]
        await gateway.close()
        return chunks

    chunks = asyncio.run(scenario())

    assert fake_client.seen["method"] == "POST"
    assert fake_client.seen["url"] == "/chat/completions"
    assert fake_client.seen["json"]["messages"] == _deepseek_messages(_messages())
    assert "tool_name" not in json.dumps(fake_client.seen["json"]["messages"])
    calls_chunk = chunks[-1]
    assert [call.call_id for call in calls_chunk.tool_calls] == [
        "model-call-a",
        "model-call-b",
    ]
    assert [call.name for call in calls_chunk.tool_calls] == [
        "lookup_exchange_by_contract_code",
        "lookup_exchange_by_contract_code",
    ]
    assert calls_chunk.tool_calls[0].arguments == {"contract_code": "rb2710"}
    assert calls_chunk.tool_calls[1].arguments == {"contract_code": "rb2710"}
    assert calls_chunk.usage == {"prompt_tokens": 12, "completion_tokens": 5}
    assert chunks[0].content == "已发起查询"
    assert chunks[0].reasoning_content == "确认合约代码"
    assert issubclass(DeepSeekModelGateway, ModelGateway)
