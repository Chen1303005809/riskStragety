"""Ollama gateway and Ollama-native request-message structures."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from ollama import AsyncClient

from agent.contracts import (
    AgentMessage,
    AssistantMessage,
    ModelChunk,
    ModelGateway,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from backend.adapters.provider_messages import ProviderWireMessage


@dataclass(frozen=True)
class OllamaWireMessage(ProviderWireMessage):
    role: str
    content: str

    def to_payload(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class OllamaFunctionWireCall:
    name: str
    arguments: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {"function": {"name": self.name, "arguments": self.arguments}}


@dataclass(frozen=True)
class OllamaAssistantWireMessage(OllamaWireMessage):
    tool_calls: tuple[OllamaFunctionWireCall, ...] = ()
    thinking: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        if self.tool_calls:
            payload["tool_calls"] = [call.to_payload() for call in self.tool_calls]
        if self.thinking:
            payload["thinking"] = self.thinking
        return payload


@dataclass(frozen=True)
class OllamaToolWireMessage(OllamaWireMessage):
    tool_name: str

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "tool_name": self.tool_name}


class OllamaModelGateway(ModelGateway):
    def __init__(self, base_url: str) -> None:
        self._client = AsyncClient(host=base_url)

    async def close(self) -> None:
        await self._client.close()

    async def list_models(self) -> list[str]:
        response = await self._client.list()
        names = (
            getattr(item, "model", None) or getattr(item, "name", None)
            for item in response.models
        )
        return sorted({name for name in names if isinstance(name, str) and name})

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[AgentMessage],
        tools: list[dict[str, Any]],
        *,
        options: dict[str, int | float],
        think: bool,
    ) -> AsyncIterator[ModelChunk]:
        stream = await self._client.chat(
            model=model,
            messages=_ollama_messages(messages),
            tools=tools,
            stream=True,
            options=options,
            think=think,
        )
        async for item in stream:
            message = item.message
            calls: list[ToolCall] = []
            if message and message.tool_calls:
                for call in message.tool_calls:
                    arguments = call.function.arguments
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    call_id = getattr(call, "id", None)
                    calls.append(
                        ToolCall(
                            call.function.name,
                            arguments,
                            call_id if isinstance(call_id, str) and call_id else None,
                        )
                    )
            usage = {}
            if item.done:
                usage = {
                    "prompt_tokens": item.prompt_eval_count or 0,
                    "completion_tokens": item.eval_count or 0,
                }
            thinking = getattr(message, "thinking", "") if message else ""
            yield ModelChunk(
                content=message.content or "" if message else "",
                tool_calls=tuple(calls),
                usage=usage,
                reasoning_content=thinking if isinstance(thinking, str) else "",
            )


def _ollama_messages(messages: Sequence[AgentMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    known_call_ids: set[str] = set()
    pending_call_names: dict[str, str] = {}
    for message in messages:
        if isinstance(message, SystemMessage):
            wire_message: ProviderWireMessage = OllamaWireMessage("system", message.content)
        elif isinstance(message, UserMessage):
            wire_message = OllamaWireMessage("user", message.content)
        elif isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if not call.call_id:
                    raise ValueError("Ollama 工具调用缺少 call_id")
                if call.call_id in known_call_ids:
                    raise ValueError(f"Ollama 工具调用 ID 重复：{call.call_id}")
                known_call_ids.add(call.call_id)
                pending_call_names[call.call_id] = call.name
            wire_message = OllamaAssistantWireMessage(
                role="assistant",
                content=message.content,
                tool_calls=tuple(
                    OllamaFunctionWireCall(call.name, call.arguments)
                    for call in message.tool_calls
                ),
                thinking=message.reasoning_content,
            )
        elif isinstance(message, ToolResultMessage):
            expected_tool_name = pending_call_names.get(message.call_id)
            if expected_tool_name is None:
                raise ValueError(
                    f"Ollama 工具结果没有对应的调用：{message.call_id}"
                )
            if expected_tool_name != message.tool_name:
                raise ValueError(
                    f"Ollama 工具结果名称与调用不匹配：{message.call_id}"
                )
            del pending_call_names[message.call_id]
            wire_message = OllamaToolWireMessage(
                role="tool",
                content=message.content,
                tool_name=message.tool_name,
            )
        else:
            raise TypeError(f"不支持的 Agent 消息类型：{type(message).__name__}")
        converted.append(wire_message.to_payload())
    if pending_call_names:
        missing = ", ".join(sorted(pending_call_names))
        raise ValueError(f"Ollama 工具调用缺少结果：{missing}")
    return converted
