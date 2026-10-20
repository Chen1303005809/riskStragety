"""DeepSeek gateway and Chat Completions request-message structures."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

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
class DeepSeekWireMessage(ProviderWireMessage):
    role: str
    content: str

    def to_payload(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class DeepSeekFunctionWireCall:
    call_id: str
    name: str
    arguments: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class DeepSeekAssistantWireMessage(DeepSeekWireMessage):
    tool_calls: tuple[DeepSeekFunctionWireCall, ...] = ()
    reasoning_content: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        if self.tool_calls:
            payload["tool_calls"] = [call.to_payload() for call in self.tool_calls]
        if self.reasoning_content:
            payload["reasoning_content"] = self.reasoning_content
        return payload


@dataclass(frozen=True)
class DeepSeekToolWireMessage(DeepSeekWireMessage):
    tool_call_id: str

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "tool_call_id": self.tool_call_id}


class DeepSeekModelGateway(ModelGateway):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        response = await self._client.get("/models")
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError("DeepSeek 返回了无效的模型列表")
        return sorted(
            {
                item["id"]
                for item in entries
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"]
            }
        )

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[AgentMessage],
        tools: list[dict[str, Any]],
        *,
        options: dict[str, int | float],
        think: bool,
    ) -> AsyncIterator[ModelChunk]:
        request: dict[str, Any] = {
            "model": model,
            "messages": _deepseek_messages(messages),
            "stream": True,
            "thinking": {"type": "enabled" if think else "disabled"},
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        if think:
            request["reasoning_effort"] = "high"
        else:
            request["temperature"] = options.get("temperature", 0.3)
            request["top_p"] = options.get("top_p", 0.8)

        pending_calls: dict[int, dict[str, str]] = {}
        usage: dict[str, int] = {}
        async with self._client.stream(
            "POST", "/chat/completions", json=request
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ValueError("DeepSeek 返回了无效的流式数据") from exc
                if not isinstance(event, dict):
                    continue

                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    prompt_tokens = event_usage.get("prompt_tokens")
                    completion_tokens = event_usage.get("completion_tokens")
                    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                        usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        }

                choices = event.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    delta_calls = delta.get("tool_calls")
                    if isinstance(delta_calls, list):
                        for call in delta_calls:
                            if not isinstance(call, dict):
                                continue
                            index = call.get("index", 0)
                            if not isinstance(index, int):
                                continue
                            pending = pending_calls.setdefault(
                                index, {"id": "", "name": "", "arguments": ""}
                            )
                            call_id = call.get("id")
                            if isinstance(call_id, str) and call_id:
                                if pending["id"] and pending["id"] != call_id:
                                    raise ValueError("DeepSeek 流式工具调用 ID 不一致")
                                pending["id"] = call_id
                            function = call.get("function")
                            if isinstance(function, dict):
                                name = function.get("name")
                                arguments = function.get("arguments")
                                if isinstance(name, str):
                                    pending["name"] += name
                                if isinstance(arguments, str):
                                    pending["arguments"] += arguments
                    if isinstance(content, str) or isinstance(reasoning, str):
                        yield ModelChunk(
                            content=content if isinstance(content, str) else "",
                            reasoning_content=reasoning if isinstance(reasoning, str) else "",
                        )

        calls: list[ToolCall] = []
        for index in sorted(pending_calls):
            item = pending_calls[index]
            try:
                arguments = json.loads(item["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            if item["name"]:
                calls.append(
                    ToolCall(
                        item["name"],
                        arguments,
                        item["id"] or None,
                    )
                )
        yield ModelChunk(tool_calls=tuple(calls), usage=usage)


def _deepseek_messages(messages: Sequence[AgentMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    known_call_ids: set[str] = set()
    pending_call_names: dict[str, str] = {}
    for message in messages:
        if isinstance(message, SystemMessage):
            wire_message: ProviderWireMessage = DeepSeekWireMessage("system", message.content)
        elif isinstance(message, UserMessage):
            wire_message = DeepSeekWireMessage("user", message.content)
        elif isinstance(message, AssistantMessage):
            calls: list[DeepSeekFunctionWireCall] = []
            for call in message.tool_calls:
                if not call.call_id:
                    raise ValueError("DeepSeek 工具调用缺少 call_id")
                if call.call_id in known_call_ids:
                    raise ValueError(f"DeepSeek 工具调用 ID 重复：{call.call_id}")
                known_call_ids.add(call.call_id)
                pending_call_names[call.call_id] = call.name
                calls.append(
                    DeepSeekFunctionWireCall(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=json.dumps(call.arguments, ensure_ascii=False),
                    )
                )
            wire_message = DeepSeekAssistantWireMessage(
                role="assistant",
                content=message.content,
                tool_calls=tuple(calls),
                reasoning_content=message.reasoning_content,
            )
        elif isinstance(message, ToolResultMessage):
            expected_tool_name = pending_call_names.get(message.call_id)
            if expected_tool_name is None:
                raise ValueError(
                    f"DeepSeek 工具结果没有对应的调用：{message.call_id}"
                )
            if expected_tool_name != message.tool_name:
                raise ValueError(
                    f"DeepSeek 工具结果名称与调用不匹配：{message.call_id}"
                )
            del pending_call_names[message.call_id]
            wire_message = DeepSeekToolWireMessage(
                role="tool",
                content=message.content,
                tool_call_id=message.call_id,
            )
        else:
            raise TypeError(f"不支持的 Agent 消息类型：{type(message).__name__}")
        converted.append(wire_message.to_payload())

    if pending_call_names:
        missing = ", ".join(sorted(pending_call_names))
        raise ValueError(f"DeepSeek 工具调用缺少结果：{missing}")
    return converted
