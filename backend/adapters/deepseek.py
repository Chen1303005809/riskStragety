"""OpenAI-compatible streaming adapter for the DeepSeek API."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent.contracts import ModelChunk, ToolCall


class DeepSeekModelGateway:
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
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        options: dict[str, int | float],
        think: bool,
    ) -> AsyncIterator[ModelChunk]:
        request: dict[str, Any] = {
            "model": model,
            "messages": _openai_messages(messages),
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
                                index, {"name": "", "arguments": ""}
                            )
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
                calls.append(ToolCall(item["name"], arguments))
        yield ModelChunk(tool_calls=tuple(calls), usage=usage)


def _openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate AgentRunner's provider-neutral tool messages to OpenAI's format."""
    tool_call_ids: dict[str, deque[str]] = defaultdict(deque)
    converted: list[dict[str, Any]] = []
    next_call_id = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            tool_calls = []
            for call in message["tool_calls"]:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue
                supplied_id = call.get("id") if isinstance(call, dict) else None
                call_id = (
                    supplied_id
                    if isinstance(supplied_id, str) and supplied_id
                    else f"call_{next_call_id}"
                )
                next_call_id += 1
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_call_ids[name].append(call_id)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            converted_assistant = {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                converted_assistant["reasoning_content"] = reasoning
            converted.append(converted_assistant)
            continue

        if role == "tool":
            name = message.get("tool_name")
            if isinstance(name, str) and tool_call_ids[name]:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_ids[name].popleft(),
                        "content": message.get("content", ""),
                    }
                )
            continue

        converted_message = {
            key: message[key]
            for key in ("role", "content", "reasoning_content")
            if key in message and message[key] is not None
        }
        converted.append(converted_message)
    return converted
