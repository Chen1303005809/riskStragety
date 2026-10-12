"""Translate Ollama SDK streaming chunks into the agent's model interface."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ollama import AsyncClient

from agent.contracts import ModelChunk, ToolCall


class OllamaModelGateway:
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
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ModelChunk]:
        stream = await self._client.chat(
            model=model, messages=messages, tools=tools, stream=True
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
                    calls.append(ToolCall(call.function.name, arguments))
            usage = {}
            if item.done:
                usage = {
                    "prompt_tokens": item.prompt_eval_count or 0,
                    "completion_tokens": item.eval_count or 0,
                }
            yield ModelChunk(
                content=message.content or "" if message else "",
                tool_calls=tuple(calls),
                usage=usage,
            )
