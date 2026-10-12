"""Prompt assembly and the bounded model/tool loop."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent.contracts import (
    AgentEvent,
    Artifact,
    ArtifactWriter,
    ChatTurn,
    InterfaceReader,
    ModelGateway,
    ToolCall,
)
from agent.prompt import build_system_prompt


MAX_TOOL_ROUNDS = 8
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_interface_files",
            "description": "列出价格接口目录中可供阅读的 Python 源码文件。生成代码前先调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_interface_file",
            "description": "阅读价格接口目录中指定 Python 文件的完整源码。",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "纯 .py 文件名"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_generated_file",
            "description": "把完成的 Python 策略保存为新的可下载文件；不得用于参数追问轮。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "下载时显示的纯 .py 文件名"},
                    "source": {"type": "string", "description": "完整 Python 源码"},
                },
                "required": ["filename", "source"],
            },
        },
    },
]


class AgentError(RuntimeError):
    """A request cannot finish after a bounded number of tool rounds."""


class AgentRunner:
    def __init__(
        self,
        model: ModelGateway,
        interfaces: InterfaceReader,
        artifacts: ArtifactWriter,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._model = model
        self._interfaces = interfaces
        self._artifacts = artifacts
        self._max_tool_rounds = max_tool_rounds

    async def run(
        self,
        *,
        message: str,
        history: Sequence[ChatTurn],
        model: str,
        account_id: str,
        options: dict[str, int | float] | None = None,
        think: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        model_options = options if options is not None else {
            "temperature": 0.3,
            "top_p": 0.8,
            "num_ctx": 16_384,
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            *({"role": turn.role, "content": turn.content} for turn in history),
            {"role": "user", "content": message},
        ]
        visible_content: list[str] = []
        created: list[Artifact] = []
        yield AgentEvent("status", {"phase": "started", "model": model})

        for round_index in range(self._max_tool_rounds + 1):
            round_content: list[str] = []
            calls: list[ToolCall] = []
            usage: dict[str, int] = {}
            async for chunk in self._model.stream_chat(
                model,
                messages,
                TOOL_SCHEMAS,
                options=model_options,
                think=think,
            ):
                if chunk.content:
                    round_content.append(chunk.content)
                    visible_content.append(chunk.content)
                    yield AgentEvent("delta", {"text": chunk.content})
                calls.extend(chunk.tool_calls)
                if chunk.usage:
                    usage = chunk.usage

            if "".join(round_content).lstrip().startswith("【参数补充请求】"):
                if created:
                    raise AgentError("已保存文件后不能再发出参数补充请求")
                yield AgentEvent(
                    "done",
                    {"content": "".join(visible_content), "model": model, "artifacts": [], "usage": usage},
                )
                return

            if not calls:
                content = "".join(visible_content)
                if not content.strip():
                    if not created:
                        raise AgentError("模型没有返回内容")
                    content = "策略文件已生成，请通过产物卡片预览或下载。"
                if not created and "【参数补充请求】" not in content:
                    raise AgentError("模型未保存策略文件，也未发出参数补充请求")
                yield AgentEvent(
                    "done",
                    {
                        "content": content,
                        "model": model,
                        "artifacts": [
                            {"artifact_id": item.artifact_id, "filename": item.filename}
                            for item in created
                        ],
                        "usage": usage,
                    },
                )
                return

            if round_index >= self._max_tool_rounds:
                raise AgentError(f"工具调用超过 {self._max_tool_rounds} 轮上限")

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(round_content),
                    "tool_calls": [
                        {"function": {"name": call.name, "arguments": call.arguments}}
                        for call in calls
                    ],
                }
            )
            for call in calls:
                yield AgentEvent("status", {"phase": "tool_started", "tool": call.name})
                try:
                    result, artifact = await self._call_tool(call, account_id)
                except (ValueError, OSError) as exc:
                    result, artifact = {"error": str(exc)}, None
                if artifact:
                    created.append(artifact)
                    yield AgentEvent(
                        "artifact",
                        {"artifact_id": artifact.artifact_id, "filename": artifact.filename},
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                yield AgentEvent("status", {"phase": "tool_finished", "tool": call.name})

        raise AgentError("Agent 未能完成请求")

    async def _call_tool(
        self, call: ToolCall, account_id: str
    ) -> tuple[dict[str, Any], Artifact | None]:
        args = call.arguments
        if call.name == "list_interface_files":
            return {"files": await self._interfaces.list_files()}, None
        if call.name == "read_interface_file":
            filename = args.get("filename")
            if not isinstance(filename, str):
                raise ValueError("filename 必须是文件名字符串")
            return {"filename": filename, "source": await self._interfaces.read_file(filename)}, None
        if call.name == "write_generated_file":
            filename, source = args.get("filename"), args.get("source")
            if not isinstance(filename, str) or not isinstance(source, str):
                raise ValueError("filename 和 source 必须是字符串")
            artifact = await self._artifacts.create(filename, source, account_id)
            return {"artifact_id": artifact.artifact_id, "filename": artifact.filename}, artifact
        raise ValueError(f"不支持的工具：{call.name}")
