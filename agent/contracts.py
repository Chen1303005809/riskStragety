"""The agent's small interface to its callers and external resources."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelChunk:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    filename: str
    account_id: str


@dataclass(frozen=True)
class AgentEvent:
    kind: Literal["status", "delta", "artifact", "done", "error"]
    data: dict[str, Any]


class ModelGateway(Protocol):
    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        options: dict[str, int | float],
        think: bool,
    ) -> AsyncIterator[ModelChunk]: ...

    async def list_models(self) -> list[str]: ...


class InterfaceReader(Protocol):
    async def list_files(self) -> list[str]: ...

    async def read_file(self, filename: str) -> str: ...


class ArtifactWriter(Protocol):
    async def create(
        self, filename: str, source: str, account_id: str, conversation_id: str | None = None
    ) -> Artifact: ...
