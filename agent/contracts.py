"""The agent's small interface to its callers and external resources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


class AgentMessage:
    """Marker base for provider-neutral messages exchanged with a model."""

    role: str
    content: str


@dataclass(frozen=True)
class SystemMessage(AgentMessage):
    content: str
    role: Literal["system"] = field(default="system", init=False)


@dataclass(frozen=True)
class UserMessage(AgentMessage):
    content: str
    role: Literal["user"] = field(default="user", init=False)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class AssistantToolCall:
    """A tool call stored in conversation history with a stable correlation ID."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantMessage(AgentMessage):
    content: str
    tool_calls: tuple[AssistantToolCall, ...] = ()
    reasoning_content: str = ""
    role: Literal["assistant"] = field(default="assistant", init=False)


@dataclass(frozen=True)
class ToolResultMessage(AgentMessage):
    content: str
    call_id: str
    tool_name: str
    role: Literal["tool"] = field(default="tool", init=False)


ModelMessage: TypeAlias = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True)
class ModelChunk:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str = ""


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    filename: str
    account_id: str


@dataclass(frozen=True)
class AgentEvent:
    kind: Literal["status", "delta", "artifact", "done", "error"]
    data: dict[str, Any]


class ModelGateway(ABC):
    """Common gateway interface implemented by concrete model providers."""

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: Sequence[ModelMessage],
        tools: list[dict[str, Any]],
        *,
        options: dict[str, int | float],
        think: bool,
    ) -> AsyncIterator[ModelChunk]:
        raise NotImplementedError

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    async def close(self) -> None:
        """Release provider resources; lightweight test gateways may do nothing."""


class InterfaceReader(Protocol):
    async def list_files(self) -> list[str]: ...

    async def read_file(self, filename: str) -> str: ...


class ArtifactWriter(Protocol):
    async def create(
        self, filename: str, source: str, account_id: str, conversation_id: str | None = None
    ) -> Artifact: ...


class FunctionalTester(Protocol):
    async def run(
        self,
        source: str,
        *,
        requirements: Sequence[ChatTurn],
        model: str,
        options: dict[str, int | float],
    ) -> dict[str, Any]: ...


class ConversationRecorder(Protocol):
    async def record(
        self,
        conversation_id: str,
        turn_id: str,
        event: str,
        data: dict[str, Any],
        *,
        flush: bool = False,
    ) -> None: ...
