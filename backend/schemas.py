"""HTTP request shapes; internal agent types stay in the agent package."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=100)
    model: str | None = Field(default=None, min_length=1, max_length=256)


class DownloadRequest(BaseModel):
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")
