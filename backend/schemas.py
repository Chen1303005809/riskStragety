"""HTTP request shapes; internal agent types stay in the agent package."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, field_validator


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ModelOptions(BaseModel):
    temperature: float = Field(default=0.3, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    num_ctx: int = Field(default=16_384, ge=512, le=131_072, strict=True)

    @field_validator("temperature", "top_p", mode="before")
    @classmethod
    def require_json_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=100)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    options: ModelOptions = Field(default_factory=ModelOptions)
    think: StrictBool = False


class DownloadRequest(BaseModel):
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")
