"""HTTP request shapes; internal agent types stay in the agent package."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, StrictBool, StrictStr, field_validator


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
    account_id: StrictStr = Field(min_length=1, max_length=128)
    conversation_id: UUID
    message: str = Field(min_length=1, max_length=20_000)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    options: ModelOptions = Field(default_factory=ModelOptions)
    think: StrictBool = False

    @field_validator("account_id")
    @classmethod
    def require_nonblank_account_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("account_id must not be blank")
        return normalized


class DownloadRequest(BaseModel):
    account_id: StrictStr = Field(min_length=1, max_length=128)
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")

    @field_validator("account_id")
    @classmethod
    def require_nonblank_account_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("account_id must not be blank")
        return normalized
