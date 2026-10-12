"""Environment-owned service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    api_key: str
    host: str = "0.0.0.0"
    port: int = 8765
    cors_origins: tuple[str, ...] = ()
    project_root: Path = PROJECT_ROOT

    @classmethod
    def from_env(cls) -> Settings:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL 必须是有效的 http(s) 地址")
        api_key = os.environ.get("AGENT_API_KEY", "")
        if not api_key or api_key == "replace-with-a-long-random-secret":
            raise ValueError("启动前必须设置非示例的 AGENT_API_KEY")
        model = os.environ.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise ValueError("启动前必须设置 OLLAMA_MODEL")
        try:
            port = int(os.environ.get("APP_PORT", "8765"))
        except ValueError as exc:
            raise ValueError("APP_PORT 必须是整数") from exc
        if not 1 <= port <= 65535:
            raise ValueError("APP_PORT 必须在 1-65535 之间")
        cors_origins = tuple(
            origin.strip()
            for origin in os.environ.get("CORS_ORIGINS", "").split(",")
            if origin.strip()
        )
        return cls(
            ollama_base_url=base_url,
            ollama_model=model,
            api_key=api_key,
            host=os.environ.get("APP_HOST", "0.0.0.0"),
            port=port,
            cors_origins=cors_origins,
        )
