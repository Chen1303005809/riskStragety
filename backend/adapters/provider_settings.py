"""Persistent settings for external model providers."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path


class ProviderSettingsStore:
    """Store provider credentials separately from chat history and never expose them."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.is_symlink():
            raise ValueError("模型服务配置数据库不能是符号链接")
        descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        with closing(sqlite3.connect(self._path, timeout=30)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS provider_settings "
                "(provider TEXT PRIMARY KEY, api_key TEXT NOT NULL)"
            )
            connection.commit()
        self._path.chmod(0o600)

    def get_deepseek_api_key(self) -> str | None:
        with closing(sqlite3.connect(self._path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT api_key FROM provider_settings WHERE provider = ?",
                ("deepseek",),
            ).fetchone()
        return row[0] if row else None

    def set_deepseek_api_key(self, api_key: str) -> None:
        with closing(sqlite3.connect(self._path, timeout=30)) as connection:
            connection.execute(
                "INSERT INTO provider_settings(provider, api_key) VALUES (?, ?) "
                "ON CONFLICT(provider) DO UPDATE SET api_key = excluded.api_key",
                ("deepseek", api_key),
            )
            connection.commit()
        self._path.chmod(0o600)

    def delete_deepseek_api_key(self) -> None:
        with closing(sqlite3.connect(self._path, timeout=30)) as connection:
            connection.execute(
                "DELETE FROM provider_settings WHERE provider = ?", ("deepseek",)
            )
            connection.commit()
        self._path.chmod(0o600)
