"""Append-only JSONL archives for complete agent conversation traces."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ConversationLogStore:
    """Store append-only JSONL traces grouped by account and conversation."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._write_lock = threading.Lock()

    def for_account(self, account_id: str) -> _AccountConversationLogRecorder:
        normalized_account_id = account_id.strip()
        if not normalized_account_id or len(normalized_account_id) > 128:
            raise ValueError("account log ID must contain between 1 and 128 characters")
        return _AccountConversationLogRecorder(self, normalized_account_id)

    async def record(
        self,
        account_id: str,
        conversation_id: str,
        turn_id: str,
        event: str,
        data: dict[str, Any],
        *,
        flush: bool = False,
    ) -> None:
        await asyncio.to_thread(
            self._record, account_id, conversation_id, turn_id, event, data, flush
        )

    def _record(
        self,
        account_id: str,
        conversation_id: str,
        turn_id: str,
        event: str,
        data: dict[str, Any],
        flush: bool,
    ) -> None:
        normalized_account_id = account_id.strip()
        if not normalized_account_id or len(normalized_account_id) > 128:
            raise ValueError("account log ID must contain between 1 and 128 characters")
        normalized_id = str(UUID(conversation_id))
        if not event or not isinstance(data, dict):
            raise ValueError("conversation log event must have a name and object data")

        _ensure_private_directory(self._directory)
        account_directory = self._directory / _account_directory_name(
            normalized_account_id
        )
        _ensure_private_directory(account_directory)
        path = account_directory / f"{normalized_id}.jsonl"
        record = {
            "timestamp": _now(),
            "account_id": normalized_account_id,
            "conversation_id": normalized_id,
            "turn_id": turn_id,
            "event": event,
            "data": data,
        }
        serialized = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        line = f"{serialized}\n".encode("utf-8")

        with self._write_lock:
            if path.is_symlink():
                raise OSError("conversation log file cannot be a symbolic link")
            if not path.exists():
                self._migrate_legacy_trace(
                    self._directory / f"{normalized_id}.jsonl",
                    path,
                    normalized_account_id,
                )
            if path.is_symlink():
                raise OSError("conversation log file cannot be a symbolic link")
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                written = 0
                while written < len(line):
                    written += os.write(descriptor, line[written:])
                if flush:
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)

    @staticmethod
    def _migrate_legacy_trace(
        legacy_path: Path, destination: Path, account_id: str
    ) -> None:
        if not legacy_path.exists():
            return
        if legacy_path.is_symlink():
            raise OSError("legacy conversation log cannot be a symbolic link")

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(legacy_path, flags)
        except FileNotFoundError:
            return

        has_matching_account = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                for line in source:
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return
                    if not isinstance(entry, dict):
                        return
                    logged_account_id = entry.get("account_id")
                    if logged_account_id is None and entry.get("event") == "turn_started":
                        data = entry.get("data")
                        if isinstance(data, dict):
                            logged_account_id = data.get("account_id")
                    if logged_account_id is not None:
                        if logged_account_id != account_id:
                            return
                        has_matching_account = True
        finally:
            os.close(descriptor)

        if not has_matching_account:
            return
        try:
            os.link(legacy_path, destination, follow_symlinks=False)
        except FileExistsError:
            return
        os.chmod(destination, 0o600)
        legacy_path.unlink()


class _AccountConversationLogRecorder:
    """Bind a conversation recorder to the account that owns its traces."""

    def __init__(self, store: ConversationLogStore, account_id: str) -> None:
        self._store = store
        self._account_id = account_id

    async def record(
        self,
        conversation_id: str,
        turn_id: str,
        event: str,
        data: dict[str, Any],
        *,
        flush: bool = False,
    ) -> None:
        await self._store.record(
            self._account_id,
            conversation_id,
            turn_id,
            event,
            data,
            flush=flush,
        )


def _account_directory_name(account_id: str) -> str:
    if account_id.casefold().startswith("guest"):
        return "guest"
    if re.fullmatch(r"[A-Za-z0-9_-]+", account_id):
        return account_id
    digest = hashlib.sha256(account_id.encode("utf-8")).hexdigest()
    return f"~{digest}"


def _ensure_private_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink() or not directory.is_dir():
        raise OSError("conversation log directory is not a regular directory")
    os.chmod(directory, 0o700)
