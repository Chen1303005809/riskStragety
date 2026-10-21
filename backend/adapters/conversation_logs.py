"""Append-only JSONL archives for complete agent conversation traces."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ConversationLogStore:
    """Store one append-only JSONL trace per conversation UUID."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._write_lock = threading.Lock()

    async def record(
        self,
        conversation_id: str,
        turn_id: str,
        event: str,
        data: dict[str, Any],
        *,
        flush: bool = False,
    ) -> None:
        await asyncio.to_thread(
            self._record, conversation_id, turn_id, event, data, flush
        )

    def _record(
        self,
        conversation_id: str,
        turn_id: str,
        event: str,
        data: dict[str, Any],
        flush: bool,
    ) -> None:
        normalized_id = str(UUID(conversation_id))
        if not event or not isinstance(data, dict):
            raise ValueError("conversation log event must have a name and object data")

        self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self._directory.is_symlink() or not self._directory.is_dir():
            raise OSError("conversation log directory is not a regular directory")
        os.chmod(self._directory, 0o700)
        path = self._directory / f"{normalized_id}.jsonl"
        record = {
            "timestamp": _now(),
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
