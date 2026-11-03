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

    def for_account(
        self, account_id: str, *, redact_sensitive: bool = False
    ) -> _AccountConversationLogRecorder:
        normalized_account_id = account_id.strip()
        if not normalized_account_id or len(normalized_account_id) > 128:
            raise ValueError("account log ID must contain between 1 and 128 characters")
        return _AccountConversationLogRecorder(
            self, normalized_account_id, redact_sensitive=redact_sensitive
        )

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

    def __init__(
        self, store: ConversationLogStore, account_id: str, *, redact_sensitive: bool = False
    ) -> None:
        self._store = store
        self._account_id = account_id
        self._redact_sensitive = redact_sensitive

    async def record(
        self,
        conversation_id: str,
        turn_id: str,
        event: str,
        data: dict[str, Any],
        *,
        flush: bool = False,
    ) -> None:
        if self._redact_sensitive:
            data = _redact_sensitive_log_data(event, data)
        await self._store.record(
            self._account_id,
            conversation_id,
            turn_id,
            event,
            data,
            flush=flush,
        )


def _redact_sensitive_log_data(event: str, data: dict[str, Any]) -> dict[str, Any]:
    """Keep operational metadata without persisting workbook-derived values."""
    safe: dict[str, Any] = {"sensitive_content_redacted": True}
    for key in (
        "round", "call_index", "call_id", "name", "phase", "model", "provider",
        "options", "think", "usage", "status", "exception_type", "ErrCode",
    ):
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)) or (
            key in {"options", "usage"} and isinstance(value, dict)
        ):
            safe[key] = value

    if event == "turn_started":
        history = data.get("history")
        safe["history"] = [
            {
                "role": item.get("role"),
                "content_chars": len(item.get("content", ""))
                if isinstance(item.get("content"), str)
                else 0,
            }
            for item in history
            if isinstance(item, dict)
        ] if isinstance(history, list) else []
        safe["user_message_chars"] = len(data.get("user_message", "")) if isinstance(
            data.get("user_message"), str
        ) else 0
    elif event == "turn_finished":
        safe["content_chars"] = len(data.get("content", "")) if isinstance(
            data.get("content"), str
        ) else 0
        safe["artifact_count"] = len(data.get("artifacts", [])) if isinstance(
            data.get("artifacts"), list
        ) else 0
    elif event in {"model_request", "functional_test_model_request"}:
        messages = data.get("messages")
        safe["messages"] = [
            {
                "role": item.get("role"),
                "tool_name": item.get("tool_name"),
                "content_chars": len(item.get("content", ""))
                if isinstance(item.get("content"), str)
                else 0,
            }
            for item in messages
            if isinstance(item, dict)
        ] if isinstance(messages, list) else []
        safe["tool_names"] = [
            tool.get("function", {}).get("name")
            for tool in data.get("tools", [])
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        ] if isinstance(data.get("tools"), list) else []
    elif event in {"model_chunk", "functional_test_model_chunk"}:
        safe["content_chars"] = len(data.get("content", "")) if isinstance(
            data.get("content"), str
        ) else 0
        safe["reasoning_chars"] = len(data.get("reasoning_content", "")) if isinstance(
            data.get("reasoning_content"), str
        ) else 0
        calls = data.get("tool_calls")
        safe["tool_calls"] = [
            {"name": item.get("name"), "call_id": item.get("call_id")}
            for item in calls
            if isinstance(item, dict)
        ] if isinstance(calls, list) else []
    elif event == "tool_call_started":
        arguments = data.get("arguments")
        if data.get("name") == "read_excel" and isinstance(arguments, dict):
            safe["arguments"] = {
                key: arguments[key]
                for key in ("file_id", "sheet_name", "cell_range")
                if isinstance(arguments.get(key), str)
            }
        else:
            safe["argument_keys"] = sorted(arguments) if isinstance(arguments, dict) else []
    elif event == "tool_result":
        result = data.get("result")
        if data.get("name") == "read_excel" and isinstance(result, dict):
            for key in ("file_id", "sheet_name", "range", "row_count", "column_count"):
                value = result.get(key)
                if isinstance(value, (str, int, float, bool)):
                    safe[key] = value
            safe["outcome"] = "error" if result.get("error") else "ok"
        elif data.get("name") == "list_excel_files" and isinstance(result, dict):
            files = result.get("files")
            safe["file_count"] = len(files) if isinstance(files, list) else 0
            safe["outcome"] = "error" if result.get("error") else "ok"
        else:
            safe["result_keys"] = sorted(result) if isinstance(result, dict) else []
            safe["outcome"] = (
                "error" if isinstance(result, dict) and result.get("error") else "ok"
            )
    elif event in {"functional_test_cases_built", "functional_sandbox_result"}:
        cases = data.get("cases")
        safe["case_count"] = len(cases) if isinstance(cases, list) else 0
        if isinstance(cases, list):
            safe["passed_count"] = sum(
                1 for case in cases if isinstance(case, dict) and case.get("passed") is True
            )
    elif event == "functional_sandbox_error":
        safe["outcome"] = "error"
    return safe


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
