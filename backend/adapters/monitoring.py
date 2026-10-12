"""SQLite-backed accounts, conversations, messages, and artifact index."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_ID = re.compile(r"^[0-9a-f]{32}$")
LEGACY_METADATA_MAX_BYTES = 16 * 1024
SCHEMA_V1 = (
    """CREATE TABLE accounts (
        account_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE conversations (
        conversation_id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL REFERENCES accounts(account_id),
        title TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('idle', 'active', 'completed', 'failed', 'interrupted')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX conversations_account_updated ON conversations(account_id, updated_at DESC, conversation_id ASC)",
    """CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
        sequence INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'failed', 'interrupted')),
        created_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE (conversation_id, sequence)
    )""",
    "CREATE INDEX messages_conversation_sequence ON messages(conversation_id, sequence ASC)",
    """CREATE TABLE artifacts (
        artifact_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        account_id TEXT NOT NULL REFERENCES accounts(account_id),
        conversation_id TEXT REFERENCES conversations(conversation_id),
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX artifacts_account_created ON artifacts(account_id, created_at DESC, artifact_id ASC)",
    "CREATE INDEX artifacts_conversation_created ON artifacts(conversation_id, created_at DESC, artifact_id ASC)",
    "CREATE TABLE monitoring_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)


class ConversationNotFound(LookupError):
    """The conversation does not exist for the requested account."""


class AccountNotFound(LookupError):
    """The requested account does not exist."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp(path: Path) -> str:
    try:
        value = path.stat().st_mtime
    except OSError:
        return _now()
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class MonitoringStore:
    """Small synchronous SQLite operations exposed as async methods."""

    def __init__(self, database_path: Path, artifact_directory: Path) -> None:
        self._path = database_path
        self._artifact_directory = artifact_directory
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()
        self._import_legacy_artifacts()
        self._recover_incomplete_turns()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version < 1:
                for statement in SCHEMA_V1:
                    connection.execute(statement)
                connection.execute("PRAGMA user_version = 1")
            elif version != 1:
                connection.rollback()
                raise RuntimeError(f"不支持的监控数据库版本：{version}")
            connection.commit()

    def _import_legacy_artifacts(self) -> None:
        if self._artifact_directory.is_symlink() or not self._artifact_directory.is_dir():
            return
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            imported = connection.execute(
                "SELECT value FROM monitoring_meta WHERE key = 'legacy_artifacts_imported'"
            ).fetchone()
            if imported:
                connection.commit()
                return
            for metadata_path in sorted(self._artifact_directory.glob("*.json")):
                artifact_id = metadata_path.stem
                if (
                    not ARTIFACT_ID.fullmatch(artifact_id)
                    or metadata_path.is_symlink()
                    or not metadata_path.is_file()
                ):
                    continue
                source_path = self._artifact_directory / f"{artifact_id}.py"
                if source_path.is_symlink() or not source_path.is_file():
                    continue
                try:
                    if metadata_path.stat().st_size > LEGACY_METADATA_MAX_BYTES:
                        raise ValueError("metadata too large")
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    account_id = metadata.get("account_id")
                    filename = metadata.get("filename")
                    if not isinstance(account_id, str) or not isinstance(filename, str):
                        raise ValueError("missing owner or filename")
                    account_id = account_id.strip()
                    if not account_id or len(account_id) > 128 or not filename or len(filename) > 120:
                        raise ValueError("invalid owner or filename")
                    if (
                        filename.startswith(".")
                        or "/" in filename
                        or "\\" in filename
                        or not filename.lower().endswith(".py")
                        or any(ord(char) < 32 or ord(char) == 127 for char in filename)
                    ):
                        raise ValueError("invalid filename")
                except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                    logging.warning("跳过无法迁移的旧产物元数据 %s：%s", metadata_path.name, exc)
                    continue
                created_at = _timestamp(metadata_path)
                connection.execute(
                    "INSERT OR IGNORE INTO accounts(account_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (account_id, created_at, created_at),
                )
                connection.execute(
                    """INSERT OR IGNORE INTO artifacts
                       (artifact_id, filename, account_id, conversation_id, created_at)
                       VALUES (?, ?, ?, NULL, ?)""",
                    (artifact_id, filename, account_id, created_at),
                )
            connection.execute(
                "INSERT INTO monitoring_meta(key, value) VALUES ('legacy_artifacts_imported', '1')"
            )
            connection.commit()

    def _recover_incomplete_turns(self) -> None:
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE messages SET status = 'interrupted', completed_at = ? WHERE status = 'pending'",
                (now,),
            )
            connection.execute(
                "UPDATE conversations SET status = 'interrupted', updated_at = ? WHERE status = 'active'",
                (now,),
            )
            connection.commit()

    async def prepare_conversation(self, account_id: str, conversation_id: str, title: str) -> None:
        await asyncio.to_thread(self._prepare_conversation, account_id, conversation_id, title)

    def _prepare_conversation(self, account_id: str, conversation_id: str, title: str) -> None:
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO accounts(account_id, created_at, updated_at)
                   VALUES (?, ?, ?)""",
                (account_id, now, now),
            )
            row = connection.execute(
                "SELECT account_id FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            if row and row["account_id"] != account_id:
                connection.rollback()
                raise ConversationNotFound(conversation_id)
            if not row:
                connection.execute(
                    """INSERT INTO conversations
                       (conversation_id, account_id, title, status, created_at, updated_at)
                       VALUES (?, ?, ?, 'idle', ?, ?)""",
                    (conversation_id, account_id, title[:120], now, now),
                )
            connection.commit()

    async def begin_turn(
        self, account_id: str, conversation_id: str, message: str
    ) -> tuple[list[tuple[str, str]], int, int]:
        task = asyncio.create_task(
            asyncio.to_thread(self._begin_turn, account_id, conversation_id, message)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                _, _, assistant_message_id = await asyncio.shield(task)
            except Exception:
                pass
            else:
                finish_task = asyncio.create_task(
                    self.finish_turn(conversation_id, assistant_message_id, "", "interrupted")
                )
                await asyncio.shield(finish_task)
            raise

    def _begin_turn(
        self, account_id: str, conversation_id: str, message: str
    ) -> tuple[list[tuple[str, str]], int, int]:
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            conversation = connection.execute(
                "SELECT account_id FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            if not conversation or conversation["account_id"] != account_id:
                connection.rollback()
                raise ConversationNotFound(conversation_id)
            history = [
                (row["role"], row["content"])
                for row in connection.execute(
                    """SELECT role, content FROM messages
                       WHERE conversation_id = ? AND status = 'complete'
                       ORDER BY sequence ASC""",
                    (conversation_id,),
                )
            ]
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), -1) + 1 FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO messages
                   (conversation_id, sequence, role, content, status, created_at, completed_at)
                   VALUES (?, ?, 'user', ?, 'complete', ?, ?)""",
                (conversation_id, sequence, message, now, now),
            )
            user_message_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            connection.execute(
                """INSERT INTO messages
                   (conversation_id, sequence, role, content, status, created_at)
                   VALUES (?, ?, 'assistant', '', 'pending', ?)""",
                (conversation_id, sequence + 1, now),
            )
            assistant_message_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            connection.execute(
                "UPDATE conversations SET status = 'active', updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            connection.execute(
                "UPDATE accounts SET updated_at = ? WHERE account_id = ?", (now, account_id)
            )
            connection.commit()
        return history, user_message_id, assistant_message_id

    async def finish_turn(
        self,
        conversation_id: str,
        assistant_message_id: int,
        content: str,
        status: str,
    ) -> None:
        await asyncio.to_thread(
            self._finish_turn, conversation_id, assistant_message_id, content, status
        )

    def _finish_turn(self, conversation_id: str, assistant_message_id: int, content: str, status: str) -> None:
        if status not in {"complete", "failed", "interrupted"}:
            raise ValueError("invalid message status")
        now = _now()
        conversation_status = "completed" if status == "complete" else status
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE messages SET content = ?, status = ?, completed_at = ?
                   WHERE id = ? AND conversation_id = ? AND role = 'assistant' AND status = 'pending'""",
                (content, status, now, assistant_message_id, conversation_id),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE conversations SET status = ?, updated_at = ? WHERE conversation_id = ?",
                    (conversation_status, now, conversation_id),
                )
                connection.execute(
                    """UPDATE accounts SET updated_at = ?
                       WHERE account_id = (SELECT account_id FROM conversations WHERE conversation_id = ?)""",
                    (now, conversation_id),
                )
            connection.commit()

    async def record_artifact(
        self,
        artifact_id: str,
        filename: str,
        account_id: str,
        conversation_id: str,
    ) -> None:
        await asyncio.to_thread(
            self._record_artifact, artifact_id, filename, account_id, conversation_id
        )

    def _record_artifact(
        self, artifact_id: str, filename: str, account_id: str, conversation_id: str
    ) -> None:
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT account_id FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            if not row or row["account_id"] != account_id:
                connection.rollback()
                raise ConversationNotFound(conversation_id)
            connection.execute(
                """INSERT INTO artifacts
                   (artifact_id, filename, account_id, conversation_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (artifact_id, filename, account_id, conversation_id, now),
            )
            connection.commit()

    async def remove_artifact_record(self, artifact_id: str) -> None:
        await asyncio.to_thread(self._remove_artifact_record, artifact_id)

    def _remove_artifact_record(self, artifact_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            connection.commit()

    async def artifact_for_account(
        self, artifact_id: str, account_id: str
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._artifact_for_account, artifact_id, account_id)

    def _artifact_for_account(
        self, artifact_id: str, account_id: str
    ) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT artifact_id, filename, account_id, created_at, conversation_id
                   FROM artifacts WHERE artifact_id = ? AND account_id = ?""",
                (artifact_id, account_id),
            ).fetchone()
        return dict(row) if row else None

    async def accounts(self, offset: int, limit: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._accounts, offset, limit)

    def _accounts(self, offset: int, limit: int) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0])
            rows = connection.execute(
                """SELECT a.account_id, a.created_at, a.updated_at,
                          (SELECT COUNT(*) FROM conversations c WHERE c.account_id = a.account_id) AS conversation_count,
                          (SELECT COUNT(*) FROM messages m JOIN conversations c USING (conversation_id)
                           WHERE c.account_id = a.account_id) AS message_count,
                          (SELECT COUNT(*) FROM artifacts f WHERE f.account_id = a.account_id) AS artifact_count
                   FROM accounts a ORDER BY a.updated_at DESC, a.account_id ASC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return {"items": [dict(row) for row in rows], "offset": offset, "limit": limit, "total": total}

    async def conversations(
        self, account_id: str, offset: int, limit: int, orphan_offset: int, orphan_limit: int
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._conversations, account_id, offset, limit, orphan_offset, orphan_limit
        )

    def _conversations(
        self, account_id: str, offset: int, limit: int, orphan_offset: int, orphan_limit: int
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            if not connection.execute("SELECT 1 FROM accounts WHERE account_id = ?", (account_id,)).fetchone():
                raise AccountNotFound(account_id)
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM conversations WHERE account_id = ?", (account_id,)
                ).fetchone()[0]
            )
            rows = connection.execute(
                """SELECT c.conversation_id, c.title, c.status, c.created_at, c.updated_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count,
                          (SELECT COUNT(*) FROM artifacts a WHERE a.conversation_id = c.conversation_id) AS artifact_count
                   FROM conversations c WHERE c.account_id = ?
                   ORDER BY c.updated_at DESC, c.conversation_id ASC LIMIT ? OFFSET ?""",
                (account_id, limit, offset),
            ).fetchall()
            orphan_total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE account_id = ? AND conversation_id IS NULL",
                    (account_id,),
                ).fetchone()[0]
            )
            orphan_rows = connection.execute(
                """SELECT artifact_id, filename, created_at FROM artifacts
                   WHERE account_id = ? AND conversation_id IS NULL
                   ORDER BY created_at DESC, artifact_id ASC LIMIT ? OFFSET ?""",
                (account_id, orphan_limit, orphan_offset),
            ).fetchall()
        return {
            "items": [dict(row) for row in rows],
            "offset": offset,
            "limit": limit,
            "total": total,
            "unassigned_artifacts": [dict(row) for row in orphan_rows],
            "unassigned_offset": orphan_offset,
            "unassigned_limit": orphan_limit,
            "unassigned_total": orphan_total,
        }

    async def conversation_detail(
        self,
        account_id: str,
        conversation_id: str,
        message_offset: int,
        message_limit: int,
        artifact_offset: int,
        artifact_limit: int,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._conversation_detail,
            account_id,
            conversation_id,
            message_offset,
            message_limit,
            artifact_offset,
            artifact_limit,
        )

    def _conversation_detail(
        self,
        account_id: str,
        conversation_id: str,
        message_offset: int,
        message_limit: int,
        artifact_offset: int,
        artifact_limit: int,
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            conversation = connection.execute(
                """SELECT c.conversation_id, c.account_id, c.title, c.status, c.created_at, c.updated_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count,
                          (SELECT COUNT(*) FROM artifacts a WHERE a.conversation_id = c.conversation_id) AS artifact_count
                   FROM conversations c WHERE c.conversation_id = ? AND c.account_id = ?""",
                (conversation_id, account_id),
            ).fetchone()
            if not conversation:
                raise ConversationNotFound(conversation_id)
            message_total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
                ).fetchone()[0]
            )
            messages = connection.execute(
                """SELECT id, sequence, role, content, status, created_at, completed_at
                   FROM messages WHERE conversation_id = ? ORDER BY sequence ASC LIMIT ? OFFSET ?""",
                (conversation_id, message_limit, message_offset),
            ).fetchall()
            artifact_total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE conversation_id = ?", (conversation_id,)
                ).fetchone()[0]
            )
            artifacts = connection.execute(
                """SELECT artifact_id, filename, created_at FROM artifacts
                   WHERE conversation_id = ? ORDER BY created_at DESC, artifact_id ASC LIMIT ? OFFSET ?""",
                (conversation_id, artifact_limit, artifact_offset),
            ).fetchall()
        return {
            "conversation": dict(conversation),
            "messages": [dict(row) for row in messages],
            "message_offset": message_offset,
            "message_limit": message_limit,
            "message_total": message_total,
            "artifacts": [dict(row) for row in artifacts],
            "artifact_offset": artifact_offset,
            "artifact_limit": artifact_limit,
            "artifact_total": artifact_total,
        }
