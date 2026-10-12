"""Restricted interface reading and persistent, immutable generated files."""

from __future__ import annotations

import ast
import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from agent.contracts import Artifact, ArtifactWriter
from backend.adapters.monitoring import MonitoringStore


MAX_INTERFACE_BYTES = 512 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024
ARTIFACT_ID = re.compile(r"^[0-9a-f]{32}$")


def _plain_python_name(filename: str) -> str:
    name = filename.strip()
    if (
        not name
        or len(name) > 120
        or name.startswith(".")
        or "/" in name
        or "\\" in name
        or not name.lower().endswith(".py")
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError("文件名必须是长度不超过 120 的纯 .py 文件名")
    return name


class LocalInterfaceReader:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    async def list_files(self) -> list[str]:
        return await asyncio.to_thread(self._list_files)

    def _list_files(self) -> list[str]:
        if self._directory.is_symlink() or not self._directory.is_dir():
            raise ValueError("接口目录不可用")
        return sorted(
            path.name
            for path in self._directory.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".py"
        )

    async def read_file(self, filename: str) -> str:
        return await asyncio.to_thread(self._read_file, filename)

    def _read_file(self, filename: str) -> str:
        name = _plain_python_name(filename)
        if self._directory.is_symlink() or not self._directory.is_dir():
            raise ValueError("接口目录不可用")
        path = self._directory / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self._directory.resolve():
            raise ValueError("接口文件不存在")
        with path.open("rb") as handle:
            data = handle.read(MAX_INTERFACE_BYTES + 1)
        if len(data) > MAX_INTERFACE_BYTES:
            raise ValueError("接口文件过大")
        return data.decode("utf-8")


@dataclass(frozen=True)
class StoredArtifact:
    filename: str
    content: bytes


class LocalArtifactStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    async def create(
        self, filename: str, source: str, account_id: str, conversation_id: str | None = None
    ) -> Artifact:
        task = asyncio.create_task(asyncio.to_thread(self._create, filename, source, account_id))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                artifact = await asyncio.shield(task)
            except Exception:
                pass
            else:
                await asyncio.shield(self.remove(artifact.artifact_id))
            raise

    def _create(self, filename: str, source: str, account_id: str) -> Artifact:
        name = _plain_python_name(filename)
        if not source.strip():
            raise ValueError("源码不能为空")
        content = source.encode("utf-8")
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError("生成文件超过 1 MiB")
        try:
            ast.parse(source, filename=name, feature_version=(3, 7))
        except SyntaxError as exc:
            raise ValueError(f"Python 3.7 语法检查失败：{exc.msg}，第 {exc.lineno} 行") from exc
        if self._directory.is_symlink():
            raise ValueError("产物目录不能是符号链接")
        self._directory.mkdir(parents=True, exist_ok=True)

        for _ in range(3):
            artifact_id = uuid.uuid4().hex
            source_path = self._directory / f"{artifact_id}.py"
            source_created = False
            try:
                with source_path.open("xb") as output:
                    source_created = True
                    output.write(content)
            except FileExistsError:
                if source_created:
                    source_path.unlink(missing_ok=True)
                continue
            except OSError:
                if source_created:
                    source_path.unlink(missing_ok=True)
                raise
            return Artifact(artifact_id, name, account_id)
        raise OSError("无法分配唯一产物 ID")

    async def read(self, artifact_id: str, account_id: str) -> StoredArtifact:
        return await asyncio.to_thread(self._read, artifact_id, account_id)

    async def read_source(self, artifact_id: str) -> bytes:
        return await asyncio.to_thread(self._read_source, artifact_id)

    def _read_source(self, artifact_id: str) -> bytes:
        if not ARTIFACT_ID.fullmatch(artifact_id) or self._directory.is_symlink():
            raise FileNotFoundError("产物不存在")
        source_path = self._directory / f"{artifact_id}.py"
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError("产物不存在")
        try:
            with source_path.open("rb") as handle:
                content = handle.read(MAX_ARTIFACT_BYTES + 1)
        except OSError as exc:
            raise FileNotFoundError("产物不存在") from exc
        if len(content) > MAX_ARTIFACT_BYTES:
            raise FileNotFoundError("产物不存在")
        return content

    async def remove(self, artifact_id: str) -> None:
        await asyncio.to_thread(self._remove, artifact_id)

    def _remove(self, artifact_id: str) -> None:
        if not ARTIFACT_ID.fullmatch(artifact_id) or self._directory.is_symlink():
            return
        (self._directory / f"{artifact_id}.py").unlink(missing_ok=True)
        (self._directory / f"{artifact_id}.json").unlink(missing_ok=True)


class IndexedArtifactWriter:
    """Write immutable source first, then index its ownership in SQLite."""

    def __init__(self, files: LocalArtifactStore, monitoring: MonitoringStore) -> None:
        self._files = files
        self._monitoring = monitoring

    async def _cleanup(self, artifact_id: str) -> None:
        try:
            await self._monitoring.remove_artifact_record(artifact_id)
        finally:
            await self._files.remove(artifact_id)

    async def create(
        self, filename: str, source: str, account_id: str, conversation_id: str | None = None
    ) -> Artifact:
        if conversation_id is None:
            raise ValueError("conversation_id is required for indexed artifact writes")
        artifact = await self._files.create(filename, source, account_id, conversation_id)
        indexing = asyncio.create_task(
            self._monitoring.record_artifact(
                artifact.artifact_id, artifact.filename, account_id, conversation_id
            )
        )
        try:
            await asyncio.shield(indexing)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(indexing)
            except Exception:
                pass
            await self._cleanup(artifact.artifact_id)
            raise
        except Exception:
            await self._cleanup(artifact.artifact_id)
            raise
        return artifact

    def _read(self, artifact_id: str, account_id: str) -> StoredArtifact:
        if not ARTIFACT_ID.fullmatch(artifact_id) or self._directory.is_symlink():
            raise FileNotFoundError("产物不存在")
        metadata_path = self._directory / f"{artifact_id}.json"
        source_path = self._directory / f"{artifact_id}.py"
        if metadata_path.is_symlink() or source_path.is_symlink():
            raise FileNotFoundError("产物不存在")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("account_id") != account_id:
                raise FileNotFoundError("产物不存在")
            name = _plain_python_name(metadata["filename"])
            with source_path.open("rb") as handle:
                content = handle.read(MAX_ARTIFACT_BYTES + 1)
            if len(content) > MAX_ARTIFACT_BYTES:
                raise FileNotFoundError("产物不存在")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise FileNotFoundError("产物不存在") from exc
        return StoredArtifact(name, content)
