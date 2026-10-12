"""Filesystem adapters have a narrow read root and immutable write root."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.adapters.files import LocalArtifactStore, LocalInterfaceReader


def test_interface_reader_lists_and_reads_only_direct_python_files(tmp_path: Path) -> None:
    interface_dir = tmp_path / "interfaces"
    interface_dir.mkdir()
    (interface_dir / "external_head.py").write_text("class RHTemplate: pass\n", encoding="utf-8")
    (interface_dir / "note.txt").write_text("hidden", encoding="utf-8")
    (interface_dir / "link.py").symlink_to(tmp_path / "secret.py")
    reader = LocalInterfaceReader(interface_dir)
    assert asyncio.run(reader.list_files()) == ["external_head.py"]
    assert "RHTemplate" in asyncio.run(reader.read_file("external_head.py"))
    for name in ("../secret.py", "link.py", "sub/file.py", "note.txt", ".hidden.py"):
        with pytest.raises(ValueError):
            asyncio.run(reader.read_file(name))


def test_artifacts_are_new_files_and_account_scoped(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = asyncio.run(store.create("risk.py", "x = 1\n", "alice"))
    second = asyncio.run(store.create("risk.py", "x = 2\n", "alice"))
    assert first.artifact_id != second.artifact_id
    assert asyncio.run(store.read(first.artifact_id, "alice")).content == b"x = 1\n"
    assert asyncio.run(store.read(second.artifact_id, "alice")).content == b"x = 2\n"
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read(first.artifact_id, "bob"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read("../risk.py", "alice"))
    metadata = json.loads((tmp_path / "artifacts" / f"{first.artifact_id}.json").read_text())
    assert metadata == {"filename": "risk.py", "account_id": "alice"}


def test_artifact_store_rejects_names_invalid_code_and_corrupt_metadata(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    for filename in ("../risk.py", "/tmp/risk.py", "risk.txt", ".hidden.py", "bad\nname.py"):
        with pytest.raises(ValueError):
            asyncio.run(store.create(filename, "x = 1\n", "alice"))
    with pytest.raises(ValueError, match="3.7"):
        asyncio.run(store.create("risk.py", "match 1:\n    case 1: pass\n", "alice"))
    artifact = asyncio.run(store.create("risk.py", "x = 1\n", "alice"))
    (tmp_path / "artifacts" / f"{artifact.artifact_id}.json").write_text("[]")
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read(artifact.artifact_id, "alice"))
