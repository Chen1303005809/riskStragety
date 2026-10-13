"""Filesystem adapters have a narrow read root and immutable write root."""

from __future__ import annotations

import asyncio
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


def test_artifact_store_writes_unique_immutable_sources(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = asyncio.run(store.create("risk.py", "x = 1\n", "alice"))
    second = asyncio.run(store.create("risk.py", "x = 2\n", "alice"))
    assert first.artifact_id != second.artifact_id
    assert asyncio.run(store.read_source(first.artifact_id)) == b"x = 1\n"
    assert asyncio.run(store.read_source(second.artifact_id)) == b"x = 2\n"
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_source("../risk.py"))


def test_artifact_store_rejects_names_and_invalid_python_code(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    for filename in ("../risk.py", "/tmp/risk.py", "risk.txt", ".hidden.py", "bad\nname.py"):
        with pytest.raises(ValueError):
            asyncio.run(store.create(filename, "x = 1\n", "alice"))
    with pytest.raises(ValueError, match="3.7"):
        asyncio.run(store.create("risk.py", "match 1:\n    case 1: pass\n", "alice"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_source("0" * 32))
