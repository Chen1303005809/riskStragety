"""Filesystem adapters have a narrow read root and immutable write root."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend.adapters.files import LocalArtifactStore


def test_artifact_store_writes_unique_immutable_sources(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = asyncio.run(store.create("risk.py", "x = 1\n", "alice"))
    second = asyncio.run(store.create("risk.py", "x = 2\n", "alice"))
    assert first.artifact_id != second.artifact_id
    assert asyncio.run(store.read_source(first.artifact_id)) == b"x = 1\n"
    assert asyncio.run(store.read_source(second.artifact_id)) == b"x = 2\n"
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_source("../risk.py"))


def test_artifact_store_rejects_names_without_python_37_grammar_gate(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    for filename in ("../risk.py", "/tmp/risk.py", "risk.txt", ".hidden.py", "bad\nname.py"):
        with pytest.raises(ValueError):
            asyncio.run(store.create(filename, "x = 1\n", "alice"))
    artifact = asyncio.run(
        store.create("risk.py", "match 1:\n    case 1: pass\n", "alice")
    )
    assert artifact.filename == "risk.py"
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_source("0" * 32))
