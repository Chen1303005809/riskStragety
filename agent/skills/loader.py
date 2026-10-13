"""Load reusable Agent skills from a confined directory tree."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_SKILL_BYTES = 128 * 1024


@dataclass(frozen=True)
class Skill:
    skill_id: str
    instructions: str


class SkillLoadError(ValueError):
    """A skill is missing, malformed, or outside the configured skill root."""


class SkillLoader(Protocol):
    async def load(self, skill_id: str) -> Skill: ...


class FilesystemSkillLoader:
    """Load ``<root>/<skill_id>/SKILL.md`` without allowing path traversal."""

    def __init__(self, root: Path | None = None, *, max_bytes: int = MAX_SKILL_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self._root = root or Path(__file__).parent
        self._max_bytes = max_bytes

    async def load(self, skill_id: str) -> Skill:
        return await asyncio.to_thread(self._load, skill_id)

    def _load(self, skill_id: str) -> Skill:
        if not isinstance(skill_id, str) or not SKILL_ID_PATTERN.fullmatch(skill_id):
            raise SkillLoadError("skill_id 格式无效")

        root = self._root
        if root.is_symlink() or not root.is_dir():
            raise SkillLoadError("Skill 目录不可用")

        skill_dir = root / skill_id
        skill_file = skill_dir / "SKILL.md"
        if skill_dir.is_symlink() or skill_file.is_symlink() or not skill_file.is_file():
            raise SkillLoadError(f"Skill 不存在或不可读取：{skill_id}")

        root_resolved = root.resolve()
        try:
            skill_resolved = skill_file.resolve(strict=True)
            skill_resolved.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise SkillLoadError("Skill 文件不在允许的目录内") from exc

        try:
            with skill_resolved.open("rb") as handle:
                data = handle.read(self._max_bytes + 1)
        except OSError as exc:
            raise SkillLoadError(f"无法读取 Skill：{skill_id}") from exc
        if len(data) > self._max_bytes:
            raise SkillLoadError("Skill 文件超过大小限制")
        try:
            instructions = data.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise SkillLoadError("Skill 文件必须使用 UTF-8 编码") from exc
        if not instructions:
            raise SkillLoadError("Skill 内容不能为空")
        return Skill(skill_id=skill_id, instructions=instructions)
