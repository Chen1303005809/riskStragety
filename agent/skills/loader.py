"""Discover and load Markdown skills from a confined directory tree."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
MAX_SKILL_BYTES = 128 * 1024
ENTRYPOINT = "SKILL.md"


@dataclass(frozen=True)
class SkillSummary:
    skill_id: str
    name: str
    description: str


@dataclass(frozen=True)
class Skill:
    skill_id: str
    instructions: str
    name: str = ""
    description: str = ""
    visibility: str = "agent"


class SkillLoadError(ValueError):
    """A skill is missing, malformed, or outside the configured skill root."""


class SkillLoader(Protocol):
    async def list_skills(self) -> list[SkillSummary]: ...

    async def load(self, skill_id: str) -> Skill: ...

    async def load_resource(self, skill_id: str, resource_path: str) -> str: ...


class FilesystemSkillLoader:
    """Load skill entrypoints and Markdown references without leaving ``root``."""

    def __init__(self, root: Path | None = None, *, max_bytes: int = MAX_SKILL_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self._root = root or Path(__file__).parent
        self._max_bytes = max_bytes

    async def list_skills(self) -> list[SkillSummary]:
        return await asyncio.to_thread(self._list_skills)

    def _list_skills(self) -> list[SkillSummary]:
        root = self._validated_root()
        summaries: list[SkillSummary] = []
        for skill_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if skill_dir.is_symlink() or not skill_dir.is_dir():
                continue
            entrypoint = skill_dir / ENTRYPOINT
            if not entrypoint.exists():
                continue
            skill = self._load(skill_dir.name)
            if skill.visibility == "internal":
                continue
            summaries.append(
                SkillSummary(skill.skill_id, skill.name, skill.description)
            )
        return summaries

    async def load(self, skill_id: str) -> Skill:
        return await asyncio.to_thread(self._load, skill_id)

    def _load(self, skill_id: str) -> Skill:
        skill_id = self._validate_skill_id(skill_id)
        content = self._read_confined_text(skill_id, ENTRYPOINT)
        metadata, instructions = self._parse_entrypoint(content, skill_id)
        return Skill(
            skill_id=skill_id,
            instructions=instructions,
            name=metadata["name"],
            description=metadata["description"],
            visibility=metadata.get("visibility", "agent"),
        )

    async def load_resource(self, skill_id: str, resource_path: str) -> str:
        return await asyncio.to_thread(
            self._load_resource, skill_id, resource_path
        )

    def _load_resource(self, skill_id: str, resource_path: str) -> str:
        skill_id = self._validate_skill_id(skill_id)
        relative_path = self._validate_resource_path(resource_path)
        if relative_path == ENTRYPOINT:
            return self._load(skill_id).instructions
        if Path(relative_path).suffix.lower() != ".md":
            raise SkillLoadError("Skill 参考资料只允许 Markdown 文件")
        return self._read_confined_text(skill_id, relative_path).strip()

    @staticmethod
    def _validate_skill_id(skill_id: str) -> str:
        if not isinstance(skill_id, str) or not SKILL_ID_PATTERN.fullmatch(skill_id):
            raise SkillLoadError("skill_id 格式无效")
        return skill_id

    @staticmethod
    def _validate_resource_path(resource_path: str) -> str:
        if not isinstance(resource_path, str) or "\\" in resource_path:
            raise SkillLoadError("resource_path 格式无效")
        path = PurePosixPath(resource_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise SkillLoadError("resource_path 必须位于 Skill 目录内")
        return path.as_posix()

    def _validated_root(self) -> Path:
        root = self._root
        if root.is_symlink() or not root.is_dir():
            raise SkillLoadError("Skill 目录不可用")
        return root

    def _read_confined_text(self, skill_id: str, relative_path: str) -> str:
        root = self._validated_root()
        skill_dir = root / skill_id
        if skill_dir.is_symlink() or not skill_dir.is_dir():
            raise SkillLoadError(f"Skill 不存在或不可读取：{skill_id}")

        candidate = skill_dir.joinpath(*PurePosixPath(relative_path).parts)
        current = skill_dir
        for part in PurePosixPath(relative_path).parts:
            current = current / part
            if current.is_symlink():
                raise SkillLoadError("Skill 文件不能是符号链接")
        if not candidate.is_file():
            raise SkillLoadError(f"Skill 文件不存在或不可读取：{relative_path}")

        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(skill_dir.resolve(strict=True))
            with resolved.open("rb") as handle:
                data = handle.read(self._max_bytes + 1)
        except (OSError, ValueError) as exc:
            raise SkillLoadError("Skill 文件不在允许的目录内或无法读取") from exc
        if len(data) > self._max_bytes:
            raise SkillLoadError("Skill 文件超过大小限制")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillLoadError("Skill 文件必须使用 UTF-8 编码") from exc

    @staticmethod
    def _parse_entrypoint(content: str, skill_id: str) -> tuple[dict[str, str], str]:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillLoadError("SKILL.md 必须以 YAML frontmatter 开始")
        try:
            end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
        except StopIteration as exc:
            raise SkillLoadError("SKILL.md frontmatter 缺少结束标记") from exc

        metadata: dict[str, str] = {}
        for line in lines[1:end]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition(":")
            if not separator or key.strip() not in {"name", "description", "visibility"}:
                raise SkillLoadError("SKILL.md frontmatter 只支持 name、description、visibility")
            normalized = value.strip().strip("\"'")
            if not normalized:
                raise SkillLoadError(f"SKILL.md frontmatter 的 {key.strip()} 不能为空")
            metadata[key.strip()] = normalized

        if not metadata.get("name") or not metadata.get("description"):
            raise SkillLoadError("SKILL.md frontmatter 必须包含 name 和 description")
        if metadata.get("visibility", "agent") not in {"agent", "internal"}:
            raise SkillLoadError("Skill visibility 只能是 agent 或 internal")
        instructions = "\n".join(lines[end + 1 :]).strip()
        if not instructions:
            raise SkillLoadError("Skill 内容不能为空")
        return metadata, instructions
