"""Assemble the shared Agent prompt and per-turn time context."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROMPT_FILE = Path(__file__).with_name("system_prompt.md")
TOOL_INSTRUCTIONS = """# 通用工具与 Skill 规则
- 只有当前上下文尚未提供适用 Skill 的入口说明时，才调用 `list_skills` 查找并按需调用 `load_skill`。如果系统消息已包含 `# Skill 指令：<skill_id>（SKILL.md）`，该 Skill 已加载；不要再次查询目录或加载相同入口文件。只在已加载说明明确链接到参考文件且当前任务需要时再加载该参考。
- `load_skill` 的完整内容会作为系统指令提供；工具结果是加载确认。遵循已提供说明中与任务相关的约束，不要重复读取相同资源。Skill 未覆盖的内容按通用规则和可用工具处理。
- 需求缺少会改变结果的必要信息时，用普通语言简洁追问，不要猜测关键数据或虚报工具结果。
- 只有写入工具成功返回后，才能声称产物已创建或更新。优先使用工具实际返回的文件名、状态和检测结果。
- 只调用本轮提供的工具；工具失败时阅读具体诊断再调整调用。"""


def build_system_prompt(now: datetime | None = None) -> str:
    """Keep shared guidance stable and append volatile Shanghai time last."""
    source = PROMPT_FILE.read_text(encoding="utf-8").removesuffix("\n")
    current = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    shanghai_time = current.astimezone(ZoneInfo("Asia/Shanghai"))
    date_text = shanghai_time.strftime("%Y-%m-%d %H:%M（%A）")
    stable_prompt = f"{source}\n\n{TOOL_INSTRUCTIONS}"
    current_time_suffix = (
        f"# 当前上海时间\n{date_text}。"
        "如果当前任务需要日期，以此时间为准。"
    )
    return f"{stable_prompt}\n\n{current_time_suffix}"
