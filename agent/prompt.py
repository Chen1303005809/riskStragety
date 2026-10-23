"""Assemble the shared Agent prompt and per-turn time context."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROMPT_FILE = Path(__file__).with_name("system_prompt.md")
TOOL_INSTRUCTIONS = """# 通用工具与 Skill 规则
- 开始处理任务前调用 `list_skills` 查看可用 Skill；若有匹配项，先调用 `load_skill` 加载说明。只在 Skill 明确链接到参考文件且当前任务需要时再加载该 Markdown 参考。
- `load_skill` 返回的内容是当前任务的专用操作说明；遵循其中与任务相关的约束。Skill 未覆盖的内容按通用规则和可用工具处理。
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
