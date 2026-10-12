"""Load the preserved domain prompt and add per-turn operating context."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROMPT_FILE = Path(__file__).with_name("system_prompt.md")
TOOL_INSTRUCTIONS = """# 工具执行规则
- 生成策略之前，先调用 list_interface_files，再用 read_interface_file 阅读接口源码，尤其是 external_head.py 与 external_baseid.py。接口示例只是契约参考，不是用户需求。
- 必要参数缺失时按上文格式追问；这一轮不得调用 write_generated_file。
- 需求完整时，先调用 write_generated_file 保存最终、完整的单文件 Python 策略，再按上文格式回复。写工具的 source 必须与回复中的 Python 代码块一致；文件名由继承 RHTemplate 的策略类名自动生成，不传 filename。
- 只能使用提供的接口读取工具和生成文件写入工具。工具报错时根据错误调整调用；不要猜测读取结果或声称已保存失败的文件。"""


def build_system_prompt(now: datetime | None = None) -> str:
    """Keep the old prompt unchanged while supplying its required date and tools."""
    source = PROMPT_FILE.read_text(encoding="utf-8").removesuffix("\n")
    current = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    shanghai_time = current.astimezone(ZoneInfo("Asia/Shanghai"))
    date_text = shanghai_time.strftime("%Y-%m-%d %H:%M（%A）")
    return (
        f"{source}\n\n# 当前时间\n当前日期：{date_text}。"
        "代码文件头、元数据及所有输出中的日期必须使用该当天日期。"
        f"\n\n{TOOL_INSTRUCTIONS}"
    )
