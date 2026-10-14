"""Load the preserved domain prompt and add per-turn operating context."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROMPT_FILE = Path(__file__).with_name("system_prompt.md")
TOOL_INSTRUCTIONS = """# 工具执行规则
- 用户给出具体期货合约代码但未写交易所时，必须先调用 lookup_exchange_by_contract_code 查询；查询成功后沿用返回的交易所代码继续处理，不得再问用户交易所。生成源码时，必须把结果中的 canonical_contract_code 作为唯一合约代码来源，逐字保留其大小写；contract_code 是统一大写的识别字段，禁止用于生成源码。用户已明确给出交易所时沿用用户提供的信息，不调用查询工具。
- 查询结果 status 为 unrecognized 时，只询问用户该合约所属交易所；status 为 expired 时，说明交割月已过期并要求用户提供有效合约代码；status 为 invalid_format 但已识别交易所时，要求用户补充完整且格式正确的合约代码。
- 查询工具只支持境内六家交易所期货，不支持期权。用户只给中文品种名、“主力”或没有具体交割月的品种描述时，要求提供具体合约代码；不要猜品种代码。
- 生成策略之前，先调用 list_interface_files，再用 read_interface_file 阅读接口源码，尤其是 external_head.py 与 external_baseid.py。接口示例只是契约参考，不是用户需求。
- 完整策略源码只能通过工具调用在内部传递、检查并写入文件产物；面向用户的聊天回复不得包含完整策略源码，也不得通过分段内容拼出完整源码。工具调用中的 source 参数不受此限制。
- 必要参数缺失时按上文格式追问；这一轮不得调用任何代码检测或写入工具。
- 需求完整时，先对候选源码调用 check_python_syntax。只有通过后，才能对完全相同的 source 调用 run_functional_test；该工具会单独调用测试用例 Skill 构造用例并执行沙盒测试。
- 语法或功能检测失败时，根据诊断修改源码，再重新运行语法检测和功能检测。不要自行替代 Skill 编造测试结果。
- 如果历史中存在包含「【上轮失败诊断】」的助手消息，必须先读取其中的工具失败结果和最终错误，结合原始用户需求针对性修正；不要忽略诊断或原样重复失败调用。
- 仅当同一份源码通过两项检测后，才调用 write_generated_file；写工具的 source 必须与检测通过的源码一致。文件名由继承 RHTemplate 的策略类名自动生成，不传 filename。
- 只能使用提供的工具。工具报错时根据错误调整调用；不要猜测读取结果或声称已保存失败的文件。"""


def build_system_prompt(now: datetime | None = None) -> str:
    """Assemble the domain prompt with current-time and tool-use instructions."""
    source = PROMPT_FILE.read_text(encoding="utf-8").removesuffix("\n")
    current = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    shanghai_time = current.astimezone(ZoneInfo("Asia/Shanghai"))
    date_text = shanghai_time.strftime("%Y-%m-%d %H:%M（%A）")
    return (
        f"{source}\n\n# 当前时间\n当前日期：{date_text}。"
        "代码文件头、元数据及所有输出中的日期必须使用该当天日期。"
        f"\n\n{TOOL_INSTRUCTIONS}"
    )
