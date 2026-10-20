"""Load the preserved domain prompt and add per-turn operating context."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROMPT_FILE = Path(__file__).with_name("system_prompt.md")
TOOL_INSTRUCTIONS = """# 工具执行规则
- 用户给出可识别的品种名称和交割月份但没有写合约代码时，必须调用 lookup_exchange_by_contract_code，并传 product_name 与数字 month；用户明确了年份或交易所时一并传 year 或 exchange_code。不要让模型凭记忆拼代码，也不要要求用户重复提供可推导的合约信息。年份未写时，由工具按本轮上海日期处理：目标月份尚未过去（含当前月）取当年，已过去取下一年。若结果为 expired，先复核年份；只有明确年份或用户明确提供的代码确已过期，才询问用户要改用哪个有效月份。
- 用户给出具体期货合约代码但未写交易所时，必须以 contract_code 调用 lookup_exchange_by_contract_code；查询成功后沿用返回的交易所代码继续处理，不得再问用户交易所。用户已明确给出代码和交易所时沿用用户信息，不必重复查询。生成源码时，必须把查询结果中的 canonical_contract_code 作为唯一合约代码来源，逐字保留其大小写；contract_code 是统一大写的识别字段，禁止用于生成源码。
- 品种与月份查询若返回唯一候选且 status 为 matched，直接进入后续流程，不进行确认式追问。若确实有缺项，把所有仍缺少且影响策略的参数合并为一次追问。
- 查询工具只支持境内六家交易所期货，不支持期权。只有品种没有交割月份/具体合约，或“主力”等动态合约描述无法映射到固定代码时，才询问交割月份或合约代码。对于模型推导的代码若返回 unrecognized 或 invalid_format，先复核品种映射和交易所代码格式；仍无法可靠确定时再问造成歧义的信息，不要一律要求用户补代码。
- 生成策略之前，先调用 list_interface_files，再用 read_interface_file 阅读接口源码，尤其是 external_head.py 与 external_baseid.py。接口示例只是契约参考，不是用户需求。
- 完整策略源码只能通过工具调用在内部传递、检查并写入文件产物；面向用户的聊天回复不得包含完整策略源码，也不得通过分段内容拼出完整源码。工具调用中的 source 参数不受此限制。
- 只有实质必要信息无法从对话或常规语义推导时才追问，并以「【参数补充请求】」开头简洁指出缺项；这一轮不得调用任何代码检测或写入工具。
- 需求完整时，先对候选源码调用 check_python_syntax。只有通过后，才能对完全相同的 source 调用 run_functional_test；该工具会单独调用测试用例 Skill 构造用例并执行沙盒测试。
- 语法或功能检测失败时，根据诊断修改源码，再重新运行语法检测和功能检测。不要自行替代 Skill 编造测试结果。
- 如果历史中存在包含「【上轮失败诊断】」的助手消息，必须先读取其中的工具失败结果和最终错误，结合原始用户需求针对性修正；不要忽略诊断或原样重复失败调用。
- 仅当同一份源码通过两项检测后，才调用 write_generated_file；写工具的 source 必须与检测通过的源码一致。文件名由策略类的 name 属性自动生成，保存为 `<name>.py`，不传 filename。
- 只能使用提供的工具。工具报错时根据错误调整调用，不要原样重复失败的工具名和参数；不要猜测读取结果或声称已保存失败的文件。"""


def build_system_prompt(now: datetime | None = None) -> str:
    """Keep the stable prompt prefix cacheable and append volatile time last."""
    source = PROMPT_FILE.read_text(encoding="utf-8").removesuffix("\n")
    current = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    shanghai_time = current.astimezone(ZoneInfo("Asia/Shanghai"))
    date_text = shanghai_time.strftime("%Y-%m-%d %H:%M（%A）")
    stable_prompt = f"{source}\n\n{TOOL_INSTRUCTIONS}"
    current_time_suffix = (
        f"# 当前时间\n当前日期：{date_text}。"
        "代码文件头、元数据及所有输出中的日期必须使用该当天日期。"
    )
    return f"{stable_prompt}\n\n{current_time_suffix}"
