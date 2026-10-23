"""Prompt assembly and the bounded model/tool loop."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.contracts import (
    AgentEvent,
    Artifact,
    ArtifactWriter,
    AssistantMessage,
    AssistantToolCall,
    ChatTurn,
    ConversationRecorder,
    FunctionalTester,
    ModelGateway,
    ModelMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent.exchange_lookup import (
    lookup_exchange_by_contract_code,
    resolve_contract_by_product_month,
)
from agent.prompt import build_system_prompt
from agent.skills import FilesystemSkillLoader, SkillLoader


MAX_TOOL_ROUNDS = 30
MAX_IDENTICAL_FAILED_TOOL_RETRIES = 3
MAX_UNSAVED_ARTIFACT_REPAIR_RETRIES = 3
logger = logging.getLogger(__name__)
TOOL_FAILURE_REPAIR_PROMPT = (
    "上一个工具结果包含 error 或 passed=false。请先检查该工具结果中的具体诊断，"
    "功能检测失败时，以结果中的 failed_source 作为被测源码，直接修改它来修复失败原因；"
    "若工具结果提示复用了之前的失败诊断，说明源码没有变化；必须先修改实现逻辑，"
    "不得只改注释或空白后重试，也不得原样返回失败源码。"
    "若功能检测结果包含 failed_cases，逐笔按其中的 ticks 推演源码状态变化，"
    "保留已通过用例的行为，不得修改测试预期。修复源码后重新调用 run_functional_test。"
    "不要只向用户描述失败，也不要保存未通过检测的源码。"
)
UNCHANGED_SOURCE_REPAIR_PROMPT = (
    "你刚刚提交的源码与失败版本具有相同的 Python 语法树，说明没有修改实现逻辑；"
    "上面的测试结果是复用的，没有重新运行沙盒。现在必须直接修复代码：先按 failed_cases 中"
    "每个 tick 的顺序追踪相关状态变量，再指出实际出错的判断或状态转换，并据此修改源码。"
    "不要改测试预期，也不要只加注释、改空白或改变量名。修改后重新调用 run_functional_test。"
)
UNSAVED_ARTIFACT_REPAIR_PROMPT = (
    "本轮没有成功保存产物，但你刚才声称文件已生成、更新或保存。请纠正这一点："
    "如果现有信息足以完成需求，就按适用 Skill 完成要求的功能测试，并成功调用 "
    "write_generated_file；如果缺少会改变结果的必要信息，就直接用普通中文询问具体缺项，"
    "并且不要调用写入工具。write_generated_file 成功返回之前，不得声称本轮产物已保存。"
)
ARTIFACT_GENERATION_CLAIM_PATTERN = re.compile(
    r"(?:已|已经|成功(?:地)?)\s*(?:为您|帮您|为你|帮你)?\s*"
    r"(?:生成|创建|保存|写入|更新|修改|完成|做好|写好)(?:了)?\s*(?:完整的?)?"
    r"(?:策略(?:文件)?|文件|产物|源码|版本)"
    r"|(?:策略(?:文件)?|文件|产物|源码|版本)\s*"
    r"(?:(?:已|已经)\s*(?:成功(?:地)?\s*)?|成功(?:地)?\s*)"
    r"(?:生成|创建|保存|写入|更新|修改|完成|做好|写好)"
)
PARAMETER_CLARIFICATION_PATTERN = re.compile(
    r"(?:请(?:您)?(?:提供|补充|告知|说明|明确|确认|告诉我)|"
    r"(?:还|仍)?需要(?:您)?(?:提供|补充|告知|说明|明确|确认)|"
    r"(?:我)?(?:还|仍)?需要(?:知道|确认|了解)|"
    r"需要我.{0,16}(?:补充|提供|说明|确认)|"
    r"(?:缺少|尚缺|还缺少|尚未提供).{0,32}|"
    r"(?:方便|可以|能否).{0,32}(?:告诉|提供|补充|说明|明确)|"
    r"(?:阈值|合约|交易所|交割月份|价格|规则|条件).{0,24}"
    r"(?:多少|哪个|哪一个|什么|是否|如何).{0,12}[?？]|"
    r"(?:请问|是否|哪个|哪一个|什么|多少).{0,48}[?？])"
)
HISTORICAL_ARTIFACT_REFERENCE_PATTERN = re.compile(
    r"(?:历史|此前|先前|之前|上轮|上次)"
)
CURRENT_TURN_REFERENCE_PATTERN = re.compile(r"(?:本轮|本次|此次|刚刚|当前)")
CONTRACT_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z]{1,3}\d{3,4})(?![A-Za-z0-9_])"
)
CONTRACT_LOOKUP_PREFLIGHT_PROMPT = """你负责为本轮期货价格策略请求预查询品种、合约和交易所信息。
只检查当前请求及仍与当前请求相关的上下文；本轮与期货价格策略无关时，不调用工具。对本轮策略请求中用户明确给出的每个具体期货合约代码，必须调用 lookup_exchange_by_contract_code 并传 contract_code；对用户明确给出的品种和交割月份，必须调用该工具并传 product_name 与 month，显式年份和交易所一并传入。用户明确给出交易所时，也要将 exchange_code 一并传入以核对。不得猜测合约代码、交割月份、年份或交易所。只有品种但没有月份或具体代码时，不调用工具。
本次请求只用于调用查询工具，不要向用户回复文字；多个品种可在本次模型请求中分别调用工具。"""
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_exchange_by_contract_code",
            "description": "离线识别境内六家期货交易所的期货合约。可按具体合约代码查询，也可按品种和交割月份推导标准代码，并可传 year、exchange_code。CZCE 合约后缀为三位数字（一位年份加两位月份），其他交易所为四位数字。unrecognized 表示工具无法识别，不代表合约不存在。canonical_contract_code 保留映射表大小写；不支持期权。",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_code": {"type": "string", "description": "用户明确给出的境内期货合约代码，例如 rb2710"},
                    "product_name": {"type": "string", "description": "用户给出的期货品种名称或代码，例如 豆粕、螺纹钢、m"},
                    "month": {"type": "integer", "minimum": 1, "maximum": 12, "description": "交割月份；将‘七月’等自然语言月份转为 7"},
                    "year": {"type": "integer", "minimum": 2000, "maximum": 2099, "description": "用户明确给出的交割年份；未明确时省略，由工具按上海当前日期推导"},
                    "exchange_code": {"type": "string", "enum": ["SHFE", "INE", "DCE", "CZCE", "CFFEX", "GFEX"], "description": "用户明确给出的交易所代码；未明确时省略"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "列出当前 Agent 可用 Skill 的 ID、名称和简短描述。先检查目录，再加载与当前任务匹配的 Skill。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "加载 Skill 的 SKILL.md；也可按需加载该 Skill 包内的 Markdown 参考文件。resource_path 省略时加载 SKILL.md。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "list_skills 返回的 Skill ID"},
                    "resource_path": {"type": "string", "description": "Skill 包内的相对 Markdown 路径，例如 references/interface-contract.md"},
                },
                "required": ["skill_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_functional_test",
            "description": "按适用 Skill 和用户规则构造测试用例，并在 Python 3.12 隔离沙盒中运行候选源码。此前失败版本的 Python 语法树相同则复用上次失败用例，不重复运行沙盒，并要求先修改实现逻辑。失败时返回 failed_source（本次实际测试的完整源码）及失败用例诊断。",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string", "description": "完整 Python 源码"}},
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_generated_file",
            "description": "将通过 run_functional_test 的完整源码保存为产物。必须使用与功能测试完全相同的源码；确实缺少必要信息时先用普通中文追问。只有工具成功返回后才能声称产物已生成或更新。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "完整 Python 源码"},
                },
                "required": ["source"],
            },
        },
    },
]


class AgentError(RuntimeError):
    """A request cannot finish after a bounded number of tool rounds."""

    def __init__(self, message: str, *, failure_context: str | None = None) -> None:
        super().__init__(message)
        self.failure_context = failure_context


def _failure_context(message: str, tool_failures: Sequence[str]) -> str:
    """Build a concise assistant turn that lets a later request correct the failure."""
    lines = ["【上轮失败诊断】", f"本轮失败：{message}"]
    if tool_failures:
        lines.append("工具失败结果：")
        lines.extend(f"- {failure}" for failure in tool_failures)
    return "\n".join(lines)


def _tool_failure_context_entry(name: str, result: dict[str, Any]) -> str:
    """Keep failed scenarios in turn diagnostics without embedding full source."""
    failed_cases = result.get("failed_cases")
    if name == "run_functional_test":
        details: dict[str, Any] = {"passed": False}
        if "case_count" in result:
            details["case_count"] = result["case_count"]
        if isinstance(failed_cases, list):
            details["failed_cases"] = failed_cases
        error = result.get("error")
        if error is None:
            error = result.get("repeat_notice")
        if error is not None:
            details["error"] = str(error)
        result = details
    return f"{name}: {json.dumps(result, ensure_ascii=False)}"


def _functional_failure_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep only the useful diagnostics for replaying a failed source."""
    summary: dict[str, Any] = {"passed": False}
    if "case_count" in result:
        summary["case_count"] = result["case_count"]
    failed_cases = result.get("failed_cases")
    if isinstance(failed_cases, list):
        summary["failed_cases"] = failed_cases
    error = result.get("error")
    if error is not None:
        summary["error"] = str(error)
    return summary


@dataclass
class _ValidationState:
    functional_passed: set[str] = field(default_factory=set)
    canonical_contract_codes: dict[str, str] = field(default_factory=dict)
    contract_code_validity: dict[str, bool] = field(default_factory=dict)
    failed_functional_sources: dict[str, dict[str, Any]] = field(default_factory=dict)


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _source_behavior_digest(source: str) -> str:
    """Hash Python syntax while ignoring comments and formatting-only changes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _source_digest(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_digest_for_log(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", errors="backslashreplace")).hexdigest()


def _tool_call_signature(name: str, arguments: dict[str, Any]) -> str:
    """Return a stable signature for detecting an unchanged failed tool call."""
    return json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _tool_call_signature_digest(name: str, arguments: dict[str, Any]) -> str:
    signature = _tool_call_signature(name, arguments)
    return hashlib.sha256(signature.encode("utf-8", errors="backslashreplace")).hexdigest()[:16]


def _model_message_for_log(message: ModelMessage) -> dict[str, Any]:
    """Serialize the exact provider-neutral message shape sent to a model."""
    record: dict[str, Any] = {"role": message.role, "content": message.content}
    if isinstance(message, AssistantMessage):
        record["reasoning_content"] = message.reasoning_content
        record["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        ]
    elif isinstance(message, ToolResultMessage):
        record["call_id"] = message.call_id
        record["tool_name"] = message.tool_name
    return record


def _model_chunk_for_log(chunk: ModelChunk) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": chunk.content,
        "reasoning_content": chunk.reasoning_content,
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in chunk.tool_calls
        ],
        "usage": chunk.usage,
    }


def _exception_detail(exc: Exception) -> str:
    detail = str(exc).strip()
    return detail or f"{type(exc).__name__}（异常消息为空）"


def _claims_artifact_generated(content: str) -> bool:
    """Detect common claims that a file was generated without a successful write."""
    for match in ARTIFACT_GENERATION_CLAIM_PATTERN.finditer(content):
        preceding_context = content[max(0, match.start() - 32) : match.start()]
        if (
            HISTORICAL_ARTIFACT_REFERENCE_PATTERN.search(preceding_context) is not None
            and CURRENT_TURN_REFERENCE_PATTERN.search(preceding_context) is None
        ):
            continue
        return True
    return False


def _is_parameter_clarification(content: str) -> bool:
    """Recognize natural-language requests for missing information."""
    return PARAMETER_CLARIFICATION_PATTERN.search(content) is not None


def _tool_result_log_summary(result: dict[str, Any]) -> str:
    """Summarize failed tool results without logging candidate source code."""
    summary: dict[str, Any] = {"passed": result.get("passed")}
    error = result.get("error")
    if error is not None:
        summary["error"] = str(error)[:2_000]

    cases = result.get("cases")
    if isinstance(cases, list):
        failed_cases = []
        for index, case in enumerate(cases, start=1):
            if not isinstance(case, dict) or case.get("passed") is False:
                item: dict[str, Any] = {"index": index}
                if isinstance(case, dict):
                    for key in ("expected_warning_count", "actual_warning_count"):
                        if key in case:
                            item[key] = case[key]
                    if case.get("error") is not None:
                        item["error"] = str(case["error"])[:1_000]
                failed_cases.append(item)
        summary["case_count"] = len(cases)
        summary["failed_cases"] = failed_cases
    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def _strategy_filename(strategy_class: ast.ClassDef) -> str:
    """Return a safe artifact filename from the strategy class's literal name."""
    if strategy_class.name != "My_PythonObj":
        raise ValueError("策略主类的 Python 类名必须是 My_PythonObj")

    name_values: list[ast.expr] = []
    for statement in strategy_class.body:
        if isinstance(statement, ast.Assign):
            name_values.extend(
                statement.value
                for target in statement.targets
                if isinstance(target, ast.Name) and target.id == "name"
            )
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "name"
            and statement.value is not None
        ):
            name_values.append(statement.value)

    if len(name_values) != 1:
        raise ValueError("策略类必须且只能定义一次 name 字符串属性")
    try:
        strategy_name = ast.literal_eval(name_values[0])
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("策略类的 name 属性必须是字符串字面量") from exc
    if not isinstance(strategy_name, str):
        raise ValueError("策略类的 name 属性必须是字符串")

    filename = f"{strategy_name}.py"
    if (
        not strategy_name
        or strategy_name != strategy_name.strip()
        or len(filename) > 120
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", strategy_name) is None
    ):
        raise ValueError("策略类的 name 属性必须是英文名称（英文字母开头，仅含英文字母、数字、下划线和连字符）")
    return filename


def _contract_code_casing_error(
    source: str, canonical_contract_codes: dict[str, str]
) -> str | None:
    """Reject every source occurrence of a mapped contract with noncanonical case."""
    for normalized_code, canonical_code in canonical_contract_codes.items():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(normalized_code)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        wrong_case = [
            match.group(0)
            for match in pattern.finditer(source)
            if match.group(0) != canonical_code
        ]
        if wrong_case:
            examples = ", ".join(repr(value) for value in wrong_case[:3])
            return (
                f"源码中的合约代码大小写不符合映射工具结果：应逐字使用 "
                f"{canonical_code!r}，发现 {examples}。请修正所有对应代码字符串。"
            )
    return None


def _contract_code_validation_error(
    source: str, validation: _ValidationState
) -> str | None:
    """Require every literal contract code in the candidate to have matched lookup data."""
    for match in CONTRACT_CODE_PATTERN.finditer(source):
        normalized_code = match.group(1).upper()
        if validation.contract_code_validity.get(normalized_code) is not True:
            return (
                f"源码中的合约代码 {match.group(1)!r} 尚未通过有效性查询；"
                "请先查询该合约并确认结果 status 为 matched，再运行功能检测。"
            )
    return None


def _coalesce_adjacent_turns(turns: Sequence[ChatTurn]) -> list[ChatTurn]:
    """Keep the chat role sequence valid when failed turns leave orphaned messages."""
    merged: list[ChatTurn] = []
    for turn in turns:
        if merged and merged[-1].role == turn.role:
            previous = merged[-1]
            merged[-1] = ChatTurn(turn.role, f"{previous.content}\n\n{turn.content}")
        else:
            merged.append(turn)
    return merged


class AgentRunner:
    def __init__(
        self,
        model: ModelGateway,
        artifacts: ArtifactWriter,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        functional_tester: FunctionalTester | None = None,
        skill_loader: SkillLoader | None = None,
    ) -> None:
        self._model = model
        self._artifacts = artifacts
        self._max_tool_rounds = max_tool_rounds
        self._functional_tester = functional_tester
        self._skill_loader = skill_loader or FilesystemSkillLoader()

    async def run(
        self,
        *,
        message: str,
        history: Sequence[ChatTurn],
        model: str,
        account_id: str,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        conversation_recorder: ConversationRecorder | None = None,
        options: dict[str, int | float] | None = None,
        think: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        async def record_trace(event: str, data: dict[str, Any]) -> None:
            if (
                conversation_recorder is not None
                and conversation_id is not None
                and turn_id is not None
            ):
                await conversation_recorder.record(
                    conversation_id, turn_id, event, data
                )

        model_options = options if options is not None else {
            "temperature": 0.3,
            "top_p": 0.8,
            "num_ctx": 16_384,
        }
        conversation_turns = _coalesce_adjacent_turns(
            [*history, ChatTurn("user", message)]
        )
        visible_content: list[str] = []
        created: list[Artifact] = []
        tool_failures: list[str] = []
        failed_tool_call_counts: dict[str, int] = {}
        used_call_ids: set[str] = set()
        generated_call_id = 0
        tool_failure_repair_pending = False
        unsaved_artifact_repair_count = 0
        functional_repair_mode = any(
            turn.role == "assistant" and "【上轮失败诊断】" in turn.content
            for turn in history
        )
        validation = _ValidationState()
        requirements = [*history, ChatTurn("user", message)]
        logger.info(
            "agent_turn_started conversation_id=%s model=%s max_tool_rounds=%d",
            conversation_id or "-",
            model,
            self._max_tool_rounds,
        )
        yield AgentEvent("status", {"phase": "started", "model": model})

        preflight_messages: list[ModelMessage] = [
            SystemMessage(CONTRACT_LOOKUP_PREFLIGHT_PROMPT)
        ]
        preflight_messages.extend(
            UserMessage(turn.content)
            if turn.role == "user"
            else AssistantMessage(turn.content)
            for turn in conversation_turns
        )
        lookup_tool = next(
            tool for tool in TOOL_SCHEMAS
            if tool["function"]["name"] == "lookup_exchange_by_contract_code"
        )
        lookup_tools = [lookup_tool]
        await record_trace(
            "model_request",
            {
                "phase": "contract_lookup_preflight",
                "model": model,
                "options": model_options,
                "think": think,
                "tools": lookup_tools,
                "messages": [_model_message_for_log(item) for item in preflight_messages],
            },
        )
        preflight_calls: list[ToolCall] = []
        async for chunk in self._model.stream_chat(
            model,
            preflight_messages,
            lookup_tools,
            options=model_options,
            think=think,
        ):
            await record_trace(
                "model_chunk",
                {"phase": "contract_lookup_preflight", **_model_chunk_for_log(chunk)},
            )
            preflight_calls.extend(chunk.tool_calls)

        contract_lookup_results: list[dict[str, Any]] = []
        for index, tool_call in enumerate(preflight_calls):
            if tool_call.name != "lookup_exchange_by_contract_code":
                continue
            call_id = tool_call.call_id or f"agent_preflight_lookup_{index}"
            call = AssistantToolCall(call_id, tool_call.name, tool_call.arguments)
            yield AgentEvent(
                "status", {"phase": "tool_started", "tool": call.name}
            )
            try:
                result, _ = await self._call_tool(
                    call,
                    account_id,
                    conversation_id,
                    turn_id=turn_id,
                    conversation_recorder=conversation_recorder,
                    validation=validation,
                    requirements=requirements,
                    model=model,
                    options=model_options,
                )
            except (ValueError, OSError) as exc:
                result = {"error": _exception_detail(exc)}
            contract_lookup_results.append(
                {"arguments": tool_call.arguments, "result": result}
            )
            await record_trace(
                "contract_lookup_preflight_result",
                {
                    "call_id": call_id,
                    "arguments": tool_call.arguments,
                    "result": result,
                },
            )
            yield AgentEvent(
                "status", {"phase": "tool_finished", "tool": call.name}
            )

        messages: list[ModelMessage] = [SystemMessage(build_system_prompt())]
        if contract_lookup_results:
            messages.append(
                SystemMessage(
                    "# 合约预查询工具结果\n"
                    "以下是主会话开始前取得的查询结果，请遵循适用 Skill 对结果的处理要求；"
                    "不要猜测、忽略或无故重复查询。\n"
                    f"{json.dumps(contract_lookup_results, ensure_ascii=False, indent=2)}"
                )
            )
        messages.extend(
            UserMessage(turn.content)
            if turn.role == "user"
            else AssistantMessage(turn.content)
            for turn in conversation_turns
        )
        skill_context_insert_index = 1
        loaded_skill_contexts: set[tuple[str, str]] = set()

        for round_index in range(self._max_tool_rounds + 1):
            round_options = model_options
            qwen38_repair_mode = (
                model.casefold().startswith("qwen3.8") and functional_repair_mode
            )
            round_think = think or qwen38_repair_mode
            if qwen38_repair_mode and not think:
                round_options = {
                    **model_options,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 0.0,
                    "repeat_penalty": 1.0,
                }
            logger.info(
                "agent_model_round_started conversation_id=%s model=%s round=%d",
                conversation_id or "-",
                model,
                round_index + 1,
            )
            round_content: list[str] = []
            round_reasoning_content: list[str] = []
            calls: list[ToolCall] = []
            usage: dict[str, int] = {}
            await record_trace(
                "model_request",
                {
                    "round": round_index + 1,
                    "model": model,
                    "options": round_options,
                    "think": round_think,
                    "tools": TOOL_SCHEMAS,
                    "messages": [_model_message_for_log(item) for item in messages],
                },
            )
            async for chunk in self._model.stream_chat(
                model,
                messages,
                TOOL_SCHEMAS,
                options=round_options,
                think=round_think,
            ):
                await record_trace(
                    "model_chunk",
                    {"round": round_index + 1, **_model_chunk_for_log(chunk)},
                )
                if chunk.content:
                    round_content.append(chunk.content)
                    visible_content.append(chunk.content)
                    yield AgentEvent("delta", {"text": chunk.content})
                if chunk.reasoning_content:
                    round_reasoning_content.append(chunk.reasoning_content)
                calls.extend(chunk.tool_calls)
                if chunk.usage:
                    usage = chunk.usage

            logger.info(
                "agent_model_round_finished conversation_id=%s model=%s round=%d "
                "response_chars=%d tool_calls=%s usage=%s",
                conversation_id or "-",
                model,
                round_index + 1,
                len("".join(round_content)),
                [call.name for call in calls],
                usage,
            )

            if not calls:
                round_reply = "".join(round_content)
                turn_content = "".join(visible_content)
                if not created and _claims_artifact_generated(turn_content):
                    if unsaved_artifact_repair_count >= MAX_UNSAVED_ARTIFACT_REPAIR_RETRIES:
                        message = "模型多次声称已生成策略文件，但始终没有成功写入"
                        logger.error(
                            "agent_unsaved_artifact_repair_exhausted conversation_id=%s "
                            "model=%s retries=%d",
                            conversation_id or "-",
                            model,
                            unsaved_artifact_repair_count,
                        )
                        raise AgentError(
                            message,
                            failure_context=_failure_context(message, tool_failures),
                        )

                    unsaved_artifact_repair_count += 1
                    assistant_content = round_reply
                    if assistant_content.strip() or round_reasoning_content:
                        messages.append(
                            AssistantMessage(
                                assistant_content,
                                reasoning_content="".join(round_reasoning_content),
                            )
                        )
                    repair_prompt = UNSAVED_ARTIFACT_REPAIR_PROMPT
                    if tool_failure_repair_pending:
                        repair_prompt = TOOL_FAILURE_REPAIR_PROMPT + "\n\n" + repair_prompt
                        tool_failure_repair_pending = False
                    messages.append(UserMessage(repair_prompt))
                    # The next completion replaces this unsubstantiated claim in the
                    # authoritative done payload; streamed deltas remain transient.
                    visible_content.clear()
                    logger.warning(
                        "agent_unsaved_artifact_repair_started conversation_id=%s "
                        "model=%s retry=%d retry_limit=%d",
                        conversation_id or "-",
                        model,
                        unsaved_artifact_repair_count,
                        MAX_UNSAVED_ARTIFACT_REPAIR_RETRIES,
                    )
                    continue

                if tool_failure_repair_pending:
                    assistant_content = "".join(round_content)
                    if assistant_content.strip() or round_reasoning_content:
                        messages.append(
                            AssistantMessage(
                                assistant_content,
                                reasoning_content="".join(round_reasoning_content),
                            )
                        )
                    messages.append(UserMessage(TOOL_FAILURE_REPAIR_PROMPT))
                    tool_failure_repair_pending = False
                    continue
                content = "".join(visible_content)
                if not content.strip():
                    if not created:
                        message = "模型没有返回内容"
                        raise AgentError(
                            message,
                            failure_context=_failure_context(message, tool_failures),
                        )
                    content = "策略文件已生成，请通过产物卡片预览或下载。"
                if not created:
                    if not _is_parameter_clarification(content):
                        message = "模型未保存策略文件，也未有效询问缺少的信息"
                        raise AgentError(
                            message,
                            failure_context=_failure_context(message, tool_failures),
                        )
                logger.info(
                    "agent_turn_completed conversation_id=%s model=%s artifacts=%d",
                    conversation_id or "-",
                    model,
                    len(created),
                )
                done_data = {
                    "content": content,
                    "model": model,
                    "artifacts": [
                        {"artifact_id": item.artifact_id, "filename": item.filename}
                        for item in created
                    ],
                    "usage": usage,
                }
                if not created:
                    done_data["_completion"] = "needs_input"
                yield AgentEvent("done", done_data)
                return

            if round_index >= self._max_tool_rounds:
                message = f"工具调用超过 {self._max_tool_rounds} 轮上限"
                logger.error(
                    "agent_tool_round_limit_reached conversation_id=%s model=%s round=%d "
                    "max_tool_rounds=%d pending_tools=%s failed_tool_calls=%d",
                    conversation_id or "-",
                    model,
                    round_index + 1,
                    self._max_tool_rounds,
                    [call.name for call in calls],
                    len(tool_failures),
                )
                raise AgentError(
                    message,
                    failure_context=_failure_context(message, tool_failures),
                )

            reserved_provider_call_ids = {
                call.call_id for call in calls if call.call_id
            }
            resolved_calls: list[AssistantToolCall] = []
            for call in calls:
                call_id = call.call_id
                if not call_id or call_id in used_call_ids:
                    while True:
                        call_id = f"agent_call_{generated_call_id}"
                        generated_call_id += 1
                        if (
                            call_id not in used_call_ids
                            and call_id not in reserved_provider_call_ids
                        ):
                            break
                used_call_ids.add(call_id)
                resolved_calls.append(
                    AssistantToolCall(call_id, call.name, call.arguments)
                )

            messages.append(
                AssistantMessage(
                    content="".join(round_content),
                    tool_calls=tuple(resolved_calls),
                    reasoning_content="".join(round_reasoning_content),
                )
            )
            tool_failure_repair_pending = False
            unchanged_source_replayed_this_round = False
            for call_index, call in enumerate(resolved_calls, start=1):
                signature = _tool_call_signature(call.name, call.arguments)
                failure_count = failed_tool_call_counts.get(signature, 0)
                source = call.arguments.get("source")
                previous_failure: dict[str, Any] | None = None
                if call.name == "run_functional_test" and isinstance(source, str):
                    source_digest = _source_behavior_digest(source)
                    previous_failure = validation.failed_functional_sources.get(
                        source_digest
                    )

                if failure_count > MAX_IDENTICAL_FAILED_TOOL_RETRIES:
                    retry_count = failure_count - 1
                    message = (
                        f"模型在第 {round_index + 1} 轮第 {call_index} 个工具调用中，"
                        f"对工具 {call.name} 使用相同参数再次调用；此前已连续重试 "
                        f"{retry_count} 次，达到 {MAX_IDENTICAL_FAILED_TOOL_RETRIES} 次上限，"
                        "已停止重复调用"
                    )
                    logger.error(
                        "agent_repeated_failed_tool_call_stopped conversation_id=%s model=%s "
                        "round=%d call=%d tool=%s signature_sha256=%s "
                        "previous_failures=%d retry_limit=%d",
                        conversation_id or "-",
                        model,
                        round_index + 1,
                        call_index,
                        call.name,
                        _tool_call_signature_digest(call.name, call.arguments),
                        failure_count,
                        MAX_IDENTICAL_FAILED_TOOL_RETRIES,
                    )
                    raise AgentError(
                        message,
                        failure_context=_failure_context(message, tool_failures),
                    )

                source_summary = ""
                if isinstance(source, str):
                    source_summary = (
                        f" source_sha256={_source_digest_for_log(source)}"
                        f" source_bytes={len(source.encode('utf-8', errors='replace'))}"
                    )
                logger.info(
                    "agent_tool_started conversation_id=%s model=%s round=%d call=%d/%d "
                    "tool=%s call_id=%s signature_sha256=%s argument_keys=%s%s",
                    conversation_id or "-",
                    model,
                    round_index + 1,
                    call_index,
                    len(resolved_calls),
                    call.name,
                    call.call_id,
                    _tool_call_signature_digest(call.name, call.arguments),
                    sorted(call.arguments),
                    source_summary,
                )
                await record_trace(
                    "tool_call_started",
                    {
                        "round": round_index + 1,
                        "call_index": call_index,
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                )
                yield AgentEvent("status", {"phase": "tool_started", "tool": call.name})
                try:
                    if call.name == "run_functional_test" and previous_failure is not None:
                        unchanged_source_replayed_this_round = True
                        await record_trace(
                            "tool_call_blocked",
                            {
                                "round": round_index + 1,
                                "call_index": call_index,
                                "call_id": call.call_id,
                                "name": call.name,
                                "reason": "unchanged_failed_source_reuses_diagnostics",
                                "source_sha256": _source_digest(source),
                                "behavior_sha256": _source_behavior_digest(source),
                            },
                        )
                        result = {
                            **previous_failure,
                            "failed_source": source,
                            "repeat_notice": (
                                "源码语法树与此前失败版本相同，本次未重新运行沙盒；"
                                "请按失败用例修改实现逻辑后再测试。"
                            ),
                        }
                        artifact = None
                    else:
                        result, artifact = await self._call_tool(
                            call,
                            account_id,
                            conversation_id,
                            turn_id=turn_id,
                            conversation_recorder=conversation_recorder,
                            validation=validation,
                            requirements=requirements,
                            model=model,
                            options=model_options,
                        )
                except (ValueError, OSError) as exc:
                    detail = _exception_detail(exc)
                    logger.warning(
                        "agent_tool_raised conversation_id=%s model=%s round=%d call=%d "
                        "tool=%s exception_type=%s detail=%s",
                        conversation_id or "-",
                        model,
                        round_index + 1,
                        call_index,
                        call.name,
                        type(exc).__name__,
                        detail[:2_000],
                    )
                    result = {"error": detail}
                    if call.name == "run_functional_test" and isinstance(source, str):
                        result["failed_source"] = source
                    artifact = None
                except Exception as exc:
                    await record_trace(
                        "tool_result",
                        {
                            "round": round_index + 1,
                            "call_index": call_index,
                            "call_id": call.call_id,
                            "name": call.name,
                            "error": _exception_detail(exc),
                            "exception_type": type(exc).__name__,
                        },
                    )
                    raise
                await record_trace(
                    "tool_result",
                    {
                        "round": round_index + 1,
                        "call_index": call_index,
                        "call_id": call.call_id,
                        "name": call.name,
                        "result": result,
                    },
                )
                if result.get("error") is not None or result.get("passed") is False:
                    tool_failure_repair_pending = True
                    if call.name == "run_functional_test":
                        functional_repair_mode = True
                    if not (
                        call.name == "run_functional_test"
                        and previous_failure is not None
                    ):
                        failed_tool_call_counts[signature] = failure_count + 1
                    if (
                        call.name == "run_functional_test"
                        and result.get("passed") is False
                        and isinstance(source, str)
                    ):
                        validation.failed_functional_sources.setdefault(
                            _source_behavior_digest(source),
                            _functional_failure_summary(result),
                        )
                    logger.warning(
                        "agent_tool_failed conversation_id=%s model=%s round=%d call=%d/%d "
                        "tool=%s signature_sha256=%s%s result=%s",
                        conversation_id or "-",
                        model,
                        round_index + 1,
                        call_index,
                        len(resolved_calls),
                        call.name,
                        _tool_call_signature_digest(call.name, call.arguments),
                        source_summary,
                        _tool_result_log_summary(result),
                    )
                    tool_failures.append(
                        f"第 {round_index + 1} 轮第 {call_index}/{len(calls)} 个工具调用，"
                        f"{_tool_failure_context_entry(call.name, result)}"
                    )
                else:
                    logger.info(
                        "agent_tool_succeeded conversation_id=%s model=%s round=%d call=%d/%d "
                        "tool=%s signature_sha256=%s result_keys=%s",
                        conversation_id or "-",
                        model,
                        round_index + 1,
                        call_index,
                        len(resolved_calls),
                        call.name,
                        _tool_call_signature_digest(call.name, call.arguments),
                        sorted(result),
                    )
                    failed_tool_call_counts.pop(signature, None)
                    if call.name == "run_functional_test" and result.get("passed") is True:
                        write_signature = _tool_call_signature(
                            "write_generated_file", call.arguments
                        )
                        failed_tool_call_counts.pop(write_signature, None)
                if artifact:
                    created.append(artifact)
                    yield AgentEvent(
                        "artifact",
                        {"artifact_id": artifact.artifact_id, "filename": artifact.filename},
                    )
                messages.append(
                    ToolResultMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        call_id=call.call_id,
                        tool_name=call.name,
                    )
                )
                if call.name == "load_skill" and isinstance(result.get("content"), str):
                    skill_key = (
                        str(result.get("skill_id", "")),
                        str(result.get("resource_path", "SKILL.md")),
                    )
                    if skill_key not in loaded_skill_contexts:
                        skill_context = (
                            f"# Skill 指令：{skill_key[0]}"
                            f"（{skill_key[1]}）\n\n{result['content']}"
                        )
                        messages.insert(
                            skill_context_insert_index,
                            SystemMessage(skill_context),
                        )
                        skill_context_insert_index += 1
                        loaded_skill_contexts.add(skill_key)
                yield AgentEvent("status", {"phase": "tool_finished", "tool": call.name})

            if tool_failure_repair_pending:
                repair_prompt = TOOL_FAILURE_REPAIR_PROMPT
                if unchanged_source_replayed_this_round:
                    repair_prompt += "\n\n" + UNCHANGED_SOURCE_REPAIR_PROMPT
                messages.append(UserMessage(repair_prompt))

        message = "Agent 未能完成请求"
        logger.error(
            "agent_loop_exhausted conversation_id=%s model=%s max_tool_rounds=%d "
            "failed_tool_calls=%d",
            conversation_id or "-",
            model,
            self._max_tool_rounds,
            len(tool_failures),
        )
        raise AgentError(
            message,
            failure_context=_failure_context(message, tool_failures),
        )

    async def _call_tool(
        self,
        call: AssistantToolCall,
        account_id: str,
        conversation_id: str | None,
        *,
        turn_id: str | None,
        conversation_recorder: ConversationRecorder | None,
        validation: _ValidationState,
        requirements: Sequence[ChatTurn],
        model: str,
        options: dict[str, int | float],
    ) -> tuple[dict[str, Any], Artifact | None]:
        args = call.arguments
        if call.name == "lookup_exchange_by_contract_code":
            contract_code = args.get("contract_code")
            if isinstance(contract_code, str) and contract_code.strip():
                result = lookup_exchange_by_contract_code(contract_code)
                requested_exchange = args.get("exchange_code")
                resolved_exchange = result.get("exchange_code")
                if (
                    isinstance(requested_exchange, str)
                    and isinstance(resolved_exchange, str)
                    and requested_exchange.upper() != resolved_exchange.upper()
                ):
                    result = {
                        **result,
                        "status": "exchange_mismatch",
                        "is_valid": False,
                        "requested_exchange_code": requested_exchange.upper(),
                        "message": "合约代码对应的交易所与用户给出的交易所不一致。",
                    }
            else:
                product_name = args.get("product_name")
                month = args.get("month")
                if not isinstance(product_name, str):
                    raise ValueError("请提供 contract_code，或同时提供 product_name 与 month")
                if isinstance(month, bool) or not isinstance(month, int):
                    raise ValueError("品种和交割月份查询必须提供整数 month")
                year = args.get("year")
                exchange_code = args.get("exchange_code")
                if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
                    raise ValueError("year 必须是整数")
                if exchange_code is not None and not isinstance(exchange_code, str):
                    raise ValueError("exchange_code 必须是字符串")
                result = resolve_contract_by_product_month(
                    product_name,
                    month,
                    year=year,
                    exchange_code=exchange_code,
                )
            normalized_code = result.get("contract_code")
            canonical_code = result.get("canonical_contract_code")
            if isinstance(normalized_code, str):
                validation.contract_code_validity[normalized_code.upper()] = (
                    result.get("status") == "matched"
                    and result.get("is_valid") is True
                )
            if isinstance(normalized_code, str) and isinstance(canonical_code, str):
                validation.canonical_contract_codes[normalized_code] = canonical_code
            return result, None
        if call.name == "list_skills":
            skills = await self._skill_loader.list_skills()
            return {
                "skills": [
                    {
                        "skill_id": skill.skill_id,
                        "name": skill.name,
                        "description": skill.description,
                    }
                    for skill in skills
                ]
            }, None
        if call.name == "load_skill":
            skill_id = args.get("skill_id")
            if not isinstance(skill_id, str):
                raise ValueError("skill_id 必须是字符串")
            resource_path = args.get("resource_path")
            skill = await self._skill_loader.load(skill_id)
            if skill.visibility == "internal":
                raise ValueError("此 Skill 仅供内部组件使用")
            if resource_path is None:
                return {
                    "skill_id": skill.skill_id,
                    "resource_path": "SKILL.md",
                    "content": skill.instructions,
                }, None
            if not isinstance(resource_path, str):
                raise ValueError("resource_path 必须是字符串")
            content = await self._skill_loader.load_resource(skill_id, resource_path)
            return {
                "skill_id": skill_id,
                "resource_path": resource_path,
                "content": content,
            }, None
        if call.name == "run_functional_test":
            source = args.get("source")
            if not isinstance(source, str):
                raise ValueError("source 必须是字符串")
            digest = _source_digest(source)
            validation_error = _contract_code_validation_error(source, validation)
            if validation_error:
                raise ValueError(validation_error)
            casing_error = _contract_code_casing_error(
                source, validation.canonical_contract_codes
            )
            if casing_error:
                raise ValueError(casing_error)
            if self._functional_tester is None:
                raise ValueError("功能检测服务未配置，无法保存策略")
            try:
                tester_run = self._functional_tester.run
                run_kwargs: dict[str, Any] = {
                    "requirements": requirements,
                    "model": model,
                    "options": options,
                }
                try:
                    parameters = inspect.signature(tester_run).parameters
                except (TypeError, ValueError):
                    parameters = {}
                accepts_extra_keywords = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                optional_trace_args = {
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "conversation_recorder": conversation_recorder,
                }
                for name, value in optional_trace_args.items():
                    if name in parameters or accepts_extra_keywords:
                        run_kwargs[name] = value
                result = await tester_run(source, **run_kwargs)
            except (ValueError, OSError):
                validation.functional_passed.discard(digest)
                raise
            except Exception as exc:
                validation.functional_passed.discard(digest)
                detail = _exception_detail(exc)
                logger.exception(
                    "agent_functional_test_exception conversation_id=%s model=%s "
                    "source_sha256=%s source_bytes=%d exception_type=%s detail=%s",
                    conversation_id or "-",
                    model,
                    digest,
                    len(source.encode("utf-8", errors="replace")),
                    type(exc).__name__,
                    detail[:2_000],
                )
                raise ValueError(f"功能检测失败：{detail}") from exc
            if not isinstance(result, dict):
                validation.functional_passed.discard(digest)
                raise ValueError("功能检测服务返回了无效结果")
            if result.get("passed") is True:
                validation.functional_passed.add(digest)
            else:
                validation.functional_passed.discard(digest)
                result = {**result, "failed_source": source}
            return result, None
        if call.name == "write_generated_file":
            source = args.get("source")
            if not isinstance(source, str):
                raise ValueError("source 必须是字符串")
            digest = _source_digest(source)
            validation_error = _contract_code_validation_error(source, validation)
            if validation_error:
                raise ValueError(validation_error)
            if digest not in validation.functional_passed:
                raise ValueError("保存前必须先通过同一份源码的 run_functional_test")
            casing_error = _contract_code_casing_error(
                source, validation.canonical_contract_codes
            )
            if casing_error:
                raise ValueError(casing_error)
            try:
                source_tree = ast.parse(source)
            except SyntaxError as exc:
                raise ValueError(f"策略源码语法无效，无法确定策略类名：{exc.msg}") from exc
            strategy_classes = [
                node
                for node in source_tree.body
                if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Name) and base.id == "RHTemplate"
                    for base in node.bases
                )
            ]
            if len(strategy_classes) > 1:
                raise ValueError("策略源码只能包含一个直接继承 RHTemplate 的顶层策略类")
            if strategy_classes:
                filename = _strategy_filename(strategy_classes[0])
            else:
                raise ValueError("策略源码必须包含一个直接继承 RHTemplate 的顶层策略类")
            create = self._artifacts.create
            if "conversation_id" in inspect.signature(create).parameters:
                artifact = await create(filename, source, account_id, conversation_id)
            else:
                artifact = await create(filename, source, account_id)
            return {"artifact_id": artifact.artifact_id, "filename": artifact.filename}, artifact
        raise ValueError(f"不支持的工具：{call.name}")
