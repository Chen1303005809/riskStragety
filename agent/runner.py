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
    InterfaceReader,
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


MAX_TOOL_ROUNDS = 12
MAX_IDENTICAL_FAILED_TOOL_CALLS = 1
logger = logging.getLogger(__name__)
TOOL_FAILURE_REPAIR_PROMPT = (
    "上一个工具结果包含 error 或 passed=false。请先检查该工具结果中的具体诊断，"
    "直接修改候选源码来修复失败原因；不得对完全相同的参数再次调用失败的工具。"
    "功能检测失败时，先重新调用 check_python_syntax，通过后再调用 run_functional_test。"
    "不要只向用户描述失败，也不要保存未通过检测的源码。"
)
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_exchange_by_contract_code",
            "description": "离线识别境内六家期货交易所的期货合约。用户给出合约代码时传 contract_code；用户给出品种名称和交割月份时传 product_name 与 month，并可传 year、exchange_code，让工具直接推导并核验标准代码。CZCE 合约后缀为三位数字（一位年份加两位月份），其他交易所为四位数字。唯一匹配且状态为 matched 时无需向用户确认。canonical_contract_code 保留映射表大小写，是生成源码时唯一可用的合约代码；contract_code 仅供识别且统一大写。不支持期权。",
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
            "name": "list_interface_files",
            "description": "列出价格接口目录中可供阅读的 Python 源码文件。生成代码前先调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_interface_file",
            "description": "阅读价格接口目录中指定 Python 文件的完整源码。",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "纯 .py 文件名"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_python_syntax",
            "description": "静态检查候选策略是否符合 Python 3.7 语法；修正源码后必须重新检查。",
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
            "name": "run_functional_test",
            "description": "按用户规则构造测试用例并在隔离沙盒中运行候选策略；必须先通过 check_python_syntax。",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string", "description": "通过语法检查的完整 Python 源码"}},
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_generated_file",
            "description": "所有需求完整的策略生成、重新生成或业务规则更新请求在语法与功能检测通过后都必须调用此工具写入完整策略文件；不得只回复文字或源码。策略类标识符必须为 My_PythonObj，文件根据其 name 属性自动命名为 <name>.py；不得用于参数追问轮。",
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


@dataclass
class _ValidationState:
    syntax_passed: set[str] = field(default_factory=set)
    functional_passed: set[str] = field(default_factory=set)
    canonical_contract_codes: dict[str, str] = field(default_factory=dict)


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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


def check_python_syntax(source: str) -> dict[str, Any]:
    """Parse as Python 3.7 syntax and compile without executing the source."""
    try:
        ast.parse(source, filename="<generated strategy>", feature_version=(3, 7))
        compile(source, "<generated strategy>", "exec")
    except SyntaxError as exc:
        return {
            "passed": False,
            "grammar": "Python 3.7",
            "error": exc.msg,
            "line": exc.lineno,
            "column": exc.offset,
        }
    return {"passed": True, "grammar": "Python 3.7", "error": None}


class AgentRunner:
    def __init__(
        self,
        model: ModelGateway,
        interfaces: InterfaceReader,
        artifacts: ArtifactWriter,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        functional_tester: FunctionalTester | None = None,
    ) -> None:
        self._model = model
        self._interfaces = interfaces
        self._artifacts = artifacts
        self._max_tool_rounds = max_tool_rounds
        self._functional_tester = functional_tester

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
        messages: list[ModelMessage] = [SystemMessage(build_system_prompt())]
        messages.extend(
            UserMessage(turn.content)
            if turn.role == "user"
            else AssistantMessage(turn.content)
            for turn in conversation_turns
        )
        visible_content: list[str] = []
        created: list[Artifact] = []
        tool_failures: list[str] = []
        failed_tool_call_counts: dict[str, int] = {}
        used_call_ids: set[str] = set()
        generated_call_id = 0
        tool_failure_repair_pending = False
        validation = _ValidationState()
        requirements = [*history, ChatTurn("user", message)]
        logger.info(
            "agent_turn_started conversation_id=%s model=%s max_tool_rounds=%d",
            conversation_id or "-",
            model,
            self._max_tool_rounds,
        )
        yield AgentEvent("status", {"phase": "started", "model": model})

        for round_index in range(self._max_tool_rounds + 1):
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
                    "options": model_options,
                    "think": think,
                    "tools": TOOL_SCHEMAS,
                    "messages": [_model_message_for_log(item) for item in messages],
                },
            )
            async for chunk in self._model.stream_chat(
                model,
                messages,
                TOOL_SCHEMAS,
                options=model_options,
                think=think,
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

            if "".join(round_content).lstrip().startswith("【参数补充请求】"):
                if created:
                    message = "已保存文件后不能再发出参数补充请求"
                    raise AgentError(
                        message,
                        failure_context=_failure_context(message, tool_failures),
                    )
                logger.info(
                    "agent_turn_completed conversation_id=%s model=%s artifacts=0 parameter_request=true",
                    conversation_id or "-",
                    model,
                )
                yield AgentEvent(
                    "done",
                    {"content": "".join(visible_content), "model": model, "artifacts": [], "usage": usage},
                )
                return

            if not calls:
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
                if not created and "【参数补充请求】" not in content:
                    message = "模型未保存策略文件，也未发出参数补充请求"
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
                yield AgentEvent(
                    "done",
                    {
                        "content": content,
                        "model": model,
                        "artifacts": [
                            {"artifact_id": item.artifact_id, "filename": item.filename}
                            for item in created
                        ],
                        "usage": usage,
                    },
                )
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
            for call_index, call in enumerate(resolved_calls, start=1):
                signature = _tool_call_signature(call.name, call.arguments)
                failure_count = failed_tool_call_counts.get(signature, 0)
                if failure_count >= MAX_IDENTICAL_FAILED_TOOL_CALLS:
                    message = (
                        f"模型在第 {round_index + 1} 轮第 {call_index} 个工具调用中，"
                        f"对工具 {call.name} 使用相同参数再次调用；此前已失败 {failure_count} 次，"
                        "已停止重复调用"
                    )
                    logger.error(
                        "agent_repeated_failed_tool_call_stopped conversation_id=%s model=%s "
                        "round=%d call=%d tool=%s signature_sha256=%s previous_failures=%d",
                        conversation_id or "-",
                        model,
                        round_index + 1,
                        call_index,
                        call.name,
                        _tool_call_signature_digest(call.name, call.arguments),
                        failure_count,
                    )
                    raise AgentError(
                        message,
                        failure_context=_failure_context(message, tool_failures),
                    )

                source = call.arguments.get("source")
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
                    result, artifact = {"error": detail}, None
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
                    failed_tool_call_counts[signature] = failure_count + 1
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
                        f"{call.name}: {json.dumps(result, ensure_ascii=False)}"
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
                yield AgentEvent("status", {"phase": "tool_finished", "tool": call.name})

            if tool_failure_repair_pending:
                messages.append(UserMessage(TOOL_FAILURE_REPAIR_PROMPT))

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
            if isinstance(normalized_code, str) and isinstance(canonical_code, str):
                validation.canonical_contract_codes[normalized_code] = canonical_code
            return result, None
        if call.name == "list_interface_files":
            return {"files": await self._interfaces.list_files()}, None
        if call.name == "read_interface_file":
            filename = args.get("filename")
            if not isinstance(filename, str):
                raise ValueError("filename 必须是文件名字符串")
            return {"filename": filename, "source": await self._interfaces.read_file(filename)}, None
        if call.name == "check_python_syntax":
            source = args.get("source")
            if not isinstance(source, str):
                raise ValueError("source 必须是字符串")
            digest = _source_digest(source)
            result = check_python_syntax(source)
            if result["passed"]:
                casing_error = _contract_code_casing_error(
                    source, validation.canonical_contract_codes
                )
                if casing_error:
                    result = {**result, "passed": False, "error": casing_error}
            if result["passed"]:
                validation.syntax_passed.add(digest)
            else:
                validation.syntax_passed.discard(digest)
                validation.functional_passed.discard(digest)
            return result, None
        if call.name == "run_functional_test":
            source = args.get("source")
            if not isinstance(source, str):
                raise ValueError("source 必须是字符串")
            digest = _source_digest(source)
            if digest not in validation.syntax_passed:
                raise ValueError("请先对完全相同的源码调用 check_python_syntax 并通过")
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
            return result, None
        if call.name == "write_generated_file":
            source = args.get("source")
            if not isinstance(source, str):
                raise ValueError("source 必须是字符串")
            digest = _source_digest(source)
            if digest not in validation.syntax_passed:
                raise ValueError("保存前必须先通过同一份源码的 check_python_syntax")
            if digest not in validation.functional_passed:
                raise ValueError("保存前必须先通过同一份源码的 run_functional_test")
            casing_error = _contract_code_casing_error(
                source, validation.canonical_contract_codes
            )
            if casing_error:
                raise ValueError(casing_error)
            try:
                source_tree = ast.parse(source, feature_version=(3, 7))
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
