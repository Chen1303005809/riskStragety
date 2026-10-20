"""Prompt assembly and the bounded model/tool loop."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.contracts import (
    AgentEvent,
    Artifact,
    ArtifactWriter,
    ChatTurn,
    FunctionalTester,
    InterfaceReader,
    ModelGateway,
    ToolCall,
)
from agent.exchange_lookup import (
    lookup_exchange_by_contract_code,
    resolve_contract_by_product_month,
)
from agent.prompt import build_system_prompt


MAX_TOOL_ROUNDS = 12
TOOL_FAILURE_REPAIR_PROMPT = (
    "上一个工具结果包含 error 或 passed=false。请先检查该工具结果中的具体诊断，"
    "直接通过工具修正候选策略；功能检测失败时，先重新调用 check_python_syntax，"
    "通过后再调用 run_functional_test。不要只向用户描述失败，也不要保存未通过检测的源码。"
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
            "description": "保存已通过语法和功能检测的完整策略；策略类标识符必须为 My_PythonObj，文件根据其 name 属性自动命名为 <name>.py；不得用于参数追问轮。",
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
        options: dict[str, int | float] | None = None,
        think: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        model_options = options if options is not None else {
            "temperature": 0.3,
            "top_p": 0.8,
            "num_ctx": 16_384,
        }
        conversation_turns = _coalesce_adjacent_turns(
            [*history, ChatTurn("user", message)]
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            *({"role": turn.role, "content": turn.content} for turn in conversation_turns),
        ]
        visible_content: list[str] = []
        created: list[Artifact] = []
        tool_failures: list[str] = []
        tool_failure_repair_pending = False
        validation = _ValidationState()
        requirements = [*history, ChatTurn("user", message)]
        yield AgentEvent("status", {"phase": "started", "model": model})

        for round_index in range(self._max_tool_rounds + 1):
            round_content: list[str] = []
            calls: list[ToolCall] = []
            usage: dict[str, int] = {}
            async for chunk in self._model.stream_chat(
                model,
                messages,
                TOOL_SCHEMAS,
                options=model_options,
                think=think,
            ):
                if chunk.content:
                    round_content.append(chunk.content)
                    visible_content.append(chunk.content)
                    yield AgentEvent("delta", {"text": chunk.content})
                calls.extend(chunk.tool_calls)
                if chunk.usage:
                    usage = chunk.usage

            if "".join(round_content).lstrip().startswith("【参数补充请求】"):
                if created:
                    message = "已保存文件后不能再发出参数补充请求"
                    raise AgentError(
                        message,
                        failure_context=_failure_context(message, tool_failures),
                    )
                yield AgentEvent(
                    "done",
                    {"content": "".join(visible_content), "model": model, "artifacts": [], "usage": usage},
                )
                return

            if not calls:
                if tool_failure_repair_pending:
                    assistant_content = "".join(round_content)
                    if assistant_content.strip():
                        messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({"role": "user", "content": TOOL_FAILURE_REPAIR_PROMPT})
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
                raise AgentError(
                    message,
                    failure_context=_failure_context(message, tool_failures),
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(round_content),
                    "tool_calls": [
                        {"function": {"name": call.name, "arguments": call.arguments}}
                        for call in calls
                    ],
                }
            )
            tool_failure_repair_pending = False
            for call in calls:
                yield AgentEvent("status", {"phase": "tool_started", "tool": call.name})
                try:
                    result, artifact = await self._call_tool(
                        call,
                        account_id,
                        conversation_id,
                        validation=validation,
                        requirements=requirements,
                        model=model,
                        options=model_options,
                    )
                except (ValueError, OSError) as exc:
                    result, artifact = {"error": str(exc)}, None
                if result.get("error") or result.get("passed") is False:
                    tool_failure_repair_pending = True
                    tool_failures.append(
                        f"{call.name}: {json.dumps(result, ensure_ascii=False)}"
                    )
                if artifact:
                    created.append(artifact)
                    yield AgentEvent(
                        "artifact",
                        {"artifact_id": artifact.artifact_id, "filename": artifact.filename},
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                yield AgentEvent("status", {"phase": "tool_finished", "tool": call.name})

        message = "Agent 未能完成请求"
        raise AgentError(
            message,
            failure_context=_failure_context(message, tool_failures),
        )

    async def _call_tool(
        self,
        call: ToolCall,
        account_id: str,
        conversation_id: str | None,
        *,
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
                result = await self._functional_tester.run(
                    source,
                    requirements=requirements,
                    model=model,
                    options=options,
                )
            except (ValueError, OSError):
                validation.functional_passed.discard(digest)
                raise
            except Exception as exc:
                validation.functional_passed.discard(digest)
                raise ValueError(f"功能检测失败：{exc}") from exc
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
