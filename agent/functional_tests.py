"""Skill-driven test-case construction and functional test orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agent.contracts import (
    ChatTurn,
    ConversationRecorder,
    ModelGateway,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from agent.skills import SkillLoader


TEST_CASE_SKILL_ID = "test_case_builder"
logger = logging.getLogger(__name__)
MAX_TEST_CASES = 24
MAX_TICKS_PER_CASE = 200
MAX_EVENTS_PER_CASE = 200
MAX_TOTAL_EVENTS = 2_000
MAX_BARS_PER_RESPONSE = 2_000
MAX_CASE_NAME_CHARS = 120
MAX_SYMBOL_CHARS = 80
MAX_CANDIDATE_SOURCE_BYTES = 1024 * 1024

KLINE_BAR_STRING_FIELDS = {
    "vtSymbol", "exchange", "tradingDay", "tradeDate", "time"
}
KLINE_BAR_NUMBER_FIELDS = {
    "openPrice", "highPrice", "lowPrice", "closePrice",
    "openInterest", "volume", "volumeDelta", "turnover",
}
KLINE_BAR_INTEGER_FIELDS = {"cycleType", "globalId"}
KLINE_BAR_FIELDS = (
    KLINE_BAR_STRING_FIELDS | KLINE_BAR_NUMBER_FIELDS | KLINE_BAR_INTEGER_FIELDS
)

_BAR_SCHEMA = {
    "type": "object",
    "properties": {
        "vtSymbol": {"type": "string"},
        "exchange": {"type": "string"},
        "cycleType": {"type": "integer"},
        "tradingDay": {"type": "string"},
        "tradeDate": {"type": "string"},
        "time": {"type": "string"},
        "openPrice": {"type": "number"},
        "highPrice": {"type": "number"},
        "lowPrice": {"type": "number"},
        "closePrice": {"type": "number"},
        "openInterest": {"type": "number"},
        "volume": {"type": "number"},
        "volumeDelta": {"type": "number"},
        "turnover": {"type": "number"},
        "globalId": {"type": "integer"},
    },
    "required": ["vtSymbol", "cycleType"],
}

_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["tick", "history_response", "kline"],
        },
        "vtSymbol": {"type": "string"},
        "lastPrice": {"type": "number"},
        "request_id": {"type": "integer"},
        "bars": {"type": "array", "items": _BAR_SCHEMA},
        "error": {"type": ["object", "null"]},
        "bar": _BAR_SCHEMA,
    },
    "required": ["type"],
}

SUBMIT_TEST_CASES_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_test_cases",
        "description": "提交由用户规则推导出的结构化行情测试用例。",
        "parameters": {
            "type": "object",
            "properties": {
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "ticks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "vtSymbol": {"type": "string"},
                                        "lastPrice": {"type": "number"},
                                    },
                                    "required": ["vtSymbol", "lastPrice"],
                                },
                            },
                            "events": {
                                "type": "array",
                                "items": _EVENT_SCHEMA,
                                "description": (
                                    "按序注入 tick、history_response 或实时 kline；"
                                    "与 ticks 二选一。"
                                ),
                            },
                            "expected_calls": {
                                "type": "object",
                                "properties": {
                                    "queryHistoryKline": {"type": "integer"},
                                    "subscribeKline": {"type": "integer"},
                                    "unsubscribeKline": {"type": "integer"},
                                },
                            },
                            "expected_warning_count": {"type": "integer"},
                        },
                        "required": ["name", "expected_warning_count"],
                    },
                }
            },
            "required": ["cases"],
        },
    },
}


class FunctionalTestError(ValueError):
    """Test cases could not be constructed or executed safely."""


@dataclass(frozen=True)
class TestTick:
    vt_symbol: str
    last_price: float

    def as_payload(self) -> dict[str, str | float]:
        return {"vtSymbol": self.vt_symbol, "lastPrice": self.last_price}


@dataclass(frozen=True)
class FunctionalTestCase:
    name: str
    ticks: tuple[TestTick, ...]
    expected_warning_count: int
    events: tuple[dict[str, Any], ...] = ()
    expected_calls: dict[str, int] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "expected_warning_count": self.expected_warning_count,
        }
        if self.events:
            payload["events"] = list(self.events)
        else:
            payload["ticks"] = [tick.as_payload() for tick in self.ticks]
        if self.expected_calls is not None:
            payload["expected_calls"] = self.expected_calls
        return payload

    @property
    def event_count(self) -> int:
        return len(self.events) if self.events else len(self.ticks)


class FunctionalSandbox(Protocol):
    async def run(
        self, source: str, cases: Sequence[FunctionalTestCase]
    ) -> list[dict[str, Any]]: ...


class SkillTestCaseBuilder:
    """Ask the selected model for tests using an independently loaded skill."""

    def __init__(self, model_gateway: ModelGateway, skill_loader: SkillLoader) -> None:
        self._model_gateway = model_gateway
        self._skill_loader = skill_loader

    async def build(
        self,
        *,
        source: str,
        requirements: Sequence[ChatTurn],
        model: str,
        options: dict[str, int | float],
        conversation_id: str | None = None,
        turn_id: str | None = None,
        conversation_recorder: ConversationRecorder | None = None,
    ) -> tuple[FunctionalTestCase, ...]:
        async def record_trace(event: str, data: dict[str, Any]) -> None:
            if (
                conversation_recorder is not None
                and conversation_id is not None
                and turn_id is not None
            ):
                await conversation_recorder.record(
                    conversation_id, turn_id, event, data
                )

        skill = await self._skill_loader.load(TEST_CASE_SKILL_ID)
        request = {
            "conversation": [
                {"role": turn.role, "content": turn.content} for turn in requirements
            ],
            "candidate_source": source,
        }
        messages = [
            SystemMessage(skill.instructions),
            UserMessage(json.dumps(request, ensure_ascii=False)),
        ]
        model_options = {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_ctx": int(options.get("num_ctx", 16_384)),
        }
        calls: list[ToolCall] = []
        await record_trace(
            "functional_test_model_request",
            {
                "model": model,
                "options": model_options,
                "think": False,
                "tools": [SUBMIT_TEST_CASES_SCHEMA],
                "messages": [
                    {"role": item.role, "content": item.content}
                    for item in messages
                ],
            },
        )
        try:
            async for chunk in self._model_gateway.stream_chat(
                model,
                messages,
                [SUBMIT_TEST_CASES_SCHEMA],
                options=model_options,
                think=False,
            ):
                await record_trace(
                    "functional_test_model_chunk",
                    {
                        "role": "assistant",
                        "content": chunk.content,
                        "reasoning_content": chunk.reasoning_content,
                        "tool_calls": [
                            {
                                "name": call.name,
                                "arguments": call.arguments,
                                "call_id": call.call_id,
                            }
                            for call in chunk.tool_calls
                        ],
                        "usage": chunk.usage,
                    },
                )
                calls.extend(chunk.tool_calls)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = str(exc).strip() or f"{type(exc).__name__}（异常消息为空）"
            await record_trace(
                "functional_test_model_error",
                {"exception_type": type(exc).__name__, "detail": detail},
            )
            logger.exception(
                "functional_test_case_builder_failed model=%s source_bytes=%d "
                "exception_type=%s detail=%s",
                model,
                len(source.encode("utf-8", errors="replace")),
                type(exc).__name__,
                detail[:2_000],
            )
            raise FunctionalTestError(f"测试用例构造模型调用失败：{detail}") from exc

        if len(calls) != 1 or calls[0].name != "submit_test_cases":
            raise FunctionalTestError("测试用例 Skill 必须且只能提交一次 submit_test_cases")
        cases = _parse_test_cases(calls[0].arguments)
        await record_trace(
            "functional_test_cases_built",
            {"cases": [case.as_payload() for case in cases]},
        )
        return cases


class SkillFunctionalTester:
    def __init__(self, case_builder: SkillTestCaseBuilder, sandbox: FunctionalSandbox) -> None:
        self._case_builder = case_builder
        self._sandbox = sandbox

    async def run(
        self,
        source: str,
        *,
        requirements: Sequence[ChatTurn],
        model: str,
        options: dict[str, int | float],
        conversation_id: str | None = None,
        turn_id: str | None = None,
        conversation_recorder: ConversationRecorder | None = None,
    ) -> dict[str, Any]:
        try:
            source_size = len(source.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise FunctionalTestError("策略源码必须是有效的 UTF-8 文本") from exc
        if source_size > MAX_CANDIDATE_SOURCE_BYTES:
            raise FunctionalTestError("策略源码超过 1 MiB")
        cases = await self._case_builder.build(
            source=source,
            requirements=requirements,
            model=model,
            options=options,
            conversation_id=conversation_id,
            turn_id=turn_id,
            conversation_recorder=conversation_recorder,
        )
        logger.info(
            "functional_test_cases_built model=%s case_count=%d event_count=%d",
            model,
            len(cases),
            sum(case.event_count for case in cases),
        )
        try:
            results = await self._sandbox.run(source, cases)
        except Exception as exc:
            if conversation_recorder is not None and conversation_id is not None and turn_id is not None:
                await conversation_recorder.record(
                    conversation_id,
                    turn_id,
                    "functional_sandbox_error",
                    {
                        "exception_type": type(exc).__name__,
                        "detail": str(exc).strip()
                        or f"{type(exc).__name__}（异常消息为空）",
                    },
                )
            logger.exception(
                "functional_test_sandbox_failed model=%s case_count=%d "
                "exception_type=%s detail=%s",
                model,
                len(cases),
                type(exc).__name__,
                (str(exc).strip() or f"{type(exc).__name__}（异常消息为空）")[:2_000],
            )
            raise
        if conversation_recorder is not None and conversation_id is not None and turn_id is not None:
            await conversation_recorder.record(
                conversation_id,
                turn_id,
                "functional_sandbox_result",
                {"cases": results},
            )
        if not isinstance(results, list) or len(results) != len(cases):
            raise FunctionalTestError("沙盒返回的测试用例数量与提交数量不一致")
        for case, result in zip(cases, results):
            if (
                not isinstance(result, dict)
                or result.get("name") != case.name
                or not isinstance(result.get("passed"), bool)
            ):
                raise FunctionalTestError("沙盒返回了无效的单项测试结果")
            expected = result.get("expected_warning_count")
            actual = result.get("actual_warning_count")
            if case.expected_calls is not None:
                reported_expected_calls = result.get("expected_calls")
                reported_actual_calls = result.get("actual_calls")
                if (
                    not isinstance(reported_actual_calls, dict)
                    or reported_expected_calls != case.expected_calls
                    or any(
                        isinstance(reported_actual_calls.get(name), bool)
                        or reported_actual_calls.get(name) != count
                        for name, count in case.expected_calls.items()
                    )
                ):
                    if result["passed"]:
                        raise FunctionalTestError("沙盒通过结果与记录的 K 线 API 调用次数不一致")
            if result["passed"]:
                if (
                    isinstance(expected, bool)
                    or expected != case.expected_warning_count
                    or isinstance(actual, bool)
                    or not isinstance(actual, int)
                    or actual != case.expected_warning_count
                ):
                    raise FunctionalTestError("沙盒通过结果与记录的预警数量不一致")
            elif expected is not None or actual is not None:
                if (
                    isinstance(expected, bool)
                    or expected != case.expected_warning_count
                    or isinstance(actual, bool)
                    or not isinstance(actual, int)
                ):
                    raise FunctionalTestError("沙盒返回了无效的预警数量记录")
        passed = all(result["passed"] for result in results)
        failed_case_indexes = [
            index for index, result in enumerate(results, start=1)
            if result["passed"] is False
        ]
        failed_cases: list[dict[str, Any]] = []
        for index, (case, result) in enumerate(zip(cases, results), start=1):
            if result["passed"] is not False:
                continue
            failure: dict[str, Any] = {
                "index": index,
                "name": case.name,
                "expected_warning_count": case.expected_warning_count,
            }
            if case.events:
                failure["events"] = list(case.events)
            else:
                failure["ticks"] = [tick.as_payload() for tick in case.ticks]
            if case.expected_calls:
                failure["expected_calls"] = case.expected_calls
            actual = result.get("actual_warning_count")
            if actual is not None:
                failure["actual_warning_count"] = actual
            actual_calls = result.get("actual_calls")
            if actual_calls is not None:
                failure["actual_calls"] = actual_calls
            error = result.get("error")
            if error is not None:
                failure["error"] = str(error)
            failed_cases.append(failure)
        logger.info(
            "functional_test_sandbox_finished model=%s passed=%s case_count=%d "
            "failed_case_indexes=%s",
            model,
            passed,
            len(results),
            failed_case_indexes,
        )
        result: dict[str, Any] = {
            "passed": passed,
            "case_count": len(results),
            "cases": results,
            "failed_cases": failed_cases,
        }
        return result


def _parse_test_cases(arguments: Any) -> tuple[FunctionalTestCase, ...]:
    if not isinstance(arguments, dict):
        raise FunctionalTestError("测试用例参数必须是 JSON 对象")
    if set(arguments) != {"cases"}:
        raise FunctionalTestError("测试用例参数只能包含 cases")
    raw_cases = arguments.get("cases")
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= MAX_TEST_CASES:
        raise FunctionalTestError(f"测试用例数量必须在 1 到 {MAX_TEST_CASES} 之间")

    cases: list[FunctionalTestCase] = []
    names: set[str] = set()
    total_events = 0
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise FunctionalTestError("每个测试用例必须是 JSON 对象")
        allowed_case_keys = {
            "name", "ticks", "events", "expected_calls", "expected_warning_count"
        }
        if set(raw_case) - allowed_case_keys:
            raise FunctionalTestError("测试用例包含不支持的字段")
        if ("ticks" in raw_case) == ("events" in raw_case):
            raise FunctionalTestError("每个测试用例必须且只能提供 ticks 或 events")
        if "name" not in raw_case or "expected_warning_count" not in raw_case:
            raise FunctionalTestError("每个测试用例必须包含 name 和 expected_warning_count")
        name = raw_case.get("name")
        ticks_data = raw_case.get("ticks")
        events_data = raw_case.get("events")
        expected_count = raw_case.get("expected_warning_count")
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > MAX_CASE_NAME_CHARS
            or name in names
        ):
            raise FunctionalTestError("测试用例名称无效或重复")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or not 0 <= expected_count <= 100
        ):
            raise FunctionalTestError("expected_warning_count 必须是 0 到 100 的整数")

        ticks: list[TestTick] = []
        events: list[dict[str, Any]] = []
        if ticks_data is not None:
            if not isinstance(ticks_data, list) or not 1 <= len(ticks_data) <= MAX_TICKS_PER_CASE:
                raise FunctionalTestError(
                    f"每个测试用例必须包含 1 到 {MAX_TICKS_PER_CASE} 个 tick"
                )
            ticks = [_parse_tick(raw_tick) for raw_tick in ticks_data]
            event_count = len(ticks)
        else:
            if not isinstance(events_data, list) or not 1 <= len(events_data) <= MAX_EVENTS_PER_CASE:
                raise FunctionalTestError(
                    f"每个测试用例必须包含 1 到 {MAX_EVENTS_PER_CASE} 个事件"
                )
            events = [_parse_event(raw_event) for raw_event in events_data]
            event_count = len(events)
        total_events += event_count
        if total_events > MAX_TOTAL_EVENTS:
            raise FunctionalTestError(f"测试事件总数不能超过 {MAX_TOTAL_EVENTS}")

        expected_calls = _parse_expected_calls(raw_case.get("expected_calls"))
        names.add(name)
        cases.append(
            FunctionalTestCase(
                name,
                tuple(ticks),
                expected_count,
                tuple(events),
                expected_calls,
            )
        )
    return tuple(cases)


def _parse_tick(raw_tick: Any) -> TestTick:
    if not isinstance(raw_tick, dict) or set(raw_tick) != {"vtSymbol", "lastPrice"}:
        raise FunctionalTestError("每个 tick 必须只包含 vtSymbol 和 lastPrice")
    symbol = raw_tick.get("vtSymbol")
    price = raw_tick.get("lastPrice")
    if (
        not isinstance(symbol, str)
        or not symbol.strip()
        or len(symbol) > MAX_SYMBOL_CHARS
    ):
        raise FunctionalTestError("tick.vtSymbol 无效")
    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(price)
        or price <= 0
    ):
        raise FunctionalTestError("tick.lastPrice 必须是有限正数")
    return TestTick(symbol, float(price))


def _parse_bar(raw_bar: Any) -> dict[str, Any]:
    if not isinstance(raw_bar, dict) or set(raw_bar) - KLINE_BAR_FIELDS:
        raise FunctionalTestError("K 线 bar 包含不支持的字段")
    symbol = raw_bar.get("vtSymbol")
    cycle_type = raw_bar.get("cycleType")
    if (
        not isinstance(symbol, str)
        or not symbol.strip()
        or len(symbol) > MAX_SYMBOL_CHARS
    ):
        raise FunctionalTestError("bar.vtSymbol 无效")
    if (
        isinstance(cycle_type, bool)
        or not isinstance(cycle_type, int)
        or not 1 <= cycle_type <= 14
    ):
        raise FunctionalTestError("bar.cycleType 必须是 1 到 14 的周期枚举")

    parsed = dict(raw_bar)
    for field in KLINE_BAR_STRING_FIELDS - {"vtSymbol"}:
        if field in parsed and not isinstance(parsed[field], str):
            raise FunctionalTestError(f"bar.{field} 必须是字符串")
    for field in KLINE_BAR_NUMBER_FIELDS:
        value = parsed.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise FunctionalTestError(f"bar.{field} 必须是有限数值")
    for field in KLINE_BAR_INTEGER_FIELDS - {"cycleType"}:
        value = parsed.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise FunctionalTestError(f"bar.{field} 必须是整数")
    return parsed


def _parse_event(raw_event: Any) -> dict[str, Any]:
    if not isinstance(raw_event, dict):
        raise FunctionalTestError("每个事件必须是 JSON 对象")
    kind = raw_event.get("type")
    if kind == "tick":
        if set(raw_event) != {"type", "vtSymbol", "lastPrice"}:
            raise FunctionalTestError("tick 事件必须包含 type、vtSymbol、lastPrice")
        tick = _parse_tick(
            {"vtSymbol": raw_event["vtSymbol"], "lastPrice": raw_event["lastPrice"]}
        )
        return {"type": "tick", **tick.as_payload()}
    if kind == "history_response":
        if set(raw_event) != {"type", "request_id", "bars", "error"}:
            raise FunctionalTestError(
                "history_response 必须包含 type、request_id、bars、error"
            )
        request_id = raw_event.get("request_id")
        bars = raw_event.get("bars")
        error = raw_event.get("error")
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 1:
            raise FunctionalTestError("history_response.request_id 必须是正整数")
        if not isinstance(bars, list) or len(bars) > MAX_BARS_PER_RESPONSE:
            raise FunctionalTestError(
                f"history_response.bars 必须是最多 {MAX_BARS_PER_RESPONSE} 根的数组"
            )
        if error is not None and not isinstance(error, dict):
            raise FunctionalTestError("history_response.error 必须是对象或 null")
        return {
            "type": "history_response",
            "request_id": request_id,
            "bars": [_parse_bar(bar) for bar in bars],
            "error": error,
        }
    if kind == "kline":
        if set(raw_event) != {"type", "bar"}:
            raise FunctionalTestError("kline 事件必须只包含 type 和 bar")
        return {"type": "kline", "bar": _parse_bar(raw_event.get("bar"))}
    raise FunctionalTestError("事件 type 只能是 tick、history_response 或 kline")


def _parse_expected_calls(raw_calls: Any) -> dict[str, int] | None:
    if raw_calls is None:
        return None
    supported = {"queryHistoryKline", "subscribeKline", "unsubscribeKline"}
    if not isinstance(raw_calls, dict) or set(raw_calls) - supported:
        raise FunctionalTestError("expected_calls 只允许 K 线查询、订阅和退订计数")
    parsed: dict[str, int] = {}
    for name, count in raw_calls.items():
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 100:
            raise FunctionalTestError("expected_calls 中的调用次数必须是 0 到 100 的整数")
        parsed[name] = count
    return parsed
