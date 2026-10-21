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
MAX_TOTAL_TICKS = 2_000
MAX_CASE_NAME_CHARS = 120
MAX_SYMBOL_CHARS = 80
MAX_CANDIDATE_SOURCE_BYTES = 1024 * 1024

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
                            "expected_warning_count": {"type": "integer"},
                        },
                        "required": ["name", "ticks", "expected_warning_count"],
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

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ticks": [tick.as_payload() for tick in self.ticks],
            "expected_warning_count": self.expected_warning_count,
        }


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
            "functional_test_cases_built model=%s case_count=%d tick_count=%d",
            model,
            len(cases),
            sum(len(case.ticks) for case in cases),
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
        logger.info(
            "functional_test_sandbox_finished model=%s passed=%s case_count=%d "
            "failed_case_indexes=%s",
            model,
            passed,
            len(results),
            failed_case_indexes,
        )
        return {
            "passed": passed,
            "case_count": len(results),
            "cases": results,
        }


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
    total_ticks = 0
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict) or set(raw_case) != {
            "name", "ticks", "expected_warning_count"
        }:
            raise FunctionalTestError("每个测试用例必须包含 name、ticks、expected_warning_count")
        name = raw_case.get("name")
        ticks_data = raw_case.get("ticks")
        expected_count = raw_case.get("expected_warning_count")
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > MAX_CASE_NAME_CHARS
            or name in names
        ):
            raise FunctionalTestError("测试用例名称无效或重复")
        if not isinstance(ticks_data, list) or not 1 <= len(ticks_data) <= MAX_TICKS_PER_CASE:
            raise FunctionalTestError(
                f"每个测试用例必须包含 1 到 {MAX_TICKS_PER_CASE} 个 tick"
            )
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or not 0 <= expected_count <= 100
        ):
            raise FunctionalTestError("expected_warning_count 必须是 0 到 100 的整数")

        ticks: list[TestTick] = []
        for raw_tick in ticks_data:
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
            ticks.append(TestTick(symbol, float(price)))

        total_ticks += len(ticks)
        if total_ticks > MAX_TOTAL_TICKS:
            raise FunctionalTestError(f"测试 tick 总数不能超过 {MAX_TOTAL_TICKS}")
        names.add(name)
        cases.append(FunctionalTestCase(name, tuple(ticks), expected_count))
    return tuple(cases)
