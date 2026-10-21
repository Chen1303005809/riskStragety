"""Reusable skill loading and isolated test-case construction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent.contracts import ChatTurn, ModelChunk, ModelGateway, ToolCall
from agent.functional_tests import (
    FunctionalTestCase,
    FunctionalTestError,
    SkillFunctionalTester,
    SkillTestCaseBuilder,
)
from agent.skills import FilesystemSkillLoader, Skill, SkillLoadError


class MemorySkillLoader:
    def __init__(self, instructions: str = "test skill") -> None:
        self.instructions = instructions
        self.loaded: list[str] = []

    async def load(self, skill_id: str) -> Skill:
        self.loaded.append(skill_id)
        return Skill(skill_id, self.instructions)


class CaseModel(ModelGateway):
    def __init__(self, arguments: dict[str, Any]) -> None:
        self.arguments = arguments
        self.seen: list[dict[str, Any]] = []

    async def list_models(self) -> list[str]:
        return ["model-a"]

    async def stream_chat(
        self, model, messages, tools, *, options, think
    ) -> AsyncIterator[ModelChunk]:
        self.seen.append(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "options": options,
                "think": think,
            }
        )
        yield ModelChunk(tool_calls=(ToolCall("submit_test_cases", self.arguments),))


class MemorySandbox:
    def __init__(self) -> None:
        self.source: str | None = None
        self.cases: Sequence[FunctionalTestCase] = ()
        self.results: list[dict[str, Any]] | None = None

    async def run(self, source: str, cases: Sequence[FunctionalTestCase]):
        self.source = source
        self.cases = cases
        if self.results is not None:
            return self.results
        return [
            {
                "name": case.name,
                "passed": True,
                "expected_warning_count": case.expected_warning_count,
                "actual_warning_count": case.expected_warning_count,
            }
            for case in cases
        ]


def test_filesystem_skill_loader_loads_confined_skill(tmp_path: Path) -> None:
    skill_file = tmp_path / "test_case_builder" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("# reusable instructions\n", encoding="utf-8")

    skill = asyncio.run(FilesystemSkillLoader(tmp_path).load("test_case_builder"))

    assert skill.skill_id == "test_case_builder"
    assert skill.instructions == "# reusable instructions"


@pytest.mark.parametrize("skill_id", ["../secret", "test/case", "UpperCase", ""])
def test_filesystem_skill_loader_rejects_invalid_ids(tmp_path: Path, skill_id: str) -> None:
    with pytest.raises(SkillLoadError, match="skill_id"):
        asyncio.run(FilesystemSkillLoader(tmp_path).load(skill_id))


def test_filesystem_skill_loader_reports_missing_and_oversized_skills(tmp_path: Path) -> None:
    loader = FilesystemSkillLoader(tmp_path, max_bytes=8)
    with pytest.raises(SkillLoadError, match="不存在"):
        asyncio.run(loader.load("missing"))

    skill_file = tmp_path / "large" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("0123456789", encoding="utf-8")
    with pytest.raises(SkillLoadError, match="大小限制"):
        asyncio.run(loader.load("large"))


def test_filesystem_skill_loader_rejects_invalid_utf8_and_symlinks(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid" / "SKILL.md"
    invalid.parent.mkdir()
    invalid.write_bytes(b"\xff")
    with pytest.raises(SkillLoadError, match="UTF-8"):
        asyncio.run(FilesystemSkillLoader(tmp_path).load("invalid"))

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("outside", encoding="utf-8")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SkillLoadError, match="不存在或不可读取"):
        asyncio.run(FilesystemSkillLoader(tmp_path).load("linked"))


def test_case_builder_uses_skill_and_separate_structured_model_request() -> None:
    model = CaseModel(
        {
            "cases": [
                {
                    "name": "突破阈值",
                    "ticks": [{"vtSymbol": "rb2710", "lastPrice": 3801}],
                    "expected_warning_count": 1,
                }
            ]
        }
    )
    skill_loader = MemorySkillLoader("construct tests from requirements")
    requirements = [
        ChatTurn("user", "SHFE rb2710 超过 3800 时提醒"),
        ChatTurn("assistant", "请先查看接口"),
        ChatTurn("user", "补充：只提醒一次"),
    ]
    source = "class RiskStrategy(RHTemplate):\n    pass\n"

    cases = asyncio.run(
        SkillTestCaseBuilder(model, skill_loader).build(
            source=source,
            requirements=requirements,
            model="model-a",
            options={"temperature": 0.4, "top_p": 0.8, "num_ctx": 32_768},
        )
    )

    request = model.seen[0]
    user_payload = json.loads(request["messages"][1].content)
    assert skill_loader.loaded == ["test_case_builder"]
    assert request["messages"][0].content == "construct tests from requirements"
    assert user_payload["candidate_source"] == source
    assert user_payload["conversation"][-1]["content"] == "补充：只提醒一次"
    assert request["model"] == "model-a"
    assert [tool["function"]["name"] for tool in request["tools"]] == ["submit_test_cases"]
    assert request["options"] == {"temperature": 0.0, "top_p": 1.0, "num_ctx": 32_768}
    assert request["think"] is False
    assert cases[0].name == "突破阈值"
    assert cases[0].ticks[0].vt_symbol == "rb2710"
    assert cases[0].expected_warning_count == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"cases": []},
        {
            "cases": [
                {
                    "name": "坏价格",
                    "ticks": [{"vtSymbol": "rb2710", "lastPrice": float("nan")}],
                    "expected_warning_count": 0,
                }
            ]
        },
        {
            "cases": [
                {
                    "name": "布尔价格",
                    "ticks": [{"vtSymbol": "rb2710", "lastPrice": True}],
                    "expected_warning_count": 0,
                }
            ]
        },
    ],
)
def test_case_builder_rejects_invalid_test_data(arguments: dict[str, Any]) -> None:
    builder = SkillTestCaseBuilder(CaseModel(arguments), MemorySkillLoader())
    with pytest.raises(FunctionalTestError):
        asyncio.run(
            builder.build(
                source="class RiskStrategy: pass",
                requirements=[ChatTurn("user", "测试")],
                model="model-a",
                options={"num_ctx": 16_384},
            )
        )


def test_case_builder_fails_closed_if_skill_model_does_not_submit_cases() -> None:
    class NoToolModel(CaseModel):
        async def stream_chat(self, model, messages, tools, *, options, think):
            self.seen.append({"model": model, "messages": messages})
            yield ModelChunk(content="没有结构化用例")

    with pytest.raises(FunctionalTestError, match="submit_test_cases"):
        asyncio.run(
            SkillTestCaseBuilder(NoToolModel({}), MemorySkillLoader()).build(
                source="class RiskStrategy: pass",
                requirements=[ChatTurn("user", "测试")],
                model="model-a",
                options={"num_ctx": 16_384},
            )
        )


def test_case_builder_does_not_fallback_when_skill_load_fails() -> None:
    class MissingSkillLoader:
        async def load(self, skill_id: str) -> Skill:
            raise SkillLoadError(f"missing: {skill_id}")

    model = CaseModel({"cases": []})
    with pytest.raises(SkillLoadError, match="missing: test_case_builder"):
        asyncio.run(
            SkillTestCaseBuilder(model, MissingSkillLoader()).build(
                source="class RiskStrategy: pass",
                requirements=[ChatTurn("user", "测试")],
                model="model-a",
                options={"num_ctx": 16_384},
            )
        )
    assert model.seen == []


def test_skill_functional_tester_runs_built_cases() -> None:
    model = CaseModel(
        {
            "cases": [
                {
                    "name": "no trigger",
                    "ticks": [{"vtSymbol": "rb2710", "lastPrice": 3799}],
                    "expected_warning_count": 0,
                }
            ]
        }
    )
    sandbox = MemorySandbox()
    source = "class RiskStrategy(RHTemplate):\n    pass\n"

    result = asyncio.run(
        SkillFunctionalTester(
            SkillTestCaseBuilder(model, MemorySkillLoader()), sandbox
        ).run(
            source,
            requirements=[ChatTurn("user", "价格超过 3800 时提醒")],
            model="model-a",
            options={"num_ctx": 16_384},
        )
    )

    assert result["passed"] is True
    assert result["case_count"] == 1
    assert sandbox.source == source
    assert sandbox.cases[0].expected_warning_count == 0


def test_skill_functional_tester_returns_failed_case_inputs() -> None:
    ticks = [
        {"vtSymbol": "AP701", "lastPrice": 5976.0},
        {"vtSymbol": "AP701", "lastPrice": 5969.0},
    ]
    model = CaseModel(
        {
            "cases": [
                {
                    "name": "第二个完整周期再次触发",
                    "ticks": ticks,
                    "expected_warning_count": 2,
                }
            ]
        }
    )
    sandbox = MemorySandbox()
    sandbox.results = [
        {
            "name": "第二个完整周期再次触发",
            "passed": False,
            "expected_warning_count": 2,
            "actual_warning_count": 1,
        }
    ]

    result = asyncio.run(
        SkillFunctionalTester(
            SkillTestCaseBuilder(model, MemorySkillLoader()), sandbox
        ).run(
            "class RiskStrategy(RHTemplate):\n    pass\n",
            requirements=[ChatTurn("user", "测试重复触发")],
            model="model-a",
            options={"num_ctx": 16_384},
        )
    )

    assert result["passed"] is False
    assert result["failed_cases"] == [
        {
            "index": 1,
            "name": "第二个完整周期再次触发",
            "ticks": ticks,
            "expected_warning_count": 2,
            "actual_warning_count": 1,
        }
    ]
