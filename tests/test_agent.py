"""Agent behavior with deliberately small, deterministic adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from agent.contracts import (
    AgentMessage,
    Artifact,
    AssistantMessage,
    AssistantToolCall,
    ChatTurn,
    ModelChunk,
    ModelGateway,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
)
from agent.prompt import PROMPT_FILE, build_system_prompt
from agent.runner import AgentError, AgentRunner, TOOL_SCHEMAS, _ValidationState, _source_digest
from agent.skills import FilesystemSkillLoader


class MemoryArtifacts:
    def __init__(self) -> None:
        self.created: list[tuple[str, str, str]] = []

    async def create(self, filename: str, source: str, account_id: str) -> Artifact:
        self.created.append((filename, source, account_id))
        return Artifact(f"{len(self.created):032x}", filename, account_id)


class PassingFunctionalTester:
    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.calls: list[dict[str, Any]] = []

    async def run(self, source, *, requirements, model, options):
        self.calls.append(
            {"source": source, "requirements": list(requirements), "model": model, "options": options}
        )
        return {
            "passed": self.passed,
            "case_count": 1,
            "cases": [{"name": "case", "passed": self.passed}],
        }


class ScriptedModel(ModelGateway):
    def __init__(self, rounds: list[list[ModelChunk]]) -> None:
        self.rounds = rounds
        self.seen: list[list[AgentMessage]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []
        self.settings_seen: list[tuple[dict[str, int | float], bool]] = []
        self.preflight_seen: list[list[AgentMessage]] = []
        self.preflight_tools_seen: list[list[dict[str, Any]]] = []
        self.preflight_settings_seen: list[tuple[dict[str, int | float], bool]] = []
        self.next_round = 0

    async def list_models(self) -> list[str]:
        return ["model-a"]

    async def stream_chat(
        self, model, messages, tools, *, options, think
    ) -> AsyncIterator[ModelChunk]:
        assert model == "model-a"
        if (
            len(tools) == 1
            and tools[0]["function"]["name"] == "lookup_exchange_by_contract_code"
        ):
            self.preflight_seen.append(messages.copy())
            self.preflight_tools_seen.append(tools.copy())
            self.preflight_settings_seen.append((options.copy(), think))
            scripted = self.rounds[self.next_round] if self.next_round < len(self.rounds) else []
            if scripted and any(
                call.name == "lookup_exchange_by_contract_code"
                for chunk in scripted
                for call in chunk.tool_calls
            ):
                self.next_round += 1
                for chunk in scripted:
                    yield chunk
            else:
                yield ModelChunk()
            return
        self.seen.append(messages.copy())
        self.tools_seen.append(tools.copy())
        self.settings_seen.append((options.copy(), think))
        scripted = self.rounds[self.next_round]
        self.next_round += 1
        for chunk in scripted:
            yield chunk


def collect(runner: AgentRunner, **kwargs):
    async def read():
        return [event async for event in runner.run(**kwargs)]

    return asyncio.run(read())


def _strategy_source(name: str = "RiskStrategy", body: str = "    pass\n") -> str:
    return f"class My_PythonObj(RHTemplate):\n    name = {name!r}\n{body}"


def _is_contract_lookup_preflight(tools: list[dict[str, Any]]) -> bool:
    return (
        len(tools) == 1
        and tools[0]["function"]["name"] == "lookup_exchange_by_contract_code"
    )


def test_current_time_is_a_volatile_suffix_after_the_stable_prompt() -> None:
    original = PROMPT_FILE.read_text(encoding="utf-8").removesuffix("\n")
    first = build_system_prompt(
        now=datetime(2026, 10, 16, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    second = build_system_prompt(
        now=datetime(2026, 10, 17, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    stable_first, time_first = first.rsplit("\n\n# 当前上海时间\n", 1)
    stable_second, time_second = second.rsplit("\n\n# 当前上海时间\n", 1)

    assert stable_first == stable_second
    assert stable_first.startswith(original + "\n\n# 通用工具与 Skill 规则")
    assert time_first != time_second
    assert first.endswith("如果当前任务需要日期，以此时间为准。")
    assert "list_skills" in stable_first
    assert "load_skill" in stable_first
    assert "check_python_syntax" not in stable_first
    assert "R1–R6" not in stable_first


def test_exchange_lookup_success_flows_into_strategy_generation_without_exchange_question() -> None:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
    contract_code = f"rb{next_month.strftime('%y%m')}"
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall(
                "lookup_exchange_by_contract_code", {"contract_code": contract_code}
            ),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="已生成策略。")],
        ]
    )
    artifacts = MemoryArtifacts()

    events = collect(
        AgentRunner(model, artifacts, functional_tester=PassingFunctionalTester()),
        message=f"{contract_code} 超过 3800 时提醒",
        history=[], model="model-a", account_id="alice",
    )

    lookup_result = next(
        item.content for item in model.seen[0]
        if isinstance(item, SystemMessage) and item.content.startswith("# 合约预查询工具结果")
    )
    assert len(model.preflight_seen) == 1
    assert model.preflight_tools_seen[0][0]["function"]["name"] == "lookup_exchange_by_contract_code"
    assert '"exchange_code": "SHFE"' in lookup_result
    assert '"status": "matched"' in lookup_result
    assert artifacts.created == [("RiskStrategy.py", source, "alice")]
    assert events[-1].data["content"] == "已生成策略。"
    assert "交易所" not in events[-1].data["content"]


def test_unrecognized_contract_asks_only_for_exchange() -> None:
    contract_code = "UNKNOWN2611"
    expected_status = "unrecognized"
    expected_followup = "合约所属交易所"
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall(
                "lookup_exchange_by_contract_code", {"contract_code": contract_code}
            ),))],
            [ModelChunk(content=f"请补充以下信息：{expected_followup}")],
        ]
    )

    events = collect(
        AgentRunner(model, MemoryArtifacts()),
        message=f"{contract_code} 超过 3800 时提醒",
        history=[], model="model-a", account_id="alice",
    )

    lookup_result = next(
        item.content for item in model.seen[0]
        if isinstance(item, SystemMessage) and item.content.startswith("# 合约预查询工具结果")
    )
    assert f'"status": "{expected_status}"' in lookup_result
    assert events[-1].kind == "done"
    assert expected_followup in events[-1].data["content"]


def test_expired_contract_asks_for_a_valid_contract_code() -> None:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    previous_month = now.replace(day=1) - timedelta(days=1)
    contract_code = f"rb{previous_month.strftime('%y%m')}"
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall(
                "lookup_exchange_by_contract_code", {"contract_code": contract_code}
            ),))],
            [ModelChunk(content="请提供有效合约代码。")],
        ]
    )

    events = collect(
        AgentRunner(model, MemoryArtifacts()),
        message=f"{contract_code} 超过 3800 时提醒",
        history=[], model="model-a", account_id="alice",
    )

    lookup_result = next(
        item.content for item in model.seen[0]
        if isinstance(item, SystemMessage) and item.content.startswith("# 合约预查询工具结果")
    )
    assert '"status": "expired"' in lookup_result
    assert events[-1].kind == "done"
    assert "提供有效合约代码" in events[-1].data["content"]


def test_functional_test_write_reply_flow() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall(
                "lookup_exchange_by_contract_code", {"contract_code": "rb2710"}
            ),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="已生成。", usage={"prompt_tokens": 42})],
        ]
    )
    artifacts = MemoryArtifacts()
    functional_tester = PassingFunctionalTester()
    events = collect(
        AgentRunner(model, artifacts, functional_tester=functional_tester),
        message="SHFE rb2710 超过 3800 时提醒",
        history=[],
        model="model-a",
        account_id="alice",
        options={"temperature": 0.55, "top_p": 0.9, "num_ctx": 32_768},
        think=True,
    )
    assert artifacts.created == [("RiskStrategy.py", source, "alice")]
    assert functional_tester.calls[0]["source"] == source
    assert functional_tester.calls[0]["requirements"] == [
        ChatTurn("user", "SHFE rb2710 超过 3800 时提醒")
    ]
    assert [event.kind for event in events if event.kind in {"artifact", "delta", "done"}] == [
        "artifact", "delta", "done"
    ]
    assert events[-1].data["artifacts"] == [
        {"artifact_id": "00000000000000000000000000000001", "filename": "RiskStrategy.py"}
    ]
    assert isinstance(model.seen[1][-1], ToolResultMessage)
    assert model.seen[1][-1].tool_name == "run_functional_test"
    assert '"passed": true' in model.seen[1][-1].content
    assert isinstance(model.seen[2][-1], ToolResultMessage)
    assert model.seen[2][-1].tool_name == "write_generated_file"
    assert model.settings_seen == [
        ({"temperature": 0.55, "top_p": 0.9, "num_ctx": 32_768}, True)
    ] * 3


def test_python_312_syntax_is_not_blocked_by_a_python_37_gate() -> None:
    source = _strategy_source(
        body=(
            "    def onTick(self, tick):\n"
            "        match tick.vtSymbol:\n"
            "            case 'contract':\n"
            "                return\n"
        )
    )
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="策略已生成。")],
        ]
    )
    tester = PassingFunctionalTester()
    artifacts = MemoryArtifacts()

    events = collect(
        AgentRunner(model, artifacts, functional_tester=tester),
        message="生成策略",
        history=[],
        model="model-a",
        account_id="alice",
    )

    assert tester.calls[0]["source"] == source
    assert artifacts.created == [("RiskStrategy.py", source, "alice")]
    assert events[-1].data["content"] == "策略已生成。"


def test_runner_preserves_unique_provider_call_ids_and_pairs_results() -> None:
    model = ScriptedModel(
        [
            [
                ModelChunk(
                    tool_calls=(
                        ToolCall("list_skills", {}),
                        ToolCall("list_skills", {}, "agent_call_0"),
                        ToolCall("list_skills", {}, "provider-call-id"),
                        ToolCall("list_skills", {}, "provider-call-id"),
                    )
                )
            ],
            [ModelChunk(content="请补充策略条件。")],
        ]
    )
    runner = AgentRunner(model, MemoryArtifacts())

    events = collect(
        runner,
        message="创建策略",
        history=[],
        model="model-a",
        account_id="alice",
    )

    assistant_message = model.seen[1][-5]
    tool_results = model.seen[1][-4:]
    assert isinstance(assistant_message, AssistantMessage)
    assert [call.call_id for call in assistant_message.tool_calls] == [
        "agent_call_1",
        "agent_call_0",
        "provider-call-id",
        "agent_call_2",
    ]
    assert all(isinstance(item, ToolResultMessage) for item in tool_results)
    assert [item.call_id for item in tool_results] == [
        call.call_id for call in assistant_message.tool_calls
    ]
    assert all(event.kind != "error" for event in events)


def test_tool_schemas_expose_skills_but_no_static_or_interface_readers() -> None:
    tool_names = {tool["function"]["name"] for tool in TOOL_SCHEMAS}
    assert {"list_skills", "load_skill", "run_functional_test", "write_generated_file"} <= tool_names
    assert not {
        "check_python_syntax",
        "list_interface_files",
        "read_interface_file",
    } & tool_names


def test_agent_discovers_and_loads_skill_instructions_and_markdown_references() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall(
                "lookup_exchange_by_contract_code", {"contract_code": "rb2710"}
            ),))],
            [ModelChunk(tool_calls=(ToolCall("list_skills", {}),))],
            [ModelChunk(tool_calls=(ToolCall("load_skill", {
                "skill_id": "price-alert-strategy"
            }),))],
            [ModelChunk(tool_calls=(ToolCall("load_skill", {
                "skill_id": "price-alert-strategy",
                "resource_path": "references/interface-contract.md",
            }),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="策略已生成。")],
        ]
    )
    artifacts = MemoryArtifacts()

    events = collect(
        AgentRunner(
            model,
            artifacts,
            functional_tester=PassingFunctionalTester(),
            skill_loader=FilesystemSkillLoader(),
        ),
        message="为 rb2710 生成价格预警策略",
        history=[],
        model="model-a",
        account_id="alice",
    )

    listed = json.loads(model.seen[1][-1].content)
    price_skill = next(
        item for item in listed["skills"]
        if item["skill_id"] == "price-alert-strategy"
    )
    assert not any(item["skill_id"] == "test_case_builder" for item in listed["skills"])
    assert price_skill == {
        "skill_id": "price-alert-strategy",
        "name": "价格预警策略",
        "description": (
            "创建或修改基于价格行情回调的期货预警策略源码；"
            "当任务涉及本项目价格接口代码时使用。"
        ),
    }
    assert all(isinstance(message, SystemMessage) for message in model.seen[3][:4])
    assert "R1 单阈值状态报警" in model.seen[3][1].content
    assert "# 价格接口契约参考" in model.seen[3][2].content
    assert '"exchange_code": "SHFE"' in model.seen[3][3].content
    assert len(model.preflight_seen) == 1
    assert len(model.preflight_tools_seen[0]) == 1
    assert model.preflight_tools_seen[0][0]["function"]["name"] == "lookup_exchange_by_contract_code"
    assert all(
        tool["function"]["name"] not in {
            "check_python_syntax",
            "list_interface_files",
            "read_interface_file",
        }
        for round_tools in [*model.tools_seen, *model.preflight_tools_seen]
        for tool in round_tools
    )
    assert artifacts.created == [("RiskStrategy.py", source, "alice")]
    assert events[-1].data["content"] == "策略已生成。"


def test_contract_code_casing_is_checked_for_testing_and_writing() -> None:
    source = _strategy_source(body="    contract_code = 'RB2710'\n")
    tester = PassingFunctionalTester()
    runner = AgentRunner(ScriptedModel([]), MemoryArtifacts(), functional_tester=tester)
    requirements = [ChatTurn("user", "SHFE rb2710")]

    async def call_tool(name: str, arguments: dict[str, Any], state: _ValidationState):
        return await runner._call_tool(
            AssistantToolCall(f"call-{name}", name, arguments),
            "alice",
            None,
            turn_id=None,
            conversation_recorder=None,
            validation=state,
            requirements=requirements,
            model="model-a",
            options={},
        )

    async def scenario() -> None:
        validation_before_test = _ValidationState()
        await call_tool(
            "lookup_exchange_by_contract_code",
            {"contract_code": "rb2710"},
            validation_before_test,
        )
        with pytest.raises(ValueError, match="大小写"):
            await call_tool(
                "run_functional_test",
                {"source": source},
                validation_before_test,
            )
        assert tester.calls == []

        validation_before_write = _ValidationState()
        await call_tool(
            "lookup_exchange_by_contract_code",
            {"contract_code": "rb2710"},
            validation_before_write,
        )
        validation_before_write.functional_passed.add(_source_digest(source))
        with pytest.raises(ValueError, match="大小写"):
            await call_tool(
                "write_generated_file",
                {"source": source},
                validation_before_write,
            )

    asyncio.run(scenario())


def test_invalid_or_unqueried_contract_cannot_reach_functional_testing() -> None:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    previous_month = now.replace(day=1) - timedelta(days=1)
    contract_code = f"rb{previous_month.strftime('%y%m')}"
    source = _strategy_source(body=f"    contract_code = {contract_code!r}\n")
    tester = PassingFunctionalTester()
    runner = AgentRunner(ScriptedModel([]), MemoryArtifacts(), functional_tester=tester)

    async def call_tool(
        name: str, arguments: dict[str, Any], state: _ValidationState
    ) -> dict[str, Any]:
        result, _ = await runner._call_tool(
            AssistantToolCall(f"call-{name}", name, arguments),
            "alice",
            None,
            turn_id=None,
            conversation_recorder=None,
            validation=state,
            requirements=[ChatTurn("user", f"{contract_code} 价格提醒")],
            model="model-a",
            options={},
        )
        return result

    async def scenario() -> None:
        unqueried = _ValidationState()
        with pytest.raises(ValueError, match="尚未通过有效性查询"):
            await call_tool("run_functional_test", {"source": source}, unqueried)

        expired = _ValidationState()
        result = await call_tool(
            "lookup_exchange_by_contract_code",
            {"contract_code": contract_code},
            expired,
        )
        assert result["status"] == "expired"
        with pytest.raises(ValueError, match="尚未通过有效性查询"):
            await call_tool("run_functional_test", {"source": source}, expired)
        assert tester.calls == []

    asyncio.run(scenario())


def test_missing_parameters_ask_without_creating_file() -> None:
    model = ScriptedModel([[
        ModelChunk(content="之前的策略文件已生成。请提供修改所需的触发阈值。")
    ]])
    artifacts = MemoryArtifacts()
    events = collect(
        AgentRunner(model, artifacts),
        message="螺纹钢突破时提醒",
        history=[
            ChatTurn("user", "生成初版策略"),
            ChatTurn("assistant", "已有策略文件可继续修改。"),
        ],
        model="model-a",
        account_id="alice",
    )
    assert artifacts.created == []
    assert events[-1].kind == "done"
    assert events[-1].data["content"] == "之前的策略文件已生成。请提供修改所需的触发阈值。"
    assert events[-1].data["_completion"] == "needs_input"


def test_empty_reply_is_not_reported_as_success() -> None:
    model = ScriptedModel([[ModelChunk(content="")]])
    with pytest.raises(AgentError, match="没有返回内容"):
        collect(
            AgentRunner(model, MemoryArtifacts()),
            message="生成", history=[], model="model-a", account_id="alice"
        )


def test_non_question_reply_without_artifact_is_still_an_agent_failure() -> None:
    model = ScriptedModel([[ModelChunk(content="好的，我会继续处理。")]])
    with pytest.raises(AgentError, match="未有效询问缺少的信息"):
        collect(
            AgentRunner(model, MemoryArtifacts()),
            message="生成策略", history=[], model="model-a", account_id="alice"
        )


def test_clarification_cannot_write_even_if_model_also_calls_write_tool() -> None:
    model = ScriptedModel([
        [ModelChunk(
            content="请提供阈值。",
            tool_calls=(ToolCall("write_generated_file", {
                "filename": "should-not-exist.py", "source": "x = 1\n"
            }),),
        )],
        [ModelChunk(content="请提供阈值。")],
        [ModelChunk(content="请提供阈值。")],
    ])
    artifacts = MemoryArtifacts()
    events = collect(
        AgentRunner(model, artifacts),
        message="螺纹钢突破时提醒", history=[], model="model-a", account_id="alice"
    )
    assert artifacts.created == []
    assert [event.kind for event in events].count("done") == 1
    assert events[-1].data["_completion"] == "needs_input"


def test_saved_artifact_can_be_followed_by_parameter_request() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="请补充修改后的触发阈值。")],
        ]
    )
    artifacts = MemoryArtifacts()
    events = collect(
        AgentRunner(model, artifacts, functional_tester=PassingFunctionalTester()),
        message="先按已知规则生成，后续还需确认一个参数",
        history=[],
        model="model-a",
        account_id="alice",
    )

    assert artifacts.created == [("RiskStrategy.py", source, "alice")]
    assert events[-1].kind == "done"
    assert events[-1].data["artifacts"] == [
        {"artifact_id": "0" * 31 + "1", "filename": "RiskStrategy.py"}
    ]
    assert "_completion" not in events[-1].data


def test_unsaved_artifact_claim_is_reworked_until_file_is_written() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(content="策略文件已生成。")],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="策略文件已保存。")],
        ]
    )
    artifacts = MemoryArtifacts()
    events = collect(
        AgentRunner(model, artifacts, functional_tester=PassingFunctionalTester()),
        message="生成策略",
        history=[],
        model="model-a",
        account_id="alice",
    )

    assert artifacts.created == [("RiskStrategy.py", source, "alice")]
    assert events[-1].kind == "done"
    assert len(events[-1].data["artifacts"]) == 1
    assert events[-1].data["content"] == "策略文件已保存。"
    assert any(
        item.role == "user" and "本轮没有成功保存产物" in item.content
        for item in model.seen[1]
    )


def test_unsaved_artifact_claim_can_be_corrected_to_a_natural_question() -> None:
    model = ScriptedModel([
        [ModelChunk(content="策略文件已生成。")],
        [ModelChunk(content="请提供触发阈值。")],
    ])
    events = collect(
        AgentRunner(model, MemoryArtifacts()),
        message="生成策略",
        history=[],
        model="model-a",
        account_id="alice",
    )

    assert events[-1].kind == "done"
    assert events[-1].data["_completion"] == "needs_input"
    assert events[-1].data["content"] == "请提供触发阈值。"


def test_repeated_unsaved_artifact_claim_stops_after_bounded_rework() -> None:
    model = ScriptedModel([[ModelChunk(content="策略文件已生成。")]] * 4)
    with pytest.raises(AgentError, match="多次声称已生成"):
        collect(
            AgentRunner(model, MemoryArtifacts()),
            message="生成策略",
            history=[],
            model="model-a",
            account_id="alice",
        )
    assert len(model.seen) == 4


def test_followup_receives_prior_turns_and_can_create_revision() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall(
                "lookup_exchange_by_contract_code", {"contract_code": "rb2710"}
            ),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="已将阈值改为 3900。")],
        ]
    )
    artifacts = MemoryArtifacts()
    functional_tester = PassingFunctionalTester()
    history = [
        ChatTurn("user", "SHFE rb2710 超过 3800 时提醒"),
        ChatTurn("assistant", "原策略已生成。"),
    ]
    events = collect(
        AgentRunner(
            model, artifacts, functional_tester=functional_tester
        ),
        message="改为 3900",
        history=history,
        model="model-a",
        account_id="alice",
    )
    assert [item.role for item in model.seen[0]] == [
        "system", "system", "user", "assistant", "user"
    ]
    assert '"status": "matched"' in model.seen[0][1].content
    assert model.seen[0][-1].content == "改为 3900"
    assert artifacts.created == [("RiskStrategy.py", source, "alice")]
    assert events[-1].data["artifacts"][0]["filename"] == "RiskStrategy.py"
    assert functional_tester.calls[0]["requirements"] == [
        *history,
        ChatTurn("user", "改为 3900"),
    ]


def test_modified_source_must_pass_functional_test_again_before_saving() -> None:
    original = _strategy_source()
    revised = _strategy_source(body="    def onTick(self, tick):\n        pass\n")
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": original}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": revised}),))],
            [ModelChunk(content="已完成修改。")],
        ]
    )
    artifacts = MemoryArtifacts()
    tester = PassingFunctionalTester()

    events = collect(
        AgentRunner(model, artifacts, functional_tester=tester),
        message="生成并修正策略",
        history=[],
        model="model-a",
        account_id="alice",
    )

    assert [call["source"] for call in tester.calls] == [original, revised]
    assert artifacts.created == [("RiskStrategy.py", revised, "alice")]
    assert any(
        '"error": "保存前必须先通过同一份源码的 run_functional_test"'
        in item.content
        for item in model.seen[4]
        if isinstance(item, ToolResultMessage)
    )
    assert events[-1].data["artifacts"][0]["filename"] == "RiskStrategy.py"


def test_failed_functional_result_prevents_artifact_creation() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="当前版本尚未通过检测。")],
            [ModelChunk(content="本轮无法保存策略。")],
        ]
    )
    artifacts = MemoryArtifacts()

    with pytest.raises(AgentError, match="未保存"):
        collect(
            AgentRunner(
                model,
                artifacts,
                functional_tester=PassingFunctionalTester(passed=False),
            ),
            message="生成策略",
            history=[],
            model="model-a",
            account_id="alice",
        )

    assert artifacts.created == []
    assert "run_functional_test" in model.seen[3][-1].content


def test_unchanged_source_after_functional_failure_is_not_retested() -> None:
    source = _strategy_source()

    class FailingFunctionalTester:
        def __init__(self) -> None:
            self.sources: list[str] = []

        async def run(self, candidate, *, requirements, model, options):
            self.sources.append(candidate)
            return {
                "passed": False,
                "case_count": 1,
                "cases": [
                    {
                        "name": "第二个完整周期再次触发",
                        "passed": False,
                        "expected_warning_count": 2,
                        "actual_warning_count": 1,
                    }
                ],
                "failed_cases": [
                    {
                        "index": 1,
                        "name": "第二个完整周期再次触发",
                        "ticks": [
                            {"vtSymbol": "AP701", "lastPrice": 5976.0},
                            {"vtSymbol": "AP701", "lastPrice": 5969.0},
                        ],
                        "expected_warning_count": 2,
                        "actual_warning_count": 1,
                    }
                ],
            }

    tester = FailingFunctionalTester()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(content="仍然无法修复。")],
            [ModelChunk(content="仍然无法修复。")],
        ]
    )

    with pytest.raises(AgentError, match="未保存策略文件") as error:
        collect(
            AgentRunner(
                model,
                MemoryArtifacts(),
                functional_tester=tester,
            ),
            message="生成策略",
            history=[],
            model="model-a",
            account_id="alice",
        )

    assert tester.sources == [source]
    assert "第二个完整周期再次触发" in (error.value.failure_context or "")
    assert '"lastPrice": 5976.0' in (error.value.failure_context or "")
    assert any(
        isinstance(item, ToolResultMessage) and "failed_cases" in item.content
        for item in model.seen[2]
    )
    assert any(
        item.role == "user" and "逐笔按其中的 ticks" in item.content
        for item in model.seen[2]
    )


def test_changed_source_can_be_functionally_tested_after_a_failure() -> None:
    original = _strategy_source()
    revised = _strategy_source(body="    def onTick(self, tick):\n        pass\n")

    class FailThenPassFunctionalTester:
        def __init__(self) -> None:
            self.sources: list[str] = []

        async def run(self, candidate, *, requirements, model, options):
            self.sources.append(candidate)
            if len(self.sources) == 1:
                return {
                    "passed": False,
                    "case_count": 1,
                    "cases": [{"name": "case", "passed": False}],
                    "failed_cases": [
                        {
                            "index": 1,
                            "name": "case",
                            "ticks": [{"vtSymbol": "rb2710", "lastPrice": 3801.0}],
                            "expected_warning_count": 1,
                            "actual_warning_count": 0,
                        }
                    ],
                }
            return {"passed": True, "case_count": 1, "cases": [{"name": "case", "passed": True}]}

    tester = FailThenPassFunctionalTester()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": original}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": revised}),))],
            [ModelChunk(content="已完成修改。")],
        ]
    )
    artifacts = MemoryArtifacts()

    events = collect(
        AgentRunner(
            model,
            artifacts,
            functional_tester=tester,
        ),
        message="生成并修正策略",
        history=[],
        model="model-a",
        account_id="alice",
    )

    assert tester.sources == [original, revised]
    assert artifacts.created == [("RiskStrategy.py", revised, "alice")]
    assert events[-1].data["content"] == "已完成修改。"


def test_tool_round_limit() -> None:
    model = ScriptedModel([[ModelChunk(tool_calls=(ToolCall("list_skills", {}),))]] * 3)
    runner = AgentRunner(model, MemoryArtifacts(), max_tool_rounds=2)
    with pytest.raises(AgentError, match="上限"):
        collect(runner, message="生成", history=[], model="model-a", account_id="alice")


def test_concurrent_runs_keep_messages_and_artifacts_separate() -> None:
    class ConcurrentModel(ModelGateway):
        def __init__(self):
            self.rounds: dict[str, int] = {}

        async def list_models(self):
            return ["model-a"]

        async def stream_chat(self, model, messages, tools, *, options, think):
            if _is_contract_lookup_preflight(tools):
                yield ModelChunk()
                return
            user_message = next(item.content for item in messages if item.role == "user")
            await asyncio.sleep(0.001)
            round_index = self.rounds.get(user_message, 0)
            self.rounds[user_message] = round_index + 1
            source = _strategy_source(f"{user_message.upper()}Strategy")
            calls = [
                ToolCall("run_functional_test", {"source": source}),
                ToolCall("write_generated_file", {"source": source}),
            ]
            if round_index < len(calls):
                yield ModelChunk(tool_calls=(calls[round_index],))
            else:
                yield ModelChunk(content=f"已完成 {user_message}")

    class ConcurrentArtifacts:
        async def create(self, filename, source, account_id):
            await asyncio.sleep(0.001)
            return Artifact(f"{filename[0] * 32}", filename, account_id)

    runner = AgentRunner(
        ConcurrentModel(),
        ConcurrentArtifacts(),
        functional_tester=PassingFunctionalTester(),
    )

    async def one(label):
        return [event async for event in runner.run(
            message=label, history=[ChatTurn("assistant", f"只属于 {label}")],
            model="model-a", account_id=label
        )]

    first, second = asyncio.run(_both(one))
    assert first[-1].data["content"] == "已完成 a"
    assert second[-1].data["content"] == "已完成 b"
    assert first[-1].data["artifacts"] == [{"artifact_id": "A" * 32, "filename": "AStrategy.py"}]
    assert second[-1].data["artifacts"] == [{"artifact_id": "B" * 32, "filename": "BStrategy.py"}]


async def _both(one):
    return await asyncio.gather(one("a"), one("b"))


def test_cancelling_a_running_request_closes_the_model_stream() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        closed = asyncio.Event()

        class SlowModel(ModelGateway):
            async def list_models(self):
                return ["model-a"]

            async def stream_chat(self, model, messages, tools, *, options, think):
                if _is_contract_lookup_preflight(tools):
                    yield ModelChunk()
                    return
                try:
                    yield ModelChunk(content="partial")
                    entered.set()
                    await asyncio.sleep(10)
                finally:
                    closed.set()

        runner = AgentRunner(SlowModel(), MemoryArtifacts())
        stream = runner.run(message="生成", history=[], model="model-a", account_id="alice")
        assert (await anext(stream)).kind == "status"
        assert (await anext(stream)).kind == "delta"
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(entered.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(closed.wait(), 1)
        await stream.aclose()

    asyncio.run(scenario())
