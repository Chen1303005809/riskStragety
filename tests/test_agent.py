"""Agent behavior with deliberately small, deterministic adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from agent.contracts import (
    AgentMessage,
    Artifact,
    AssistantMessage,
    ChatTurn,
    ModelChunk,
    ModelGateway,
    ToolCall,
    ToolResultMessage,
)
from agent.prompt import PROMPT_FILE, build_system_prompt
from agent.runner import AgentError, AgentRunner, check_python_syntax


class MemoryInterfaces:
    def __init__(self) -> None:
        self.files = {"external_head.py": "class RHTemplate: pass\n"}
        self.reads: list[str] = []

    async def list_files(self) -> list[str]:
        return list(self.files)

    async def read_file(self, filename: str) -> str:
        self.reads.append(filename)
        return self.files[filename]


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
        self.settings_seen: list[tuple[dict[str, int | float], bool]] = []

    async def list_models(self) -> list[str]:
        return ["model-a"]

    async def stream_chat(
        self, model, messages, tools, *, options, think
    ) -> AsyncIterator[ModelChunk]:
        assert model == "model-a"
        assert len(tools) == 6
        self.seen.append(messages.copy())
        self.settings_seen.append((options.copy(), think))
        for chunk in self.rounds[len(self.seen) - 1]:
            yield chunk


def collect(runner: AgentRunner, **kwargs):
    async def read():
        return [event async for event in runner.run(**kwargs)]

    return asyncio.run(read())


def _strategy_source(name: str = "RiskStrategy", body: str = "    pass\n") -> str:
    return f"class My_PythonObj(RHTemplate):\n    name = {name!r}\n{body}"


def test_current_time_is_a_volatile_suffix_after_the_stable_prompt() -> None:
    original = PROMPT_FILE.read_text(encoding="utf-8").removesuffix("\n")
    first = build_system_prompt(
        now=datetime(2026, 10, 16, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    second = build_system_prompt(
        now=datetime(2026, 10, 17, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    stable_first, time_first = first.rsplit("\n\n# 当前时间\n", 1)
    stable_second, time_second = second.rsplit("\n\n# 当前时间\n", 1)

    assert stable_first == stable_second
    assert stable_first.startswith(original + "\n\n# 工具执行规则")
    assert time_first != time_second
    assert first.endswith("代码文件头、元数据及所有输出中的日期必须使用该当天日期。")
    assert "list_interface_files" in stable_first
    assert "read_interface_file" in stable_first
    assert "check_python_syntax" in stable_first
    assert "run_functional_test" in stable_first
    assert "write_generated_file" in stable_first
    assert "lookup_exchange_by_contract_code" in stable_first
    assert "不要再要求用户提供合约代码" in stable_first


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
            [ModelChunk(tool_calls=(ToolCall("list_interface_files", {}),))],
            [ModelChunk(tool_calls=(ToolCall("read_interface_file", {"filename": "external_head.py"}),))],
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="已生成策略。")],
        ]
    )
    artifacts = MemoryArtifacts()

    events = collect(
        AgentRunner(model, MemoryInterfaces(), artifacts, functional_tester=PassingFunctionalTester()),
        message=f"{contract_code} 超过 3800 时提醒",
        history=[], model="model-a", account_id="alice",
    )

    lookup_result = model.seen[1][-1].content
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
            [ModelChunk(content=f"【参数补充请求】\n{expected_followup}")],
        ]
    )

    events = collect(
        AgentRunner(model, MemoryInterfaces(), MemoryArtifacts()),
        message=f"{contract_code} 超过 3800 时提醒",
        history=[], model="model-a", account_id="alice",
    )

    assert f'"status": "{expected_status}"' in model.seen[1][-1].content
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
            [ModelChunk(content="【参数补充请求】\n请提供有效合约代码。")],
        ]
    )

    events = collect(
        AgentRunner(model, MemoryInterfaces(), MemoryArtifacts()),
        message=f"{contract_code} 超过 3800 时提醒",
        history=[], model="model-a", account_id="alice",
    )

    assert '"status": "expired"' in model.seen[1][-1].content
    assert events[-1].kind == "done"
    assert "提供有效合约代码" in events[-1].data["content"]


def test_read_write_reply_flow() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("list_interface_files", {}),))],
            [ModelChunk(tool_calls=(ToolCall("read_interface_file", {"filename": "external_head.py"}),))],
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": source}),))],
            [ModelChunk(content="已生成。", usage={"prompt_tokens": 42})],
        ]
    )
    interfaces = MemoryInterfaces()
    artifacts = MemoryArtifacts()
    functional_tester = PassingFunctionalTester()
    events = collect(
        AgentRunner(model, interfaces, artifacts, functional_tester=functional_tester),
        message="SHFE rb2710 超过 3800 时提醒",
        history=[],
        model="model-a",
        account_id="alice",
        options={"temperature": 0.55, "top_p": 0.9, "num_ctx": 32_768},
        think=True,
    )
    assert interfaces.reads == ["external_head.py"]
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
    assert "external_head.py" in model.seen[1][-1].content
    assert isinstance(model.seen[3][-1], ToolResultMessage)
    assert '"passed": true' in model.seen[3][-1].content
    assert isinstance(model.seen[4][-1], ToolResultMessage)
    assert model.seen[3][-1].tool_name == "check_python_syntax"
    assert model.seen[4][-1].tool_name == "run_functional_test"
    assert isinstance(model.seen[5][-1], ToolResultMessage)
    assert model.seen[5][-1].tool_name == "write_generated_file"
    assert model.settings_seen == [
        ({"temperature": 0.55, "top_p": 0.9, "num_ctx": 32_768}, True)
    ] * 6


def test_runner_preserves_unique_provider_call_ids_and_pairs_results() -> None:
    model = ScriptedModel(
        [
            [
                ModelChunk(
                    tool_calls=(
                        ToolCall("list_interface_files", {}),
                        ToolCall("list_interface_files", {}, "agent_call_0"),
                        ToolCall("list_interface_files", {}, "provider-call-id"),
                        ToolCall("list_interface_files", {}, "provider-call-id"),
                    )
                )
            ],
            [ModelChunk(content="【参数补充请求】请补充策略条件。")],
        ]
    )
    runner = AgentRunner(model, MemoryInterfaces(), MemoryArtifacts())

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


def test_static_checker_uses_python_37_grammar() -> None:
    assert check_python_syntax("value = 1\n")["passed"] is True
    result = check_python_syntax("match value:\n    case _: pass\n")
    assert result["passed"] is False
    assert result["grammar"] == "Python 3.7"
    assert result["line"] in {1, 2}


def test_missing_parameters_ask_without_creating_file() -> None:
    model = ScriptedModel([[ModelChunk(content="【参数补充请求】\n请提供合约与阈值。")]])
    artifacts = MemoryArtifacts()
    events = collect(
        AgentRunner(model, MemoryInterfaces(), artifacts),
        message="螺纹钢突破时提醒",
        history=[],
        model="model-a",
        account_id="alice",
    )
    assert artifacts.created == []
    assert events[-1].kind == "done"
    assert "参数补充请求" in events[-1].data["content"]


def test_unpersisted_or_empty_reply_is_not_reported_as_success() -> None:
    for response in ("", "已生成策略，但没有保存文件。"):
        model = ScriptedModel([[ModelChunk(content=response)]])
        with pytest.raises(AgentError):
            collect(
                AgentRunner(model, MemoryInterfaces(), MemoryArtifacts()),
                message="生成", history=[], model="model-a", account_id="alice"
            )


def test_clarification_cannot_write_even_if_model_also_calls_write_tool() -> None:
    model = ScriptedModel([[ModelChunk(
        content="【参数补充请求】\n请提供阈值。",
        tool_calls=(ToolCall("write_generated_file", {
            "filename": "should-not-exist.py", "source": "x = 1\n"
        }),),
    )]])
    artifacts = MemoryArtifacts()
    events = collect(
        AgentRunner(model, MemoryInterfaces(), artifacts),
        message="螺纹钢突破时提醒", history=[], model="model-a", account_id="alice"
    )
    assert artifacts.created == []
    assert [event.kind for event in events] == ["status", "delta", "done"]


def test_followup_receives_prior_turns_and_can_create_revision() -> None:
    source = _strategy_source()
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": source}),))],
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
            model, MemoryInterfaces(), artifacts, functional_tester=functional_tester
        ),
        message="改为 3900",
        history=history,
        model="model-a",
        account_id="alice",
    )
    assert [item.role for item in model.seen[0]] == ["system", "user", "assistant", "user"]
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
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": original}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": original}),))],
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": revised}),))],
            [ModelChunk(content="已完成修改。")],
        ]
    )
    artifacts = MemoryArtifacts()
    tester = PassingFunctionalTester()

    events = collect(
        AgentRunner(model, MemoryInterfaces(), artifacts, functional_tester=tester),
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
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": source}),))],
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
                MemoryInterfaces(),
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
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": source}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": source}),))],
        ]
    )

    with pytest.raises(AgentError, match="源码与此前失败版本完全相同") as error:
        collect(
            AgentRunner(
                model,
                MemoryInterfaces(),
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
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": original}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": original}),))],
            [ModelChunk(tool_calls=(ToolCall("check_python_syntax", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("run_functional_test", {"source": revised}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"source": revised}),))],
            [ModelChunk(content="已完成修改。")],
        ]
    )
    artifacts = MemoryArtifacts()

    events = collect(
        AgentRunner(
            model,
            MemoryInterfaces(),
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
    model = ScriptedModel([[ModelChunk(tool_calls=(ToolCall("list_interface_files", {}),))]] * 3)
    runner = AgentRunner(model, MemoryInterfaces(), MemoryArtifacts(), max_tool_rounds=2)
    with pytest.raises(AgentError, match="上限"):
        collect(runner, message="生成", history=[], model="model-a", account_id="alice")


def test_concurrent_runs_keep_messages_and_artifacts_separate() -> None:
    class ConcurrentModel(ModelGateway):
        def __init__(self):
            self.rounds: dict[str, int] = {}

        async def list_models(self):
            return ["model-a"]

        async def stream_chat(self, model, messages, tools, *, options, think):
            user_message = next(item.content for item in messages if item.role == "user")
            await asyncio.sleep(0.001)
            round_index = self.rounds.get(user_message, 0)
            self.rounds[user_message] = round_index + 1
            source = _strategy_source(f"{user_message.upper()}Strategy")
            calls = [
                ToolCall("check_python_syntax", {"source": source}),
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
        MemoryInterfaces(),
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
                try:
                    yield ModelChunk(content="partial")
                    entered.set()
                    await asyncio.sleep(10)
                finally:
                    closed.set()

        runner = AgentRunner(SlowModel(), MemoryInterfaces(), MemoryArtifacts())
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
