"""Agent behavior with deliberately small, deterministic adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent.contracts import Artifact, ChatTurn, ModelChunk, ToolCall
from agent.prompt import PROMPT_FILE, build_system_prompt
from agent.runner import AgentError, AgentRunner


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


class ScriptedModel:
    def __init__(self, rounds: list[list[ModelChunk]]) -> None:
        self.rounds = rounds
        self.seen: list[list[dict[str, Any]]] = []

    async def list_models(self) -> list[str]:
        return ["model-a"]

    async def stream_chat(self, model, messages, tools) -> AsyncIterator[ModelChunk]:
        assert model == "model-a"
        assert len(tools) == 3
        self.seen.append(messages.copy())
        for chunk in self.rounds[len(self.seen) - 1]:
            yield chunk


def collect(runner: AgentRunner, **kwargs):
    async def read():
        return [event async for event in runner.run(**kwargs)]

    return asyncio.run(read())


def test_preserved_prompt_is_prefixed_without_editing_original() -> None:
    original = PROMPT_FILE.read_text(encoding="utf-8").removesuffix("\n")
    assembled = build_system_prompt()
    assert assembled.startswith(original + "\n\n# 当前时间")
    assert "list_interface_files" in assembled
    assert "read_interface_file" in assembled
    assert "write_generated_file" in assembled


def test_read_write_reply_flow() -> None:
    source = "class My_TEST:\n    pass\n"
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("list_interface_files", {}),))],
            [ModelChunk(tool_calls=(ToolCall("read_interface_file", {"filename": "external_head.py"}),))],
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"filename": "risk.py", "source": source}),))],
            [ModelChunk(content="已生成。", usage={"prompt_tokens": 42})],
        ]
    )
    interfaces = MemoryInterfaces()
    artifacts = MemoryArtifacts()
    events = collect(
        AgentRunner(model, interfaces, artifacts),
        message="SHFE rb2710 超过 3800 时提醒",
        history=[],
        model="model-a",
        account_id="alice",
    )
    assert interfaces.reads == ["external_head.py"]
    assert artifacts.created == [("risk.py", source, "alice")]
    assert [event.kind for event in events if event.kind in {"artifact", "delta", "done"}] == [
        "artifact", "delta", "done"
    ]
    assert events[-1].data["artifacts"] == [{"artifact_id": "00000000000000000000000000000001", "filename": "risk.py"}]
    assert model.seen[1][-1]["role"] == "tool"
    assert "external_head.py" in model.seen[1][-1]["content"]
    assert model.seen[2][-1]["role"] == "tool"
    assert "RHTemplate" in model.seen[2][-1]["content"]


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
    model = ScriptedModel(
        [
            [ModelChunk(tool_calls=(ToolCall("write_generated_file", {"filename": "revision.py", "source": "x = 3900\n"}),))],
            [ModelChunk(content="已将阈值改为 3900。")],
        ]
    )
    artifacts = MemoryArtifacts()
    history = [
        ChatTurn("user", "SHFE rb2710 超过 3800 时提醒"),
        ChatTurn("assistant", "原策略已生成。"),
    ]
    events = collect(
        AgentRunner(model, MemoryInterfaces(), artifacts),
        message="改为 3900",
        history=history,
        model="model-a",
        account_id="alice",
    )
    assert [item["role"] for item in model.seen[0]] == ["system", "user", "assistant", "user"]
    assert model.seen[0][-1]["content"] == "改为 3900"
    assert artifacts.created == [("revision.py", "x = 3900\n", "alice")]
    assert events[-1].data["artifacts"][0]["filename"] == "revision.py"


def test_tool_round_limit() -> None:
    model = ScriptedModel([[ModelChunk(tool_calls=(ToolCall("list_interface_files", {}),))]] * 3)
    runner = AgentRunner(model, MemoryInterfaces(), MemoryArtifacts(), max_tool_rounds=2)
    with pytest.raises(AgentError, match="上限"):
        collect(runner, message="生成", history=[], model="model-a", account_id="alice")


def test_concurrent_runs_keep_messages_and_artifacts_separate() -> None:
    class ConcurrentModel:
        async def list_models(self):
            return ["model-a"]

        async def stream_chat(self, model, messages, tools):
            user_message = next(item["content"] for item in messages if item["role"] == "user")
            await asyncio.sleep(0.001)
            if messages[-1]["role"] == "user":
                yield ModelChunk(tool_calls=(ToolCall("write_generated_file", {
                    "filename": f"{user_message}.py", "source": f"label = '{user_message}'\n"
                }),))
            else:
                yield ModelChunk(content=f"已完成 {user_message}")

    class ConcurrentArtifacts:
        async def create(self, filename, source, account_id):
            await asyncio.sleep(0.001)
            return Artifact(f"{filename[0] * 32}", filename, account_id)

    runner = AgentRunner(ConcurrentModel(), MemoryInterfaces(), ConcurrentArtifacts())

    async def one(label):
        return [event async for event in runner.run(
            message=label, history=[ChatTurn("assistant", f"只属于 {label}")],
            model="model-a", account_id=label
        )]

    first, second = asyncio.run(_both(one))
    assert first[-1].data["content"] == "已完成 a"
    assert second[-1].data["content"] == "已完成 b"
    assert first[-1].data["artifacts"] == [{"artifact_id": "a" * 32, "filename": "a.py"}]
    assert second[-1].data["artifacts"] == [{"artifact_id": "b" * 32, "filename": "b.py"}]


async def _both(one):
    return await asyncio.gather(one("a"), one("b"))


def test_cancelling_a_running_request_closes_the_model_stream() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        closed = asyncio.Event()

        class SlowModel:
            async def list_models(self):
                return ["model-a"]

            async def stream_chat(self, model, messages, tools):
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
