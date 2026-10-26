"""Trusted harness copied into the isolated functional-test container."""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _warning_count(messages: list[Any]) -> int:
    from external_baseid import PyToSpiNotification

    count = 0
    for message in messages:
        try:
            payload = json.loads(message) if isinstance(message, str) else message
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("MsgType") == PyToSpiNotification
            and payload.get("level") == "warning"
        ):
            count += 1
    return count


def run() -> dict[str, Any]:
    workdir = Path(__file__).resolve().parent
    sys.path.insert(0, str(workdir))
    from external_head import RHTemplate, VtBarData, VtTickData
    import PyEngine

    spec = importlib.util.spec_from_file_location("candidate_strategy", workdir / "strategy.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载候选策略")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategies = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and value is not RHTemplate
        and value.__module__ == module.__name__
        and any(base is RHTemplate for base in value.__bases__)
    ]
    if len(strategies) != 1:
        raise RuntimeError("策略必须且只能包含一个直接继承 RHTemplate 的顶层类")
    strategy_class = strategies[0]
    cases = json.loads((workdir / "cases.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for case in cases:
        PyEngine.MESSAGES[:] = []
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    inspect.signature(strategy_class).bind(1)
                    accepts_pid = True
                except (TypeError, ValueError):
                    accepts_pid = False
                # 生成策略通常显式接收 pid；RHTemplate 本身无参。
                strategy = strategy_class(1) if accepts_pid else strategy_class()
                if not hasattr(strategy, "npid"):
                    strategy.npid = 1
                symbols = _case_symbols(case)
                if not hasattr(strategy, "symbolList"):
                    strategy.symbolList = symbols
                if not hasattr(strategy, "exchangeList"):
                    strategy.exchangeList = [""] * len(strategy.symbolList)
                if not hasattr(strategy, "vtSymbol"):
                    strategy.vtSymbol = ";".join(strategy.symbolList)
                if not hasattr(strategy, "exchange"):
                    strategy.exchange = ";".join(strategy.exchangeList)
                if not hasattr(strategy, "instrument"):
                    strategy.instrument = strategy.vtSymbol
                try:
                    strategy.onStart()
                    if case.get("events"):
                        _deliver_events(strategy, case["events"], VtTickData, VtBarData)
                    else:
                        for tick_data in case["ticks"]:
                            tick = VtTickData()
                            tick.vtSymbol = tick_data["vtSymbol"]
                            tick.lastPrice = tick_data["lastPrice"]
                            strategy.onTick(tick)
                finally:
                    strategy.onStop()
            actual = _warning_count(PyEngine.MESSAGES)
            expected = case["expected_warning_count"]
            call_state = getattr(strategy, "_sandbox_kline_state", None)
            actual_calls = _kline_call_counts(call_state)
            expected_calls = case.get("expected_calls")
            calls_match = expected_calls is None or all(
                actual_calls.get(name, 0) == count
                for name, count in expected_calls.items()
            )
            result = {
                "name": case["name"],
                "passed": actual == expected and calls_match,
                "expected_warning_count": expected,
                "actual_warning_count": actual,
            }
            if case.get("events"):
                result["actual_calls"] = actual_calls
            if expected_calls is not None:
                result["expected_calls"] = expected_calls
            results.append(
                result
            )
        except Exception as exc:
            results.append(
                {
                    "name": case.get("name", "unnamed"),
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"passed": bool(results) and all(case["passed"] for case in results), "cases": results}


def _case_symbols(case: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for tick in case.get("ticks", []):
        symbols.append(tick["vtSymbol"])
    for event in case.get("events", []):
        if event["type"] == "tick":
            symbols.append(event["vtSymbol"])
        elif event["type"] == "kline":
            symbols.append(event["bar"]["vtSymbol"])
        elif event["type"] == "history_response":
            symbols.extend(bar["vtSymbol"] for bar in event["bars"])
    return list(dict.fromkeys(symbols))


def _make_bar(payload: dict[str, Any], bar_class: type) -> Any:
    bar = bar_class()
    for field, value in payload.items():
        setattr(bar, field, value)
    return bar


def _deliver_events(
    strategy: Any,
    events: list[dict[str, Any]],
    tick_class: type,
    bar_class: type,
) -> None:
    delivered_history_ids: set[int] = set()
    for event in events:
        if event["type"] == "tick":
            tick = tick_class()
            tick.vtSymbol = event["vtSymbol"]
            tick.lastPrice = event["lastPrice"]
            strategy.onTick(tick)
        elif event["type"] == "history_response":
            state = getattr(strategy, "_sandbox_kline_state", None)
            query_ids = {
                query["request_id"] for query in (state or {}).get("queries", [])
            }
            if event["request_id"] not in query_ids:
                raise RuntimeError(
                    "history_response.request_id 没有对应的已提交查询"
                )
            if event["request_id"] in delivered_history_ids:
                raise RuntimeError("一个历史查询只能注入一次完成回调")
            bars = [_make_bar(payload, bar_class) for payload in event["bars"]]
            strategy.onHistoryKline(event["request_id"], bars, event["error"])
            delivered_history_ids.add(event["request_id"])
        elif event["type"] == "kline":
            bar = _make_bar(event["bar"], bar_class)
            state = getattr(strategy, "_sandbox_kline_state", None) or {}
            matching_subscription = any(
                subscription[1] == bar.vtSymbol
                and subscription[2] == bar.cycleType
                and subscription[3] == bar.exchange
                and subscription[5] == bar.globalId
                for subscription in state.get("active_subscriptions", set())
            )
            if matching_subscription:
                strategy.onKline(bar)
        else:
            raise RuntimeError("不支持的测试事件：{}".format(event.get("type")))


def _kline_call_counts(state: dict[str, Any] | None) -> dict[str, int]:
    state = state or {}
    return {
        "queryHistoryKline": len(state.get("queries", [])),
        "subscribeKline": len(state.get("subscriptions", [])),
        "unsubscribeKline": len(state.get("unsubscriptions", [])),
    }


if __name__ == "__main__":
    try:
        result = run()
    except Exception as exc:
        result = {
            "passed": False,
            "cases": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
