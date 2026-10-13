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
    from external_head import RHTemplate, VtTickData
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
                if not hasattr(strategy, "symbolList"):
                    strategy.symbolList = list(
                        dict.fromkeys(tick["vtSymbol"] for tick in case["ticks"])
                    )
                if not hasattr(strategy, "exchangeList"):
                    strategy.exchangeList = [""] * len(strategy.symbolList)
                if not hasattr(strategy, "vtSymbol"):
                    strategy.vtSymbol = ";".join(strategy.symbolList)
                if not hasattr(strategy, "exchange"):
                    strategy.exchange = ";".join(strategy.exchangeList)
                if not hasattr(strategy, "instrument"):
                    strategy.instrument = strategy.vtSymbol
                strategy.onStart()
                try:
                    for tick_data in case["ticks"]:
                        tick = VtTickData()
                        tick.vtSymbol = tick_data["vtSymbol"]
                        tick.lastPrice = tick_data["lastPrice"]
                        strategy.onTick(tick)
                finally:
                    strategy.onStop()
            actual = _warning_count(PyEngine.MESSAGES)
            expected = case["expected_warning_count"]
            results.append(
                {
                    "name": case["name"],
                    "passed": actual == expected,
                    "expected_warning_count": expected,
                    "actual_warning_count": actual,
                }
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
