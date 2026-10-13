"""Docker sandbox command boundaries and failure handling."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.functional_tests import (
    FunctionalTestCase,
    FunctionalTestError,
    TestTick as FunctionalTestTick,
)
from backend.adapters.sandbox import DockerFunctionalSandbox


def case() -> FunctionalTestCase:
    return FunctionalTestCase(
        name="below threshold",
        ticks=(FunctionalTestTick("rb2710", 3799.0),),
        expected_warning_count=0,
    )


def write_fake_docker(tmp_path: Path, *, mode: str = "success") -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "docker"
    capture = tmp_path / "docker-args.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "args = sys.argv[1:]\n"
        "if args and args[0] == 'kill':\n"
        "    raise SystemExit(0)\n"
        "with open(os.environ['DOCKER_CAPTURE'], 'w', encoding='utf-8') as handle:\n"
        "    json.dump(args, handle)\n"
        f"if {mode!r} == 'timeout':\n"
        "    time.sleep(10)\n"
        f"if {mode!r} == 'failure':\n"
        "    print('daemon unavailable', file=sys.stderr)\n"
        "    raise SystemExit(125)\n"
        "print(json.dumps({'passed': True, 'cases': [{'name': 'below threshold', 'passed': True, 'expected_warning_count': 0, 'actual_warning_count': 0}]}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir, capture


def make_sandbox(tmp_path: Path, *, timeout_seconds: int = 2) -> DockerFunctionalSandbox:
    interface_dir = tmp_path / "interfaces"
    interface_dir.mkdir()
    (interface_dir / "external_head.py").write_text("# interface head\n", encoding="utf-8")
    (interface_dir / "external_baseid.py").write_text("# ids\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return DockerFunctionalSandbox(
        interface_dir,
        scratch,
        image="test-python:3.12",
        timeout_seconds=timeout_seconds,
    )


def test_docker_sandbox_uses_isolated_read_only_limited_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir, capture = write_fake_docker(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_CAPTURE", str(capture))

    results = asyncio.run(
        make_sandbox(tmp_path).run("class RiskStrategy: pass\n", [case()])
    )

    args = json.loads(capture.read_text(encoding="utf-8"))
    assert results[0]["passed"] is True
    assert args[0] == "run"
    assert "--pull=never" in args
    assert args[args.index("--network") + 1] == "none"
    assert "--read-only" in args
    assert "--cap-drop" in args and args[args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges:true" in args
    assert args[args.index("--memory") + 1] == "256m"
    assert args[args.index("--memory-swap") + 1] == "256m"
    assert args[args.index("--cpus") + 1] == "1"
    assert args[args.index("--pids-limit") + 1] == "32"
    mount = args[args.index("--mount") + 1]
    assert "target=/work,readonly" in mount
    assert args[-4:] == ["python", "-I", "-B", "/work/sandbox_harness.py"]


def test_docker_sandbox_reports_daemon_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir, capture = write_fake_docker(tmp_path, mode="failure")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_CAPTURE", str(capture))

    with pytest.raises(FunctionalTestError, match="daemon unavailable"):
        asyncio.run(make_sandbox(tmp_path).run("class RiskStrategy: pass\n", [case()]))


def test_docker_sandbox_stops_timed_out_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir, capture = write_fake_docker(tmp_path, mode="timeout")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_CAPTURE", str(capture))

    with pytest.raises(FunctionalTestError, match="超过 1 秒时限"):
        asyncio.run(make_sandbox(tmp_path, timeout_seconds=1).run(
            "class RiskStrategy: pass\n", [case()]
        ))


def test_real_python312_docker_harness_when_image_and_daemon_are_available(
    tmp_path: Path,
) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")
    daemon = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=3, check=False
    )
    image = subprocess.run(
        ["docker", "image", "inspect", "python:3.12-slim"],
        capture_output=True,
        timeout=3,
        check=False,
    )
    if daemon.returncode != 0 or image.returncode != 0:
        pytest.skip("Docker daemon or python:3.12-slim image is unavailable")

    interface_dir = Path(__file__).parents[1] / "interfaces" / "价格接口"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = '''from external_head import RHTemplate, VtTickData

class ExampleStrategy(RHTemplate):
    def __init__(self, pid):
        self.symbolList = ["rb2710"]
        self.exchangeList = ["SHFE"]
        self.npid = pid
        self.alerted = False
        super().__init__()
        self.vtSymbol = "rb2710"
        self.exchange = "SHFE"
        self.instrument = self.vtSymbol

    def onTick(self, tick):
        if tick.vtSymbol == "rb2710" and tick.lastPrice > 3800 and not self.alerted:
            self.alerted = True
            self.putEvent("above", "warning")
'''
    cases = [
        FunctionalTestCase(
            "below threshold",
            (FunctionalTestTick("rb2710", 3799.0),),
            0,
        ),
        FunctionalTestCase(
            "above threshold",
            (FunctionalTestTick("rb2710", 3801.0),),
            1,
        ),
    ]

    results = asyncio.run(DockerFunctionalSandbox(interface_dir, scratch).run(source, cases))

    assert [result["passed"] for result in results] == [True, True]
