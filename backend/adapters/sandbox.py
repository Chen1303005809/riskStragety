"""Docker-backed execution of generated strategies in a constrained container."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent.functional_tests import FunctionalTestCase, FunctionalTestError


DOCKER_IMAGE = "python:3.12-slim"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_INTERFACE_BYTES = 512 * 1024
MAX_CASES_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
SANDBOX_TIMEOUT_SECONDS = 12
MAX_CONCURRENT_SANDBOXES = 2

_PYENGINE_SHIM = """MESSAGES = []

class PyEngine:
    def sendMsg(self, message, _extra=""):
        MESSAGES.append(message)
"""

_RHCLIENT_SHIM = """class RHclient:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None
        return no_op
"""


class DockerFunctionalSandbox:
    """Run each supplied scenario in one isolated, short-lived container."""

    def __init__(
        self,
        interface_directory: Path,
        scratch_root: Path,
        *,
        image: str = DOCKER_IMAGE,
        timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS,
        max_concurrent: int = MAX_CONCURRENT_SANDBOXES,
    ) -> None:
        if timeout_seconds < 1 or max_concurrent < 1:
            raise ValueError("sandbox limits must be positive")
        self._interface_directory = interface_directory
        self._scratch_root = scratch_root
        self._image = image
        self._timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run(
        self, source: str, cases: Sequence[FunctionalTestCase]
    ) -> list[dict[str, Any]]:
        if not source.strip():
            raise FunctionalTestError("策略源码不能为空")
        source_bytes = source.encode("utf-8")
        if len(source_bytes) > MAX_SOURCE_BYTES:
            raise FunctionalTestError("策略源码超过 1 MiB")
        if not cases:
            raise FunctionalTestError("没有可执行的测试用例")

        async with self._semaphore:
            with self._stage(source_bytes, cases) as stage_directory:
                payload = await self._run_container(Path(stage_directory))
        if not payload.get("passed"):
            return payload.get("cases", [])
        results = payload.get("cases")
        if not isinstance(results, list) or len(results) != len(cases):
            raise FunctionalTestError("沙盒返回了无效的测试结果")
        return results

    def _stage(self, source: bytes, cases: Sequence[FunctionalTestCase]):
        root = self._scratch_root
        if root.is_symlink() or not root.is_dir():
            raise FunctionalTestError("沙盒暂存目录不可用")
        interface_root = self._interface_directory
        if interface_root.is_symlink() or not interface_root.is_dir():
            raise FunctionalTestError("价格接口目录不可用")

        head = self._read_interface(interface_root / "external_head.py", interface_root)
        baseid = self._read_interface(interface_root / "external_baseid.py", interface_root)
        cases_json = json.dumps(
            [case.as_payload() for case in cases], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(cases_json) > MAX_CASES_BYTES:
            raise FunctionalTestError("测试用例数据超过暂存大小限制")

        try:
            context = tempfile.TemporaryDirectory(prefix="agent-functional-", dir=root)
            stage = Path(context.name)
            (stage / "strategy.py").write_bytes(source)
            (stage / "cases.json").write_bytes(cases_json)
            (stage / "external_head.py").write_bytes(
                head + b"\nfrom RHclient import RHclient\n"
            )
            (stage / "external_baseid.py").write_bytes(baseid)
            (stage / "PyEngine.py").write_text(_PYENGINE_SHIM, encoding="utf-8")
            (stage / "RHclient.py").write_text(_RHCLIENT_SHIM, encoding="utf-8")
            harness = Path(__file__).with_name("sandbox_harness.py")
            shutil.copyfile(harness, stage / "sandbox_harness.py")
            return context
        except OSError as exc:
            if "context" in locals():
                context.cleanup()
            raise FunctionalTestError(f"无法准备沙盒上下文：{exc}") from exc

    @staticmethod
    def _read_interface(path: Path, root: Path) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise FunctionalTestError(f"接口文件不存在或不可用：{path.name}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root.resolve())
            with resolved.open("rb") as handle:
                content = handle.read(MAX_INTERFACE_BYTES + 1)
        except (OSError, ValueError) as exc:
            raise FunctionalTestError(f"无法读取接口文件：{path.name}") from exc
        if len(content) > MAX_INTERFACE_BYTES:
            raise FunctionalTestError(f"接口文件超过 512 KiB：{path.name}")
        return content

    async def _run_container(self, stage: Path) -> dict[str, Any]:
        container_name = f"agent-test-{uuid.uuid4().hex}"
        uid = getattr(os, "getuid", lambda: 65534)()
        gid = getattr(os, "getgid", lambda: 65534)()
        command = [
            "docker",
            "run",
            "--pull=never",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=16m",
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "32",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--user",
            f"{uid}:{gid}",
            "--mount",
            f"type=bind,source={stage},target=/work,readonly",
            "--workdir",
            "/work",
            self._image,
            "python",
            "-I",
            "-B",
            "/work/sandbox_harness.py",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise FunctionalTestError(f"无法启动 Docker 沙盒：{exc}") from exc

        stdout_task = asyncio.create_task(_read_bounded(process.stdout, MAX_OUTPUT_BYTES))
        stderr_task = asyncio.create_task(_read_bounded(process.stderr, MAX_OUTPUT_BYTES))
        try:
            await asyncio.wait_for(process.wait(), timeout=self._timeout_seconds)
        except asyncio.TimeoutError as exc:
            await self._stop_container(container_name, process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise FunctionalTestError(
                f"功能检测超过 {self._timeout_seconds} 秒时限"
            ) from exc
        except asyncio.CancelledError:
            await self._stop_container(container_name, process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise

        stdout_result, stderr_result = await asyncio.gather(stdout_task, stderr_task)
        stdout, stdout_truncated = stdout_result
        stderr, stderr_truncated = stderr_result
        if stdout_truncated or stderr_truncated:
            raise FunctionalTestError("沙盒输出超过 64 KiB 限制")
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            detail = stderr_text or stdout.decode("utf-8", errors="replace").strip()
            raise FunctionalTestError(f"Docker 沙盒执行失败：{detail[:2_000]}")
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FunctionalTestError("沙盒未返回有效的 JSON 测试结果") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("passed"), bool):
            raise FunctionalTestError("沙盒返回了无效的测试结果")
        if not isinstance(payload.get("cases"), list):
            raise FunctionalTestError("沙盒结果缺少 cases")
        if payload.get("error"):
            raise FunctionalTestError(f"策略加载失败：{payload['error']}")
        return payload

    async def _stop_container(self, name: str, process: asyncio.subprocess.Process) -> None:
        try:
            killer = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=2)
        except (OSError, asyncio.TimeoutError):
            pass
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


async def _read_bounded(
    stream: asyncio.StreamReader | None, limit: int
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    parts: list[bytes] = []
    kept = 0
    truncated = False
    while True:
        chunk = await stream.read(8 * 1024)
        if not chunk:
            break
        remaining = limit - kept
        if remaining > 0:
            kept_chunk = chunk[:remaining]
            parts.append(kept_chunk)
            kept += len(kept_chunk)
        if len(chunk) > remaining:
            truncated = True
    return b"".join(parts), truncated
