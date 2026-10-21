"""HTTP routes, authentication, and SSE conversion around AgentRunner."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator
from contextlib import asynccontextmanager
from urllib.parse import quote
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent.contracts import AgentEvent, ChatTurn, FunctionalTester, ModelGateway
from agent.functional_tests import SkillFunctionalTester, SkillTestCaseBuilder
from agent.runner import AgentRunner
from agent.skills import FilesystemSkillLoader
from backend.adapters.conversation_logs import ConversationLogStore
from backend.adapters.files import IndexedArtifactWriter, LocalArtifactStore, LocalInterfaceReader
from backend.adapters.provider_settings import ProviderSettingsStore
from backend.adapters.monitoring import (
    AccountNotFound,
    ConversationNotFound,
    MonitoringStore,
)
from backend.adapters.sandbox import DockerFunctionalSandbox
from backend.adapters.deepseek import DeepSeekModelGateway
from backend.adapters.ollama import OllamaModelGateway
from backend.config import Settings
from backend.schemas import ChatRequest, DeepSeekKeyRequest, DownloadRequest
from backend.security import Principal, require_principal


ARTIFACT_STREAM_CHUNK_SIZE = 64 * 1024
logger = logging.getLogger(__name__)

ERROR_MESSAGES = {
    "ok": "",
    "file_not_generated": "本轮已完成，但未生成策略文件。",
    "auth_invalid": "API Key 无效",
    "request_invalid": "请求参数无效",
    "provider_not_configured": "请先配置 DeepSeek API Key",
    "provider_key_invalid": "DeepSeek API Key 验证失败，请检查 Key 或网络连接",
    "conversation_not_found": "会话不存在",
    "account_not_found": "账号不存在",
    "artifact_not_found": "产物不存在",
    "model_service_unavailable": "模型服务暂不可用，请稍后重试",
    "agent_incomplete": "本轮生成未正常结束，请重试",
    "agent_failed": "本轮处理失败，请稍后重试",
    "resource_not_found": "请求的资源不存在",
    "method_not_allowed": "请求方法不受支持",
    "internal_error": "服务内部错误，请稍后重试",
}


class MappedHTTPException(HTTPException):
    """An HTTP failure with a stable, client-safe error code."""

    def __init__(self, status_code: int, err_code: str) -> None:
        if err_code not in ERROR_MESSAGES:
            raise ValueError(f"unknown client error code: {err_code}")
        super().__init__(status_code=status_code, detail=ERROR_MESSAGES[err_code])
        self.err_code = err_code


def _error_response(
    status_code: int,
    err_code: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    message = ERROR_MESSAGES[err_code]
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={"detail": message, "ErrCode": err_code, "ErrMsg": message},
    )


def _framework_http_error_code(status_code: int) -> str:
    return {
        401: "auth_invalid",
        404: "resource_not_found",
        405: "method_not_allowed",
        422: "request_invalid",
        502: "model_service_unavailable",
    }.get(status_code, "internal_error" if status_code >= 500 else "request_invalid")


def _artifact_chunks(content: bytes) -> Iterator[bytes]:
    for start in range(0, len(content), ARTIFACT_STREAM_CHUNK_SIZE):
        yield content[start : start + ARTIFACT_STREAM_CHUNK_SIZE]


def _sse(event: AgentEvent, *, conversation_id: str) -> bytes:
    payload = {**event.data, "conversation_id": conversation_id}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.kind}\ndata: {data}\n\n".encode("utf-8")


def _attachment_header(filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9_.-]", "_", ascii_name).strip("._")
    if not ascii_name.lower().endswith(".py"):
        ascii_name = f"{ascii_name or 'strategy'}.py"
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\x27\x27{quote(filename, safe="")}'


class _ConversationLockPool:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entries: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, key: str):
        async with self._guard:
            entry = self._entries.get(key)
            lock, users = entry if entry else (asyncio.Lock(), 0)
            self._entries[key] = (lock, users + 1)
        acquired = False
        try:
            await lock.acquire()
            acquired = True
            yield
        finally:
            async with self._guard:
                current, users = self._entries[key]
                if acquired:
                    lock.release()
                if users == 1:
                    del self._entries[key]
                else:
                    self._entries[key] = (current, users - 1)


def create_app(
    settings: Settings | None = None,
    *,
    model_gateway: ModelGateway | None = None,
    artifact_store: LocalArtifactStore | None = None,
    agent_runner: AgentRunner | None = None,
    functional_tester: FunctionalTester | None = None,
) -> FastAPI:
    config = settings or Settings.from_env()
    model = model_gateway or OllamaModelGateway(config.ollama_base_url)
    artifacts = artifact_store or LocalArtifactStore(config.project_root / "artifacts")
    monitoring = MonitoringStore(config.data_dir / "monitoring.sqlite3", artifacts.directory)
    conversation_logs = ConversationLogStore(config.data_dir / "conversation_logs")
    provider_settings = ProviderSettingsStore(config.data_dir / "provider_settings.sqlite3")
    interface_directory = config.project_root / "interfaces" / "价格接口"
    interfaces = LocalInterfaceReader(interface_directory)
    artifact_writer = IndexedArtifactWriter(artifacts, monitoring)
    sandbox = DockerFunctionalSandbox(interface_directory, config.project_root)

    def build_runner(gateway: ModelGateway) -> AgentRunner:
        selected_tester = functional_tester or SkillFunctionalTester(
            SkillTestCaseBuilder(gateway, FilesystemSkillLoader()), sandbox
        )
        return AgentRunner(
            gateway,
            interfaces,
            artifact_writer,
            functional_tester=selected_tester,
        )

    runner = agent_runner or build_runner(model)
    conversation_locks = _ConversationLockPool()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(model, "close", None)
        if close is not None:
            await close()

    app = FastAPI(title="风控策略 Agent", lifespan=lifespan)
    app.state.settings = config
    app.state.monitoring = monitoring
    app.state.conversation_logs = conversation_logs
    app.state.provider_settings = provider_settings

    @app.exception_handler(MappedHTTPException)
    async def mapped_http_exception_handler(
        _: Request, exc: MappedHTTPException
    ) -> JSONResponse:
        return _error_response(exc.status_code, exc.err_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(422, "request_invalid")

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return _error_response(
            exc.status_code,
            _framework_http_error_code(exc.status_code),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(_: Request, __: Exception) -> JSONResponse:
        return _error_response(500, "internal_error")

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.get("/api/providers")
    async def providers(_: Principal = Depends(require_principal)) -> dict[str, object]:
        return {
            "providers": [
                {"id": "ollama", "name": "Ollama", "configured": True},
                {
                    "id": "deepseek",
                    "name": "DeepSeek",
                    "configured": provider_settings.get_deepseek_api_key() is not None,
                },
            ]
        }

    @app.put("/api/providers/deepseek")
    async def configure_deepseek(
        payload: DeepSeekKeyRequest, _: Principal = Depends(require_principal)
    ) -> dict[str, object]:
        gateway = DeepSeekModelGateway(payload.api_key)
        try:
            names = await gateway.list_models()
        except Exception as exc:
            raise MappedHTTPException(400, "provider_key_invalid") from exc
        finally:
            await gateway.close()
        provider_settings.set_deepseek_api_key(payload.api_key)
        return {"provider": "deepseek", "configured": True, "models": names}

    @app.delete("/api/providers/deepseek")
    async def remove_deepseek(_: Principal = Depends(require_principal)) -> dict[str, object]:
        provider_settings.delete_deepseek_api_key()
        return {"provider": "deepseek", "configured": False}

    @app.get("/api/models")
    async def models(
        provider: str = Query(default="ollama", pattern="^(ollama|deepseek)$"),
        _: Principal = Depends(require_principal),
    ) -> dict[str, object]:
        if provider == "deepseek":
            api_key = provider_settings.get_deepseek_api_key()
            if api_key is None:
                raise MappedHTTPException(409, "provider_not_configured")
            gateway = DeepSeekModelGateway(api_key)
            try:
                names = await gateway.list_models()
                default_model = "deepseek-flash" if "deepseek-flash" in names else (names[0] if names else "")
                return {
                    "models": names,
                    "default_model": default_model,
                    "deepseek_configured": True,
                }
            except Exception as exc:
                raise MappedHTTPException(502, "model_service_unavailable") from exc
            finally:
                await gateway.close()
        try:
            return {
                "models": await model.list_models(),
                "default_model": config.ollama_model,
                "deepseek_configured": provider_settings.get_deepseek_api_key() is not None,
            }
        except Exception as exc:
            raise MappedHTTPException(502, "model_service_unavailable") from exc

    @app.post("/api/agent/chat/stream")
    async def chat(
        payload: ChatRequest, principal: Principal = Depends(require_principal)
    ) -> StreamingResponse:
        selected_model = payload.model or (
            config.ollama_model if payload.provider == "ollama" else "deepseek-flash"
        )
        deepseek_api_key = None
        if payload.provider == "deepseek":
            deepseek_api_key = provider_settings.get_deepseek_api_key()
            if deepseek_api_key is None:
                raise MappedHTTPException(409, "provider_not_configured")
        conversation_id = str(payload.conversation_id)
        try:
            await monitoring.prepare_conversation(
                payload.account_id, conversation_id, payload.message
            )
        except ConversationNotFound as exc:
            raise MappedHTTPException(404, "conversation_not_found") from exc

        async def stream():
            provider_gateway = (
                DeepSeekModelGateway(deepseek_api_key)
                if deepseek_api_key is not None
                else None
            )
            selected_runner = (
                build_runner(provider_gateway)
                if provider_gateway is not None and agent_runner is None
                else runner
            )
            partial_content: list[str] = []
            assistant_message_id: int | None = None
            turn_finalized = False

            async def finalize_failed_turn(
                content: str,
                *,
                status: str = "failed",
                details: dict[str, object] | None = None,
            ) -> None:
                nonlocal turn_finalized
                if assistant_message_id is None or turn_finalized:
                    return
                try:
                    await conversation_logs.record(
                        conversation_id,
                        str(assistant_message_id),
                        "turn_finished",
                        {"status": status, "content": content, **(details or {})},
                        flush=True,
                    )
                except Exception:
                    logger.exception(
                        "conversation_log_finish_failed conversation_id=%s turn_id=%s",
                        conversation_id,
                        assistant_message_id,
                    )
                try:
                    await monitoring.finish_turn(
                        conversation_id, assistant_message_id, content, status
                    )
                except Exception:
                    logger.exception(
                        "agent_turn_persistence_failed conversation_id=%s status=%s",
                        conversation_id,
                        status,
                    )
                turn_finalized = True

            try:
                logger.info(
                    "agent_request_started conversation_id=%s provider=%s model=%s",
                    conversation_id,
                    payload.provider,
                    selected_model,
                )
                async with conversation_locks.hold(conversation_id):
                    history_rows, _, assistant_message_id = await monitoring.begin_turn(
                        payload.account_id, conversation_id, payload.message
                    )
                    history = [ChatTurn(role, content) for role, content in history_rows]
                    await conversation_logs.record(
                        conversation_id,
                        str(assistant_message_id),
                        "turn_started",
                        {
                            "account_id": payload.account_id,
                            "provider": payload.provider,
                            "model": selected_model,
                            "options": payload.options.model_dump(),
                            "think": payload.think,
                            "history": [
                                {"role": turn.role, "content": turn.content}
                                for turn in history
                            ],
                            "user_message": payload.message,
                        },
                    )
                    async for event in selected_runner.run(
                        message=payload.message,
                        history=history,
                        model=selected_model,
                        account_id=payload.account_id,
                        conversation_id=conversation_id,
                        turn_id=str(assistant_message_id),
                        conversation_recorder=conversation_logs,
                        options=payload.options.model_dump(),
                        think=payload.think,
                    ):
                        if event.kind == "delta":
                            text = event.data.get("text")
                            if isinstance(text, str):
                                partial_content.append(text)
                        elif event.kind == "done":
                            content = event.data.get("content")
                            if isinstance(content, str):
                                partial_content = [content]
                            artifacts_data = event.data.get("artifacts")
                            has_artifacts = isinstance(artifacts_data, list) and bool(artifacts_data)
                            err_code = "ok" if has_artifacts else "file_not_generated"
                            err_msg = ERROR_MESSAGES[err_code]
                            done_event = AgentEvent(
                                "done",
                                {
                                    **event.data,
                                    "ErrCode": err_code,
                                    "ErrMsg": err_msg,
                                },
                            )
                            await conversation_logs.record(
                                conversation_id,
                                str(assistant_message_id),
                                "turn_finished",
                                {
                                    "status": "complete",
                                    "content": "".join(partial_content),
                                    "model": selected_model,
                                    "artifacts": artifacts_data or [],
                                    "usage": event.data.get("usage", {}),
                                    "ErrCode": err_code,
                                    "ErrMsg": err_msg,
                                },
                                flush=True,
                            )
                            await monitoring.finish_turn(
                                conversation_id,
                                assistant_message_id,
                                "".join(partial_content),
                                "complete",
                            )
                            turn_finalized = True
                            yield _sse(done_event, conversation_id=conversation_id)
                            return
                        if event.kind == "error":
                            logger.error(
                                "agent_error_event conversation_id=%s provider=%s model=%s "
                                "partial_chars=%d event_data=%s",
                                conversation_id,
                                payload.provider,
                                selected_model,
                                len("".join(partial_content)),
                                json.dumps(event.data, ensure_ascii=False, default=str)[:2_000],
                            )
                            await finalize_failed_turn(
                                "".join(partial_content),
                                details={"error": event.data},
                            )
                            yield _sse(
                                AgentEvent(
                                    "error",
                                    {
                                        "ErrCode": "agent_failed",
                                        "ErrMsg": ERROR_MESSAGES["agent_failed"],
                                    },
                                ),
                                conversation_id=conversation_id,
                            )
                            return
                        yield _sse(event, conversation_id=conversation_id)

                    logger.warning(
                        "agent_stream_incomplete conversation_id=%s provider=%s model=%s "
                        "partial_chars=%d",
                        conversation_id,
                        payload.provider,
                        selected_model,
                        len("".join(partial_content)),
                    )
                    await finalize_failed_turn(
                        "".join(partial_content),
                        details={"error": "agent_incomplete"},
                    )
                    yield _sse(
                        AgentEvent(
                            "error",
                            {
                                "ErrCode": "agent_incomplete",
                                "ErrMsg": ERROR_MESSAGES["agent_incomplete"],
                            },
                        ),
                        conversation_id=conversation_id,
                    )
            except asyncio.CancelledError:
                logger.info(
                    "agent_request_interrupted conversation_id=%s provider=%s model=%s",
                    conversation_id,
                    payload.provider,
                    selected_model,
                )
                if assistant_message_id is not None and not turn_finalized:
                    await finalize_failed_turn(
                        "".join(partial_content),
                        status="interrupted",
                        details={"error": "request_cancelled"},
                    )
                raise
            except Exception as exc:
                exception_detail = str(exc).strip() or (
                    f"{type(exc).__name__}（异常消息为空）"
                )
                logger.exception(
                    "agent_request_failed conversation_id=%s provider=%s model=%s "
                    "assistant_message_id=%s exception_type=%s detail=%s",
                    conversation_id,
                    payload.provider,
                    selected_model,
                    assistant_message_id,
                    type(exc).__name__,
                    exception_detail[:2_000],
                )
                if assistant_message_id is not None and not turn_finalized:
                    failure_context = getattr(exc, "failure_context", None)
                    if not isinstance(failure_context, str) or not failure_context.strip():
                        failure_context = (
                            f"【上轮失败诊断】\n本轮失败：{exception_detail}"
                        )
                    elif failure_context.rstrip().endswith("本轮失败："):
                        failure_context = f"{failure_context}{exception_detail}"
                    failed_content = "".join(partial_content)
                    if failed_content.strip():
                        failed_content = f"{failed_content}\n\n{failure_context}"
                    else:
                        failed_content = failure_context
                    await finalize_failed_turn(
                        failed_content,
                        details={
                            "error": exception_detail,
                            "exception_type": type(exc).__name__,
                            "failure_context": failure_context,
                        },
                    )
                yield _sse(
                    AgentEvent(
                        "error",
                        {
                            "ErrCode": "agent_failed",
                            "ErrMsg": ERROR_MESSAGES["agent_failed"],
                        },
                    ),
                    conversation_id=conversation_id,
                )
            finally:
                if provider_gateway is not None:
                    await provider_gateway.close()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/artifacts/download")
    async def download(
        payload: DownloadRequest, principal: Principal = Depends(require_principal)
    ) -> StreamingResponse:
        artifact_record = await monitoring.artifact_for_account(
            payload.artifact_id, payload.account_id
        )
        if artifact_record is None:
            raise MappedHTTPException(404, "artifact_not_found")
        try:
            content = await artifacts.read_source(payload.artifact_id)
        except FileNotFoundError as exc:
            raise MappedHTTPException(404, "artifact_not_found") from exc

        return StreamingResponse(
            _artifact_chunks(content),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": _attachment_header(artifact_record["filename"]),
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/monitor/accounts")
    async def monitor_accounts(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        _: Principal = Depends(require_principal),
    ) -> dict[str, object]:
        return await monitoring.accounts(offset, limit)

    @app.get("/api/monitor/accounts/{account_id:path}/conversations")
    async def monitor_conversations(
        account_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        unassigned_offset: int = Query(default=0, ge=0),
        unassigned_limit: int = Query(default=50, ge=1, le=100),
        _: Principal = Depends(require_principal),
    ) -> dict[str, object]:
        try:
            return await monitoring.conversations(
                account_id.strip(), offset, limit, unassigned_offset, unassigned_limit
            )
        except AccountNotFound as exc:
            raise MappedHTTPException(404, "account_not_found") from exc

    @app.get("/api/monitor/accounts/{account_id:path}/conversations/{conversation_id}")
    async def monitor_conversation_detail(
        account_id: str,
        conversation_id: str,
        message_offset: int = Query(default=0, ge=0),
        message_limit: int = Query(default=100, ge=1, le=200),
        artifact_offset: int = Query(default=0, ge=0),
        artifact_limit: int = Query(default=50, ge=1, le=100),
        _: Principal = Depends(require_principal),
    ) -> dict[str, object]:
        try:
            normalized_conversation_id = str(UUID(conversation_id))
        except ValueError as exc:
            raise MappedHTTPException(422, "request_invalid") from exc
        try:
            return await monitoring.conversation_detail(
                account_id.strip(),
                normalized_conversation_id,
                message_offset,
                message_limit,
                artifact_offset,
                artifact_limit,
            )
        except ConversationNotFound as exc:
            raise MappedHTTPException(404, "conversation_not_found") from exc

    dist = config.project_root / "frontend" / "dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    async def index() -> Response:
        path = dist / "index.html"
        if not path.is_file():
            return HTMLResponse(
                "<h1>前端尚未构建</h1><p>请在 frontend 目录执行 npm ci 和 npm run build。</p>",
                status_code=503,
            )
        return FileResponse(path)

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> Response:
        path = dist / "favicon.svg"
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/svg+xml")

    return app
