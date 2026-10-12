"""HTTP routes, authentication, and SSE conversion around AgentRunner."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from contextlib import asynccontextmanager
from urllib.parse import quote
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.contracts import AgentEvent, ChatTurn, ModelGateway
from agent.runner import AgentRunner
from backend.adapters.files import IndexedArtifactWriter, LocalArtifactStore, LocalInterfaceReader
from backend.adapters.monitoring import (
    AccountNotFound,
    ConversationNotFound,
    MonitoringStore,
)
from backend.adapters.ollama import OllamaModelGateway
from backend.config import Settings
from backend.schemas import ChatRequest, DownloadRequest
from backend.security import Principal, require_principal


ARTIFACT_STREAM_CHUNK_SIZE = 64 * 1024


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
) -> FastAPI:
    config = settings or Settings.from_env()
    model = model_gateway or OllamaModelGateway(config.ollama_base_url)
    artifacts = artifact_store or LocalArtifactStore(config.project_root / "artifacts")
    monitoring = MonitoringStore(config.data_dir / "monitoring.sqlite3", artifacts.directory)
    runner = agent_runner or AgentRunner(
        model,
        LocalInterfaceReader(config.project_root / "interfaces" / "价格接口"),
        IndexedArtifactWriter(artifacts, monitoring),
    )
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
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.get("/api/models")
    async def models(_: Principal = Depends(require_principal)) -> dict[str, object]:
        try:
            return {
                "models": await model.list_models(),
                "default_model": config.ollama_model,
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Ollama 模型列表不可用：{exc}") from exc

    @app.post("/api/agent/chat/stream")
    async def chat(
        payload: ChatRequest, principal: Principal = Depends(require_principal)
    ) -> StreamingResponse:
        selected_model = payload.model or config.ollama_model
        conversation_id = str(payload.conversation_id)
        try:
            await monitoring.prepare_conversation(
                payload.account_id, conversation_id, payload.message
            )
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="会话不存在") from exc

        async def stream():
            async with conversation_locks.hold(conversation_id):
                history_rows, _, assistant_message_id = await monitoring.begin_turn(
                    payload.account_id, conversation_id, payload.message
                )
                history = [ChatTurn(role, content) for role, content in history_rows]
                partial_content: list[str] = []
                finished = False
                try:
                    async for event in runner.run(
                        message=payload.message,
                        history=history,
                        model=selected_model,
                        account_id=payload.account_id,
                        conversation_id=conversation_id,
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
                            await monitoring.finish_turn(
                                conversation_id,
                                assistant_message_id,
                                "".join(partial_content),
                                "complete",
                            )
                            finished = True
                        yield _sse(event, conversation_id=conversation_id)
                    if not finished:
                        await monitoring.finish_turn(
                            conversation_id,
                            assistant_message_id,
                            "".join(partial_content),
                            "failed",
                        )
                        yield _sse(
                            AgentEvent(
                                "error",
                                {"code": "agent_incomplete", "message": "Agent 响应提前结束"},
                            ),
                            conversation_id=conversation_id,
                        )
                except asyncio.CancelledError:
                    if not finished:
                        await monitoring.finish_turn(
                            conversation_id,
                            assistant_message_id,
                            "".join(partial_content),
                            "interrupted",
                        )
                    raise
                except Exception as exc:
                    if not finished:
                        await monitoring.finish_turn(
                            conversation_id,
                            assistant_message_id,
                            "".join(partial_content),
                            "failed",
                        )
                    yield _sse(
                        AgentEvent("error", {"code": "agent_failed", "message": str(exc)}),
                        conversation_id=conversation_id,
                    )

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
            raise HTTPException(status_code=404, detail="产物不存在")
        try:
            content = await artifacts.read_source(payload.artifact_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="产物不存在") from exc

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
            raise HTTPException(status_code=404, detail="账号不存在") from exc

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
            raise HTTPException(status_code=422, detail="conversation_id 必须是 UUID") from exc
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
            raise HTTPException(status_code=404, detail="会话不存在") from exc

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
