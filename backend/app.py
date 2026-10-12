"""HTTP routes, authentication, and SSE conversion around AgentRunner."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.contracts import AgentEvent, ChatTurn, ModelGateway
from agent.runner import AgentRunner
from backend.adapters.files import LocalArtifactStore, LocalInterfaceReader
from backend.adapters.ollama import OllamaModelGateway
from backend.config import Settings
from backend.schemas import ChatRequest, DownloadRequest
from backend.security import Principal, require_principal


ARTIFACT_STREAM_CHUNK_SIZE = 64 * 1024


def _artifact_chunks(content: bytes) -> Iterator[bytes]:
    for start in range(0, len(content), ARTIFACT_STREAM_CHUNK_SIZE):
        yield content[start : start + ARTIFACT_STREAM_CHUNK_SIZE]


def _sse(event: AgentEvent) -> bytes:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.kind}\ndata: {data}\n\n".encode("utf-8")


def _attachment_header(filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9_.-]", "_", ascii_name).strip("._")
    if not ascii_name.lower().endswith(".py"):
        ascii_name = f"{ascii_name or 'strategy'}.py"
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\x27\x27{quote(filename, safe="")}'


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
    runner = agent_runner or AgentRunner(
        model, LocalInterfaceReader(config.project_root / "interfaces" / "价格接口"), artifacts
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(model, "close", None)
        if close is not None:
            await close()

    app = FastAPI(title="风控策略 Agent", lifespan=lifespan)
    app.state.settings = config
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
        history = [ChatTurn(item.role, item.content) for item in payload.history]

        async def stream():
            try:
                async for event in runner.run(
                    message=payload.message,
                    history=history,
                    model=selected_model,
                    account_id=principal.account_id,
                    options=payload.options.model_dump(),
                    think=payload.think,
                ):
                    yield _sse(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield _sse(AgentEvent("error", {"code": "agent_failed", "message": str(exc)}))

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/artifacts/download")
    async def download(
        payload: DownloadRequest, principal: Principal = Depends(require_principal)
    ) -> StreamingResponse:
        try:
            artifact = await artifacts.read(payload.artifact_id, principal.account_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="产物不存在") from exc

        return StreamingResponse(
            _artifact_chunks(artifact.content),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": _attachment_header(artifact.filename),
                "Cache-Control": "no-store",
            },
        )

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
