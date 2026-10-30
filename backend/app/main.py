from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import __version__
from app.api import agents, autoresearch, evaluation, experiments, graph, memory, runs
from app.core.config import get_settings
from app.core.database import session_scope
from app.core.logging import configure_logging
from app.experiments.versioning import GitWorkspace
from app.orchestration.manager import RunManager, lab_is_busy
from app.orchestration.recovery import bootstrap

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app.state.recovered_runs = bootstrap(settings)
    app.state.manager = RunManager(settings)
    yield


app = FastAPI(
    title="AI Agent Memory Lab",
    version=__version__,
    description="Agents that run reproducible ML experiments, share persistent memory, and plan from evidence.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["X-Request-ID", "X-Process-Time-Ms"])
app.add_middleware(GZipMiddleware, minimum_size=get_settings().gzip_minimum_size)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info("request", extra={"event_type": "http_request", "data": {
        "request_id": request_id, "method": request.method, "path": request.url.path,
        "status": response.status_code, "duration_ms": round(elapsed_ms, 1)}})
    return response


def _error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return _error(exc.status_code, exc.detail["code"], exc.detail["message"], exc.detail.get("details"))
    return _error(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return _error(422, "validation_error", "Request validation failed.", details)


@app.exception_handler(SQLAlchemyError)
async def database_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("database error", extra={"event_type": "database_error"})
    return _error(503, "database_error", "A database error occurred; see server logs.")


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", extra={"event_type": "internal_error"})
    return _error(500, "internal_error", "Internal server error; see server logs.")


@app.get("/api/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health/ready", tags=["health"])
def readiness() -> JSONResponse:
    try:
        with session_scope() as s:
            s.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(status_code=503, content={"status": "unavailable", "database": "error"})
    return JSONResponse(content={"status": "ok", "database": "ok"})


@app.get("/api/health", tags=["health"])
def health(request: Request) -> dict[str, Any]:
    settings = get_settings()
    with session_scope() as s:
        s.execute(text("SELECT 1"))
    ws = GitWorkspace(settings.workspace_dir)
    return {"status": "ok", "database": "ok", "workspace_git": ws.commit_exists(ws.root_commit()),
            "busy": lab_is_busy(), "execution_mode": settings.execution_mode,
            "llm_provider": settings.llm_provider if settings.anthropic_api_key else "none",
            "recovered_runs_on_startup": request.app.state.recovered_runs}


for router in (runs.router, experiments.router, memory.router, agents.router, graph.router, evaluation.router,
               autoresearch.router):
    app.include_router(router)
