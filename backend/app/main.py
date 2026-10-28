from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api import agents, evaluation, experiments, graph, memory, runs
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
    version="0.1.0",
    description="Agents that run reproducible ML experiments, share persistent memory, and plan from evidence.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"],
                   allow_headers=["*"])


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


for router in (runs.router, experiments.router, memory.router, agents.router, graph.router, evaluation.router):
    app.include_router(router)
