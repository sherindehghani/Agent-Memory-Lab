from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.common import api_error, get_manager, not_found
from app.core.database import get_db
from app.models import EvaluationResult
from app.orchestration.evaluation import start_comparison
from app.orchestration.manager import ConflictError, RunManager
from app.schemas.api import CompareRequest

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


def _out(r: EvaluationResult) -> dict[str, Any]:
    return {"id": r.id, "status": r.status, "params": r.params, "results": r.results, "error": r.error,
            "created_at": r.created_at.isoformat(), "finished_at": r.finished_at.isoformat() if r.finished_at else None}


@router.post("/compare", status_code=202)
def compare(body: CompareRequest, db: Session = Depends(get_db),
            manager: RunManager = Depends(get_manager)) -> dict[str, Any]:
    """Run memory-enabled vs. memory-disabled trials in isolated namespaces (background)."""
    try:
        eval_id = start_comparison(manager, body)
    except ConflictError as exc:
        raise api_error(409, "run_conflict", str(exc)) from None
    return _out(db.get(EvaluationResult, eval_id))


@router.get("/results")
def results(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(EvaluationResult).order_by(EvaluationResult.created_at.desc()).limit(limit)).all()
    return [_out(r) for r in rows]


@router.get("/results/{evaluation_id}")
def result(evaluation_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.get(EvaluationResult, evaluation_id)
    if row is None:
        raise not_found("evaluation", evaluation_id)
    return _out(row)
