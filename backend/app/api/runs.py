from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.common import MAX_PAGE, api_error, experiments_out, get_manager, namespace_filter, not_found, paginate
from app.core.database import get_db
from app.models import AgentMessage, Experiment, Run, RunEvent
from app.orchestration.manager import ConflictError, NotFoundError, RunManager
from app.schemas.api import Page, RunCreate, RunOut

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", status_code=201, response_model=RunOut)
def create_run(body: RunCreate, db: Session = Depends(get_db), manager: RunManager = Depends(get_manager)) -> Run:
    try:
        run_id = manager.launch(body)
    except ConflictError as exc:
        raise api_error(409, "run_conflict", str(exc)) from None
    return db.get(Run, run_id)


@router.get("", response_model=Page[RunOut])
def list_runs(namespace: str = "main", state: str | None = None, limit: int = Query(50, ge=1, le=MAX_PAGE),
              offset: int = Query(0, ge=0), db: Session = Depends(get_db)) -> dict[str, Any]:
    stmt = select(Run).where(namespace_filter(Run.namespace, namespace)).order_by(Run.created_at.desc())
    if state:
        stmt = stmt.where(Run.state == state.upper())
    items, total = paginate(db, stmt, limit, offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(Run, run_id)
    if run is None:
        raise not_found("run", run_id)
    events = db.scalars(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.id)).all()
    exps = db.scalars(select(Experiment).where(Experiment.run_id == run_id).order_by(Experiment.sequence)).all()
    return {
        "run": RunOut.model_validate(run).model_dump(mode="json"),
        "events": [{"id": e.id, "event_type": e.event_type, "from_state": e.from_state, "to_state": e.to_state,
                    "experiment_id": e.experiment_id, "agent_id": e.agent_id, "message": e.message, "data": e.data,
                    "created_at": e.created_at.isoformat()} for e in events],
        "experiments": [e.model_dump(mode="json") for e in experiments_out(db, list(exps))],
    }


@router.get("/{run_id}/messages")
def run_messages(run_id: str, limit: int = Query(200, ge=1, le=1000), db: Session = Depends(get_db)) -> list[dict]:
    if db.get(Run, run_id) is None:
        raise not_found("run", run_id)
    rows = db.scalars(select(AgentMessage).where(AgentMessage.run_id == run_id)
                      .order_by(AgentMessage.created_at).limit(limit)).all()
    return [{"id": m.id, "sender": m.sender, "recipient": m.recipient, "message_type": m.message_type,
             "experiment_id": m.experiment_id, "payload": m.payload, "created_at": m.created_at.isoformat()}
            for m in rows]


@router.post("/{run_id}/stop", status_code=202, response_model=RunOut)
def stop_run(run_id: str, manager: RunManager = Depends(get_manager)) -> Run:
    try:
        return manager.request_stop(run_id)
    except NotFoundError:
        raise not_found("run", run_id) from None
    except ConflictError as exc:
        raise api_error(409, "run_not_active", str(exc)) from None


@router.post("/{run_id}/resume", status_code=201, response_model=RunOut)
def resume_run(run_id: str, db: Session = Depends(get_db), manager: RunManager = Depends(get_manager)) -> Run:
    try:
        new_id = manager.resume(run_id)
    except NotFoundError:
        raise not_found("run", run_id) from None
    except ConflictError as exc:
        raise api_error(409, "run_conflict", str(exc)) from None
    return db.get(Run, new_id)
