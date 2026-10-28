from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.common import api_error, experiments_out, get_manager, not_found
from app.autoresearch.repo import AutoresearchRepo
from app.autoresearch.session import NAMESPACE, AutoresearchCreate, launch
from app.core.database import get_db
from app.experiments.versioning import GitError
from app.memory.store import MemoryStore
from app.models import Experiment, Run
from app.orchestration.manager import ConflictError, RunManager
from app.schemas.api import RunOut

router = APIRouter(prefix="/api/autoresearch", tags=["autoresearch"])


class ProgramBody(BaseModel):
    content: str = Field(min_length=1, max_length=50_000)


def _repo(manager: RunManager) -> AutoresearchRepo:
    repo = AutoresearchRepo(manager.settings.workspace_dir)
    repo.init()
    return repo


@router.get("/program")
def get_program(manager: RunManager = Depends(get_manager)) -> dict[str, str]:
    return {"content": _repo(manager).read_program()}


@router.put("/program")
def put_program(body: ProgramBody, manager: RunManager = Depends(get_manager)) -> dict[str, str]:
    repo = _repo(manager)
    repo.write_program(body.content)
    return {"content": repo.read_program()}


@router.post("/sessions", status_code=201, response_model=RunOut)
def start_session(body: AutoresearchCreate, db: Session = Depends(get_db),
                  manager: RunManager = Depends(get_manager)) -> Run:
    try:
        run_id = launch(manager, body)
    except ConflictError as exc:
        raise api_error(409, "run_conflict", str(exc)) from None
    return db.get(Run, run_id)


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    runs = db.scalars(select(Run).where(Run.namespace == NAMESPACE).order_by(Run.created_at.desc()).limit(50)).all()
    return [RunOut.model_validate(r).model_dump(mode="json") for r in runs]


@router.get("/sessions/{run_id}")
def get_session(run_id: str, db: Session = Depends(get_db), manager: RunManager = Depends(get_manager)) -> dict:
    run = db.get(Run, run_id)
    if run is None or run.namespace != NAMESPACE:
        raise not_found("autoresearch session", run_id)
    exps = db.scalars(select(Experiment).where(Experiment.run_id == run_id).order_by(Experiment.sequence)).all()
    rows = []
    for e in experiments_out(db, list(exps)):
        rows.append({"experiment_id": e.id, "commit": (e.git_commit or "")[:7], "val_f1": e.metrics.get("f1"),
                     "f1_std": e.metrics.get("f1_std"), "memory_mb": e.metrics.get("peak_memory_mb"),
                     "status": {"accepted": "keep", "rejected": "discard"}.get(e.decision or "",
                                                                             "crash" if e.decision else e.status),
                     "description": e.configuration.get("description", ""), "idea": e.configuration.get("idea"),
                     "lines": e.configuration.get("lines"), "error": e.error, "decision_reason": e.decision_reason,
                     "attempts": e.attempts})
    repo = _repo(manager)
    best = next((e for e in reversed(exps) if e.decision == "accepted"), None)
    best_source = None
    if best and best.git_commit:
        try:
            best_source = repo.read_train(best.git_commit)
        except GitError:
            best_source = None
    return {"run": RunOut.model_validate(run).model_dump(mode="json"), "results": rows,
            "best": {"experiment_id": best.id, "commit": best.git_commit, "source": best_source,
                     "test_metrics": MemoryStore(db, NAMESPACE).metrics_of(best.id, "test")} if best else None}


@router.get("/experiments/{experiment_id}/diff")
def experiment_diff(experiment_id: str, db: Session = Depends(get_db),
                    manager: RunManager = Depends(get_manager)) -> dict[str, Any]:
    exp = db.get(Experiment, experiment_id)
    if exp is None or exp.namespace != NAMESPACE:
        raise not_found("autoresearch experiment", experiment_id)
    repo = _repo(manager)
    parent = exp.configuration.get("parent_commit")
    try:
        source = repo.read_train(exp.git_commit) if exp.git_commit else None
    except GitError:
        source = None
    log_path = repo.logs / f"{exp.id}.log"
    return {"experiment_id": exp.id, "commit": exp.git_commit, "parent_commit": parent,
            "diff": repo.diff(parent, exp.git_commit) if parent and exp.git_commit else "",
            "source": source, "log_tail": "\n".join(log_path.read_text().splitlines()[-50:]) if log_path.exists()
            else ""}
