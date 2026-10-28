from __future__ import annotations

import logging
import threading
from statistics import mean
from typing import Any

from app.core.database import new_session
from app.models import EvaluationResult, Run, new_id, utcnow
from app.orchestration.manager import _RUN_LOCK, ConflictError, RunManager
from app.schemas.api import CompareRequest, RunCreate

logger = logging.getLogger(__name__)

CONDITIONS = (("memory_disabled", False), ("memory_enabled", True))
TRIAL_METRICS = ("best_validation_f1", "experiments", "accepted", "repeated_configurations",
                 "prior_findings_reused", "experiment_seconds_total", "memory_retrievals",
                 "memory_retrieval_ms_mean")


def start_comparison(manager: RunManager, request: CompareRequest, *, background: bool = True) -> str:
    if not _RUN_LOCK.acquire(blocking=False):
        raise ConflictError("Another run or evaluation is currently executing.")
    try:
        with new_session() as s:
            row = EvaluationResult(id=new_id("eval"), status="running", params=request.model_dump())
            s.add(row)
            s.commit()
            eval_id = row.id
    except Exception:
        _RUN_LOCK.release()
        raise

    def target() -> None:
        try:
            _run_comparison(manager, eval_id, request)
        finally:
            _RUN_LOCK.release()

    if background:
        threading.Thread(target=target, name=f"eval-{eval_id}", daemon=True).start()
    else:
        target()
    return eval_id


def _run_comparison(manager: RunManager, eval_id: str, request: CompareRequest) -> None:
    trials: list[dict[str, Any]] = []
    try:
        for seed in request.seeds:
            for condition, memory_enabled in CONDITIONS:
                namespace = f"eval:{eval_id}:{condition}:seed{seed}"
                run_id = manager.create_run(RunCreate(
                    objective=request.objective, memory_enabled=memory_enabled, max_experiments=request.budget,
                    seed=seed, initial_model_type=request.initial_model_type,
                    initial_configuration=request.initial_configuration), namespace=namespace)
                manager.execute_run(run_id)
                with new_session() as s:
                    run = s.get(Run, run_id)
                    summary = run.summary or {}
                    trials.append({"seed": seed, "condition": condition, "run_id": run_id, "namespace": namespace,
                                   "state": run.state, **{k: summary.get(k) for k in TRIAL_METRICS},
                                   "final_baseline_test_f1": (summary.get("final_baseline_test_metrics") or {})
                                   .get("f1")})
        results = {"trials": trials, "aggregate": aggregate(trials),
                   "note": "Demonstration on one dataset and few seeds; differences are measured, not "
                           "statistically tested. Test F1 is reported for information only and was never used "
                           "for selection."}
        status, error = "completed", None
    except Exception as exc:
        logger.exception("evaluation %s failed", eval_id)
        results, status, error = {"trials": trials, "aggregate": aggregate(trials)}, "failed", str(exc)
    with new_session() as s:
        row = s.get(EvaluationResult, eval_id)
        row.status, row.results, row.error, row.finished_at = status, results, error, utcnow()
        s.commit()


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition, _ in CONDITIONS:
        rows = [t for t in trials if t["condition"] == condition]
        out[condition] = {
            m: round(mean(r[m] for r in rows if r.get(m) is not None), 6) if any(r.get(m) is not None for r in rows)
            else None
            for m in (*TRIAL_METRICS, "final_baseline_test_f1")
        }
        out[condition]["trials"] = len(rows)
    on, off = out.get("memory_enabled", {}), out.get("memory_disabled", {})
    out["difference_memory_minus_no_memory"] = {
        m: round(on[m] - off[m], 6) for m in TRIAL_METRICS if on.get(m) is not None and off.get(m) is not None
    }
    return out
