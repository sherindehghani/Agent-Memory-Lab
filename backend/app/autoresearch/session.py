from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.agents.base import AgentContext
from app.agents.evaluator import EvaluatorAgent
from app.agents.llm import build_llm_client
from app.autoresearch.agents import AutoresearcherAgent, CodeExecutorAgent, CodeProposalInput, HarnessInput
from app.autoresearch.guard import count_code_lines
from app.autoresearch.harness import run_harness
from app.autoresearch.proposers import source_hash
from app.autoresearch.repo import AutoresearchRepo
from app.experiments.acceptance import EPS
from app.experiments.datasets import load_splits
from app.memory.retrieval import MemoryQuery, MemoryRetriever
from app.memory.store import MemoryStore
from app.models import AgentAction, Experiment, Run, new_id, utcnow
from app.orchestration.manager import _RUN_LOCK, ConflictError, RunManager, build_summary
from app.orchestration.state import ACTIVE, RunState, record_event, transition
from app.schemas.agents import BaselineSnapshot, EvaluatorInput

logger = logging.getLogger(__name__)

NAMESPACE = "autoresearch"
PROTOCOL = "autoresearch-cv5"


class AutoresearchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposer: Literal["offline", "llm"] = "offline"
    max_experiments: int = Field(20, ge=1, le=500)
    max_minutes: float = Field(30, ge=1, le=720)
    run_timeout_seconds: int | None = Field(None, ge=10, le=900)
    max_fix_attempts: int = Field(2, ge=0, le=5)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    tag: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")


def launch(manager: RunManager, request: AutoresearchCreate, *, background: bool = True) -> str:
    if not _RUN_LOCK.acquire(blocking=False):
        raise ConflictError("Another run or evaluation is currently executing.")
    try:
        with manager.session_factory() as s:
            active = s.scalar(select(Run.id).where(Run.state.in_([x.value for x in ACTIVE])).limit(1))
            if active:
                raise ConflictError(f"Run {active} is still active; stop it or wait for it to finish.")
            tag = request.tag or datetime.now(timezone.utc).strftime("%b%d").lower()
            timeout = request.run_timeout_seconds or manager.settings.autoresearch_run_timeout_seconds
            run = Run(id=new_id("run"), namespace=NAMESPACE, objective="Maximise cross-validated F1 by editing train.py",
                      state=RunState.CREATED.value, memory_enabled=True, planner_mode=f"autoresearch:{request.proposer}",
                      seed=request.seed, max_experiments=request.max_experiments,
                      max_duration_seconds=int(request.max_minutes * 60), max_retries=request.max_fix_attempts,
                      initial_model_type="train.py",
                      initial_configuration={"tag": tag, "proposer": request.proposer, "run_timeout_seconds": timeout})
            s.add(run)
            s.flush()
            record_event(s, run.id, "run_created", "Autoresearch session created", data=run.initial_configuration)
            s.commit()
            run_id = run.id
    except Exception:
        _RUN_LOCK.release()
        raise

    def target() -> None:
        try:
            AutoresearchSession(manager, run_id).execute()
        finally:
            _RUN_LOCK.release()

    if background:
        threading.Thread(target=target, name=f"autoresearch-{run_id}", daemon=True).start()
    else:
        target()
    return run_id


class AutoresearchSession:
    def __init__(self, manager: RunManager, run_id: str):
        self.manager = manager
        self.run_id = run_id
        self.repo = AutoresearchRepo(manager.settings.workspace_dir)
        self.proposer_agent = AutoresearcherAgent()
        self.executor = CodeExecutorAgent()
        self.evaluator = EvaluatorAgent()

    def execute(self) -> None:
        manager = self.manager
        session = manager.session_factory()
        try:
            run = session.get(Run, self.run_id)
            cfg = run.initial_configuration
            llm = build_llm_client(manager.settings) if cfg["proposer"] == "llm" else None
            ctx = AgentContext(session=session, run_id=run.id, namespace=NAMESPACE, settings=manager.settings,
                               store=MemoryStore(session, NAMESPACE), retriever=MemoryRetriever(session, NAMESPACE),
                               workspace=manager.workspace, rng=random.Random(run.seed),
                               should_stop=lambda: manager._stop_requested(self.run_id), llm=llm)
            self.repo.init()
            prior = ctx.store.get_baseline()
            start = prior.git_commit if prior and self.repo.commit_exists(prior.git_commit) else "master"
            branch = self.repo.start_branch(cfg["tag"], start)
            run.initial_configuration = {**cfg, "branch": branch, "start_commit": self.repo.head(),
                                         "start_experiment_id": prior.id if start != "master" else None}
            run.start_baseline_experiment_id = prior.id if start != "master" else None
            record_event(session, run.id, "autoresearch_setup",
                         f"Working on branch {branch}, starting from "
                         + (f"the best train.py of earlier sessions ({prior.id}, {start[:7]})" if start != "master"
                            else "master"), data={"branch": branch, "head": self.repo.head()})
            if cfg["proposer"] == "llm" and llm is None:
                record_event(session, run.id, "llm_unavailable", "LLM proposer requested but no provider is "
                             "configured; using the offline idea bank.", level=logging.WARNING)
            transition(session, run, RunState.PLANNING, "Autoresearch loop started")
            session.commit()

            self.best: dict[str, float] | None = None
            self.best_exp: Experiment | None = None
            self.best_lines = 0
            t0 = time.monotonic()
            done = 0
            while True:
                if manager._stop_requested(self.run_id):
                    run.stop_reason = "stop requested"
                    transition(session, run, RunState.CANCELLED, "Stopped by user")
                    break
                remaining = run.max_duration_seconds - (time.monotonic() - t0)
                if remaining < 5:
                    self._complete(ctx, run, f"time budget exhausted ({run.max_duration_seconds}s)")
                    break
                result = self._iteration(ctx, run, done, remaining)
                if result == "cancelled":
                    run.stop_reason = "stop requested during an experiment"
                    break
                if result == "exhausted":
                    self._complete(ctx, run, "the offline idea bank has no untried idea left")
                    break
                done += 1
                session.commit()
                if manager._stop_requested(self.run_id):
                    run.stop_reason = "stop requested"
                    transition(session, run, RunState.CANCELLED, "Stopped by user")
                    break
                if done >= run.max_experiments:
                    self._complete(ctx, run, f"experiment limit reached ({run.max_experiments})")
                    break
                transition(session, run, RunState.PLANNING, "Next idea")
                session.commit()
            run.summary = {**build_summary(session, run), "branch": branch,
                           "best_commit": self.best_exp.git_commit if self.best_exp else None}
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.exception("autoresearch session %s failed", self.run_id)
            manager._mark_failed(self.run_id, f"{type(exc).__name__}: {exc}")
        finally:
            session.close()

    def _history(self, session: Any, run: Run) -> list[dict[str, Any]]:
        exps = session.scalars(select(Experiment).where(Experiment.namespace == NAMESPACE)
                               .order_by(Experiment.created_at)).all()
        store = MemoryStore(session, NAMESPACE)
        rows = []
        for e in exps[-60:]:
            status = {"accepted": "keep", "rejected": "discard"}.get(e.decision or "", "crash")
            rows.append({"commit": e.git_commit or "0000000", "val_f1": store.metrics_of(e.id).get("f1", 0.0),
                         "status": status, "description": e.configuration.get("description", ""),
                         "session": e.run_id})
        return rows

    def _memories(self, ctx: AgentContext) -> list[dict[str, Any]]:
        results = ctx.retriever.search(MemoryQuery(kinds=["experiment_outcome", "failure", "primary_improvement"],
                                                   include_global=False, limit=20))
        return [{"memory_id": r.memory.id, "content": r.memory.content[:400]} for r in results]

    def _iteration(self, ctx: AgentContext, run: Run, index: int, remaining: float) -> str:
        session = ctx.session
        ctx.experiment_id = None
        parent_sha = self.repo.head()
        current = self.repo.read_train()
        action_ids: list[str] = []
        usage: dict[str, Any] | None = None
        if index == 0:
            description, hypothesis, source, idea = ("baseline", "Measure the unchanged train.py as the session "
                                                     "baseline.", current, "baseline")
            referenced: list[str] = []
        else:
            memories = self._memories(ctx)
            tried = ctx.retriever.tried_configurations(PROTOCOL)
            p = self.proposer_agent.run(ctx, CodeProposalInput(
                proposer=run.initial_configuration["proposer"], program=self.repo.read_program(),
                current_source=current, best_f1=self.best.get("f1") if self.best else None,
                history=self._history(session, run), memories=memories, tried_hashes=sorted(tried)))
            action_ids.append(p.action_id)
            if p.status != "success":
                raise RuntimeError(f"autoresearcher failed: {p.error}")
            out = p.output
            if out.exhausted:
                record_event(session, run.id, "search_exhausted", "No untried idea remains", agent_id="autoresearcher_01")
                return "exhausted"
            description, hypothesis, source, idea, usage = out.description, out.hypothesis, out.source, out.idea, \
                out.llm_usage
            referenced = [m["memory_id"] for m in memories][:10]

        exp = Experiment(
            id=new_id("exp"), run_id=run.id, namespace=NAMESPACE, sequence=index + 1,
            parent_experiment_id=self.best_exp.id if self.best_exp else None,
            baseline_experiment_id=self.best_exp.id if self.best_exp else None, proposed_by="autoresearcher_01",
            objective=run.objective, hypothesis=hypothesis, expected_effects=[{"metric": "f1", "direction": "increase"}]
            if index else [], reasoning_summary=f"idea: {idea}" + (f"; cites {len(referenced)} memories" if referenced
                                                                  else ""),
            plan_source="baseline" if index == 0 else run.initial_configuration["proposer"],
            referenced_memory_ids=referenced, model_type="train.py",
            configuration={"description": description, "idea": idea, "lines": count_code_lines(source),
                           "parent_commit": parent_sha, **({"llm_usage": usage} if usage else {})},
            config_hash=source_hash(source), config_diff={}, moves=[f"code:{idea}"], status="planned",
            acceptance_threshold=self.manager.settings.acceptance_threshold, seed=self.manager.settings.random_seed,
            dataset_version=load_splits(self.manager.settings.random_seed).version, protocol_version=PROTOCOL,
        )
        session.add(exp)
        session.flush()
        ctx.experiment_id = exp.id
        for aid in action_ids:
            session.get(AgentAction, aid).experiment_id = exp.id
        self.manager._send(ctx, "autoresearcher_01", "executor_01", "experiment_proposal", exp.id,
                           {"description": description, "hypothesis": hypothesis, "idea": idea})
        transition(session, run, RunState.READY, f"Proposed {exp.id}: {description}", experiment_id=exp.id)

        sha = self.repo.commit_train(source, f"{description}") if index else parent_sha
        exp.git_commit = sha
        record_event(session, run.id, "git_commit", f"{sha[:7]} {description}", experiment_id=exp.id)
        transition(session, run, RunState.EXECUTING, f"Running {exp.id}", experiment_id=exp.id)
        exp.status, exp.started_at = "running", utcnow()
        session.commit()

        timeout = float(run.initial_configuration["run_timeout_seconds"])
        attempts, executed = 0, None
        while True:
            attempts += 1
            e = self.executor.run(ctx, HarnessInput(experiment_id=exp.id, source=source, seed=exp.seed,
                                                    timeout_seconds=min(timeout, max(10.0, remaining))))
            executed = e
            out = e.output
            if out is None or out.status in ("completed", "cancelled") or attempts > run.max_retries or index == 0:
                break
            fix = self.proposer_agent.run(ctx, CodeProposalInput(
                proposer=run.initial_configuration["proposer"], program=self.repo.read_program(),
                current_source=current, best_f1=None, history=[], memories=[], tried_hashes=[],
                fix_error=f"{out.error}\n{out.log_tail}", failed_source=source))
            if fix.status != "success" or fix.output.exhausted:
                break
            source = fix.output.source
            sha = self.repo.commit_train(source, f"fix: {description}")
            exp.git_commit, exp.config_hash = sha, source_hash(source)
            record_event(session, run.id, "crash_fix_attempt", f"Fix attempt {attempts} committed as {sha[:7]}",
                         experiment_id=exp.id, agent_id="autoresearcher_01")
            session.commit()
        out = executed.output
        status = out.status if out else "failed"
        metrics = out.metrics if out else None
        error = (out.error if out else executed.error)
        exp.attempts, exp.executed_by, exp.error, exp.finished_at = attempts, "executor_01", error, utcnow()
        self.repo.keep_attempt(exp.id, sha)
        if status == "cancelled":
            exp.status = "cancelled"
            self.repo.reset_to(parent_sha)
            transition(session, run, RunState.CANCELLED, "Stopped during an experiment", experiment_id=exp.id)
            session.commit()
            return "cancelled"
        self.manager._send(ctx, "executor_01", "evaluator_01", "execution_result", exp.id,
                           {"status": status, "metrics": metrics, "error": error, "attempts": attempts})

        transition(session, run, RunState.EVALUATING, f"Evaluating {exp.id}", experiment_id=exp.id)
        snapshot = (BaselineSnapshot(experiment_id=self.best_exp.id, model_type="train.py", configuration={},
                                     metrics=self.best, git_commit=self.best_exp.git_commit) if self.best_exp else None)
        v = self.evaluator.run(ctx, EvaluatorInput(experiment_id=exp.id, status=status if status != "timeout" else
                                                   "timeout", metrics=metrics, error=error, baseline=snapshot,
                                                   expected_effects=exp.expected_effects,
                                                   threshold=exp.acceptance_threshold))
        ev = v.output
        decision, reason = ev.decision, ev.reason
        lines = count_code_lines(source)
        if decision == "rejected" and ev.primary_delta is not None and abs(ev.primary_delta) <= EPS \
                and lines < self.best_lines:
            decision = "accepted"
            reason = (f"Equal cross-validated F1 with simpler code ({lines} vs {self.best_lines} lines): kept under the "
                      f"simplicity rule.")
        exp.status = "completed" if status == "completed" else "failed"
        exp.decision, exp.decision_reason = decision, reason
        exp.primary_delta, exp.secondary_deltas = ev.primary_delta, ev.secondary_deltas
        exp.hypothesis_outcome, exp.hypothesis_details = ev.hypothesis_outcome, ev.hypothesis_details
        record_event(session, run.id, "evaluation_decision", f"{decision}: {reason}", experiment_id=exp.id,
                     agent_id="evaluator_01")

        transition(session, run, RunState.STORING_MEMORY, f"Storing {exp.id}", experiment_id=exp.id)
        if metrics:
            ctx.store.save_metrics(exp.id, metrics)
        tsv_status = {"accepted": "keep", "rejected": "discard"}.get(decision, "crash")
        if decision == "accepted":
            self.best, self.best_exp, self.best_lines = metrics, exp, lines
            ctx.store.set_baseline(exp, reason)
            record_event(session, run.id, "branch_advanced", f"Kept {sha[:7]}; branch advances", experiment_id=exp.id)
        else:
            self.repo.reset_to(parent_sha)
            record_event(session, run.id, "branch_reset", f"Discarded {sha[:7]}; reset to {parent_sha[:7]}",
                         experiment_id=exp.id)
        self.repo.append_result(sha, (metrics or {}).get("f1", 0.0), (metrics or {}).get("peak_memory_mb", 0.0),
                                tsv_status, description)
        self.manager._store_memories(ctx, run, exp, ev.secondary_improvements if decision == "rejected" else [],
                                     metrics, {"plan": action_ids[0] if action_ids else executed.action_id,
                                               "execute": executed.action_id, "evaluate": v.action_id}, None)
        self.manager._send(ctx, "evaluator_01", "broadcast", "evaluation_decision", exp.id,
                           {"decision": decision, "reason": reason, "status": tsv_status})
        session.flush()
        return "done"

    def _complete(self, ctx: AgentContext, run: Run, reason: str) -> None:
        run.stop_reason = reason
        run.end_baseline_experiment_id = self.best_exp.id if self.best_exp else None
        if self.best_exp is not None:
            ctx.session.commit()
            source = self.repo.read_train(self.best_exp.git_commit)
            out = run_harness(source, self.best_exp.seed, split="test",
                              timeout=float(run.initial_configuration["run_timeout_seconds"]))
            if out.status == "completed" and out.metrics:
                ctx.store.save_metrics(self.best_exp.id, {k: out.metrics[k] for k in
                                                          ("accuracy", "f1", "precision", "recall")}, split="test")
                record_event(ctx.session, run.id, "final_test_evaluation",
                             f"Held-out test F1 of best train.py ({self.best_exp.git_commit[:7]}): "
                             f"{out.metrics['f1']:.4f}", experiment_id=self.best_exp.id)
        transition(ctx.session, run, RunState.COMPLETED, reason)
