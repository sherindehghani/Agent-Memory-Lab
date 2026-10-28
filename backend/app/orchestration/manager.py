"""Experiment Manager: drives agents through the run state machine.

One iteration:  PLANNING (Researcher -> Planner) -> READY -> EXECUTING (Executor)
-> EVALUATING (Evaluator) -> STORING_MEMORY (metrics, baseline pointer, memories,
findings, graph relationships) -> PLANNING ... -> COMPLETED.

Only one run (or evaluation) executes at a time in a process (``_RUN_LOCK``),
and the database rejects a new run while another is active, so two runs can
never race on the same baseline pointer.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.base import AgentContext
from app.agents.evaluator import EvaluatorAgent
from app.agents.executor import ExecutorAgent
from app.agents.llm import build_llm_client
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.core.config import Settings, get_settings
from app.core.database import new_session
from app.experiments import runner
from app.experiments.datasets import load_splits
from app.experiments.search_space import config_diff, config_hash, moves_from_diff
from app.experiments.versioning import GitWorkspace
from app.memory.graph import GraphEvidenceError, add_relationship
from app.memory.retrieval import MemoryRetriever
from app.memory.store import MemoryStore
from app.models import (
    AgentAction,
    AgentMessage,
    Experiment,
    MemoryRecord,
    Run,
    new_id,
    utcnow,
)
from app.orchestration.state import ACTIVE, RunState, record_event, transition
from app.schemas.agents import (
    AgentMessageSchema,
    BaselineSnapshot,
    EvaluatorInput,
    ExecutorInput,
    PlannerInput,
    PlannerOutput,
    ResearcherInput,
    ResearchSuggestion,
)
from app.schemas.api import RunCreate

logger = logging.getLogger(__name__)

_RUN_LOCK = threading.Lock()


class ConflictError(RuntimeError):
    pass


class NotFoundError(LookupError):
    pass


def lab_is_busy() -> bool:
    return _RUN_LOCK.locked()


class RunManager:
    def __init__(self, settings: Settings | None = None, session_factory: Callable[[], Session] = new_session,
                 workspace: GitWorkspace | None = None):
        self.settings = settings or get_settings()
        self.session_factory = session_factory
        self.workspace = workspace or GitWorkspace(self.settings.workspace_dir)
        self.planner, self.executor = PlannerAgent(), ExecutorAgent()
        self.evaluator, self.researcher = EvaluatorAgent(), ResearcherAgent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def launch(self, request: RunCreate, *, background: bool = True, namespace: str = "main",
               parent_run_id: str | None = None) -> str:
        """Create a run and execute it (in a thread by default). Raises ConflictError if busy."""
        if not _RUN_LOCK.acquire(blocking=False):
            raise ConflictError("Another run or evaluation is currently executing.")
        try:
            run_id = self.create_run(request, namespace=namespace, parent_run_id=parent_run_id)
        except Exception:
            _RUN_LOCK.release()
            raise

        def target() -> None:
            try:
                self.execute_run(run_id)
            finally:
                _RUN_LOCK.release()

        if background:
            threading.Thread(target=target, name=f"run-{run_id}", daemon=True).start()
        else:
            target()
        return run_id

    def create_run(self, request: RunCreate, *, namespace: str = "main", parent_run_id: str | None = None) -> str:
        with self.session_factory() as session:
            active = session.scalar(select(Run.id).where(Run.state.in_([s.value for s in ACTIVE])).limit(1))
            if active:
                raise ConflictError(f"Run {active} is still active; stop it or wait for it to finish.")
            s = self.settings
            run = Run(
                id=new_id("run"), namespace=namespace, objective=request.objective, state=RunState.CREATED.value,
                memory_enabled=request.memory_enabled, planner_mode=request.planner_mode, seed=request.seed,
                max_experiments=request.max_experiments or s.max_experiments,
                max_duration_seconds=request.max_duration_seconds or s.max_run_seconds,
                max_retries=s.max_retries if request.max_retries is None else request.max_retries,
                initial_model_type=request.initial_model_type,
                initial_configuration=request.initial_configuration or {}, parent_run_id=parent_run_id,
            )
            session.add(run)
            session.flush()
            record_event(session, run.id, "run_created", f"Run created in namespace '{namespace}'",
                         data={"memory_enabled": run.memory_enabled, "max_experiments": run.max_experiments,
                               "planner_mode": run.planner_mode})
            session.commit()
            return run.id

    def request_stop(self, run_id: str) -> Run:
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise NotFoundError(run_id)
            if RunState(run.state) not in ACTIVE:
                raise ConflictError(f"Run {run_id} is already {run.state}.")
            run.stop_requested = True
            record_event(session, run_id, "stop_requested", "Stop requested by user")
            session.commit()
            return run

    def resume(self, run_id: str, *, background: bool = True) -> str:
        """Start a new run continuing from the namespace's last accepted baseline."""
        with self.session_factory() as session:
            old = session.get(Run, run_id)
            if old is None:
                raise NotFoundError(run_id)
            if old.state not in (RunState.INTERRUPTED, RunState.CANCELLED, RunState.FAILED):
                raise ConflictError(f"Run {run_id} is {old.state}; only interrupted, cancelled or failed runs "
                                    f"can be resumed.")
            request = RunCreate(objective=old.objective, memory_enabled=old.memory_enabled,
                                planner_mode=old.planner_mode, max_experiments=old.max_experiments,
                                max_duration_seconds=old.max_duration_seconds, max_retries=old.max_retries,
                                seed=old.seed, initial_model_type=old.initial_model_type,  # type: ignore[arg-type]
                                initial_configuration=old.initial_configuration)
            namespace = old.namespace
        new_run_id = self.launch(request, background=background, namespace=namespace, parent_run_id=run_id)
        with self.session_factory() as session:
            record_event(session, run_id, "recovery", f"Resumed as {new_run_id} from the last accepted baseline",
                         data={"new_run_id": new_run_id})
            session.commit()
        return new_run_id

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _stop_requested(self, run_id: str) -> bool:
        with self.session_factory() as s:
            return bool(s.scalar(select(Run.stop_requested).where(Run.id == run_id)))

    def execute_run(self, run_id: str) -> None:
        session = self.session_factory()
        try:
            run = session.get(Run, run_id)
            if run is None:
                raise NotFoundError(run_id)
            ns = run.namespace
            ctx = AgentContext(
                session=session, run_id=run.id, namespace=ns, settings=self.settings,
                store=MemoryStore(session, ns), retriever=MemoryRetriever(session, ns), workspace=self.workspace,
                rng=random.Random(run.seed), should_stop=lambda: self._stop_requested(run_id),
                execution_mode=self.settings.execution_mode,
                llm=build_llm_client(self.settings) if run.planner_mode == "llm" else None,
            )
            baseline = ctx.store.get_baseline()
            run.start_baseline_experiment_id = baseline.id if baseline else None
            transition(session, run, RunState.PLANNING, "Run started")
            session.commit()

            t0 = time.monotonic()
            executed = 0
            while True:  # invariant: the run is in PLANNING here
                if self._stop_requested(run_id):
                    run.stop_reason = "stop requested"
                    transition(session, run, RunState.CANCELLED, "Stopped by user before planning")
                    break
                remaining = run.max_duration_seconds - (time.monotonic() - t0)
                if remaining < 1:
                    self._complete(ctx, run, f"time budget exhausted ({run.max_duration_seconds}s)")
                    break
                result = self._iteration(ctx, run, executed, remaining)
                if result == "cancelled":
                    run.stop_reason = "stop requested during execution"
                    break
                if result == "exhausted":
                    self._complete(ctx, run, "no untested bounded neighbour of the baseline remains")
                    break
                executed += 1
                session.commit()
                if self._stop_requested(run_id):
                    run.stop_reason = "stop requested"
                    transition(session, run, RunState.CANCELLED, "Stopped by user")
                    break
                if executed >= run.max_experiments:
                    self._complete(ctx, run, f"experiment limit reached ({run.max_experiments})")
                    break
                transition(session, run, RunState.PLANNING, "Next iteration")
                session.commit()
            run.summary = build_summary(session, run, self.workspace)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.exception("run %s failed", run_id)
            self._mark_failed(run_id, f"{type(exc).__name__}: {exc}")
        finally:
            session.close()

    def _mark_failed(self, run_id: str, error: str) -> None:
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if run is None or RunState(run.state) not in ACTIVE:
                return
            run.error = error
            for exp in session.scalars(select(Experiment).where(Experiment.run_id == run_id,
                                                                Experiment.status.in_(["planned", "running"]))):
                exp.status, exp.error = "failed", f"run failed: {error}"
            transition(session, run, RunState.FAILED, error)
            run.summary = build_summary(session, run, self.workspace)
            session.commit()

    def _send(self, ctx: AgentContext, sender: str, recipient: str, message_type: str,
              experiment_id: str | None, payload: dict[str, Any]) -> None:
        msg = AgentMessageSchema(sender=sender, recipient=recipient, message_type=message_type,  # type: ignore[arg-type]
                                 run_id=ctx.run_id, experiment_id=experiment_id, payload=payload)
        ctx.session.add(AgentMessage(id=new_id("msg"), run_id=msg.run_id, experiment_id=msg.experiment_id,
                                     sender=msg.sender, recipient=msg.recipient, message_type=msg.message_type,
                                     payload=msg.model_dump(mode="json")["payload"], created_at=msg.timestamp))
        ctx.session.flush()

    def _snapshot(self, ctx: AgentContext, exp: Experiment | None) -> BaselineSnapshot | None:
        if exp is None:
            return None
        return BaselineSnapshot(experiment_id=exp.id, model_type=exp.model_type, configuration=exp.configuration,
                                metrics=ctx.store.metrics_of(exp.id), git_commit=exp.git_commit)

    def _iteration(self, ctx: AgentContext, run: Run, index: int, remaining: float) -> str:
        session, store = ctx.session, ctx.store
        ctx.experiment_id = None
        baseline = store.get_baseline()
        initial_type, initial_cfg = run.initial_model_type, run.initial_configuration
        if baseline is not None and baseline.protocol_version != runner.PROTOCOL_VERSION:
            # Scores from another protocol are not comparable: re-measure the old baseline's
            # configuration under the current protocol and make that the new baseline.
            record_event(session, run.id, "protocol_changed",
                         f"Baseline {baseline.id} was measured with {baseline.protocol_version}; re-baselining its "
                         f"configuration under {runner.PROTOCOL_VERSION}.")
            initial_type, initial_cfg = baseline.model_type, baseline.configuration
            baseline = None
        snapshot = self._snapshot(ctx, baseline)

        # -- PLANNING: Researcher (memory enabled) then Planner -------------
        suggestion: ResearchSuggestion | None = None
        action_ids: list[str] = []
        if run.memory_enabled and snapshot is not None:
            r = self.researcher.run(ctx, ResearcherInput(objective=run.objective, baseline=snapshot))
            action_ids.append(r.action_id)
            if r.status == "success" and r.output.suggestion:  # type: ignore[union-attr]
                suggestion = r.output.suggestion  # type: ignore[union-attr]
                self._send(ctx, "researcher_01", "planner_01", "research_suggestion", None,
                           suggestion.model_dump(mode="json"))
        p = self.planner.run(ctx, PlannerInput(
            objective=run.objective, baseline=snapshot, initial_model_type=initial_type,
            initial_configuration=initial_cfg, memory_enabled=run.memory_enabled,
            mode=run.planner_mode, iteration=index, research_suggestion=suggestion))  # type: ignore[arg-type]
        action_ids.append(p.action_id)
        if p.status != "success":
            raise RuntimeError(f"planner failed: {p.error}")
        plan: PlannerOutput = p.output  # type: ignore[assignment]
        if plan.exhausted:
            record_event(session, run.id, "search_exhausted", plan.reasoning_summary, agent_id="planner_01")
            return "exhausted"

        parent = session.get(Experiment, plan.parent_experiment_id) if plan.parent_experiment_id else None
        diff = config_diff(parent.model_type if parent else None, parent.configuration if parent else None,
                           plan.model_type, plan.proposed_configuration)
        sequence = (session.scalar(select(func.count()).select_from(Experiment)
                                   .where(Experiment.run_id == run.id)) or 0) + 1
        exp = Experiment(
            id=new_id("exp"), run_id=run.id, namespace=run.namespace, sequence=sequence,
            parent_experiment_id=parent.id if parent else None, baseline_experiment_id=baseline.id if baseline else None,
            proposed_by="planner_01", objective=run.objective, hypothesis=plan.hypothesis,
            expected_effects=[e.model_dump() for e in plan.expected_effects], reasoning_summary=plan.reasoning_summary,
            plan_source=plan.plan_source, referenced_memory_ids=plan.referenced_memory_ids,
            model_type=plan.model_type, configuration=plan.proposed_configuration,
            config_hash=config_hash(plan.model_type, plan.proposed_configuration), config_diff=diff,
            moves=moves_from_diff(diff), status="planned", acceptance_threshold=self.settings.acceptance_threshold,
            seed=self.settings.random_seed, dataset_version=load_splits(self.settings.random_seed).version,
            protocol_version=runner.PROTOCOL_VERSION,
        )
        session.add(exp)
        session.flush()
        ctx.experiment_id = exp.id
        for aid in action_ids:
            session.get(AgentAction, aid).experiment_id = exp.id  # type: ignore[union-attr]
        self._send(ctx, "planner_01", "executor_01", "experiment_proposal", exp.id, plan.model_dump(mode="json"))
        transition(session, run, RunState.READY, f"Planned {exp.id}", experiment_id=exp.id)
        session.commit()

        # -- EXECUTING ---------------------------------------------------------
        transition(session, run, RunState.EXECUTING, f"Executing {exp.id}", experiment_id=exp.id)
        exp.status, exp.started_at = "running", utcnow()
        session.commit()
        record_event(session, run.id, "experiment_started", f"{exp.model_type} {exp.configuration}",
                     experiment_id=exp.id, agent_id="executor_01")
        session.commit()  # never hold SQLite's write lock while a model trains (stop requests must get through)
        artifact = {"experiment_id": exp.id, "run_id": run.id, "model_type": exp.model_type,
                    "configuration": exp.configuration, "parent_experiment_id": exp.parent_experiment_id,
                    "hypothesis": exp.hypothesis, "seed": exp.seed, "dataset_version": exp.dataset_version,
                    "protocol_version": exp.protocol_version, "proposed_by": exp.proposed_by}
        e = self.executor.run(ctx, ExecutorInput(
            experiment_id=exp.id, model_type=exp.model_type, configuration=exp.configuration, seed=exp.seed,
            timeout_seconds=max(1.0, min(self.settings.experiment_timeout_seconds, remaining)),
            max_retries=run.max_retries, parent_commit=parent.git_commit if parent else None, artifact=artifact,
            commit_message=f"{exp.id}: {', '.join(exp.moves) or 'initial'} — {exp.hypothesis}"))
        if e.status == "success":
            out = e.output
            status, metrics, error = out.status, out.metrics, out.error  # type: ignore[union-attr]
            exp.git_commit, exp.attempts = out.git_commit, out.attempts  # type: ignore[union-attr]
            if out.git_commit:  # type: ignore[union-attr]
                record_event(session, run.id, "git_commit", f"Committed config as {out.git_commit}",  # type: ignore[union-attr]
                             experiment_id=exp.id, agent_id="executor_01")
            if out.git_error:  # type: ignore[union-attr]
                record_event(session, run.id, "git_error", out.git_error, experiment_id=exp.id,  # type: ignore[union-attr]
                             level=logging.WARNING)
        else:
            status, metrics, error = "failed", None, e.error
        exp.executed_by = "executor_01" if exp.attempts else None
        exp.error = error
        exp.finished_at = utcnow()
        if status == "cancelled":
            exp.status = "cancelled"
            record_event(session, run.id, "experiment_cancelled", error or "cancelled", experiment_id=exp.id)
            transition(session, run, RunState.CANCELLED, "Stopped by user during execution", experiment_id=exp.id)
            session.commit()
            return "cancelled"
        record_event(session, run.id, "experiment_completed" if status == "completed" else "experiment_failed",
                     error or f"metrics: {metrics}", experiment_id=exp.id, agent_id="executor_01",
                     level=logging.INFO if status == "completed" else logging.WARNING)
        self._send(ctx, "executor_01", "evaluator_01", "execution_result", exp.id,
                   {"status": status, "metrics": metrics, "error": error, "attempts": exp.attempts,
                    "git_commit": exp.git_commit})

        # -- EVALUATING ----------------------------------------------------------
        transition(session, run, RunState.EVALUATING, f"Evaluating {exp.id}", experiment_id=exp.id)
        v = self.evaluator.run(ctx, EvaluatorInput(
            experiment_id=exp.id, status=status, metrics=metrics, error=error, baseline=snapshot,
            expected_effects=plan.expected_effects, threshold=self.settings.acceptance_threshold))
        if v.status != "success":
            raise RuntimeError(f"evaluator failed: {v.error}")
        ev = v.output
        exp.status = "completed" if status == "completed" else "failed"
        exp.decision, exp.decision_reason = ev.decision, ev.reason  # type: ignore[union-attr]
        exp.primary_delta, exp.secondary_deltas = ev.primary_delta, ev.secondary_deltas  # type: ignore[union-attr]
        exp.hypothesis_outcome, exp.hypothesis_details = ev.hypothesis_outcome, ev.hypothesis_details  # type: ignore[union-attr]
        record_event(session, run.id, "evaluation_decision", f"{ev.decision}: {ev.reason}",  # type: ignore[union-attr]
                     experiment_id=exp.id, agent_id="evaluator_01")
        self._send(ctx, "evaluator_01", "broadcast", "evaluation_decision", exp.id, ev.model_dump(mode="json"))  # type: ignore[union-attr]

        # -- STORING_MEMORY ----------------------------------------------------------
        transition(session, run, RunState.STORING_MEMORY, f"Storing memory for {exp.id}", experiment_id=exp.id)
        if metrics:
            finite = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
            store.save_metrics(exp.id, finite)
        if exp.decision == "accepted":
            store.set_baseline(exp, ev.reason)  # type: ignore[union-attr]
            record_event(session, run.id, "baseline_updated", f"Accepted baseline is now {exp.id}",
                         experiment_id=exp.id, data={"previous": baseline.id if baseline else None})
        self._store_memories(ctx, run, exp, ev.secondary_improvements, metrics,  # type: ignore[union-attr]
                             {"plan": p.action_id, "execute": e.action_id, "evaluate": v.action_id}, suggestion
                             if plan.plan_source == "research_suggestion" else None)
        self._send(ctx, "evaluator_01", "broadcast", "memory_stored", exp.id,
                   {"memory_count": store.count_memories()})
        session.flush()
        return "done"

    # ------------------------------------------------------------------
    # Memory + graph writes
    # ------------------------------------------------------------------
    def _store_memories(self, ctx: AgentContext, run: Run, exp: Experiment, improvements: list[dict[str, Any]],
                        metrics: dict[str, float] | None, actions: dict[str, str],
                        suggestion: ResearchSuggestion | None) -> None:
        store, session, ns = ctx.store, ctx.session, run.namespace
        vm = store.metrics_of(exp.id)
        f1 = vm.get("f1")
        f1_txt = f"{f1:.4f}" if f1 is not None else "n/a"
        delta_txt = f" ({exp.primary_delta:+.4f} vs. {exp.baseline_experiment_id})" if exp.primary_delta is not None else ""
        mem_txt = f" Peak memory {vm['peak_memory_mb']:.3f} MB." if "peak_memory_mb" in vm else ""
        moves = ", ".join(exp.moves) or "initial configuration"
        content = (f"Experiment {exp.id} ({exp.model_type}, {exp.configuration}; move: {moves}) {exp.status}. "
                   f"Validation F1 {f1_txt}{delta_txt}.{mem_txt} Decision: {exp.decision} — {exp.decision_reason} "
                   f"Hypothesis: {exp.hypothesis} Outcome: {exp.hypothesis_outcome}.")
        base_evidence = {"experiment_id": exp.id, "baseline_experiment_id": exp.baseline_experiment_id,
                         "metrics": vm, "decision": exp.decision, "config_hash": exp.config_hash,
                         "config_diff": exp.config_diff, "moves": exp.moves,
                         "plan_source": exp.plan_source}
        tags = [exp.model_type, exp.decision or "none", exp.status, "experiment", *exp.moves]
        created: list[MemoryRecord] = [store.add_memory(
            category="episodic", kind="experiment_outcome", title=f"{exp.id}: {exp.decision} {exp.model_type}",
            content=content, created_by="evaluator_01", verification_status="verified", evidence=base_evidence,
            tags=tags + (["failure"] if exp.status == "failed" else []), source_run_id=run.id,
            source_experiment_id=exp.id)]

        findings: list[tuple[str, str, dict[str, Any]]] = []
        d = exp.primary_delta
        if exp.decision == "invalid":
            findings.append(("failure", f"Configuration {exp.model_type} {exp.configuration} did not produce a valid "
                                        f"result: {exp.error or exp.decision_reason}", {"error": exp.error}))
        elif exp.baseline_experiment_id is None:
            findings.append(("baseline_established", f"Initial baseline {exp.id} measured validation F1 {f1_txt}.",
                             {"metrics": vm}))
        elif d is not None and exp.decision == "accepted":
            findings.append(("primary_improvement", f"Move {moves} improved validation F1 by {d:+.4f} relative to "
                                                    f"{exp.baseline_experiment_id}.", {"f1_delta": d}))
        elif d is not None and d < 0:
            findings.append(("regression", f"Move {moves} reduced validation F1 by {d:+.4f} relative to "
                                           f"{exp.baseline_experiment_id}.", {"f1_delta": d}))
        elif d is not None:
            findings.append(("no_primary_effect", f"Move {moves} changed validation F1 by {d:+.4f} (not above the "
                                                  f"acceptance threshold) relative to {exp.baseline_experiment_id}.",
                             {"f1_delta": d}))
        if exp.decision == "rejected" and improvements:
            parts = [f"{i['metric']} by {abs(i['delta']):.4g} ({i['relative']:+.0%})" for i in improvements]
            findings.append(("secondary_improvement",
                             f"Rejected experiment {exp.id} (move {moves}) reduced {', '.join(parts)} relative to "
                             f"{exp.baseline_experiment_id}, while validation F1 changed by {d:+.4f}. Measured "
                             f"association on one fixed split, not a causal claim.",
                             {"improvements": improvements, "f1_delta": d}))
        if exp.hypothesis_outcome and exp.hypothesis_outcome != "not_evaluated":
            findings.append(("hypothesis_outcome", f"Hypothesis of {exp.id} was {exp.hypothesis_outcome}: "
                                                   f"{exp.hypothesis}", {"details": exp.hypothesis_details}))

        finding_rows = []
        for ftype, statement, extra in findings:
            evidence = {**base_evidence, **extra}
            f = store.add_finding(run_id=run.id, experiment_id=exp.id, finding_type=ftype, statement=statement,
                                  evidence=evidence, created_by="evaluator_01")
            finding_rows.append(f)
            created.append(store.add_memory(
                category="semantic", kind=ftype, title=f"{ftype.replace('_', ' ')}: {exp.id}", content=statement,
                created_by="evaluator_01", verification_status="verified", evidence={**evidence, "finding_id": f.id},
                tags=tags + [ftype], source_run_id=run.id, source_experiment_id=exp.id, finding_id=f.id))

        if suggestion is not None:
            status = {"supported": "verified"}.get(exp.hypothesis_outcome or "", "contradicted")
            if exp.hypothesis_outcome == "not_evaluated":
                status = "unverified"
            store.set_verification(suggestion.memory_id, status,
                                   f"Tested by {exp.id}: hypothesis {exp.hypothesis_outcome}")
            record_event(session, run.id, "hypothesis_tested",
                         f"Research suggestion {suggestion.memory_id} -> {status} by {exp.id}", experiment_id=exp.id,
                         agent_id="evaluator_01")

        # -- graph relationships (validated against real records) --------------
        X = f"experiment:{exp.id}"
        rels: list[tuple[str, str, str, str]] = [
            ("agent:planner_01", "PROPOSED", X, f"action:{actions['plan']}"),
            ("agent:evaluator_01", "EVALUATED", X, f"action:{actions['evaluate']}"),
            (X, "PART_OF", f"run:{run.id}", X),
            (X, "USES", f"config:{exp.config_hash}", X),
            (X, "TRAINS", f"model:{exp.model_type}", X),
        ]
        if exp.executed_by:
            rels.append(("agent:executor_01", "EXECUTED", X, f"action:{actions['execute']}"))
        if exp.parent_experiment_id:
            rels.append((X, "DERIVED_FROM", f"experiment:{exp.parent_experiment_id}", X))
        if exp.baseline_experiment_id:
            rels.append((X, "COMPARED_AGAINST", f"experiment:{exp.baseline_experiment_id}", X))
        if exp.git_commit and self.workspace.commit_exists(exp.git_commit):
            rels.append((X, "RECORDED_IN", f"commit:{exp.git_commit}", X))
        for f in finding_rows:
            rels.append((X, "PRODUCED", f"finding:{f.id}", X))
            rels.append((f"finding:{f.id}", "SUPPORTED_BY", X, X))
        for m in created:
            rels.append((f"memory:{m.id}", "REFERS_TO", X, f"memory:{m.id}"))
        for mem_id in exp.referenced_memory_ids:
            rels.append((f"memory:{mem_id}", "INFLUENCED", X, f"action:{actions['plan']}"))
        if suggestion is not None:
            rels.append((f"memory:{suggestion.memory_id}", "REFERS_TO", f"experiment:{suggestion.source_experiment_id}",
                         f"memory:{suggestion.memory_id}"))
        for src, rel, tgt, evidence in rels:
            try:
                add_relationship(session, ns, src, rel, tgt, evidence)
            except GraphEvidenceError as exc:  # never invent a relationship
                record_event(session, run.id, "graph_relationship_rejected", str(exc), experiment_id=exp.id,
                             level=logging.WARNING)

    def _complete(self, ctx: AgentContext, run: Run, reason: str) -> None:
        run.stop_reason = reason
        self._finalize(ctx, run)
        transition(ctx.session, run, RunState.COMPLETED, reason)

    # ------------------------------------------------------------------
    def _finalize(self, ctx: AgentContext, run: Run) -> None:
        """Final held-out test evaluation of the namespace's accepted baseline.

        The test split is touched only here, after all selection decisions."""
        baseline = ctx.store.get_baseline()
        run.end_baseline_experiment_id = baseline.id if baseline else None
        if baseline is None or ctx.store.metrics_of(baseline.id, "test"):
            return
        ctx.session.commit()  # release the write lock before training
        outcome = runner.execute(baseline.model_type, baseline.configuration, baseline.seed, split="test",
                                 timeout=self.settings.experiment_timeout_seconds, mode=self.settings.execution_mode)  # type: ignore[arg-type]
        if outcome.status == "completed" and outcome.metrics:
            ctx.store.save_metrics(baseline.id, {k: outcome.metrics[k] for k in
                                                 ("accuracy", "f1", "precision", "recall")}, split="test")
            record_event(ctx.session, run.id, "final_test_evaluation",
                         f"Held-out test F1 of baseline {baseline.id}: {outcome.metrics['f1']:.4f}",
                         experiment_id=baseline.id)
        else:
            record_event(ctx.session, run.id, "final_test_evaluation_failed", outcome.error or outcome.status,
                         experiment_id=baseline.id, level=logging.WARNING)


def build_summary(session: Session, run: Run, workspace: GitWorkspace | None = None) -> dict[str, Any]:
    """Run summary computed from stored records only."""
    exps = session.scalars(select(Experiment).where(Experiment.run_id == run.id).order_by(Experiment.sequence)).all()
    store = MemoryStore(session, run.namespace)
    seen_before = set(session.scalars(select(Experiment.config_hash).where(
        Experiment.namespace == run.namespace, Experiment.created_at < run.created_at)))
    repeated = 0
    for e in exps:
        if e.config_hash in seen_before:
            repeated += 1
        seen_before.add(e.config_hash)
    f1s = {e.id: store.metrics_of(e.id).get("f1") for e in exps}
    valid = [(v, k) for k, v in f1s.items() if v is not None]
    actions = session.scalars(select(AgentAction).where(AgentAction.run_id == run.id)).all()
    retrievals = sum(a.retrieval_count for a in actions)
    retrieval_ms = sum(a.retrieval_ms for a in actions)
    reused = sorted({m for e in exps for m in e.referenced_memory_ids})
    llm_usage = [a.output["llm_usage"] for a in actions if a.output and a.output.get("llm_usage")]
    baseline = store.get_baseline()
    pending = session.scalars(select(MemoryRecord).where(
        MemoryRecord.namespace == run.namespace, MemoryRecord.kind == "research_suggestion",
        MemoryRecord.verification_status == "unverified")).all()
    durations = [(e.finished_at - e.started_at).total_seconds() for e in exps if e.started_at and e.finished_at]
    unfinished = [f"experiment {e.id} is {e.status}" for e in exps if e.status in ("planned", "running", "interrupted",
                                                                                  "cancelled")]
    unfinished += [f"untested hypothesis {m.id}: {m.content[:120]}" for m in pending]
    return {
        "experiments": len(exps),
        "accepted": sum(e.decision == "accepted" for e in exps),
        "rejected": sum(e.decision == "rejected" for e in exps),
        "invalid": sum(e.decision == "invalid" for e in exps),
        "best_validation_f1": max(valid)[0] if valid else None,
        "best_experiment_id": max(valid)[1] if valid else None,
        "final_baseline_experiment_id": baseline.id if baseline else None,
        "final_baseline_validation_f1": store.metrics_of(baseline.id).get("f1") if baseline else None,
        "final_baseline_test_metrics": store.metrics_of(baseline.id, "test") if baseline else {},
        "repeated_configurations": repeated,
        "prior_findings_reused": len(reused),
        "reused_memory_ids": reused,
        "memory_retrievals": retrievals,
        "memory_retrieval_ms_total": round(retrieval_ms, 3),
        "memory_retrieval_ms_mean": round(retrieval_ms / retrievals, 3) if retrievals else 0.0,
        "experiment_seconds_total": round(sum(durations), 3),
        "hypotheses": {o: sum(e.hypothesis_outcome == o for e in exps)
                       for o in ("supported", "partially_supported", "not_supported")},
        "research_followups": [{"experiment_id": e.id, "parent_experiment_id": e.parent_experiment_id,
                                "hypothesis_outcome": e.hypothesis_outcome, "decision": e.decision}
                               for e in exps if e.plan_source == "research_suggestion"],
        "llm_usage": {"calls": len(llm_usage),
                      "input_tokens": sum(u.get("input_tokens", 0) for u in llm_usage),
                      "output_tokens": sum(u.get("output_tokens", 0) for u in llm_usage),
                      "estimated_cost_usd": round(sum(u.get("estimated_cost_usd", 0) for u in llm_usage), 6)},
        "stop_reason": run.stop_reason,
        "unfinished": unfinished,
    }
