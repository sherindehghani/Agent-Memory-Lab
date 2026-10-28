from __future__ import annotations

import json
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.evaluator import EvaluatorAgent
from app.agents.executor import ExecutorAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.autoresearch.agents import AutoresearcherAgent
from app.core.config import Settings, get_settings
from app.core.database import init_engine, run_migrations, session_scope
from app.experiments.acceptance import SECONDARY_METRICS
from app.experiments.datasets import load_splits
from app.experiments.runner import PROTOCOL_VERSION
from app.experiments.search_space import search_space_description
from app.experiments.versioning import GitWorkspace
from app.memory.store import MemoryStore
from app.models import Agent, Experiment, Run
from app.orchestration.state import ACTIVE, RunState, record_event, transition

logger = logging.getLogger(__name__)

AGENT_CLASSES = (PlannerAgent, ExecutorAgent, EvaluatorAgent, ResearcherAgent, AutoresearcherAgent)


def seed_agents(session: Session) -> None:
    for cls in AGENT_CLASSES:
        agent = session.get(Agent, cls.agent_id)
        if agent is None:
            session.add(Agent(id=cls.agent_id, name=cls.name, role=cls.role, description=cls.description,
                              tools=list(cls.tools)))
        else:
            agent.name, agent.role, agent.description, agent.tools = cls.name, cls.role, cls.description, list(cls.tools)
    session.flush()


def seed_procedural_memory(session: Session, settings: Settings) -> None:
    store = MemoryStore(session)
    rules = [
        ("acceptance_rule", "Experiment acceptance rule",
         f"Optimise validation F1. A candidate is accepted only if it completed, its metrics are valid, and "
         f"validation F1 exceeds the accepted baseline by more than {settings.acceptance_threshold}. Ties are "
         f"rejected. Failed or invalid results never replace the baseline. Lower resource use is recorded as a "
         f"secondary improvement when it beats the baseline by the per-metric relative margin.",
         {"primary_metric": "f1", "threshold": settings.acceptance_threshold,
          "secondary_margins": {k: v[1] for k, v in SECONDARY_METRICS.items()}}),
        ("evaluation_protocol", "Evaluation protocol",
         f"Protocol {PROTOCOL_VERSION}: fixed stratified 80/20 development/test split of the Breast Cancer "
         f"Wisconsin dataset (seed {settings.random_seed}); decisions use mean binary F1 over stratified 5-fold "
         f"cross-validation on the development data. The held-out test split is only used for the final "
         f"evaluation of a run's final baseline.",
         {"protocol_version": PROTOCOL_VERSION, "dataset_version": load_splits(settings.random_seed).version}),
        ("execution_constraints", "Agent execution constraints",
         "Agents propose structured configurations only; no generated code is executed. Supported models: "
         "logistic_regression, random_forest. Each experiment runs with a timeout and file writes are confined "
         "to the workspace. The LLM (optional) can only propose configurations and never decides acceptance.",
         {"experiment_timeout_seconds": settings.experiment_timeout_seconds,
          "execution_mode": settings.execution_mode}),
        ("retry_limits", "Run budgets and retry limits",
         f"Default budgets: {settings.max_experiments} experiments per run, {settings.max_run_seconds}s per run, "
         f"{settings.max_retries} retry per failed experiment. Runs never execute unbounded.",
         {"max_experiments": settings.max_experiments, "max_run_seconds": settings.max_run_seconds,
          "max_retries": settings.max_retries}),
        ("allowed_changes", "Allowed configuration changes",
         "Bounded hyperparameter ranges: " + json.dumps(search_space_description(), default=str),
         {"search_space": search_space_description()}),
    ]
    for kind, title, content, evidence in rules:
        store.upsert_procedural(kind, title, content, evidence, tags=["procedural", "policy", kind])
    session.flush()


def recover_interrupted_runs(session: Session) -> list[str]:
    recovered = []
    for run in session.scalars(select(Run).where(Run.state.in_([s.value for s in ACTIVE]))):
        previous = run.state
        for exp in session.scalars(select(Experiment).where(Experiment.run_id == run.id,
                                                            Experiment.status.in_(["planned", "running"]))):
            exp.status = "interrupted"
            exp.error = "process stopped before the experiment finished"
        run.stop_reason = f"process stopped while {previous}"
        transition(session, run, RunState.INTERRUPTED, f"Recovered on startup (was {previous})")
        record_event(session, run.id, "recovery", "Marked interrupted; completed records and the accepted "
                                                  "baseline were preserved. No experiment is re-run automatically.",
                     data={"previous_state": previous})
        recovered.append(run.id)
    return recovered


def bootstrap(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    run_migrations(settings.database_url)
    init_engine(settings.database_url)
    GitWorkspace(settings.workspace_dir).init()
    with session_scope() as session:
        seed_agents(session)
        seed_procedural_memory(session, settings)
        recovered = recover_interrupted_runs(session)
    if recovered:
        logger.warning("recovered interrupted runs: %s", recovered, extra={"event_type": "recovery"})
    return recovered
