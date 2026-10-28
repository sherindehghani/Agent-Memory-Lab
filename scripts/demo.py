#!/usr/bin/env python
"""Reproducible end-to-end demo of the AI Agent Memory Lab (real measurements only).

    python scripts/demo.py            # continue in the configured lab database
    python scripts/demo.py --reset    # wipe data/lab.db and workspace/ first (fresh demo)
    python scripts/demo.py --compare  # also run the memory on/off comparison

Steps: initialise DB + Git workspace -> initial baseline -> candidate experiments
-> accept/reject -> store memories -> find a rejected experiment with a useful
secondary result (if one exists) -> Researcher hypothesis -> follow-up
experiment -> knowledge graph -> final summary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select  # noqa: E402

from app.core.config import PROJECT_ROOT, get_settings  # noqa: E402
from app.core.database import new_session  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.memory import graph as kg  # noqa: E402
from app.memory.retrieval import MemoryRetriever, parse_natural_query  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.models import EvaluationResult, Experiment, Finding, MemoryRecord, Run  # noqa: E402
from app.orchestration.evaluation import start_comparison  # noqa: E402
from app.orchestration.manager import RunManager  # noqa: E402
from app.orchestration.recovery import bootstrap  # noqa: E402
from app.schemas.api import CompareRequest, RunCreate  # noqa: E402


def step(n: int, text: str) -> None:
    print(f"\n[{n}] {text}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the lab database and workspace first")
    parser.add_argument("--experiments", type=int, default=7, help="experiment budget for the demo run")
    parser.add_argument("--seed", type=int, default=0, help="planner policy seed")
    parser.add_argument("--compare", action="store_true", help="also run the memory on/off evaluation")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", json_logs=False)
    if args.reset:
        db_path = Path(settings.database_url.removeprefix("sqlite:///")).resolve()
        ws = Path(settings.workspace_dir).resolve()
        for p in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
            if p.is_file() and PROJECT_ROOT in p.parents:
                p.unlink()
        if ws.is_dir() and PROJECT_ROOT in ws.parents:
            shutil.rmtree(ws)
        print(f"Reset: removed {db_path.name} and {ws.name}/")

    step(1, "Initialise database (Alembic migrations), Git workspace, agents and procedural rules")
    bootstrap(settings)
    print(f"    database: {settings.database_url}\n    workspace: {settings.workspace_dir}"
          f"\n    execution mode: {settings.execution_mode}")

    manager = RunManager(settings)
    with new_session() as s:
        existing = MemoryStore(s).get_baseline()
    step(2, "Create the initial baseline" if existing is None else
         f"Accepted baseline already exists ({existing.id}); the run continues from it")

    step(3, f"Run up to {args.experiments} experiments with memory enabled (Planner, Executor, Evaluator, Researcher)")
    run_id = manager.launch(RunCreate(max_experiments=args.experiments, seed=args.seed, memory_enabled=True),
                            background=False)

    with new_session() as s:
        run = s.get(Run, run_id)
        exps = s.scalars(select(Experiment).where(Experiment.run_id == run_id).order_by(Experiment.sequence)).all()
        store = MemoryStore(s)
        step(4, "Accept/reject decisions (5-fold cross-validated F1 on the development split)")
        for e in exps:
            f1 = store.metrics_of(e.id).get("f1")
            mem = store.metrics_of(e.id).get("peak_memory_mb")
            print(f"    #{e.sequence} {e.id} [{e.plan_source}] {e.model_type} moves={e.moves or ['initial']}"
                  f"\n        F1={f1 if f1 is None else round(f1, 4)} peak_mem={mem if mem is None else round(mem, 4)}MB"
                  f" -> {e.decision}: {e.decision_reason}")

        n_mem = s.scalar(select(MemoryRecord.id).where(MemoryRecord.source_run_id == run_id).limit(1))
        step(5, f"Memories stored: {len(s.scalars(select(MemoryRecord).where(MemoryRecord.source_run_id == run_id)).all())}"
                f" records for this run (episodic outcomes, semantic findings, research hypotheses)" if n_mem else
             "No memories stored")

        step(6, 'Second agent query: "Find rejected experiments that reduced memory use"')
        results = MemoryRetriever(s).search(parse_natural_query("Find rejected experiments that reduced memory use"))
        if not results:
            print("    None found. No rejected experiment in this lab reduced memory use — nothing is fabricated.")
        for r in results[:5]:
            ev = r.evidence
            print(f"    {r.memory.id} -> {ev['experiment_id']}: F1 delta {ev['f1_delta']:+.4f}, "
                  f"peak memory delta {ev['memory_delta_mb']:+.4f} MB  ({'; '.join(r.reasons)})")

        step(7, "Researcher hypotheses and follow-up experiments")
        followups = [e for e in exps if e.plan_source == "research_suggestion"]
        if not followups:
            print("    No follow-up ran in this run (no eligible rejected experiment with a secondary improvement,"
                  " or the budget ended first).")
        for e in followups:
            sugg = next((s.get(MemoryRecord, m) for m in e.referenced_memory_ids
                         if (rec := s.get(MemoryRecord, m)) and rec.kind == "research_suggestion"), None)
            print(f"    {e.id} derived from rejected {e.parent_experiment_id}")
            print(f"        hypothesis: {e.hypothesis}")
            print(f"        cited memories: {e.referenced_memory_ids}")
            print(f"        outcome: {e.hypothesis_outcome} (decision {e.decision}); "
                  f"suggestion {sugg.id if sugg else '?'} is now '{sugg.verification_status if sugg else '?'}'")

        step(8, "Knowledge graph (built from persisted relationships)")
        graph, dangling = kg.build_graph(s, "main")
        print(f"    {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {len(dangling)} dangling")
        finding = s.scalars(select(Finding).where(Finding.run_id == run_id,
                                                  Finding.finding_type == "secondary_improvement")).first()
        if finding:
            path = kg.evidence_path(graph, finding.id)
            print(f"    Evidence path for finding {finding.id}: \"{finding.statement[:110]}...\"")
            for p in path["paths"][:3]:
                print("      " + " ".join(f"{st['source'].split(':')[0]} -{st['relation']}-> {st['target']}" for st in p))

        step(9, "Run summary")
        summary = {k: v for k, v in (run.summary or {}).items() if k not in ("reused_memory_ids",)}
        print("    " + json.dumps({"state": run.state, **summary}, indent=2, default=str).replace("\n", "\n    "))

    if args.compare:
        step(10, "Memory evaluation: memory-enabled vs memory-disabled (seeds 0,1,2; budget 6)")
        eval_id = start_comparison(manager, CompareRequest(seeds=[0, 1, 2], budget=6), background=False)
        with new_session() as s:
            row = s.get(EvaluationResult, eval_id)
            print("    " + json.dumps(row.results["aggregate"] if row.results else row.error, indent=2)
                  .replace("\n", "\n    "))
    print("\nOpen the dashboard (make backend + make frontend) to inspect everything above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
