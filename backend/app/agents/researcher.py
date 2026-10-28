"""Researcher agent: mines rejected experiments for useful secondary findings.

Policy: find verified ``secondary_improvement`` findings from rejected
experiments (ranked by memory reduction) that have not been followed up yet.
For the best one, keep the change that saved resources and add one more
bounded move that does not touch the same parameter, preferring moves whose
past F1 effect was non-negative and that are not expected to grow the model.
The output is a *hypothesis* (stored as unverified semantic memory), not a result.

Limits: only memory or model-size savings count (timing is too noisy); only rejected experiments that lost at most ``MAX_F1_LOSS`` validation
F1 qualify, and research follow-ups are never followed up again (no chains).
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from app.agents.base import Agent, AgentContext
from app.experiments.search_space import UPSIZING_MOVES, neighbours
from app.memory.retrieval import MemoryQuery, MetricGoal
from app.schemas.agents import ExpectedEffect, ResearcherInput, ResearcherOutput, ResearchSuggestion


MAX_F1_LOSS = 0.02


def _fmt(v: Any) -> str:
    return "unlimited" if v is None else str(v)


class ResearcherAgent(Agent):
    agent_id = "researcher_01"
    name = "Researcher"
    role = "researcher"
    description = ("Searches rejected experiments for verified secondary improvements (e.g. lower memory) and "
                   "turns them into new, evidence-linked hypotheses for the Planner.")
    tools = ("memory.search", "memory.tried_configurations", "memory.move_evidence", "memory.write_suggestion")
    input_schema = ResearcherInput
    output_schema = ResearcherOutput
    action_type = "research"

    def execute(self, ctx: AgentContext, inp: ResearcherInput) -> ResearcherOutput:
        results = ctx.retriever.search(MemoryQuery(
            text="rejected experiment reduced memory", kinds=["secondary_improvement"], decision="rejected",
            verification_status=["verified"], include_global=False, require_metric_goal=False,
            metric_goal=MetricGoal(metric="peak_memory_mb", direction="decrease"), limit=10,
        ))
        examined = [r.memory.id for r in results]
        if not results:
            return ResearcherOutput(suggestion=None, examined_memory_ids=[],
                                    summary="No rejected experiment with a verified secondary improvement exists.")
        followed = ctx.retriever.followed_up_sources()
        tried = ctx.retriever.tried_configurations()
        move_ev = ctx.retriever.move_evidence()

        for res in results:
            ev = res.evidence
            source_id = ev.get("experiment_id")
            if not source_id or source_id in followed or not isinstance(ev.get("configuration"), dict):
                continue
            if res.memory.evidence.get("plan_source") == "research_suggestion":
                continue  # no chains of follow-ups
            if ev.get("f1_delta") is None or ev["f1_delta"] < -MAX_F1_LOSS:
                continue
            source_diff = res.memory.evidence.get("config_diff", {})
            if "model_type" in source_diff:
                continue  # a model switch is not a "keep this change" refinement
            touched = set(source_diff)
            options = []
            for idx, cand in enumerate(neighbours(ev["model_type"], ev["configuration"])):
                if "model_type" in cand.diff or touched & set(cand.diff) or cand.config_hash in tried:
                    continue
                deltas = [e["f1_delta"] for m in cand.moves for e in move_ev.get(m, [])]
                past = mean(deltas) if deltas else 0.0
                upsizing = any(m in UPSIZING_MOVES for m in cand.moves)
                options.append((past < 0, upsizing, -past, idx, cand, deltas))
            if not options:
                continue
            options.sort(key=lambda o: o[:4])
            *_, cand, deltas = options[0]
            kept = ", ".join(f"{p}={_fmt(c['to'])}" for p, c in source_diff.items()) or "its configuration"
            added = ", ".join(f"{p} {_fmt(c['from'])}→{_fmt(c['to'])}" for p, c in cand.diff.items())
            # Timing is too noisy to steer research; only memory / model-size savings qualify.
            improvements = [i for i in res.memory.evidence.get("improvements") or []
                            if i.get("metric") in ("peak_memory_mb", "model_size_kb")]
            best = min(improvements, key=lambda i: i.get("relative", 0), default=None)
            if best is None:
                continue
            metric, f1_delta = best["metric"], ev.get("f1_delta")
            saved = ", ".join(f"{i['metric']} by {abs(i['delta']):.4g} ({i['relative']:+.0%})" for i in improvements)
            hypothesis = (
                f"Rejected experiment {source_id} changed cross-validated F1 by {f1_delta:+.4f} but reduced {saved}. "
                f"Keeping {kept} and additionally changing {added} may preserve the lower {metric} while keeping "
                f"cross-validated F1 at least at the baseline level."
            )
            evidence = {"source_experiment_id": source_id, "source_memory_id": res.memory.id,
                        "f1_delta": f1_delta, "improvements": improvements,
                        "memory_delta_mb": ev.get("memory_delta_mb"),
                        "model_size_delta_kb": ev.get("model_size_delta_kb"),
                        "added_move_past_f1_deltas": deltas}
            effects = [ExpectedEffect(metric="f1", direction="no_worse"),
                       ExpectedEffect(metric=metric, direction="decrease")]
            record = ctx.store.add_memory(
                category="semantic", kind="research_suggestion", title=f"Hypothesis from rejected {source_id}",
                content=hypothesis, created_by=self.agent_id, verification_status="unverified",
                evidence={**evidence, "proposed_model_type": cand.model_type,
                          "proposed_configuration": cand.configuration, "config_hash": cand.config_hash,
                          "expected_effects": [e.model_dump() for e in effects]},
                tags=["hypothesis", "research", "memory", cand.model_type], source_run_id=ctx.run_id,
                source_experiment_id=source_id,
            )
            suggestion = ResearchSuggestion(
                memory_id=record.id, source_experiment_id=source_id, source_memory_ids=[res.memory.id],
                hypothesis=hypothesis, model_type=cand.model_type, configuration=cand.configuration,
                config_hash=cand.config_hash, expected_effects=effects, evidence=evidence,
            )
            return ResearcherOutput(suggestion=suggestion, examined_memory_ids=examined,
                                    summary=f"Suggested follow-up of rejected experiment {source_id}.")
        return ResearcherOutput(suggestion=None, examined_memory_ids=examined,
                                summary="All rejected experiments with secondary improvements were already "
                                        "followed up or have no untested refinement.")
