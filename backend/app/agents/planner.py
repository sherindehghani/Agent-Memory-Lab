from __future__ import annotations

from statistics import mean
from typing import Any

from app.agents.base import Agent, AgentContext
from app.experiments.search_space import (
    DOWNSIZING_MOVES,
    Candidate,
    config_hash,
    neighbours,
    validate_configuration,
)
from app.schemas.agents import ExpectedEffect, PlannerInput, PlannerOutput


def _fmt(v: Any) -> str:
    return "unlimited" if v is None else str(v)


def hypothesis_for(cand: Candidate) -> tuple[str, list[ExpectedEffect]]:
    if "model_type" in cand.diff:
        change = cand.diff["model_type"]
        return (f"Switching from {change['from']} to {change['to']} (default settings) may improve validation F1.",
                [ExpectedEffect(metric="f1", direction="increase")])
    change_txt = ", ".join(f"{p} {_fmt(c['from'])}→{_fmt(c['to'])}" for p, c in cand.diff.items())
    if any(m in DOWNSIZING_MOVES for m in cand.moves):
        return (f"Changing {change_txt} may reduce peak training memory while keeping validation F1 at least "
                f"as high as the baseline.",
                [ExpectedEffect(metric="f1", direction="no_worse"),
                 ExpectedEffect(metric="peak_memory_mb", direction="decrease")])
    return f"Changing {change_txt} may improve validation F1.", [ExpectedEffect(metric="f1", direction="increase")]


class PlannerAgent(Agent):
    agent_id = "planner_01"
    name = "Planner"
    role = "planner"
    description = ("Reads the objective and accepted baseline, retrieves relevant past experiments (when memory is "
                   "enabled) and proposes the next bounded configuration with a hypothesis and cited memories.")
    tools = ("memory.tried_configurations", "memory.move_evidence", "search_space.neighbours", "llm.propose")
    input_schema = PlannerInput
    output_schema = PlannerOutput
    action_type = "plan"

    def execute(self, ctx: AgentContext, inp: PlannerInput) -> PlannerOutput:
        if inp.baseline is None:
            cfg = validate_configuration(inp.initial_model_type, inp.initial_configuration)
            return PlannerOutput(
                hypothesis=f"Establish the accepted baseline for '{inp.objective}' with the initial configuration.",
                model_type=inp.initial_model_type, proposed_configuration=cfg, parent_experiment_id=None,
                reasoning_summary="No accepted baseline exists in this namespace, so the initial configuration is "
                                  "run first to create one.",
                plan_source="initial_baseline",
            )
        fallback_note = ""
        if inp.mode == "llm":
            if ctx.llm is None:
                fallback_note = "LLM mode requested but no LLM provider is configured; used deterministic policy. "
            else:
                from app.agents.llm import llm_plan

                try:
                    return llm_plan(ctx, inp)
                except Exception as exc:
                    fallback_note = f"LLM proposal rejected ({type(exc).__name__}: {exc}); used deterministic policy. "
        out = self._memory_policy(ctx, inp) if inp.memory_enabled else self._memoryless_policy(ctx, inp)
        out.reasoning_summary = fallback_note + out.reasoning_summary
        return out

    def _memoryless_policy(self, ctx: AgentContext, inp: PlannerInput) -> PlannerOutput:
        base = inp.baseline
        assert base is not None
        cands = neighbours(base.model_type, base.configuration)
        cand = ctx.rng.choice(cands)
        hypothesis, effects = hypothesis_for(cand)
        return PlannerOutput(
            hypothesis=hypothesis, model_type=cand.model_type, proposed_configuration=cand.configuration,
            parent_experiment_id=base.experiment_id, expected_effects=effects,
            reasoning_summary=(f"Memory disabled: picked move {', '.join(cand.moves)} at random (seeded) from "
                               f"{len(cands)} bounded neighbours of baseline {base.experiment_id}; no earlier "
                               f"experiments were consulted."),
            plan_source="policy",
        )

    def _memory_policy(self, ctx: AgentContext, inp: PlannerInput) -> PlannerOutput:
        base = inp.baseline
        assert base is not None
        tried = ctx.retriever.tried_configurations()
        sugg = inp.research_suggestion
        if sugg is not None and config_hash(sugg.model_type, sugg.configuration) not in tried:
            return PlannerOutput(
                hypothesis=sugg.hypothesis, model_type=sugg.model_type, proposed_configuration=sugg.configuration,
                parent_experiment_id=sugg.source_experiment_id, expected_effects=sugg.expected_effects,
                referenced_memory_ids=[sugg.memory_id, *sugg.source_memory_ids],
                reasoning_summary=(f"Memory enabled: adopted Researcher suggestion {sugg.memory_id}, derived from "
                                   f"rejected experiment {sugg.source_experiment_id} "
                                   f"(evidence: {sugg.evidence}). The configuration has not been tested before."),
                plan_source="research_suggestion",
            )

        evidence = ctx.retriever.move_evidence()
        cands = neighbours(base.model_type, base.configuration)
        skipped_ids: list[str] = []
        scored: list[tuple[float, Candidate, list[dict[str, Any]]]] = []
        for cand in cands:
            if cand.config_hash in tried:
                skipped_ids.extend(tried[cand.config_hash])
                continue
            ev = [e for m in cand.moves for e in evidence.get(m, [])]
            score = mean(e["f1_delta"] for e in ev) if ev else 0.0
            scored.append((score, cand, ev))
        parent_id = base.experiment_id
        if not scored:
            scored, parent_id, via = self._second_step(ctx, cands, tried, evidence)
            if via:
                skipped_ids.append(via)
        if not scored:
            return PlannerOutput(
                hypothesis="No untested configuration within two bounded steps of the baseline remains.",
                model_type=base.model_type, proposed_configuration=base.configuration,
                parent_experiment_id=base.experiment_id, referenced_memory_ids=sorted(set(skipped_ids))[:20],
                reasoning_summary=f"Memory enabled: every configuration within two bounded steps of "
                                  f"{base.experiment_id} already has a recorded outcome; stopping instead of "
                                  f"repeating experiments.",
                plan_source="policy", exhausted=True,
            )
        best = max(s for s, _, _ in scored)
        ties = [(s, c, e) for s, c, e in scored if s == best]
        score, cand, ev = ctx.rng.choice(ties)
        hypothesis, effects = hypothesis_for(cand)
        parts = []
        if parent_id != base.experiment_id:
            parts.append(f"all {len(cands)} one-step neighbours were already tested, so the search extends one "
                         f"more step from tested experiment {parent_id}")
        elif skipped_ids:
            parts.append(f"skipped {len(cands) - len(scored)} neighbour(s) already tested")
        if ev:
            parts.append(f"move {', '.join(cand.moves)} has mean measured F1 delta {score:+.4f} across "
                         f"{len(ev)} past experiment(s)")
        else:
            parts.append(f"no past evidence for move {', '.join(cand.moves)}; chosen among {len(ties)} "
                         f"equally ranked untested moves (seeded tie-break)")
        negatives = sum(1 for s, _, _ in scored if s < 0)
        if negatives:
            parts.append(f"{negatives} candidate move(s) ranked lower because past F1 deltas were negative")
        referenced = list(dict.fromkeys([e["memory_id"] for e in ev] + skipped_ids))[:20]
        return PlannerOutput(
            hypothesis=hypothesis, model_type=cand.model_type, proposed_configuration=cand.configuration,
            parent_experiment_id=parent_id, expected_effects=effects, referenced_memory_ids=referenced,
            reasoning_summary="Memory enabled: " + "; ".join(parts) + ".", plan_source="policy",
        )

    @staticmethod
    def _second_step(ctx: AgentContext, first: list[Candidate], tried: dict[str, list[str]],
                     evidence: dict[str, list[dict[str, Any]]]
                     ) -> tuple[list[tuple[float, Candidate, list[dict[str, Any]]]], str | None, str | None]:
        tested = [t for c in first if (t := ctx.retriever.tested_experiment(c.config_hash))]
        tested.sort(key=lambda t: -(t["f1_delta"] if t["f1_delta"] is not None else -1.0))
        for t in tested:
            scored = []
            for cand in neighbours(t["model_type"], t["configuration"]):
                if cand.config_hash in tried:
                    continue
                ev = [e for m in cand.moves for e in evidence.get(m, [])]
                scored.append((mean(e["f1_delta"] for e in ev) if ev else 0.0, cand, ev))
            if scored:
                return scored, t["experiment_id"], t["memory_id"]
        return [], None, None
