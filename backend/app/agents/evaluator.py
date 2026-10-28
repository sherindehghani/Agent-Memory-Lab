"""Evaluator agent: applies the deterministic acceptance rule (never an LLM)."""

from __future__ import annotations

from app.agents.base import Agent, AgentContext
from app.experiments.acceptance import check_hypothesis, decide
from app.schemas.agents import EvaluatorInput, EvaluatorOutput


class EvaluatorAgent(Agent):
    agent_id = "evaluator_01"
    name = "Evaluator"
    role = "evaluator"
    description = ("Compares a result with the accepted baseline using fixed rules, records metric deltas, "
                   "accepts or rejects, preserves secondary improvements and checks the stated hypothesis.")
    tools = ("acceptance.decide", "acceptance.check_hypothesis")
    input_schema = EvaluatorInput
    output_schema = EvaluatorOutput
    action_type = "evaluate"

    def execute(self, ctx: AgentContext, inp: EvaluatorInput) -> EvaluatorOutput:
        baseline_metrics = inp.baseline.metrics if inp.baseline else None
        d = decide(inp.status, inp.metrics, baseline_metrics, inp.threshold, inp.error)
        valid = d.decision != "invalid"
        outcome, details = check_hypothesis(
            [e.model_dump() for e in inp.expected_effects], inp.metrics if valid else None, baseline_metrics
        )
        return EvaluatorOutput(
            decision=d.decision, reason=d.reason, primary_delta=d.primary_delta, secondary_deltas=d.secondary_deltas,
            secondary_improvements=d.secondary_improvements, threshold=d.threshold,
            hypothesis_outcome=outcome, hypothesis_details=details,  # type: ignore[arg-type]
        )
