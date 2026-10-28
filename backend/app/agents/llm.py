from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.config import Settings
from app.experiments.search_space import config_hash, search_space_description, validate_configuration
from app.memory.retrieval import MemoryQuery
from app.schemas.agents import PlannerOutput

if TYPE_CHECKING:
    from app.agents.base import AgentContext
    from app.schemas.agents import PlannerInput

PRICING = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "model_type": {"type": "string", "enum": ["logistic_regression", "random_forest"]},
        "configuration_json": {"type": "string", "description": "JSON object of hyperparameters"},
        "referenced_memory_ids": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary": {"type": "string"},
    },
    "required": ["hypothesis", "model_type", "configuration_json", "referenced_memory_ids", "reasoning_summary"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You plan the next machine-learning experiment for an automated lab. Propose exactly one configuration "
    "inside the allowed search space that is likely to improve validation F1 over the accepted baseline. "
    "Prefer configurations that have not been tested. Cite only memory ids that appear in the provided "
    "memories. Keep reasoning_summary to two sentences."
)


@dataclass
class LLMClient:
    model: str
    api_key: str
    timeout: float

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout, max_retries=2)
        response = client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request")
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                 "model": self.model}
        price = PRICING.get(self.model)
        if price:
            usage["estimated_cost_usd"] = round(
                (usage["input_tokens"] * price[0] + usage["output_tokens"] * price[1]) / 1e6, 6)
        return json.loads(text), usage


def build_llm_client(settings: Settings) -> LLMClient | None:
    if settings.llm_provider != "anthropic" or settings.anthropic_api_key is None:
        return None
    return LLMClient(settings.llm_model, settings.anthropic_api_key.get_secret_value(),
                     settings.llm_timeout_seconds)


def llm_plan(ctx: AgentContext, inp: PlannerInput) -> PlannerOutput:
    base = inp.baseline
    assert base is not None
    memories: list[dict[str, Any]] = []
    if inp.memory_enabled:
        results = ctx.retriever.search(MemoryQuery(kinds=["experiment_outcome", "secondary_improvement",
                                                          "research_suggestion"], include_global=False, limit=15))
        memories = [{"memory_id": r.memory.id, "content": r.memory.content, "evidence": r.evidence}
                    for r in results]
    user = json.dumps({
        "objective": inp.objective,
        "accepted_baseline": base.model_dump(),
        "allowed_search_space": search_space_description(),
        "memories": memories,
        "research_suggestion": inp.research_suggestion.model_dump() if inp.research_suggestion else None,
    }, default=str)
    data, usage = ctx.llm.complete_json(SYSTEM_PROMPT, user)
    configuration = validate_configuration(data["model_type"], json.loads(data["configuration_json"]))
    known = {m["memory_id"] for m in memories}
    cited = [m for m in data.get("referenced_memory_ids", []) if m in known]
    tried = ctx.retriever.tried_configurations() if inp.memory_enabled else {}
    note = " (configuration was already tested)" if config_hash(data["model_type"], configuration) in tried else ""
    return PlannerOutput(
        hypothesis=data["hypothesis"], model_type=data["model_type"], proposed_configuration=configuration,
        parent_experiment_id=base.experiment_id, referenced_memory_ids=cited,
        reasoning_summary=f"LLM ({usage['model']}): {data['reasoning_summary']}{note}", plan_source="llm",
        llm_usage=usage,
    )
