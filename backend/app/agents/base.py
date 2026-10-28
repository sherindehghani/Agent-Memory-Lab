"""Common agent interface.

Each agent declares a name, role, input/output schema and the tools it may use.
``Agent.run`` validates the input, executes, validates the output, and persists
an :class:`~app.models.AgentAction` including which memories were retrieved.
Agents reach shared memory only through the ``store``/``retriever`` interfaces
in :class:`AgentContext`.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import log_event
from app.experiments.versioning import GitWorkspace
from app.memory.retrieval import MemoryRetriever
from app.memory.store import MemoryStore
from app.models import AgentAction, new_id

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    session: Session
    run_id: str
    namespace: str
    settings: Settings
    store: MemoryStore
    retriever: MemoryRetriever
    workspace: GitWorkspace
    rng: random.Random
    should_stop: Callable[[], bool] = lambda: False
    execution_mode: str = "subprocess"
    llm: Any = None  # optional app.agents.llm.LLMClient
    experiment_id: str | None = None


@dataclass
class AgentResult:
    agent_id: str
    status: str  # success | error
    output: BaseModel | None
    error: str | None
    action_id: str
    retrieved_memory_ids: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    retrieval_ms: float = 0.0
    duration_ms: float = 0.0


class Agent(ABC):
    agent_id: ClassVar[str]
    name: ClassVar[str]
    role: ClassVar[str]
    description: ClassVar[str]
    tools: ClassVar[tuple[str, ...]]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    action_type: ClassVar[str]

    def run(self, ctx: AgentContext, payload: BaseModel | dict[str, Any]) -> AgentResult:
        data = payload.model_dump() if isinstance(payload, BaseModel) else payload
        inp = self.input_schema.model_validate(data)
        stats = ctx.retriever.stats
        before_count, before_ms, before_ids = stats.count, stats.total_ms, len(stats.memory_ids)
        t0 = time.perf_counter()
        output: BaseModel | None = None
        error: str | None = None
        try:
            output = self.output_schema.model_validate(self.execute(ctx, inp).model_dump())
            status = "success"
        except Exception as exc:  # recorded, never swallowed silently
            status, error = "error", f"{type(exc).__name__}: {exc}"
            log_event(logger, "agent_error", error, run_id=ctx.run_id, experiment_id=ctx.experiment_id,
                      agent_id=self.agent_id, level=logging.ERROR)
        duration_ms = (time.perf_counter() - t0) * 1000
        retrieved = stats.memory_ids[before_ids:]
        action = AgentAction(
            id=new_id("act"), agent_id=self.agent_id, run_id=ctx.run_id, experiment_id=ctx.experiment_id,
            action_type=self.action_type, status=status, input=inp.model_dump(mode="json"),
            output=output.model_dump(mode="json") if output else None, error=error,
            retrieved_memory_ids=retrieved, retrieval_count=stats.count - before_count,
            retrieval_ms=round(stats.total_ms - before_ms, 3), duration_ms=round(duration_ms, 3),
        )
        ctx.session.add(action)
        ctx.session.flush()
        log_event(logger, f"agent_{self.action_type}", f"{self.name} finished with {status}", run_id=ctx.run_id,
                  experiment_id=ctx.experiment_id, agent_id=self.agent_id, duration_ms=round(duration_ms, 1))
        return AgentResult(self.agent_id, status, output, error, action.id, retrieved,
                           action.retrieval_count, action.retrieval_ms, duration_ms)

    @abstractmethod
    def execute(self, ctx: AgentContext, inp: Any) -> BaseModel: ...
