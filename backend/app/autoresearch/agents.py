from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.agents.base import Agent, AgentContext
from app.autoresearch.guard import check_source
from app.autoresearch.harness import run_harness
from app.autoresearch.proposers import LLMProposer, OfflineProposer, ProposalContext


class CodeProposalInput(BaseModel):
    proposer: Literal["offline", "llm"]
    program: str
    current_source: str
    best_f1: float | None
    history: list[dict[str, Any]]
    memories: list[dict[str, Any]]
    tried_hashes: list[str]
    fix_error: str | None = None
    failed_source: str | None = None


class CodeProposalOutput(BaseModel):
    exhausted: bool = False
    description: str = ""
    hypothesis: str = ""
    source: str = ""
    idea: str | None = None
    guard_errors: list[str] = []
    llm_usage: dict[str, Any] | None = None


class AutoresearcherAgent(Agent):
    agent_id = "autoresearcher_01"
    name = "Autoresearcher"
    role = "code researcher"
    description = ("Reads program.md, the results history and shared memory, then edits train.py with one "
                   "focused idea per experiment (Claude when configured, otherwise a built-in idea bank).")
    tools = ("memory.search", "memory.tried_configurations", "git.read_train", "llm.propose_code", "idea_bank")
    input_schema = CodeProposalInput
    output_schema = CodeProposalOutput
    action_type = "propose_code"

    def execute(self, ctx: AgentContext, inp: CodeProposalInput) -> CodeProposalOutput:
        if inp.proposer == "llm" and ctx.llm is not None:
            proposer: OfflineProposer | LLMProposer = LLMProposer(ctx.llm)
        else:
            proposer = OfflineProposer()
        pctx = ProposalContext(program=inp.program, current_source=inp.current_source, best_f1=inp.best_f1,
                               history=inp.history, memories=inp.memories, tried_hashes=set(inp.tried_hashes),
                               fix_error=inp.fix_error, failed_source=inp.failed_source)
        proposal = proposer.fix(pctx) if inp.fix_error else proposer.propose(pctx)
        if proposal is None:
            return CodeProposalOutput(exhausted=True)
        return CodeProposalOutput(description=proposal.description, hypothesis=proposal.hypothesis,
                                  source=proposal.source, idea=proposal.idea, guard_errors=check_source(proposal.source),
                                  llm_usage=proposal.usage)


class HarnessInput(BaseModel):
    experiment_id: str
    source: str
    seed: int
    timeout_seconds: float


class HarnessOutput(BaseModel):
    status: Literal["completed", "failed", "timeout", "cancelled"]
    metrics: dict[str, float] | None
    error: str | None
    log_tail: str
    wall_seconds: float


class CodeExecutorAgent(Agent):
    agent_id = "executor_01"
    name = "Executor"
    role = "executor"
    description = "Runs train.py inside the fixed evaluation harness (subprocess, time limit) and reports metrics."
    tools = ("harness.run",)
    input_schema = HarnessInput
    output_schema = HarnessOutput
    action_type = "run_train_py"

    def execute(self, ctx: AgentContext, inp: HarnessInput) -> HarnessOutput:
        errors = check_source(inp.source)
        if errors:
            return HarnessOutput(status="failed", metrics=None, error="guard: " + "; ".join(errors), log_tail="",
                                 wall_seconds=0.0)
        log_path = ctx.workspace.root / "autoresearch" / "logs" / f"{inp.experiment_id}.log"
        out = run_harness(inp.source, inp.seed, timeout=inp.timeout_seconds, should_stop=ctx.should_stop,
                          log_path=log_path)
        return HarnessOutput(status=out.status, metrics=out.metrics, error=out.error, log_tail=out.log_tail,
                             wall_seconds=round(out.wall_seconds, 3))
