from __future__ import annotations

from app.agents.base import Agent, AgentContext
from app.experiments import runner
from app.experiments.search_space import validate_configuration
from app.experiments.versioning import GitError
from app.schemas.agents import ExecutorInput, ExecutorOutput


class ExecutorAgent(Agent):
    agent_id = "executor_01"
    name = "Executor"
    role = "executor"
    description = ("Validates the proposed configuration, commits it to the experiment workspace, trains and "
                   "evaluates the model with bounded retries and timeouts, and reports measured metrics or errors.")
    tools = ("search_space.validate", "git.commit_experiment", "runner.execute")
    input_schema = ExecutorInput
    output_schema = ExecutorOutput
    action_type = "execute"

    def execute(self, ctx: AgentContext, inp: ExecutorInput) -> ExecutorOutput:
        configuration = validate_configuration(inp.model_type, inp.configuration)
        commit, git_error = None, None
        try:
            commit = ctx.workspace.commit_experiment(inp.experiment_id, inp.artifact, inp.parent_commit,
                                                     inp.commit_message)
        except GitError as exc:
            git_error = str(exc)

        attempts, total_wall = 0, 0.0
        outcome = None
        for attempt in range(1, inp.max_retries + 2):
            attempts = attempt
            outcome = runner.execute(inp.model_type, configuration, inp.seed, timeout=inp.timeout_seconds,
                                     mode=ctx.execution_mode, should_stop=ctx.should_stop)
            total_wall += outcome.wall_seconds
            if outcome.status in ("completed", "cancelled") or ctx.should_stop():
                break
        assert outcome is not None
        if outcome.status == "completed" and outcome.metrics:
            ctx.workspace.save_artifact(inp.experiment_id, "metrics.json", outcome.metrics)
        return ExecutorOutput(status=outcome.status, metrics=outcome.metrics, error=outcome.error,
                              attempts=attempts, git_commit=commit, git_error=git_error,
                              wall_seconds=round(total_wall, 3))
