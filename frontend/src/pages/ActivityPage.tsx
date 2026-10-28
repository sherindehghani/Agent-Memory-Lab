import { useState } from "react";
import { Link } from "react-router-dom";
import { ComparisonCharts } from "../components/charts";
import { Button, Card, ErrorBanner, Json, Pill, fmt } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";
import type { AgentAction, AgentInfo, AgentMessage, EvaluationRow, Page, Run } from "../types";

function Evaluation() {
  const results = useApi(() => api.get<EvaluationRow[]>("/api/evaluation/results"), [], 3000);
  const [seeds, setSeeds] = useState("0,1,2");
  const [budget, setBudget] = useState(6);
  const [err, setErr] = useState<string | null>(null);
  const latest = results.data?.[0];
  const done = results.data?.find((r) => r.status === "completed");

  async function start() {
    setErr(null);
    try {
      await api.post("/api/evaluation/compare", { seeds: seeds.split(",").map((s) => Number(s.trim())), budget });
      results.reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <Card title="Memory evaluation: memory-enabled vs. memory-disabled planning" actions={
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <label className="text-ink2">Seeds <input value={seeds} onChange={(e) => setSeeds(e.target.value)} className="w-24 rounded border border-line bg-surface px-2 py-1 text-ink" /></label>
        <label className="text-ink2">Budget <input type="number" min={2} max={20} value={budget} onChange={(e) => setBudget(Number(e.target.value))} className="w-16 rounded border border-line bg-surface px-2 py-1 text-ink" /></label>
        <Button onClick={start} disabled={latest?.status === "running"}>Run comparison</Button>
      </div>}>
      <ErrorBanner error={err ?? results.error} />
      <p className="mb-3 text-xs text-ink3">
        Each seed runs both conditions from an empty, isolated namespace with the same dataset, initial configuration,
        search space, budget and acceptance rule. This is a small demonstration, not a statistically conclusive study.
      </p>
      {latest?.status === "running" && <p className="mb-2 text-sm text-ink2">Evaluation {latest.id} is running…</p>}
      {done?.results ? (
        <>
          <ComparisonCharts aggregate={done.results.aggregate} />
          <details className="mt-3"><summary className="cursor-pointer text-xs text-ink2">Per-trial results ({done.id})</summary>
            <div className="overflow-x-auto">
              <table className="mt-2 w-full min-w-[700px] text-xs tabular-nums">
                <thead><tr className="text-left text-ink2"><th>Seed</th><th>Condition</th><th>Run</th><th>Best F1</th><th>Accepted</th>
                  <th>Repeated</th><th>Findings reused</th><th>Retrievals</th><th>Test F1</th></tr></thead>
                <tbody>{done.results.trials.map((t, i) => (
                  <tr key={i} className="border-t border-line">
                    <td>{String(t.seed)}</td><td>{String(t.condition)}</td><td className="font-mono">{String(t.run_id)}</td>
                    <td>{fmt.f1(t.best_validation_f1)}</td><td>{String(t.accepted)}</td><td>{String(t.repeated_configurations)}</td>
                    <td>{String(t.prior_findings_reused)}</td><td>{String(t.memory_retrievals)}</td><td>{fmt.f1(t.final_baseline_test_f1)}</td>
                  </tr>))}</tbody>
              </table>
            </div>
            <p className="mt-1 text-xs text-ink3">{done.results.note}</p>
          </details>
        </>
      ) : <p className="text-sm text-ink3">No completed comparison yet.</p>}
    </Card>
  );
}

export default function ActivityPage() {
  const agents = useApi(() => api.get<AgentInfo[]>("/api/agents"), [], 5000);
  const runs = useApi(() => api.get<Page<Run>>("/api/runs", { namespace: "*", limit: 50 }), [], 5000);
  const [agentId, setAgentId] = useState("planner_01");
  const [runId, setRunId] = useState("");
  const activity = useApi(() => api.get<{ actions: Page<AgentAction>; messages: AgentMessage[] }>(
    `/api/agents/${agentId}/activity`, { run_id: runId || undefined, limit: 30 }), [agentId, runId], 4000);
  const runDetail = useApi(() => (runId ? api.get<{ events: { id: number; event_type: string; to_state: string | null; message: string; created_at: string }[] }>(`/api/runs/${runId}`) : Promise.resolve(null)), [runId], 4000);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {agents.data?.map((a) => (
          <button key={a.id} onClick={() => setAgentId(a.id)}
            className={`rounded-lg border bg-card p-3 text-left ${agentId === a.id ? "border-accent" : "border-line"}`}>
            <div className="flex items-center justify-between"><span className="font-semibold">{a.name}</span><span className="text-xs text-ink3">{a.action_count} actions</span></div>
            <p className="mt-1 text-xs text-ink2">{a.description}</p>
            <p className="mt-1 font-mono text-[10px] text-ink3">{a.tools.join(" · ")}</p>
          </button>
        ))}
      </div>

      <Card title={`Activity — ${agentId}`} actions={
        <label className="flex items-center gap-1 text-sm text-ink2">Run
          <select value={runId} onChange={(e) => setRunId(e.target.value)} className="rounded border border-line bg-surface px-2 py-1 text-ink">
            <option value="">all runs</option>
            {runs.data?.items.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.namespace === "main" ? "main" : "eval"} · {r.state}</option>)}
          </select></label>}>
        <ErrorBanner error={activity.error} />
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <h3 className="mb-1 text-sm font-semibold">Actions ({activity.data?.actions.total ?? 0})</h3>
            <div className="space-y-1.5">
              {activity.data?.actions.items.map((a) => (
                <details key={a.id} className="rounded border border-line p-2 text-sm">
                  <summary className="cursor-pointer">
                    <span className="font-medium">{a.action_type}</span>{" "}
                    <Pill tone={a.status === "success" ? "good" : "critical"}>{a.status}</Pill>{" "}
                    {a.experiment_id && <Link to={`/experiments/${a.experiment_id}`} className="font-mono text-xs text-accent">{a.experiment_id}</Link>}{" "}
                    <span className="text-xs text-ink3">{a.retrieval_count} retrievals ({a.retrieval_ms.toFixed(1)} ms) · {a.duration_ms.toFixed(0)} ms</span>
                  </summary>
                  {typeof a.output?.reasoning_summary === "string" && <p className="mt-1 text-ink2">{a.output.reasoning_summary}</p>}
                  {typeof a.output?.reason === "string" && <p className="mt-1 text-ink2">{a.output.reason}</p>}
                  {typeof a.output?.summary === "string" && <p className="mt-1 text-ink2">{a.output.summary}</p>}
                  {a.retrieved_memory_ids.length > 0 && <p className="mt-1 text-xs text-ink2">Retrieved memories: {a.retrieved_memory_ids.join(", ")}</p>}
                  {a.error && <p className="mt-1 text-critical">{a.error}</p>}
                  <Json value={a.output} />
                </details>
              ))}
            </div>
          </div>
          <div>
            <h3 className="mb-1 text-sm font-semibold">Messages</h3>
            <div className="space-y-1.5">
              {activity.data?.messages.map((m) => (
                <details key={m.id} className="rounded border border-line p-2 text-sm">
                  <summary className="cursor-pointer"><span className="font-mono text-xs">{m.sender} → {m.recipient}</span>{" "}
                    <Pill>{m.message_type}</Pill> <span className="text-xs text-ink3">{fmt.time(m.created_at)}</span></summary>
                  <Json value={m.payload} />
                </details>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {runId && runDetail.data && (
        <Card title={`Run lifecycle — ${runId}`}>
          <ol className="max-h-72 space-y-0.5 overflow-auto text-xs">
            {runDetail.data.events.map((e) => (
              <li key={e.id}><span className="text-ink3">{fmt.time(e.created_at)}</span> <span className="font-mono">{e.event_type}</span>
                {e.to_state && <Pill>{e.to_state}</Pill>} <span className="text-ink2">{e.message}</span></li>
            ))}
          </ol>
        </Card>
      )}

      <Evaluation />
    </div>
  );
}
