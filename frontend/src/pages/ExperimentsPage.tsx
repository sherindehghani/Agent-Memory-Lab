import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Card, DecisionBadge, ErrorBanner, Json, Pill, fmt } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";
import type { Experiment, Page } from "../types";

interface Detail {
  experiment: Experiment;
  test_metrics: Record<string, number>;
  baseline: Experiment | null;
  git: { commit: string | null; exists: boolean; config: unknown };
  influenced_by: { memory_id: string; kind: string; title: string; content: string; verification_status: string }[];
  missing_memory_references: string[];
}
interface History {
  lineage: Experiment[];
  children: Experiment[];
  git_log: { commit: string; message: string; date: string }[];
  git_diff_from_parent: string;
  git_commit_missing: boolean;
}

const COMPARE = ["f1", "accuracy", "precision", "recall", "peak_memory_mb", "model_size_kb", "duration_seconds"];

function ExperimentDetail({ id }: { id: string }) {
  const detail = useApi(() => api.get<Detail>(`/api/experiments/${id}`), [id]);
  const history = useApi(() => api.get<History>(`/api/experiments/${id}/history`), [id]);
  const d = detail.data;
  if (detail.error) return <ErrorBanner error={detail.error} />;
  if (!d) return <p className="text-sm text-ink3">Loading…</p>;
  const e = d.experiment;
  const b = d.baseline;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-mono text-base font-semibold">{e.id}</h2>
        <DecisionBadge decision={e.decision} />
        <Pill>{e.status}</Pill>
        <Pill tone="accent">{e.plan_source}</Pill>
        {e.hypothesis_outcome && <Pill tone={e.hypothesis_outcome === "supported" ? "good" : "neutral"}>hypothesis {e.hypothesis_outcome}</Pill>}
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2 text-sm">
          <div><span className="text-ink2">Hypothesis:</span> {e.hypothesis}</div>
          <div><span className="text-ink2">Planner reasoning:</span> {e.reasoning_summary}</div>
          <div><span className="text-ink2">Decision:</span> {e.decision_reason}</div>
          <div><span className="text-ink2">Configuration:</span> <span className="font-mono text-xs">{e.model_type} · {fmt.config(e.configuration)}</span></div>
          <div><span className="text-ink2">Change from parent:</span> {e.moves.join(", ") || "—"}{" "}
            {e.parent_experiment_id && <Link className="font-mono text-accent" to={`/experiments/${e.parent_experiment_id}`}>{e.parent_experiment_id}</Link>}</div>
          {e.error && <div className="text-critical">Error: {e.error}</div>}
          <div className="text-xs text-ink3">Dataset {e.dataset_version} · protocol {e.protocol_version} · threshold {e.acceptance_threshold} · attempts {e.attempts}</div>
        </div>
        <div>
          <table className="w-full text-sm tabular-nums">
            <thead><tr className="text-left text-xs text-ink2"><th className="py-1">Metric (validation)</th><th>This</th>
              <th>Baseline {b ? <Link className="font-mono text-accent" to={`/experiments/${b.id}`}>{fmt.short(b.id)}</Link> : ""}</th><th>Δ</th></tr></thead>
            <tbody>
              {COMPARE.map((m) => {
                const v = e.metrics[m];
                const bv = b?.metrics[m];
                return (
                  <tr key={m} className="border-t border-line">
                    <td className="py-1 text-ink2">{m}</td><td>{fmt.num(v, 4)}</td><td>{fmt.num(bv, 4)}</td>
                    <td>{v != null && bv != null ? fmt.delta(v - bv) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-ink3">Time and memory vary between machines and runs; scores are deterministic for a fixed seed.</p>
          {Object.keys(d.test_metrics).length > 0 && (
            <p className="mt-1 text-xs text-ink2">Held-out test F1 (final evaluation only, never used for selection): {fmt.f1(d.test_metrics.f1)}</p>
          )}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <h3 className="mb-1 text-sm font-semibold">Memories that influenced this proposal</h3>
          {d.influenced_by.length === 0 ? <p className="text-sm text-ink3">None cited.</p> : (
            <ul className="space-y-1 text-sm">
              {d.influenced_by.map((m) => (
                <li key={m.memory_id} className="rounded border border-line p-2">
                  <div className="flex flex-wrap gap-2"><span className="font-mono text-xs">{m.memory_id}</span><Pill>{m.kind}</Pill>
                    <Pill tone={m.verification_status === "verified" ? "good" : m.verification_status === "contradicted" ? "critical" : "neutral"}>{m.verification_status}</Pill></div>
                  <div className="mt-1 text-ink2">{m.content}</div>
                </li>
              ))}
            </ul>
          )}
          {d.missing_memory_references.length > 0 && <p className="text-xs text-critical">Missing referenced memories: {d.missing_memory_references.join(", ")}</p>}
        </div>
        <div>
          <h3 className="mb-1 text-sm font-semibold">Git</h3>
          <div className="text-sm">Commit: <span className="font-mono text-xs">{d.git.commit ?? "—"}</span>{" "}
            {d.git.commit && !d.git.exists && <span className="text-critical">(missing from workspace)</span>}</div>
          {history.data && (
            <>
              <ul className="mt-1 space-y-0.5 text-xs text-ink2">
                {history.data.git_log.slice(0, 6).map((c) => <li key={c.commit}><span className="font-mono">{c.commit.slice(0, 10)}</span> {c.message}</li>)}
              </ul>
              {history.data.git_diff_from_parent && <pre className="mt-2 max-h-48 overflow-auto rounded bg-sunken p-2 font-mono text-xs">{history.data.git_diff_from_parent}</pre>}
              {history.data.children.length > 0 && (
                <div className="mt-2 text-sm">Derived experiments: {history.data.children.map((c) =>
                  <Link key={c.id} className="mr-2 font-mono text-accent" to={`/experiments/${c.id}`}>{c.id}</Link>)}</div>
              )}
            </>
          )}
        </div>
      </div>
      <details><summary className="cursor-pointer text-xs text-ink2">Raw record</summary><Json value={e} /></details>
    </div>
  );
}

export default function ExperimentsPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState("");
  const [model, setModel] = useState("");
  const [decision, setDecision] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 25;
  const list = useApi(() => api.get<Page<Experiment>>("/api/experiments",
    { status, model_type: model, decision, limit, offset }), [status, model, decision, offset], 4000);
  const select = (value: string, set: (v: string) => void, options: string[], label: string) => (
    <label className="flex items-center gap-1 text-sm text-ink2">{label}
      <select value={value} onChange={(e) => { set(e.target.value); setOffset(0); }}
        className="rounded border border-line bg-surface px-2 py-1 text-ink">
        <option value="">all</option>{options.map((o) => <option key={o}>{o}</option>)}
      </select></label>
  );

  return (
    <div className="space-y-4">
      {id && <Card><ExperimentDetail id={id} /></Card>}
      <Card title="Experiment history" actions={
        <div className="flex flex-wrap gap-3">
          {select(status, setStatus, ["completed", "failed", "cancelled", "interrupted"], "Status")}
          {select(decision, setDecision, ["accepted", "rejected", "invalid"], "Decision")}
          {select(model, setModel, ["random_forest", "logistic_regression"], "Model")}
        </div>}>
        <ErrorBanner error={list.error} />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm tabular-nums">
            <thead><tr className="text-left text-xs text-ink2">
              <th className="py-1.5">Experiment</th><th>Model</th><th>Configuration</th><th>Val F1</th><th>Duration (s)</th>
              <th>Peak mem (MB)</th><th>Status</th><th>Decision</th><th>Created</th></tr></thead>
            <tbody>
              {list.data?.items.map((e) => (
                <tr key={e.id} onClick={() => navigate(`/experiments/${e.id}`)}
                  className={`cursor-pointer border-t border-line hover:bg-sunken ${e.id === id ? "bg-sunken" : ""}`}>
                  <td className="py-1.5 font-mono text-xs">{e.id}</td>
                  <td>{e.model_type === "random_forest" ? "RF" : "LR"}</td>
                  <td className="max-w-xs truncate font-mono text-xs" title={fmt.config(e.configuration)}>{fmt.config(e.configuration)}</td>
                  <td>{fmt.f1(e.metrics.f1)}</td>
                  <td>{fmt.num(e.metrics.duration_seconds, 2)}</td>
                  <td>{fmt.num(e.metrics.peak_memory_mb)}</td>
                  <td>{e.status}</td>
                  <td><DecisionBadge decision={e.decision} /></td>
                  <td className="text-xs text-ink2">{fmt.time(e.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {list.data?.items.length === 0 && <p className="py-4 text-sm text-ink3">No experiments match.</p>}
        </div>
        {list.data && list.data.total > limit && (
          <div className="mt-3 flex items-center gap-3 text-sm">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))} className="text-accent disabled:text-ink3">← Prev</button>
            <span className="text-ink2">{offset + 1}–{Math.min(offset + limit, list.data.total)} of {list.data.total}</span>
            <button disabled={offset + limit >= list.data.total} onClick={() => setOffset(offset + limit)} className="text-accent disabled:text-ink3">Next →</button>
          </div>
        )}
      </Card>
    </div>
  );
}
