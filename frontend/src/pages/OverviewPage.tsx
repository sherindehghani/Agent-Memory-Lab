import { useState } from "react";
import { Link } from "react-router-dom";
import { F1Chart } from "../components/charts";
import { Button, Card, ErrorBanner, Pill, Stat, fmt } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";
import type { Overview, Run } from "../types";

export default function OverviewPage() {
  const overview = useApi(() => api.get<Overview>("/api/overview"), [], 2500);
  const [form, setForm] = useState({ max_experiments: 5, memory_enabled: true, seed: 0,
    initial_model_type: "random_forest", planner_mode: "deterministic" });
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const o = overview.data;
  const run = o?.latest_run;

  async function startRun() {
    setErr(null);
    try {
      const r = await api.post<Run>("/api/runs", form);
      setMsg(`Started ${r.id}`);
      overview.reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }
  async function stopRun() {
    if (!run) return;
    try {
      await api.post(`/api/runs/${run.id}/stop`);
      setMsg(`Stop requested for ${run.id}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="space-y-4">
      <ErrorBanner error={overview.error} />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-7">
        <Stat label="Runs" value={o?.total_runs ?? "—"} />
        <Stat label="Experiments" value={o?.total_experiments ?? "—"} />
        <Stat label="Accepted" value={o?.accepted ?? "—"} />
        <Stat label="Rejected" value={o?.rejected ?? "—"} hint={o?.invalid ? `${o.invalid} invalid` : undefined} />
        <Stat label="Best validation F1" value={fmt.f1(o?.best_validation_f1)} />
        <Stat label="Latest run" value={<span className="text-base">{run?.state ?? "—"}</span>} hint={run?.id} />
        <Stat label="Memory records" value={o?.total_memory_records ?? "—"} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Validation F1 across experiments" className="lg:col-span-2">
          {o && <F1Chart series={o.f1_series} />}
        </Card>

        <Card title="Start a run">
          <ErrorBanner error={err} />
          <div className="space-y-2 text-sm">
            <label className="flex items-center justify-between gap-2">Experiments (budget)
              <input type="number" min={1} max={50} value={form.max_experiments}
                onChange={(e) => setForm({ ...form, max_experiments: Number(e.target.value) })}
                className="w-20 rounded border border-line bg-surface px-2 py-1" /></label>
            <label className="flex items-center justify-between gap-2">Planner seed
              <input type="number" min={0} value={form.seed} onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })}
                className="w-20 rounded border border-line bg-surface px-2 py-1" /></label>
            <label className="flex items-center justify-between gap-2">Initial model
              <select value={form.initial_model_type} onChange={(e) => setForm({ ...form, initial_model_type: e.target.value })}
                className="rounded border border-line bg-surface px-2 py-1">
                <option value="random_forest">Random Forest</option>
                <option value="logistic_regression">Logistic Regression</option>
              </select></label>
            <label className="flex items-center justify-between gap-2">Planner
              <select value={form.planner_mode} onChange={(e) => setForm({ ...form, planner_mode: e.target.value })}
                className="rounded border border-line bg-surface px-2 py-1">
                <option value="deterministic">deterministic</option>
                <option value="llm">LLM (falls back if not configured)</option>
              </select></label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={form.memory_enabled} onChange={(e) => setForm({ ...form, memory_enabled: e.target.checked })} />
              Memory enabled</label>
            <p className="text-xs text-ink3">The initial model only matters when no accepted baseline exists yet.</p>
            <div className="flex gap-2 pt-1">
              <Button onClick={startRun} disabled={o?.busy}>Start new run</Button>
              <Button variant="secondary" onClick={stopRun} disabled={!o?.busy}>Stop</Button>
            </div>
            {msg && <p className="text-xs text-ink2">{msg}</p>}
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Current accepted baseline">
          {o?.current_baseline ? (
            <div className="space-y-1 text-sm">
              <Link className="font-mono text-accent" to={`/experiments/${o.current_baseline.id}`}>{o.current_baseline.id}</Link>
              <div>{o.current_baseline.model_type}: <span className="font-mono text-xs">{fmt.config(o.current_baseline.configuration)}</span></div>
              <div>Validation F1 {fmt.f1(o.current_baseline.metrics.f1)} · peak memory {fmt.num(o.current_baseline.metrics.peak_memory_mb)} MB</div>
              <div className="text-xs text-ink3">Stored as an explicit database pointer, not inferred from Git.</div>
            </div>
          ) : <p className="text-sm text-ink3">No baseline yet — start a run.</p>}
        </Card>
        <Card title="Latest run summary">
          {run ? (
            <div className="space-y-1 text-sm">
              <div className="flex flex-wrap items-center gap-2"><span className="font-mono">{run.id}</span><Pill>{run.state}</Pill>
                <Pill tone={run.memory_enabled ? "accent" : "neutral"}>{run.memory_enabled ? "memory on" : "memory off"}</Pill></div>
              <div className="text-ink2">Stop reason: {run.stop_reason ?? "—"}</div>
              {run.error && <div className="text-critical">Error: {run.error}</div>}
              {run.summary && (
                <ul className="list-inside list-disc text-ink2">
                  <li>{String(run.summary.experiments)} experiments, {String(run.summary.accepted)} accepted, {String(run.summary.rejected)} rejected</li>
                  <li>{String(run.summary.prior_findings_reused)} prior memories cited; {String(run.summary.repeated_configurations)} repeated configurations</li>
                  <li>Held-out test F1 of final baseline: {fmt.f1((run.summary.final_baseline_test_metrics as Record<string, number>)?.f1)}</li>
                  {(run.summary.unfinished as string[] | undefined)?.slice(0, 3).map((u) => <li key={u}>Unfinished: {u}</li>)}
                </ul>
              )}
            </div>
          ) : <p className="text-sm text-ink3">No runs yet.</p>}
        </Card>
      </div>
    </div>
  );
}
