import { useEffect, useState } from "react";
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { Button, Card, ErrorBanner, Pill, fmt } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";
import type { Run } from "../types";

interface ResultRow {
  experiment_id: string;
  commit: string;
  val_f1: number | null;
  f1_std: number | null;
  memory_mb: number | null;
  status: string;
  description: string;
  idea: string | null;
  lines: number | null;
  error: string | null;
  decision_reason: string | null;
  attempts: number;
}
interface SessionDetail {
  run: Run;
  results: ResultRow[];
  best: { experiment_id: string; commit: string; source: string | null; test_metrics: Record<string, number> } | null;
}
interface DiffResponse { diff: string; source: string | null; log_tail: string; commit: string | null }

const STATUS_COLOR: Record<string, string> = {
  keep: "var(--status-good)", discard: "var(--neutral-mark)", crash: "var(--status-critical)",
};
const STATUS_ICON: Record<string, string> = { keep: "✓", discard: "–", crash: "✕" };

function StatusBadge({ status }: { status: string }) {
  const tone = status === "keep" ? "good" : status === "crash" ? "critical" : "neutral";
  return <Pill tone={tone}>{STATUS_ICON[status] ?? "•"} {status}</Pill>;
}

function ProgressChart({ rows }: { rows: ResultRow[] }) {
  let best: number | null = null;
  const data = rows.map((r, i) => {
    if (r.status === "keep" && r.val_f1 != null) best = r.val_f1;
    return { i: i + 1, f1: r.status === "crash" ? null : r.val_f1, best, status: r.status, description: r.description };
  });
  const values = data.flatMap((d) => [d.f1, d.best]).filter((v): v is number => v != null);
  if (!values.length) return <p className="text-sm text-ink3">No measured experiments yet.</p>;
  const lo = Math.floor((Math.min(...values) - 0.005) * 200) / 200;
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.003) * 200) / 200);
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-4 text-xs text-ink2">
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-accent" />best kept val_f1</span>
        <span>✓ keep · – discard · ✕ crash (not plotted)</span>
      </div>
      <div className="h-56">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="i" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false}
              stroke="var(--border)" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
            <YAxis domain={[lo, hi]} tickFormatter={(v) => v.toFixed(3)} width={48} stroke="var(--border)"
              tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as (typeof data)[number];
              return (
                <div className="rounded border border-line bg-card px-2 py-1.5 text-xs text-ink shadow">
                  <div>#{p.i} {p.description}</div>
                  <div>val_f1: {fmt.f1(p.f1)} · best: {fmt.f1(p.best)}</div>
                  <div>status: {p.status}</div>
                </div>
              );
            }} />
            <Line type="stepAfter" dataKey="best" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Scatter dataKey="f1" isAnimationActive={false}
              shape={(props: { cx?: number; cy?: number; payload?: { status: string } }) => {
                const { cx, cy, payload } = props;
                if (cx == null || cy == null || !payload) return <g />;
                const color = STATUS_COLOR[payload.status] ?? "var(--neutral-mark)";
                const hollow = payload.status === "discard";
                return <circle cx={cx} cy={cy} r={5} fill={hollow ? "var(--surface-1)" : color}
                  stroke={hollow ? color : "var(--surface-1)"} strokeWidth={2} />;
              }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ProgramEditor() {
  const program = useApi(() => api.get<{ content: string }>("/api/autoresearch/program"), []);
  const [text, setText] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { if (program.data) setText(program.data.content); }, [program.data]);
  async function save() {
    try {
      await api.put("/api/autoresearch/program", { content: text });
      setMsg("Saved. The next experiment reads the new instructions.");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  }
  return (
    <Card title="program.md — your instructions to the agent" actions={<Button variant="secondary" onClick={save}>Save</Button>}>
      <ErrorBanner error={program.error} />
      <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
        className="h-72 w-full rounded border border-line bg-surface p-2 font-mono text-xs text-ink" />
      <p className="mt-1 text-xs text-ink3">{msg ?? "Read by the LLM proposer before every experiment. The offline idea bank ignores it."}</p>
    </Card>
  );
}

export default function AutoresearchPage() {
  const sessions = useApi(() => api.get<Run[]>("/api/autoresearch/sessions"), [], 3000);
  const [selected, setSelected] = useState("");
  const runId = selected || sessions.data?.[0]?.id || "";
  const detail = useApi(() => (runId ? api.get<SessionDetail>(`/api/autoresearch/sessions/${runId}`) : Promise.resolve(null)), [runId], 2500);
  const [form, setForm] = useState({ proposer: "offline", max_experiments: 20, max_minutes: 30, tag: "" });
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const diff = useApi(() => (open ? api.get<DiffResponse>(`/api/autoresearch/experiments/${open}/diff`) : Promise.resolve(null)), [open]);
  const run = detail.data?.run;
  const active = run && !["COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"].includes(run.state);

  async function start() {
    setErr(null);
    try {
      const r = await api.post<Run>("/api/autoresearch/sessions", { ...form, tag: form.tag || undefined });
      setSelected(r.id);
      sessions.reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }
  async function stop() {
    if (run) await api.post(`/api/runs/${run.id}/stop`).catch((e) => setErr(String(e)));
  }

  const rows = detail.data?.results ?? [];
  const kept = rows.filter((r) => r.status === "keep").length;

  return (
    <div className="space-y-4">
      <Card title="Autoresearch — the agent edits train.py, keeps what improves val_f1 and resets the rest">
        <ErrorBanner error={err ?? sessions.error} />
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <label className="flex flex-col text-xs text-ink2">Proposer
            <select value={form.proposer} onChange={(e) => setForm({ ...form, proposer: e.target.value })}
              className="rounded border border-line bg-surface px-2 py-1 text-sm text-ink">
              <option value="offline">offline idea bank</option>
              <option value="llm">Claude (needs API key)</option>
            </select></label>
          <label className="flex flex-col text-xs text-ink2">Max experiments
            <input type="number" min={1} max={500} value={form.max_experiments}
              onChange={(e) => setForm({ ...form, max_experiments: Number(e.target.value) })}
              className="w-24 rounded border border-line bg-surface px-2 py-1 text-sm text-ink" /></label>
          <label className="flex flex-col text-xs text-ink2">Max minutes
            <input type="number" min={1} max={720} value={form.max_minutes}
              onChange={(e) => setForm({ ...form, max_minutes: Number(e.target.value) })}
              className="w-24 rounded border border-line bg-surface px-2 py-1 text-sm text-ink" /></label>
          <label className="flex flex-col text-xs text-ink2">Tag (branch name)
            <input value={form.tag} placeholder="e.g. oct29" onChange={(e) => setForm({ ...form, tag: e.target.value })}
              className="w-32 rounded border border-line bg-surface px-2 py-1 text-sm text-ink" /></label>
          <Button onClick={start} disabled={!!active}>Start session</Button>
          <Button variant="secondary" onClick={stop} disabled={!active}>Stop</Button>
          <label className="ml-auto flex flex-col text-xs text-ink2">Session
            <select value={runId} onChange={(e) => { setSelected(e.target.value); setOpen(null); }}
              className="rounded border border-line bg-surface px-2 py-1 text-sm text-ink">
              {sessions.data?.map((s) => (
                <option key={s.id} value={s.id}>{String(s.initial_configuration?.branch ?? s.id)} · {s.state}</option>
              ))}
            </select></label>
        </div>
        {run && (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm text-ink2">
            <Pill>{run.state}</Pill>
            <span className="font-mono text-xs">{String(run.initial_configuration?.branch ?? "")}</span>
            <span>{rows.length} experiments · {kept} kept</span>
            {run.stop_reason && <span>· {run.stop_reason}</span>}
            {run.error && <span className="text-critical">· {run.error}</span>}
          </div>
        )}
      </Card>

      {run && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card title="Progress" className="lg:col-span-2"><ProgressChart rows={rows} /></Card>
          <Card title="Best train.py">
            {detail.data?.best ? (
              <>
                <div className="mb-1 text-xs text-ink2">
                  commit <span className="font-mono">{detail.data.best.commit?.slice(0, 7)}</span>
                  {detail.data.best.test_metrics.f1 != null && <> · held-out test F1 {fmt.f1(detail.data.best.test_metrics.f1)}</>}
                </div>
                <pre className="max-h-64 overflow-auto rounded bg-sunken p-2 font-mono text-xs text-ink">{detail.data.best.source}</pre>
              </>
            ) : <p className="text-sm text-ink3">Nothing kept yet.</p>}
          </Card>
        </div>
      )}

      {run && (
        <Card title="results.tsv">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm tabular-nums">
              <thead><tr className="text-left text-xs text-ink2">
                <th className="py-1.5">#</th><th>Commit</th><th>val_f1</th><th>± std</th><th>Peak mem (MB)</th><th>Lines</th>
                <th>Status</th><th>Description</th></tr></thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.experiment_id} onClick={() => setOpen(open === r.experiment_id ? null : r.experiment_id)}
                    className={`cursor-pointer border-t border-line hover:bg-sunken ${open === r.experiment_id ? "bg-sunken" : ""}`}>
                    <td className="py-1.5 text-ink3">{i + 1}</td>
                    <td className="font-mono text-xs">{r.commit}</td>
                    <td>{fmt.num(r.val_f1, 6)}</td>
                    <td className="text-ink2">{fmt.num(r.f1_std, 4)}</td>
                    <td>{fmt.num(r.memory_mb)}</td>
                    <td>{r.lines ?? "—"}</td>
                    <td><StatusBadge status={r.status} /></td>
                    <td>{r.description}{r.attempts > 1 && <span className="text-xs text-ink3"> ({r.attempts} attempts)</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {open && (
            <div className="mt-3 space-y-2">
              <div className="flex items-center gap-3 text-sm">
                <Link to={`/experiments/${open}`} className="text-accent">Open full experiment record →</Link>
                <span className="text-xs text-ink2">{rows.find((r) => r.experiment_id === open)?.decision_reason}</span>
              </div>
              <ErrorBanner error={diff.error} />
              {diff.data?.diff ? <pre className="max-h-72 overflow-auto rounded bg-sunken p-2 font-mono text-xs">{diff.data.diff}</pre>
                : <p className="text-xs text-ink3">No diff (baseline, or the code did not change).</p>}
              {diff.data?.log_tail && (
                <details><summary className="cursor-pointer text-xs text-ink2">Run log (tail)</summary>
                  <pre className="max-h-60 overflow-auto rounded bg-sunken p-2 font-mono text-xs">{diff.data.log_tail}</pre></details>
              )}
            </div>
          )}
        </Card>
      )}

      <ProgramEditor />
    </div>
  );
}
