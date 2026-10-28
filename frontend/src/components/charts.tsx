import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  LabelList,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Overview } from "../types";
import { fmt } from "./ui";

const DECISION_COLOR: Record<string, string> = {
  accepted: "var(--status-good)",
  rejected: "var(--neutral-mark)",
  invalid: "var(--status-critical)",
};
const axis = { stroke: "var(--border)", tick: { fill: "var(--text-muted)", fontSize: 11 } };

function LegendItem({ swatch, label }: { swatch: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink2">
      {swatch}
      {label}
    </span>
  );
}

const dot = (color: string, ring = false) => (
  <svg width="10" height="10" aria-hidden>
    <circle cx="5" cy="5" r="4" fill={ring ? "none" : color} stroke={color} strokeWidth={ring ? 2 : 0} />
  </svg>
);

export function F1Chart({ series }: { series: Overview["f1_series"] }) {
  let baseline: number | null = null;
  const data = series.map((p) => {
    if (p.decision === "accepted" && p.f1 != null) baseline = p.f1;
    return { ...p, baseline };
  });
  const values = data.flatMap((d) => [d.f1, d.baseline]).filter((v): v is number => v != null);
  if (!values.length) return <p className="text-sm text-ink3">No measured experiments yet.</p>;
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.01) * 100) / 100);
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.005) * 100) / 100);
  const invalid = series.filter((s) => s.decision === "invalid").length;

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-4">
        <LegendItem swatch={<span className="inline-block h-0.5 w-4 bg-accent" />} label="accepted baseline F1" />
        <LegendItem swatch={dot(DECISION_COLOR.accepted)} label="✓ accepted" />
        <LegendItem swatch={dot(DECISION_COLOR.rejected, true)} label="– rejected" />
        {invalid > 0 && <LegendItem swatch={dot(DECISION_COLOR.invalid)} label={`✕ invalid (${invalid}, no valid F1)`} />}
      </div>
      <div className="h-64">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="index" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} {...axis}
              label={{ value: "experiment #", position: "insideBottomRight", offset: -2, fill: "var(--text-muted)", fontSize: 11 }} />
            <YAxis domain={[lo, hi]} tickFormatter={(v) => v.toFixed(2)} width={44} {...axis} />
            <Tooltip
              cursor={{ stroke: "var(--border)" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as (typeof data)[number];
                return (
                  <div className="rounded border border-line bg-card px-2 py-1.5 text-xs text-ink shadow">
                    <div className="font-mono">{p.experiment_id}</div>
                    <div>{p.model_type}</div>
                    <div>validation F1: {fmt.f1(p.f1)}</div>
                    <div>baseline F1: {fmt.f1(p.baseline)}</div>
                    <div>decision: {p.decision ?? "—"}</div>
                  </div>
                );
              }}
            />
            <Line type="stepAfter" dataKey="baseline" stroke="var(--series-1)" strokeWidth={2} dot={false}
              isAnimationActive={false} connectNulls />
            <Scatter dataKey="f1" isAnimationActive={false}
              shape={(props: { cx?: number; cy?: number; payload?: { decision: string } }) => {
                const { cx, cy, payload } = props;
                if (cx == null || cy == null || !payload) return <g />;
                const color = DECISION_COLOR[payload.decision] ?? "var(--neutral-mark)";
                const hollow = payload.decision === "rejected";
                return (
                  <circle cx={cx} cy={cy} r={5} fill={hollow ? "var(--surface-1)" : color} stroke={hollow ? color : "var(--surface-1)"}
                    strokeWidth={2} />
                );
              }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export const COMPARISON_METRICS: { key: string; label: string; digits: number }[] = [
  { key: "best_validation_f1", label: "Best validation F1", digits: 4 },
  { key: "accepted", label: "Accepted experiments", digits: 2 },
  { key: "repeated_configurations", label: "Repeated configurations", digits: 2 },
  { key: "prior_findings_reused", label: "Prior findings reused", digits: 2 },
  { key: "memory_retrievals", label: "Memory retrievals", digits: 1 },
  { key: "memory_retrieval_ms_mean", label: "Mean retrieval latency (ms)", digits: 3 },
  { key: "experiment_seconds_total", label: "Experiment time (s)", digits: 2 },
  { key: "final_baseline_test_f1", label: "Held-out test F1 (info only)", digits: 4 },
];

export function ComparisonCharts({ aggregate }: { aggregate: Record<string, Record<string, number | null>> }) {
  const off = aggregate.memory_disabled ?? {};
  const on = aggregate.memory_enabled ?? {};
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-4">
        <LegendItem swatch={<span className="inline-block h-3 w-3 rounded-sm" style={{ background: "var(--series-1)" }} />} label="memory enabled" />
        <LegendItem swatch={<span className="inline-block h-3 w-3 rounded-sm" style={{ background: "var(--series-2)" }} />} label="memory disabled" />
        <span className="text-xs text-ink3">means across seeds</span>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {COMPARISON_METRICS.map((m) => {
          const data = [
            { name: "enabled", value: on[m.key] ?? null, fill: "var(--series-1)" },
            { name: "disabled", value: off[m.key] ?? null, fill: "var(--series-2)" },
          ];
          return (
            <div key={m.key} className="rounded border border-line p-2">
              <div className="mb-1 text-xs text-ink2">{m.label}</div>
              <div className="h-28">
                <ResponsiveContainer>
                  <BarChart data={data} margin={{ top: 16, right: 4, bottom: 0, left: 4 }} barCategoryGap={18}>
                    <XAxis dataKey="name" {...axis} />
                    <YAxis hide domain={[0, "auto"]} />
                    <Tooltip cursor={{ fill: "var(--surface-2)" }} formatter={(v: number) => fmt.num(v, m.digits)}
                      contentStyle={{ background: "var(--surface-1)", border: "1px solid var(--border)", fontSize: 12 }} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive={false} stroke="var(--surface-1)" strokeWidth={2}>
                      <LabelList dataKey="value" position="top" formatter={(v: number) => fmt.num(v, m.digits)}
                        style={{ fill: "var(--text-secondary)", fontSize: 11 }} />
                      {data.map((d) => <Cell key={d.name} fill={d.fill} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
