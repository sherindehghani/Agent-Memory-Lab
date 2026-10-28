import { Background, Controls, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Card, ErrorBanner, Json, Pill } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";
import type { GraphNode, GraphResponse, Page, Run } from "../types";

// Column layout by node type; experiments sit in the middle ordered by sequence.
const COLUMNS = ["agent", "memory", "experiment", "finding", "config", "model", "commit", "run"];
const TYPE_ACCENT: Record<string, string> = {
  experiment: "var(--series-1)", finding: "var(--series-2)", memory: "#1baf7a",
};

function nodeLabel(n: GraphNode) {
  const label = String(n.label ?? n.id);
  return label.length > 42 ? `${label.slice(0, 40)}…` : label;
}

export default function GraphPage() {
  const runs = useApi(() => api.get<Page<Run>>("/api/runs", { limit: 30 }), []);
  const [runId, setRunId] = useState("");
  const [hidden, setHidden] = useState<Set<string>>(new Set(["memory"]));
  const graph = useApi(() => api.get<GraphResponse>("/api/graph", { run_id: runId || undefined }), [runId]);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [edgeInfo, setEdgeInfo] = useState<Record<string, unknown> | null>(null);
  const findingId = selected?.type === "finding" ? selected.id.split(":")[1] : null;
  const evidence = useApi(
    () => (findingId ? api.get<Record<string, unknown>>(`/api/graph/findings/${findingId}/evidence`) : Promise.resolve(null)),
    [findingId]);

  const { nodes, edges } = useMemo(() => {
    const g = graph.data;
    if (!g) return { nodes: [] as Node[], edges: [] as Edge[] };
    const visible = g.nodes.filter((n) => !hidden.has(n.type));
    const ids = new Set(visible.map((n) => n.id));
    const byCol: Record<string, GraphNode[]> = {};
    visible.forEach((n) => (byCol[n.type] ??= []).push(n));
    byCol.experiment?.sort((a, b) => Number(a.sequence ?? 0) - Number(b.sequence ?? 0));
    const cols = COLUMNS.filter((c) => byCol[c]?.length);
    const nodes: Node[] = cols.flatMap((c, ci) =>
      byCol[c].map((n, i) => ({
        id: n.id,
        position: { x: ci * 260, y: i * 74 },
        data: { label: (
          <div className="text-left">
            <div className="text-[10px] uppercase tracking-wide text-ink3">{n.type}{n.decision ? ` · ${n.decision}` : ""}</div>
            <div className="text-xs text-ink">{nodeLabel(n)}</div>
          </div>) },
        style: { background: "var(--surface-1)", color: "var(--text-primary)", border: "1px solid var(--border)",
          borderLeft: `4px solid ${TYPE_ACCENT[n.type] ?? "var(--neutral-mark)"}`, borderRadius: 6, width: 220, padding: 6 },
      })));
    const edges: Edge[] = g.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({
      id: e.id, source: e.source, target: e.target, label: e.relation,
      data: { relation: e.relation, evidence_ref: e.evidence_ref },
      labelStyle: { fontSize: 10, fill: "var(--text-secondary)" },
      labelBgStyle: { fill: "var(--surface-1)" },
      style: { stroke: "var(--neutral-mark)", strokeWidth: 1.2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "var(--neutral-mark)" },
    }));
    return { nodes, edges };
  }, [graph.data, hidden]);

  const toggle = (t: string) => setHidden((h) => { const n = new Set(h); n.has(t) ? n.delete(t) : n.add(t); return n; });

  return (
    <div className="space-y-4">
      <Card title="Knowledge graph" actions={
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label className="flex items-center gap-1 text-ink2">Run
            <select value={runId} onChange={(e) => { setRunId(e.target.value); setSelected(null); }}
              className="rounded border border-line bg-surface px-2 py-1 text-ink">
              <option value="">latest</option>
              {runs.data?.items.map((r) => <option key={r.id} value={r.id}>{r.id} ({r.state})</option>)}
            </select></label>
          {graph.data && <span className="text-xs text-ink3">{graph.data.stats.nodes} nodes · {graph.data.stats.edges} edges · {graph.data.stats.dangling} dangling (excluded)</span>}
        </div>}>
        <ErrorBanner error={graph.error} />
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-ink2">
          <span>Legend / show:</span>
          {COLUMNS.map((t) => (
            <label key={t} className="flex items-center gap-1 rounded border border-line px-1.5 py-0.5">
              <input type="checkbox" checked={!hidden.has(t)} onChange={() => toggle(t)} />
              <span className="inline-block h-3 w-1" style={{ background: TYPE_ACCENT[t] ?? "var(--neutral-mark)" }} />{t}
            </label>
          ))}
          <span className="text-ink3">Edges are labelled with their relation; click one to see its evidence reference.</span>
        </div>
        <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
          <div className="h-[600px] rounded border border-line bg-surface">
            <ReactFlow nodes={nodes} edges={edges} fitView minZoom={0.1} nodesConnectable={false}
              onNodeClick={(_, n) => { setSelected(graph.data?.nodes.find((x) => x.id === n.id) ?? null); setEdgeInfo(null); }}
              onEdgeClick={(_, e) => setEdgeInfo({ source: e.source, target: e.target, ...(e.data as object) })}>
              <Background color="var(--grid)" />
              <Controls />
            </ReactFlow>
          </div>
          <div className="space-y-3 text-sm">
            {edgeInfo && (
              <div className="rounded border border-line p-2">
                <div className="mb-1 font-semibold">Relationship</div>
                <Json value={edgeInfo} />
              </div>
            )}
            {!selected && !edgeInfo && <p className="text-ink3">Select a node to inspect it. Selecting a finding shows its evidence path.</p>}
            {selected && (
              <div className="rounded border border-line p-2">
                <div className="flex flex-wrap items-center gap-2"><Pill tone="accent">{selected.type}</Pill>
                  <span className="break-all font-mono text-xs">{selected.id}</span></div>
                {selected.type === "experiment" && (
                  <Link className="mt-1 block text-accent" to={`/experiments/${selected.id.split(":")[1]}`}>Open experiment →</Link>
                )}
                <Json value={selected} />
              </div>
            )}
            {findingId && evidence.data && (
              <div className="rounded border border-line p-2">
                <div className="mb-1 font-semibold">Evidence path</div>
                <p className="mb-1 text-ink2">{String(evidence.data.statement)}</p>
                {(evidence.data.paths as { source: string; relation: string; target: string }[][]).map((p, i) => (
                  <div key={i} className="mb-1 font-mono text-xs">
                    {p.map((s, j) => <span key={j}>{j === 0 && s.source.split(":")[0]} —{s.relation}→ {s.target.split(":")[0]}:{s.target.split(":")[1].slice(0, 10)} </span>)}
                  </div>
                ))}
                <p className="mt-1 text-xs text-ink3">{String(evidence.data.caveat)}</p>
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
