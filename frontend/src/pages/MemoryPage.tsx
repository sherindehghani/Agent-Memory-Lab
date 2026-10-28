import { useState } from "react";
import { Link } from "react-router-dom";
import { Button, Card, ErrorBanner, Json, Pill, fmt, verificationTone } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";
import type { MemoryRecord, Page, SearchResponse } from "../types";

const EXAMPLES = [
  "Find rejected experiments that reduced memory use",
  "Find experiments with better F1 scores",
  "Find experiments using Random Forest",
  "Find experiments that used n_estimators",
  "Find earlier failures",
];

function EvidencePanel({ memoryId }: { memoryId: string }) {
  const ev = useApi(() => api.get<Record<string, unknown>>(`/api/memory/${memoryId}/evidence`), [memoryId]);
  if (ev.error) return <ErrorBanner error={ev.error} />;
  if (!ev.data) return <p className="text-xs text-ink3">Loading evidence…</p>;
  const status = ev.data.evidence_status as string;
  return (
    <div className="mt-2 space-y-1">
      <div className="text-xs">Evidence status: <Pill tone={status === "ok" ? "good" : status === "no_experiment_reference" ? "neutral" : "critical"}>{status}</Pill></div>
      <Json value={{ experiment: ev.data.experiment, baseline: ev.data.baseline, finding: ev.data.finding, problems: ev.data.problems }} />
    </div>
  );
}

export default function MemoryPage() {
  const [category, setCategory] = useState("");
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [search, setSearch] = useState<SearchResponse | null>(null);
  const [searchErr, setSearchErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const list = useApi(() => api.get<Page<MemoryRecord>>("/api/memory", { category, limit: 50 }), [category], 5000);

  async function runSearch(q = query) {
    setSearchErr(null);
    setQuery(q);
    try {
      setSearch(await api.post<SearchResponse>("/api/memory/search", { query: q, limit: 10 }));
    } catch (e) {
      setSearchErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="space-y-4">
      <Card title="Search shared memory">
        <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); runSearch(); }}>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask about earlier experiments…"
            className="min-w-[240px] flex-1 rounded border border-line bg-surface px-3 py-1.5 text-sm" />
          <Button type="submit">Search</Button>
        </form>
        <div className="mt-2 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => <button key={ex} onClick={() => runSearch(ex)} className="rounded border border-line px-2 py-0.5 text-xs text-ink2 hover:bg-sunken">{ex}</button>)}
        </div>
        <ErrorBanner error={searchErr} />
        {search && (
          <div className="mt-3 space-y-2">
            <div className="text-xs text-ink3">
              Parsed as <span className="font-mono">{JSON.stringify(search.parsed_query)}</span> · ranking: {search.ranking_formula} · {search.latency_ms} ms
              {search.skipped.length > 0 && <> · skipped {search.skipped.length} unusable record(s)</>}
            </div>
            {search.results.length === 0 && <p className="text-sm text-ink3">No matching memory. Nothing is invented when evidence is absent.</p>}
            {search.results.map((r) => (
              <div key={r.memory_id} className="rounded border border-line p-3 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs">{r.memory_id}</span><Pill>{r.category}/{r.kind}</Pill>
                  <Pill tone={verificationTone(r.verification_status)}>{r.verification_status}</Pill>
                  <span className="text-xs text-ink2">score {r.score.toFixed(3)}</span>
                  {r.source_experiment_id && <Link to={`/experiments/${r.source_experiment_id}`} className="font-mono text-xs text-accent">{r.source_experiment_id}</Link>}
                </div>
                <div className="mt-1"><span className="text-ink2">Why retrieved:</span> {r.reason}</div>
                <div className="mt-1 grid gap-x-4 text-xs text-ink2 sm:grid-cols-4">
                  <div>F1 Δ: {fmt.delta(r.evidence.f1_delta)}</div>
                  <div>Peak memory Δ: {fmt.delta(r.evidence.memory_delta_mb, 4)} MB</div>
                  <div>Model size Δ: {fmt.delta(r.evidence.model_size_delta_kb, 1)} KB</div>
                  <div>Decision: {String(r.evidence.decision ?? "—")}</div>
                </div>
                <div className="mt-1 text-xs text-ink3">components: {Object.entries(r.score_components).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(", ")}</div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card title="Memory records" actions={
        <div className="flex gap-1">
          {["", "episodic", "semantic", "procedural"].map((c) => (
            <button key={c} onClick={() => setCategory(c)}
              className={`rounded px-2 py-0.5 text-xs ${category === c ? "bg-sunken font-medium text-ink" : "text-ink2"}`}>{c || "all"}</button>
          ))}
        </div>}>
        <ErrorBanner error={list.error} />
        <p className="mb-2 text-xs text-ink3">{list.data?.total ?? 0} records. Click a record to inspect the stored evidence behind it.</p>
        <div className="space-y-1.5">
          {list.data?.items.map((m) => (
            <div key={m.id} className="rounded border border-line p-2 text-sm">
              <button className="w-full text-left" onClick={() => setOpen(open === m.id ? null : m.id)}>
                <div className="flex flex-wrap items-center gap-2">
                  <Pill tone="accent">{m.category}</Pill><Pill>{m.kind}</Pill>
                  <Pill tone={verificationTone(m.verification_status)}>{m.verification_status}</Pill>
                  <span className="font-medium">{m.title}</span>
                  <span className="ml-auto text-xs text-ink3">{m.created_by} · {fmt.time(m.created_at)}</span>
                </div>
                <div className="mt-1 text-ink2">{m.content}</div>
              </button>
              {open === m.id && <EvidencePanel memoryId={m.id} />}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
