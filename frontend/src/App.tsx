import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useApi } from "./hooks/useApi";
import ActivityPage from "./pages/ActivityPage";
import AutoresearchPage from "./pages/AutoresearchPage";
import ExperimentsPage from "./pages/ExperimentsPage";
import GraphPage from "./pages/GraphPage";
import MemoryPage from "./pages/MemoryPage";
import OverviewPage from "./pages/OverviewPage";
import { api } from "./services/api";

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/autoresearch", label: "Autoresearch" },
  { to: "/experiments", label: "Experiments" },
  { to: "/memory", label: "Shared memory" },
  { to: "/graph", label: "Knowledge graph" },
  { to: "/activity", label: "Agents & evaluation" },
];

export default function App() {
  const health = useApi(() => api.get<{ busy: boolean; execution_mode: string; llm_provider: string }>("/api/health"), [], 4000);
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-card">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <div className="font-semibold">AI Agent Memory Lab</div>
          <nav className="flex flex-wrap gap-1">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.to === "/"}
                className={({ isActive }) =>
                  `rounded px-2.5 py-1 text-sm ${isActive ? "bg-sunken font-medium text-ink" : "text-ink2 hover:text-ink"}`}>
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto text-xs text-ink3">
            {health.error ? <span className="text-critical">backend unreachable</span>
              : health.data ? <>{health.data.busy ? "● running" : "○ idle"} · {health.data.execution_mode} · LLM: {health.data.llm_provider}</>
              : "…"}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-5">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/autoresearch" element={<AutoresearchPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/experiments/:id" element={<ExperimentsPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/activity" element={<ActivityPage />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}
