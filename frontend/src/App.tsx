import { Suspense, lazy, useEffect, useState, type ReactNode } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useApi } from "./hooks/useApi";
import { api } from "./services/api";
import { Skeleton } from "./components/ui";

const OverviewPage = lazy(() => import("./pages/OverviewPage"));
const AutoresearchPage = lazy(() => import("./pages/AutoresearchPage"));
const ExperimentsPage = lazy(() => import("./pages/ExperimentsPage"));
const MemoryPage = lazy(() => import("./pages/MemoryPage"));
const GraphPage = lazy(() => import("./pages/GraphPage"));
const ActivityPage = lazy(() => import("./pages/ActivityPage"));

const PageFallback = () => (
  <div className="space-y-3"><Skeleton className="h-24 w-full" /><Skeleton className="h-64 w-full" /></div>
);

const icon = (d: string) => (
  <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={1.8}
    strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d={d} /></svg>
);

const NAV: { to: string; label: string; icon: ReactNode }[] = [
  { to: "/", label: "Overview", icon: icon("M3 12l9-8 9 8M5 10v10h14V10") },
  { to: "/autoresearch", label: "Autoresearch", icon: icon("M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM21 21l-5-5") },
  { to: "/experiments", label: "Experiments", icon: icon("M9 3h6M10 3v6L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2L14 9V3") },
  { to: "/memory", label: "Shared memory", icon: icon("M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6") },
  { to: "/graph", label: "Knowledge graph", icon: icon("M6 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M18 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M12 18m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M7.5 7.5l3.5 9M16.5 7.5l-3.5 9M8 6h8") },
  { to: "/activity", label: "Agents & evaluation", icon: icon("M3 12h4l3-8 4 16 3-8h4") },
];

type Theme = "system" | "light" | "dark";
const THEMES: Theme[] = ["system", "light", "dark"];

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try { return (localStorage.getItem("lab-theme") as Theme) || "system"; } catch { return "system"; }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("lab-theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  return [theme, () => setTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length])];
}

function StatusIndicator() {
  const health = useApi(() => api.get<{ busy: boolean; execution_mode: string; llm_provider: string }>("/api/health"), [], 4000);
  const [dot, label] = health.error ? ["bg-critical", "Backend unreachable"]
    : !health.data ? ["bg-ink3", "Connecting…"]
    : health.data.busy ? ["bg-accent animate-pulse", "Run in progress"] : ["bg-good", "Idle"];
  return (
    <div className="rounded-lg border border-line bg-sunken/60 px-3 py-2.5 text-xs">
      <div className="flex items-center gap-2 font-medium text-ink">
        <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />{label}
      </div>
      {health.data && (
        <div className="mt-1 text-ink3">{health.data.execution_mode} · LLM: {health.data.llm_provider}</div>
      )}
    </div>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const [theme, cycleTheme] = useTheme();
  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <div className="flex items-center gap-2.5 px-1">
        <div className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-sm font-bold text-white">M</div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-ink">Agent Memory Lab</div>
          <div className="text-[11px] text-ink3">Research workspace</div>
        </div>
      </div>
      <nav className="flex flex-col gap-0.5" aria-label="Main">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.to === "/"} onClick={onNavigate}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors ${isActive
                ? "bg-accent/10 font-medium text-accent"
                : "text-ink2 hover:bg-sunken hover:text-ink"}`}>
            {n.icon}{n.label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto flex flex-col gap-2">
        <StatusIndicator />
        <button type="button" onClick={cycleTheme}
          className="flex items-center justify-between rounded-md px-2.5 py-2 text-xs text-ink2 hover:bg-sunken hover:text-ink">
          <span>Theme</span><span className="capitalize text-ink">{theme}</span>
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const current = NAV.find((n) => (n.to === "/" ? location.pathname === "/" : location.pathname.startsWith(n.to)));
  return (
    <div className="min-h-screen lg:pl-60">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 border-r border-line bg-card lg:block">
        <Sidebar />
      </aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64 border-r border-line bg-card shadow-xl">
            <Sidebar onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
      <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-line bg-card/80 px-4 py-3 backdrop-blur lg:px-8">
        <button type="button" onClick={() => setOpen(true)} aria-label="Open navigation"
          className="rounded-md p-1.5 text-ink2 hover:bg-sunken lg:hidden">
          {icon("M4 6h16M4 12h16M4 18h16")}
        </button>
        <h1 className="text-base font-semibold text-ink">{current?.label ?? "Agent Memory Lab"}</h1>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6 lg:px-8">
        <Suspense fallback={<PageFallback />}>
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
        </Suspense>
      </main>
    </div>
  );
}
