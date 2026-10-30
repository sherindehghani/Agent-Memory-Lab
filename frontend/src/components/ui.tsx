import type { ReactNode } from "react";
import type { Decision } from "../types";

export function Card({ title, children, actions, className = "" }: {
  title?: ReactNode; children: ReactNode; actions?: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-card p-5 shadow-sm ${className}`}>
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-xl border border-line bg-card px-5 py-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-ink3">{label}</div>
      <div className="mt-1.5 text-3xl font-semibold tabular-nums tracking-tight text-ink">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-ink3">{hint}</div>}
    </div>
  );
}

const DECISION_STYLE: Record<string, { icon: string; cls: string }> = {
  accepted: { icon: "✓", cls: "text-good bg-good/10 border-good/30" },
  rejected: { icon: "–", cls: "text-ink2 bg-sunken border-line" },
  invalid: { icon: "✕", cls: "text-critical bg-critical/10 border-critical/30" },
};

export function DecisionBadge({ decision }: { decision: Decision | string | null }) {
  if (!decision) return <span className="text-xs text-ink3">—</span>;
  const s = DECISION_STYLE[decision] ?? { icon: "•", cls: "text-ink2 border-line" };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${s.cls}`}>
      <span aria-hidden>{s.icon}</span>
      {decision}
    </span>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" | "good" | "critical" }) {
  const cls = {
    neutral: "border-line bg-sunken text-ink2",
    accent: "border-accent/30 bg-accent/10 text-accent",
    good: "border-good/30 bg-good/10 text-good",
    critical: "border-critical/30 bg-critical/10 text-critical",
  }[tone];
  return <span className={`inline-block rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>{children}</span>;
}

export function verificationTone(s: string): "good" | "critical" | "neutral" {
  return s === "verified" ? "good" : s === "contradicted" ? "critical" : "neutral";
}

export function ErrorBanner({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <div role="alert" className="mb-4 rounded-lg border border-critical/40 bg-critical/10 px-3 py-2 text-sm text-critical">
      {error}
    </div>
  );
}

export function Json({ value }: { value: unknown }) {
  return (
    <pre className="max-h-72 overflow-auto rounded-lg border border-line bg-sunken p-3 font-mono text-xs leading-relaxed text-ink2">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function Button({ children, onClick, disabled, variant = "primary", type = "button" }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; variant?: "primary" | "secondary"; type?: "button" | "submit";
}) {
  const cls = variant === "primary"
    ? "bg-accent text-white shadow-sm hover:brightness-110 active:brightness-95"
    : "border border-line bg-card text-ink shadow-sm hover:bg-sunken";
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={`inline-flex items-center gap-1.5 rounded-md px-3.5 py-2 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 disabled:cursor-not-allowed disabled:opacity-50 ${cls}`}>
      {children}
    </button>
  );
}

export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-sunken ${className}`} />;
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-4 py-10 text-center">
      <div className="text-sm font-medium text-ink2">{title}</div>
      {hint && <div className="mt-1 text-xs text-ink3">{hint}</div>}
    </div>
  );
}

export const fmt = {
  f1: (v: unknown) => (typeof v === "number" ? v.toFixed(4) : "—"),
  delta: (v: unknown, digits = 4) => (typeof v === "number" ? `${v >= 0 ? "+" : ""}${v.toFixed(digits)}` : "—"),
  num: (v: unknown, digits = 3) => (typeof v === "number" ? v.toFixed(digits) : "—"),
  time: (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : "—"),
  config: (c: Record<string, unknown>) => Object.entries(c).map(([k, v]) => `${k}=${v ?? "None"}`).join(", "),
  short: (id: string | null | undefined) => (id ? id.replace(/^(\w+)_/, "$1_").slice(0, 16) : "—"),
};
