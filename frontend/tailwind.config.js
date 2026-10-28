export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "var(--surface-0)",
        card: "var(--surface-1)",
        sunken: "var(--surface-2)",
        line: "var(--border)",
        ink: "var(--text-primary)",
        ink2: "var(--text-secondary)",
        ink3: "var(--text-muted)",
        accent: "var(--accent)",
        good: "var(--status-good)",
        critical: "var(--status-critical)",
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
    },
  },
  plugins: [],
};
