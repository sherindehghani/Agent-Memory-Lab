import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API calls go to /api; in dev they are proxied to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000" },
  },
});
