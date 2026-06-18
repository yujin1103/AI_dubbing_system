import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
const BACKEND_HTTP = env?.VITE_BACKEND_URL ?? "http://webapp-backend:8000";
const BACKEND_WS = BACKEND_HTTP.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": new URL("./src", import.meta.url).pathname,
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    allowedHosts: ["desktop-71pbjk3", "fedora", ".ts.net"],
    proxy: {
      "/api": { target: BACKEND_HTTP, changeOrigin: true, ws: true },
      "/static": { target: BACKEND_HTTP, changeOrigin: true },
      "/ws": { target: BACKEND_WS, ws: true, changeOrigin: true },
      // 실시간 통역 서버(controller:8910) — 페이지+WS 를 /rt 로 프록시
      "/rt": { target: "http://controller:8910", ws: true, changeOrigin: true, rewrite: (p) => p.replace(/^\/rt/, "") },
    },
    watch: {
      usePolling: true,
      interval: 300,
    },
  },
});
