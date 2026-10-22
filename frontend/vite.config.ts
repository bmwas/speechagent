import fs from "node:fs";
import path from "node:path";
import { defineConfig, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";

const media = process.env.MEDIA_UPSTREAM || "http://127.0.0.1:7860";
const livekit = process.env.LIVEKIT_UPSTREAM || "http://127.0.0.1:7880";

const proxyOne: ProxyOptions = {
  target: media,
  changeOrigin: true,
  xfwd: true,
  configure(proxy) {
    proxy.on("proxyReq", (proxyReq, req) => {
      const host = req.headers.host;
      if (typeof host === "string" && host) {
        proxyReq.setHeader("x-forwarded-host", host);
      }
    });
  },
};

const proxy = {
  "/health": proxyOne,
  "/api": proxyOne,
  "/rtc": {
    target: livekit,
    ws: true,
    changeOrigin: true,
  } satisfies ProxyOptions,
};

const certDir = process.env.TLS_CERT_DIR || path.resolve("certs");
const keyPath = path.join(certDir, "key.pem");
const certPath = path.join(certDir, "cert.pem");
const https =
  fs.existsSync(keyPath) && fs.existsSync(certPath)
    ? { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }
    : undefined;

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, https, proxy, allowedHosts: true },
  preview: { host: true, port: 5173, strictPort: true, https, proxy, allowedHosts: true },
});
