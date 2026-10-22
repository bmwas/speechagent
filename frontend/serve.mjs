#!/usr/bin/env node
/**
 * Capture UI: HTTP + HTTPS on the same port (default 6173; msestream uses 5173).
 *
 * Chrome hides getUserMedia on http://<LAN-IP>, so a laptop on the LAN cannot
 * open camera/mic from that origin. This process:
 *   - serves HTTPS (self-signed) on :6173                 → secure context
 *   - serves HTTP  on :6173 and 308s to https://<same>:6173/... → never 127.0.0.1
 * The muxing is a TCP byte-sniff proxy to two loopback servers; using
 * `Server.emit('connection', socket)` on https.Server does not drive the TLS
 * handshake, so we hand off a raw TCP pipe instead.
 */
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { httpsLocation } from "./https-location.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 6173);
const INNER_HTTP = Number(process.env.INNER_HTTP_PORT || 15173);
const INNER_HTTPS = Number(process.env.INNER_HTTPS_PORT || 15174);
const DIST = path.resolve(process.env.DIST_DIR || path.join(__dirname, "dist"));
const MEDIA = process.env.MEDIA_UPSTREAM || "http://127.0.0.1:8860";
const LIVEKIT = process.env.LIVEKIT_UPSTREAM || "http://127.0.0.1:8880";
const CERT_DIR = process.env.TLS_CERT_DIR || path.resolve(__dirname, "certs");
const SPA = new Set(["/", "/patient", "/clinician"]);
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".map": "application/json",
  ".txt": "text/plain; charset=utf-8",
};
const HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
  "host",
]);

function log(...args) {
  console.log(new Date().toISOString(), ...args);
}

function loadTls() {
  const key = path.join(CERT_DIR, "key.pem");
  const cert = path.join(CERT_DIR, "cert.pem");
  if (!fs.existsSync(key) || !fs.existsSync(cert)) {
    console.error(`capture-ui: missing TLS files in ${CERT_DIR} (entrypoint should have written them)`);
    process.exit(1);
  }
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
}

function pathnameOf(reqUrl) {
  try {
    return decodeURIComponent(new URL(reqUrl || "/", "http://local").pathname);
  } catch {
    return "/";
  }
}

function isMediaPath(pathname) {
  return pathname === "/health" || pathname === "/drops" || pathname.startsWith("/api");
}

function isLivekitPath(pathname) {
  return pathname === "/rtc" || pathname.startsWith("/rtc/");
}

function send(res, status, body, headers = {}) {
  const buf = Buffer.from(body);
  res.writeHead(status, {
    "content-type": "text/plain; charset=utf-8",
    "content-length": buf.length,
    "cache-control": "no-store",
    ...headers,
  });
  res.end(buf);
}

function redirectHttps(req, res) {
  const loc = httpsLocation(req.url, req.headers.host, PORT);
  log("308", req.url, "→", loc);
  send(res, 308, `redirect ${loc}\n`, { location: loc });
}

function safeFile(urlPath) {
  const rel = pathnameOf(urlPath);
  const full = path.normalize(path.join(DIST, rel));
  const root = DIST.endsWith(path.sep) ? DIST : DIST + path.sep;
  if (full !== DIST && !full.startsWith(root)) return null;
  return full;
}

function proxyMedia(req, res) {
  let target;
  try {
    target = new URL(MEDIA);
  } catch {
    send(res, 502, "MEDIA_UPSTREAM is not a URL\n");
    return;
  }
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (v == null || HOP.has(k.toLowerCase())) continue;
    headers[k] = v;
  }
  headers.host = target.host;
  headers["x-forwarded-host"] = req.headers.host || "";
  headers["x-forwarded-proto"] = "https";
  const pReq = http.request(
    {
      hostname: target.hostname,
      port: target.port || (target.protocol === "https:" ? 443 : 80),
      path: req.url,
      method: req.method,
      headers,
    },
    (pRes) => {
      const out = { ...pRes.headers };
      delete out.connection;
      res.writeHead(pRes.statusCode || 502, out);
      pRes.pipe(res);
    },
  );
  pReq.on("error", (err) => {
    log("media proxy error", err.message);
    if (!res.headersSent) send(res, 502, `media proxy: ${err.message}\n`);
    else res.destroy();
  });
  req.pipe(pReq);
}

function proxyLivekitHttp(req, res) {
  let target;
  try {
    target = new URL(LIVEKIT);
  } catch {
    send(res, 502, "LIVEKIT_UPSTREAM is not a URL\n");
    return;
  }
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (v == null || HOP.has(k.toLowerCase())) continue;
    headers[k] = v;
  }
  headers.host = target.host;
  const pReq = http.request(
    {
      hostname: target.hostname,
      port: target.port || 8880,
      path: req.url,
      method: req.method,
      headers,
    },
    (pRes) => {
      const out = { ...pRes.headers };
      delete out.connection;
      res.writeHead(pRes.statusCode || 502, out);
      pRes.pipe(res);
    },
  );
  pReq.on("error", (err) => {
    log("livekit proxy error", err.message);
    if (!res.headersSent) send(res, 502, `livekit proxy: ${err.message}\n`);
    else res.destroy();
  });
  req.pipe(pReq);
}

function proxyLivekitUpgrade(req, socket, head) {
  let target;
  try {
    target = new URL(LIVEKIT);
  } catch {
    socket.destroy();
    return;
  }
  const port = Number(target.port) || 8880;
  const upstream = net.connect(port, target.hostname, () => {
    const lines = [`${req.method} ${req.url} HTTP/1.1`];
    for (const [k, v] of Object.entries(req.headers)) {
      if (v == null) continue;
      const val = Array.isArray(v) ? v.join(", ") : v;
      if (k.toLowerCase() === "host") lines.push(`Host: ${target.host}`);
      else lines.push(`${k}: ${val}`);
    }
    if (!req.headers.host) lines.push(`Host: ${target.host}`);
    upstream.write(lines.join("\r\n") + "\r\n\r\n");
    if (head && head.length) upstream.write(head);
    socket.pipe(upstream);
    upstream.pipe(socket);
  });
  upstream.on("error", (err) => {
    log("livekit ws proxy error", err.message);
    try {
      socket.destroy();
    } catch {
      /* already closed */
    }
  });
  socket.on("error", () => {
    try {
      upstream.destroy();
    } catch {
      /* already closed */
    }
  });
}

function serveStatic(req, res) {
  const pathname = pathnameOf(req.url);
  if (isMediaPath(pathname)) {
    proxyMedia(req, res);
    return;
  }
  if (isLivekitPath(pathname)) {
    proxyLivekitHttp(req, res);
    return;
  }
  let file = safeFile(req.url);
  if (file && fs.existsSync(file) && fs.statSync(file).isDirectory()) {
    file = path.join(file, "index.html");
  }
  if ((!file || !fs.existsSync(file)) && SPA.has(pathname.replace(/\/$/, "") || "/")) {
    file = path.join(DIST, "index.html");
  }
  if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    send(res, 404, "not found\n");
    return;
  }
  const ext = path.extname(file).toLowerCase();
  const html = ext === ".html";
  res.writeHead(200, {
    "content-type": MIME[ext] || "application/octet-stream",
    "cache-control": html ? "no-store" : "public, max-age=60",
  });
  fs.createReadStream(file).pipe(res);
}

function onHttps(req, res) {
  try {
    serveStatic(req, res);
  } catch (err) {
    log("https handler", err);
    if (!res.headersSent) send(res, 500, "internal error\n");
  }
}

async function listen(server, port, host) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => {
      server.off("error", reject);
      resolve();
    });
  });
}

async function main() {
  if (!fs.existsSync(path.join(DIST, "index.html"))) {
    console.error("capture-ui: dist/index.html missing — image build did not run vite");
    process.exit(1);
  }
  const tls = loadTls();
  const httpsServer = https.createServer(tls, onHttps);
  const httpServer = http.createServer(redirectHttps);
  httpsServer.on("upgrade", (req, socket, head) => {
    const pathname = pathnameOf(req.url);
    if (isLivekitPath(pathname)) {
      proxyLivekitUpgrade(req, socket, head);
      return;
    }
    socket.destroy();
  });
  httpsServer.on("clientError", (err, socket) => {
    log("https clientError", err.code || err.message);
    try { socket.destroy(); } catch {}
  });
  httpServer.on("clientError", (err, socket) => {
    try { socket.destroy(); } catch {}
    log("http clientError", err.code || err.message);
  });

  await listen(httpsServer, INNER_HTTPS, "127.0.0.1");
  await listen(httpServer, INNER_HTTP, "127.0.0.1");

  const mux = net.createServer((client) => {
    client.once("error", () => client.destroy());
    client.once("data", (first) => {
      if (!first || first.length === 0) {
        client.destroy();
        return;
      }
      const tlsHello = first[0] === 0x16 || first[0] === 0x80;
      const target = tlsHello ? INNER_HTTPS : INNER_HTTP;
      const upstream = net.connect(target, "127.0.0.1");
      upstream.once("error", (err) => {
        log("mux upstream error", err.message);
        try { client.destroy(); } catch {}
      });
      upstream.once("connect", () => {
        upstream.write(first);
        client.pipe(upstream);
        upstream.pipe(client);
      });
    });
  });
  mux.on("error", (err) => {
    console.error("capture-ui mux listen failed", err);
    process.exit(1);
  });
  await listen(mux, PORT, "0.0.0.0");
  log(`capture-ui mux :${PORT}  http→https  media=${MEDIA}  livekit=${LIVEKIT}  inner http=${INNER_HTTP} https=${INNER_HTTPS}`);
}

main().catch((err) => {
  console.error("capture-ui boot failed", err);
  process.exit(1);
});
