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
import crypto from "node:crypto";
import { httpsLocation, muxLooksLikeHttp } from "./https-location.mjs";

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

const WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

function acceptProbeUpgrade(req, socket) {
  const key = req.headers["sec-websocket-key"];
  if (typeof key !== "string" || !key) {
    log("ws probe missing key");
    socket.destroy();
    return;
  }
  const accept = crypto.createHash("sha1").update(key + WS_GUID).digest("base64");
  try {
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\n" +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    socket.end();
    log("ws probe 101", req.url);
  } catch (err) {
    log("ws probe failed", err.message);
    try { socket.destroy(); } catch {}
  }
}

function livekitTarget() {
  const raw = LIVEKIT.includes("://") ? LIVEKIT : `http://${LIVEKIT}`;
  const u = new URL(raw);
  const tls = u.protocol === "https:" || u.protocol === "wss:";
  return {
    hostname: u.hostname || "127.0.0.1",
    port: Number(u.port) || (tls ? 443 : 8880),
    tls,
    host: u.host || `127.0.0.1:${u.port || 8880}`,
  };
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
    target = livekitTarget();
  } catch (err) {
    send(res, 502, `LIVEKIT_UPSTREAM is not a URL: ${err.message}\n`);
    return;
  }
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (v == null || HOP.has(k.toLowerCase())) continue;
    headers[k] = v;
  }
  headers.host = target.host;
  const lib = target.tls ? https : http;
  const pReq = lib.request(
    {
      hostname: target.hostname,
      port: target.port,
      path: req.url,
      method: req.method,
      headers,
      timeout: 8000,
    },
    (pRes) => {
      const out = { ...pRes.headers };
      delete out.connection;
      res.writeHead(pRes.statusCode || 502, out);
      pRes.pipe(res);
    },
  );
  pReq.on("error", (err) => {
    log("livekit http proxy error", err.message, `${target.hostname}:${target.port}`, req.url);
    if (!res.headersSent) send(res, 502, `livekit proxy: ${err.message}\n`);
    else res.destroy();
  });
  pReq.on("timeout", () => {
    log("livekit http proxy timeout", `${target.hostname}:${target.port}`, req.url);
    pReq.destroy();
  });
  req.pipe(pReq);
}

function proxyLivekitUpgrade(req, socket, head) {
  let target;
  try {
    target = livekitTarget();
  } catch (err) {
    log("livekit ws bad LIVEKIT_UPSTREAM", err.message);
    socket.destroy();
    return;
  }
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (v == null) continue;
    if (k.toLowerCase() === "host") continue;
    headers[k] = v;
  }
  headers.host = target.host;
  if (!headers.connection) headers.connection = "Upgrade";
  if (!headers.upgrade) headers.upgrade = "websocket";
  log("livekit ws upgrade", req.url, "→", `${target.hostname}:${target.port}`);
  const lib = target.tls ? https : http;
  const pReq = lib.request({
    hostname: target.hostname,
    port: target.port,
    path: req.url || "/rtc",
    method: req.method || "GET",
    headers,
    timeout: 8000,
  });
  pReq.on("upgrade", (pRes, pSocket, pHead) => {
    const lines = [`HTTP/1.1 ${pRes.statusCode || 101} ${pRes.statusMessage || "Switching Protocols"}`];
    for (const [k, v] of Object.entries(pRes.headers)) {
      if (v == null) continue;
      const val = Array.isArray(v) ? v.join(", ") : v;
      lines.push(`${k}: ${val}`);
    }
    try {
      socket.write(lines.join("\r\n") + "\r\n\r\n");
      if (pHead && pHead.length) socket.write(pHead);
      if (head && head.length) pSocket.write(head);
      pSocket.pipe(socket);
      socket.pipe(pSocket);
    } catch (err) {
      log("livekit ws pipe failed", err.message);
      try { pSocket.destroy(); } catch {}
      try { socket.destroy(); } catch {}
      return;
    }
    pSocket.on("error", () => {
      try { socket.destroy(); } catch {}
    });
    socket.on("error", () => {
      try { pSocket.destroy(); } catch {}
    });
    log("livekit ws 101", req.url);
  });
  pReq.on("response", (pRes) => {
    log("livekit ws expected 101, got HTTP", pRes.statusCode, req.url);
    try {
      socket.write(`HTTP/1.1 ${pRes.statusCode} ${pRes.statusMessage || ""}\r\nConnection: close\r\n\r\n`);
    } catch {
      /* ignore */
    }
    pRes.pipe(socket);
  });
  pReq.on("error", (err) => {
    log("livekit ws proxy error", err.message, "upstream", `${target.hostname}:${target.port}`);
    try { socket.destroy(); } catch {}
  });
  pReq.on("timeout", () => {
    log("livekit ws proxy timeout", `${target.hostname}:${target.port}`);
    pReq.destroy();
    try { socket.destroy(); } catch {}
  });
  pReq.end();
}

function probeLivekit() {
  let target;
  try {
    target = livekitTarget();
  } catch (err) {
    log("livekit probe skipped", err.message);
    return;
  }
  const lib = target.tls ? https : http;
  const req = lib.get(
    {
      hostname: target.hostname,
      port: target.port,
      path: "/",
      timeout: 3000,
    },
    (res) => {
      log("livekit probe", `${target.hostname}:${target.port}`, "HTTP", res.statusCode);
      res.resume();
    },
  );
  req.on("error", (err) => {
    log(
      "livekit probe FAILED",
      `${target.hostname}:${target.port}`,
      err.message,
      "— browser Connect will time out on wss /rtc until LiveKit listens here",
    );
  });
  req.on("timeout", () => {
    log("livekit probe timeout", `${target.hostname}:${target.port}`);
    req.destroy();
  });
}

function sendJson(res, status, obj) {
  const buf = Buffer.from(JSON.stringify(obj));
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": buf.length,
    "cache-control": "no-store",
  });
  res.end(buf);
}

function livekitDiag(req, res) {
  let target;
  try {
    target = livekitTarget();
  } catch (err) {
    sendJson(res, 500, { ok: false, error: err.message, upstream: LIVEKIT });
    return;
  }
  let sent = false;
  const done = (status, obj) => {
    if (sent) return;
    sent = true;
    sendJson(res, status, obj);
  };
  const started = Date.now();
  const socket = net.connect(target.port, target.hostname, () => {
    const ms = Date.now() - started;
    socket.end();
    done(200, {
      ok: true,
      upstream: `${target.hostname}:${target.port}`,
      tcp_ms: ms,
      path: "/rtc",
    });
  });
  socket.setTimeout(3000);
  socket.on("timeout", () => {
    socket.destroy();
    done(503, {
      ok: false,
      upstream: `${target.hostname}:${target.port}`,
      error: "tcp timeout",
    });
  });
  socket.on("error", (err) => {
    done(503, {
      ok: false,
      upstream: `${target.hostname}:${target.port}`,
      error: err.message,
    });
  });
}

function serveStatic(req, res) {
  const pathname = pathnameOf(req.url);
  if (pathname === "/lk-diag") {
    livekitDiag(req, res);
    return;
  }
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
    if (pathname === "/lk-ws-probe") {
      acceptProbeUpgrade(req, socket);
      return;
    }
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
      const httpPlain = muxLooksLikeHttp(first);
      if (!httpPlain && first[0] !== 0x16 && first[0] !== 0x80) {
        log("mux tls-ish first=0x" + first[0].toString(16));
      }
      const target = httpPlain ? INNER_HTTP : INNER_HTTPS;
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
  probeLivekit();
}

main().catch((err) => {
  console.error("capture-ui boot failed", err);
  process.exit(1);
});
