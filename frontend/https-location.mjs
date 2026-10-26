/** Host header → host:port. Never drop the UI port (https://10.x.x.x without :6173 is :443). */
export function hostWithPort(hostHdr, port) {
  const p = Number(port) || 6173;
  if (!hostHdr) return `127.0.0.1:${p}`;
  if (hostHdr.startsWith("[")) {
    if (/\]:\d+$/.test(hostHdr)) return hostHdr;
    return `${hostHdr}:${p}`;
  }
  if (/:\d+$/.test(hostHdr)) return hostHdr;
  return `${hostHdr}:${p}`;
}

export function httpsLocation(reqUrl, hostHdr, port) {
  return `https://${hostWithPort(hostHdr, port)}${reqUrl || "/"}`;
}

export function isLoopbackHostname(hostname) {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "[::1]" ||
    hostname === "::1"
  );
}

/** Browser-side: LAN http:// must become https:// on the same host:port. */
export function lanHttpsUrl(loc) {
  if (!loc || loc.protocol !== "http:") return null;
  if (isLoopbackHostname(loc.hostname)) return null;
  return `https://${loc.host}${loc.pathname || "/"}${loc.search || ""}${loc.hash || ""}`;
}

/**
 * LiveKit Room.connect URL. Always same-origin (capture-ui muxes /rtc).
 */
export function livekitConnectUrl(page, bodyUrl) {
  const host = page && page.host ? page.host : "";
  const protocol = page && page.protocol ? page.protocol : "https:";
  const hostname = page && page.hostname ? page.hostname : "";
  const wantSecure = protocol === "https:";
  const pageWs = `${wantSecure ? "wss" : "ws"}://${host}`;
  const given = String(bodyUrl || "").trim();
  if (!given) return pageWs;
  let u;
  try {
    u = new URL(given);
  } catch {
    return pageWs;
  }
  if (u.hostname !== hostname) return pageWs;
  const givenSecure = u.protocol === "wss:" || u.protocol === "https:";
  if (wantSecure && !givenSecure) return pageWs;
  if (isLoopbackHostname(u.hostname) && !isLoopbackHostname(hostname)) return pageWs;
  return pageWs;
}

export function isFailedToFetch(msg) {
  const s = String(msg || "").toLowerCase();
  return s === "failed to fetch" || s.includes("failed to fetch") || s.includes("networkerror");
}

/**
 * TCP mux sniff. Only clear HTTP methods go to the redirector.
 * Everything else — including TLS 1.3 appdata (0x17), alerts, and
 * resumed handshakes — must stay on the HTTPS server. Treating 0x17 as
 * HTTP sent LiveKit wss:/rtc into the 308 path and the browser timed out.
 */
export function muxLooksLikeHttp(first) {
  if (!first || first.length === 0) return false;
  const b = first[0];
  if (b === 0x14 || b === 0x15 || b === 0x16 || b === 0x17 || b === 0x80) return false;
  const s = Buffer.from(first.subarray(0, Math.min(first.length, 12))).toString("latin1");
  return /^(GET|POST|HEAD|PUT|PATCH|DELETE|OPTIONS|CONNECT)[ /]/i.test(s);
}
