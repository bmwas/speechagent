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
 * LiveKit Room.connect URL. Prefer the page origin (capture-ui muxes /rtc)
 * whenever the token URL would be mixed-content or GPU loopback on a laptop.
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
  const givenSecure = u.protocol === "wss:" || u.protocol === "https:";
  if (wantSecure && !givenSecure) return pageWs;
  if (isLoopbackHostname(u.hostname) && !isLoopbackHostname(hostname)) return pageWs;
  if (u.protocol === "http:" || u.protocol === "https:") {
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.toString().replace(/\/$/, "");
  }
  return given.replace(/\/$/, "");
}

export function isFailedToFetch(msg) {
  const s = String(msg || "").toLowerCase();
  return s === "failed to fetch" || s.includes("failed to fetch") || s.includes("networkerror");
}
