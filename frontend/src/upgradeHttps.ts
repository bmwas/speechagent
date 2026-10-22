/** LAN http:// is not a secure context. Never send the user to GPU 127.0.0.1. */
export function isLoopbackHostname(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "[::1]" ||
    hostname === "::1"
  );
}

export function lanHttpsUrl(loc: Location | URL): string | null {
  if (loc.protocol !== "http:") return null;
  if (isLoopbackHostname(loc.hostname)) {
    return null;
  }
  return `https://${loc.host}${loc.pathname}${loc.search}${loc.hash}`;
}

export function upgradeLanHttp(): boolean {
  if (typeof window === "undefined") return false;
  const next = lanHttpsUrl(window.location);
  if (!next) return false;
  window.location.replace(next);
  return true;
}

/**
 * LiveKit Room.connect URL. Prefer the page origin (capture-ui muxes /rtc)
 * whenever the token URL would be mixed-content or GPU loopback on a laptop.
 */
export function livekitConnectUrl(
  page: { protocol: string; host: string; hostname: string },
  bodyUrl: string,
): string {
  const wantSecure = page.protocol === "https:";
  const pageWs = `${wantSecure ? "wss" : "ws"}://${page.host}`;
  const given = (bodyUrl || "").trim();
  if (!given) return pageWs;
  let u: URL;
  try {
    u = new URL(given);
  } catch {
    return pageWs;
  }
  const givenSecure = u.protocol === "wss:" || u.protocol === "https:";
  if (wantSecure && !givenSecure) return pageWs;
  if (isLoopbackHostname(u.hostname) && !isLoopbackHostname(page.hostname)) return pageWs;
  if (u.protocol === "http:" || u.protocol === "https:") {
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.toString().replace(/\/$/, "");
  }
  return given.replace(/\/$/, "");
}

export function isFailedToFetch(msg: string): boolean {
  const s = (msg || "").toLowerCase();
  return s === "failed to fetch" || s.includes("failed to fetch") || s.includes("networkerror");
}
