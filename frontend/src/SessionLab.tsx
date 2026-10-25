import { useEffect, useState, useRef } from "react";

type TurnMetrics = {
  ttfa_ms: number;
  ttfa_anchor: string;
  ttft_ms: number;
  rtf: number;
  gap_ms: number;
  turn_audio_ms: number;
};

type SessionStats = {
  p50: number;
  p95: number;
};

type SessionAggregates = {
  n: number;
  duration_s: number;
  ttfa: SessionStats;
  ttft: SessionStats;
  rtf: SessionStats;
  gap: SessionStats;
  prebuffer: SessionStats;
  e2e: SessionStats;
  underruns_total: number;
  clock_slips: number;
};

type LiveData = {
  ok: boolean;
  model?: string;
  family?: string;
  backend?: string;
  voice_id?: string;
  turn?: TurnMetrics;
  session?: SessionAggregates;
};

type WebRTCStats = {
  rtt_ms: number;
  jitter_ms: number;
  packets_lost: number;
  packets_received: number;
  concealed_samples: number;
  total_samples: number;
};

type Props = {
  connected: boolean;
  room: string;
  peerConnection?: RTCPeerConnection | null;
};

function formatMs(v: number | undefined | null): string {
  if (v == null || v < 0) return "—";
  return `${Math.round(v)}ms`;
}

function formatRtf(v: number | undefined | null): string {
  if (v == null || v < 0) return "—";
  return v.toFixed(2);
}

function formatPercent(num: number, denom: number): string {
  if (denom <= 0) return "—";
  const pct = (num / denom) * 100;
  return `${pct.toFixed(2)}%`;
}

export default function SessionLab({ connected, room, peerConnection }: Props) {
  const [live, setLive] = useState<LiveData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [webrtc, setWebrtc] = useState<WebRTCStats | null>(null);
  const prevInboundRef = useRef<{
    packetsLost: number;
    packetsReceived: number;
    concealedSamples: number;
    totalSamplesReceived: number;
  } | null>(null);

  // Poll /api/live while connected
  useEffect(() => {
    if (!connected) {
      setLive(null);
      setError(null);
      return;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const r = await fetch(`/api/live?room=${encodeURIComponent(room)}`, {
          signal: AbortSignal.timeout(4000),
        });
        const j = await r.json();
        if (!cancelled) {
          if (j.ok) {
            setLive(j);
            setError(null);
          } else {
            setLive(null);
            setError(j.error || "no session");
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    };

    poll();
    const id = setInterval(poll, 500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [connected, room]);

  // Poll WebRTC stats
  useEffect(() => {
    if (!connected || !peerConnection) {
      setWebrtc(null);
      prevInboundRef.current = null;
      return;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const stats = await peerConnection.getStats();
        let rtt_ms = -1;
        let jitter_ms = -1;
        let packets_lost = 0;
        let packets_received = 0;
        let concealed_samples = 0;
        let total_samples = 0;

        stats.forEach((report) => {
          // Get RTT from candidate-pair
          if (report.type === "candidate-pair" && report.nominated) {
            if (report.currentRoundTripTime != null) {
              rtt_ms = report.currentRoundTripTime * 1000;
            }
          }
          // Get jitter and packet stats from inbound-rtp for audio
          if (report.type === "inbound-rtp" && report.kind === "audio") {
            if (report.jitter != null) {
              jitter_ms = report.jitter * 1000;
            }
            if (report.packetsLost != null) {
              packets_lost = report.packetsLost;
            }
            if (report.packetsReceived != null) {
              packets_received = report.packetsReceived;
            }
            if (report.concealedSamples != null) {
              concealed_samples = report.concealedSamples;
            }
            if (report.totalSamplesReceived != null) {
              total_samples = report.totalSamplesReceived;
            }
          }
        });

        // Compute deltas from previous poll for rates
        const prev = prevInboundRef.current;
        if (prev) {
          const deltaLost = packets_lost - prev.packetsLost;
          const deltaReceived = packets_received - prev.packetsReceived;
          const deltaConcealed = concealed_samples - prev.concealedSamples;
          const deltaSamples = total_samples - prev.totalSamplesReceived;

          if (!cancelled) {
            setWebrtc({
              rtt_ms,
              jitter_ms,
              packets_lost: deltaLost,
              packets_received: deltaReceived,
              concealed_samples: deltaConcealed,
              total_samples: deltaSamples,
            });
          }
        }

        prevInboundRef.current = {
          packetsLost: packets_lost,
          packetsReceived: packets_received,
          concealedSamples: concealed_samples,
          totalSamplesReceived: total_samples,
        };
      } catch {
        // getStats can fail if connection is closed
      }
    };

    poll();
    const id = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [connected, peerConnection]);

  if (!connected) {
    return (
      <div className="session-lab session-lab-idle">
        <h3>Session Lab</h3>
        <div className="lab-row">
          <span className="lab-label">Connect to start scoring</span>
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="session-lab session-lab-error">
        <span>session lab unavailable: {error}</span>
      </div>
    );
  }
  if (!live) {
    return (
      <div className="session-lab session-lab-loading">
        <span>loading session lab...</span>
      </div>
    );
  }

  const { model, family, backend, voice_id, turn, session } = live;

  return (
    <div className="session-lab">
      <h3>Session Lab</h3>

      {/* Model info */}
      <div className="lab-section">
        <div className="lab-section-title">Model</div>
        <div className="lab-row">
          <span className="lab-label">Model:</span>
          <span className="lab-value">{model || "—"}</span>
        </div>
        <div className="lab-row">
          <span className="lab-label">Family:</span>
          <span className="lab-value">{family || "—"}</span>
        </div>
        <div className="lab-row">
          <span className="lab-label">Backend:</span>
          <span className="lab-value">{backend || "—"}</span>
        </div>
        <div className="lab-row">
          <span className="lab-label">Voice:</span>
          <span className="lab-value">{voice_id || "—"}</span>
        </div>
      </div>

      {/* WebRTC network stats */}
      <div className="lab-section">
        <div className="lab-section-title">Network (WebRTC)</div>
        {webrtc ? (
          <>
            <div className="lab-row">
              <span className="lab-label">RTT:</span>
              <span className="lab-value">{formatMs(webrtc.rtt_ms)}</span>
            </div>
            <div className="lab-row">
              <span className="lab-label">Jitter:</span>
              <span className="lab-value">{formatMs(webrtc.jitter_ms)}</span>
            </div>
            <div className="lab-row">
              <span className="lab-label">Packet loss:</span>
              <span className="lab-value">
                {webrtc.packets_lost}/{webrtc.packets_received + webrtc.packets_lost} (
                {formatPercent(webrtc.packets_lost, webrtc.packets_received + webrtc.packets_lost)})
              </span>
            </div>
            <div className="lab-row">
              <span className="lab-label">Concealed:</span>
              <span className="lab-value">
                {formatPercent(webrtc.concealed_samples, webrtc.total_samples)}
              </span>
            </div>
          </>
        ) : (
          <div className="lab-row">
            <span className="lab-label">Waiting for stats...</span>
          </div>
        )}
      </div>

      {/* Turn metrics */}
      <div className="lab-section">
        <div className="lab-section-title">Last Turn</div>
        {turn ? (
          <>
            <div className="lab-row">
              <span className="lab-label">TTFA ({turn.ttfa_anchor || "query"}):</span>
              <span className="lab-value">{formatMs(turn.ttfa_ms)}</span>
            </div>
            <div className="lab-row">
              <span className="lab-label">TTFT:</span>
              <span className="lab-value">{formatMs(turn.ttft_ms)}</span>
            </div>
            <div className="lab-row">
              <span className="lab-label">RTF:</span>
              <span className="lab-value">{formatRtf(turn.rtf)}</span>
            </div>
            <div className="lab-row">
              <span className="lab-label">Max gap:</span>
              <span className="lab-value">{formatMs(turn.gap_ms)}</span>
            </div>
            <div className="lab-row">
              <span className="lab-label">Audio:</span>
              <span className="lab-value">{formatMs(turn.turn_audio_ms)}</span>
            </div>
          </>
        ) : (
          <div className="lab-row">
            <span className="lab-label">No turn data yet</span>
          </div>
        )}
      </div>

      {/* Session aggregates */}
      <div className="lab-section">
        <div className="lab-section-title">Session ({session?.n || 0} turns, {session?.duration_s?.toFixed(0) || 0}s)</div>
        {session && session.n > 0 ? (
          <table className="lab-stats-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>p50</th>
                <th>p95</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>TTFA</td>
                <td>{formatMs(session.ttfa?.p50)}</td>
                <td>{formatMs(session.ttfa?.p95)}</td>
              </tr>
              <tr>
                <td>TTFT</td>
                <td>{formatMs(session.ttft?.p50)}</td>
                <td>{formatMs(session.ttft?.p95)}</td>
              </tr>
              <tr>
                <td>RTF</td>
                <td>{formatRtf(session.rtf?.p50)}</td>
                <td>{formatRtf(session.rtf?.p95)}</td>
              </tr>
              <tr>
                <td>Gap</td>
                <td>{formatMs(session.gap?.p50)}</td>
                <td>{formatMs(session.gap?.p95)}</td>
              </tr>
              <tr>
                <td>Prebuffer</td>
                <td>{formatMs(session.prebuffer?.p50)}</td>
                <td>{formatMs(session.prebuffer?.p95)}</td>
              </tr>
              <tr>
                <td>E2E</td>
                <td>{formatMs(session.e2e?.p50)}</td>
                <td>{formatMs(session.e2e?.p95)}</td>
              </tr>
            </tbody>
          </table>
        ) : (
          <div className="lab-row">
            <span className="lab-label">No session data yet</span>
          </div>
        )}
        {session && (
          <div className="lab-row">
            <span className="lab-label">Underruns:</span>
            <span className="lab-value">{session.underruns_total || 0}</span>
            <span className="lab-label" style={{ marginLeft: "1em" }}>Slips:</span>
            <span className="lab-value">{session.clock_slips || 0}</span>
          </div>
        )}
      </div>
    </div>
  );
}
