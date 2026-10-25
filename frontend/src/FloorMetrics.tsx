import { useEffect, useState } from "react";
import type { VoiceFamily } from "./voices";

type LiveData = {
  ok: boolean;
  floor?: string;
  user_speaking?: boolean;
  agent_state?: string;
  agent_audible?: boolean;
  ready?: boolean;
  speaking?: boolean;
  // Qwen
  frame_n?: number;
  utt_ms?: number;
  delta_age_ms?: number;
  speaker?: string;
  // MiniCPM vLLM
  turn_waiting?: boolean;
  rtf?: number;
  buf_ms?: number;
  // Native MiniCPM
  kv?: number;
  cost_all_ms_p50?: number;
  // Common
  ring_ms?: number;
  mic_age_ms?: number;
  family?: VoiceFamily;
  backend?: string;
};

type Props = {
  connected: boolean;
  room: string;
};

export default function FloorMetrics({ connected, room }: Props) {
  const [live, setLive] = useState<LiveData | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  if (!connected) return null;
  if (error) {
    return (
      <div className="floor-strip floor-error">
        <span className="floor-label">metrics unavailable</span>
      </div>
    );
  }
  if (!live) {
    return (
      <div className="floor-strip floor-loading">
        <span className="floor-label">connecting...</span>
      </div>
    );
  }

  const { floor, user_speaking, agent_state, family, backend } = live;

  // Floor indicators
  const userActive = user_speaking || floor === "user";
  const agentActive = agent_state === "speaking" || floor === "agent";

  // Agent state emoji/icon
  const agentIcon =
    agent_state === "speaking"
      ? "🗣️"
      : agent_state === "thinking"
        ? "💭"
        : agent_state === "listening"
          ? "👂"
          : "⏳";

  // Metrics chips based on family/backend
  const chips: { label: string; value: string | number }[] = [];

  if (family === "qwen") {
    if (live.frame_n != null) chips.push({ label: "frames", value: live.frame_n });
    if (live.utt_ms != null && live.utt_ms > 0) chips.push({ label: "utt", value: `${live.utt_ms}ms` });
    if (live.delta_age_ms != null && live.delta_age_ms >= 0)
      chips.push({ label: "delta", value: `${live.delta_age_ms}ms` });
    if (live.speaker) chips.push({ label: "voice", value: live.speaker });
  } else if (backend === "native") {
    if (live.kv != null && live.kv > 0) chips.push({ label: "KV", value: live.kv });
    if (live.cost_all_ms_p50 != null && live.cost_all_ms_p50 >= 0)
      chips.push({ label: "p50", value: `${live.cost_all_ms_p50}ms` });
    if (live.buf_ms != null) chips.push({ label: "buf", value: `${live.buf_ms}ms` });
  } else {
    // MiniCPM vLLM
    if (live.turn_waiting != null) chips.push({ label: "turn", value: live.turn_waiting ? "wait" : "ok" });
    if (live.rtf != null && live.rtf > 0) chips.push({ label: "RTF", value: live.rtf.toFixed(2) });
    if (live.buf_ms != null) chips.push({ label: "buf", value: `${live.buf_ms}ms` });
  }

  // Common metrics
  if (live.ring_ms != null && live.ring_ms > 0) chips.push({ label: "ring", value: `${live.ring_ms}ms` });

  return (
    <div className="floor-strip">
      <div className="floor-actors">
        <div className={`floor-actor ${userActive ? "active" : ""}`}>
          <span className="floor-icon">🎤</span>
          <span className="floor-name">You</span>
          {floor === "user" && <span className="floor-marker">●</span>}
        </div>
        <div className="floor-divider" />
        <div className={`floor-actor ${agentActive ? "active" : ""}`}>
          <span className="floor-icon">{agentIcon}</span>
          <span className="floor-name">Agent</span>
          {floor === "agent" && <span className="floor-marker">●</span>}
        </div>
      </div>
      {chips.length > 0 && (
        <div className="floor-chips">
          {chips.map((chip, i) => (
            <span key={i} className="floor-chip">
              <span className="chip-label">{chip.label}</span>
              <span className="chip-value">{chip.value}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
