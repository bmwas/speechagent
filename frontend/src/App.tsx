import { useCallback, useEffect, useRef, useState } from "react";
import {
  ConnectionState,
  LocalAudioTrack,
  LocalVideoTrack,
  type LocalTrack,
  type RemoteTrack,
  Room,
  RoomEvent,
  Track,
} from "livekit-client";
import { isFailedToFetch, livekitConnectUrl, upgradeLanHttp } from "./upgradeHttps";

const ROOM = "default";
const PATH = "/patient";

const AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  channelCount: 1,
  sampleRate: 48000,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

const VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  width: { ideal: 960, min: 448 },
  height: { ideal: 540, min: 448 },
  frameRate: { ideal: 15, max: 30 },
};

function isUsableMic(d: MediaDeviceInfo): boolean {
  if (d.kind !== "audioinput" || !d.deviceId) return false;
  if (d.deviceId === "default" || d.deviceId === "communications") return false;
  const label = (d.label || "").toLowerCase();
  if (label.includes("monitor") || label.includes("loopback") || label.includes("stereo mix")) {
    return false;
  }
  return true;
}

function wrapStream(stream: MediaStream): LocalTrack[] {
  const out: LocalTrack[] = [];
  for (const media of stream.getAudioTracks()) {
    const track = new LocalAudioTrack(media, undefined, true);
    track.source = Track.Source.Microphone;
    out.push(track);
  }
  for (const media of stream.getVideoTracks()) {
    const track = new LocalVideoTrack(media, undefined, true);
    track.source = Track.Source.Camera;
    out.push(track);
  }
  return out;
}

function gumMessage(err: unknown): string {
  if (err instanceof DOMException) return `${err.name}: ${err.message}`;
  if (err instanceof Error) return err.message;
  return String(err);
}

async function acquireLocalTracks(): Promise<LocalTrack[]> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: AUDIO_CONSTRAINTS,
      video: VIDEO_CONSTRAINTS,
    });
    return wrapStream(stream);
  } catch {
    /* fall through */
  }
  const videoStream = await navigator.mediaDevices.getUserMedia({
    video: VIDEO_CONSTRAINTS,
    audio: false,
  });
  const audioAttempts: MediaStreamConstraints[] = [
    { audio: AUDIO_CONSTRAINTS, video: false },
    { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false },
    { audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false },
    { audio: true, video: false },
  ];
  let devices: MediaDeviceInfo[] = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch {
    /* ignore */
  }
  for (const d of devices) {
    if (!isUsableMic(d)) continue;
    audioAttempts.push({
      audio: { ...AUDIO_CONSTRAINTS, deviceId: { exact: d.deviceId } },
      video: false,
    });
  }
  let last = "Could not start audio source";
  for (const constraints of audioAttempts) {
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia(constraints);
      return [...wrapStream(videoStream), ...wrapStream(audioStream)];
    } catch (err) {
      last = gumMessage(err);
    }
  }
  videoStream.getTracks().forEach((t) => t.stop());
  throw new Error(
    `Microphone failed (${last}). Chrome on Linux cannot open the "default" input when it is a monitor. Pick a real mic in chrome://settings/content/microphone, then Connect again.`,
  );
}

export default function App() {
  const [error, setError] = useState<string | null>(null);
  const [state, setState] = useState("idle");
  const [lkOn, setLkOn] = useState(false);
  const [model, setModel] = useState("");
  const localRef = useRef<HTMLVideoElement>(null);
  const remoteAudioRef = useRef<HTMLAudioElement>(null);
  const lkRoomRef = useRef<Room | null>(null);
  const lkIdentityRef = useRef<string | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const insecure = typeof window !== "undefined" && !navigator.mediaDevices?.getUserMedia;
  const httpPage = typeof window !== "undefined" && window.location.protocol === "http:";
  const host = typeof window !== "undefined" ? window.location.host : "host";

  useEffect(() => {
    if (upgradeLanHttp()) return;
    if (httpPage) {
      setError(
        `This page is HTTP, so /health 308s onto HTTPS and Chrome reports Failed to fetch. Open https://${host}${PATH} — Advanced → Proceed — then reload.`,
      );
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/health", { signal: AbortSignal.timeout(8000) });
        const j = await r.json();
        if (!cancelled) {
          setModel(j.model || "");
          setError(null);
        }
      } catch (err) {
        if (cancelled) return;
        const raw = err instanceof Error ? err.message : String(err);
        setError(
          isFailedToFetch(raw)
            ? `Health check could not reach https://${host}/health (Failed to fetch). Accept the self-signed certificate (Advanced → Proceed), then reload. From a laptop use the GPU LAN IP, not 127.0.0.1.`
            : raw,
        );
      }
    })();
    return () => {
      cancelled = true;
      const lk = lkRoomRef.current;
      const identity = lkIdentityRef.current;
      if (identity) {
        fetch("/api/livekit/leave", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ room: ROOM, identity }),
        }).catch(() => undefined);
      }
      if (lk) lk.disconnect().catch(() => undefined);
    };
  }, [httpPage, host]);

  const attachRemote = (track: RemoteTrack) => {
    if (track.kind === Track.Kind.Audio && remoteAudioRef.current) {
      try {
        if (typeof track.setPlayoutDelay === "function") track.setPlayoutDelay(0.02);
        const hinted = track as unknown as { playoutDelayHint?: number };
        if ("playoutDelayHint" in hinted) hinted.playoutDelayHint = 0;
      } catch {
        /* ignore */
      }
      track.attach(remoteAudioRef.current);
      remoteAudioRef.current.play().catch(() => undefined);
    }
  };

  const connect = useCallback(async () => {
    try {
      if (!audioCtxRef.current) {
        const Ctor =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (Ctor) audioCtxRef.current = new Ctor({ latencyHint: "interactive", sampleRate: 48000 });
      }
      if (audioCtxRef.current?.state === "suspended") void audioCtxRef.current.resume();
    } catch {
      /* ignore */
    }
    if (upgradeLanHttp()) return;
    if (insecure) {
      setError(
        `Camera/mic APIs are off on ${window.location.origin}. Open https://${window.location.host}${PATH} — Advanced → Proceed — then reload. Do not switch to 127.0.0.1 — that is the GPU box, not this laptop.`,
      );
      return;
    }
    const stale = lkRoomRef.current;
    if (stale) {
      const oldId = lkIdentityRef.current;
      lkIdentityRef.current = null;
      if (oldId) {
        fetch("/api/livekit/leave", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ room: ROOM, identity: oldId }),
        }).catch(() => undefined);
      }
      try {
        await stale.disconnect();
      } catch {
        /* ignore */
      }
      lkRoomRef.current = null;
    }
    setError(null);
    setState("connecting");
    const localTracks = await acquireLocalTracks();
    const micTrack = localTracks.find((t) => t.kind === Track.Kind.Audio);
    const micDeviceId = micTrack?.mediaStreamTrack.getSettings().deviceId;
    const tokenUrl = `/api/livekit/token?role=patient&room=${ROOM}&consent=1`;
    try {
      const r = await fetch(tokenUrl, {
        method: "POST",
        signal: AbortSignal.timeout(20000),
      });
      let body: {
        ok?: boolean;
        url?: string;
        token?: string;
        identity?: string;
        error?: string;
        model?: string;
      } = {};
      try {
        body = (await r.json()) as typeof body;
      } catch {
        throw new Error(
          r.ok
            ? "token response was not JSON"
            : `token HTTP ${r.status} (not JSON — is the agent up behind this origin?)`,
        );
      }
      if (!r.ok || !body.token) throw new Error(body.error || `token HTTP ${r.status}`);
      if (body.model) setModel(body.model);
      const url = livekitConnectUrl(window.location, body.url || "");
      const videoEncoding = { maxBitrate: 1_500_000, maxFramerate: 30 };
      const room = new Room({
        adaptiveStream: false,
        dynacast: false,
        audioCaptureDefaults: {
          ...(micDeviceId ? { deviceId: { exact: micDeviceId } } : {}),
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
          voiceIsolation: false,
        },
        publishDefaults: {
          dtx: false,
          red: false,
          simulcast: false,
          degradationPreference: "maintain-resolution",
          videoEncoding,
          stopMicTrackOnMute: false,
        },
      });
      lkRoomRef.current = room;
      lkIdentityRef.current = body.identity || null;
      room.on(RoomEvent.ConnectionStateChanged, (s: ConnectionState) => setState(s));
      room.on(RoomEvent.Disconnected, () => {
        setLkOn(false);
        setState("disconnected");
      });
      room.on(RoomEvent.TrackSubscribed, (track) => attachRemote(track));
      room.on(RoomEvent.TrackUnsubscribed, (track) => track.detach());
      room.on(RoomEvent.LocalTrackPublished, (pub) => {
        if (pub.track?.kind === Track.Kind.Video && localRef.current) {
          pub.track.attach(localRef.current);
          localRef.current.play().catch(() => undefined);
        }
      });
      await room.connect(url, body.token, { websocketTimeout: 10000 });
      room.remoteParticipants.forEach((p) => {
        p.trackPublications.forEach((pub) => {
          if (pub.track) attachRemote(pub.track);
        });
      });
      for (const track of localTracks) {
        if (track.kind === Track.Kind.Audio) {
          await room.localParticipant.publishTrack(track, { source: Track.Source.Microphone });
        } else {
          await room.localParticipant.publishTrack(track, {
            source: Track.Source.Camera,
            simulcast: false,
            degradationPreference: "maintain-resolution",
            videoEncoding,
          });
          if (localRef.current) {
            track.attach(localRef.current);
            localRef.current.play().catch(() => undefined);
          }
        }
      }
      if (room.localParticipant.audioTrackPublications.size === 0) {
        throw new Error("Microphone track did not publish");
      }
      setLkOn(true);
    } catch (err) {
      for (const t of localTracks) {
        try {
          t.stop();
        } catch {
          /* ignore */
        }
      }
      const raw = err instanceof Error ? err.message : String(err);
      const msg = isFailedToFetch(raw)
        ? `Connect failed (Failed to fetch). LiveKit signaling is same-origin wss://${window.location.host}/rtc — accept the self-signed cert, stay on this host (not 127.0.0.1 from a laptop), then Connect again. (${raw})`
        : raw;
      setError(msg);
      setState("idle");
      setLkOn(false);
      const lk = lkRoomRef.current;
      const identity = lkIdentityRef.current;
      lkRoomRef.current = null;
      lkIdentityRef.current = null;
      if (identity) {
        fetch("/api/livekit/leave", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ room: ROOM, identity }),
        }).catch(() => undefined);
      }
      if (lk) lk.disconnect().catch(() => undefined);
    }
  }, [insecure]);

  const disconnect = useCallback(async () => {
    const lk = lkRoomRef.current;
    lkRoomRef.current = null;
    const identity = lkIdentityRef.current;
    lkIdentityRef.current = null;
    setLkOn(false);
    if (identity) {
      fetch("/api/livekit/leave", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ room: ROOM, identity }),
      }).catch(() => undefined);
    }
    if (lk) {
      try {
        await lk.disconnect();
      } catch {
        /* ignore */
      }
    }
    setState("idle");
  }, []);

  const connected = state !== "idle" && state !== "disconnected";

  return (
    <div className="shell">
      <div className="eyebrow">speechagent · livekit · /patient</div>
      <h1>Talk to the agent</h1>
      <p className="muted">
        Headphones on. Open this page over HTTPS, click Connect, allow camera and mic, then speak.
        The agent sees you and answers in speech.
        {model ? ` Model: ${model}.` : ""}
      </p>
      {insecure || httpPage ? (
        <div className="warn">
          Stay on <code>https://{host}{PATH}</code> — Advanced → Proceed — then reload. HTTP is not
          a secure context, so camera/mic stay blocked.
        </div>
      ) : null}
      <div className="row">
        <button type="button" onClick={() => void connect()} disabled={state === "connecting"}>
          {connected ? "Reconnect" : "Connect"}
        </button>
        <button type="button" className="secondary" onClick={() => void disconnect()} disabled={!lkOn && state === "idle"}>
          Disconnect
        </button>
        <span className="status">transport: {state}</span>
      </div>
      {error ? <p className="muted">error: {error}</p> : null}
      <div className="previews">
        <div className="tile">
          <h2>self-preview (local)</h2>
          <video ref={localRef} autoPlay muted playsInline />
        </div>
        <div className="tile">
          <h2>agent audio</h2>
          <video autoPlay playsInline />
        </div>
      </div>
      <audio ref={remoteAudioRef} autoPlay playsInline />
    </div>
  );
}
