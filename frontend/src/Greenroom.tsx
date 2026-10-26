import type { ReactNode } from "react";

export type Admission = {
  admit?: boolean;
  phase?: string;
  message?: string;
  livekit_ok?: boolean;
  model_tcp?: boolean;
  model_http?: boolean;
  models_ok?: boolean;
  stream_ok?: boolean;
  engine_ready?: boolean;
  model?: string;
};

type Props = {
  ready: Admission | null;
  error: string | null;
  connecting: boolean;
  needGesture: boolean;
  signalOk: boolean;
  signalDetail: string;
  voicePicker: ReactNode;
  insecure: boolean;
  host: string;
  onEnter: () => void;
};

type Check = {
  id: string;
  label: string;
  done: boolean;
};

function checks(ready: Admission | null, signalOk: boolean): Check[] {
  const modelServing = Boolean(ready?.model_http && ready?.models_ok);
  const engineUp = Boolean(ready?.stream_ok || ready?.engine_ready);
  return [
    { id: "sfu", label: "LiveKit is listening", done: Boolean(ready?.livekit_ok) },
    { id: "model", label: "Model HTTP is serving", done: modelServing },
    { id: "engine", label: "Speech engine accepted a socket", done: engineUp },
    { id: "signal", label: "Browser completed wss to this page", done: signalOk },
  ];
}

export default function Greenroom({
  ready,
  error,
  connecting,
  needGesture,
  signalOk,
  signalDetail,
  voicePicker,
  insecure,
  host,
  onEnter,
}: Props) {
  const admit = Boolean(ready?.admit);
  const items = checks(ready, signalOk);
  const canEnter = admit && signalOk;
  const title = canEnter
    ? needGesture
      ? "Engine is ready — click Enter"
      : connecting
        ? "Entering the room"
        : "Engine is ready"
    : admit
      ? "Waiting for this browser"
      : "Waiting for the engine";
  const detail = !admit
    ? ready?.message ||
      "The model is still loading. Camera, mic, and Connect stay locked until it can hear you."
    : signalOk
      ? "Engine is ready. We will walk you in."
      : "Chrome has not finished a secure WebSocket to this origin. Reload, Advanced → Proceed, then click Enter. LiveKit is already up.";

  return (
    <div className="greenroom" role="dialog" aria-modal="true" aria-labelledby="greenroom-title">
      <div className="greenroom-card">
        <p className="greenroom-kicker">greenroom</p>
        <h1 id="greenroom-title">{title}</h1>
        <p className="muted">{detail}</p>
        {ready?.model ? <p className="greenroom-model">Model: {ready.model}</p> : null}
        <ol className="greenroom-checks">
          {items.map((item) => (
            <li key={item.id} className={item.done ? "on" : ""}>
              <span className="greenroom-dot" aria-hidden="true" />
              {item.label}
            </li>
          ))}
        </ol>
        {voicePicker}
        {insecure ? (
          <div className="warn">
            Stay on <code>https://{host}/patient</code> — Advanced → Proceed — then reload.
            Camera and mic stay blocked on HTTP.
          </div>
        ) : null}
        <div className="row">
          <button
            type="button"
            onClick={onEnter}
            disabled={!canEnter || connecting || insecure}
          >
            {connecting ? "Entering…" : canEnter ? "Enter" : "Waiting…"}
          </button>
          <span className="status">
            {canEnter
              ? needGesture
                ? "browser needs a click for camera/mic"
                : "admitting automatically"
              : signalOk
                ? ready?.phase || "checking"
                : signalDetail || "checking browser socket"}
          </span>
        </div>
        {error ? <p className="muted">error: {error}</p> : null}
      </div>
    </div>
  );
}
