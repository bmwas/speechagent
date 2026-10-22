# Real-Time Omni Speech Agent

Talk to **MiniCPM-o 4.5** or **Qwen2.5-Omni-3B** in the browser. The agent hears you, sees your camera, and answers in speech over LiveKit. Inference runs on the remote **RTX 6000 Ada (48 GB, CUDA index 1)**.

Everything starts and stops with Docker Compose. Grafana shows every agent action. `docker compose down` **deletes all logs**.

## Architecture

```
Browser  https://<GPU-LAN>:5173/patient
   HTTP :5173 308s to HTTPS on the same host:port (msestream capture-ui mux)
   wss://<same>/rtc  -->  LiveKit SFU (:7880, host net, ICE=MSE_LAN_IP)
                              |
                         Python agent (:7860)
                              |
                   vLLM-Omni model-server (:8099)
                              |
                    RTX 6000 Ada 48 GB (device 1)

stdout SA_TRACE  -->  Promtail  -->  Loki (:3100)  -->  Grafana (:3000)
```

| Service | Role |
|---|---|
| `livekit` | Self-hosted SFU. Host network so ICE/TURN advertise the GPU LAN IP. |
| `model-server` | `vllm/vllm-omni:v0.28.0` (prebuilt, includes flash-attn). Pinned to Ada GPU. |
| `agent` | Joins the room as a bot, streams A/V into the model, publishes speech back. |
| `frontend` | Same capture-ui as msestream: HTTPS+HTTP mux on **:5173**, SPA at `/patient`, proxies `/api` and `/rtc`. |
| `loki` / `promtail` / `grafana` | Log pipeline. Anonymous Grafana Admin. |
| `log-collector` | Copies container stdout into `./logs/*.log`. |
| `log-janitor` | On `compose down` (SIGTERM), deletes everything under `./logs/`. |

## What Compose does

On the GPU server, **`docker compose up -d --build` is the entire install**. You do not install Python, Node, vLLM, LiveKit, or model weights on the host.

That one command:

1. Builds `agent` (Python 3.11 + LiveKit SDK) inside Docker
2. Builds `frontend` (`npm ci` + Vite + `serve.mjs` HTTPS mux on :5173) and mints a self-signed TLS cert (SAN includes `MSE_LAN_IP`)
3. Pulls `livekit/livekit-server:v1.9.0`, `vllm/vllm-omni:v0.28.0` (flash-attn included), Loki, Promtail, Grafana
4. Pins vLLM to the Ada GPU (`ADA_GPU_INDEX`, default host index **1**)
5. Downloads HuggingFace weights into `./models` (kept across `down`)
6. Starts LiveKit on host network so ICE/TURN advertise `MSE_LAN_IP`
7. Starts Promtail → Loki → Grafana; `log-collector` copies stdout into `./logs`
8. On `docker compose down`, `log-janitor` **deletes `./logs`** (weights stay)

Do **not** run Compose on a laptop or any host without the Ada GPU.

## Prerequisites (GPU server)

- Docker Compose v2 + **NVIDIA Container Toolkit**
- CUDA map matching msestream: index `0` = Blackwell, index `1` = **RTX 6000 Ada 48 GB**
- This repo copied onto that box; edit `.env` there
- `HUGGINGFACE_TOKEN` already in the GPU-box `~/.bashrc` — Compose passes it through (do not paste the token into `.env`)
- From a laptop you only open the HTTPS UI

VRAM: MiniCPM-o 4.5 ~12–20 GB; Qwen2.5-Omni-3B ~7 GB. The 48 GB Ada holds either, not both at once.

## Run on the GPU server

Copy from a laptop:

```bash
rsync -az --delete --exclude node_modules --exclude dist --exclude logs \
  ./speechagent/ gpu-box:/opt/speechagent/
```

SSH to the GPU box and run **all** of the following there:

```bash
cd /opt/speechagent

# 1. Confirm Ada is index 1 (change ADA_GPU_INDEX in .env if not)
nvidia-smi -L

# 2. LAN IP the laptop can ping (ICE/TURN)
hostname -I | awk '{print $1}'
# paste that value into MSE_LAN_IP and LIVEKIT_NODE_IP in .env
# HUGGINGFACE_TOKEN is already in ~/.bashrc — compose interpolates it.

# 3. Start everything (images, builds, GPU, weights, TLS, logs)
docker compose up -d --build

# 4. Watch first-boot weight download / CUDA init
docker compose logs -f model-server
# Ctrl-C when you see the server listening on :8099 (can take several minutes)

docker compose ps
curl -sS http://127.0.0.1:7860/health
```

Open from a **laptop on the LAN** (not `127.0.0.1` of the GPU box):

- Patient UI: `https://<MSE_LAN_IP>:5173/patient` — Chrome **Advanced → Proceed** (self-signed)
- HTTP on the **same** `:5173` 308s to that HTTPS URL (never to 127.0.0.1, never drops the port)
- LiveKit signaling is same-origin `wss://<MSE_LAN_IP>:5173/rtc`
- Click **Connect**, allow camera + mic, headphones on, talk
- Grafana: `http://<MSE_LAN_IP>:3000` (anonymous Admin)
  - **Agent traces** — every `SA_TRACE` event
  - **Agent logs** — raw service stdout

Do not run this stack at the same time as msestream: both bind LiveKit `:7880` and capture-ui `:5173`.

Stop (this **wipes `./logs`**, keeps `./models`):

```bash
docker compose down
```

## Compose commands (GPU server only)

| Command | What it does |
|---|---|
| `docker compose up -d --build` | Build + start the full stack |
| `docker compose ps` | Service status |
| `docker compose logs -f agent` | Agent stdout including `SA_TRACE` |
| `docker compose logs -f model-server` | vLLM-Omni / weight download |
| `docker compose logs -f livekit` | SFU / ICE |
| `docker compose restart agent` | Bounce the bot without unloading weights |
| `docker compose up -d --build model-server agent` | Reload after changing `MODEL_NAME` |
| `docker compose down` | Stop **and delete `./logs`** |

## Switch models

In `.env`:

```env
# Full-duplex, most natural conversation
MODEL_NAME=openbmb/MiniCPM-o-4_5

# Turn-based, lighter VRAM, Chelsie/Ethan voices
MODEL_NAME=Qwen/Qwen2.5-Omni-3B
QWEN_SPEAKER=Chelsie
```

Then:

```bash
docker compose up -d --build model-server agent
```

Only one omni model is loaded. Do not run both.

## Voices

**MiniCPM-o 4.5** clones `assets/HT_ref_audio.wav` from the HuggingFace snapshot (auto-downloaded with the weights). Override with `REF_AUDIO_PATH` pointing at a clean **16 kHz mono WAV, 3–10 seconds**.

**Qwen2.5-Omni-3B** has built-in speakers: `Chelsie` (female) and `Ethan` (male). Set `QWEN_SPEAKER`. No reference WAV.

## How conversation stays real-time

- Mic audio is sent in **100 ms** chunks; MiniCPM-o decides listen vs speak every ~1 s internally.
- LiveKit `AudioSource` queue is **120 ms** (shallow) so barge-in is tens of ms, not seconds.
- A **20 ms playout clock** feeds WebRTC; the producer never burst-dumps a 1 s wav into the SDK.
- Leading silence on the first model chunk is trimmed.
- MiniCPM-o 4.5 **does not self-yield** when you talk over it. The agent runs energy VAD, sends `response.cancel`, and clears the LiveKit queue (client-side barge-in).
- While MiniCPM-o is synthesizing speech, camera frames are suppressed so Token2Wav keeps the GPU.
- Qwen uses VAD start 150 ms / stop 800 ms, then streams `/v1/chat/completions` audio deltas.
- Browser: AudioContext prewarm on the Connect click, 20 ms inbound playout delay, AEC/NS/AGC, skip Chrome's Linux `"default"` mic.

## GPU pinning

```yaml
environment:
  CUDA_DEVICE_ORDER: PCI_BUS_ID
  CUDA_VISIBLE_DEVICES: "0"   # inside the container the reserved card is cuda:0
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["1"]    # host nvidia-smi index 1 = Ada 48 GB
          capabilities: [gpu]
```

If `nvidia-smi -L` shows Ada at index 0 on your box, change `device_ids` to `["0"]`. Do **not** set `CUDA_VISIBLE_DEVICES` to the host index after `device_ids` has already selected one card.

## Observability

Every meaningful action is one stdout line:

```
SA_TRACE {"ts":..., "agent":"speech", "event":"session.open", "session_id":"lk-default", ...}
```

Promtail parses the prefix, labels `agent` + `event`, and stores `session_id` / `trace_id` as structured metadata.

### Event catalog

| Event | When |
|---|---|
| `session.open` / `session.close` | Connect / hangup |
| `session.brain_ready` | Omni engine started |
| `livekit.connect` / `livekit.disconnect` | Bot or user joined/left |
| `track.audio_sub` / `track.video_sub` / `track.audio_pub` | Media tracks |
| `vad.start` / `vad.stop` | Energy VAD edges |
| `turn.barge_in` | User interrupted the agent |
| `duplex.ws_open` / `duplex.session_created` / `duplex.ws_close` | MiniCPM WS |
| `duplex.append` | 1 Hz audio+video heartbeat into the model |
| `duplex.speak` / `duplex.listen` / `duplex.cancel` | Model turn decision |
| `qwen.request` / `qwen.first_audio` / `qwen.done` | Turn-based path |
| `playout.push` / `playout.underrun` / `playout.block` / `playout.clear` | Speaker path |
| `video.frame` | JPEG still sent (1 fps) |
| `voice.ref_loaded` | Which WAV / Qwen speaker |
| `error.*` | Failures with traceback |

`TRACE_DEBUG=1` (default) also emits per-chunk `audio.chunk` / `video.chunk` **metadata** (rms, bytes). Raw PCM and JPEG are never logged.

On `docker compose down`, `log-janitor` deletes `./logs/**` (Loki chunks + collector files). Docker json-file logs die with the containers. Weights in `./models` are kept.

## Code map

```
speechagent/
  docker-compose.yml          sole runtime
  .env                        keys, model, LAN IP, tuning
  agent/src/
    agent.py                  session: LiveKit + engine + VAD + playout
    duplex_model.py           MiniCPM-o Realtime WebSocket
    qwen_omni.py              Qwen2.5-Omni chat completions + audio
    token_server.py           POST /api/livekit/token, GET /health
    livekit_room.py           rtc.Room subscribe / publish
    audio.py                  resample, silence trim, 20 ms clock
    video.py                  RGB → JPEG sampler
    barge_in.py               energy VAD + echo-aware interrupt
    trace.py                  SA_TRACE JSON
    logging_setup.py          LOG_LEVEL, quiet WebRTC loggers
  frontend/                   Vite + React + serve.mjs HTTPS mux (`/patient`)
  livekit/                    SFU config + entrypoint
  model-server/entrypoint.sh  MODEL_NAME → vllm serve --omni
  obs/                        Loki, Promtail, Grafana, janitor
```

## Ports

| Port | Service |
|---|---|
| 5173 TCP | Capture UI: HTTP 308 → HTTPS `/patient`, plus `/api` and `/rtc` |
| 3000 TCP | Grafana |
| 3100 TCP | Loki |
| 7860 TCP | Agent token/health (host network) |
| 7880–7882 | LiveKit signaling / TCP media / TURN |
| 50000–50100 UDP | LiveKit WebRTC |
| 8099 TCP | vLLM-Omni |

## Configuration

See [`.env`](.env). Important knobs:

| Variable | Default | Meaning |
|---|---|---|
| `MSE_LAN_IP` | `192.168.1.34` | ICE/TURN advertise this address; TLS cert SAN; capture-ui `:5173/patient` |
| `ADA_GPU_INDEX` | `1` | Host `nvidia-smi` index of the Ada 48 GB |
| `MODEL_NAME` | `openbmb/MiniCPM-o-4_5` | Which omni weights to load |
| `HUGGINGFACE_TOKEN` | bashrc | Hub auth for weight download. Read from the GPU-box shell, not `.env` |
| `AUDIO_SOURCE_QUEUE_MS` | `120` | LiveKit speaker queue depth |
| `VAD_STOP_SECS` | `0.80` | Silence before a Qwen turn ends |
| `VIDEO_FPS` | `1` | Stills sent to the model |
| `TRACE_DEBUG` | `1` | Per-chunk metadata traces |
| `REF_AUDIO_PATH` | empty | MiniCPM custom voice WAV |
| `QWEN_SPEAKER` | `Chelsie` | Qwen built-in voice |

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Chrome: camera/mic blocked | Use **https://LAN:5173/patient**, accept the self-signed cert. Not `http://`, not GPU `127.0.0.1` from a laptop. |
| Mic: `Could not start audio source` | Chrome Linux `"default"` device. Pick a real input in chrome://settings/content/microphone. |
| Connect hangs / ICE failed | `MSE_LAN_IP` must be the GPU box address the laptop can ping. UDP 50000–50100 + 7882 open on the LAN. |
| Silent agent | `docker compose logs -f model-server` — weights still downloading or CUDA OOM. Grafana `error.*` events. |
| Grafana empty | Wait ~15 s for Promtail. Filter `{compose_project="speechagent"}`. |
| Logs still on disk after down | `docker compose logs log-janitor`. The janitor needs 20 s grace. Then `ls logs/`. |
| Ada not used | `nvidia-smi` — compose `device_ids: ["1"]`. Inside the container `CUDA_VISIBLE_DEVICES=0`. |

## References

- [vLLM-Omni MiniCPM-o 4.5 recipe](https://github.com/vllm-project/vllm-omni)
- [vLLM-Omni Realtime Duplex API](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/realtime_duplex_api.md)
- [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [LiveKit Agents](https://docs.livekit.io/agents/)
- Logging / GPU pin / `/patient` UI patterns from the local `msestream` stack
