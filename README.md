# Real-Time Omni Speech Agent

Talk to **MiniCPM-o 4.5** or **Qwen2.5-Omni-3B** in the browser. The agent hears you, sees your camera, and answers in speech over LiveKit. Inference runs on the remote **RTX 6000 Ada (48 GB, CUDA index 1)**.

Everything starts and stops with Docker Compose. Grafana shows every agent action. `docker compose down` **deletes all logs**.

## Architecture

```
Browser  https://<GPU-LAN>:6173/patient
   HTTP :6173 308s to HTTPS on the same host:port
   wss://<same>/rtc  -->  LiveKit SFU (:8880, host net, ICE=MSE_LAN_IP)
                              |
                         Python agent (:8860)
                              |
                   vLLM-Omni model-server (:8199)
                              |
                    RTX 6000 Ada 48 GB (device 1)

stdout SA_TRACE  -->  Promtail  -->  Loki (:3310)  -->  Grafana (:3300)
```

msestream keeps `:5173` / `:7860` / `:7880` / `:3000` / `:3100`. This stack uses the offsets above so both can listen at once. They still **share Ada GPU 1** — two omni models can OOM.

| Service | Role |
|---|---|
| `livekit` | Self-hosted SFU. Host network so ICE/TURN advertise the GPU LAN IP. |
| `model-server` | `vllm/vllm-omni:v0.30.0` (prebuilt, includes flash-attn). Pinned to Ada GPU. |
| `agent` | Joins the room as a bot, streams A/V into the model, publishes speech back. |
| `frontend` | Same capture-ui mux as msestream, on **:6173** host network, SPA at `/patient`, proxies `/api` and `/rtc` to loopback. |
| `loki` / `promtail` / `grafana` | Log pipeline. Anonymous Grafana Admin. |
| `gpu-stats` | Host `nvidia-smi` sampler (query only, no VRAM). `SA_TRACE gpu.sample` / `gpu.summary` / `gpu.verdict` → Grafana **Ada GPU**. |
| `log-collector` | Copies container stdout into `./logs/*.log`. |
| `log-janitor` | On `compose down` (SIGTERM), deletes everything under `./logs/`. |

## What Compose does

On the GPU server, **`docker compose up -d --build` is the entire install**. You do not install Python, Node, vLLM, LiveKit, or model weights on the host.

That one command:

1. Builds `agent` (Python 3.11 + LiveKit SDK) inside Docker
2. Builds `frontend` (`npm ci` + Vite + `serve.mjs` HTTPS mux on :6173) and mints a self-signed TLS cert (SAN includes `MSE_LAN_IP`)
3. Pulls `livekit/livekit-server:v1.9.0`, `vllm/vllm-omni:v0.30.0` (flash-attn included), Loki, Promtail, Grafana
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
# gpu-free stops msestream llama.cpp / interviewer-engine on Ada first.
docker compose up -d --build

# 4. Watch first-boot weight download / CUDA init
docker compose logs -f model-server
# Ctrl-C when you see the server listening on :8199 (can take several minutes)

docker compose ps
curl -sS http://127.0.0.1:8860/health
```

Open from a **laptop on the LAN** (not `127.0.0.1` of the GPU box):

- Patient UI: `https://<MSE_LAN_IP>:6173/patient` — Chrome **Advanced → Proceed** (self-signed)
- HTTP on the **same** `:6173` 308s to that HTTPS URL (never to 127.0.0.1, never drops the port)
- LiveKit signaling is same-origin `wss://<MSE_LAN_IP>:6173/rtc`
- Click **Connect**, allow camera + mic, headphones on, talk
- Grafana: `http://<MSE_LAN_IP>:3300` (anonymous Admin). **Not** msestream `:3000`.
  - **Agent logs** (home dashboard) — model-server KV/CUDA errors + per-service stdout
  - **Agent traces** — `SA_TRACE` events (`duplex.*`, barge-in, video.frame)
  - **Realtime stream** — duplex (`duplex.video_attach` → `duplex.first_audio`) or Qwen (`qwen.video_frame` → `qwen.request` → `qwen.first_audio`) then playout (`lane=stream`)
  - **Ada GPU** — VRAM / util / power over time plus sampler-computed 1m/5m/15m avg, p50, p95, peak, min headroom, duty cycle (`/d/speechagent-gpu`)

msestream `/patient` stays at `:5173`. This agent is `:6173`.

Stop (this **wipes `./logs`**, keeps `./models`):

```bash
docker compose down
```

## Compose commands (GPU server only)

| Command | What it does |
|---|---|
| `docker compose up -d --build` | Build + start the full stack (`gpu-free` clears Ada first) |
| `docker compose run --rm gpu-free` | Stop other Ada users without starting vLLM |
| `FREE_ADA_GPU=0 docker compose up -d model-server` | Load vLLM without touching msestream |
| `docker compose ps` | Service status |
| `docker compose logs -f agent` | Agent stdout including `SA_TRACE` |
| `docker compose logs -f gpu-stats` | Ada VRAM/util samples + rolling avg/p95/peak/headroom verdicts |
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

# Turn-based live video stream, lighter VRAM, Chelsie/Ethan voices
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

This is a **native omni** path on vLLM-Omni, not ASR → LLM → TTS. The two models use different sockets. They cannot share one.

**MiniCPM-o 4.5** — native duplex (`/v1/realtime?duplex=1`):

- Browser camera + mic publish over LiveKit. The agent streams both into MiniCPM-o **while you talk**.
- Mic PCM is appended every **100 ms**. Camera stills (~**1 JPEG/s**, long side 448) ride on those appends as `video_frames` — the OpenBMB duplex contract is one still per ~1 s audio unit.
- `auto_response` is on, so the model can start speaking from the live stream. On VAD stop the agent also sends `input_audio_buffer.commit`.
- Stage-0 `max_model_len` is **8192**. The session also opens with MiniCPM's **basic sliding window** (trim at 3000 tokens, resume at 2000, system/ref-audio prefix kept) so a long call does not hit `context_length_exceeded`.
- Language is locked by the **official English clone prompt** (`AGENT_SYSTEM_PROMPT`), not by `HT_ref_audio.wav`. That WAV is bilingual; without the English prefix MiniCPM falls back to Chinese after a thinker overflow.
- While the model is speaking, silence and camera stills are **not** appended (they were filling the thinker KV and stalling the vocoder). User speech still is, so barge-in works. Stills resume on the next listen unit.
- If a speak stalls (≥4 s with no audio delta), the agent sends `response.cancel` and **reopens the duplex socket** (`duplex.reconnect`). vllm-omni drops `context_length_exceeded` as `dropping error for unknown req` and otherwise leaves a zombie session.
- MiniCPM-o 4.5 **does not self-yield** when you talk over it. The agent runs energy VAD, sends `response.cancel`, and clears the LiveKit queue (client-side barge-in).

**Qwen2.5-Omni-3B** — live video stream (`/v1/video/chat/stream`), not duplex, not one JPEG, not an MP4 at VAD stop:

- Camera ticks become `video.frame` at **5 FPS** as they happen. Mic ticks become `audio.chunk` as they happen. The server keeps a sliding video session (EVS near-duplicate filter, sample `QWEN_VIDEO_NUM_FRAMES` at query time).
- VAD stop (150 ms start / 800 ms stop) sends `video.query`. Speech streams back on `response.output_audio.delta`.
- Wire frames are JPEG because that is vllm-omni's video-stream transport. The session is a video buffer, not a still we chose. The agent never sends `image_url`.

**Both:**

- Model speech is **streamed out as it is generated**: each audio delta is decoded immediately, resampled, and clocked into LiveKit at **20 ms**. The agent never waits for a full WAV.
- LiveKit `AudioSource` queue is **120 ms** (shallow) so barge-in is tens of ms, not seconds.
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

If `nvidia-smi -L` shows Ada at index 0 on your box, change `ADA_GPU_INDEX` in `.env`. Do **not** set `NVIDIA_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` to the **host** index after `device_ids` has already selected one card — StageEngineCoreProc children then look for cuda:1, fall back to CPU, and die with `Available KV cache memory: 0.0 GiB`.

MiniCPM-o 4.5 is three colocated stages and needs a **clear 48 GB Ada**. `gpu-free` runs before `model-server` and `docker stop`s other containers pinned to GPU 1 (msestream `interviewer-engine`, `observer-engine`, llama.cpp). Blackwell (`device_ids: ["0"]`, sglang) is not touched. `docker stop` is a manual stop, so MSE `restart: unless-stopped` stays down until you start those engines again. Set `FREE_ADA_GPU=0` to skip.

## Observability

Every meaningful action is one stdout line:

```
SA_TRACE {"ts":..., "agent":"speech", "event":"session.open", "session_id":"lk-default", ...}
```

Promtail parses the prefix, labels `agent` + `event` + `lane`, and stores `session_id` / `trace_id` as structured metadata.

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
| `duplex.append_speech` | First `is_speech=true` append of an utterance |
| `duplex.video_attach` | JPEG still attached to an audio append (~1 / s) |
| `duplex.commit` | VAD end-of-speech `input_audio_buffer.commit` |
| `duplex.first_audio` | Time-to-first-byte from the model (`latency_ms`) — first delta, not full turn |
| `duplex.audio_delta` | 1 Hz model PCM out (bytes, qsize, rms) |
| `duplex.speak` / `duplex.listen` / `duplex.cancel` | Model turn decision |
| `duplex.rx_stats` | 1 Hz WebSocket event-type histogram |
| `duplex.model_speech_start` / `duplex.model_speech_stop` | Model-side VAD |
| `stream.health` | 1 Hz pipeline snapshot (ready, ring, mic age) |
| `stream.mic_in` / `stream.mic_first` / `stream.mic_gap` | LiveKit mic hop |
| `stream.stall` / `stream.drop_not_ready` | Duplex not producing / WS not ready |
| `qwen.ws_open` / `qwen.ready` / `qwen.ws_close` | Qwen `/v1/video/chat/stream` session |
| `qwen.video_frame` | Live `video.frame` sent into the server buffer |
| `qwen.frames_consumed` | Server sampled this query's frames |
| `qwen.request` / `qwen.first_audio` / `qwen.sse` / `qwen.done` | `video.query` then streamed speech |
| `playout.push` / `playout.first_frame` / `playout.underrun` / `playout.block` / `playout.clear` | Speaker path |
| `video.frame` | JPEG encoded for the active engine (1 fps MiniCPM, 5 fps Qwen) |
| `voice.ref_loaded` | Which WAV / Qwen speaker |
| `error.*` | Failures with traceback |

`TRACE_DEBUG=1` and `STREAM_TRACE=1` (defaults) emit VAD-edge plus 1 Hz pipeline meters (`lane=stream` in Loki). `TRACE_HIRES=1` logs every WebSocket message, audio delta, mic append, playout push, and 10 Hz ring/LiveKit samples. Raw PCM and JPEG are never logged.

On `docker compose down`, `log-janitor` deletes `./logs/**` (Loki chunks + collector files). Docker json-file logs die with the containers. Weights in `./models` are kept.

## Code map

```
speechagent/
  docker-compose.yml          sole runtime
  .env                        keys, model, LAN IP, tuning
  agent/src/
    agent.py                  session: LiveKit + engine + VAD + playout
    duplex_model.py           MiniCPM-o Realtime WebSocket
    qwen_omni.py              Qwen2.5-Omni `/v1/video/chat/stream` + speech out
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
  model-server/deploy/        MiniCPM-o Ada KV/VRAM split YAML
  obs/                        Loki, Promtail, Grafana, janitor
```

## Ports

| Port | Service |
|---|---|
| 6173 TCP | Capture UI: HTTP 308 → HTTPS `/patient`, plus `/api` and `/rtc` |
| 3300 TCP | Grafana |
| 3310 TCP | Loki |
| 8860 TCP | Agent token/health (host network) |
| 8880–8882 | LiveKit signaling / TCP media / TURN |
| 51000–51100 UDP | LiveKit WebRTC |
| 8199 TCP | vLLM-Omni |

## Configuration

See [`.env`](.env). Important knobs:

| Variable | Default | Meaning |
|---|---|---|
| `MSE_LAN_IP` | `10.4.7.193` | ICE/TURN advertise this address; TLS cert SAN; capture-ui `:6173/patient`. Must be the GPU box LAN IP, not a laptop. |
| `ADA_GPU_INDEX` | `1` | Host `nvidia-smi` index of the Ada 48 GB |
| `FREE_ADA_GPU` | `1` | Stop other Ada GPU users before vLLM loads |
| `MODEL_NAME` | `openbmb/MiniCPM-o-4_5` | Which omni weights to load |
| `HUGGINGFACE_TOKEN` | bashrc | Hub auth for weight download. Read from the GPU-box shell, not `.env` |
| `AUDIO_SOURCE_QUEUE_MS` | `80` | LiveKit speaker queue depth |
| `PLAYOUT_HOLD_S` | `0.25` | Brief silence bridge after real audio drains; longer stalls are treated as idle |
| `PLAYOUT_MAX_WAIT_MS` | `300` | Maximum prebuffer wait for a short delta that cannot reach the 150 ms waterline |
| `BOT_AUDIBLE_GRACE_MS` | `120` | Recent-real-audio window used for barge-in detection |
| `STALL_CANCEL_MS` | `45000` | Last-resort cancellation threshold for a genuinely stuck model response |
| `VAD_STOP_SECS` | `0.80` | Silence before a Qwen turn ends |
| `VIDEO_FPS` | `1` | MiniCPM-o camera cadence (~1 JPEG per 1 s audio unit). Do not raise this for Qwen. |
| `VIDEO_MAX_SIDE` | `448` | MiniCPM JPEG long-side cap so vision tokens fit the sliding window |
| `AGENT_SYSTEM_PROMPT` | English clone prompt | MiniCPM language lock. Official English prefix; do not replace with a generic assistant string. |
| `QWEN_VIDEO_FPS` | `5` | Live `video.frame` rate into Qwen's video-stream session |
| `QWEN_VIDEO_NUM_FRAMES` | `40` | Frames the server samples from the session buffer on `video.query` |
| `QWEN_VIDEO_MAX_FRAMES` | `50` | Server session buffer cap |
| `QWEN_VIDEO_MAX_SIDE` | `448` | Qwen JPEG long-side cap (3B token budget) |
| `TRACE_DEBUG` | `1` | Extra debug traces (VAD edges). Dropped video/audio chunks are not logged. |
| `STREAM_TRACE` | `1` | Realtime pipeline traces (`lane=stream`) for the Grafana **Realtime stream** dashboard |
| `TRACE_HIRES` | `1` | Every packet/frame plus 10 Hz state samples for chopping debug. Grafana: **Audio pipeline debug**. Set `0` for aggregation. |
| `TRACE_ASYNC_WRITE` | `1` | Batch trace writes off the realtime event loop; required for continuous frame-level logging |
| `GPU_STATS_INTERVAL_S` | `2` | `nvidia-smi` sample period (`gpu.sample`) |
| `GPU_STATS_SUMMARY_S` | `30` | Rolling 1m/5m/15m stats line (`gpu.summary`) |
| `GPU_STATS_VERDICT_S` | `300` | Lifetime avg/p95/peak + “would another model fit?” (`gpu.verdict`) |
| `STREAM_TRACE_PERIOD_S` | `1` | Aggregation window for mic/append/delta meters |
| `REF_AUDIO_PATH` | empty | MiniCPM voice clone WAV. Empty uses the snapshot `assets/HT_ref_audio.wav` (required by v0.30.0 duplex). |
| `QWEN_SPEAKER` | `Chelsie` | Qwen built-in voice |

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Chrome: camera/mic blocked | Use **https://LAN:6173/patient**, accept the self-signed cert. Not `http://`, not GPU `127.0.0.1` from a laptop. |
| Mic: `Could not start audio source` | Chrome Linux `"default"` device. Pick a real input in chrome://settings/content/microphone. |
| Connect hangs / ICE failed | `MSE_LAN_IP` must be the GPU box address the laptop can ping. UDP 51000–51100 + 8882 open on the LAN. |
| `room connection has timed out (signal)` | Browser never reached `wss://<GPU-LAN>:6173/rtc`. Open **https://10.4.7.193:6173/patient** (not the laptop IP). Recreate `livekit` + `frontend` (frontend is host-network so `/rtc` hits `127.0.0.1:8880`). Check Grafana frontend logs for `livekit ws`. |
| Silent agent / `duplex.ws_retry` / `stream.drop_not_ready` | Model never serving on `:8199`. Grafana: vocoder import errors or `0.0 GiB` KV. Recreate `model-server` (`vllm/vllm-omni:v0.30.0`). Wait for `Application startup complete`, then speak. MiniCPM: `duplex.video_attach` → `duplex.first_audio` → `playout.first_frame`. Qwen: `qwen.video_frame` (while talking) → `qwen.request` → `qwen.first_audio` → `playout.first_frame`. |
| `MiniCPM-o duplex audio output requires ref_audio` / `error.brain` `never sent session.created` | v0.30.0 will not open a duplex session without a clone WAV. Recreate **agent** after this fix so it sends `assets/HT_ref_audio.wav` from `/models/hf`. Override with `REF_AUDIO_PATH`. Model-server does not need a reload. |
| First turns good, then buffering / quiet, then Chinese | Stage-0 `context_length_exceeded` (prompt ~4167 vs old cap 4096). Server logs `dropping error for unknown req`; agent sees `stream.stall` and `ring_ms=0` while `speaking=true`. Recreate **model-server** (8192 cap) **and agent** (sliding window + English clone prompt + reconnect). Grafana: `duplex.session_update` should show `sliding_window=basic`; a hung speak should emit `duplex.stall_cancel` then `duplex.reconnect` within ~4 s. |
| `audio_past_key_values length 1500 exceeds 1500, reset` | Expected MiniCPM audio-encoder window (`max_source_positions=1500`), about every 30 s. Not a crash. Harmless unless it coincides with thinker overflow. |
| `This model does not officially support disabling chunked prefill` | Expected on MiniCPM stage-2 (vocoder yaml sets `enable_chunked_prefill: false`). Ignore. |
| SageMaker `model.py` not found / NVFP4 NaN-clamp skipped | Image noise from the vllm-omni container. Ignore. |
| Qwen `qwen.request skipped=no_frames` | Camera never produced a JPEG. Check `video.frame` and `QWEN_VIDEO_FPS`. Server `video.query` with an empty buffer returns `No frames buffered`. |
| `Available KV cache memory: 0.0 GiB` | GPU not visible in StageEngineCoreProc, or Ada full. Recreate `model-server` after pull. Confirm `runtime: nvidia` and `CUDA_VISIBLE_DEVICES=0`. Free Ada VRAM (`nvidia-smi` on the GPU box). |
| Grafana empty / wrong stack | Open **http://10.4.7.193:3300**, not msestream `:3000`. Filter `{compose_project="speechagent"}`. |
| Ada GPU dashboard empty | Recreate **gpu-stats** (and Grafana if the **Ada GPU** link is missing). `docker compose logs gpu-stats` should show `gpu-stats hello` then a `gpu.summary` line every 30 s. Query-only `nvidia-smi` — it does not take VRAM. |
| Logs still on disk after down | `docker compose logs log-janitor`. The janitor needs 20 s grace. Then `ls logs/`. |
| Ada not used | `nvidia-smi` on the GPU box — compose `device_ids: ["1"]`. Inside the container `CUDA_VISIBLE_DEVICES=0`. |

## References

- [vLLM-Omni MiniCPM-o 4.5 recipe](https://github.com/vllm-project/vllm-omni)
- [vLLM-Omni Realtime Duplex API](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/realtime_duplex_api.md)
- [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [vLLM-Omni Streaming Video Input API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/video_stream_api/)
- [LiveKit Agents](https://docs.livekit.io/agents/)
- Logging / GPU pin / `/patient` UI patterns from the local `msestream` stack
