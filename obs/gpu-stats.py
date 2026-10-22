#!/usr/bin/env python3
"""Ada GPU sampler → stdout SA_TRACE (Grafana) + human summary lines.

Query-only nvidia-smi via the host namespace (same trick as gpu-free). Does
not allocate VRAM. Rolling windows (1m / 5m / 15m / lifetime) live in this
process so the logs themselves carry avg / p50 / p95 / peak / headroom /
duty-cycle, not only Grafana.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics
import subprocess
import time
from collections import deque
from typing import Any

INTERVAL_S = max(1.0, float(os.environ.get("GPU_STATS_INTERVAL_S") or 2))
SUMMARY_S = max(INTERVAL_S, float(os.environ.get("GPU_STATS_SUMMARY_S") or 30))
VERDICT_S = max(SUMMARY_S, float(os.environ.get("GPU_STATS_VERDICT_S") or 300))
IDX = str(os.environ.get("ADA_GPU_INDEX") or "1").strip()
HOST = os.environ.get("HOSTNAME") or ""
MAX_SAMPLES = int(6 * 3600 / INTERVAL_S)  # ~6 h ring

_SEQ = 0
_NVSMI: list[str] | None = None


def _trace(event: str, **fields: Any) -> None:
    global _SEQ
    _SEQ += 1
    payload: dict[str, Any] = {
        "ts": time.time(),
        "agent": "gpu",
        "event": event,
        "lane": "gpu",
        "gpu_index": int(IDX) if IDX.isdigit() else IDX,
        "seq": _SEQ,
    }
    if HOST:
        payload["host"] = HOST
    payload.update(fields)
    print("SA_TRACE " + json.dumps(payload, default=str, separators=(",", ":")), flush=True)


def _human(line: str) -> None:
    print(line, flush=True)


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:240]
        raise RuntimeError(err or f"failed: {' '.join(cmd[:6])}")
    return proc.stdout


def _nvsmi(*args: str) -> str:
    global _NVSMI
    if _NVSMI is None:
        candidates = (
            ["nvidia-smi"],
            ["nsenter", "--target", "1", "--mount", "--uts", "--ipc", "--net", "--pid", "nvidia-smi"],
        )
        last = "nvidia-smi not found"
        for prefix in candidates:
            try:
                _run([*prefix, "-L"], timeout=5.0)
                _NVSMI = list(prefix)
                break
            except Exception as exc:
                last = str(exc)
        if _NVSMI is None:
            raise RuntimeError(f"cannot run nvidia-smi: {last}")
        _human(f"gpu-stats nvidia-smi via {' '.join(_NVSMI)} index={IDX}")
    return _run([*_NVSMI, *args])


def _num(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NA", "-"}:
        return None
    text = text.replace("MiB", "").replace("W", "").replace("%", "").replace("C", "").replace("MHz", "")
    try:
        return float(text.split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _csv_row(text: str) -> list[str]:
    row = next(csv.reader(io.StringIO(text.strip().splitlines()[0] if text.strip() else " ")))
    return [cell.strip() for cell in row]


_GPU_QUERY = (
    "index,name,uuid,memory.used,memory.total,memory.free,"
    "utilization.gpu,utilization.memory,temperature.gpu,"
    "power.draw,power.limit,clocks.sm,clocks.mem,fan.speed,pstate,"
    "utilization.encoder,utilization.decoder"
)
_APP_QUERY = "pid,process_name,used_gpu_memory"


def sample_gpu() -> dict[str, Any]:
    raw = _nvsmi(
        "-i",
        IDX,
        f"--query-gpu={_GPU_QUERY}",
        "--format=csv,noheader,nounits",
    )
    cols = _csv_row(raw)
    if len(cols) < 9:
        raise RuntimeError(f"unexpected nvidia-smi row: {raw[:200]}")

    def col(i: int) -> str:
        return cols[i] if i < len(cols) else ""

    used = _num(col(3)) or 0.0
    total = _num(col(4)) or 0.0
    free = _num(col(5))
    if free is None:
        free = max(total - used, 0.0)
    util = _num(col(6)) or 0.0
    mem_util = _num(col(7)) or 0.0
    temp = _num(col(8))
    power = _num(col(9))
    power_lim = _num(col(10))
    sm = _num(col(11))
    mem_clk = _num(col(12))
    fan = _num(col(13))
    pstate = col(14) or ""
    enc = _num(col(15))
    dec = _num(col(16))
    used_pct = (100.0 * used / total) if total > 0 else 0.0
    headroom = max(total - used, 0.0)

    apps_raw = _nvsmi(
        "-i",
        IDX,
        f"--query-compute-apps={_APP_QUERY}",
        "--format=csv,noheader,nounits",
    )
    procs: list[dict[str, Any]] = []
    proc_mem = 0.0
    vllm_mem = 0.0
    for line in apps_raw.splitlines():
        if not line.strip() or "No running" in line:
            continue
        cells = _csv_row(line)
        if len(cells) < 3:
            continue
        mem = _num(cells[2]) or 0.0
        name = os.path.basename(cells[1] or "")[:48]
        procs.append({"pid": cells[0], "name": name, "mem_mib": round(mem, 1)})
        proc_mem += mem
        low = name.lower()
        if "vllm" in low or "enginecore" in low or "pt_main" in low or "python" in low:
            vllm_mem += mem

    names = ",".join(f"{p['name']}:{int(p['mem_mib'])}" for p in procs[:8])
    return {
        "gpu_name": col(1)[:80],
        "gpu_uuid": col(2)[:64],
        "mem_used_mib": round(used, 1),
        "mem_total_mib": round(total, 1),
        "mem_free_mib": round(free, 1),
        "mem_used_gib": round(used / 1024.0, 3),
        "mem_total_gib": round(total / 1024.0, 3),
        "mem_free_gib": round(free / 1024.0, 3),
        "mem_used_pct": round(used_pct, 2),
        "headroom_mib": round(headroom, 1),
        "headroom_gib": round(headroom / 1024.0, 3),
        "util_gpu_pct": round(util, 1),
        "util_mem_pct": round(mem_util, 1),
        "temp_c": None if temp is None else round(temp, 1),
        "power_w": None if power is None else round(power, 1),
        "power_limit_w": None if power_lim is None else round(power_lim, 1),
        "power_pct": round(100.0 * power / power_lim, 1) if power and power_lim else None,
        "sm_clock_mhz": None if sm is None else int(sm),
        "mem_clock_mhz": None if mem_clk is None else int(mem_clk),
        "fan_pct": None if fan is None else round(fan, 1),
        "pstate": pstate,
        "util_enc_pct": None if enc is None else round(enc, 1),
        "util_dec_pct": None if dec is None else round(dec, 1),
        "procs": len(procs),
        "proc_mem_mib": round(proc_mem, 1),
        "vllm_mem_mib": round(vllm_mem, 1),
        "proc_names": names[:200],
    }


class Series:
    def __init__(self, maxlen: int) -> None:
        self.values: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float | None) -> None:
        if value is None:
            return
        self.values.append(float(value))

    def stats(self, last_n: int | None = None) -> dict[str, float]:
        data = list(self.values)
        if last_n is not None:
            data = data[-last_n:]
        n = len(data)
        if n == 0:
            return {"n": 0}
        ordered = sorted(data)
        mean = statistics.fmean(data)
        stdev = statistics.pstdev(data) if n >= 2 else 0.0

        def pct(p: float) -> float:
            if n == 1:
                return ordered[0]
            idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
            return ordered[idx]

        return {
            "n": n,
            "min": round(ordered[0], 3),
            "max": round(ordered[-1], 3),
            "avg": round(mean, 3),
            "stdev": round(stdev, 3),
            "p50": round(pct(50), 3),
            "p95": round(pct(95), 3),
            "p99": round(pct(99), 3),
        }


def _duty(values: list[float], thresh: float) -> float:
    if not values:
        return 0.0
    return round(100.0 * sum(1 for v in values if v >= thresh) / len(values), 1)


def _room(headroom_min_gib: float, headroom_avg_gib: float) -> tuple[str, str]:
    # MiniCPM-o 4.5 three-stage ~12–20 GiB; Qwen2.5-Omni-3B ~7 GiB.
    if headroom_min_gib < 2:
        return "critical", "Ada is nearly full; do not load another model"
    if headroom_min_gib < 8:
        return "tight", "Qwen-3B (~8 GiB) would not fit at the peak; MiniCPM extra copy will OOM"
    if headroom_min_gib < 18:
        return "comfortable", "Qwen-3B (~8 GiB) would fit; a second MiniCPM (~18 GiB) would not"
    if headroom_avg_gib >= 20:
        return "plenty", "another MiniCPM-sized load (~18 GiB) would still fit at the worst sample"
    return "comfortable", "steady headroom; watch the peak before adding a second model"


def _prefix(stats: dict[str, float], name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in stats.items():
        out[f"{name}_{key}"] = value
    return out


def _window_n(seconds: float) -> int:
    return max(1, int(math.ceil(seconds / INTERVAL_S)))


def main() -> None:
    ring = MAX_SAMPLES
    mem = Series(ring)
    head = Series(ring)
    util = Series(ring)
    power = Series(ring)
    temp = Series(ring)
    peak_used = 0.0
    peak_t = time.time()
    t0 = time.monotonic()
    last_summary = 0.0
    last_verdict = 0.0
    fails = 0

    _human(f"gpu-stats start index={IDX} interval={INTERVAL_S}s summary={SUMMARY_S}s verdict={VERDICT_S}s")
    try:
        first = sample_gpu()
    except Exception as exc:
        _trace("gpu.error", error=str(exc)[:240], where="hello")
        _human(f"gpu-stats FATAL nvidia-smi: {exc}")
        raise SystemExit(1) from exc

    _trace(
        "gpu.hello",
        gpu_name=first["gpu_name"],
        gpu_uuid=first.get("gpu_uuid"),
        mem_total_gib=first["mem_total_gib"],
        interval_s=INTERVAL_S,
        summary_s=SUMMARY_S,
        verdict_s=VERDICT_S,
    )
    _human(
        f"gpu-stats hello {first['gpu_name']} total={first['mem_total_gib']} GiB "
        f"used={first['mem_used_gib']} GiB headroom={first['headroom_gib']} GiB"
    )

    while True:
        loop = time.monotonic()
        try:
            snap = sample_gpu()
            fails = 0
        except Exception as exc:
            fails += 1
            _trace("gpu.error", error=str(exc)[:240], fails=fails)
            _human(f"gpu-stats sample failed ({fails}): {exc}")
            time.sleep(min(15.0, INTERVAL_S * fails))
            continue

        mem.add(snap["mem_used_gib"])
        head.add(snap["headroom_gib"])
        util.add(snap["util_gpu_pct"])
        if snap.get("power_w") is not None:
            power.add(snap["power_w"])
        if snap.get("temp_c") is not None:
            temp.add(snap["temp_c"])
        if snap["mem_used_gib"] >= peak_used:
            peak_used = snap["mem_used_gib"]
            peak_t = time.time()

        life = mem.stats()
        snap_out = dict(snap)
        snap_out.update(
            {
                "uptime_s": round(time.monotonic() - t0, 1),
                "n": int(life.get("n") or 0),
                "mem_used_avg_gib": life.get("avg"),
                "mem_used_peak_gib": round(peak_used, 3),
                "peak_age_s": round(time.time() - peak_t, 1),
                "headroom_avg_gib": head.stats().get("avg"),
                "headroom_min_gib": head.stats().get("min"),
                "util_avg_pct": util.stats().get("avg"),
                "util_max_pct": util.stats().get("max"),
            }
        )
        _trace("gpu.sample", **{k: v for k, v in snap_out.items() if v is not None})

        now = time.monotonic()
        if last_summary == 0.0 or (now - last_summary) >= SUMMARY_S:
            last_summary = now
            n1, n5, n15 = _window_n(60), _window_n(300), _window_n(900)
            w1, w5, w15, wall = mem.stats(n1), mem.stats(n5), mem.stats(n15), mem.stats()
            h1, h5, h15, hall = head.stats(n1), head.stats(n5), head.stats(n15), head.stats()
            u1, u5, u15, uall = util.stats(n1), util.stats(n5), util.stats(n15), util.stats()
            p1, p5 = power.stats(n1), power.stats(n5)
            t5 = temp.stats(n5)
            util_all = list(util.values)
            util_5 = util_all[-n5:]
            room, why = _room(float(hall.get("min") or 0.0), float(hall.get("avg") or 0.0))
            summary = {
                "uptime_s": round(now - t0, 1),
                "interval_s": INTERVAL_S,
                "gpu_name": snap["gpu_name"],
                "mem_total_gib": snap["mem_total_gib"],
                "now_used_gib": snap["mem_used_gib"],
                "now_headroom_gib": snap["headroom_gib"],
                "now_util_pct": snap["util_gpu_pct"],
                "now_power_w": snap.get("power_w"),
                "now_temp_c": snap.get("temp_c"),
                "now_pstate": snap.get("pstate"),
                "procs": snap["procs"],
                "vllm_mem_mib": snap["vllm_mem_mib"],
                "peak_used_gib": round(peak_used, 3),
                "peak_age_s": round(time.time() - peak_t, 1),
                "room": room,
                "why": why,
                "duty_compute_1m_pct": _duty(list(util.values)[-n1:], 10),
                "duty_compute_5m_pct": _duty(util_5, 10),
                "duty_hot_5m_pct": _duty(util_5, 80),
                "duty_tight_5m_pct": _duty([v for v in list(mem.values)[-n5:]], 0.90 * snap["mem_total_gib"])
                if snap["mem_total_gib"]
                else 0.0,
            }
            summary.update(_prefix(w1, "w1m_used"))
            summary.update(_prefix(w5, "w5m_used"))
            summary.update(_prefix(w15, "w15m_used"))
            summary.update(_prefix(wall, "life_used"))
            summary.update(_prefix(h1, "w1m_head"))
            summary.update(_prefix(h5, "w5m_head"))
            summary.update(_prefix(h15, "w15m_head"))
            summary.update(_prefix(hall, "life_head"))
            summary.update(_prefix(u1, "w1m_util"))
            summary.update(_prefix(u5, "w5m_util"))
            summary.update(_prefix(u15, "w15m_util"))
            summary.update(_prefix(uall, "life_util"))
            summary.update(_prefix(p1, "w1m_power"))
            summary.update(_prefix(p5, "w5m_power"))
            summary.update(_prefix(t5, "w5m_temp"))
            _trace("gpu.summary", **{k: v for k, v in summary.items() if v is not None})
            _human(
                "gpu-stats "
                f"now {snap['mem_used_gib']:.2f}/{snap['mem_total_gib']:.2f} GiB "
                f"({snap['mem_used_pct']:.1f}%) headroom={snap['headroom_gib']:.2f} GiB "
                f"util={snap['util_gpu_pct']:.0f}% "
                f"power={snap.get('power_w') or 0:.0f}W temp={snap.get('temp_c') or 0:.0f}C "
                f"pstate={snap.get('pstate') or '?'} procs={snap['procs']} | "
                f"1m used avg={w1.get('avg')} p95={w1.get('p95')} "
                f"5m used avg={w5.get('avg')} peak={w5.get('max')} "
                f"head min={h5.get('min')} avg={h5.get('avg')} | "
                f"5m util avg={u5.get('avg')} max={u5.get('max')} "
                f"duty={summary['duty_compute_5m_pct']}% hot={summary['duty_hot_5m_pct']}% | "
                f"life peak={peak_used:.2f} GiB room={room}"
            )

        if last_verdict == 0.0 or (now - last_verdict) >= VERDICT_S:
            last_verdict = now
            hall = head.stats()
            wall = mem.stats()
            uall = util.stats()
            room, why = _room(float(hall.get("min") or 0.0), float(hall.get("avg") or 0.0))
            n = int(wall.get("n") or 0)
            minutes = (now - t0) / 60.0
            verdict = (
                f"Ada {snap['mem_total_gib']:.1f} GiB over {minutes:.1f} min (n={n}): "
                f"used avg {wall.get('avg')} GiB (p50 {wall.get('p50')}, p95 {wall.get('p95')}, "
                f"peak {peak_used:.2f}), headroom avg {hall.get('avg')} GiB "
                f"(worst {hall.get('min')} GiB). Compute duty "
                f"{_duty(list(util.values), 10)}% (hot {_duty(list(util.values), 80)}%). "
                f"Room: {room} — {why}."
            )
            _trace(
                "gpu.verdict",
                room=room,
                why=why,
                verdict=verdict,
                uptime_s=round(now - t0, 1),
                n=n,
                mem_total_gib=snap["mem_total_gib"],
                life_used_avg_gib=wall.get("avg"),
                life_used_p50_gib=wall.get("p50"),
                life_used_p95_gib=wall.get("p95"),
                life_used_peak_gib=round(peak_used, 3),
                life_head_avg_gib=hall.get("avg"),
                life_head_min_gib=hall.get("min"),
                life_util_avg_pct=uall.get("avg"),
                life_util_max_pct=uall.get("max"),
                duty_compute_pct=_duty(list(util.values), 10),
                duty_hot_pct=_duty(list(util.values), 80),
            )
            _human(f"gpu-stats VERDICT {verdict}")

        slept = time.monotonic() - loop
        time.sleep(max(0.2, INTERVAL_S - slept))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(0)
