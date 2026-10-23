#!/usr/bin/env python3
"""Apply the MiniCPM response-required patch to vLLM-Omni 0.30.0.

Fail closed: unknown version or missing source anchors abort startup.
Idempotent: a second run leaves an already-patched tree unchanged.
"""

from __future__ import annotations

import shutil
import sys
from importlib import metadata
from pathlib import Path

EXPECTED_VERSION = "0.30.0"
POLICY_NAME = "require_response_policy.py"

OMNI_IMPORT_ANCHOR = "from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy"
OMNI_IMPORT_PATCH = """from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.require_response_policy import (
    LOG_PREFIX as _REQUIRE_LOG_PREFIX,
    RequireLogGate,
    decide_require_response,
    env_require_response_enabled,
    payload_response_required,
    payload_turn_id,
)
"""

OMNI_HELPERS = '''
    def _minicpmo45_require_opened_turns(self) -> set:
        opened = getattr(self, "_minicpmo45_require_opened_turn_ids", None)
        if not isinstance(opened, set):
            opened = set()
            self._minicpmo45_require_opened_turn_ids = opened
        return opened

    def _minicpmo45_require_latches(self) -> dict:
        latches = getattr(self, "_minicpmo45_require_latched_turns", None)
        if not isinstance(latches, dict):
            latches = {}
            self._minicpmo45_require_latched_turns = latches
        return latches

    def _minicpmo45_mark_require_opened(self, session_id: str | None, turn_id: object) -> None:
        if session_id is None or turn_id is None:
            return
        self._minicpmo45_require_opened_turns().add((session_id, turn_id))
        latches = self._minicpmo45_require_latches()
        if latches.get(session_id) == turn_id:
            latches.pop(session_id, None)

    def _minicpmo45_require_log_gate(self):
        gate = getattr(self, "_minicpmo45_require_log_gate_obj", None)
        if gate is None:
            gate = RequireLogGate(period_s=1.0)
            self._minicpmo45_require_log_gate_obj = gate
        return gate

    def _minicpmo45_require_log(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        logger.info("%s %s", _REQUIRE_LOG_PREFIX, payload)

    def _minicpmo45_require_decision(
        self,
        payload: object,
        *,
        session_id: str | None,
        state: object,
        sampled_is_listen: bool | None,
        force_listen: bool,
    ):
        payload_dict = payload if isinstance(payload, dict) else None
        latches = self._minicpmo45_require_latches()
        turn_id = payload_turn_id(payload_dict)
        if payload_response_required(payload_dict) and session_id is not None and turn_id is not None:
            if latches.get(session_id) != turn_id:
                latches[session_id] = turn_id
                receipt = self._minicpmo45_require_log_gate().offer(
                    "marker_receipt",
                    turn_id,
                    session_id=session_id,
                )
                if receipt.immediate is not None:
                    self._minicpmo45_require_log(receipt.immediate)
                if receipt.summary is not None:
                    self._minicpmo45_require_log(receipt.summary)
        opened = {tid for sid, tid in self._minicpmo45_require_opened_turns() if sid == session_id}
        decision = decide_require_response(
            enabled=env_require_response_enabled(),
            sampled_is_listen=sampled_is_listen,
            force_listen=bool(force_listen),
            is_speech=bool(payload_dict.get("is_speech") is True) if payload_dict else False,
            response_required=payload_response_required(payload_dict),
            pending_speech_context=bool(getattr(state, "pending_speech_context", False)) if state is not None else False,
            current_turn_ended=bool(getattr(state, "current_turn_ended", True)) if state is not None else True,
            pending_speech_response_open=(
                bool(getattr(state, "pending_speech_response_open", False)) if state is not None else False
            ),
            turn_id=turn_id,
            opened_turn_ids=opened,
            latched_turn_id=latches.get(session_id) if session_id is not None else None,
        )
        offered = self._minicpmo45_require_log_gate().offer(
            decision.reason,
            decision.turn_id,
            session_id=session_id,
            required=int(decision.required),
            override=int(decision.override_listen),
            block_force_listen=int(decision.block_force_listen),
            sampled_listen=None if sampled_is_listen is None else int(sampled_is_listen),
        )
        if offered.immediate is not None:
            self._minicpmo45_require_log(offered.immediate)
        if offered.summary is not None:
            self._minicpmo45_require_log(offered.summary)
        return decision

'''

OMNI_PREPARE_ANCHOR = """            if not force_listen:
                continue
            if force_listen and segment_key in force_listen_segments:
                continue
            if force_listen:
                logits[row_idx, :] = float("-inf")
                logits[row_idx, listen_id] = 0.0
                force_listen_segments.add(segment_key)
"""

OMNI_PREPARE_PATCH = """            require_state = None
            if session_key is not None and isinstance(helper_sessions, dict):
                require_state = helper_sessions.get(session_key)
            require_decision = self._minicpmo45_require_decision(
                payload,
                session_id=session_key if isinstance(session_key, str) else None,
                state=require_state,
                sampled_is_listen=None,
                force_listen=force_listen,
            )
            if require_decision.block_force_listen:
                force_listen = False
            if not force_listen:
                continue
            if force_listen and segment_key in force_listen_segments:
                continue
            if force_listen:
                logits[row_idx, :] = float("-inf")
                logits[row_idx, listen_id] = 0.0
                force_listen_segments.add(segment_key)
"""

OMNI_FINALIZE_ANCHOR = """        if (
            sampled == listen_id
            and 0 <= tts_bos_id
            and state is not None
            and not getattr(state, "current_turn_ended", True)
            and not force_listen
        ):
            return int(tts_bos_id)
        return int(sampled)
"""

OMNI_FINALIZE_PATCH = """        session_id = None
        row_sessions = getattr(self, "_minicpmo45_duplex_row_sessions", None)
        if isinstance(row_sessions, dict):
            session_id = row_sessions.get(row_idx)
        require_decision = self._minicpmo45_require_decision(
            payload,
            session_id=session_id if isinstance(session_id, str) else None,
            state=state,
            sampled_is_listen=sampled == listen_id,
            force_listen=force_listen,
        )
        if require_decision.reason == "natural_speak":
            self._minicpmo45_mark_require_opened(
                session_id if isinstance(session_id, str) else None,
                require_decision.turn_id,
            )
        if require_decision.override_listen and 0 <= tts_bos_id:
            self._minicpmo45_mark_require_opened(
                session_id if isinstance(session_id, str) else None,
                require_decision.turn_id,
            )
            return int(tts_bos_id)
        if (
            sampled == listen_id
            and 0 <= tts_bos_id
            and state is not None
            and not getattr(state, "current_turn_ended", True)
            and not force_listen
        ):
            return int(tts_bos_id)
        return int(sampled)
"""

OMNI_FINISHED_ANCHOR = """        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        sessions = getattr(helper, "sessions", None) if helper is not None else None
        if isinstance(request_sessions, dict):
            for request_id in finished_req_ids:
                session_key = request_sessions.pop(request_id, None)
                if session_key is not None and isinstance(sessions, dict):
                    sessions.pop(session_key, None)
"""

OMNI_FINISHED_PATCH = """        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        sessions = getattr(helper, "sessions", None) if helper is not None else None
        finished_sessions: set = set()
        if isinstance(request_sessions, dict):
            for request_id in finished_req_ids:
                session_key = request_sessions.pop(request_id, None)
                if isinstance(session_key, str):
                    finished_sessions.add(session_key)
                if session_key is not None and isinstance(sessions, dict):
                    sessions.pop(session_key, None)
        if finished_sessions:
            opened = getattr(self, "_minicpmo45_require_opened_turn_ids", None)
            if isinstance(opened, set):
                opened.difference_update({item for item in opened if item[0] in finished_sessions})
            latches = getattr(self, "_minicpmo45_require_latched_turns", None)
            if isinstance(latches, dict):
                for session_key in finished_sessions:
                    latches.pop(session_key, None)
"""

OMNI_PREPARE_DEF = "    def prepare_duplex_sampling("

RUNNER_ANCHOR = """        payload: dict[str, object] = {
            "type": "audio",
            "force_listen": force_listen,
        }
"""

RUNNER_PATCH = """        payload: dict[str, object] = {
            "type": "audio",
            "force_listen": force_listen,
        }
        if event.get("response_required") is True:
            payload["response_required"] = True
        require_turn_id = event.get("turn_id")
        if isinstance(require_turn_id, int) or (isinstance(require_turn_id, str) and require_turn_id):
            payload["turn_id"] = require_turn_id
"""

INPUT_INIT_ANCHOR = """        self._turn_had_speech = False
        self._reservation_seq = 0
"""

INPUT_INIT_PATCH = """        self._turn_had_speech = False
        self._require_response = False
        self._require_turn_id = None
        self._reservation_seq = 0
"""

INPUT_CLEAR_ANCHOR = """        self._frame_queue.clear()
        self._sample_rate_hz = None
        self._turn_had_speech = False
"""

INPUT_CLEAR_PATCH = """        self._frame_queue.clear()
        self._sample_rate_hz = None
        self._turn_had_speech = False
        self._require_response = False
        self._require_turn_id = None
"""

INPUT_SPEECH_ANCHOR = """        self._turn_had_speech = self._turn_had_speech or bool(payload.get("is_speech", False))
"""

INPUT_SPEECH_PATCH = """        self._turn_had_speech = self._turn_had_speech or bool(payload.get("is_speech", False))
        if payload.get("response_required") is True:
            self._require_response = True
        require_turn_id = payload.get("turn_id")
        if isinstance(require_turn_id, int) or (isinstance(require_turn_id, str) and require_turn_id):
            self._require_turn_id = require_turn_id
"""

INPUT_OUT_ANCHOR = """        out["force_listen"] = any(span.force_listen for span in reserved_spans)
        out["is_speech"] = any(span.is_speech for span in reserved_spans)
"""

INPUT_OUT_PATCH = """        out["force_listen"] = any(span.force_listen for span in reserved_spans)
        out["is_speech"] = any(span.is_speech for span in reserved_spans)
        if self._require_response or payload.get("response_required") is True:
            out["response_required"] = True
        if self._require_turn_id is not None:
            out["turn_id"] = self._require_turn_id
"""

INPUT_COMMIT_ANCHOR = """            payload: dict[str, object] = {
                "type": "audio",
                "audio": "",
                "format": "pcm_f32le",
                "sample_rate_hz": self._sample_rate_hz or 16000,
                "force_listen": self._buffer_force_listen(),
                "is_speech": self._buffer_is_speech(),
            }
"""

INPUT_COMMIT_PATCH = """            payload: dict[str, object] = {
                "type": "audio",
                "audio": "",
                "format": "pcm_f32le",
                "sample_rate_hz": self._sample_rate_hz or 16000,
                "force_listen": self._buffer_force_listen(),
                "is_speech": self._buffer_is_speech(),
            }
            if self._require_response:
                payload["response_required"] = True
            if self._require_turn_id is not None:
                payload["turn_id"] = self._require_turn_id
"""

INPUT_COMMIT_RESET_ANCHOR = """        self._sample_rate_hz = None
        self._turn_had_speech = False
        return reservation
"""

INPUT_COMMIT_RESET_PATCH = """        self._sample_rate_hz = None
        self._turn_had_speech = False
        self._require_response = False
        self._require_turn_id = None
        return reservation
"""

INPUT_FLUSH_ANCHOR = """        payload: dict[str, object] = {
            "type": "audio",
            "audio": "",
            "format": "pcm_f32le",
            "sample_rate_hz": self._sample_rate_hz or 16000,
            "force_listen": self._buffer_force_listen(),
            "is_speech": self._buffer_is_speech(),
        }
        return self.append(payload, chunk_period_ms=chunk_period_ms, flush=True)
"""

INPUT_FLUSH_PATCH = """        payload: dict[str, object] = {
            "type": "audio",
            "audio": "",
            "format": "pcm_f32le",
            "sample_rate_hz": self._sample_rate_hz or 16000,
            "force_listen": self._buffer_force_listen(),
            "is_speech": self._buffer_is_speech(),
        }
        if self._require_response:
            payload["response_required"] = True
        if self._require_turn_id is not None:
            payload["turn_id"] = self._require_turn_id
        return self.append(payload, chunk_period_ms=chunk_period_ms, flush=True)
"""


class PatchError(RuntimeError):
    pass


def _package_version() -> str:
    for name in ("vllm-omni", "vllm_omni"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    raise PatchError("vllm-omni is not installed")


def _omni_root() -> Path:
    import vllm_omni

    return Path(vllm_omni.__file__).resolve().parent


def _replace_once(path: Path, anchor: str, patch: str, *, already: str | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    if already and already in text and anchor not in text:
        return "already"
    if patch.strip() and patch in text:
        return "already"
    if anchor not in text:
        raise PatchError(f"{path} is missing required v0.30.0 anchor")
    if text.count(anchor) != 1:
        raise PatchError(f"{path} anchor is not unique ({text.count(anchor)} matches)")
    path.write_text(text.replace(anchor, patch, 1), encoding="utf-8")
    return "applied"


def _insert_before(path: Path, needle: str, insertion: str, *, already: str) -> str:
    text = path.read_text(encoding="utf-8")
    if already in text:
        return "already"
    if needle not in text:
        raise PatchError(f"{path} is missing helper insertion point")
    if text.count(needle) != 1:
        raise PatchError(f"{path} helper insertion point is not unique")
    path.write_text(text.replace(needle, insertion + needle, 1), encoding="utf-8")
    return "applied"


def apply(
    *,
    omni_root: Path | None = None,
    policy_src: Path | None = None,
    expected_version: str = EXPECTED_VERSION,
    installed_version: str | None = None,
) -> dict[str, str]:
    if installed_version is not None:
        version = installed_version
    elif omni_root is None:
        version = _package_version()
    else:
        version = expected_version
    if version != expected_version and not str(version).startswith(expected_version):
        raise PatchError(f"vllm-omni {version} is not {expected_version}; refusing to patch")
    root = omni_root if omni_root is not None else _omni_root()
    src = policy_src if policy_src is not None else Path(__file__).resolve().parent / POLICY_NAME
    if not src.is_file():
        raise PatchError(f"policy module missing: {src}")

    omni = root / "model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py"
    runner = root / "engine/duplex/session/runner.py"
    inp = root / "model_executor/models/minicpmo_4_5/duplex/input.py"
    dest_policy = root / "model_executor/models/minicpmo_4_5/duplex" / POLICY_NAME
    for path in (omni, runner, inp):
        if not path.is_file():
            raise PatchError(f"vLLM-Omni source missing: {path}")

    shutil.copy2(src, dest_policy)
    status = {
        "policy": "copied",
        "omni_import": _replace_once(omni, OMNI_IMPORT_ANCHOR, OMNI_IMPORT_PATCH, already="require_response_policy"),
        "omni_helpers": _insert_before(
            omni,
            OMNI_PREPARE_DEF,
            OMNI_HELPERS,
            already="_minicpmo45_require_decision",
        ),
        "omni_prepare": _replace_once(omni, OMNI_PREPARE_ANCHOR, OMNI_PREPARE_PATCH, already="sampled_is_listen=None"),
        "omni_finalize": _replace_once(omni, OMNI_FINALIZE_ANCHOR, OMNI_FINALIZE_PATCH, already="require_decision.override_listen"),
        "omni_finished": _replace_once(omni, OMNI_FINISHED_ANCHOR, OMNI_FINISHED_PATCH, already="finished_sessions"),
        "runner": _replace_once(runner, RUNNER_ANCHOR, RUNNER_PATCH, already='payload["response_required"] = True'),
        "input_init": _replace_once(inp, INPUT_INIT_ANCHOR, INPUT_INIT_PATCH, already="self._require_response = False"),
        "input_clear": _replace_once(inp, INPUT_CLEAR_ANCHOR, INPUT_CLEAR_PATCH, already="self._require_turn_id = None"),
        "input_speech": _replace_once(inp, INPUT_SPEECH_ANCHOR, INPUT_SPEECH_PATCH, already="self._require_response = True"),
        "input_out": _replace_once(inp, INPUT_OUT_ANCHOR, INPUT_OUT_PATCH, already='out["response_required"] = True'),
        "input_commit": _replace_once(inp, INPUT_COMMIT_ANCHOR, INPUT_COMMIT_PATCH, already='payload["response_required"] = True'),
        "input_commit_reset": _replace_once(inp, INPUT_COMMIT_RESET_ANCHOR, INPUT_COMMIT_RESET_PATCH, already="self._require_turn_id = None"),
        "input_flush": _replace_once(
            inp,
            INPUT_FLUSH_ANCHOR,
            INPUT_FLUSH_PATCH,
            already='payload["turn_id"] = self._require_turn_id\n        return self.append',
        ),
        "version": version,
    }
    for path, marker in (
        (omni, "_minicpmo45_require_decision"),
        (runner, 'payload["response_required"] = True'),
        (inp, 'out["response_required"] = True'),
        (dest_policy, "def decide_require_response"),
    ):
        if marker not in path.read_text(encoding="utf-8"):
            raise PatchError(f"patch verification failed for {path}")
    return status


def main() -> int:
    try:
        status = apply()
    except PatchError as exc:
        print(f"model-server FATAL: MiniCPM require-response patch failed: {exc}", file=sys.stderr)
        return 1
    print("model-server MiniCPM require-response patch:", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
