"""Validate vLLM-Omni deploy YAML before boot (fail fast with readable errors)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _stage_limits(stage: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    mml = stage.get("max_model_len")
    mbt = stage.get("max_num_batched_tokens")
    sampling = stage.get("default_sampling_params") or {}
    max_tokens = sampling.get("max_tokens")
    return (
        int(mml) if mml is not None else None,
        int(mbt) if mbt is not None else None,
        int(max_tokens) if max_tokens is not None else None,
    )


def validate_deploy(path: Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"PyYAML required to validate deploy config: {exc}") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    stages = data.get("stages") or []
    if not stages:
        raise RuntimeError(f"deploy config has no stages: {path}")

    errors: list[str] = []
    summary: list[str] = []
    for stage in stages:
        sid = stage.get("stage_id", "?")
        mml, mbt, max_tokens = _stage_limits(stage)
        eager = stage.get("enforce_eager")
        summary.append(
            f"stage {sid}: max_model_len={mml} max_num_batched_tokens={mbt} "
            f"max_tokens={max_tokens} enforce_eager={eager}"
        )
        if mbt is not None and mml is None:
            errors.append(
                f"stage {sid}: max_model_len unset but max_num_batched_tokens={mbt}; "
                "vLLM may default max_model_len above batched tokens and crash the orchestrator"
            )
        if mml is not None and mbt is not None and mbt < mml:
            errors.append(
                f"stage {sid}: max_num_batched_tokens ({mbt}) < max_model_len ({mml})"
            )
        if mml is not None and max_tokens is not None and max_tokens > mml:
            errors.append(
                f"stage {sid}: default_sampling_params.max_tokens ({max_tokens}) "
                f"> max_model_len ({mml})"
            )
    for line in summary:
        print(f"model-server deploy {line}")
    if errors:
        for err in errors:
            print(f"model-server FATAL deploy validation: {err}", file=sys.stderr)
        sys.exit(1)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: validate_deploy.py /path/to/deploy.yaml", file=sys.stderr)
        sys.exit(2)
    validate_deploy(Path(sys.argv[1]))
