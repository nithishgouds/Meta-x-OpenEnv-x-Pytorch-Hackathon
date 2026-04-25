"""Shared training observability helpers used by train_sft.py and train_grpo.py.

Goals
-----
- Capture every stdout/stderr line into a persistent train.log so HF Jobs runs
  remain debuggable after the container exits.
- Provide a single function to invoke plot_training_logs.py at the end of
  training so plots ship inside the run's output directory (no manual step).
- Keep a tiny final_metrics.json with headline numbers for quick inspection.

All helpers write incrementally where possible and never block on console
output, so they are safe inside Hugging Face Jobs and other non-interactive
runners.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Iterable


class TeeStream(io.TextIOBase):
    """Duplicate writes to multiple streams. Used to mirror stdout/stderr to
    a persistent log file on disk while still showing them in the container
    console.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):  # type: ignore[override]
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):  # type: ignore[override]
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def install_console_logger(output_dir: str, stage: str) -> str:
    """Tee stdout and stderr into ``<output_dir>/train.log``.

    Returns the path to the log file. Safe to call multiple times — only the
    first call installs the tee.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    if getattr(sys.stdout, "_opssim_tee", False):
        return log_path

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    log_file.write(
        f"\n=== [{stage}] start {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} ===\n"
    )
    log_file.flush()

    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    sys.stdout._opssim_tee = True  # type: ignore[attr-defined]
    sys.stderr._opssim_tee = True  # type: ignore[attr-defined]
    return log_path


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    """Append a single JSON object as one JSONL line and flush immediately.

    Each writer opens/closes the file so a kill mid-step does not lose the
    previous line. Use for step-wise metrics whose data loss would hurt
    debugging more than the open/close overhead.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_safe) + "\n")
        handle.flush()


def _json_safe(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def write_final_metrics(output_dir: str, payload: dict[str, Any]) -> str:
    """Persist a small headline-numbers JSON to ``<output_dir>/final_metrics.json``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "final_metrics.json")
    payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_safe)
    return path


def auto_generate_plots(
    output_dir: str,
    stage: str,
    extra_args: Iterable[str] = (),
) -> list[str]:
    """Invoke plot_training_logs.py against ``output_dir`` so plots ship
    inside the run output. Returns the list of created files (best-effort).

    Failures are logged but never raised — plots are nice-to-have, never a
    reason to fail a multi-hour training run.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    arg = "--sft-dir" if stage == "sft" else "--grpo-dir"
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_training_logs.py")
    cmd = [
        sys.executable,
        script,
        arg, output_dir,
        "--output-dir", plots_dir,
        *extra_args,
    ]
    print(f"[plots] running: {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)
        if result.stdout:
            print(result.stdout, flush=True)
        if result.returncode != 0:
            print(f"[plots] non-zero exit {result.returncode}; stderr:\n{result.stderr}", flush=True)
    except Exception as exc:
        print(f"[plots] failed to run plot_training_logs.py: {exc}", flush=True)
        return []

    if not os.path.isdir(plots_dir):
        return []
    return sorted(os.path.join(plots_dir, name) for name in os.listdir(plots_dir))
