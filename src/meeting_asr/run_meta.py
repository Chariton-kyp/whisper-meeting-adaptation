"""Provenance recorded next to every result: code version, environment and hardware state."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect() -> dict:
    import datasets
    import peft
    import torch
    import transformers

    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total / 1e9, 2),
            "vram_free_at_start_gb": round(free / 1e9, 2),
        }
    smi = None
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total,temperature.gpu,clocks.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "gpu": gpu,
        "nvidia_smi": smi,
    }
