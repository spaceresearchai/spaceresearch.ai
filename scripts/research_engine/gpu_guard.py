#!/usr/bin/env python3
"""
GPU Safety Guard & Runtime Resolver (`gpu_guard.py`).

Checks live NVIDIA GPU memory via `nvidia-smi` before launching experiments so
autonomous research jobs never starve or OOM live inference services (e.g. Basalt).
Also locates the local PyTorch+Transformers CUDA Python environment automatically.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_CONTAINER = Path(
    "/home/marcio/Documents/Codex/2026-11-19/i-want-to-serve-inference-create/longinus/.runtime/video/lightwan-qa/container"
)
DEFAULT_CORERANK_ROOT = Path("/home/marcio/Projects/corerank-5090")


@dataclass
class GpuSnapshot:
    available: bool
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    util_pct: int
    driver: str
    processes: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "available": self.available,
            "name": self.name,
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "util_pct": self.util_pct,
            "driver": self.driver,
            "processes": self.processes,
        }


def query_gpu() -> GpuSnapshot:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        if not out:
            raise RuntimeError("empty nvidia-smi output")
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        name = parts[0]
        total_mb, used_mb, free_mb, util_pct = (int(float(parts[i])) for i in (1, 2, 3, 4))
        driver = parts[5] if len(parts) > 5 else "unknown"

        procs: List[Dict[str, str]] = []
        try:
            pout = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                text=True,
                timeout=5,
            ).strip()
            for line in pout.splitlines():
                if not line.strip():
                    continue
                pp = [x.strip() for x in line.split(",")]
                if len(pp) >= 3:
                    procs.append({"pid": pp[0], "name": Path(pp[1]).name, "used_memory": pp[2]})
        except Exception:
            pass

        return GpuSnapshot(
            available=True,
            name=name,
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=free_mb,
            util_pct=util_pct,
            driver=driver,
            processes=procs,
        )
    except Exception:
        return GpuSnapshot(
            available=False,
            name="CPU-only",
            total_mb=0,
            used_mb=0,
            free_mb=0,
            util_pct=0,
            driver="none",
            processes=[],
        )


def check_vram_headroom(min_free_mb: int = 2800) -> Tuple[bool, GpuSnapshot, str]:
    snap = query_gpu()
    if not snap.available:
        return False, snap, "No NVIDIA GPU detected via nvidia-smi."
    if snap.free_mb < min_free_mb:
        return (
            False,
            snap,
            f"Insufficient free VRAM on {snap.name}: {snap.free_mb} MiB free < {min_free_mb} MiB safety threshold "
            f"(used={snap.used_mb}/{snap.total_mb} MiB).",
        )
    return True, snap, f"GPU ready: {snap.name} ({snap.free_mb} MiB free / {snap.total_mb} MiB total)."


def resolve_torch_python() -> Tuple[str, Dict[str, str]]:
    """
    Return (python_executable, env_dict) that has PyTorch CUDA, transformers, numpy, scipy, and corerank importable.
    """
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    py_cand = DEFAULT_CONTAINER / "rootfs" / "app" / "miniconda" / "bin" / "python3.11"
    site_cand = DEFAULT_CONTAINER / "user" / "lib" / "python3.11" / "site-packages"
    extra_paths: List[str] = []
    if site_cand.exists():
        extra_paths.append(str(site_cand))
    if DEFAULT_CORERANK_ROOT.exists():
        extra_paths.append(str(DEFAULT_CORERANK_ROOT))
    if env.get("PYTHONPATH"):
        extra_paths.append(env["PYTHONPATH"])
    if extra_paths:
        env["PYTHONPATH"] = ":".join(extra_paths)

    if py_cand.exists():
        return str(py_cand), env
    return sys.executable, env


if __name__ == "__main__":
    ok, snap, msg = check_vram_headroom(2500)
    print(f"[gpu_guard] ok={ok} | {msg} | procs={snap.processes}")
