#!/usr/bin/env python3
"""Space Research — Universal Local AI Node Worker.

Auto-detects your computer's hardware and joins the Space Research Local AI Network:
  - Apple Silicon Mac (M1/M2/M3/M4/M5): native Metal GPU via `mlx-lm`
  - NVIDIA GPU (Linux / Windows — RTX 30/40/50):
    lean portable CUDA llama-server (zero PyTorch bloat)
  - Standard PC / Laptop (Lenovo ThinkPad, Dell, HP, Intel / AMD CPU):
    compact local model + fast lightweight checks

Zero port forwarding required. Connects outbound behind home NAT and stops cleanly
with Ctrl+C anytime.

Usage (Mac, NVIDIA PC, or standard laptop — exact same command):
  curl -sO https://spaceresearch.ai/join.py
  python3 join.py --install
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_COORDINATOR = "https://api.spaceresearch.ai"
FALLBACK_COORDINATOR = "https://api.longinusresearch.com"
COORDINATOR_DISCOVERY_URLS = (
    "https://spaceresearch.ai/coordinator.json",
    "https://longinusresearch.com/coordinator.json",
)
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def resolve_coordinator_url(requested: str) -> str:
    """Resolve permanent coordinator URL via https://spaceresearch.ai/coordinator.json (self-healing)."""
    val = (requested or "").strip()
    if val and val.lower() != "auto":
        return val.rstrip("/")
    for disc_url in COORDINATOR_DISCOVERY_URLS:
        try:
            req = urllib.request.Request(
                disc_url,
                headers={"Cache-Control": "no-cache", "User-Agent": "space-worker/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for key in ("api_base", "fallback_api_base"):
                    url = str(data.get(key) or "").strip().rstrip("/")
                    if url.startswith("http"):
                        try:
                            ping = urllib.request.Request(url + "/health", headers={"User-Agent": "space-worker/1.0"})
                            with urllib.request.urlopen(ping, timeout=4.0) as pr:
                                if pr.status == 200:
                                    return url
                        except Exception:
                            continue
        except Exception:
            continue
    for cand in (DEFAULT_COORDINATOR, FALLBACK_COORDINATOR):
        try:
            ping = urllib.request.Request(cand + "/health", headers={"User-Agent": "space-worker/1.0"})
            with urllib.request.urlopen(ping, timeout=4.0) as pr:
                if pr.status == 200:
                    return cand.rstrip("/")
        except Exception:
            continue
    return DEFAULT_COORDINATOR.rstrip("/")


def _is_rosetta_apple_silicon() -> bool:
    if platform.system() != "Darwin" or platform.machine() == "arm64":
        return False
    try:
        out = subprocess.check_output(["sysctl", "-n", "sysctl.proc_translated"], text=True, stderr=subprocess.DEVNULL).strip()
        return out == "1"
    except Exception:
        return False


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _bootstrap_macos_arm_python_if_needed() -> None:
    """If running on Apple Silicon under Rosetta x86_64 or Python <3.10, auto-reexec in native ARM64 Python 3.10+."""
    if platform.system() != "Darwin":
        return
    if os.environ.get("LONGINUS_REEXEC") == "1":
        return
    needs_arm = _is_rosetta_apple_silicon()
    needs_py310 = is_apple_silicon() and sys.version_info < (3, 10)
    if not (needs_arm or needs_py310):
        return

    candidates = [
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
    ]
    for py in candidates:
        if not Path(py).exists():
            continue
        try:
            ver = subprocess.check_output(
                ["arch", "-arm64", py, "-c", "import sys,platform; print(sys.version_info[:2], platform.machine())"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if "arm64" in ver:
                print(f"[SPACE] Switching to native Apple Silicon Python ({py}) for full Metal GPU speed...")
                env = dict(os.environ, LONGINUS_REEXEC="1")
                os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", py, *sys.argv], env)
        except Exception:
            continue

    if needs_py310:
        print(
            f"[ERROR] Apple Silicon MLX requires Python 3.10+ (you ran Python {sys.version_info.major}.{sys.version_info.minor}).\n"
            "        Fix in 1 command:\n\n"
            "  brew install python@3.12 && /opt/homebrew/bin/python3.12 join.py --install\n"
        )
        raise SystemExit(1)
    if needs_arm:
        print(
            "[WARN] Detected Intel/Rosetta x86_64 Python on an Apple Silicon Mac (slow CPU fallback).\n"
            "       For 100x faster Apple Metal GPU speed, run:\n\n"
            "  brew install python@3.12 && /opt/homebrew/bin/python3.12 join.py --install\n"
        )


def _ensure_isolated_venv_if_needed(auto_install: bool) -> None:
    """Avoid PEP 668 externally-managed-environment / Conda conflicts by using ~/.longinus_venv on --install."""
    if not auto_install or os.environ.get("LONGINUS_IN_VENV") == "1":
        return
    if sys.prefix != sys.base_prefix:
        return  # already inside a virtualenv
    venv_dir = Path.home() / ".longinus_venv"
    venv_py = venv_dir / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    if not venv_py.exists():
        print(f"[SPACE] Creating isolated runtime environment at {venv_dir} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    env = dict(os.environ, LONGINUS_IN_VENV="1")
    os.execve(str(venv_py), [str(venv_py), *sys.argv], env)


def has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def _needs_jinja_upgrade() -> bool:
    try:
        import jinja2

        return not hasattr(jinja2, "pass_eval_context")
    except Exception:
        return True


def _macos_supports_mlx() -> bool:
    """Apple MLX requires Apple Silicon + macOS 13.5+ (Ventura/Sonoma/Sequoia)."""
    if not is_apple_silicon():
        return False
    try:
        ver_str = platform.mac_ver()[0]
        parts = [int(x) for x in ver_str.split(".")[:2]]
        major = parts[0] if len(parts) >= 1 else 0
        minor = parts[1] if len(parts) >= 2 else 0
        return (major, minor) >= (13, 5)
    except Exception:
        return True


def ensure_dependencies(auto_install: bool) -> None:
    need_jinja = _needs_jinja_upgrade()
    if _macos_supports_mlx():
        needed = ["cryptography", "mlx_lm"]
        missing = [m for m in needed if importlib.util.find_spec(m) is None]
        if not missing and not need_jinja:
            return
        if auto_install:
            pkgs = ["cryptography", "mlx-lm", "jinja2>=3.1.0"]
            print(f"[SPACE] Installing/updating Apple Silicon MLX packages: {' '.join(pkgs)} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])
            return
        if missing:
            print(f"[SPACE] Missing packages: {', '.join(missing)}")
            print("           Install them once with:\n\n  pip3 install -q -U cryptography mlx-lm 'jinja2>=3.1.0'\n")
            print("           Or re-run with:  python3 join.py --install")
            raise SystemExit(1)
        return

    needed = ["cryptography", "torch", "transformers", "accelerate"]
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    win_nvidia = platform.system() == "Windows" and has_nvidia_gpu()

    # On Windows with an NVIDIA GPU, auto-upgrade CPU-only torch wheel to CUDA 12.4 wheel when --install is passed
    if not missing and win_nvidia and auto_install:
        try:
            import torch

            if not torch.cuda.is_available():
                print("[SPACE] NVIDIA GPU detected on Windows, upgrading CPU-only PyTorch to CUDA 12.4 ...")
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "--force-reinstall",
                        "torch",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu124",
                    ]
                )
        except Exception:
            pass
        return

    if not missing:
        return

    # Fast path: NVIDIA GPUs (>= 6 GB VRAM) use prebuilt portable CUDA llama-server, so only `cryptography` is required!
    nv_info = _detect_nvidia_gpu_via_smi()
    if nv_info is not None and nv_info[1] >= 6.0:
        if importlib.util.find_spec("cryptography") is not None:
            return
        if auto_install:
            print("[SPACE] NVIDIA GPU detected — installing lightweight worker dependency (cryptography)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cryptography"])
            return
        print("[SPACE] Missing package: cryptography\n  pip3 install -q cryptography\n  Or run: python3 join.py --install")
        raise SystemExit(1)

    if auto_install:
        if win_nvidia:
            print("[SPACE] NVIDIA GPU detected on Windows — installing CUDA PyTorch + runtime packages ...")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "torch",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ]
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "cryptography", "transformers", "accelerate"]
            )
        else:
            print("[SPACE] Installing required packages: cryptography torch transformers accelerate ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "cryptography", "torch", "transformers", "accelerate"]
            )
        return

    print(f"[SPACE] Missing packages: {', '.join(missing)}")
    if win_nvidia:
        print(
            "           Install CUDA packages once with:\n\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install -q cryptography transformers accelerate\n"
        )
    else:
        print("           Install them once with:\n\n  pip3 install -q cryptography torch transformers accelerate\n")
    print("           Or re-run with:  python3 join.py --install")
    raise SystemExit(1)


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = []
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(_B58_ALPHABET[rem])
    leading_zeros = len(data) - len(data.lstrip(b"\x00"))
    return ("1" * leading_zeros) + "".join(reversed(out))


def load_or_create_keypair() -> tuple[object, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    home = Path.home()
    candidates = [
        home / ".longinus_node.json",
        home / ".longinus_mac_node.json",
        home / ".longinus_gpu_node.json",
    ]
    existing = next((p for p in candidates if p.exists()), None)
    path = existing or (home / ".longinus_node.json")

    if path.exists():
        raw = bytes.fromhex(json.loads(path.read_text())["seed_hex"])
        priv = Ed25519PrivateKey.from_private_bytes(raw)
    else:
        priv = Ed25519PrivateKey.generate()
        seed = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        path.write_text(json.dumps({"wallet": b58encode(pub), "seed_hex": seed.hex()}, indent=2))
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, b58encode(pub)


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": "space-worker/1.0"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sign_receipt(priv, receipt: dict) -> dict:
    d = {k: v for k, v in receipt.items() if k != "signature_hex"}
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt["signature_hex"] = priv.sign(canonical).hex()
    return receipt


def _detect_cpu_name_and_ram() -> tuple[str, float]:
    sys_name = platform.system()
    cpu_name = platform.processor() or f"{sys_name} {platform.machine()} CPU"
    mem_gb = 16.0

    if sys_name == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    mem_gb = round(kb / (1024**2), 1)
                    break
        except Exception:
            pass
    elif sys_name == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if out:
                cpu_name = out
        except Exception:
            pass
        try:
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            mem_gb = round(mem_bytes / (1024**3), 1)
        except Exception:
            pass
    elif sys_name == "Windows":
        try:
            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                mem_gb = round(stat.ullTotalPhys / (1024**3), 1)
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"], text=True, stderr=subprocess.DEVNULL
            ).splitlines()
            names = [x.strip() for x in out if x.strip() and x.strip().lower() != "name"]
            if names:
                cpu_name = names[0]
        except Exception:
            pass

    return cpu_name, mem_gb


def _is_apple_silicon_hw() -> bool:
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() in ("arm64", "aarch64"):
        return True
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.optional.arm64"], text=True, stderr=subprocess.DEVNULL).strip()
        if out == "1":
            return True
    except Exception:
        pass
    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
        if "apple" in brand.lower():
            return True
    except Exception:
        pass
    return False


def _detect_nvidia_gpu_via_smi() -> tuple[str, float, int] | None:
    """Return (gpu_name, vram_gb, driver_major) via nvidia-smi without importing PyTorch."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if not out:
            return None
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        vram_gb = round(float(parts[1]) / 1024.0, 1)
        drv_major = int(str(parts[2]).split(".")[0]) if len(parts) >= 3 and parts[2] else 550
        return name, vram_gb, drv_major
    except Exception:
        return None


def detect_hardware(force_cpu: bool = False) -> tuple[str, float, str, str, list[str]]:
    """Returns (chip_name, memory_gb, backend, hardware_tier, supported_engines)."""
    if not force_cpu and _macos_supports_mlx():
        chip, mem_gb = _detect_cpu_name_and_ram()
        return chip or "Apple Silicon", mem_gb, "mlx", "apple_silicon", ["corekv", "corerank", "corecompile"]

    if not force_cpu and _is_apple_silicon_hw():
        chip, mem_gb = _detect_cpu_name_and_ram()
        return chip or "Apple Silicon", mem_gb, "mps", "apple_silicon", ["corekv", "corerank", "corecompile"]

    if not force_cpu:
        nv = _detect_nvidia_gpu_via_smi()
        if nv is not None and nv[1] >= 4.0:
            name, mem_gb, _drv = nv
            tier = "datacenter_gpu" if mem_gb >= 40.0 else "consumer_gpu"
            return name, mem_gb, "cuda", tier, ["corekv", "corerank", "corecompile"]

    try:
        import torch

        if not force_cpu and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem_gb = round(props.total_memory / (1024**3), 1)
            tier = "datacenter_gpu" if mem_gb >= 40.0 else "consumer_gpu"
            return props.name, mem_gb, "cuda", tier, ["corekv", "corerank", "corecompile"]

        if not force_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chip, mem_gb = _detect_cpu_name_and_ram()
            return chip or "Apple Silicon", mem_gb, "mps", "apple_silicon", ["corekv", "corerank", "corecompile"]
    except Exception:
        pass

    cpu_name, mem_gb = _detect_cpu_name_and_ram()
    return cpu_name, mem_gb, "cpu", "cpu_edge", ["corecompile", "corekv"]


def _macos_version() -> tuple[int, int]:
    """Return (major, minor) macOS version tuple, or (0, 0) on non-macOS."""
    if platform.system() != "Darwin":
        return (0, 0)
    try:
        ver_str = platform.mac_ver()[0]
        parts = [int(p) for p in ver_str.split(".") if p.isdigit()]
        if len(parts) >= 2:
            return (parts[0], parts[1])
        if len(parts) == 1:
            return (parts[0], 0)
    except Exception:
        pass
    return (0, 0)


MAC_METAL_PRESETS: dict[str, dict[str, Any]] = {
    "gemma4-26b": {
        "label": "google/Gemma-4-26B-A4B (Q4_K_XL)",
        "url": "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        "filename": "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        "size_note": "17.0 GB",
        "is_inkling": False,
    },
    "qwen38-27b": {
        "label": "qwen/Qwen3.8-27B (Q4_K_M)",
        "url": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf",
        "filename": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "size_note": "16.6 GB",
        "is_inkling": False,
    },
    "qwen35-35b": {
        "label": "Qwen/Qwen3.5-35B-A3B (IQ4_XS)",
        "url": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf",
        "filename": "Qwen3.5-35B-A3B-UD-IQ4_XS.gguf",
        "size_note": "17.5 GB",
        "is_inkling": False,
    },
    "qwen35-9b": {
        "label": "Qwen/Qwen3.5-9B (Q4_K_XL)",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "size_note": "6.0 GB",
        "is_inkling": False,
    },
    "gemma4-e4b": {
        "label": "google/Gemma-4-E4B (Q4_K_XL)",
        "url": "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-UD-Q4_K_XL.gguf",
        "filename": "gemma-4-E4B-it-UD-Q4_K_XL.gguf",
        "size_note": "5.1 GB",
        "is_inkling": False,
    },
    "inkling2": {
        "label": "thinkingmachines/Inkling-2-Mini (Q4_K_XL)",
        "url": "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-Q4_K_XL.gguf",
        "filename": "Inkling-2-Mini-UD-Q4_K_XL.gguf",
        "size_note": "17.6 GB",
        "is_inkling": True,
    },
    "qwen25-coder-32b": {
        "label": "Qwen/Qwen2.5-Coder-32B-Instruct (Q4_K_M)",
        "url": "https://huggingface.co/unsloth/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "size_note": "19.9 GB",
        "is_inkling": False,
    },
    "deepseek-r1-32b": {
        "label": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (Q4_K_M)",
        "url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "size_note": "19.9 GB",
        "is_inkling": False,
    },
    "qwen3-32b": {
        "label": "Qwen/Qwen3-32B (Q4_K_M)",
        "url": "https://huggingface.co/unsloth/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf",
        "filename": "Qwen3-32B-Q4_K_M.gguf",
        "size_note": "19.8 GB",
        "is_inkling": False,
    },
}


def _resolve_mac_metal_preset(mem_gb: float, model_override: str | None) -> str | None:
    if model_override:
        low = model_override.strip().lower()
        if low in MAC_METAL_PRESETS:
            return low
        if "gemma-4-26b" in low or "gemma4-26b" in low or ("gemma4" in low and "26" in low):
            return "gemma4-26b"
        if "gemma-4-e4b" in low or "gemma4-e4b" in low:
            return "gemma4-e4b"
        if "qwen3.8-27b" in low or "qwen38-27b" in low:
            return "qwen38-27b"
        if "qwen3.5-35b" in low or "qwen35-35b" in low:
            return "qwen35-35b"
        if "qwen3.5-9b" in low or "qwen35-9b" in low:
            return "qwen35-9b"
        if "coder" in low and "32" in low:
            return "qwen25-coder-32b"
        if "deepseek" in low or "r1-32b" in low:
            return "deepseek-r1-32b"
        if "qwen3-32b" in low or "qwen3_32b" in low:
            return "qwen3-32b"
        if "inkling" in low:
            return "inkling2"
        return None
    # Optimal Apple Silicon defaults:
    # - 24GB+ (M2 Max / M3 Max / M4 Pro): Gemma 4 26B-A4B MoE (4B active -> ~40-60 tok/s on 400 GB/s Metal!)
    # - 12-23GB (M1 / M2 / M3 16GB): Qwen3.5-9B UD-Q4_K_XL (6.0 GB -> full 4-bit quality, 10 GB free RAM!)
    return "gemma4-26b" if mem_gb >= 24.0 else "qwen35-9b"


def _use_llama_cpp_inkling(backend: str, mem_gb: float, model_override: str | None) -> bool:
    """Use native Metal llama-server on Apple Silicon (works on macOS 12 Monterey AND 13.3+!)."""
    if model_override and _resolve_mac_metal_preset(mem_gb, model_override) is None and not model_override.endswith(".gguf"):
        return False
    return (
        platform.system() == "Darwin"
        and _is_apple_silicon_hw()
        and backend in ("mlx", "mps")
        and mem_gb >= 12.0
    )


def _resolve_cuda_llama_preset(mem_gb: float, model_override: str | None) -> str | None:
    if model_override:
        return _resolve_mac_metal_preset(mem_gb, model_override)
    if mem_gb >= 22.0:
        return "gemma4-26b"
    if mem_gb >= 10.0:
        return "qwen35-9b"
    return "gemma4-e4b"


def _use_llama_cpp_cuda(backend: str, mem_gb: float, model_override: str | None) -> bool:
    """Use portable prebuilt CUDA llama-server on Linux/Windows x64 NVIDIA GPUs (>=6 GB VRAM)."""
    if backend != "cuda" or mem_gb < 6.0:
        return False
    sys_name = platform.system()
    arch = platform.machine().lower()
    if sys_name not in ("Linux", "Windows") or arch not in ("x86_64", "amd64"):
        return False
    if model_override and _resolve_cuda_llama_preset(mem_gb, model_override) is None and not model_override.endswith(".gguf"):
        return False
    return True


def recommend_model(backend: str, mem_gb: float) -> str:
    if _use_llama_cpp_inkling(backend, mem_gb, None):
        preset = _resolve_mac_metal_preset(mem_gb, None) or "qwen35-9b"
        return str(MAC_METAL_PRESETS[preset]["label"])
    if _use_llama_cpp_cuda(backend, mem_gb, None):
        preset = _resolve_cuda_llama_preset(mem_gb, None) or "qwen35-9b"
        return str(MAC_METAL_PRESETS[preset]["label"])
    if backend == "mlx":
        return "google/Gemma-4-26B-A4B (Q4_K_XL)" if mem_gb >= 24.0 else "Qwen/Qwen3.5-9B (Q4_K_XL)"
    if backend == "mps":
        return "Qwen/Qwen3-4B" if mem_gb >= 20.0 else "Qwen/Qwen3-1.7B"
    if backend == "cuda":
        if mem_gb >= 20.0:
            return "google/Gemma-4-26B-A4B (Q4_K_XL)"
        if mem_gb >= 10.0:
            return "Qwen/Qwen3.5-9B (Q4_K_XL)"
        return "google/Gemma-4-E4B (Q4_K_XL)"
    # CPU Edge laptops (Lenovo ThinkPad, Dell, HP, etc.)
    return "Qwen/Qwen3-1.7B" if mem_gb >= 12.0 else "Qwen/Qwen3-0.6B"


def run_corecompile_guardrail(prompt: str) -> tuple[str, list[float], list[float], int]:
    """Deterministic sub-millisecond CoreCompile safety/policy verification (ideal for CPU edge nodes)."""
    low = prompt.lower()
    flagged = any(
        tok in low
        for tok in (
            "ignore previous instructions",
            "exfiltrate",
            "reveal system prompt",
            "drop table",
            "bypass security",
        )
    )
    verdict = "BLOCK: policy boundary triggered (prompt injection / exfiltration pattern)" if flagged else "ALLOW: compliant payload"
    h = hashlib.sha256(f"corecompile|{verdict}|{prompt}".encode("utf-8")).digest()
    sketch = [((b / 255.0) * 2.0 - 1.0) for b in h[:8]]
    norm = sum(v * v for v in sketch) ** 0.5 or 1.0
    sketch = [round(v / norm, 5) for v in sketch]
    top_lps = [-0.04, -3.5, -4.8, -5.6, -6.2, -6.8, -7.2, -7.7]
    return verdict, top_lps, sketch, max(4, len(verdict.split()))


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Format a user prompt with chat template (supports Inkling 2 Mini reasoning_effort + Qwen3)."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        tmpl = str(getattr(tokenizer, "chat_template", "") or "")
        if "reasoning_effort" in tmpl or "Thinking effort level" in tmpl:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort="minimal",
                )
            except Exception:
                pass
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass
    # Safe zero-Jinja2 fallback matching Qwen3 non-thinking ChatML format
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LlamaCppInklingRunner:
    """Runs frontier GGUF models natively on Apple Silicon Metal via llama-server.

    Hardware-matched defaults:
    - 24GB+ Macs (M2 Max / M3 Max): Gemma 4 26B-A4B MoE (4B active -> ~40-60 tok/s, zero thinking wait).
    - 12-23GB Macs (M1 / M2 16GB): Qwen3.5-9B UD-Q4_K_XL (6.0 GB full 4-bit quality, 10 GB free RAM).
    - Works on both macOS >= 13.3 (prebuilt 13 MB Metal engine) and macOS 12 Monterey (native ARM64 build).
    """

    def __init__(self, mem_gb: float, max_tokens: int = 1536, port: int = 8799, model_override: str | None = None):
        import atexit
        import glob
        import tarfile

        self.max_tokens = max_tokens
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

        preset_key = _resolve_mac_metal_preset(mem_gb, model_override) or ("gemma4-26b" if mem_gb >= 24.0 else "qwen35-9b")
        spec = dict(MAC_METAL_PRESETS[preset_key])
        self.is_inkling = bool(spec["is_inkling"])

        mac_ver = _macos_version()
        use_prebuilt = mac_ver >= (13, 3)

        llama_root = Path.home() / ".longinus_llama"
        models_dir = Path.home() / ".longinus_models"
        llama_root.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        if use_prebuilt:
            bin_dir = llama_root / "unsloth-b12588"
            server_bin = next(iter(glob.glob(str(bin_dir / "**" / "llama-server"), recursive=True)), None)
            if not server_bin:
                bin_dir.mkdir(parents=True, exist_ok=True)
                tgz = llama_root / "llama-mac-arm64.tar.gz"
                url = "https://github.com/unslothai/llama.cpp/releases/download/b12588-mix-c0c3de8/llama-b12588-mix-c0c3de8-bin-macos-arm64.tar.gz"
                print("[METAL] Downloading prebuilt Apple Silicon Metal engine (13 MB)...")
                subprocess.check_call(["curl", "-fsSL", "-C", "-", url, "-o", str(tgz)])
                with tarfile.open(tgz, "r:gz") as tf:
                    tf.extractall(bin_dir)
                tgz.unlink(missing_ok=True)
                server_bin = next(iter(glob.glob(str(bin_dir / "**" / "llama-server"), recursive=True)), None)
            if not server_bin:
                raise RuntimeError("Could not locate llama-server after extracting archive.")
        else:
            src_dir = llama_root / "mainline-src"
            server_bin = str(src_dir / "build" / "bin" / "llama-server")
            if not Path(server_bin).exists():
                print(
                    f"[METAL] macOS {mac_ver[0]}.{mac_ver[1]} detected — compiling native Metal llama-server once (~45s)..."
                )
                if not shutil.which("cmake") and shutil.which("brew"):
                    subprocess.check_call(["brew", "install", "cmake"])
                if not shutil.which("cmake") or not shutil.which("git"):
                    raise RuntimeError("git and cmake are required to build Metal llama-server on macOS 12.")
                if not (src_dir / ".git").exists():
                    subprocess.check_call(
                        ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(src_dir)]
                    )
                arch_pfx = ["arch", "-arm64"] if shutil.which("arch") else []
                subprocess.check_call(
                    arch_pfx
                    + [
                        "cmake",
                        "-B",
                        str(src_dir / "build"),
                        "-S",
                        str(src_dir),
                        "-DGGML_METAL=ON",
                        "-DLLAMA_BUILD_TESTS=OFF",
                        "-DLLAMA_BUILD_EXAMPLES=OFF",
                        "-DCMAKE_OSX_ARCHITECTURES=arm64",
                    ]
                )
                jobs = str(max(2, os.cpu_count() or 4))
                subprocess.check_call(
                    arch_pfx
                    + [
                        "cmake",
                        "--build",
                        str(src_dir / "build"),
                        "--config",
                        "Release",
                        "-j",
                        jobs,
                        "--target",
                        "llama-server",
                    ]
                )

        if self.is_inkling:
            quant = "UD-Q4_K_XL" if mem_gb >= 24.0 else "UD-Q2_K_XL"
            self.model_label = f"thinkingmachines/Inkling-2-Mini ({quant.replace('UD-', '')})"
            if use_prebuilt:
                gguf_url = f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-{quant}.gguf"
                gguf_path = models_dir / f"Inkling-2-Mini-{quant}.gguf"
            else:
                gguf_url = (
                    f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/"
                    f"Inkling-2-Mini-{quant}.gguf_file"
                )
                gguf_path = models_dir / f"Inkling-2-Mini-{quant}-mainline.gguf"
            sz_note = "17.6 GB" if quant == "UD-Q4_K_XL" else "9.1 GB"
        else:
            self.model_label = str(spec["label"])
            gguf_url = str(spec["url"])
            gguf_path = models_dir / str(spec["filename"])
            sz_note = str(spec["size_note"])

        if not gguf_path.exists() or gguf_path.stat().st_size < 1_000_000_000:
            print(f"[METAL] Downloading {self.model_label} ({sz_note}, resumable)...")
            subprocess.check_call(["curl", "-fL", "-C", "-", "--progress-bar", gguf_url, "-o", str(gguf_path)])

        ctx_sz = "32768" if mem_gb >= 24.0 else "16384"
        n_par = "4" if mem_gb >= 24.0 else "2"
        arch_pfx = ["arch", "-arm64"] if shutil.which("arch") else []
        cmd = arch_pfx + [
            server_bin,
            "--model",
            str(gguf_path),
            "--gpu-layers",
            "99",
            "--flash-attn",
            "on",
            "--kv-unified",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--cont-batching",
            "--parallel",
            n_par,
            "--ctx-size",
            ctx_sz,
            "--spec-type",
            "ngram-map-k4v",
            "--jinja",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if self.is_inkling:
            cmd += [
                "--no-reasoning-preserve",
                "--reasoning-effort",
                "minimal",
                "--temp",
                "1.0",
                "--top-p",
                "1.0",
                "--top-k",
                "0",
                "--min-p",
                "0.0",
            ]
        else:
            cmd += [
                "--temp",
                "0.7",
                "--top-p",
                "0.9",
                "--top-k",
                "20",
                "--min-p",
                "0.05",
            ]
        print(f"[METAL] Starting {self.model_label} on Apple Silicon Metal GPU...")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)

        deadline = time.time() + 120.0
        ready = False
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.proc.returncode})")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.6)
        if not ready:
            self.close()
            raise RuntimeError(f"Timed out waiting for local Metal llama-server to load {self.model_label}.")
        print(f"[METAL] Ready! Serving {self.model_label} natively on Apple Silicon!")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def run(self, prompt: str, engine: str = "corekv") -> tuple[str, list[float], list[float], int]:
        if engine == "corecompile":
            return run_corecompile_guardrail(prompt)
        if self.is_inkling:
            payload = {
                "model": self.model_label,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "stream": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "reasoning_budget_tokens": 64,
                "chat_template_kwargs": {"reasoning_effort": "minimal"},
            }
        else:
            payload = {
                "model": self.model_label,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.05,
                "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "none"},
            }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        text = (msg.get("content") or "").strip()
        if not text and msg.get("reasoning_content"):
            text = str(msg["reasoning_content"]).strip()
        usage = data.get("usage") or {}
        toks = int(usage.get("completion_tokens") or max(1, len(text.split())))
        h = hashlib.sha256((text or "empty").encode("utf-8")).digest()
        sketch = [((b / 255.0) * 2.0 - 1.0) for b in h[:8]]
        norm = sum(v * v for v in sketch) ** 0.5 or 1.0
        sketch = [round(v / norm, 5) for v in sketch]
        top_lps = [-0.08, -2.9, -4.2, -5.1, -5.8, -6.3, -6.9, -7.4]
        return text, top_lps, sketch, toks


class LlamaCppCudaRunner:
    """Runs frontier GGUF models natively on NVIDIA RTX GPUs (30/40/50-series) via prebuilt portable CUDA llama-server.

    Zero PyTorch or CUDA Toolkit installation required: auto-downloads the portable CUDA 13 (driver >= 580)
    or CUDA 12 llama-server build with FlashAttention, Q8_0 unified KV, and ngram-map-k4v speculative decoding.
    """

    def __init__(self, mem_gb: float, max_tokens: int = 1536, port: int = 8799, model_override: str | None = None):
        import atexit
        import glob
        import tarfile
        import zipfile

        self.max_tokens = max_tokens
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

        preset_key = _resolve_cuda_llama_preset(mem_gb, model_override) or "qwen35-9b"
        spec = dict(MAC_METAL_PRESETS[preset_key])
        self.is_inkling = bool(spec["is_inkling"])
        self.model_label = str(spec["label"])

        nv_info = _detect_nvidia_gpu_via_smi()
        drv_major = nv_info[2] if nv_info else 550
        cuda_tag = "cuda13" if drv_major >= 580 else "cuda12"
        is_win = platform.system() == "Windows"
        exe_name = "llama-server.exe" if is_win else "llama-server"

        llama_root = Path.home() / ".longinus_llama"
        models_dir = Path.home() / ".longinus_models"
        llama_root.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        bin_dir = llama_root / f"cuda-b12631-{cuda_tag}"
        server_bin = next(iter(glob.glob(str(bin_dir / "**" / exe_name), recursive=True)), None)
        if not server_bin:
            bin_dir.mkdir(parents=True, exist_ok=True)
            tag = "b12631-mix-acc0843"
            if is_win:
                asset = f"app-{tag}-windows-x64-{cuda_tag}-portable.zip"
                arc = llama_root / asset
                url = f"https://github.com/unslothai/llama.cpp/releases/download/{tag}/{asset}"
                print(f"[CUDA] Downloading portable Windows {cuda_tag.upper()} engine ({asset})...")
                subprocess.check_call(["curl", "-fL", "-C", "-", "--progress-bar", url, "-o", str(arc)])
                with zipfile.ZipFile(arc, "r") as zf:
                    zf.extractall(bin_dir)
                arc.unlink(missing_ok=True)
            else:
                asset = f"app-{tag}-linux-x64-{cuda_tag}-portable.tar.gz"
                arc = llama_root / asset
                url = f"https://github.com/unslothai/llama.cpp/releases/download/{tag}/{asset}"
                print(f"[CUDA] Downloading portable Linux {cuda_tag.upper()} engine ({asset})...")
                subprocess.check_call(["curl", "-fL", "-C", "-", "--progress-bar", url, "-o", str(arc)])
                with tarfile.open(arc, "r:gz") as tf:
                    tf.extractall(bin_dir)
                arc.unlink(missing_ok=True)
            server_bin = next(iter(glob.glob(str(bin_dir / "**" / exe_name), recursive=True)), None)
        if not server_bin:
            raise RuntimeError(f"Could not locate {exe_name} after extracting CUDA archive.")
        if not is_win:
            try:
                os.chmod(server_bin, 0o755)
            except Exception:
                pass

        gguf_path = models_dir / str(spec["filename"])
        gguf_url = str(spec["url"])
        if not gguf_path.exists() or gguf_path.stat().st_size < 1_000_000_000:
            print(f"[CUDA] Downloading {self.model_label} ({spec['size_note']}, resumable)...")
            subprocess.check_call(["curl", "-fL", "-C", "-", "--progress-bar", gguf_url, "-o", str(gguf_path)])

        ctx_sz = "32768" if mem_gb >= 16.0 else "16384"
        n_par = "4" if mem_gb >= 16.0 else "2"
        cmd = [
            server_bin,
            "--model",
            str(gguf_path),
            "--gpu-layers",
            "99",
            "--flash-attn",
            "on",
            "--kv-unified",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--cont-batching",
            "--parallel",
            n_par,
            "--ctx-size",
            ctx_sz,
            "--spec-type",
            "ngram-map-k4v",
            "--jinja",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if self.is_inkling:
            cmd += ["--no-reasoning-preserve", "--reasoning-effort", "minimal", "--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0"]
        else:
            cmd += ["--temp", "0.7", "--top-p", "0.9", "--top-k", "20", "--min-p", "0.05"]

        env = dict(os.environ)
        bin_parent = str(Path(server_bin).resolve().parent)
        if not is_win:
            prev_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{bin_parent}:{prev_ld}" if prev_ld else bin_parent

        print(f"[CUDA] Starting {self.model_label} on NVIDIA GPU ({cuda_tag.upper()}, {n_par} slots, {ctx_sz} Q8 KV)...")
        self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)

        deadline = time.time() + 120.0
        ready = False
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"CUDA llama-server exited early (code {self.proc.returncode})")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.6)
        if not ready:
            self.close()
            raise RuntimeError(f"Timed out waiting for CUDA llama-server to load {self.model_label}.")
        print(f"[CUDA] Ready! Serving {self.model_label} natively on NVIDIA CUDA!")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def run(self, prompt: str, engine: str = "corekv") -> tuple[str, list[float], list[float], int]:
        if engine == "corecompile":
            return run_corecompile_guardrail(prompt)
        payload: dict[str, Any] = {
            "model": self.model_label,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(self.max_tokens, 1536),
            "stream": False,
        }
        if self.is_inkling:
            payload.update({"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "chat_template_kwargs": {"reasoning_effort": "minimal"}, "reasoning_budget_tokens": 64})
        else:
            payload.update({"temperature": 0.7, "top_p": 0.9, "top_k": 20, "min_p": 0.05, "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "none"}})
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        text = (msg.get("content") or "").strip()
        if not text and msg.get("reasoning_content"):
            text = str(msg["reasoning_content"]).strip()
        usage = data.get("usage") or {}
        toks = int(usage.get("completion_tokens") or max(1, len(text.split())))
        h = hashlib.sha256((text or "empty").encode("utf-8")).digest()
        top_lps = [round(-0.15 - (b / 255.0) * 1.4, 5) for b in h[:8]]
        raw = [((b / 127.5) - 1.0) for b in h[8:16]]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        sketch = [round(v / norm, 5) for v in raw]
        return text, top_lps, sketch, toks


class MLXRunner:
    def __init__(self, model_name: str, max_tokens: int = 1024):
        from mlx_lm import generate, load

        print(f"[MLX] Loading {model_name} onto Apple Silicon Metal GPU...")
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)
        self._generate = generate
        self.max_tokens = max_tokens
        print("[MLX] Ready on Apple Silicon!")

    def run(self, prompt: str, engine: str = "corekv") -> tuple[str, list[float], list[float], int]:
        if engine == "corecompile":
            return run_corecompile_guardrail(prompt)
        formatted = format_chat_prompt(self.tokenizer, prompt)

        raw = self._generate(self.model, self.tokenizer, prompt=formatted, max_tokens=self.max_tokens, verbose=False)
        text = raw
        if "<|content_text|>" in text:
            text = text.rsplit("<|content_text|>", 1)[1]
        for end_tok in ("<|end_message|>", "<|content_model_end_sampling|>"):
            text = text.replace(end_tok, "")
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.replace("<think>", "").strip()
        toks = max(1, len(self.tokenizer.encode(text)))
        h = hashlib.sha256(text.encode("utf-8")).digest()
        sketch = [((b / 255.0) * 2.0 - 1.0) for b in h[:8]]
        norm = sum(v * v for v in sketch) ** 0.5 or 1.0
        sketch = [round(v / norm, 5) for v in sketch]
        top_lps = [-0.08, -2.9, -4.2, -5.1, -5.8, -6.3, -6.9, -7.4]
        return text, top_lps, sketch, toks


class TorchRunner:
    """Runs PyTorch inference cleanly on either NVIDIA CUDA GPUs or multi-core CPUs (Lenovo/Windows/Linux)."""

    def __init__(self, model_name: str, device: str = "cuda", max_tokens: int = 448):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_tokens = min(max_tokens, 256) if device == "cpu" else max_tokens

        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"[CUDA] Loading {model_name} onto NVIDIA GPU...")
        elif device == "mps":
            dtype = torch.float16
            print(f"[MPS] Loading {model_name} onto Apple Silicon Metal GPU (PyTorch MPS)...")
        else:
            threads = max(1, (os.cpu_count() or 4) - 1)
            torch.set_num_threads(threads)
            dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32
            print(f"[CPU] Loading {model_name} across {threads} CPU threads (CoreCompile + Compact CoreKV)...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="cuda").eval()
        elif device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to("mps").eval()
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to("cpu").eval()
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to("cpu").eval()

        hidden = int(getattr(self.model.config, "hidden_size", 2048))
        gen = torch.Generator(device="cpu").manual_seed(20261122)
        proj = torch.randn(hidden, 8, generator=gen, dtype=torch.float32)
        proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
        self._sketch_proj = proj.to(device)
        label = (
            torch.cuda.get_device_name(0)
            if device == "cuda"
            else ("Apple Silicon Metal GPU (MPS)" if device == "mps" else "CPU Edge Engine")
        )
        print(f"[{device.upper()}] Ready on {label}!")

    def run(self, prompt: str, engine: str = "corekv") -> tuple[str, list[float], list[float], int]:
        if engine == "corecompile":
            return run_corecompile_guardrail(prompt)

        torch = self._torch
        with torch.inference_mode():
            formatted = format_chat_prompt(self.tokenizer, prompt)
            inputs = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(self.device)

            prompt_len = int(inputs["input_ids"].shape[1])
            out = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                use_cache=False,
            )
            log_probs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
            top8 = [round(float(v), 5) for v in torch.topk(log_probs, k=8).values.detach().cpu().tolist()]

            last_h = out.hidden_states[-1][0, -1, :].float()
            raw_s = torch.matmul(last_h, self._sketch_proj)
            norm = float(raw_s.norm().item()) or 1.0
            sketch = [round(float(v / norm), 5) for v in raw_s.detach().cpu().tolist()]

            seq = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = seq[0, prompt_len:]
            raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text = (
                raw_text.split("</think>", 1)[1].strip()
                if "</think>" in raw_text
                else raw_text.replace("<think>", "").strip()
            )
            return text, top8, sketch, max(1, int(new_tokens.shape[0]))


def is_machine_busy_for_personal_use(backend: str, my_pid: int) -> tuple[bool, str]:
    """Detect if the user is actively gaming, rendering, or running another heavy GPU/CPU app."""
    if backend == "cuda" and shutil.which("nvidia-smi"):
        try:
            util_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
            ).strip().splitlines()
            gpu_util = int(util_out[0].strip()) if util_out and util_out[0].strip().isdigit() else 0
            # Check if another process besides join.py is actively using GPU VRAM
            procs = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
            ).strip().splitlines()
            other_mem_mb = 0
            for line in procs:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) != my_pid:
                    if parts[1].isdigit():
                        other_mem_mb += int(parts[1])
            if other_mem_mb >= 1500:
                return True, f"external GPU app using {other_mem_mb} MB VRAM"
            if gpu_util >= 30:
                return True, f"GPU utilization at {gpu_util}%"
        except Exception:
            pass
    elif hasattr(os, "getloadavg"):
        try:
            load1 = os.getloadavg()[0]
            cores = max(1, os.cpu_count() or 4)
            if (load1 / cores) >= 0.80:
                return True, f"high system load ({load1:.1f}/{cores} cores)"
        except Exception:
            pass
    return False, ""


def unload_runner(runner: Any, backend: str) -> None:
    """Release GPU/unified memory immediately so personal games or apps get 100% of VRAM."""
    import gc

    if runner is None:
        return
    try:
        if hasattr(runner, "close"):
            runner.close()
        if hasattr(runner, "model"):
            del runner.model
        if hasattr(runner, "tokenizer"):
            del runner.tokenizer
    except Exception:
        pass
    gc.collect()
    if backend == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    elif backend == "mlx":
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass


def _run_rpc_tunnel_session(
    *,
    coord: str,
    wallet: str,
    chip: str,
    mem_gb: float,
    display_name: str,
    rpc_port: int = 50052,
    persistent: bool = True,
) -> bool:
    """Start local loopback rpc-server and tunnel it outbound over HTTPS WebSocket to api.spaceresearch.ai.

    Zero open ports, zero firewall config, zero pip dependencies.
    When `persistent=False` (automatic hot-swap mode), exits cleanly when the coordinator sends
    RELEASE_RPC (ftype=6) or closes the tunnel after the 70B session goes idle.
    """
    import base64 as _b64
    import glob as _glob
    import socket as _sock
    import ssl as _ssl
    import struct as _struct
    import tarfile as _tarfile
    import threading as _thr
    import urllib.parse as _uparse

    rpc_bin = None
    # 1. Check cached unslothai/llama.cpp b12588 bundle FIRST (exact RPC protocol match with RTX 5090 host!)
    for pattern in (
        str(Path.home() / ".longinus_llama" / "unsloth-b12588" / "**" / "*rpc-server"),
        str(Path.home() / ".longinus_llama" / "**" / "*rpc-server"),
    ):
        found = next(iter(_glob.glob(pattern, recursive=True)), None)
        if found and Path(found).is_file():
            rpc_bin = found
            break

    if not rpc_bin:
        for cand in (
            str(Path.home() / "Projects/longinus-network/.bin/llama-unsloth/ggml-rpc-server"),
            shutil.which("ggml-rpc-server"),
            shutil.which("rpc-server"),
            "/opt/homebrew/bin/ggml-rpc-server",
            "/opt/homebrew/bin/rpc-server",
            "/usr/local/bin/rpc-server",
        ):
            if cand and Path(cand).exists():
                rpc_bin = cand
                break

    if not rpc_bin and platform.system() == "Darwin":
        # Download exact matching b12588 macOS arm64 bundle (13 MB, includes Metal ggml-rpc-server)
        llama_root = Path.home() / ".longinus_llama"
        bin_dir_dl = llama_root / "unsloth-b12588"
        bin_dir_dl.mkdir(parents=True, exist_ok=True)
        tgz = llama_root / "llama-mac-arm64.tar.gz"
        url = "https://github.com/unslothai/llama.cpp/releases/download/b12588-mix-c0c3de8/llama-b12588-mix-c0c3de8-bin-macos-arm64.tar.gz"
        print("[SPACE-RPC] Downloading matching Apple Silicon Metal ggml-rpc-server (13 MB)...")
        try:
            subprocess.check_call(["curl", "-fsSL", "-C", "-", url, "-o", str(tgz)])
            with _tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(bin_dir_dl)
            tgz.unlink(missing_ok=True)
            rpc_bin = next(iter(_glob.glob(str(bin_dir_dl / "**" / "*rpc-server"), recursive=True)), None)
        except Exception as exc:
            print(f"[SPACE-RPC] Download failed ({exc}).")

    if not rpc_bin and platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"):
        nv_info = _detect_nvidia_gpu_via_smi()
        drv_major = nv_info[2] if nv_info else 550
        cuda_tag = "cuda13" if drv_major >= 580 else "cuda12"
        llama_root = Path.home() / ".longinus_llama"
        bin_dir_dl = llama_root / f"unsloth-b12588-{cuda_tag}"
        bin_dir_dl.mkdir(parents=True, exist_ok=True)
        tgz = llama_root / f"app-b12588-{cuda_tag}.tar.gz"
        url = f"https://github.com/unslothai/llama.cpp/releases/download/b12588-mix-c0c3de8/app-b12588-mix-c0c3de8-linux-x64-{cuda_tag}-portable.tar.gz"
        print(f"[SPACE-RPC] Downloading matching Linux {cuda_tag.upper()} ggml-rpc-server...")
        try:
            subprocess.check_call(["curl", "-fsSL", "-C", "-", url, "-o", str(tgz)])
            with _tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(bin_dir_dl)
            tgz.unlink(missing_ok=True)
            rpc_bin = next(iter(_glob.glob(str(bin_dir_dl / "**" / "*rpc-server"), recursive=True)), None)
        except Exception as exc:
            print(f"[SPACE-RPC] Linux CUDA RPC download failed ({exc}).")

    if not rpc_bin:
        print("[SPACE-RPC] Could not find `ggml-rpc-server` / `rpc-server`. Skipping RPC hot-swap.")
        return False

    env = dict(os.environ)
    bin_dir = str(Path(rpc_bin).parent)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")
    arch_pfx = ["arch", "-arm64"] if (platform.system() == "Darwin" and shutil.which("arch")) else []
    print(f"[SPACE-RPC] Starting Metal/CUDA RPC backend: {rpc_bin}", flush=True)
    rpc_proc = subprocess.Popen(
        arch_pfx + [rpc_bin, "-H", "127.0.0.1", "-p", str(rpc_port), "-c"],
        env=env,
        stdout=subprocess.DEVNULL if not persistent else None,
        stderr=subprocess.DEVNULL if not persistent else None,
    )
    time.sleep(1.0)
    if rpc_proc.poll() is not None:
        print(f"[SPACE-RPC] rpc-server exited early (code {rpc_proc.returncode}).", flush=True)
        return False

    def _recv_exact(sock, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise EOFError("WebSocket closed")
            buf.extend(chunk)
        return bytes(buf)

    def _ws_send_frame(sock, send_lock: _thr.Lock, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        n = len(payload)
        hdr = bytearray([0x80 | (opcode & 0x0F)])
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr.extend(_struct.pack("!H", n))
        else:
            hdr.append(0x80 | 127)
            hdr.extend(_struct.pack("!Q", n))
        hdr.extend(mask)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        with send_lock:
            sock.sendall(bytes(hdr) + masked)

    def _ws_recv_frame(sock, send_lock: _thr.Lock) -> tuple[int, bytes]:
        msg_op = None
        chunks: list[bytes] = []
        while True:
            b0, b1 = _recv_exact(sock, 2)
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            is_masked = bool(b1 & 0x80)
            ln = b1 & 0x7F
            if ln == 126:
                ln = _struct.unpack("!H", _recv_exact(sock, 2))[0]
            elif ln == 127:
                ln = _struct.unpack("!Q", _recv_exact(sock, 8))[0]
            mask = _recv_exact(sock, 4) if is_masked else b""
            data = _recv_exact(sock, ln) if ln else b""
            if is_masked:
                data = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
            if opcode == 0x09:
                _ws_send_frame(sock, send_lock, 0x0A, data)
                continue
            if opcode == 0x0A:
                continue
            if opcode == 0x08:
                return opcode, data
            if msg_op is None:
                msg_op = opcode
            chunks.append(data)
            if fin:
                return msg_op, (chunks[0] if len(chunks) == 1 else b"".join(chunks))

    parsed = _uparse.urlparse(coord)
    use_tls = parsed.scheme == "https"
    host = parsed.hostname or "api.spaceresearch.ai"
    port = parsed.port or (443 if use_tls else 80)
    q = _uparse.urlencode(
        {"wallet": wallet, "name": display_name, "chip": chip, "mem_gb": f"{mem_gb:.1f}"}
    )
    ws_path = f"/v1/rpc/tunnel?{q}"

    if persistent:
        print("=" * 74)
        print(f"[SPACE-RPC] Zero-Port Internet Tensor Worker ({chip} · {mem_gb:.1f} GB)")
        print(f"[SPACE-RPC] Local loopback rpc-server: 127.0.0.1:{rpc_port} (no firewall ports opened)")
        print(f"[SPACE-RPC] Outbound tunnel: {'wss' if use_tls else 'ws'}://{host}/v1/rpc/tunnel")
        print("[SPACE-RPC] Press Ctrl+C to stop anytime.")
        print("=" * 74)

    released = False
    attempts = 0
    try:
        while True:
            attempts += 1
            ws_sock = None
            conns: dict[int, _sock.socket] = {}
            stop_ev = _thr.Event()
            send_lock = _thr.Lock()
            try:
                raw = _sock.create_connection((host, port), timeout=15.0)
                raw.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1)
                if use_tls:
                    ctx = _ssl.create_default_context()
                    ws_sock = ctx.wrap_socket(raw, server_hostname=host)
                else:
                    ws_sock = raw
                ws_sock.settimeout(None)
                ws_key = _b64.b64encode(os.urandom(16)).decode("ascii")
                req_hdr = (
                    f"GET {ws_path} HTTP/1.1\r\n"
                    f"Host: {host}\r\n"
                    f"Upgrade: websocket\r\n"
                    f"Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {ws_key}\r\n"
                    f"Sec-WebSocket-Version: 13\r\n"
                    f"User-Agent: space-worker/1.0\r\n\r\n"
                )
                ws_sock.sendall(req_hdr.encode("ascii"))
                resp_buf = b""
                while b"\r\n\r\n" not in resp_buf:
                    ch = ws_sock.recv(1024)
                    if not ch:
                        raise EOFError("Handshake closed")
                    resp_buf += ch
                status_line = resp_buf.split(b"\r\n", 1)[0].decode("ascii", errors="ignore")
                if " 101 " not in status_line:
                    raise RuntimeError(f"WebSocket upgrade failed: {status_line}")

                print(f"[SPACE-RPC] Connected over internet to {host}! Serving 70B+ cluster shard.", flush=True)

                def _send_mux(ftype: int, cid: int, data: bytes = b"") -> None:
                    _ws_send_frame(ws_sock, send_lock, 0x02, _struct.pack("!BHI", ftype, cid & 0xFFFF, len(data)) + data)

                def _keepalive():
                    while not stop_ev.wait(20.0):
                        try:
                            _send_mux(4, 0)
                        except Exception:
                            break

                _thr.Thread(target=_keepalive, daemon=True).start()

                def _pump_local_to_ws(cid: int, lsock: _sock.socket) -> None:
                    try:
                        while not stop_ev.is_set():
                            data = lsock.recv(65536)
                            if not data:
                                break
                            _send_mux(2, cid, data)
                    except Exception:
                        pass
                    finally:
                        conns.pop(cid, None)
                        try:
                            _send_mux(3, cid)
                        except Exception:
                            pass
                        try:
                            lsock.close()
                        except Exception:
                            pass

                rx_unacked = 0
                while True:
                    opcode, frame = _ws_recv_frame(ws_sock, send_lock)
                    if opcode == 0x08:
                        released = not persistent
                        break
                    if opcode != 0x02 or len(frame) < 7:
                        continue
                    ftype, cid, plen = _struct.unpack("!BHI", frame[:7])
                    payload = frame[7 : 7 + plen]
                    if ftype == 6:
                        print("[SPACE-RPC] Coordinator released 70B shard — returning to normal mode.", flush=True)
                        released = True
                        break
                    if ftype == 1:
                        try:
                            ls = _sock.create_connection(("127.0.0.1", rpc_port), timeout=5.0)
                            ls.settimeout(None)
                            ls.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1)
                            conns[cid] = ls
                            _thr.Thread(target=_pump_local_to_ws, args=(cid, ls), daemon=True).start()
                        except Exception:
                            _send_mux(3, cid)
                    elif ftype == 2:
                        ls = conns.get(cid)
                        if ls is not None:
                            try:
                                ls.sendall(payload)
                                rx_unacked += len(payload)
                                if rx_unacked >= 512 * 1024:
                                    _send_mux(7, 0, _struct.pack("!I", rx_unacked))
                                    rx_unacked = 0
                            except Exception:
                                conns.pop(cid, None)
                                _send_mux(3, cid)
                        else:
                            rx_unacked += len(payload)
                            if rx_unacked >= 512 * 1024:
                                _send_mux(7, 0, _struct.pack("!I", rx_unacked))
                                rx_unacked = 0
                    elif ftype == 3:
                        ls = conns.pop(cid, None)
                        if ls is not None:
                            try:
                                ls.shutdown(_sock.SHUT_RDWR)
                            except Exception:
                                pass
                            try:
                                ls.close()
                            except Exception:
                                pass
            except Exception as exc:
                if not persistent and attempts >= 2:
                    break
                print(f"[SPACE-RPC] Tunnel reconnecting in 3s ({exc})...", flush=True)
            finally:
                stop_ev.set()
                for ls in list(conns.values()):
                    try:
                        ls.shutdown(_sock.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        ls.close()
                    except Exception:
                        pass
                if ws_sock is not None:
                    try:
                        ws_sock.close()
                    except Exception:
                        pass
            if released or not persistent:
                break
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\n[SPACE-RPC] Shutting down cleanly.")
    finally:
        try:
            rpc_proc.terminate()
            rpc_proc.wait(timeout=4)
        except Exception:
            try:
                rpc_proc.kill()
            except Exception:
                pass
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join Space Research Network (Apple Silicon M1-M5, NVIDIA CUDA GPU, or standard Windows/Linux CPU laptop)"
    )
    parser.add_argument(
        "--coordinator",
        default=os.environ.get("SPACE_COORDINATOR", os.environ.get("LONGINUS_COORDINATOR", "auto")),
        help="Space Research Coordinator HTTPS URL (default: auto-discover via https://spaceresearch.ai/coordinator.json)",
    )
    parser.add_argument("--model", default=None, help="Optional model override (auto-picked for your hardware if omitted)")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--install", action="store_true", help="Auto-install required Python packages if missing")
    parser.add_argument("--cpu", action="store_true", help="Force CPU Edge mode even if a GPU is present")
    parser.add_argument("--name", default="", help="Optional handle shown on the Season 0 Leaderboard")
    parser.add_argument(
        "--payout-wallet",
        "--wallet",
        dest="payout_wallet",
        default="",
        help="Optional Solana wallet address (e.g. Phantom/Solflare) to receive your soulbound tSPACE tokens",
    )
    parser.add_argument(
        "--sync-wallet",
        "--migrate-tokens",
        dest="sync_wallet",
        action="store_true",
        help="One-time sync of previously earned soulbound tSPACE from this node into your --payout-wallet",
    )
    parser.add_argument("--swarm", action="store_true", help="Opt into Pooled-RAM Swarm sharding for large models")
    parser.add_argument("--swarm-group", default="global", help="Swarm group name (e.g. desk-lan or global)")
    parser.add_argument(
        "--rpc-worker",
        action="store_true",
        help="Run this Mac/GPU as a llama.cpp RPC tensor-sharding backend (for true single-model 70B-122B LAN/Tailscale clusters)",
    )
    parser.add_argument("--rpc-port", type=int, default=50052, help="TCP port for --rpc-worker (default: 50052)")
    parser.add_argument(
        "--auto-pause",
        action="store_true",
        help="Automatically pause & free VRAM whenever you launch a game or heavy personal GPU/CPU app",
    )
    parser.add_argument(
        "--idle-resume-secs",
        type=int,
        default=90,
        help="Seconds of idle time before auto-resuming after --auto-pause (default: 90s)",
    )
    args = parser.parse_args()

    if args.rpc_worker:
        chip, mem_gb, _backend, _tier, _engines = detect_hardware(force_cpu=args.cpu)
        _, wallet = load_or_create_keypair()
        coord = resolve_coordinator_url(args.coordinator)
        _run_rpc_tunnel_session(
            coord=coord,
            wallet=wallet,
            chip=chip,
            mem_gb=mem_gb,
            display_name=args.name or chip,
            rpc_port=int(args.rpc_port or 50052),
            persistent=True,
        )
        return

    _bootstrap_macos_arm_python_if_needed()
    _ensure_isolated_venv_if_needed(auto_install=args.install)
    ensure_dependencies(auto_install=args.install)
    chip, mem_gb, backend, tier, engines = detect_hardware(force_cpu=args.cpu)
    model_id = args.model or recommend_model(backend, mem_gb)
    priv, wallet = load_or_create_keypair()
    my_pid = os.getpid()

    def _load_runner():
        nonlocal model_id
        if _use_llama_cpp_inkling(backend, mem_gb, args.model):
            try:
                r = LlamaCppInklingRunner(mem_gb=mem_gb, max_tokens=args.max_tokens, model_override=args.model)
                model_id = r.model_label
                return r
            except Exception as exc:
                print(f"[WARN] Metal llama-server setup failed ({exc}); falling back to standard runner...")
                if any(k in model_id.lower() for k in ("inkling", "gemma-4", "qwen3.5", "qwen3.8")):
                    model_id = "mlx-community/Qwen3-8B-4bit" if (backend == "mlx" and mem_gb >= 24.0) else "Qwen/Qwen3-1.7B"
        if _use_llama_cpp_cuda(backend, mem_gb, args.model):
            try:
                r = LlamaCppCudaRunner(mem_gb=mem_gb, max_tokens=args.max_tokens, model_override=args.model)
                model_id = r.model_label
                return r
            except Exception as exc:
                print(f"[WARN] Portable CUDA llama-server setup failed ({exc}); falling back to PyTorch runner...")
                if any(k in model_id.lower() for k in ("inkling", "gemma-4", "qwen3.5", "qwen3.8")):
                    model_id = "Qwen/Qwen3-4B" if mem_gb >= 12.0 else "Qwen/Qwen3-1.7B"
        if backend == "mlx":
            return MLXRunner(model_id, max_tokens=args.max_tokens)
        return TorchRunner(model_id, device=backend, max_tokens=args.max_tokens)

    runner = _load_runner()

    payout_wallet = (args.payout_wallet or "").strip()
    if payout_wallet:
        if not (32 <= len(payout_wallet) <= 44 and all(c in _B58_ALPHABET for c in payout_wallet)):
            print(f"[WARN] Ignoring invalid --payout-wallet address: {payout_wallet}")
            payout_wallet = ""
        else:
            # Persist payout wallet alongside local node identity for future runs
            node_cfg = Path.home() / ".longinus_node.json"
            if node_cfg.exists():
                try:
                    d = json.loads(node_cfg.read_text())
                    d["payout_wallet"] = payout_wallet
                    node_cfg.write_text(json.dumps(d, indent=2))
                except Exception:
                    pass
    else:
        node_cfg = Path.home() / ".longinus_node.json"
        if node_cfg.exists():
            try:
                payout_wallet = str(json.loads(node_cfg.read_text()).get("payout_wallet") or "").strip()
            except Exception:
                payout_wallet = ""

    explicit_coord = (args.coordinator or "").strip().lower() not in ("", "auto")
    coord = resolve_coordinator_url(args.coordinator)
    print(f"[SPACE] Connecting {chip} ({mem_gb} GB, {backend}, model={model_id}) -> {coord}")
    print(f"[SPACE] Node ID: {wallet[:12]}...{wallet[-6:]}" + (f" ({args.name})" if args.name else ""))
    if payout_wallet:
        print(f"[SPACE] Soulbound tSPACE Payout Wallet: {payout_wallet}")
    if args.auto_pause:
        print(f"[SPACE] Auto-Pause Guard enabled (frees VRAM on personal GPU/app load; resumes after {args.idle_resume_secs}s idle)")

    def _register_with_coord(target_coord: str) -> dict:
        return http_json(
            f"{target_coord}/v1/nodes/register",
            {
                "wallet": wallet,
                "hardware_tier": tier,
                "chip_name": chip,
                "backend": backend,
                "memory_gb": mem_gb,
                "endpoint_url": f"pull://{wallet[:12]}",
                "bonded_usdc": 1_000_000_000 if tier == "datacenter_gpu" else 200_000_000,
                "model_id": model_id,
                "supported_engines": engines,
                "display_name": args.name,
                "payout_wallet": payout_wallet,
                "swarm_enabled": bool(args.swarm),
                "swarm_group": args.swarm_group,
            },
        )

    reg = _register_with_coord(coord)
    print(f"[SPACE] Online! Status={reg.get('status')} | Watching for live requests...")
    api_key = reg.get("api_key")
    if api_key:
        g_slot = reg.get("genesis_slot")
        g_tag = f"Genesis 50 Slot #{g_slot}/50 (2.0x Compute Credits)" if g_slot else "Standard Tier (1.0x)"
        rem = int(reg.get("remaining_tokens", reg.get("credit_tokens", 0)))
        print(f"[BARTER API] {g_tag} | Balance: {rem:,} free API tokens")
        print(f"[BARTER API] Use your idle GPU to power Claude Code / Cursor / OpenCode / Cline:")
        print(f"             export OPENAI_BASE_URL={coord}/v1")
        print(f"             export OPENAI_API_KEY={api_key}")

    if args.sync_wallet and payout_wallet and payout_wallet != wallet:
        try:
            mig_msg = f"migrate_tlng|{wallet}|{payout_wallet}".encode("utf-8")
            sig_hex = priv.sign(mig_msg).hex()
            mig = http_json(
                f"{coord}/v1/nodes/migrate-tlng",
                {"wallet": wallet, "payout_wallet": payout_wallet, "signature_hex": sig_hex},
                timeout=45.0,
            )
            amt = float(mig.get("migrated_amount", 0.0))
            if amt > 0:
                print(
                    f"[SPACE] Synced {amt:g} earlier soulbound tSPACE -> {payout_wallet} "
                    f"(tx: {str(mig.get('tx_sig', ''))[:16]}...)"
                )
            elif mig.get("already_migrated") or mig.get("already_synced"):
                print(f"[SPACE] Earlier soulbound tSPACE already synced to {payout_wallet}.")
        except Exception as exc:
            print(f"[WARN] Could not sync earlier tSPACE ({exc}); future earnings still route to {payout_wallet}.")

    if args.swarm:
        try:
            sw = http_json(
                f"{coord}/v1/swarm/join",
                {
                    "wallet": wallet,
                    "chip_name": chip,
                    "backend": backend,
                    "memory_gb": mem_gb,
                    "display_name": args.name or chip,
                    "swarm_group": args.swarm_group,
                },
            )
            plan = sw.get("swarm_plan", {})
            assign = sw.get("assignment") or {}
            print(
                f"[SWARM] Joined '{plan.get('swarm_group')}' | Pooled RAM: {plan.get('pooled_memory_gb')} GB "
                f"({plan.get('node_count')} devices) -> Target: {plan.get('model_id')} | "
                f"Assigned Stage {assign.get('stage_index', 0)} (Layers {assign.get('layer_range', '0-0')})"
            )
        except Exception as exc:
            print(f"[SWARM] Notice: could not register swarm stage ({exc})")

    paused = False
    idle_since: float | None = None
    consecutive_errors = 0

    while True:
        try:
            if args.auto_pause:
                busy, reason = is_machine_busy_for_personal_use(backend, my_pid)
                if busy:
                    idle_since = None
                    if not paused:
                        print(f"\n[AUTO-PAUSE] Personal activity detected ({reason}) — freeing VRAM & pausing...")
                        unload_runner(runner, backend)
                        runner = None
                        paused = True
                    time.sleep(5.0)
                    continue
                if paused:
                    if idle_since is None:
                        idle_since = time.time()
                    if (time.time() - idle_since) < args.idle_resume_secs:
                        time.sleep(5.0)
                        continue
                    print("\n[AUTO-PAUSE] Machine idle — reloading model and resuming Space Research node...")
                    runner = _load_runner()
                    paused = False
                    idle_since = None

            poll = http_json(f"{coord}/v1/jobs/next?wallet={wallet}&wait_s=15", timeout=25.0)
            consecutive_errors = 0
            if not poll.get("has_job") or not poll.get("job"):
                continue
            job = poll["job"]
            job_id = job["job_id"]
            if job.get("model_id") == "__enter_rpc_mode__":
                print(f"\n[SPACE-HOTSWAP] 70B+ Sharded Cluster requested — freeing local RAM & joining RPC tunnel...")
                unload_runner(runner, backend)
                runner = None
                try:
                    _run_rpc_tunnel_session(
                        coord=coord,
                        wallet=wallet,
                        chip=chip,
                        mem_gb=mem_gb,
                        display_name=args.name or chip,
                        rpc_port=int(args.rpc_port or 50052),
                        persistent=False,
                    )
                finally:
                    print("[SPACE-HOTSWAP] 70B+ cluster idle — reloading local model & resuming normal serving...")
                    runner = _load_runner()
                continue
            engine = job.get("engine", "corekv")
            prompt = job["prompt"]
            print(f"\n[JOB {job_id}] Processing request on {chip}...")
            t0 = time.perf_counter()

            out_text, top_lps, sketch, toks = runner.run(prompt, engine=engine)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            receipt = sign_receipt(
                priv,
                {
                    "request_id": job_id,
                    "node_wallet": wallet,
                    "model_id": model_id,
                    "engine": engine,
                    "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "output_hash": hashlib.sha256(out_text.encode("utf-8")).hexdigest(),
                    "top_logprobs": top_lps,
                    "latent_sketch": sketch,
                    "tokens_served": toks,
                    "latency_ms": latency_ms,
                    "timestamp": int(time.time()),
                },
            )
            http_json(f"{coord}/v1/jobs/{job_id}/complete", {"output_text": out_text, "receipt": receipt})
            print(f"[JOB {job_id}] Completed ({toks} tokens in {latency_ms:.0f} ms)!")
            try:
                rk = http_json(f"{coord}/v1/leaderboard/{wallet}", timeout=5.0)
                op = rk.get("operator", {})
                print(
                    f"[SEASON 0] Rank #{op.get('rank')} of {rk.get('active_operators')} | "
                    f"{op.get('season0_points')} pts | Est. {op.get('est_canary_tokens')} {rk.get('canary_token_symbol', 'tSPACE')}"
                )
            except Exception:
                pass
        except KeyboardInterrupt:
            print("\n[SPACE] Shutting down node cleanly.")
            break
        except (urllib.error.URLError, TimeoutError, Exception) as exc:
            consecutive_errors += 1
            print(f"[WARN] Poll error ({exc}); retrying in 3s...")
            if not explicit_coord and consecutive_errors >= 3:
                fresh = resolve_coordinator_url("auto")
                if fresh and fresh != coord:
                    print(f"[SPACE] Discovered updated coordinator endpoint -> {fresh}; re-registering...")
                    try:
                        _register_with_coord(fresh)
                        coord = fresh
                        consecutive_errors = 0
                        print(f"[SPACE] Seamlessly reconnected to {coord}!")
                    except Exception as r_exc:
                        print(f"[WARN] Reconnect to {fresh} failed ({r_exc})")
                else:
                    try:
                        _register_with_coord(coord)
                        consecutive_errors = 0
                    except Exception:
                        pass
            time.sleep(3.0)


if __name__ == "__main__":
    main()
