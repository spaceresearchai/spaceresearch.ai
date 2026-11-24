"""Modular Provider Backend Adapters for Space AI Worker Nodes.

Provides an abstract `ProviderAdapter` interface plus 5 concrete adapters:
  - `native`    : Space portable CUDA / Apple Silicon MLX worker (`scripts/join.py`)
                  — 20x Genesis High-Density Tier
  - `ollama`    : Local Ollama daemon (`http://127.0.0.1:11434`)
                  — Standard Community Adapter Tier (1.0x)
  - `llama-cpp` : Local llama.cpp `llama-server` (`http://127.0.0.1:8081`)
                  — Standard Community Adapter Tier (1.0x)
  - `vllm`      : Local vLLM OpenAI-compatible server (`http://127.0.0.1:8000`)
                  — Standard Community Adapter Tier (1.0x)
  - `mlx`       : Local `mlx_lm.server` (`http://127.0.0.1:8080`)
                  — Standard Community Adapter Tier (1.0x)

Also includes `AdapterBridgeWorker`, a pure-stdlib worker loop that bridges any
external adapter into the Space coordinator pull-job protocol (`/v1/nodes/register`,
`/v1/jobs/next`, `/v1/jobs/{job_id}/complete`).
"""

from __future__ import annotations

import abc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cli import get_repo_root, load_node_identity, save_space_config


@dataclass
class AdapterProbeResult:
    name: str
    display_name: str
    status: str  # "online", "available", "unavailable"
    detected: bool
    endpoint: str
    models: list[str] = field(default_factory=list)
    hardware: str = ""
    binary_found: bool = False
    latency_ms: float | None = None
    multiplier: float = 1.0
    tier: str = "standard_community"
    tier_label: str = "Standard Community Adapter (1x)"
    recommended: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _http_json_request(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 3.0,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], float]:
    req_headers = {"User-Agent": "space-cli-adapter/0.2"}
    if headers:
        req_headers.update(headers)
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        parsed = json.loads(raw) if raw.strip() else {}
        return int(resp.status), parsed if isinstance(parsed, dict) else {"data": parsed}, lat_ms


def _detect_local_hardware() -> tuple[str, float, str]:
    """Lightweight stdlib hardware detection (chip name, memory GB, backend kind)."""
    sys_name = platform.system()
    machine = platform.machine()

    # Check Apple Silicon first
    if sys_name == "Darwin" and machine == "arm64":
        chip = "Apple Silicon"
        mem_gb = 16.0
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip() or chip
        except Exception:
            pass
        try:
            mem_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip()
            )
            mem_gb = round(mem_bytes / (1024**3), 1)
        except Exception:
            pass
        return chip, mem_gb, "mlx"

    # Check NVIDIA GPU via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if out.returncode == 0 and out.stdout.strip():
                line = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                gpu_name = parts[0] if parts else "NVIDIA GPU"
                vram_gb = round(float(parts[1]) / 1024.0, 1) if len(parts) > 1 else 8.0
                return gpu_name, vram_gb, "cuda"
        except Exception:
            pass

    # Fallback CPU detection
    cpu_name = platform.processor() or f"{sys_name} {machine} CPU"
    mem_gb = 8.0
    if sys_name == "Linux" and Path("/proc/meminfo").exists():
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    mem_gb = round(kb / (1024 * 1024), 1)
                    break
        except Exception:
            pass
    return cpu_name, mem_gb, "cpu"


class ProviderAdapter(abc.ABC):
    """Abstract base class for Space AI provider backend adapters."""

    name: str = ""
    display_name: str = ""
    default_endpoint: str = ""
    multiplier: float = 1.0
    tier: str = "standard_community"
    tier_label: str = "Standard Community Adapter (1x)"
    is_native: bool = False

    def __init__(self, endpoint: str | None = None, repo_root: Path | None = None) -> None:
        self.endpoint = (endpoint or self.default_endpoint).rstrip("/")
        self.repo_root = repo_root or get_repo_root()

    @abc.abstractmethod
    def probe(self, timeout: float = 1.5) -> AdapterProbeResult:
        """Probe whether this backend is running or installed locally."""

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Run a text completion against this backend adapter."""

    def build_worker_command(
        self,
        coordinator: str,
        name: str = "",
        auto_pause: bool = False,
        model: str | None = None,
        payout_wallet: str = "",
        extra_args: list[str] | None = None,
    ) -> list[str] | None:
        """Return subprocess command for native workers, or None for in-process bridge adapters."""
        return None


class NativeAdapter(ProviderAdapter):
    """Space Research native portable CUDA / Apple Silicon MLX worker (`scripts/join.py`)."""

    name = "native"
    display_name = "Space Native (Portable CUDA / Apple MLX)"
    default_endpoint = "local://scripts/join.py"
    multiplier = 20.0
    tier = "genesis_high_density"
    tier_label = "20x Genesis High-Density Tier"
    is_native = True

    @property
    def join_script_path(self) -> Path:
        return self.repo_root / "scripts" / "join.py"

    def probe(self, timeout: float = 1.5) -> AdapterProbeResult:
        t0 = time.perf_counter()
        chip, mem_gb, backend = _detect_local_hardware()
        script_exists = self.join_script_path.exists()
        mlx_installed = backend == "mlx" and importlib.util.find_spec("mlx_lm") is not None
        cuda_ready = backend == "cuda"
        detected = script_exists
        status = "available" if script_exists else "unavailable"
        detail_parts = [f"hw={chip} ({mem_gb:.1f} GB, {backend})"]
        if cuda_ready:
            detail_parts.append("portable CUDA ready")
        elif mlx_installed:
            detail_parts.append("mlx-lm installed")
        elif backend == "mlx":
            detail_parts.append("Apple Silicon detected (join.py --install ready)")
        if not script_exists:
            detail_parts.append("scripts/join.py not found")
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return AdapterProbeResult(
            name=self.name,
            display_name=self.display_name,
            status=status,
            detected=detected,
            endpoint=str(self.join_script_path),
            models=["auto-selected-for-hardware"],
            hardware=f"{chip} ({mem_gb:.1f} GB, {backend})",
            binary_found=script_exists,
            latency_ms=lat_ms,
            multiplier=self.multiplier,
            tier=self.tier,
            tier_label=self.tier_label,
            recommended=detected,
            detail="; ".join(detail_parts),
        )

    def build_worker_command(
        self,
        coordinator: str,
        name: str = "",
        auto_pause: bool = False,
        model: str | None = None,
        payout_wallet: str = "",
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = [sys.executable, str(self.join_script_path)]
        if coordinator:
            cmd.extend(["--coordinator", coordinator])
        if name:
            cmd.extend(["--name", name])
        if auto_pause:
            cmd.append("--auto-pause")
        if model:
            cmd.extend(["--model", model])
        if payout_wallet:
            cmd.extend(["--payout-wallet", payout_wallet])
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "NativeAdapter delegates directly to `scripts/join.py` worker process; "
            "use `build_worker_command()` or `space up --adapter native`."
        )


class OpenAICompatibleAdapter(ProviderAdapter):
    """Base class for HTTP OpenAI-compatible local inference backends."""

    binary_names: tuple[str, ...] = ()
    health_paths: tuple[str, ...] = ("/v1/models", "/health")

    def _binary_available(self) -> bool:
        for b in self.binary_names:
            if shutil.which(b):
                return True
        return False

    def _extract_models(self, payload: dict[str, Any]) -> list[str]:
        models: list[str] = []
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
                elif isinstance(item, str):
                    models.append(item)
        raw_models = payload.get("models")
        if isinstance(raw_models, list):
            for item in raw_models:
                if isinstance(item, dict):
                    mid = item.get("name") or item.get("model") or item.get("id")
                    if mid:
                        models.append(str(mid))
                elif isinstance(item, str):
                    models.append(item)
        # Deduplicate while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for m in models:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def probe(self, timeout: float = 1.5) -> AdapterProbeResult:
        bin_found = self._binary_available()
        models: list[str] = []
        online = False
        best_lat: float | None = None
        last_err = ""

        for path in self.health_paths:
            try:
                status_code, data, lat_ms = _http_json_request(f"{self.endpoint}{path}", timeout=timeout)
                if 200 <= status_code < 300:
                    online = True
                    best_lat = lat_ms if best_lat is None else min(best_lat, lat_ms)
                    found = self._extract_models(data)
                    if found:
                        models = found
                        break
            except Exception as exc:
                last_err = str(exc)

        if online:
            status = "online"
            detail = f"reachable at {self.endpoint} ({len(models)} model(s))"
        elif bin_found:
            status = "available"
            detail = f"binary installed, endpoint {self.endpoint} not listening"
        else:
            status = "unavailable"
            detail = f"not listening at {self.endpoint}" + (f" ({last_err[:60]})" if last_err else "")

        return AdapterProbeResult(
            name=self.name,
            display_name=self.display_name,
            status=status,
            detected=online or bin_found,
            endpoint=self.endpoint,
            models=models,
            binary_found=bin_found,
            latency_ms=best_lat,
            multiplier=self.multiplier,
            tier=self.tier,
            tier_label=self.tier_label,
            recommended=False,
            detail=detail,
        )

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        chosen_model = model
        if not chosen_model:
            pr = self.probe(timeout=2.0)
            chosen_model = pr.models[0] if pr.models else "default"

        payload = {
            "model": chosen_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        status_code, data, lat_ms = _http_json_request(
            f"{self.endpoint}/v1/chat/completions",
            payload=payload,
            timeout=timeout,
        )
        if status_code >= 400:
            raise RuntimeError(f"{self.name} adapter HTTP {status_code}: {data}")

        choices = data.get("choices") or []
        text = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            text = str(msg.get("content") or choices[0].get("text") or "")
        usage = data.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or usage.get("total_tokens") or max(1, len(text.split())))
        return {
            "text": text,
            "tokens": tokens,
            "latency_ms": lat_ms,
            "model": str(data.get("model") or chosen_model),
        }


class OllamaAdapter(OpenAICompatibleAdapter):
    name = "ollama"
    display_name = "Ollama Local Server"
    default_endpoint = "http://127.0.0.1:11434"
    binary_names = ("ollama",)
    health_paths = ("/api/tags", "/v1/models")

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        try:
            return super().complete(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception:
            # Fallback to Ollama native /api/generate if /v1/chat/completions is unavailable
            chosen_model = model
            if not chosen_model:
                pr = self.probe(timeout=2.0)
                chosen_model = pr.models[0] if pr.models else "llama3.2"
            payload = {
                "model": chosen_model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": int(max_tokens), "temperature": float(temperature)},
            }
            _, data, lat_ms = _http_json_request(f"{self.endpoint}/api/generate", payload=payload, timeout=timeout)
            text = str(data.get("response") or "")
            tokens = int(data.get("eval_count") or max(1, len(text.split())))
            return {"text": text, "tokens": tokens, "latency_ms": lat_ms, "model": chosen_model}


class LlamaCppAdapter(OpenAICompatibleAdapter):
    name = "llama-cpp"
    display_name = "llama.cpp Server"
    default_endpoint = "http://127.0.0.1:8081"
    binary_names = ("llama-server", "llama-cli")
    health_paths = ("/v1/models", "/health")


class VLLMAdapter(OpenAICompatibleAdapter):
    name = "vllm"
    display_name = "vLLM OpenAI Server"
    default_endpoint = "http://127.0.0.1:8000"
    binary_names = ("vllm",)
    health_paths = ("/v1/models", "/health")


class MLXAdapter(OpenAICompatibleAdapter):
    name = "mlx"
    display_name = "Apple MLX LM Server"
    default_endpoint = "http://127.0.0.1:8080"
    binary_names = ("mlx_lm.server", "mlx_lm")
    health_paths = ("/v1/models", "/health")

    def _binary_available(self) -> bool:
        return super()._binary_available() or importlib.util.find_spec("mlx_lm") is not None


ADAPTER_CLASSES: dict[str, type[ProviderAdapter]] = {
    "native": NativeAdapter,
    "ollama": OllamaAdapter,
    "llama-cpp": LlamaCppAdapter,
    "vllm": VLLMAdapter,
    "mlx": MLXAdapter,
}


def get_adapter(
    name: str,
    endpoint: str | None = None,
    repo_root: Path | None = None,
) -> ProviderAdapter:
    key = (name or "native").strip().lower().replace("_", "-")
    if key not in ADAPTER_CLASSES:
        raise ValueError(f"Unknown adapter '{name}'. Valid adapters: {', '.join(ADAPTER_CLASSES)}")
    return ADAPTER_CLASSES[key](endpoint=endpoint, repo_root=repo_root)


def probe_all_adapters(
    timeout: float = 1.5,
    endpoints: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> list[AdapterProbeResult]:
    """Probe all supported local provider adapters and return structured results."""
    overrides = endpoints or {}
    results: list[AdapterProbeResult] = []
    for name in ("native", "ollama", "llama-cpp", "vllm", "mlx"):
        adapter = get_adapter(name, endpoint=overrides.get(name), repo_root=repo_root)
        results.append(adapter.probe(timeout=timeout))
    return results


def format_adapters_table(results: list[AdapterProbeResult]) -> str:
    """Format adapter probe results as a clean CLI summary table."""
    lines = [
        "Space AI Provider Backend Adapters",
        "=" * 78,
        f"{'ADAPTER':<11} {'STATUS':<12} {'TIER / MULTIPLIER':<32} {'ENDPOINT / DETAIL'}",
        "-" * 78,
    ]
    for r in results:
        tag = f"{r.tier_label}"
        models_hint = f" models={','.join(r.models[:2])}" if r.models else ""
        rec = " [RECOMMENDED]" if r.recommended else ""
        lines.append(f"{r.name:<11} {r.status:<12} {tag:<32} {r.endpoint}{models_hint}{rec}")
    lines.append("-" * 78)
    lines.append(
        "Tip: `native` (Space portable CUDA / Apple MLX) qualifies for the 20x Genesis\n"
        "     high-density tier. External adapters (`ollama`, `llama-cpp`, `vllm`, `mlx`)\n"
        "     bridge existing local servers at the standard community adapter tier (1x)."
    )
    return "\n".join(lines)


class AdapterBridgeWorker:
    """Bridges an external `ProviderAdapter` (`ollama`/`llama-cpp`/`vllm`/`mlx`) to a Space Coordinator.

    Registers the adapter node, long-polls `/v1/jobs/next`, dispatches prompts to
    the local adapter endpoint, and submits signed/hashed completions to `/v1/jobs/{id}/complete`.
    """

    def __init__(
        self,
        adapter: ProviderAdapter,
        coordinator: str,
        display_name: str = "",
        model_override: str | None = None,
        payout_wallet: str = "",
        home_dir: Path | str | None = None,
        poll_wait_s: int = 5,
        retry_delay_s: float = 2.0,
    ) -> None:
        self.adapter = adapter
        self.coordinator = coordinator.rstrip("/")
        self.display_name = display_name or f"{adapter.name}-bridge"
        self.model_override = model_override
        self.payout_wallet = payout_wallet
        self.home_dir = Path(home_dir) if home_dir is not None else Path.home()
        self.poll_wait_s = max(1, int(poll_wait_s))
        self.retry_delay_s = max(0.05, float(retry_delay_s))

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self.wallet, self._priv = self._load_or_generate_identity()
        self.model_id = model_override or self._discover_model()
        self.state = "initialized"
        self.barter_key: str | None = None
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.last_error: str | None = None
        self.last_job_at: float | None = None
        self.registered = False

    def _load_or_generate_identity(self) -> tuple[str, Any]:
        node_cfg, _ = load_node_identity(self.home_dir)
        wallet = str(node_cfg.get("wallet") or "").strip()
        seed_hex = str(node_cfg.get("seed_hex") or "").strip()
        priv = None
        if seed_hex:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

                priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
            except Exception:
                priv = None
        if not wallet:
            # Deterministic local bridge ID when no Ed25519 node file exists yet
            wallet = "sp_adapter_" + hashlib.sha256(
                f"{platform.node()}:{self.adapter.name}:{uuid.getnode()}".encode("utf-8")
            ).hexdigest()[:32]
        return wallet, priv

    def _discover_model(self) -> str:
        probe = self.adapter.probe(timeout=1.5)
        if probe.models:
            return probe.models[0]
        return f"{self.adapter.name}-local"

    def _sign_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        body = {k: v for k, v in receipt.items() if k != "signature_hex"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if self._priv is not None:
            receipt["signature_hex"] = self._priv.sign(canonical).hex()
        else:
            receipt["signature_hex"] = hashlib.sha256(canonical).hexdigest()
        return receipt

    def register(self, timeout: float = 10.0) -> dict[str, Any]:
        chip, mem_gb, _ = _detect_local_hardware()
        payload = {
            "wallet": self.wallet,
            "hardware_tier": "community_adapter",
            "chip_name": f"{chip} ({self.adapter.display_name})",
            "backend": self.adapter.name,
            "memory_gb": mem_gb,
            "endpoint_url": f"pull://{self.adapter.name}-{self.wallet[:12]}",
            "bonded_usdc": 200_000_000,
            "model_id": self.model_id,
            "supported_engines": [self.adapter.name, "openai_compat"],
            "display_name": self.display_name,
            "payout_wallet": self.payout_wallet,
            "adapter_tier": self.adapter.tier,
            "adapter_multiplier": self.adapter.multiplier,
        }
        _, reg, _ = _http_json_request(f"{self.coordinator}/v1/nodes/register", payload=payload, timeout=timeout)
        with self._lock:
            self.registered = True
            self.state = "online"
            api_key = str(reg.get("api_key") or "").strip()
            if api_key:
                self.barter_key = api_key
                try:
                    save_space_config({"barter_key": api_key, "coordinator": self.coordinator}, home_dir=self.home_dir)
                except Exception:
                    pass
        return reg

    def poll_once(self, timeout: float = 15.0) -> bool:
        """Poll for one job from coordinator, execute via adapter, and submit completion."""
        url = f"{self.coordinator}/v1/jobs/next?wallet={self.wallet}&wait_s={self.poll_wait_s}"
        _, poll, _ = _http_json_request(url, timeout=timeout)
        if not poll.get("has_job") or not poll.get("job"):
            return False

        job = poll["job"]
        job_id = str(job["job_id"])
        prompt = str(job.get("prompt") or "")
        engine = str(job.get("engine") or self.adapter.name)
        req_model = job.get("model_id") or self.model_id

        with self._lock:
            self.state = "processing"
        try:
            result = self.adapter.complete(prompt=prompt, model=self.model_override or req_model)
            out_text = str(result.get("text") or "")
            tokens = int(result.get("tokens") or max(1, len(out_text.split())))
            lat_ms = float(result.get("latency_ms") or 1.0)
            receipt = self._sign_receipt(
                {
                    "request_id": job_id,
                    "node_wallet": self.wallet,
                    "model_id": str(result.get("model") or self.model_id),
                    "engine": engine,
                    "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "output_hash": hashlib.sha256(out_text.encode("utf-8")).hexdigest(),
                    "top_logprobs": [],
                    "latent_sketch": [],
                    "tokens_served": tokens,
                    "latency_ms": lat_ms,
                    "timestamp": int(time.time()),
                }
            )
            _http_json_request(
                f"{self.coordinator}/v1/jobs/{job_id}/complete",
                payload={"output_text": out_text, "receipt": receipt},
                timeout=15.0,
            )
            with self._lock:
                self.jobs_completed += 1
                self.last_job_at = time.time()
                self.last_error = None
                self.state = "online"
            return True
        except Exception as exc:
            with self._lock:
                self.jobs_failed += 1
                self.last_error = str(exc)
                self.state = "online"
            raise

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self.registered:
                    self.register()
                self.poll_once(timeout=float(self.poll_wait_s + 10))
            except Exception as exc:
                with self._lock:
                    self.last_error = str(exc)
                    if not self.registered:
                        self.state = "reconnecting"
                self._stop_event.wait(self.retry_delay_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self.state = "starting"
        self._thread = threading.Thread(target=self._run_loop, name=f"space-adapter-{self.adapter.name}", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        with self._lock:
            self.state = "stopped"

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "mode": "adapter_bridge",
                "adapter": self.adapter.name,
                "tier": self.adapter.tier,
                "multiplier": self.adapter.multiplier,
                "endpoint": self.adapter.endpoint,
                "coordinator": self.coordinator,
                "wallet": self.wallet,
                "model": self.model_id,
                "state": self.state,
                "registered": self.registered,
                "jobs_completed": self.jobs_completed,
                "jobs_failed": self.jobs_failed,
                "last_error": self.last_error,
            }


__all__ = [
    "AdapterProbeResult",
    "ProviderAdapter",
    "NativeAdapter",
    "OpenAICompatibleAdapter",
    "OllamaAdapter",
    "LlamaCppAdapter",
    "VLLMAdapter",
    "MLXAdapter",
    "ADAPTER_CLASSES",
    "get_adapter",
    "probe_all_adapters",
    "format_adapters_table",
    "AdapterBridgeWorker",
]
