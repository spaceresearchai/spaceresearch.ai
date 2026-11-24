"""Space AI CLI, Unified Local Buyer+Worker Proxy, Launchers & Provider Adapters.

Zero-dependency (Python stdlib) local control plane for Space Research Network:
- One-command coding agent launchers (`space claude`, `space codex`, `space opencode`)
- Unified local OpenAI + Anthropic buyer proxy (`127.0.0.1:8377`) with multi-upstream
  latency EMA tracking, automatic failover, and Space routing header injection
- Unified worker + proxy supervisor (`space up`)
- Modular provider backend adapters (`native`, `ollama`, `llama-cpp`, `vllm`, `mlx`)
- Agent-native USDC session budget helpers (`space session create`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__version__ = "0.3.0"

DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 8377
DEFAULT_COORDINATOR = "https://api.spaceresearch.ai"
FALLBACK_COORDINATOR = "https://api.longinusresearch.com"

VALID_ROUTES = ("fastest", "cheapest", "verified-local-only")
VALID_PRIVACY_TIERS = ("standard", "zero-log", "hardened-docker", "e2ee", "tee", "local-only")
STRICT_PRIVACY_TIERS = frozenset({"hardened-docker", "e2ee", "tee", "local-only"})
PLACEHOLDER_API_KEYS = frozenset(
    {
        "",
        "space-local-proxy",
        "space-barter-auto",
        "space-auto",
        "sk-space-local",
    }
)


def get_space_config_dir(home_dir: Path | str | None = None) -> Path:
    base = Path(home_dir) if home_dir is not None else Path.home()
    return base / ".space_ai"


def get_space_config_path(home_dir: Path | str | None = None) -> Path:
    return get_space_config_dir(home_dir) / "config.json"


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_space_config(home_dir: Path | str | None = None) -> dict[str, Any]:
    """Load persisted Space CLI configuration from `~/.space_ai/config.json`."""
    return _read_json_file(get_space_config_path(home_dir))


def save_space_config(updates: dict[str, Any], home_dir: Path | str | None = None) -> Path:
    """Merge and persist updates into `~/.space_ai/config.json` with restrictive permissions."""
    cfg_path = get_space_config_path(home_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    current = load_space_config(home_dir)
    for key, val in updates.items():
        if val is None:
            current.pop(key, None)
        else:
            current[key] = val
    cfg_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(cfg_path, 0o600)
    except OSError:
        pass
    return cfg_path


def load_node_identity(home_dir: Path | str | None = None) -> tuple[dict[str, Any], Path | None]:
    """Inspect local node config (`~/.longinus_node.json` and variants) without mutating it."""
    base = Path(home_dir) if home_dir is not None else Path.home()
    for name in (".longinus_node.json", ".longinus_mac_node.json", ".longinus_gpu_node.json"):
        cand = base / name
        if cand.exists():
            return _read_json_file(cand), cand
    return {}, None


def resolve_barter_key(
    explicit_key: str | None = None,
    home_dir: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """Resolve the active Space/Longinus compute-barter API key and its source.

    Search order:
      1. Explicit CLI/config argument
      2. Environment variables (`SPACE_BARTER_KEY`, `SPACE_API_KEY`, `LONGINUS_API_KEY`)
      3. `~/.space_ai/config.json` (`barter_key`, `api_key`, `openai_api_key`)
      4. `~/.longinus_node.json` (+ mac/gpu variants) (`barter_key`, `api_key`, `longinus_api_key`)
    """
    if explicit_key and explicit_key.strip() and explicit_key.strip() not in PLACEHOLDER_API_KEYS:
        return explicit_key.strip(), "explicit"

    environ = env if env is not None else os.environ
    for env_key in ("SPACE_BARTER_KEY", "SPACE_API_KEY", "LONGINUS_API_KEY"):
        val = (environ.get(env_key) or "").strip()
        if val and val not in PLACEHOLDER_API_KEYS:
            return val, f"env:{env_key}"

    space_cfg = load_space_config(home_dir)
    for field in ("barter_key", "api_key", "openai_api_key"):
        val = str(space_cfg.get(field) or "").strip()
        if val and val not in PLACEHOLDER_API_KEYS:
            return val, "~/.space_ai/config.json"

    node_cfg, node_path = load_node_identity(home_dir)
    if node_cfg and node_path is not None:
        for field in ("barter_key", "api_key", "longinus_api_key"):
            val = str(node_cfg.get(field) or "").strip()
            if val and val not in PLACEHOLDER_API_KEYS:
                return val, f"~/{node_path.name}"

    return None, "none"


def resolve_session_token(
    explicit_session: str | None = None,
    home_dir: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> str | None:
    """Resolve active agent session budget token (`sp_sess_...`)."""
    if explicit_session and explicit_session.strip():
        return explicit_session.strip()
    environ = env if env is not None else os.environ
    env_val = (environ.get("SPACE_SESSION") or environ.get("SPACE_SESSION_TOKEN") or "").strip()
    if env_val:
        return env_val
    space_cfg = load_space_config(home_dir)
    saved = str(space_cfg.get("session_token") or "").strip()
    return saved or None


def resolve_upstreams(
    explicit_coordinators: list[str] | tuple[str, ...] | None = None,
    home_dir: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve ordered list of upstream coordinator/node URLs for failover."""
    urls: list[str] = []

    def _add(raw: str | None) -> None:
        if not raw:
            return
        for part in str(raw).split(","):
            u = part.strip().rstrip("/")
            if u and u.lower() != "auto" and u not in urls:
                urls.append(u)

    if explicit_coordinators:
        for c in explicit_coordinators:
            _add(c)
        if urls:
            return urls

    environ = env if env is not None else os.environ
    _add(environ.get("SPACE_UPSTREAMS"))
    _add(environ.get("SPACE_COORDINATOR"))
    _add(environ.get("LONGINUS_COORDINATOR"))

    space_cfg = load_space_config(home_dir)
    cfg_upstreams = space_cfg.get("upstreams")
    if isinstance(cfg_upstreams, list):
        for item in cfg_upstreams:
            _add(str(item))
    elif isinstance(cfg_upstreams, str):
        _add(cfg_upstreams)
    _add(space_cfg.get("coordinator"))

    if not urls:
        urls = [DEFAULT_COORDINATOR, FALLBACK_COORDINATOR]
    return urls


def mask_secret(secret: str | None) -> str | None:
    """Redact API keys and session tokens for safe display in `/status` and logs."""
    if not secret:
        return None
    s = secret.strip()
    if len(s) <= 10:
        return s[:2] + "..." + s[-2:] if len(s) > 4 else "***"
    return f"{s[:10]}...{s[-4:]}"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


__all__ = [
    "__version__",
    "DEFAULT_PROXY_HOST",
    "DEFAULT_PROXY_PORT",
    "DEFAULT_COORDINATOR",
    "FALLBACK_COORDINATOR",
    "VALID_ROUTES",
    "VALID_PRIVACY_TIERS",
    "STRICT_PRIVACY_TIERS",
    "PLACEHOLDER_API_KEYS",
    "get_space_config_dir",
    "get_space_config_path",
    "load_space_config",
    "save_space_config",
    "load_node_identity",
    "resolve_barter_key",
    "resolve_session_token",
    "resolve_upstreams",
    "mask_secret",
    "get_repo_root",
]
