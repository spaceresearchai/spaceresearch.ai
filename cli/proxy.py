"""Unified Local OpenAI + Anthropic Buyer Proxy for Space AI (Pure Python Stdlib).

Default bind: `127.0.0.1:8377` (zero third-party dependencies).

Supported endpoints:
  - `GET  /health`                  — lightweight liveness probe
  - `GET  /status`                  — barter key status (masked), worker state, routing policy,
                                      failover stats, and per-upstream latency EMA
  - `GET  /v1/models`               — OpenAI-compatible model listing (failover-backed)
  - `POST /v1/chat/completions`     — OpenAI-compatible chat completions (JSON + SSE streaming)
  - `POST /v1/responses`            — OpenAI Responses API passthrough (Codex / Agents SDK)
  - `POST /v1/messages`             — Anthropic Messages API (direct forward OR automatic
                                      clean translation to/from OpenAI `/v1/chat/completions`)
  - `POST /v1/messages/count_tokens`— Anthropic token count helper
  - `POST /_space/policy`           — local runtime routing policy update for launchers

Features:
  - Multi-upstream latency EMA tracking + automatic failover/retry on 5xx/429/network errors
  - Automatic injection of Space routing headers (`X-Space-Route`, `X-Space-Privacy`,
    `X-Space-Region`, `X-Space-Session`, `X-Space-Client`)
  - Automatic barter key discovery from `~/.space_ai/config.json` or `~/.longinus_node.json`
"""

from __future__ import annotations

import http.client
import json
import ssl
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from cli import (
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    PLACEHOLDER_API_KEYS,
    STRICT_PRIVACY_TIERS,
    __version__,
    mask_secret,
    resolve_barter_key,
    resolve_session_token,
    resolve_upstreams,
)
from cli.e2ee import (
    E2EE_PROTOCOL_VERSION,
    E2EEError,
    e2ee_available,
    encrypt_client_request,
    hash_private_pool_secret,
    is_loopback_url,
    verify_e2ee_pubkey_binding,
)

RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
PASSTHROUGH_RESPONSE_HEADERS = (
    "x-space-route-applied",
    "x-space-privacy-applied",
    "x-space-privacy-strict",
    "x-space-privacy-matched",
    "x-space-privacy-downgraded",
    "x-space-e2ee-protocol",
    "x-space-region-applied",
    "x-request-id",
    "x-space-node",
    "x-space-receipt-id",
)


@dataclass
class ProxyConfig:
    host: str = DEFAULT_PROXY_HOST
    port: int = DEFAULT_PROXY_PORT
    upstreams: list[str] = field(default_factory=list)
    route: str = "fastest"
    privacy: str = "standard"
    region: str = ""
    private_node: str = ""
    private_pool_secret: str = ""
    allow_privacy_fallback: bool = False
    session_token: str | None = None
    client_tag: str = ""
    default_model: str = ""
    api_key: str | None = None
    home_dir: Path | str | None = None
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    circuit_cooldown_s: float = 15.0
    ema_alpha: float = 0.30
    force_translate_anthropic: bool = False
    quiet: bool = False


@dataclass
class UpstreamState:
    url: str
    order_index: int
    latency_ema_ms: float | None = None
    requests_ok: int = 0
    requests_failed: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    last_status: int | None = None
    last_error: str | None = None
    last_used_at: float | None = None

    def is_circuit_open(self, now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        return self.circuit_open_until > t

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        t = now if now is not None else time.time()
        return {
            "url": self.url,
            "healthy": not self.is_circuit_open(t),
            "latency_ema_ms": round(self.latency_ema_ms, 2) if self.latency_ema_ms is not None else None,
            "requests_ok": self.requests_ok,
            "requests_failed": self.requests_failed,
            "consecutive_failures": self.consecutive_failures,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "last_used_at": self.last_used_at,
        }


class UpstreamPool:
    """Thread-safe multi-upstream tracker with latency EMA ranking & circuit-breaker failover."""

    def __init__(self, urls: list[str], ema_alpha: float = 0.30, circuit_cooldown_s: float = 15.0) -> None:
        cleaned: list[str] = []
        for u in urls:
            s = str(u).strip().rstrip("/")
            if s and s not in cleaned:
                cleaned.append(s)
        if not cleaned:
            cleaned = resolve_upstreams()
        self._lock = threading.Lock()
        self._ema_alpha = max(0.05, min(0.95, float(ema_alpha)))
        self._cooldown_s = max(1.0, float(circuit_cooldown_s))
        self._states: dict[str, UpstreamState] = {
            u: UpstreamState(url=u, order_index=idx) for idx, u in enumerate(cleaned)
        }
        self._order: list[str] = cleaned

    def ordered_candidates(self) -> list[UpstreamState]:
        """Return upstreams ordered: healthy first (by EMA latency, then priority), circuit-open last."""
        now = time.time()
        with self._lock:
            states = list(self._states.values())

        def _sort_key(st: UpstreamState) -> tuple[int, int, float, int]:
            is_open = 1 if st.is_circuit_open(now) else 0
            # Observed healthy upstreams ranked by EMA latency; unobserved keep configured order
            has_ema = 0 if st.latency_ema_ms is not None else 1
            ema = st.latency_ema_ms if st.latency_ema_ms is not None else 0.0
            return (is_open, has_ema, ema, st.order_index)

        return sorted(states, key=_sort_key)

    def record_success(self, url: str, latency_ms: float, status_code: int = 200) -> None:
        now = time.time()
        with self._lock:
            st = self._states.get(url)
            if st is None:
                return
            st.requests_ok += 1
            st.consecutive_failures = 0
            st.circuit_open_until = 0.0
            st.last_status = int(status_code)
            st.last_error = None
            st.last_used_at = now
            if st.latency_ema_ms is None:
                st.latency_ema_ms = float(latency_ms)
            else:
                a = self._ema_alpha
                st.latency_ema_ms = (a * float(latency_ms)) + ((1.0 - a) * st.latency_ema_ms)

    def record_failure(self, url: str, error: str, status_code: int | None = None) -> None:
        now = time.time()
        with self._lock:
            st = self._states.get(url)
            if st is None:
                return
            st.requests_failed += 1
            st.consecutive_failures += 1
            st.last_status = status_code
            st.last_error = error[:240]
            st.last_used_at = now
            # Open circuit briefly after failure so subsequent requests prefer healthy peers
            backoff = min(self._cooldown_s * (2 ** min(st.consecutive_failures - 1, 3)), 120.0)
            st.circuit_open_until = now + backoff

    def healthy_count(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for st in self._states.values() if not st.is_circuit_open(now))

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            return [self._states[u].to_dict(now) for u in self._order]


# ---------------------------------------------------------------------------
# Anthropic Messages API <-> OpenAI Chat Completions Translation Helpers
# ---------------------------------------------------------------------------


def anthropic_to_openai_request(payload: dict[str, Any], default_model: str = "") -> dict[str, Any]:
    """Translate an Anthropic `/v1/messages` JSON request into an OpenAI `/v1/chat/completions` request."""
    oai_messages: list[dict[str, Any]] = []

    system = payload.get("system")
    if isinstance(system, str) and system.strip():
        oai_messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        sys_parts = [str(b.get("text", "")) for b in system if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n".join(p for p in sys_parts if p)
        if joined:
            oai_messages.append({"role": "system", "content": joined})

    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            oai_messages.append({"role": role, "content": str(content or "")})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text") or ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif btype == "tool_result":
                res_content = block.get("content")
                if isinstance(res_content, list):
                    res_text = "\n".join(
                        str(x.get("text", "")) for x in res_content if isinstance(x, dict) and x.get("type") == "text"
                    )
                else:
                    res_text = str(res_content or "")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": res_text,
                    }
                )

        if text_parts or tool_calls:
            entry: dict[str, Any] = {"role": role, "content": "\n".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            oai_messages.append(entry)
        oai_messages.extend(tool_results)

    oai_req: dict[str, Any] = {
        "model": str(payload.get("model") or default_model or "space-auto"),
        "messages": oai_messages,
        "stream": bool(payload.get("stream", False)),
    }
    if payload.get("max_tokens") is not None:
        oai_req["max_tokens"] = int(payload["max_tokens"])
    if payload.get("temperature") is not None:
        oai_req["temperature"] = float(payload["temperature"])
    if payload.get("top_p") is not None:
        oai_req["top_p"] = float(payload["top_p"])
    if payload.get("stop_sequences"):
        oai_req["stop"] = payload["stop_sequences"]
    if isinstance(payload.get("tools"), list) and payload["tools"]:
        oai_tools = []
        for t in payload["tools"]:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            oai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        if oai_tools:
            oai_req["tools"] = oai_tools
    return oai_req


def _map_finish_reason(finish_reason: str | None) -> str:
    fr = (finish_reason or "stop").lower()
    if fr == "length":
        return "max_tokens"
    if fr in ("tool_calls", "function_call"):
        return "tool_use"
    return "end_turn"


def openai_to_anthropic_response(oai_resp: dict[str, Any], requested_model: str = "") -> dict[str, Any]:
    """Translate an OpenAI `/v1/chat/completions` JSON response into an Anthropic `/v1/messages` response."""
    choices = oai_resp.get("choices") or []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}

    content_blocks: list[dict[str, Any]] = []
    text = msg.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})

    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed_input = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            parsed_input = {"raw": raw_args}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": str(fn.get("name") or ""),
                "input": parsed_input if isinstance(parsed_input, dict) else {"value": parsed_input},
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = oai_resp.get("usage") if isinstance(oai_resp.get("usage"), dict) else {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)

    raw_id = str(oai_resp.get("id") or uuid.uuid4().hex[:16])
    msg_id = raw_id if raw_id.startswith("msg_") else f"msg_{raw_id}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": str(oai_resp.get("model") or requested_model or "space-auto"),
        "content": content_blocks,
        "stop_reason": _map_finish_reason(choice0.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Space Local Buyer Proxy Server
# ---------------------------------------------------------------------------


class _SpaceThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], proxy: SpaceLocalProxy):
        self.proxy = proxy
        super().__init__(server_address, handler_cls)


def _open_upstream_connection(
    base_url: str,
    timeout: float,
) -> tuple[http.client.HTTPConnection, str]:
    parsed = urllib.parse.urlsplit(base_url)
    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")
    if scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            host,
            port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    return conn, prefix


class SpaceLocalProxy:
    """Unified Local OpenAI/Anthropic Buyer Proxy with multi-upstream failover & Space header injection."""

    def __init__(
        self,
        config: ProxyConfig | None = None,
        worker_status_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or ProxyConfig()
        upstreams = self.config.upstreams or resolve_upstreams(home_dir=self.config.home_dir)
        self.config.upstreams = list(upstreams)
        self.pool = UpstreamPool(
            upstreams,
            ema_alpha=self.config.ema_alpha,
            circuit_cooldown_s=self.config.circuit_cooldown_s,
        )
        self._worker_status_provider = worker_status_provider
        self._stats_lock = threading.Lock()
        self._policy_lock = threading.Lock()
        self.started_at = time.time()
        self.stats: dict[str, int] = {
            "total_requests": 0,
            "total_openai_calls": 0,
            "total_anthropic_calls": 0,
            "total_e2ee_calls": 0,
            "total_airgap_blocks": 0,
            "total_translations": 0,
            "total_failovers": 0,
            "total_retries": 0,
            "total_errors": 0,
        }
        self._server: _SpaceThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self.config.host

    @property
    def port(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        return int(self.config.port)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def set_worker_status_provider(self, provider: Callable[[], dict[str, Any]] | None) -> None:
        self._worker_status_provider = provider

    def bump_stat(self, key: str, delta: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + delta

    def get_routing_policy(self) -> dict[str, Any]:
        with self._policy_lock:
            session = resolve_session_token(self.config.session_token, home_dir=self.config.home_dir)
            pool_id = hash_private_pool_secret(self.config.private_pool_secret)
            return {
                "route": self.config.route or "fastest",
                "privacy": self.config.privacy or "standard",
                "region": self.config.region or "",
                "private_node": (self.config.private_node or "").strip(),
                "private_pool_id": pool_id,
                "allow_privacy_fallback": bool(self.config.allow_privacy_fallback),
                "session_token": session,
                "client_tag": self.config.client_tag or "",
                "default_model": self.config.default_model or "",
            }

    def update_routing_policy(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._policy_lock:
            if "route" in updates and updates["route"]:
                self.config.route = str(updates["route"]).strip()
            if "privacy" in updates and updates["privacy"]:
                self.config.privacy = str(updates["privacy"]).strip()
            if "region" in updates and updates["region"] is not None:
                self.config.region = str(updates["region"]).strip()
            if "private_node" in updates and updates["private_node"] is not None:
                self.config.private_node = str(updates["private_node"]).strip()
            if "private_pool_secret" in updates and updates["private_pool_secret"] is not None:
                self.config.private_pool_secret = str(updates["private_pool_secret"]).strip()
            if "allow_privacy_fallback" in updates:
                self.config.allow_privacy_fallback = bool(updates["allow_privacy_fallback"])
            if "session_token" in updates and updates["session_token"] is not None:
                self.config.session_token = str(updates["session_token"]).strip() or None
            if "client_tag" in updates and updates["client_tag"] is not None:
                self.config.client_tag = str(updates["client_tag"]).strip()
            if "default_model" in updates and updates["default_model"] is not None:
                self.config.default_model = str(updates["default_model"]).strip()
        return self.get_routing_policy()

    def active_barter_key(self) -> tuple[str | None, str]:
        return resolve_barter_key(explicit_key=self.config.api_key, home_dir=self.config.home_dir)

    def build_upstream_headers(
        self,
        incoming_headers: Any,
        *,
        is_anthropic: bool = False,
        is_stream: bool = False,
        content_type: str = "application/json",
    ) -> dict[str, str]:
        """Build forwarded headers with Space routing policy & auto-discovered barter key."""
        policy = self.get_routing_policy()
        out: dict[str, str] = {
            "User-Agent": f"space-local-proxy/{__version__}",
            "Content-Type": content_type,
        }
        if is_stream:
            out["Accept"] = "text/event-stream"

        # Forward Anthropic-specific version/beta headers when present
        for h_name in ("anthropic-version", "anthropic-beta", "x-session-id"):
            val = incoming_headers.get(h_name)
            if val:
                out[h_name] = str(val).strip()

        # Resolve caller auth vs auto-saved barter key
        raw_auth = (incoming_headers.get("Authorization") or "").strip()
        raw_xkey = (incoming_headers.get("x-api-key") or "").strip()
        caller_token = ""
        if raw_auth:
            parts = raw_auth.split(" ", 1)
            caller_token = parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else raw_auth
        elif raw_xkey:
            caller_token = raw_xkey

        saved_key, _ = self.active_barter_key()
        effective_key = caller_token if caller_token and caller_token not in PLACEHOLDER_API_KEYS else (saved_key or "")
        if effective_key:
            out["Authorization"] = f"Bearer {effective_key}"
            out["x-api-key"] = effective_key

        # Inject Space routing headers (explicit incoming header wins over proxy default)
        route = (incoming_headers.get("X-Space-Route") or policy["route"] or "").strip()
        privacy = (incoming_headers.get("X-Space-Privacy") or policy["privacy"] or "").strip().lower()
        region = (incoming_headers.get("X-Space-Region") or policy["region"] or "").strip()
        session = (incoming_headers.get("X-Space-Session") or policy["session_token"] or "").strip()
        client = (incoming_headers.get("X-Space-Client") or policy["client_tag"] or "").strip()
        private_node = (incoming_headers.get("X-Space-Private-Node") or policy["private_node"] or "").strip()
        private_pool_id = (incoming_headers.get("X-Space-Private-Pool") or policy["private_pool_id"] or "").strip()
        fallback_hdr = (incoming_headers.get("X-Space-Privacy-Fallback") or "").strip().lower()
        if not fallback_hdr and policy.get("allow_privacy_fallback"):
            fallback_hdr = "allow"
        if not client and is_anthropic:
            client = "coding-agent"

        if route:
            out["X-Space-Route"] = route
        if privacy:
            out["X-Space-Privacy"] = privacy
        if region:
            out["X-Space-Region"] = region
        if session:
            out["X-Space-Session"] = session
        if client:
            out["X-Space-Client"] = client
        if private_node:
            out["X-Space-Private-Node"] = private_node
        if private_pool_id:
            out["X-Space-Private-Pool"] = private_pool_id
        if fallback_hdr:
            out["X-Space-Privacy-Fallback"] = fallback_hdr

        return out

    def health_payload(self) -> dict[str, Any]:
        worker_info = self._get_worker_status()
        return {
            "status": "ok",
            "service": "space-local-proxy",
            "version": __version__,
            "bind": self.base_url,
            "upstreams_healthy": self.pool.healthy_count(),
            "upstreams_total": len(self.config.upstreams),
            "worker_state": worker_info.get("state", "disabled"),
            "uptime_s": round(time.time() - self.started_at, 2),
        }

    def _get_worker_status(self) -> dict[str, Any]:
        if self._worker_status_provider is None:
            return {"enabled": False, "adapter": None, "state": "disabled"}
        try:
            data = self._worker_status_provider()
            return data if isinstance(data, dict) else {"enabled": True, "state": "unknown"}
        except Exception as exc:
            return {"enabled": True, "state": "error", "last_error": str(exc)}

    def status_payload(self) -> dict[str, Any]:
        key, key_source = self.active_barter_key()
        policy = self.get_routing_policy()
        with self._stats_lock:
            stats_copy = dict(self.stats)
        return {
            "service": "space-local-proxy",
            "version": __version__,
            "bind": self.base_url,
            "uptime_s": round(time.time() - self.started_at, 2),
            "barter_key": {
                "configured": bool(key),
                "masked": mask_secret(key),
                "source": key_source,
            },
            "worker": self._get_worker_status(),
            "routing_policy": {
                "route": policy["route"],
                "privacy": policy["privacy"],
                "privacy_strict": policy["privacy"] in STRICT_PRIVACY_TIERS and not policy["allow_privacy_fallback"],
                "e2ee_available": e2ee_available(),
                "e2ee_protocol": E2EE_PROTOCOL_VERSION,
                "private_node": policy["private_node"] or None,
                "private_pool_configured": bool(policy["private_pool_id"]),
                "region": policy["region"] or "auto",
                "session_configured": bool(policy["session_token"]),
                "session_masked": mask_secret(policy["session_token"]),
                "client": policy["client_tag"] or "auto",
                "default_model": policy["default_model"] or None,
            },
            "stats": stats_copy,
            "upstreams": self.pool.snapshot(),
        }

    def _effective_privacy(self, headers: dict[str, str] | Any | None = None) -> str:
        raw = ""
        if headers is not None:
            raw = str(headers.get("X-Space-Privacy") or headers.get("x-space-privacy") or "").strip().lower()
        if not raw:
            raw = str(self.get_routing_policy().get("privacy") or "standard").strip().lower()
        return raw or "standard"

    def execute_upstream(
        self,
        method: str,
        path: str,
        body: bytes | None,
        headers: dict[str, str],
        *,
        stream: bool = False,
        fallback_statuses: frozenset[int] = RETRYABLE_HTTP_STATUSES,
    ) -> tuple[http.client.HTTPConnection | None, http.client.HTTPResponse | None, bytes, int, dict[str, str], str, int]:
        """Execute HTTP request across upstreams with automatic failover & latency EMA tracking.

        Enforces strict air-gap isolation when `X-Space-Privacy: local-only` is active:
        non-loopback upstreams are completely filtered out before any socket is opened.

        Returns: `(conn_if_streaming, resp_if_streaming, body_bytes, status_code, resp_headers, chosen_upstream, failovers)`
        """
        candidates = self.pool.ordered_candidates()
        effective_privacy = self._effective_privacy(headers)
        if effective_privacy == "local-only":
            candidates = [c for c in candidates if is_loopback_url(c.url)]
            if not candidates:
                self.bump_stat("total_airgap_blocks")
                self.bump_stat("total_errors")
                err_payload = json.dumps(
                    {
                        "error": {
                            "type": "privacy_policy_error",
                            "code": "local_only_airgap_blocked",
                            "message": (
                                "X-Space-Privacy: local-only blocks all non-loopback network egress, "
                                "and no loopback inference upstream (127.0.0.1 / localhost / ::1) is configured."
                            ),
                            "hint": (
                                "Start a local model server (e.g. llama-server, Ollama, MLX, or vLLM) "
                                "and pass `--coordinator http://127.0.0.1:<port>`."
                            ),
                        }
                    }
                ).encode("utf-8")
                return (
                    None,
                    None,
                    err_payload,
                    503,
                    {
                        "x-space-privacy-applied": "local-only",
                        "x-space-privacy-strict": "true",
                        "x-space-privacy-matched": "false",
                    },
                    "airgap-blocked",
                    0,
                )

        failovers = 0
        last_error = "No upstream servers configured"
        last_status = 502
        last_body = b""
        last_headers: dict[str, str] = {}
        last_upstream = candidates[0].url if candidates else ""

        for idx, cand in enumerate(candidates):
            has_next = idx < (len(candidates) - 1)
            conn: http.client.HTTPConnection | None = None
            t0 = time.perf_counter()
            try:
                conn, prefix = _open_upstream_connection(cand.url, timeout=self.config.request_timeout_s)
                full_path = f"{prefix}{path}" if prefix else path
                req_headers = dict(headers)
                if body is not None:
                    req_headers["Content-Length"] = str(len(body))
                elif method in ("POST", "PUT", "PATCH"):
                    req_headers["Content-Length"] = "0"
                conn.request(method, full_path, body=body, headers=req_headers)
                resp = conn.getresponse()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                status = int(resp.status)

                if status in fallback_statuses and has_next:
                    err_body = resp.read()
                    conn.close()
                    self.pool.record_failure(cand.url, f"HTTP {status}", status_code=status)
                    failovers += 1
                    self.bump_stat("total_failovers")
                    self.bump_stat("total_retries")
                    last_status = status
                    last_body = err_body
                    last_headers = resp_headers
                    last_upstream = cand.url
                    last_error = f"HTTP {status} from {cand.url}"
                    continue

                if status >= 500:
                    self.pool.record_failure(cand.url, f"HTTP {status}", status_code=status)
                else:
                    self.pool.record_success(cand.url, lat_ms, status_code=status)

                if stream and status < 400:
                    return conn, resp, b"", status, resp_headers, cand.url, failovers

                data = resp.read()
                conn.close()
                return None, None, data, status, resp_headers, cand.url, failovers

            except Exception as exc:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                self.pool.record_failure(cand.url, str(exc), status_code=None)
                last_error = f"{cand.url}: {exc}"
                last_upstream = cand.url
                if has_next:
                    failovers += 1
                    self.bump_stat("total_failovers")
                    self.bump_stat("total_retries")
                    continue

        self.bump_stat("total_errors")
        err_payload = json.dumps(
            {
                "error": {
                    "type": "upstream_unavailable",
                    "message": f"All Space AI upstreams failed ({last_error}).",
                    "hint": "Check network connectivity or run `space status`.",
                }
            }
        ).encode("utf-8")
        return None, None, last_body or err_payload, last_status, last_headers, last_upstream, failovers

    def execute_e2ee_chat(
        self,
        oai_payload: dict[str, Any],
        incoming_headers: Any,
    ) -> tuple[int, dict[str, Any], dict[str, str], str, int]:
        """Execute a blind-relay E2EE (`space-e2ee-v1`) inference request.

        Invariant: plaintext prompts/messages are encrypted strictly inside this local
        proxy before any network call. The coordinator only relays ciphertext and never
        receives plaintext prompt or output bytes. Fails closed (503) if no worker with
        a cryptographically verified Ed25519->X25519 key binding is available.
        """
        headers = self.build_upstream_headers(incoming_headers, is_anthropic=False, is_stream=False)
        headers["X-Space-Privacy"] = "e2ee"
        pinned_node = headers.get("X-Space-Private-Node", "").strip()
        pool_id = headers.get("X-Space-Private-Pool", "").strip()

        def _err(status_code: int, code: str, message: str, chosen: str = "e2ee-fail-closed", failovers: int = 0):
            self.bump_stat("total_errors")
            return (
                status_code,
                {"error": {"type": "privacy_policy_error", "code": code, "message": message}},
                {
                    "x-space-privacy-applied": "e2ee",
                    "x-space-privacy-strict": "true",
                    "x-space-privacy-matched": "false",
                    "x-space-e2ee-protocol": E2EE_PROTOCOL_VERSION,
                },
                chosen,
                failovers,
            )

        if not e2ee_available():
            return _err(
                503,
                "e2ee_crypto_unavailable",
                "Client E2EE requires the Python `cryptography` package. Refusing to send plaintext.",
            )

        # 1. Fetch worker E2EE public key directory from coordinator (public keys only; verified locally!)
        qs = []
        if pinned_node:
            qs.append(f"wallet={urllib.parse.quote(pinned_node)}")
        if pool_id:
            qs.append(f"private_pool_id={urllib.parse.quote(pool_id)}")
        key_path = "/v1/nodes/e2ee-keys" + (f"?{'&'.join(qs)}" if qs else "")
        _, _, key_body, key_status, _, chosen, failovers = self.execute_upstream(
            "GET", key_path, None, headers, stream=False
        )
        if key_status != 200:
            return _err(
                503,
                "e2ee_directory_unavailable",
                f"Could not fetch verified E2EE worker keys (HTTP {key_status}). Refusing plaintext fallback.",
                chosen=chosen,
                failovers=failovers,
            )

        try:
            key_data = json.loads(key_body.decode("utf-8"))
            nodes = key_data.get("nodes") if isinstance(key_data, dict) else []
            if not isinstance(nodes, list):
                nodes = []
        except Exception as exc:
            return _err(
                503,
                "e2ee_directory_invalid",
                f"Malformed E2EE key directory response ({exc}). Refusing plaintext fallback.",
                chosen=chosen,
                failovers=failovers,
            )

        # 2. Locally verify Ed25519->X25519 key bindings (protects against tampered/MITM coordinator!)
        verified_worker: dict[str, Any] | None = None
        rejected_bad_sig = 0
        for item in nodes:
            if not isinstance(item, dict):
                continue
            w = str(item.get("wallet") or "").strip()
            pk = str(item.get("e2ee_pubkey_b64") or "").strip()
            sig = str(item.get("e2ee_pubkey_sig_hex") or "").strip()
            if pinned_node and w != pinned_node:
                continue
            if not w or not pk or not sig:
                continue
            if verify_e2ee_pubkey_binding(w, pk, sig):
                verified_worker = item
                break
            rejected_bad_sig += 1

        if verified_worker is None:
            detail = (
                f"Rejected {rejected_bad_sig} worker key(s) due to invalid Ed25519->X25519 signature binding! "
                if rejected_bad_sig
                else ""
            )
            return _err(
                503,
                "e2ee_key_verification_failed" if rejected_bad_sig else "e2ee_no_verified_worker",
                (
                    f"{detail}No cryptographically verified E2EE worker available"
                    + (f" matching pinned wallet {pinned_node[:12]}..." if pinned_node else "")
                    + ". Refusing to downgrade to plaintext."
                ),
                chosen=chosen,
                failovers=failovers,
            )

        # 3. Encrypt the prompt/messages payload client-side BEFORE calling the blind relay
        target_wallet = str(verified_worker["wallet"]).strip()
        model_id = str(oai_payload.get("model") or verified_worker.get("model_id") or "space-auto")
        req_id = f"e2ee_{uuid.uuid4().hex[:16]}"
        inner_payload = {
            "model": model_id,
            "messages": oai_payload.get("messages") or [],
            "max_tokens": oai_payload.get("max_tokens"),
            "temperature": oai_payload.get("temperature"),
        }
        try:
            envelope, session = encrypt_client_request(
                inner_payload,
                worker_wallet=target_wallet,
                worker_e2ee_pubkey_b64=str(verified_worker["e2ee_pubkey_b64"]),
                worker_e2ee_sig_hex=str(verified_worker["e2ee_pubkey_sig_hex"]),
                request_id=req_id,
            )
        except E2EEError as exc:
            return _err(503, "e2ee_encrypt_failed", str(exc), chosen=chosen, failovers=failovers)

        relay_body = json.dumps(
            {
                "request_id": req_id,
                "target_node": target_wallet,
                "model_id": model_id,
                "e2ee_envelope": envelope,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        # 4. Send ONLY ciphertext to /v1/e2ee/infer (coordinator never sees plaintext)
        self.bump_stat("total_e2ee_calls")
        _, _, resp_body, resp_status, resp_headers, chosen2, failovers2 = self.execute_upstream(
            "POST", "/v1/e2ee/infer", relay_body, headers, stream=False
        )
        total_failovers = failovers + failovers2
        if resp_status != 200:
            try:
                err_obj = json.loads(resp_body.decode("utf-8"))
            except Exception:
                err_obj = {"error": {"type": "e2ee_relay_error", "message": f"Blind relay returned HTTP {resp_status}"}}
            return resp_status, err_obj, resp_headers, chosen2, total_failovers

        # 5. Decrypt worker response locally & verify AES-256-GCM tag + request_id + wallet
        try:
            relay_json = json.loads(resp_body.decode("utf-8"))
            enc_resp = relay_json.get("e2ee_response")
            decrypted = session.decrypt_response(enc_resp)
        except Exception as exc:
            return _err(
                502,
                "e2ee_response_decrypt_failed",
                f"Failed to authenticate/decrypt worker E2EE response: {exc}",
                chosen=chosen2,
                failovers=total_failovers,
            )

        out_text = str(decrypted.get("output_text") or "")
        tokens_served = int(decrypted.get("tokens_served") or max(1, len(out_text.split())))
        out_model = str(decrypted.get("model_id") or model_id)
        now_i = int(time.time())
        oai_response = {
            "id": f"chatcmpl-{req_id}",
            "object": "chat.completion",
            "created": now_i,
            "model": out_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": out_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens_served,
                "total_tokens": tokens_served,
            },
            "space_privacy": {
                "mode": "e2ee",
                "protocol": E2EE_PROTOCOL_VERSION,
                "worker_wallet": target_wallet,
                "coordinator_blind": True,
                "key_binding_verified": True,
            },
        }
        out_headers = dict(resp_headers)
        out_headers.update(
            {
                "x-space-privacy-applied": "e2ee",
                "x-space-privacy-strict": "true",
                "x-space-privacy-matched": "true",
                "x-space-e2ee-protocol": E2EE_PROTOCOL_VERSION,
                "x-space-node": target_wallet,
            }
        )
        return 200, oai_response, out_headers, chosen2, total_failovers

    def start(self, background: bool = False) -> None:
        if self._server is not None:
            return
        if (self.config.privacy or "").strip().lower() == "local-only":
            if not is_loopback_url(f"http://{self.config.host}:{int(self.config.port)}"):
                raise ValueError(
                    f"Refusing to bind local-only proxy on non-loopback host {self.config.host!r}. "
                    "Use 127.0.0.1, localhost, or ::1."
                )
        self._server = _SpaceThreadingHTTPServer((self.config.host, int(self.config.port)), _SpaceProxyHandler, self)
        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="space-local-proxy",
                daemon=True,
            )
            self._thread.start()
        else:
            self._server.serve_forever(poll_interval=0.2)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


class _SpaceProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        proxy: SpaceLocalProxy = self.server.proxy  # type: ignore[attr-defined]
        if not proxy.config.quiet:
            super().log_message(format, *args)

    def _send_json(self, status: int, data: dict[str, Any], extra_headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _send_raw(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        cl = self.headers.get("Content-Length")
        if cl is not None:
            try:
                length = max(0, int(cl))
                return self.rfile.read(length) if length > 0 else b""
            except ValueError:
                return b""
        return b""

    def _copy_proxy_headers(self, upstream_headers: dict[str, str], upstream_url: str, failovers: int) -> dict[str, str]:
        out: dict[str, str] = {
            "X-Space-Proxy-Upstream": upstream_url,
            "X-Space-Proxy-Failovers": str(failovers),
        }
        for key in PASSTHROUGH_RESPONSE_HEADERS:
            if key in upstream_headers:
                out[key] = upstream_headers[key]
        return out

    def _stream_passthrough(
        self,
        conn: http.client.HTTPConnection,
        resp: http.client.HTTPResponse,
        status: int,
        resp_headers: dict[str, str],
        upstream_url: str,
        failovers: int,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", resp_headers.get("content-type", "text/event-stream"))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            for k, v in self._copy_proxy_headers(resp_headers, upstream_url, failovers).items():
                self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def do_GET(self) -> None:
        proxy: SpaceLocalProxy = self.server.proxy  # type: ignore[attr-defined]
        proxy.bump_stat("total_requests")
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json(200, proxy.health_payload())
            return
        if path == "/status":
            self._send_json(200, proxy.status_payload())
            return
        if path == "/v1/models":
            headers = proxy.build_upstream_headers(self.headers, is_anthropic=False, is_stream=False)
            _, _, body, status, resp_headers, chosen, failovers = proxy.execute_upstream(
                "GET", self.path, None, headers, stream=False
            )
            self._send_raw(
                status,
                body,
                content_type=resp_headers.get("content-type", "application/json"),
                extra_headers=self._copy_proxy_headers(resp_headers, chosen, failovers),
            )
            return

        self._send_json(404, {"error": f"Unknown GET route: {path}"})

    def do_POST(self) -> None:
        proxy: SpaceLocalProxy = self.server.proxy  # type: ignore[attr-defined]
        proxy.bump_stat("total_requests")
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        raw_body = self._read_body()

        if path == "/_space/policy":
            try:
                updates = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                if not isinstance(updates, dict):
                    self._send_json(400, {"error": "Policy payload must be a JSON object"})
                    return
                updated = proxy.update_routing_policy(updates)
                self._send_json(200, {"status": "ok", "routing_policy": updated})
            except Exception as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        if path == "/v1/messages/count_tokens":
            self._handle_count_tokens(proxy, raw_body)
            return

        if path in ("/v1/chat/completions", "/v1/responses"):
            proxy.bump_stat("total_openai_calls")
            self._handle_openai_post(proxy, self.path, raw_body)
            return

        if path == "/v1/messages":
            proxy.bump_stat("total_anthropic_calls")
            self._handle_anthropic_messages(proxy, raw_body)
            return

        self._send_json(404, {"error": f"Unknown POST route: {path}"})

    def _send_openai_sse_from_completion(
        self,
        oai_json: dict[str, Any],
        extra_headers: dict[str, str],
    ) -> None:
        comp_id = str(oai_json.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}")
        created = int(oai_json.get("created") or time.time())
        model = str(oai_json.get("model") or "space-e2ee")
        choices = oai_json.get("choices") or []
        content = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            content = str(msg.get("content") or "")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        for k, v in extra_headers.items():
            self.send_header(k, v)
        self.end_headers()
        first_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
        }
        end_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "space_privacy": oai_json.get("space_privacy"),
        }
        self.wfile.write(f"data: {json.dumps(first_chunk, separators=(',', ':'))}\n\n".encode("utf-8"))
        self.wfile.write(f"data: {json.dumps(end_chunk, separators=(',', ':'))}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_anthropic_sse_from_response(
        self,
        anth_resp: dict[str, Any],
        extra: dict[str, str],
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()

        msg_shell = dict(anth_resp)
        msg_shell["content"] = []
        msg_shell["stop_reason"] = None
        self.wfile.write(_sse_event("message_start", {"type": "message_start", "message": msg_shell}))
        for idx, block in enumerate(anth_resp["content"]):
            if block.get("type") == "text":
                self.wfile.write(
                    _sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}},
                    )
                )
                if block.get("text"):
                    self.wfile.write(
                        _sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": block["text"]},
                            },
                        )
                    )
                self.wfile.write(_sse_event("content_block_stop", {"type": "content_block_stop", "index": idx}))
            elif block.get("type") == "tool_use":
                self.wfile.write(
                    _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": {},
                            },
                        },
                    )
                )
                self.wfile.write(
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})},
                        },
                    )
                )
                self.wfile.write(_sse_event("content_block_stop", {"type": "content_block_stop", "index": idx}))
        self.wfile.write(
            _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": anth_resp["stop_reason"], "stop_sequence": None},
                    "usage": {"output_tokens": anth_resp["usage"]["output_tokens"]},
                },
            )
        )
        self.wfile.write(_sse_event("message_stop", {"type": "message_stop"}))
        self.wfile.flush()

    def _handle_count_tokens(self, proxy: SpaceLocalProxy, raw_body: bytes) -> None:
        effective_privacy = proxy._effective_privacy(self.headers)
        if effective_privacy in ("local-only", "e2ee"):
            self._send_json(
                200,
                {"input_tokens": max(1, len(raw_body) // 4)},
                extra_headers={
                    "X-Space-Privacy-Applied": effective_privacy,
                    "X-Space-Privacy-Strict": "true",
                },
            )
            return
        headers = proxy.build_upstream_headers(self.headers, is_anthropic=True, is_stream=False)
        _, _, body, status, resp_headers, chosen, failovers = proxy.execute_upstream(
            "POST", "/v1/messages/count_tokens", raw_body, headers, stream=False
        )
        if status in (404, 405, 501):
            self._send_json(200, {"input_tokens": max(1, len(raw_body) // 4)})
            return
        self._send_raw(
            status,
            body,
            content_type=resp_headers.get("content-type", "application/json"),
            extra_headers=self._copy_proxy_headers(resp_headers, chosen, failovers),
        )

    def _handle_openai_post(self, proxy: SpaceLocalProxy, req_path: str, raw_body: bytes) -> None:
        is_stream = False
        body_to_send = raw_body
        parsed_payload: dict[str, Any] | None = None
        if raw_body:
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                if isinstance(payload, dict):
                    parsed_payload = payload
                    is_stream = bool(payload.get("stream", False))
                    policy = proxy.get_routing_policy()
                    if not payload.get("model") and policy["default_model"]:
                        payload["model"] = policy["default_model"]
                        body_to_send = json.dumps(payload).encode("utf-8")
            except Exception:
                pass

        effective_privacy = proxy._effective_privacy(self.headers)
        if effective_privacy == "e2ee":
            if parsed_payload is None:
                self._send_json(400, {"error": {"type": "invalid_request_error", "message": "E2EE requires a JSON body"}})
                return
            status, oai_json, resp_headers, chosen, failovers = proxy.execute_e2ee_chat(parsed_payload, self.headers)
            extra = self._copy_proxy_headers(resp_headers, chosen, failovers)
            if status != 200:
                self._send_json(status, oai_json, extra_headers=extra)
                return
            if is_stream:
                self._send_openai_sse_from_completion(oai_json, extra)
                return
            self._send_json(200, oai_json, extra_headers=extra)
            return

        headers = proxy.build_upstream_headers(self.headers, is_anthropic=False, is_stream=is_stream)
        conn, resp, body, status, resp_headers, chosen, failovers = proxy.execute_upstream(
            "POST", req_path, body_to_send, headers, stream=is_stream
        )
        if conn is not None and resp is not None:
            self._stream_passthrough(conn, resp, status, resp_headers, chosen, failovers)
            return
        self._send_raw(
            status,
            body,
            content_type=resp_headers.get("content-type", "application/json"),
            extra_headers=self._copy_proxy_headers(resp_headers, chosen, failovers),
        )

    def _handle_anthropic_messages(self, proxy: SpaceLocalProxy, raw_body: bytes) -> None:
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception as exc:
            self._send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}})
            return
        if not isinstance(payload, dict):
            self._send_json(
                400, {"type": "error", "error": {"type": "invalid_request_error", "message": "JSON body must be an object"}}
            )
            return

        is_stream = bool(payload.get("stream", False))
        effective_privacy = proxy._effective_privacy(self.headers)

        # 0. Strict E2EE blind relay for Anthropic clients (Claude Code, etc.) — never sends plaintext upstream!
        if effective_privacy == "e2ee":
            proxy.bump_stat("total_translations")
            policy = proxy.get_routing_policy()
            oai_payload = anthropic_to_openai_request(payload, default_model=policy["default_model"])
            status, oai_json, resp_headers, chosen, failovers = proxy.execute_e2ee_chat(oai_payload, self.headers)
            extra = self._copy_proxy_headers(resp_headers, chosen, failovers)
            extra["X-Space-Proxy-Translated"] = "anthropic-e2ee-blind-relay"
            if status != 200:
                self._send_json(status, oai_json, extra_headers=extra)
                return
            anth_resp = openai_to_anthropic_response(oai_json, requested_model=str(payload.get("model") or ""))
            if not is_stream:
                self._send_json(200, anth_resp, extra_headers=extra)
                return
            self._send_anthropic_sse_from_response(anth_resp, extra)
            return

        headers = proxy.build_upstream_headers(self.headers, is_anthropic=True, is_stream=is_stream)

        # 1. Try native Anthropic forwarding first unless forced into translation mode
        if not proxy.config.force_translate_anthropic:
            conn, resp, body, status, resp_headers, chosen, failovers = proxy.execute_upstream(
                "POST", "/v1/messages", raw_body, headers, stream=is_stream
            )
            if conn is not None and resp is not None:
                self._stream_passthrough(conn, resp, status, resp_headers, chosen, failovers)
                return
            if status not in (404, 405, 501):
                self._send_raw(
                    status,
                    body,
                    content_type=resp_headers.get("content-type", "application/json"),
                    extra_headers=self._copy_proxy_headers(resp_headers, chosen, failovers),
                )
                return

        # 2. Upstream is OpenAI-only (404/405/501 on /v1/messages): cleanly translate!
        proxy.bump_stat("total_translations")
        policy = proxy.get_routing_policy()
        oai_payload = anthropic_to_openai_request(payload, default_model=policy["default_model"])
        # For translated streaming, fetch non-streaming or streaming OpenAI and emit Anthropic SSE
        oai_payload["stream"] = False
        oai_bytes = json.dumps(oai_payload).encode("utf-8")
        oai_headers = proxy.build_upstream_headers(self.headers, is_anthropic=True, is_stream=False)
        _, _, body, status, resp_headers, chosen, failovers = proxy.execute_upstream(
            "POST", "/v1/chat/completions", oai_bytes, oai_headers, stream=False
        )
        extra = self._copy_proxy_headers(resp_headers, chosen, failovers)
        extra["X-Space-Proxy-Translated"] = "anthropic-to-openai"

        if status >= 400:
            self._send_raw(
                status,
                body,
                content_type=resp_headers.get("content-type", "application/json"),
                extra_headers=extra,
            )
            return

        try:
            oai_json = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self._send_json(
                502,
                {"type": "error", "error": {"type": "api_error", "message": f"Invalid upstream JSON: {exc}"}},
                extra_headers=extra,
            )
            return

        anth_resp = openai_to_anthropic_response(oai_json, requested_model=str(payload.get("model") or ""))
        if not is_stream:
            self._send_json(200, anth_resp, extra_headers=extra)
            return

        self._send_anthropic_sse_from_response(anth_resp, extra)


def check_proxy_health(host: str = DEFAULT_PROXY_HOST, port: int = DEFAULT_PROXY_PORT, timeout: float = 1.0) -> dict[str, Any] | None:
    """Return `/health` payload if a local Space proxy is already running, else None."""
    conn: http.client.HTTPConnection | None = None
    try:
        conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
        conn.request("GET", "/health", headers={"User-Agent": f"space-cli/{__version__}"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status == 200:
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict) and data.get("service") == "space-local-proxy":
                return data
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return None


def push_proxy_policy(
    policy_updates: dict[str, Any],
    host: str = DEFAULT_PROXY_HOST,
    port: int = DEFAULT_PROXY_PORT,
    timeout: float = 1.5,
) -> bool:
    """Update routing policy on an already-running local proxy instance."""
    conn: http.client.HTTPConnection | None = None
    try:
        body = json.dumps(policy_updates).encode("utf-8")
        conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
        conn.request(
            "POST",
            "/_space/policy",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        resp.read()
        return resp.status == 200
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


__all__ = [
    "ProxyConfig",
    "UpstreamState",
    "UpstreamPool",
    "SpaceLocalProxy",
    "anthropic_to_openai_request",
    "openai_to_anthropic_response",
    "check_proxy_health",
    "push_proxy_policy",
]
