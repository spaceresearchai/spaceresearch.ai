"""Comprehensive verification suite for Space AI CLI, Unified Local Proxy, Launchers & Adapters.

Runs cleanly with either `python3 tests/test_space_cli_proxy.py` or `python3 -m pytest tests/test_space_cli_proxy.py`.
Uses live local mock HTTP servers (zero external network calls, zero root required, isolated temp HOME dirs).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli import (
    load_space_config,
    mask_secret,
    resolve_barter_key,
    save_space_config,
)
from cli.adapters import (
    ADAPTER_CLASSES,
    AdapterBridgeWorker,
    NativeAdapter,
    OllamaAdapter,
    ProviderAdapter,
    format_adapters_table,
    get_adapter,
    probe_all_adapters,
)
from cli.launchers import LaunchOptions, build_launch_plan, format_dry_run
from cli.proxy import (
    ProxyConfig,
    SpaceLocalProxy,
    anthropic_to_openai_request,
    check_proxy_health,
    openai_to_anthropic_response,
    push_proxy_policy,
)
from cli.space_cli import main as space_cli_main


# ---------------------------------------------------------------------------
# Live Local Mock Upstream & Coordinator Server
# ---------------------------------------------------------------------------


class _MockServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, handler_cls: type[BaseHTTPRequestHandler], state: dict[str, Any]):
        self.state = state
        super().__init__(("127.0.0.1", 0), handler_cls)


class _MockUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        cl = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(cl) if cl > 0 else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _send_json(self, status: int, data: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _record_request(self, method: str, path: str, body: dict[str, Any] | None = None) -> None:
        st = self.server.state  # type: ignore[attr-defined]
        with st["lock"]:
            st["requests"].append(
                {
                    "method": method,
                    "path": path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
            )

    def _space_echo_headers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.headers.get("X-Space-Route"):
            out["X-Space-Route-Applied"] = self.headers["X-Space-Route"]
        if self.headers.get("X-Space-Privacy"):
            out["X-Space-Privacy-Applied"] = self.headers["X-Space-Privacy"]
        if self.headers.get("X-Space-Region"):
            out["X-Space-Region-Applied"] = self.headers["X-Space-Region"]
        return out

    def do_GET(self) -> None:
        st = self.server.state  # type: ignore[attr-defined]
        if st.get("delay_s", 0.0) > 0:
            time.sleep(st["delay_s"])
        self._record_request("GET", self.path)

        if st.get("fail_status"):
            self._send_json(int(st["fail_status"]), {"error": "injected failure"})
            return

        if self.path == "/health":
            self._send_json(200, {"status": "ok", "name": st.get("name", "mock")})
            return

        if self.path in ("/v1/models", "/api/tags"):
            if self.path == "/api/tags":
                self._send_json(200, {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3.2:3b"}]})
            else:
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "qwen/Qwen3.8-27B", "object": "model"},
                            {"id": "thinkingmachines/Inkling-2-Mini", "object": "model"},
                        ],
                    },
                    headers=self._space_echo_headers(),
                )
            return

        if self.path.startswith("/v1/jobs/next"):
            with st["lock"]:
                job = st["jobs_queue"].pop(0) if st["jobs_queue"] else None
            if job:
                self._send_json(200, {"has_job": True, "job": job})
            else:
                self._send_json(200, {"has_job": False, "job": None})
            return

        if self.path.startswith("/v1/sessions/budget/"):
            tok = self.path.split("/v1/sessions/budget/", 1)[1]
            sess = st["sessions"].get(tok)
            if not sess:
                self._send_json(404, {"error": "session not found"})
                return
            self._send_json(200, sess)
            return

        self._send_json(404, {"error": f"not found: {self.path}"})

    def do_POST(self) -> None:
        st = self.server.state  # type: ignore[attr-defined]
        if st.get("delay_s", 0.0) > 0:
            time.sleep(st["delay_s"])
        body = self._read_json()
        self._record_request("POST", self.path, body)

        if st.get("fail_status"):
            self._send_json(int(st["fail_status"]), {"error": "injected failure"})
            return

        echo = self._space_echo_headers()

        if self.path == "/v1/nodes/register":
            with st["lock"]:
                st["registrations"].append(body)
            self._send_json(
                200,
                {
                    "status": "active",
                    "wallet": body.get("wallet"),
                    "api_key": "lng_live_bridge_auto_1234567890",
                    "credit_tokens": 50000,
                },
            )
            return

        if self.path.startswith("/v1/jobs/") and self.path.endswith("/complete"):
            job_id = self.path.split("/v1/jobs/", 1)[1].rsplit("/complete", 1)[0]
            with st["lock"]:
                st["completions"].append({"job_id": job_id, **body})
            self._send_json(200, {"status": "recorded", "job_id": job_id})
            return

        if self.path == "/v1/sessions/budget":
            tok = f"sp_sess_mock_{len(st['sessions']) + 1:04d}"
            max_usdc = float(body.get("max_usdc", 5.0))
            sess = {
                "session_token": tok,
                "owner_id": body.get("owner_id") or "anon",
                "rail": body.get("rail") or "solana_usdc",
                "max_usdc": max_usdc,
                "spent_usdc": 0.0,
                "remaining_usdc": max_usdc,
                "active": True,
                "expires_at": "2026-12-01T00:00:00Z",
            }
            st["sessions"][tok] = sess
            self._send_json(200, sess)
            return

        if self.path == "/v1/chat/completions":
            model = body.get("model") or "qwen/Qwen3.8-27B"
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                for k, v in echo.items():
                    self.send_header(k, v)
                self.end_headers()
                chunks = [
                    {"id": "chatcmpl-stream-1", "model": model, "choices": [{"index": 0, "delta": {"content": "Hello "}}]},
                    {"id": "chatcmpl-stream-1", "model": model, "choices": [{"index": 0, "delta": {"content": "Space!"}}]},
                ]
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            self._send_json(
                200,
                {
                    "id": f"chatcmpl-{st.get('name', 'mock')}",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"hello from {st.get('name', 'mock')}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                },
                headers=echo,
            )
            return

        if self.path == "/v1/responses":
            self._send_json(
                200,
                {"id": "resp_mock_1", "object": "response", "output_text": f"codex from {st.get('name', 'mock')}"},
                headers=echo,
            )
            return

        if self.path == "/v1/messages":
            if not st.get("supports_anthropic", True):
                self._send_json(404, {"error": "anthropic endpoint not supported on OpenAI-only upstream"})
                return
            self._send_json(
                200,
                {
                    "id": f"msg_native_{st.get('name', 'mock')}",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model") or "qwen/Qwen3.8-27B",
                    "content": [{"type": "text", "text": f"native anthropic from {st.get('name', 'mock')}"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 12, "output_tokens": 9},
                },
                headers=echo,
            )
            return

        self._send_json(404, {"error": f"not found: {self.path}"})


def _start_mock_server(
    name: str = "mock",
    *,
    fail_status: int | None = None,
    supports_anthropic: bool = True,
    delay_s: float = 0.0,
) -> tuple[_MockServer, str, dict[str, Any]]:
    state: dict[str, Any] = {
        "name": name,
        "fail_status": fail_status,
        "supports_anthropic": supports_anthropic,
        "delay_s": delay_s,
        "lock": threading.Lock(),
        "requests": [],
        "registrations": [],
        "jobs_queue": [],
        "completions": [],
        "sessions": {},
    }
    srv = _MockServer(_MockUpstreamHandler, state)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    return srv, url, state


def _http_call(
    url: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req_headers = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            hdrs = {k.lower(): v for k, v in resp.getheaders()}
            return int(resp.status), hdrs, resp.read()
    except urllib.error.HTTPError as exc:
        hdrs = {k.lower(): v for k, v in exc.headers.items()}
        return int(exc.code), hdrs, exc.read()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestConfigAndBarterKey(unittest.TestCase):
    def test_barter_key_discovery_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            # 1. No config -> None
            key, src = resolve_barter_key(home_dir=home, env={})
            self.assertIsNone(key)
            self.assertEqual(src, "none")

            # 2. ~/.longinus_node.json fallback
            (home / ".longinus_node.json").write_text(
                json.dumps({"wallet": "TestWallet111", "api_key": "lng_live_node_secret_987654"}),
                encoding="utf-8",
            )
            key, src = resolve_barter_key(home_dir=home, env={})
            self.assertEqual(key, "lng_live_node_secret_987654")
            self.assertEqual(src, "~/.longinus_node.json")

            # 3. ~/.space_ai/config.json overrides ~/.longinus_node.json
            save_space_config({"barter_key": "lng_live_space_cfg_112233"}, home_dir=home)
            key, src = resolve_barter_key(home_dir=home, env={})
            self.assertEqual(key, "lng_live_space_cfg_112233")
            self.assertEqual(src, "~/.space_ai/config.json")

            # 4. Env var overrides files
            key, src = resolve_barter_key(home_dir=home, env={"SPACE_BARTER_KEY": "lng_live_env_556677"})
            self.assertEqual(key, "lng_live_env_556677")
            self.assertEqual(src, "env:SPACE_BARTER_KEY")

            # 5. Secret redaction
            masked = mask_secret("lng_live_space_cfg_112233")
            self.assertIsNotNone(masked)
            self.assertNotIn("space_cfg", masked or "")
            self.assertTrue((masked or "").startswith("lng_live_s"))
            self.assertTrue((masked or "").endswith("2233"))


class TestUnifiedLocalProxy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.home = Path(self.tmpdir.name)
        save_space_config({"barter_key": "lng_live_saved_barter_key_9999"}, home_dir=self.home)
        self.servers: list[_MockServer] = []
        self.proxies: list[SpaceLocalProxy] = []

    def tearDown(self) -> None:
        for p in self.proxies:
            p.stop()
        for s in self.servers:
            s.shutdown()
            s.server_close()
        self.tmpdir.cleanup()

    def _make_proxy(self, upstreams: list[str], **kwargs: Any) -> SpaceLocalProxy:
        cfg = ProxyConfig(
            host="127.0.0.1",
            port=0,
            upstreams=upstreams,
            home_dir=self.home,
            quiet=True,
            **kwargs,
        )
        proxy = SpaceLocalProxy(cfg)
        proxy.start(background=True)
        self.proxies.append(proxy)
        return proxy

    def test_openai_anthropic_headers_and_status(self) -> None:
        srv, url, st = _start_mock_server("primary", supports_anthropic=True)
        self.servers.append(srv)

        proxy = self._make_proxy(
            [url],
            route="verified-local-only",
            privacy="zero-log",
            region="eu",
            session_token="sp_sess_budget_abc123",
            client_tag="coding-agent",
        )

        # /health check
        health = check_proxy_health("127.0.0.1", proxy.port)
        self.assertIsNotNone(health)
        assert health is not None
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["upstreams_healthy"], 1)

        # GET /v1/models
        code, hdrs, raw = _http_call(f"{proxy.base_url}/v1/models")
        self.assertEqual(code, 200)
        models_json = json.loads(raw.decode("utf-8"))
        self.assertEqual(len(models_json["data"]), 2)
        self.assertEqual(hdrs.get("x-space-route-applied"), "verified-local-only")
        self.assertEqual(hdrs.get("x-space-privacy-applied"), "zero-log")
        self.assertEqual(hdrs.get("x-space-region-applied"), "eu")

        # POST /v1/chat/completions (non-streaming + placeholder key -> auto-inject saved barter key)
        code, hdrs, raw = _http_call(
            f"{proxy.base_url}/v1/chat/completions",
            method="POST",
            body={"model": "qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer space-local-proxy"},
        )
        self.assertEqual(code, 200)
        chat = json.loads(raw.decode("utf-8"))
        self.assertEqual(chat["choices"][0]["message"]["content"], "hello from primary")

        last_req = st["requests"][-1]
        self.assertEqual(last_req["headers"]["authorization"], "Bearer lng_live_saved_barter_key_9999")
        self.assertEqual(last_req["headers"]["x-space-route"], "verified-local-only")
        self.assertEqual(last_req["headers"]["x-space-privacy"], "zero-log")
        self.assertEqual(last_req["headers"]["x-space-region"], "eu")
        self.assertEqual(last_req["headers"]["x-space-session"], "sp_sess_budget_abc123")
        self.assertEqual(last_req["headers"]["x-space-client"], "coding-agent")

        # POST /v1/chat/completions (SSE streaming)
        code, hdrs, raw = _http_call(
            f"{proxy.base_url}/v1/chat/completions",
            method="POST",
            body={"model": "qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "stream"}], "stream": True},
        )
        self.assertEqual(code, 200)
        text = raw.decode("utf-8")
        self.assertIn("Hello ", text)
        self.assertIn("Space!", text)
        self.assertIn("[DONE]", text)

        # POST /v1/messages (native Anthropic forwarding + per-request route override)
        code, hdrs, raw = _http_call(
            f"{proxy.base_url}/v1/messages",
            method="POST",
            body={
                "model": "claude-sonnet-4",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "write rust"}],
            },
            headers={"X-Space-Route": "cheapest", "anthropic-version": "2023-06-01"},
        )
        self.assertEqual(code, 200)
        anth = json.loads(raw.decode("utf-8"))
        self.assertEqual(anth["type"], "message")
        self.assertEqual(anth["content"][0]["text"], "native anthropic from primary")
        self.assertEqual(hdrs.get("x-space-route-applied"), "cheapest")
        self.assertEqual(st["requests"][-1]["headers"]["anthropic-version"], "2023-06-01")

        # POST /v1/messages/count_tokens fallback when upstream returns 404
        code, _, raw = _http_call(
            f"{proxy.base_url}/v1/messages/count_tokens",
            method="POST",
            body={"messages": [{"role": "user", "content": "count these tokens please"}]},
        )
        self.assertEqual(code, 200)
        self.assertGreater(json.loads(raw.decode("utf-8"))["input_tokens"], 0)

        # Runtime policy update via /_space/policy
        self.assertTrue(push_proxy_policy({"route": "cheapest", "region": "us"}, "127.0.0.1", proxy.port))

        # /status inspection (masked secrets + latency EMA populated)
        code, _, raw = _http_call(f"{proxy.base_url}/status")
        self.assertEqual(code, 200)
        status = json.loads(raw.decode("utf-8"))
        self.assertTrue(status["barter_key"]["configured"])
        self.assertNotIn("saved_barter_key", status["barter_key"]["masked"])
        self.assertEqual(status["routing_policy"]["route"], "cheapest")
        self.assertEqual(status["routing_policy"]["region"], "us")
        self.assertGreaterEqual(status["stats"]["total_openai_calls"], 2)
        self.assertGreaterEqual(status["stats"]["total_anthropic_calls"], 1)
        self.assertIsNotNone(status["upstreams"][0]["latency_ema_ms"])

    def test_anthropic_to_openai_translation_json_and_sse(self) -> None:
        # Upstream does NOT support /v1/messages (returns 404), only OpenAI /v1/chat/completions
        srv, url, st = _start_mock_server("openai-only", supports_anthropic=False)
        self.servers.append(srv)
        proxy = self._make_proxy([url])

        # Non-streaming Anthropic request translated cleanly to OpenAI and back
        code, hdrs, raw = _http_call(
            f"{proxy.base_url}/v1/messages",
            method="POST",
            body={
                "model": "claude-sonnet-4",
                "system": [{"type": "text", "text": "You are concise."}],
                "max_tokens": 256,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Ping"}]}],
            },
        )
        self.assertEqual(code, 200)
        self.assertEqual(hdrs.get("x-space-proxy-translated"), "anthropic-to-openai")
        anth = json.loads(raw.decode("utf-8"))
        self.assertEqual(anth["type"], "message")
        self.assertEqual(anth["role"], "assistant")
        self.assertEqual(anth["stop_reason"], "end_turn")
        self.assertEqual(anth["content"][0]["text"], "hello from openai-only")
        self.assertEqual(anth["usage"], {"input_tokens": 11, "output_tokens": 7})
        # Verify system message & coding-agent header reached OpenAI upstream
        oai_req = st["requests"][-1]
        self.assertEqual(oai_req["path"], "/v1/chat/completions")
        self.assertEqual(oai_req["headers"]["x-space-client"], "coding-agent")
        self.assertEqual(oai_req["body"]["messages"][0]["role"], "system")

        # Streaming Anthropic request translated into valid Anthropic SSE events
        code, hdrs, raw = _http_call(
            f"{proxy.base_url}/v1/messages",
            method="POST",
            body={
                "model": "claude-sonnet-4",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "Stream ping"}],
            },
        )
        self.assertEqual(code, 200)
        sse_text = raw.decode("utf-8")
        self.assertIn("event: message_start", sse_text)
        self.assertIn("event: content_block_delta", sse_text)
        self.assertIn("hello from openai-only", sse_text)
        self.assertIn("event: message_stop", sse_text)

    def test_multi_upstream_failover_and_latency_ema(self) -> None:
        # Primary fails with HTTP 503; backup succeeds
        bad_srv, bad_url, _ = _start_mock_server("bad-primary", fail_status=503)
        good_srv, good_url, _ = _start_mock_server("healthy-backup")
        self.servers.extend([bad_srv, good_srv])

        proxy = self._make_proxy([bad_url, good_url], circuit_cooldown_s=30.0)

        # First call hits bad_url (503) -> automatically fails over to good_url (200)
        code, hdrs, raw = _http_call(
            f"{proxy.base_url}/v1/chat/completions",
            method="POST",
            body={"model": "qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "failover test"}]},
        )
        self.assertEqual(code, 200)
        self.assertEqual(hdrs.get("x-space-proxy-failovers"), "1")
        self.assertEqual(hdrs.get("x-space-proxy-upstream"), good_url)
        self.assertEqual(json.loads(raw.decode("utf-8"))["choices"][0]["message"]["content"], "hello from healthy-backup")

        # Second call goes directly to healthy-backup (circuit breaker opened on bad-primary -> 0 failovers)
        code, hdrs, _ = _http_call(
            f"{proxy.base_url}/v1/chat/completions",
            method="POST",
            body={"model": "qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "second call"}]},
        )
        self.assertEqual(code, 200)
        self.assertEqual(hdrs.get("x-space-proxy-failovers"), "0")
        self.assertEqual(hdrs.get("x-space-proxy-upstream"), good_url)

        # Verify status reflects failover stats & unhealthy primary
        _, _, raw_st = _http_call(f"{proxy.base_url}/status")
        st = json.loads(raw_st.decode("utf-8"))
        self.assertEqual(st["stats"]["total_failovers"], 1)
        by_url = {u["url"]: u for u in st["upstreams"]}
        self.assertFalse(by_url[bad_url]["healthy"])
        self.assertEqual(by_url[bad_url]["requests_failed"], 1)
        self.assertTrue(by_url[good_url]["healthy"])
        self.assertEqual(by_url[good_url]["requests_ok"], 2)
        self.assertIsNotNone(by_url[good_url]["latency_ema_ms"])

    def test_anthropic_openai_unit_translation_with_tools(self) -> None:
        oai = anthropic_to_openai_request(
            {
                "model": "claude-sonnet-4",
                "system": "Be brief",
                "messages": [
                    {"role": "user", "content": "Check file"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "print(1)"}],
                    },
                ],
                "tools": [{"name": "read_file", "description": "Read", "input_schema": {"type": "object"}}],
            }
        )
        self.assertEqual(oai["messages"][0]["role"], "system")
        self.assertEqual(oai["messages"][2]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(oai["messages"][3]["role"], "tool")
        self.assertEqual(oai["tools"][0]["function"]["name"], "read_file")

        anth = openai_to_anthropic_response(
            {
                "id": "cmpl_99",
                "model": "qwen3",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Calling tool",
                            "tool_calls": [
                                {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )
        self.assertEqual(anth["stop_reason"], "tool_use")
        self.assertEqual(anth["content"][0]["type"], "text")
        self.assertEqual(anth["content"][1]["type"], "tool_use")
        self.assertEqual(anth["content"][1]["input"], {"path": "a.py"})


class TestLaunchersAndCLI(unittest.TestCase):
    def test_claude_codex_opencode_dry_run_and_zero_bashrc_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bashrc = home / ".bashrc"
            bashrc.write_text("# original bashrc\n", encoding="utf-8")
            save_space_config({"barter_key": "lng_live_cli_barter_4242"}, home_dir=home)

            # Claude launcher via CLI main() with inline pass-through flags (no `--` needed)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = space_cli_main(
                    [
                        "claude",
                        "--dry-run",
                        "--route",
                        "verified-local-only",
                        "--privacy",
                        "hardened-docker",
                        "--region",
                        "us",
                        "--model",
                        "qwen/Qwen3.8-27B",
                        "--session",
                        "sp_sess_001",
                        "-p",
                        "refactor proxy",
                    ],
                    home_dir=home,
                )
            self.assertEqual(rc, 0)
            claude_plan = json.loads(buf.getvalue())
            self.assertEqual(claude_plan["agent"], "claude")
            self.assertEqual(claude_plan["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8377")
            self.assertEqual(claude_plan["env"]["ANTHROPIC_AUTH_TOKEN"], "lng_live_cli_barter_4242")
            self.assertEqual(claude_plan["env"]["ANTHROPIC_API_KEY"], "lng_live_cli_barter_4242")
            self.assertEqual(claude_plan["env"]["SPACE_CLIENT"], "coding-agent")
            self.assertEqual(claude_plan["env"]["SPACE_ROUTE"], "verified-local-only")
            self.assertEqual(claude_plan["env"]["SPACE_PRIVACY"], "hardened-docker")
            self.assertEqual(claude_plan["env"]["SPACE_REGION"], "us")
            self.assertEqual(claude_plan["env"]["SPACE_SESSION"], "sp_sess_001")
            self.assertEqual(claude_plan["command"], ["claude", "-p", "refactor proxy"])
            self.assertIn("Rate-limit fallback", claude_plan["rate_limit_note"])
            self.assertFalse(claude_plan["config_mutated"])

            # Codex launcher dry-run (shell format)
            codex_plan = build_launch_plan(
                LaunchOptions(agent="codex", agent_args=["--approval-mode", "suggest"], home_dir=home)
            )
            self.assertEqual(codex_plan.env_overrides["OPENAI_BASE_URL"], "http://127.0.0.1:8377/v1")
            self.assertEqual(codex_plan.env_overrides["OPENAI_API_KEY"], "lng_live_cli_barter_4242")
            shell_out = format_dry_run(codex_plan, fmt="shell")
            self.assertIn("OPENAI_BASE_URL=http://127.0.0.1:8377/v1", shell_out)
            self.assertIn("codex --approval-mode suggest", shell_out)

            # OpenCode launcher dry-run with --coordinator --direct
            opencode_plan = build_launch_plan(
                LaunchOptions(
                    agent="opencode",
                    coordinator="https://coord.example.com",
                    direct=True,
                    model="qwen3",
                    home_dir=home,
                )
            )
            self.assertEqual(opencode_plan.env_overrides["OPENAI_BASE_URL"], "https://coord.example.com/v1")
            self.assertEqual(opencode_plan.env_overrides["OPENAI_API_BASE"], "https://coord.example.com/v1")
            self.assertEqual(opencode_plan.env_overrides["OPENCODE_MODEL"], "qwen3")

            # Confirm ~/.bashrc untouched!
            self.assertEqual(bashrc.read_text(encoding="utf-8"), "# original bashrc\n")

    def test_space_up_dry_run_native_and_ollama(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = space_cli_main(
                [
                    "up",
                    "--name",
                    "my-rig",
                    "--auto-pause",
                    "--adapter",
                    "native",
                    "--proxy-port",
                    "8377",
                    "--dry-run",
                ]
            )
        self.assertEqual(rc, 0)
        plan = json.loads(buf.getvalue())
        self.assertEqual(plan["proxy"]["bind"], "http://127.0.0.1:8377")
        self.assertEqual(plan["worker"]["adapter"], "native")
        self.assertEqual(plan["worker"]["multiplier"], 20.0)
        self.assertTrue(plan["worker"]["auto_pause"])
        self.assertIn("--auto-pause", plan["worker"]["command"])
        self.assertIn("--name", plan["worker"]["command"])
        self.assertIn("my-rig", plan["worker"]["command"])

        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            rc2 = space_cli_main(["up", "--adapter", "ollama", "--name", "mac-ollama", "--dry-run"])
        self.assertEqual(rc2, 0)
        plan2 = json.loads(buf2.getvalue())
        self.assertEqual(plan2["worker"]["adapter"], "ollama")
        self.assertEqual(plan2["worker"]["multiplier"], 1.0)
        self.assertIsNone(plan2["worker"]["command"])

    def test_space_session_create_and_status(self) -> None:
        srv, url, _ = _start_mock_server("coord-session")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = space_cli_main(
                        [
                            "session",
                            "create",
                            "--max-usdc",
                            "7.5",
                            "--rail",
                            "solana_usdc",
                            "--coordinator",
                            url,
                            "--json",
                        ],
                        home_dir=home,
                    )
                self.assertEqual(rc, 0)
                created = json.loads(buf.getvalue())
                token = created["session_token"]
                self.assertTrue(token.startswith("sp_sess_"))
                self.assertEqual(created["max_usdc"], 7.5)

                # Verify persisted to ~/.space_ai/config.json
                cfg = load_space_config(home)
                self.assertEqual(cfg["session_token"], token)

                # Verify `space session status` loads saved token automatically
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    rc2 = space_cli_main(["session", "status", "--coordinator", url, "--json"], home_dir=home)
                self.assertEqual(rc2, 0)
                st = json.loads(buf2.getvalue())
                self.assertEqual(st["session_token"], token)
                self.assertEqual(st["remaining_usdc"], 7.5)
        finally:
            srv.shutdown()
            srv.server_close()


class TestProviderAdaptersAndBridge(unittest.TestCase):
    def test_all_five_adapters_and_multiplier_tiers(self) -> None:
        expected = {
            "native": ("local://scripts/join.py", 20.0),
            "ollama": ("http://127.0.0.1:11434", 1.0),
            "llama-cpp": ("http://127.0.0.1:8081", 1.0),
            "vllm": ("http://127.0.0.1:8000", 1.0),
            "mlx": ("http://127.0.0.1:8080", 1.0),
        }
        self.assertEqual(set(ADAPTER_CLASSES.keys()), set(expected.keys()))
        for name, (endpoint, mult) in expected.items():
            ad = get_adapter(name, repo_root=REPO_ROOT)
            self.assertIsInstance(ad, ProviderAdapter)
            self.assertEqual(ad.default_endpoint, endpoint)
            self.assertEqual(ad.multiplier, mult)

        # Live probe with mock Ollama server + unreachable ports for others
        ollama_srv, ollama_url, _ = _start_mock_server("mock-ollama")
        try:
            results = probe_all_adapters(
                timeout=0.5,
                endpoints={
                    "ollama": ollama_url,
                    "llama-cpp": "http://127.0.0.1:1",
                    "vllm": "http://127.0.0.1:1",
                    "mlx": "http://127.0.0.1:1",
                },
                repo_root=REPO_ROOT,
            )
            by_name = {r.name: r for r in results}
            self.assertTrue(by_name["native"].detected)
            self.assertEqual(by_name["native"].multiplier, 20.0)
            self.assertTrue(by_name["native"].recommended)
            self.assertEqual(by_name["ollama"].status, "online")
            self.assertIn("qwen2.5-coder:7b", by_name["ollama"].models)
            self.assertIsNotNone(by_name["ollama"].latency_ms)

            table = format_adapters_table(results)
            self.assertIn("20x Genesis High-Density Tier", table)
            self.assertIn("Standard Community Adapter", table)
            self.assertIn("ollama", table)
        finally:
            ollama_srv.shutdown()
            ollama_srv.server_close()

    def test_adapter_bridge_worker_registers_and_completes_job(self) -> None:
        coord_srv, coord_url, coord_st = _start_mock_server("mock-coord")
        ollama_srv, ollama_url, _ = _start_mock_server("mock-ollama")
        try:
            coord_st["jobs_queue"].append(
                {"job_id": "job_adapter_42", "prompt": "Say hello", "engine": "ollama", "model_id": "qwen2.5-coder:7b"}
            )
            with tempfile.TemporaryDirectory() as tmp:
                adapter = OllamaAdapter(endpoint=ollama_url, repo_root=REPO_ROOT)
                worker = AdapterBridgeWorker(
                    adapter=adapter,
                    coordinator=coord_url,
                    display_name="test-ollama-rig",
                    home_dir=Path(tmp),
                    poll_wait_s=1,
                )
                reg = worker.register()
                self.assertEqual(reg["status"], "active")
                self.assertTrue(worker.registered)
                self.assertEqual(worker.barter_key, "lng_live_bridge_auto_1234567890")
                # Also persisted barter key automatically!
                self.assertEqual(load_space_config(Path(tmp))["barter_key"], "lng_live_bridge_auto_1234567890")

                did_job = worker.poll_once()
                self.assertTrue(did_job)
                self.assertEqual(worker.jobs_completed, 1)
                self.assertEqual(len(coord_st["completions"]), 1)
                comp = coord_st["completions"][0]
                self.assertEqual(comp["job_id"], "job_adapter_42")
                self.assertEqual(comp["output_text"], "hello from mock-ollama")
                self.assertTrue(comp["receipt"]["signature_hex"])
        finally:
            coord_srv.shutdown()
            coord_srv.server_close()
            ollama_srv.shutdown()
            ollama_srv.server_close()


class TestExecutableAndInstaller(unittest.TestCase):
    def test_scripts_space_executable_and_zero_root_installer(self) -> None:
        space_script = REPO_ROOT / "scripts" / "space"
        installer = REPO_ROOT / "scripts" / "install-space.sh"
        self.assertTrue(os.access(space_script, os.X_OK), "scripts/space must be executable")
        self.assertTrue(os.access(installer, os.X_OK), "scripts/install-space.sh must be executable")

        # Run scripts/space directly
        res = subprocess.run([str(space_script), "adapters", "--json"], capture_output=True, text=True, check=True)
        parsed = json.loads(res.stdout)
        self.assertEqual(len(parsed["adapters"]), 5)

        # Test zero-root installer in isolated temp HOME (both symlink and --copy modes)
        for extra_args in ([], ["--copy"]):
            with tempfile.TemporaryDirectory() as tmp:
                env = dict(os.environ)
                env["HOME"] = tmp
                env["SPACE_BIN_DIR"] = f"{tmp}/.local/bin"
                env["SPACE_INSTALL_DIR"] = f"{tmp}/.space_ai/app"
                env["SPACE_SOURCE_DIR"] = str(REPO_ROOT)

                inst = subprocess.run(
                    [str(installer), *extra_args],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("Installed space", inst.stdout)
                installed_bin = Path(tmp) / ".local" / "bin" / "space"
                self.assertTrue(installed_bin.exists())
                self.assertTrue(os.access(installed_bin, os.X_OK))
                self.assertFalse((Path(tmp) / ".bashrc").exists(), "installer must never mutate ~/.bashrc")

                # Execute installed ~/.local/bin/space
                dry = subprocess.run(
                    [str(installed_bin), "claude", "--dry-run", "--route", "cheapest"],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                dry_json = json.loads(dry.stdout)
                self.assertEqual(dry_json["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8377")
                self.assertEqual(dry_json["env"]["SPACE_ROUTE"], "cheapest")


if __name__ == "__main__":
    unittest.main(verbosity=2)
