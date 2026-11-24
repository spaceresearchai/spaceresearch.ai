"""Auditable Private AI Security & Anti-MITM Test Suite (`SIP-0005`).

Run directly from a fresh git checkout to verify Space AI's client & worker
privacy guarantees without trusting any hosted service:

    python3 -m unittest tests/test_private_ai_hardening.py -v
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.e2ee import (
    E2EE_PROTOCOL_VERSION,
    E2EEVerificationError,
    WorkerE2EEKeypair,
    b58encode,
    e2ee_available,
    encrypt_client_request,
    hash_private_pool_secret,
    is_loopback_url,
    verify_e2ee_pubkey_binding,
)
from cli.launchers import LaunchOptions, build_launch_plan
from cli.proxy import ProxyConfig, SpaceLocalProxy
from cli.space_cli import main as space_main


def _load_join_module():
    spec = importlib.util.spec_from_file_location("space_join_worker", REPO_ROOT / "scripts" / "join.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _http_req(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = dict(headers or {})
    if data is not None:
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        code = exc.code
        hdrs_out = {k.lower(): v for k, v in exc.headers.items()}
        exc.close()
        return code, hdrs_out, body


class _FakeRunner:
    def __init__(self, reply_prefix: str = "SECRET_ANSWER: "):
        self.reply_prefix = reply_prefix
        self.seen_prompts: list[str] = []

    def run(self, prompt: str, engine: str = "corekv"):
        self.seen_prompts.append(prompt)
        out = f"{self.reply_prefix}{prompt[::-1]}"
        return out, [-0.05] * 8, [0.1] * 8, max(1, len(out.split()))


class TestAirGapAndLoopbackValidation(unittest.TestCase):
    def test_is_loopback_url_strict_parser(self) -> None:
        self.assertTrue(is_loopback_url("http://127.0.0.1:8081"))
        self.assertTrue(is_loopback_url("http://localhost:11434/v1"))
        self.assertTrue(is_loopback_url("http://[::1]:8377"))
        self.assertTrue(is_loopback_url("https://127.0.0.2:8443"))

        self.assertFalse(is_loopback_url("https://api.spaceresearch.ai"))
        self.assertFalse(is_loopback_url("http://127.0.0.1.evil.com:8080"))
        self.assertFalse(is_loopback_url("http://localhost.evil.com:8080"))
        self.assertFalse(is_loopback_url("http://127.0.0.1@evil.com/v1"))
        self.assertFalse(is_loopback_url("file:///etc/passwd"))
        self.assertFalse(is_loopback_url(""))

    def test_local_only_airgap_blocks_remote_egress_and_non_loopback_bind(self) -> None:
        # 1. Refuses non-loopback bind host under local-only
        bad_bind = SpaceLocalProxy(
            ProxyConfig(
                host="0.0.0.0",
                port=0,
                upstreams=["http://127.0.0.1:11434"],
                privacy="local-only",
                quiet=True,
            )
        )
        with self.assertRaises(ValueError):
            bad_bind.start(background=True)

        # 2. Refuses --direct launcher against remote coordinator under local-only
        with self.assertRaises(ValueError):
            build_launch_plan(
                LaunchOptions(
                    agent="claude",
                    coordinator="https://api.spaceresearch.ai",
                    direct=True,
                    privacy="local-only",
                )
            )

        # 3. Fails closed (503) when only remote upstreams exist — zero external sockets opened
        proxy = SpaceLocalProxy(
            ProxyConfig(
                host="127.0.0.1",
                port=0,
                upstreams=["https://api.spaceresearch.ai", "https://api.longinusresearch.com"],
                privacy="local-only",
                quiet=True,
            )
        )
        proxy.start(background=True)
        try:
            status, hdrs, body = _http_req(
                f"{proxy.base_url}/v1/chat/completions",
                method="POST",
                payload={"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "TOP_SECRET_LOCAL"}]},
            )
            self.assertEqual(status, 503)
            err = json.loads(body.decode("utf-8"))
            self.assertEqual(err["error"]["code"], "local_only_airgap_blocked")
            self.assertEqual(hdrs.get("x-space-privacy-applied"), "local-only")
            self.assertEqual(hdrs.get("x-space-privacy-strict"), "true")
            self.assertEqual(hdrs.get("x-space-privacy-matched"), "false")
            self.assertEqual(proxy.stats["total_airgap_blocks"], 1)

            # count_tokens must also stay strictly local (never hit upstream)
            c_status, c_hdrs, c_body = _http_req(
                f"{proxy.base_url}/v1/messages/count_tokens",
                method="POST",
                payload={"messages": [{"role": "user", "content": "TOP_SECRET_LOCAL"}]},
            )
            self.assertEqual(c_status, 200)
            self.assertGreater(json.loads(c_body.decode("utf-8"))["input_tokens"], 0)
            self.assertEqual(c_hdrs.get("x-space-privacy-applied"), "local-only")
        finally:
            proxy.stop()

    def test_local_only_routes_cleanly_to_loopback_backend(self) -> None:
        class _LocalHandler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or "0")
                _ = self.rfile.read(n)
                raw = json.dumps(
                    {
                        "id": "chatcmpl-local",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "local-qwen",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "LOCAL_OK"}, "finish_reason": "stop"}],
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _LocalHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        local_url = f"http://127.0.0.1:{srv.server_address[1]}"
        proxy = SpaceLocalProxy(
            ProxyConfig(
                host="127.0.0.1",
                port=0,
                # Even if remote coordinator is listed FIRST, local-only must ignore it and use loopback only!
                upstreams=["https://api.spaceresearch.ai", local_url],
                privacy="local-only",
                quiet=True,
            )
        )
        proxy.start(background=True)
        try:
            status, _, body = _http_req(
                f"{proxy.base_url}/v1/chat/completions",
                method="POST",
                payload={"model": "local-qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body.decode("utf-8"))["choices"][0]["message"]["content"], "LOCAL_OK")
        finally:
            proxy.stop()
            srv.shutdown()
            srv.server_close()


@unittest.skipUnless(e2ee_available(), "cryptography package required for E2EE tests")
class TestE2EEBlindRelayAndAntiMITM(unittest.TestCase):
    def _gen_ed25519_worker(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        ed_priv = Ed25519PrivateKey.generate()
        pub_bytes = ed_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return ed_priv, b58encode(pub_bytes)

    def test_e2ee_crypto_roundtrip_tamper_and_mitm_rejection(self) -> None:
        ed_priv, wallet = self._gen_ed25519_worker()
        wk = WorkerE2EEKeypair()
        ann = wk.signed_announcement(ed_priv, wallet)
        self.assertTrue(verify_e2ee_pubkey_binding(wallet, ann["e2ee_pubkey_b64"], ann["e2ee_pubkey_sig_hex"]))

        # MITM attack: attacker substitutes their own X25519 key while keeping worker's wallet & signature
        attacker_wk = WorkerE2EEKeypair()
        self.assertFalse(verify_e2ee_pubkey_binding(wallet, attacker_wk.pubkey_b64, ann["e2ee_pubkey_sig_hex"]))
        with self.assertRaises(E2EEVerificationError):
            encrypt_client_request(
                {"prompt": "my-secret"},
                worker_wallet=wallet,
                worker_e2ee_pubkey_b64=attacker_wk.pubkey_b64,
                worker_e2ee_sig_hex=ann["e2ee_pubkey_sig_hex"],
            )

        # Honest encryption & decryption
        env, sess = encrypt_client_request(
            {"prompt": "my-secret-api-key-12345"},
            worker_wallet=wallet,
            worker_e2ee_pubkey_b64=ann["e2ee_pubkey_b64"],
            worker_e2ee_sig_hex=ann["e2ee_pubkey_sig_hex"],
        )
        self.assertNotIn("my-secret-api-key-12345", json.dumps(env))
        dec, shared, req_id = wk.decrypt_request_envelope(env, expected_wallet=wallet)
        self.assertEqual(dec["prompt"], "my-secret-api-key-12345")

        # Tampered ciphertext fails AES-256-GCM tag verification
        tampered = dict(env)
        tampered["ciphertext_b64"] = "A" + env["ciphertext_b64"][1:] if env["ciphertext_b64"][0] != "A" else "B" + env["ciphertext_b64"][1:]
        with self.assertRaises(E2EEVerificationError):
            wk.decrypt_request_envelope(tampered, expected_wallet=wallet)

        # Response round-trip + replay/request_id mismatch check
        resp_env = wk.encrypt_response_envelope(
            {"output_text": "confidential-output"},
            shared_secret=shared,
            request_id=req_id,
            wallet=wallet,
        )
        self.assertNotIn("confidential-output", json.dumps(resp_env))
        self.assertEqual(sess.decrypt_response(resp_env)["output_text"], "confidential-output")

        bad_resp = dict(resp_env, request_id="e2ee_different")
        with self.assertRaises(E2EEVerificationError):
            sess.decrypt_response(bad_resp)

    def test_worker_private_only_refuses_plaintext_and_handles_e2ee_blind_relay(self) -> None:
        join_mod = _load_join_module()
        ed_priv, wallet = self._gen_ed25519_worker()
        worker_e2ee = join_mod._WorkerE2EE()
        ann = worker_e2ee.signed_announcement(ed_priv, wallet)
        runner = _FakeRunner(reply_prefix="CONFIDENTIAL_REPLY:")

        # 1. Worker in --private-only mode MUST refuse plaintext jobs
        with self.assertRaises(PermissionError):
            join_mod.process_worker_job(
                {"job_id": "job-plain-1", "prompt": "unencrypted prompt"},
                runner=runner,
                priv=ed_priv,
                wallet=wallet,
                model_id="qwen3.8-27b",
                worker_e2ee=worker_e2ee,
                private_only=True,
            )
        self.assertEqual(len(runner.seen_prompts), 0)

        # 2. Full end-to-end blind relay through SpaceLocalProxy -> Mock Coordinator -> join.py worker
        secret_prompt = "SUPER_SECRET_SOURCE_CODE_TOKEN_99421"
        captured_coordinator_wire: list[bytes] = []
        mitm_mode = {"enabled": False}
        attacker_pk = WorkerE2EEKeypair().pubkey_b64

        class _BlindRelayHandler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path.startswith("/v1/nodes/e2ee-keys"):
                    pk_to_serve = attacker_pk if mitm_mode["enabled"] else ann["e2ee_pubkey_b64"]
                    raw = json.dumps(
                        {
                            "e2ee_protocol": E2EE_PROTOCOL_VERSION,
                            "count": 1,
                            "nodes": [
                                {
                                    "wallet": wallet,
                                    "display_name": "my-private-5090",
                                    "model_id": "qwen3.8-27b",
                                    "isolation": "hardened-docker",
                                    "private_only": True,
                                    "e2ee_protocol": E2EE_PROTOCOL_VERSION,
                                    "e2ee_pubkey_b64": pk_to_serve,
                                    "e2ee_pubkey_sig_hex": ann["e2ee_pubkey_sig_hex"],
                                }
                            ],
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or "0")
                raw_req = self.rfile.read(n)
                captured_coordinator_wire.append(raw_req)
                if self.path == "/v1/e2ee/infer":
                    body = json.loads(raw_req.decode("utf-8"))
                    comp, _, _ = join_mod.process_worker_job(
                        {
                            "job_id": body["request_id"],
                            "engine": "corekv",
                            "e2ee_envelope": body["e2ee_envelope"],
                        },
                        runner=runner,
                        priv=ed_priv,
                        wallet=wallet,
                        model_id="qwen3.8-27b",
                        worker_e2ee=worker_e2ee,
                        private_only=True,
                    )
                    # Verify worker returned empty plaintext output_text & ciphertext hashes only
                    assert comp["output_text"] == ""
                    assert comp["receipt"]["prompt_hash"] != hashlib.sha256(secret_prompt.encode()).hexdigest()
                    out_raw = json.dumps(
                        {
                            "job_id": body["request_id"],
                            "request_id": body["request_id"],
                            "target_node": wallet,
                            "e2ee_protocol": E2EE_PROTOCOL_VERSION,
                            "coordinator_blind": True,
                            "e2ee_response": comp["e2ee_response"],
                            "receipt": comp["receipt"],
                        }
                    ).encode("utf-8")
                    captured_coordinator_wire.append(out_raw)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out_raw)))
                    self.end_headers()
                    self.wfile.write(out_raw)
                    return
                self.send_response(400)
                self.end_headers()

        relay_srv = ThreadingHTTPServer(("127.0.0.1", 0), _BlindRelayHandler)
        threading.Thread(target=relay_srv.serve_forever, daemon=True).start()
        relay_url = f"http://127.0.0.1:{relay_srv.server_address[1]}"

        proxy = SpaceLocalProxy(
            ProxyConfig(
                host="127.0.0.1",
                port=0,
                upstreams=[relay_url],
                privacy="e2ee",
                private_node=wallet,
                private_pool_secret="my-secret-team-pool",
                quiet=True,
            )
        )
        proxy.start(background=True)
        try:
            # OpenAI /v1/chat/completions over E2EE
            st, hdrs, body = _http_req(
                f"{proxy.base_url}/v1/chat/completions",
                method="POST",
                payload={"model": "qwen3.8-27b", "messages": [{"role": "user", "content": secret_prompt}]},
            )
            self.assertEqual(st, 200)
            self.assertEqual(hdrs.get("x-space-privacy-applied"), "e2ee")
            self.assertEqual(hdrs.get("x-space-e2ee-protocol"), E2EE_PROTOCOL_VERSION)
            oai_out = json.loads(body.decode("utf-8"))
            expected_reply = f"CONFIDENTIAL_REPLY:{secret_prompt[::-1]}"
            self.assertEqual(oai_out["choices"][0]["message"]["content"], expected_reply)
            self.assertTrue(oai_out["space_privacy"]["coordinator_blind"])

            # Anthropic /v1/messages (Claude Code path) over E2EE SSE stream!
            st_a, hdrs_a, body_a = _http_req(
                f"{proxy.base_url}/v1/messages",
                method="POST",
                payload={
                    "model": "claude-sonnet-4",
                    "stream": True,
                    "messages": [{"role": "user", "content": secret_prompt}],
                },
            )
            self.assertEqual(st_a, 200)
            self.assertEqual(hdrs_a.get("x-space-privacy-applied"), "e2ee")
            sse_text = body_a.decode("utf-8")
            self.assertIn("event: message_start", sse_text)
            self.assertIn(expected_reply, sse_text)

            # AUDIT PROOF: Coordinator saw ZERO bytes of secret_prompt, expected_reply, or raw pool secret!
            wire_blob = b"".join(captured_coordinator_wire)
            self.assertNotIn(secret_prompt.encode("utf-8"), wire_blob)
            self.assertNotIn(expected_reply.encode("utf-8"), wire_blob)
            self.assertNotIn(b"my-secret-team-pool", wire_blob)

            # MITM PROOF: Coordinator tries to substitute worker's X25519 pubkey -> Proxy MUST fail closed (503)!
            mitm_mode["enabled"] = True
            prev_len = len(captured_coordinator_wire)
            st_m, _, body_m = _http_req(
                f"{proxy.base_url}/v1/chat/completions",
                method="POST",
                payload={"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "NEVER_SEND_ON_MITM"}]},
            )
            self.assertEqual(st_m, 503)
            self.assertEqual(json.loads(body_m.decode("utf-8"))["error"]["code"], "e2ee_key_verification_failed")
            # Zero POST calls sent to /v1/e2ee/infer during MITM!
            self.assertEqual(len(captured_coordinator_wire), prev_len)
        finally:
            proxy.stop()
            relay_srv.shutdown()
            relay_srv.server_close()


class TestCLIPrivacyAuditAndBundleIntegrity(unittest.TestCase):
    def test_space_privacy_check_and_pool_secret_hashing(self) -> None:
        h1 = hash_private_pool_secret("team-alpha")
        self.assertEqual(len(h1), 32)
        self.assertNotIn("team-alpha", h1)
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = space_main(["privacy-check", "--json"], home_dir=Path(tmp))
            self.assertEqual(rc, 0)
            rep = json.loads(buf.getvalue())
            self.assertEqual(rep["e2ee_protocol"], E2EE_PROTOCOL_VERSION)
            self.assertIn("local-only", rep["fail_closed_tiers"])
            self.assertIn("e2ee", rep["fail_closed_tiers"])
            self.assertIn("tee", rep["fail_closed_tiers"])

    def test_tarball_checksum_and_source_parity(self) -> None:
        sha_line = (REPO_ROOT / "space-cli.sha256").read_text(encoding="utf-8").strip()
        expected_hash = sha_line.split()[0]
        actual_hash = hashlib.sha256((REPO_ROOT / "space-cli.tar.gz").read_bytes()).hexdigest()
        self.assertEqual(actual_hash, expected_hash)

        with tarfile.open(REPO_ROOT / "space-cli.tar.gz", "r:gz") as tf:
            names = tf.getnames()
            self.assertIn("cli/e2ee.py", names)
            self.assertIn("cli/proxy.py", names)
            self.assertIn("scripts/join.py", names)
            for member in tf.getmembers():
                if member.isfile():
                    extracted = tf.extractfile(member).read()
                    on_disk = (REPO_ROOT / member.name).read_bytes()
                    self.assertEqual(extracted, on_disk, f"Tarball member {member.name} drifted from repo source!")


if __name__ == "__main__":
    unittest.main()
