"""Space AI Auditable End-to-End Encryption & Air-Gap Primitives (`space-e2ee-v1`).

Plain-source cryptographic layer for Private AI across the Space Research CLI,
local buyer proxy, and GPU/Apple Silicon workers:

  - Key Agreement : Ephemeral X25519 (per-request forward secrecy)
  - Key Derivation: HKDF-SHA256 with per-message random salt & domain-separated info
  - Cipher        : AES-256-GCM with authenticated request_id + worker wallet AAD
  - Anti-MITM     : Worker X25519 public key MUST be signed by the worker's Solana
                    Base58 Ed25519 identity key (`space-e2ee-v1:keybind:<wallet>:<pk>`)
  - Air-Gap Guard : Strict loopback-only URL validator for `X-Space-Privacy: local-only`

When E2EE is active, the Space coordinator acts strictly as a blind relay and
never sees plaintext prompts or completions.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any

E2EE_PROTOCOL_VERSION = "space-e2ee-v1"
KEY_BINDING_PREFIX = "space-e2ee-v1:keybind:"
PRIVATE_POOL_DOMAIN = "space-private-pool-v1:"

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_MAP = {ch: idx for idx, ch in enumerate(_B58_ALPHABET)}


class E2EEError(RuntimeError):
    """Base error for Space E2EE failures (always fail-closed)."""


class E2EEUnavailableError(E2EEError):
    """Raised when `cryptography` primitives are missing in this Python environment."""


class E2EEVerificationError(E2EEError):
    """Raised when worker key binding, AAD, or AEAD authentication fails."""


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out: list[str] = []
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(_B58_ALPHABET[rem])
    leading_zeros = len(data) - len(data.lstrip(b"\x00"))
    return ("1" * leading_zeros) + "".join(reversed(out))


def b58decode(text: str, expected_len: int | None = None) -> bytes:
    n = 0
    for ch in text:
        if ch not in _B58_MAP:
            raise ValueError(f"Invalid Base58 character: {ch!r}")
        n = n * 58 + _B58_MAP[ch]
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big") if n > 0 else b""
    leading_ones = len(text) - len(text.lstrip("1"))
    out = (b"\x00" * leading_ones) + raw
    if expected_len is not None:
        if len(out) < expected_len:
            out = (b"\x00" * (expected_len - len(out))) + out
        if len(out) != expected_len:
            raise ValueError(f"Expected {expected_len} bytes, got {len(out)}")
    return out


def _import_crypto():
    try:
        from cryptography.exceptions import InvalidSignature, InvalidTag
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
            X25519PublicKey,
        )
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        return {
            "InvalidSignature": InvalidSignature,
            "InvalidTag": InvalidTag,
            "hashes": hashes,
            "serialization": serialization,
            "Ed25519PrivateKey": Ed25519PrivateKey,
            "Ed25519PublicKey": Ed25519PublicKey,
            "X25519PrivateKey": X25519PrivateKey,
            "X25519PublicKey": X25519PublicKey,
            "AESGCM": AESGCM,
            "HKDF": HKDF,
        }
    except Exception as exc:
        raise E2EEUnavailableError(
            "Space E2EE requires the Python `cryptography` package (`pip install cryptography`). "
            "Refusing to fall back to plaintext."
        ) from exc


def e2ee_available() -> bool:
    try:
        _import_crypto()
        return True
    except E2EEUnavailableError:
        return False


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str, *, expected_len: int | None = None, field: str = "field") -> bytes:
    if not isinstance(text, str) or not text:
        raise E2EEVerificationError(f"Missing or invalid base64 {field}")
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except Exception as exc:
        raise E2EEVerificationError(f"Malformed base64 {field}") from exc
    if expected_len is not None and len(raw) != expected_len:
        raise E2EEVerificationError(
            f"Invalid {field} length: expected {expected_len} bytes, got {len(raw)}"
        )
    return raw


def key_binding_message(wallet: str, e2ee_pubkey_b64: str) -> bytes:
    """Canonical message signed by the worker's Ed25519 wallet key to bind its X25519 key."""
    return f"{KEY_BINDING_PREFIX}{wallet.strip()}:{e2ee_pubkey_b64.strip()}".encode("utf-8")


def sign_e2ee_pubkey(ed25519_priv: Any, wallet: str, e2ee_pubkey_b64: str) -> str:
    """Sign a worker's X25519 public key with its Ed25519 identity key (anti-MITM)."""
    msg = key_binding_message(wallet, e2ee_pubkey_b64)
    return ed25519_priv.sign(msg).hex()


def verify_e2ee_pubkey_binding(wallet: str, e2ee_pubkey_b64: str, signature_hex: str) -> bool:
    """Cryptographically verify that `e2ee_pubkey_b64` was signed by `wallet` (Ed25519)."""
    if not wallet or not e2ee_pubkey_b64 or not signature_hex:
        return False
    c = _import_crypto()
    try:
        x_bytes = _b64d(e2ee_pubkey_b64, expected_len=32, field="e2ee_pubkey_b64")
        c["X25519PublicKey"].from_public_bytes(x_bytes)
        ed_pub_bytes = b58decode(wallet.strip(), expected_len=32)
        sig_bytes = bytes.fromhex(signature_hex.strip())
        if len(sig_bytes) != 64:
            return False
        ed_pub = c["Ed25519PublicKey"].from_public_bytes(ed_pub_bytes)
        ed_pub.verify(sig_bytes, key_binding_message(wallet, e2ee_pubkey_b64))
        return True
    except (c["InvalidSignature"], ValueError, E2EEVerificationError):
        return False


def _derive_aead_key(shared_secret: bytes, *, direction: str, nonce: bytes, request_id: str, wallet: str) -> bytes:
    c = _import_crypto()
    info = f"{E2EE_PROTOCOL_VERSION}:{direction}:{request_id}:{wallet}".encode("utf-8")
    hkdf = c["HKDF"](
        algorithm=c["hashes"].SHA256(),
        length=32,
        salt=nonce,
        info=info,
    )
    return hkdf.derive(shared_secret)


def _aead_aad(*, direction: str, request_id: str, wallet: str) -> bytes:
    return f"{E2EE_PROTOCOL_VERSION}:{direction}:{request_id}:{wallet}".encode("utf-8")


def hash_private_pool_secret(secret: str) -> str:
    """Derive a non-reversible pool identifier from a shared private pool secret."""
    clean = (secret or "").strip()
    if not clean:
        return ""
    return hashlib.sha256((PRIVATE_POOL_DOMAIN + clean).encode("utf-8")).hexdigest()[:32]


def is_loopback_url(url: str) -> bool:
    """Return True strictly if `url` targets an HTTP/HTTPS loopback host (127.0.0.1, ::1, localhost)."""
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def ciphertext_sha256(envelope: dict[str, Any]) -> str:
    """Compute SHA-256 over the encrypted ciphertext (avoids hashing plaintext in E2EE receipts)."""
    raw = str(envelope.get("ciphertext_b64") or "").encode("ascii", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


class WorkerE2EEKeypair:
    """Ephemeral in-RAM X25519 keypair held by a Space GPU/Apple Silicon worker."""

    def __init__(self, private_key: Any | None = None):
        c = _import_crypto()
        self._priv = private_key or c["X25519PrivateKey"].generate()
        self._pub = self._priv.public_key()
        self._pub_bytes = self._pub.public_bytes(
            encoding=c["serialization"].Encoding.Raw,
            format=c["serialization"].PublicFormat.Raw,
        )

    @property
    def pubkey_b64(self) -> str:
        return _b64e(self._pub_bytes)

    def signed_announcement(self, ed25519_priv: Any, wallet: str) -> dict[str, str]:
        return {
            "e2ee_protocol": E2EE_PROTOCOL_VERSION,
            "e2ee_pubkey_b64": self.pubkey_b64,
            "e2ee_pubkey_sig_hex": sign_e2ee_pubkey(ed25519_priv, wallet, self.pubkey_b64),
        }

    def decrypt_request_envelope(
        self,
        envelope: dict[str, Any],
        *,
        expected_wallet: str,
    ) -> tuple[dict[str, Any], bytes, str]:
        """Decrypt a client's `space-e2ee-v1` request envelope in memory."""
        c = _import_crypto()
        if not isinstance(envelope, dict):
            raise E2EEVerificationError("E2EE envelope must be a JSON object")
        if envelope.get("v") != E2EE_PROTOCOL_VERSION:
            raise E2EEVerificationError(f"Unsupported E2EE version: {envelope.get('v')!r}")
        wallet = str(envelope.get("wallet") or "").strip()
        if wallet != expected_wallet:
            raise E2EEVerificationError("E2EE envelope target wallet mismatch")
        request_id = str(envelope.get("request_id") or "").strip()
        if not request_id:
            raise E2EEVerificationError("E2EE envelope missing request_id")

        client_epk_bytes = _b64d(
            str(envelope.get("client_epk_b64") or ""),
            expected_len=32,
            field="client_epk_b64",
        )
        nonce = _b64d(str(envelope.get("nonce_b64") or ""), expected_len=12, field="nonce_b64")
        ct = _b64d(str(envelope.get("ciphertext_b64") or ""), field="ciphertext_b64")

        try:
            client_epk = c["X25519PublicKey"].from_public_bytes(client_epk_bytes)
            shared = self._priv.exchange(client_epk)
        except Exception as exc:
            raise E2EEVerificationError("X25519 key exchange failed") from exc

        key = _derive_aead_key(shared, direction="req", nonce=nonce, request_id=request_id, wallet=wallet)
        aad = _aead_aad(direction="req", request_id=request_id, wallet=wallet)
        try:
            plaintext = c["AESGCM"](key).decrypt(nonce, ct, aad)
        except c["InvalidTag"] as exc:
            raise E2EEVerificationError("AES-256-GCM authentication failed (tampered or wrong key)") from exc

        try:
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise E2EEVerificationError("Decrypted E2EE payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise E2EEVerificationError("Decrypted E2EE payload must be a JSON object")
        return payload, shared, request_id

    @staticmethod
    def encrypt_response_envelope(
        payload: dict[str, Any],
        *,
        shared_secret: bytes,
        request_id: str,
        wallet: str,
    ) -> dict[str, Any]:
        """Encrypt the worker's response payload back to the client's ephemeral key."""
        c = _import_crypto()
        nonce = os.urandom(12)
        key = _derive_aead_key(shared_secret, direction="resp", nonce=nonce, request_id=request_id, wallet=wallet)
        aad = _aead_aad(direction="resp", request_id=request_id, wallet=wallet)
        pt = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ct = c["AESGCM"](key).encrypt(nonce, pt, aad)
        return {
            "v": E2EE_PROTOCOL_VERSION,
            "request_id": request_id,
            "wallet": wallet,
            "nonce_b64": _b64e(nonce),
            "ciphertext_b64": _b64e(ct),
        }


@dataclass
class ClientE2EESession:
    """Single-use client E2EE session holding ephemeral shared secret for decrypting the response."""

    request_id: str
    worker_wallet: str
    _shared_secret: bytes

    def decrypt_response(self, response_envelope: dict[str, Any]) -> dict[str, Any]:
        c = _import_crypto()
        if not isinstance(response_envelope, dict):
            raise E2EEVerificationError("E2EE response envelope must be a JSON object")
        if response_envelope.get("v") != E2EE_PROTOCOL_VERSION:
            raise E2EEVerificationError(f"Unsupported E2EE response version: {response_envelope.get('v')!r}")
        if str(response_envelope.get("request_id") or "").strip() != self.request_id:
            raise E2EEVerificationError("E2EE response request_id mismatch (possible replay/swap)")
        if str(response_envelope.get("wallet") or "").strip() != self.worker_wallet:
            raise E2EEVerificationError("E2EE response wallet mismatch")

        nonce = _b64d(str(response_envelope.get("nonce_b64") or ""), expected_len=12, field="nonce_b64")
        ct = _b64d(str(response_envelope.get("ciphertext_b64") or ""), field="ciphertext_b64")

        key = _derive_aead_key(
            self._shared_secret,
            direction="resp",
            nonce=nonce,
            request_id=self.request_id,
            wallet=self.worker_wallet,
        )
        aad = _aead_aad(direction="resp", request_id=self.request_id, wallet=self.worker_wallet)
        try:
            plaintext = c["AESGCM"](key).decrypt(nonce, ct, aad)
        except c["InvalidTag"] as exc:
            raise E2EEVerificationError("Response AES-256-GCM authentication failed (tampered response)") from exc

        try:
            data = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise E2EEVerificationError("Decrypted E2EE response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise E2EEVerificationError("Decrypted E2EE response must be a JSON object")
        return data


def encrypt_client_request(
    payload: dict[str, Any],
    *,
    worker_wallet: str,
    worker_e2ee_pubkey_b64: str,
    worker_e2ee_sig_hex: str,
    request_id: str | None = None,
) -> tuple[dict[str, Any], ClientE2EESession]:
    """Verify worker Ed25519→X25519 binding and encrypt a request envelope client-side."""
    c = _import_crypto()
    clean_wallet = (worker_wallet or "").strip()
    if not verify_e2ee_pubkey_binding(clean_wallet, worker_e2ee_pubkey_b64, worker_e2ee_sig_hex):
        raise E2EEVerificationError(
            f"Worker {clean_wallet[:12]}... failed Ed25519->X25519 key binding verification! "
            "Refusing to encrypt prompt (possible MITM or tampered coordinator key)."
        )

    req_id = (request_id or f"e2ee_{uuid.uuid4().hex[:16]}").strip()
    worker_pub_bytes = _b64d(worker_e2ee_pubkey_b64, expected_len=32, field="worker_e2ee_pubkey_b64")
    worker_pub = c["X25519PublicKey"].from_public_bytes(worker_pub_bytes)

    eph_priv = c["X25519PrivateKey"].generate()
    eph_pub_bytes = eph_priv.public_key().public_bytes(
        encoding=c["serialization"].Encoding.Raw,
        format=c["serialization"].PublicFormat.Raw,
    )
    shared = eph_priv.exchange(worker_pub)
    nonce = os.urandom(12)
    key = _derive_aead_key(shared, direction="req", nonce=nonce, request_id=req_id, wallet=clean_wallet)
    aad = _aead_aad(direction="req", request_id=req_id, wallet=clean_wallet)
    pt = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ct = c["AESGCM"](key).encrypt(nonce, pt, aad)

    envelope = {
        "v": E2EE_PROTOCOL_VERSION,
        "request_id": req_id,
        "wallet": clean_wallet,
        "client_epk_b64": _b64e(eph_pub_bytes),
        "nonce_b64": _b64e(nonce),
        "ciphertext_b64": _b64e(ct),
    }
    return envelope, ClientE2EESession(request_id=req_id, worker_wallet=clean_wallet, _shared_secret=shared)


__all__ = [
    "E2EE_PROTOCOL_VERSION",
    "KEY_BINDING_PREFIX",
    "PRIVATE_POOL_DOMAIN",
    "E2EEError",
    "E2EEUnavailableError",
    "E2EEVerificationError",
    "b58encode",
    "b58decode",
    "e2ee_available",
    "key_binding_message",
    "sign_e2ee_pubkey",
    "verify_e2ee_pubkey_binding",
    "hash_private_pool_secret",
    "is_loopback_url",
    "ciphertext_sha256",
    "WorkerE2EEKeypair",
    "ClientE2EESession",
    "encrypt_client_request",
]
