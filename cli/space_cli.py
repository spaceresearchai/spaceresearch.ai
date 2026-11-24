"""Space AI Unified Command-Line Interface (`space`).

Commands:
  space claude   [flags] [-- args...]   Launch Claude Code via local Space proxy (ephemeral env)
  space codex    [flags] [-- args...]   Launch Codex CLI via local Space proxy (ephemeral env)
  space opencode [flags] [-- args...]   Launch OpenCode via local Space proxy (ephemeral env)
  space up       [flags]                Run unified local buyer proxy + worker adapter supervisor
  space proxy    [flags]                Run standalone local OpenAI/Anthropic buyer proxy (127.0.0.1:8377)
  space adapters [--json]               Probe local hardware & provider adapters + multiplier tiers
  space session  create|status ...      Create/inspect agent-native USDC session budget tokens (`sp_sess_...`)
  space status   [--json]               Show local proxy, barter key, worker, and upstream failover status
  space config   [flags]                Inspect or update `~/.space_ai/config.json`
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from cli import (
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    VALID_PRIVACY_TIERS,
    VALID_ROUTES,
    __version__,
    get_repo_root,
    load_space_config,
    mask_secret,
    resolve_barter_key,
    resolve_session_token,
    resolve_upstreams,
    save_space_config,
)
from cli.adapters import (
    ADAPTER_CLASSES,
    AdapterBridgeWorker,
    NativeAdapter,
    format_adapters_table,
    get_adapter,
    probe_all_adapters,
)
from cli.launchers import LaunchOptions, launch_coding_agent
from cli.proxy import ProxyConfig, SpaceLocalProxy


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 15.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    req_headers = {"User-Agent": f"space-cli/{__version__}"}
    if headers:
        req_headers.update(headers)
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}


def _add_routing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--coordinator",
        action="append",
        default=None,
        help="Coordinator / upstream URL (may be passed multiple times for failover)",
    )
    parser.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST, help=f"Local proxy bind host (default: {DEFAULT_PROXY_HOST})")
    parser.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT, help=f"Local proxy bind port (default: {DEFAULT_PROXY_PORT})")
    parser.add_argument("--route", choices=VALID_ROUTES, default=None, help="Space routing preference (`X-Space-Route`)")
    parser.add_argument("--privacy", choices=VALID_PRIVACY_TIERS, default=None, help="Space privacy tier (`X-Space-Privacy`)")
    parser.add_argument("--region", default=None, help="Preferred geographic region (`X-Space-Region`, e.g. us, eu, ap)")
    parser.add_argument("--private-node", default=None, help="Pin a specific worker Solana wallet for E2EE blind-relay (`X-Space-Private-Node`)")
    parser.add_argument("--private-pool", default=None, help="Shared private pool secret (hashed locally before sending)")
    parser.add_argument("--model", default=None, help="Default model override")
    parser.add_argument("--session", default=None, help="USDC session budget token (`sp_sess_...`)")
    parser.add_argument("--api-key", default=None, help="Explicit Space/Longinus barter API key override")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="space",
        description="Space AI CLI — unified local OpenAI/Anthropic proxy, coding agent launchers, and worker adapters.",
    )
    parser.add_argument("--version", action="version", version=f"space {__version__}")
    sub = parser.add_subparsers(dest="command")

    # Coding agent launchers: claude, codex, opencode
    for agent_name, desc in (
        ("claude", "Launch Claude Code pointed at local Space proxy (ephemeral env, zero config mutation)"),
        ("codex", "Launch OpenAI Codex CLI pointed at local Space proxy (ephemeral env)"),
        ("opencode", "Launch OpenCode pointed at local Space proxy (ephemeral env)"),
    ):
        p = sub.add_parser(agent_name, help=desc, description=desc)
        _add_routing_args(p)
        p.add_argument("--direct", action="store_true", help="Point agent directly at --coordinator instead of local proxy")
        p.add_argument("--no-auto-start", action="store_true", help="Do not auto-start local proxy if not already running")
        p.add_argument("--dry-run", action="store_true", help="Print resolved env + command without executing")
        p.add_argument("--format", choices=("json", "shell"), default="json", help="Output format for --dry-run (default: json)")

    # Standalone local buyer proxy
    p_proxy = sub.add_parser("proxy", help="Run the unified local OpenAI/Anthropic buyer proxy")
    _add_routing_args(p_proxy)
    p_proxy.add_argument("--host", dest="proxy_host", default=DEFAULT_PROXY_HOST, help=f"Bind host (default: {DEFAULT_PROXY_HOST})")
    p_proxy.add_argument("--port", dest="proxy_port", type=int, default=DEFAULT_PROXY_PORT, help=f"Bind port (default: {DEFAULT_PROXY_PORT})")
    p_proxy.add_argument("--client", default="", help="Default X-Space-Client header value (e.g. coding-agent)")
    p_proxy.add_argument("--force-translate-anthropic", action="store_true", help="Always translate /v1/messages to OpenAI /v1/chat/completions")
    p_proxy.add_argument("--quiet", action="store_true", help="Suppress per-request access logs")

    # Unified Worker + Proxy (`space up`)
    p_up = sub.add_parser("up", help="Run unified local buyer proxy + supervised worker adapter")
    _add_routing_args(p_up)
    p_up.add_argument("--name", default="", help="Worker display handle on the Space leaderboard")
    p_up.add_argument("--auto-pause", action="store_true", help="Automatically pause & free VRAM on personal GPU/app load")
    p_up.add_argument(
        "--adapter",
        choices=tuple(ADAPTER_CLASSES.keys()),
        default="native",
        help="Worker backend adapter (default: native — 20x Genesis tier)",
    )
    p_up.add_argument("--adapter-endpoint", default=None, help="Custom local endpoint URL for external adapter")
    p_up.add_argument("--wallet", "--payout-wallet", dest="payout_wallet", default="", help="Solana wallet for tSPACE payouts")
    p_up.add_argument("--dry-run", action="store_true", help="Print unified proxy + worker supervisor plan without starting")
    p_up.add_argument("--quiet", action="store_true", help="Suppress per-request proxy access logs")

    # Provider adapters detection
    p_adapters = sub.add_parser("adapters", help="Probe local hardware & provider adapters with multiplier tiers")
    p_adapters.add_argument("--json", action="store_true", dest="as_json", help="Print structured JSON output")
    p_adapters.add_argument("--timeout", type=float, default=1.5, help="Per-endpoint probe timeout in seconds")

    # Agent-native USDC session budget helper
    p_session = sub.add_parser("session", help="Create and inspect agent-native USDC session budgets (`sp_sess_...`)")
    sess_sub = p_session.add_subparsers(dest="session_cmd")

    p_sc = sess_sub.add_parser("create", help="Create a capped session budget token (`POST /v1/sessions/budget`)")
    p_sc.add_argument("--max-usdc", type=float, required=True, help="Maximum USDC cap for this session (e.g. 5.0)")
    p_sc.add_argument("--rail", choices=("solana_usdc", "card_topup"), default="solana_usdc", help="Settlement rail")
    p_sc.add_argument("--owner", default="", help="Optional owner wallet or agent handle")
    p_sc.add_argument("--ttl-hours", type=float, default=24.0, help="Session expiration TTL in hours (default: 24)")
    p_sc.add_argument("--coordinator", default=None, help="Coordinator base URL")
    p_sc.add_argument("--no-save", action="store_true", help="Do not persist session token to ~/.space_ai/config.json")
    p_sc.add_argument("--json", action="store_true", dest="as_json", help="Print raw JSON response")

    p_ss = sess_sub.add_parser("status", help="Inspect live balance of a session budget token")
    p_ss.add_argument("token", nargs="?", default=None, help="Session token (`sp_sess_...`; defaults to saved session)")
    p_ss.add_argument("--coordinator", default=None, help="Coordinator base URL")
    p_ss.add_argument("--json", action="store_true", dest="as_json", help="Print raw JSON response")

    # Local proxy & config status
    p_status = sub.add_parser("status", help="Inspect running local proxy status, barter key, and upstream health")
    p_status.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    p_status.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    p_status.add_argument("--json", action="store_true", dest="as_json")

    # Config helper
    p_cfg = sub.add_parser("config", help="View or update ~/.space_ai/config.json")
    p_cfg.add_argument("--set-key", default=None, help="Save Space/Longinus barter API key")
    p_cfg.add_argument("--set-coordinator", default=None, help="Save default coordinator URL")
    p_cfg.add_argument("--set-route", choices=VALID_ROUTES, default=None, help="Save default X-Space-Route")
    p_cfg.add_argument("--set-privacy", choices=VALID_PRIVACY_TIERS, default=None, help="Save default X-Space-Privacy")
    p_cfg.add_argument("--set-region", default=None, help="Save default X-Space-Region")
    p_cfg.add_argument("--set-private-node", default=None, help="Save pinned private worker Solana wallet")
    p_cfg.add_argument("--set-private-pool", default=None, help="Save private pool secret (stored 0600, hashed before wire)")
    p_cfg.add_argument("--json", action="store_true", dest="as_json")

    # Auditable privacy self-check
    p_priv = sub.add_parser(
        "privacy-check",
        aliases=("verify-privacy",),
        help="Audit local Private AI invariants (E2EE crypto, air-gap, fail-closed policy, key permissions)",
    )
    p_priv.add_argument("--json", action="store_true", dest="as_json")

    return parser


def _cmd_launcher(args: argparse.Namespace, home_dir: Path | None = None) -> int:
    forwarded = list(args.agent_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    coord = args.coordinator[0] if args.coordinator else None
    opts = LaunchOptions(
        agent=args.command,
        agent_args=forwarded,
        coordinator=coord,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        direct=bool(args.direct),
        auto_start_proxy=not bool(args.no_auto_start),
        route=args.route,
        privacy=args.privacy,
        region=args.region,
        private_node=args.private_node,
        private_pool_secret=args.private_pool,
        model=args.model,
        session=args.session,
        api_key=args.api_key,
        dry_run=bool(args.dry_run),
        dry_run_format=args.format,
        home_dir=home_dir,
    )
    return launch_coding_agent(opts)


def _cmd_proxy(args: argparse.Namespace, home_dir: Path | None = None) -> int:
    upstreams = resolve_upstreams(args.coordinator, home_dir=home_dir)
    space_cfg = load_space_config(home_dir)
    cfg = ProxyConfig(
        host=args.proxy_host,
        port=args.proxy_port,
        upstreams=upstreams,
        route=args.route or str(space_cfg.get("route") or "fastest"),
        privacy=args.privacy or str(space_cfg.get("privacy") or "standard"),
        region=args.region if args.region is not None else str(space_cfg.get("region") or ""),
        private_node=args.private_node if args.private_node is not None else str(space_cfg.get("private_node") or ""),
        private_pool_secret=(
            args.private_pool if args.private_pool is not None else str(space_cfg.get("private_pool_secret") or "")
        ),
        session_token=args.session,
        client_tag=args.client or "",
        default_model=args.model or "",
        api_key=args.api_key,
        home_dir=home_dir,
        force_translate_anthropic=bool(args.force_translate_anthropic),
        quiet=bool(args.quiet),
    )
    proxy = SpaceLocalProxy(cfg)
    proxy.start(background=True)
    key, key_src = proxy.active_barter_key()
    print(f"[SPACE PROXY] Listening on {proxy.base_url} (OpenAI /v1/chat/completions + Anthropic /v1/messages)")
    print(f"[SPACE PROXY] Upstreams: {', '.join(upstreams)}")
    print(f"[SPACE PROXY] Policy: route={cfg.route} privacy={cfg.privacy} region={cfg.region or 'auto'}")
    print(f"[SPACE PROXY] Barter key: {mask_secret(key) or 'not configured'} ({key_src})")
    print(f"[SPACE PROXY] Status: {proxy.base_url}/status | Ctrl+C to stop.")

    stop_event = threading.Event()

    def _on_signal(_sig: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        while not stop_event.is_set():
            stop_event.wait(0.5)
    finally:
        proxy.stop()
        print("\n[SPACE PROXY] Stopped cleanly.")
    return 0


def _cmd_up(args: argparse.Namespace, home_dir: Path | None = None) -> int:
    upstreams = resolve_upstreams(args.coordinator, home_dir=home_dir)
    primary_coord = upstreams[0]
    space_cfg = load_space_config(home_dir)
    adapter = get_adapter(args.adapter, endpoint=args.adapter_endpoint, repo_root=get_repo_root())

    plan = {
        "proxy": {
            "bind": f"http://{args.proxy_host}:{args.proxy_port}",
            "upstreams": upstreams,
            "route": args.route or str(space_cfg.get("route") or "fastest"),
            "privacy": args.privacy or str(space_cfg.get("privacy") or "standard"),
            "region": args.region if args.region is not None else str(space_cfg.get("region") or ""),
        },
        "worker": {
            "adapter": adapter.name,
            "tier": adapter.tier,
            "tier_label": adapter.tier_label,
            "multiplier": adapter.multiplier,
            "endpoint": adapter.endpoint,
            "coordinator": primary_coord,
            "name": args.name or "",
            "auto_pause": bool(args.auto_pause),
            "model": args.model,
            "command": (
                adapter.build_worker_command(
                    coordinator=primary_coord,
                    name=args.name,
                    auto_pause=bool(args.auto_pause),
                    model=args.model,
                    payout_wallet=args.payout_wallet,
                )
                if isinstance(adapter, NativeAdapter)
                else None
            ),
        },
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    stop_event = threading.Event()
    worker_proc: subprocess.Popen[Any] | None = None
    bridge_worker: AdapterBridgeWorker | None = None
    native_state: dict[str, Any] = {
        "enabled": True,
        "mode": "native_subprocess",
        "adapter": adapter.name,
        "tier": adapter.tier,
        "multiplier": adapter.multiplier,
        "coordinator": primary_coord,
        "name": args.name or "",
        "auto_pause": bool(args.auto_pause),
        "state": "starting",
        "pid": None,
        "restarts": 0,
    }

    def _worker_status() -> dict[str, Any]:
        if bridge_worker is not None:
            return bridge_worker.status_dict()
        st = dict(native_state)
        if worker_proc is not None:
            rc = worker_proc.poll()
            st["pid"] = worker_proc.pid
            st["state"] = "running" if rc is None else f"exited({rc})"
        return st

    proxy_cfg = ProxyConfig(
        host=args.proxy_host,
        port=args.proxy_port,
        upstreams=upstreams,
        route=plan["proxy"]["route"],
        privacy=plan["proxy"]["privacy"],
        region=plan["proxy"]["region"],
        session_token=args.session,
        default_model=args.model or "",
        api_key=args.api_key,
        home_dir=home_dir,
        quiet=bool(args.quiet),
    )
    proxy = SpaceLocalProxy(proxy_cfg, worker_status_provider=_worker_status)
    proxy.start(background=True)
    print(f"[SPACE UP] Local buyer proxy online at {proxy.base_url} (status: {proxy.base_url}/status)")

    def _on_signal(_sig: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        if isinstance(adapter, NativeAdapter):
            cmd = adapter.build_worker_command(
                coordinator=primary_coord,
                name=args.name,
                auto_pause=bool(args.auto_pause),
                model=args.model,
                payout_wallet=args.payout_wallet,
            )
            print(f"[SPACE UP] Starting native worker ({adapter.tier_label}): {' '.join(cmd)}")
            worker_proc = subprocess.Popen(cmd)
            native_state["pid"] = worker_proc.pid
            native_state["state"] = "running"

            while not stop_event.is_set():
                rc = worker_proc.poll()
                if rc is not None:
                    native_state["state"] = f"exited({rc})"
                    print(f"[SPACE UP] Native worker exited (code={rc}); restarting in 3s... (Ctrl+C to stop)")
                    if stop_event.wait(3.0):
                        break
                    native_state["restarts"] += 1
                    worker_proc = subprocess.Popen(cmd)
                    native_state["pid"] = worker_proc.pid
                    native_state["state"] = "running"
                stop_event.wait(0.5)
        else:
            bridge_worker = AdapterBridgeWorker(
                adapter=adapter,
                coordinator=primary_coord,
                display_name=args.name or f"{adapter.name}-worker",
                model_override=args.model,
                payout_wallet=args.payout_wallet,
                home_dir=home_dir,
            )
            print(
                f"[SPACE UP] Starting `{adapter.name}` adapter bridge ({adapter.tier_label}) "
                f"{adapter.endpoint} -> {primary_coord}"
            )
            bridge_worker.start()
            while not stop_event.is_set():
                stop_event.wait(0.5)
    finally:
        print("\n[SPACE UP] Shutting down worker & local proxy...")
        if bridge_worker is not None:
            bridge_worker.stop()
        if worker_proc is not None and worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
        proxy.stop()
        print("[SPACE UP] Clean shutdown complete.")
    return 0


def _cmd_adapters(args: argparse.Namespace) -> int:
    results = probe_all_adapters(timeout=args.timeout, repo_root=get_repo_root())
    if args.as_json:
        print(json.dumps({"adapters": [r.to_dict() for r in results]}, indent=2))
    else:
        print(format_adapters_table(results))
    return 0


def _cmd_session(args: argparse.Namespace, home_dir: Path | None = None) -> int:
    if not args.session_cmd:
        print("Usage: space session {create|status} [options]", file=sys.stderr)
        return 2

    upstreams = resolve_upstreams([args.coordinator] if args.coordinator else None, home_dir=home_dir)
    coord = upstreams[0].rstrip("/")

    if args.session_cmd == "create":
        if args.max_usdc <= 0:
            print("[SPACE] --max-usdc must be greater than 0.", file=sys.stderr)
            return 2
        ttl_seconds = max(60, min(2_592_000, int(round(float(args.ttl_hours) * 3600.0))))
        payload: dict[str, Any] = {
            "max_usdc": round(float(args.max_usdc), 6),
            "rail": args.rail,
            "ttl_seconds": ttl_seconds,
        }
        if args.owner:
            payload["owner_id"] = args.owner
        try:
            resp = _http_json(f"{coord}/v1/sessions/budget", payload=payload, timeout=15.0)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[SPACE] Session create failed (HTTP {exc.code}): {detail}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[SPACE] Could not reach coordinator {coord}: {exc}", file=sys.stderr)
            return 1

        token = str(resp.get("session_token") or resp.get("token") or "").strip()
        if token and not args.no_save:
            save_space_config({"session_token": token, "coordinator": coord}, home_dir=home_dir)

        if args.as_json:
            print(json.dumps(resp, indent=2))
            return 0

        print("[SPACE SESSION] Created agent-native capped session budget:")
        print(f"  Session Token : {token}")
        print(f"  Max Budget    : ${float(resp.get('max_usdc', args.max_usdc)):.2f} USDC ({resp.get('rail', args.rail)})")
        if resp.get("expires_at"):
            print(f"  Expires At    : {resp['expires_at']}")
        if token and not args.no_save:
            print("  Saved To      : ~/.space_ai/config.json (auto-injected on `space claude/codex/opencode`)")
        print(f"  Quick Use     : space claude --session {token}")
        return 0

    if args.session_cmd == "status":
        token = args.token or resolve_session_token(home_dir=home_dir)
        if not token:
            print("[SPACE] No session token provided or saved. Run `space session create --max-usdc 5` first.", file=sys.stderr)
            return 2
        try:
            resp = _http_json(f"{coord}/v1/sessions/budget/{token}", timeout=15.0)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[SPACE] Session lookup failed (HTTP {exc.code}): {detail}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[SPACE] Could not reach coordinator {coord}: {exc}", file=sys.stderr)
            return 1

        if args.as_json:
            print(json.dumps(resp, indent=2))
            return 0

        spent = float(resp.get("spent_usdc", 0.0))
        max_u = float(resp.get("max_usdc", 0.0))
        rem = float(resp.get("remaining_usdc", max(0.0, max_u - spent)))
        print(f"[SPACE SESSION] {mask_secret(token)} | Active={resp.get('active', True)}")
        print(f"  Spent / Cap   : ${spent:.4f} / ${max_u:.2f} USDC (Remaining: ${rem:.4f} USDC)")
        return 0

    return 2


def _cmd_status(args: argparse.Namespace, home_dir: Path | None = None) -> int:
    proxy_url = f"http://{args.proxy_host}:{args.proxy_port}"
    try:
        data = _http_json(f"{proxy_url}/status", timeout=2.0)
        data["proxy_online"] = True
    except Exception:
        key, src = resolve_barter_key(home_dir=home_dir)
        sess = resolve_session_token(home_dir=home_dir)
        data = {
            "proxy_online": False,
            "bind": proxy_url,
            "barter_key": {"configured": bool(key), "masked": mask_secret(key), "source": src},
            "session_masked": mask_secret(sess),
            "upstreams": resolve_upstreams(home_dir=home_dir),
        }

    if args.as_json:
        print(json.dumps(data, indent=2))
        return 0

    if data.get("proxy_online"):
        bk = data.get("barter_key") or {}
        rp = data.get("routing_policy") or {}
        wk = data.get("worker") or {}
        st = data.get("stats") or {}
        print(f"Space Local Proxy : ONLINE at {data.get('bind')} (v{data.get('version')}, up {data.get('uptime_s')}s)")
        print(f"Barter Key        : {bk.get('masked') or 'not configured'} ({bk.get('source')})")
        print(f"Worker            : {wk.get('state', 'disabled')} (adapter={wk.get('adapter') or 'none'})")
        print(f"Routing Policy    : route={rp.get('route')} privacy={rp.get('privacy')} region={rp.get('region')}")
        print(
            f"Traffic / Failover: requests={st.get('total_requests', 0)} "
            f"openai={st.get('total_openai_calls', 0)} anthropic={st.get('total_anthropic_calls', 0)} "
            f"failovers={st.get('total_failovers', 0)}"
        )
        for u in data.get("upstreams") or []:
            ema = f"{u['latency_ema_ms']}ms" if u.get("latency_ema_ms") is not None else "n/a"
            print(
                f"  - {u['url']} | healthy={u['healthy']} ema={ema} "
                f"ok={u['requests_ok']} fail={u['requests_failed']}"
            )
    else:
        bk = data.get("barter_key") or {}
        print(f"Space Local Proxy : OFFLINE ({proxy_url})")
        print(f"Barter Key        : {bk.get('masked') or 'not configured'} ({bk.get('source')})")
        print(f"Configured Peers  : {', '.join(data.get('upstreams') or [])}")
        print("Start with        : space up --auto-pause  OR  space proxy")
    return 0


def _cmd_config(args: argparse.Namespace, home_dir: Path | None = None) -> int:
    updates: dict[str, Any] = {}
    if args.set_key is not None:
        updates["barter_key"] = args.set_key.strip()
    if args.set_coordinator is not None:
        updates["coordinator"] = args.set_coordinator.strip().rstrip("/")
    if args.set_route is not None:
        updates["route"] = args.set_route
    if args.set_privacy is not None:
        updates["privacy"] = args.set_privacy
    if args.set_region is not None:
        updates["region"] = args.set_region.strip()
    if args.set_private_node is not None:
        updates["private_node"] = args.set_private_node.strip()
    if args.set_private_pool is not None:
        updates["private_pool_secret"] = args.set_private_pool.strip()

    if updates:
        path = save_space_config(updates, home_dir=home_dir)
        if not args.as_json:
            print(f"[SPACE] Saved configuration to {path}")

    cfg = load_space_config(home_dir=home_dir)
    redacted = dict(cfg)
    for k in ("barter_key", "api_key", "openai_api_key", "session_token", "private_pool_secret"):
        if redacted.get(k):
            redacted[k] = mask_secret(str(redacted[k]))
    if args.as_json:
        print(json.dumps(redacted, indent=2))
    else:
        print(json.dumps(redacted, indent=2))
    return 0


def _cmd_privacy_check(args: argparse.Namespace, home_dir: Path | None = None) -> int:
    from cli import STRICT_PRIVACY_TIERS, get_space_config_path, load_node_identity
    from cli.e2ee import E2EE_PROTOCOL_VERSION, e2ee_available, hash_private_pool_secret, is_loopback_url

    cfg = load_space_config(home_dir)
    cfg_path = get_space_config_path(home_dir)
    _, node_path = load_node_identity(home_dir)
    upstreams = resolve_upstreams(home_dir=home_dir)
    privacy = str(cfg.get("privacy") or "standard").strip().lower()
    loopback_upstreams = [u for u in upstreams if is_loopback_url(u)]

    def _file_mode_ok(p: Path | None) -> bool | None:
        if p is None or not p.exists():
            return None
        try:
            return (p.stat().st_mode & 0o077) == 0
        except OSError:
            return None

    report = {
        "e2ee_protocol": E2EE_PROTOCOL_VERSION,
        "e2ee_crypto_available": e2ee_available(),
        "default_privacy": privacy,
        "fail_closed_tiers": sorted(STRICT_PRIVACY_TIERS),
        "local_only_airgap_ready": bool(loopback_upstreams),
        "loopback_upstreams": loopback_upstreams,
        "configured_upstreams": upstreams,
        "private_node_pinned": str(cfg.get("private_node") or "").strip() or None,
        "private_pool_id": hash_private_pool_secret(str(cfg.get("private_pool_secret") or "")) or None,
        "config_permissions_0600": _file_mode_ok(cfg_path),
        "node_key_permissions_0600": _file_mode_ok(node_path),
        "guarantees": {
            "local_only": "Strictly blocks all non-loopback network egress; fails closed (503) if no local server is configured.",
            "e2ee": "Client-to-worker X25519 + HKDF-SHA256 + AES-256-GCM with mandatory Ed25519 worker key binding; coordinator is blind.",
            "hardened_docker_and_tee": "Fail-closed (503) on coordinator unless matching isolated/attested worker is online.",
        },
    }
    if args.as_json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Space Private AI Audit (protocol={report['e2ee_protocol']})")
    print(f"  E2EE Cryptography : {'READY (X25519 + AES-256-GCM + Ed25519)' if report['e2ee_crypto_available'] else 'MISSING (`pip install cryptography`)'}")
    print(f"  Default Privacy   : {report['default_privacy']} (fail-closed tiers: {', '.join(report['fail_closed_tiers'])})")
    print(f"  Local-Only AirGap : {'READY (' + ', '.join(loopback_upstreams) + ')' if loopback_upstreams else 'No loopback upstream configured (will fail closed with 503)'}")
    print(f"  Pinned Node       : {report['private_node_pinned'] or 'none (any verified E2EE worker)'}")
    print(f"  Private Pool ID   : {report['private_pool_id'] or 'none (public pool)'}")
    return 0


def _split_launcher_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split explicit `--` separator for coding-agent pass-through arguments."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def main(argv: list[str] | None = None, home_dir: Path | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if raw_argv and raw_argv[0] in ("claude", "codex", "opencode"):
        space_part, explicit_passthrough = _split_launcher_argv(raw_argv)
        args, unknown = parser.parse_known_args(space_part)
        # Preserve exact argument ordering for both inline pass-through flags and `-- ...` args
        args.agent_args = [*unknown, *explicit_passthrough]
    else:
        args = parser.parse_args(raw_argv)

    if not args.command:
        parser.print_help()
        return 0
    if args.command in ("claude", "codex", "opencode"):
        return _cmd_launcher(args, home_dir=home_dir)
    if args.command == "proxy":
        return _cmd_proxy(args, home_dir=home_dir)
    if args.command == "up":
        return _cmd_up(args, home_dir=home_dir)
    if args.command == "adapters":
        return _cmd_adapters(args)
    if args.command == "session":
        return _cmd_session(args, home_dir=home_dir)
    if args.command == "status":
        return _cmd_status(args, home_dir=home_dir)
    if args.command == "config":
        return _cmd_config(args, home_dir=home_dir)
    if args.command in ("privacy-check", "verify-privacy"):
        return _cmd_privacy_check(args, home_dir=home_dir)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
