"""One-Command Coding Agent Launchers for Space AI (`claude`, `codex`, `opencode`).

Launches popular terminal coding agents pointed at the unified local Space proxy
(`http://127.0.0.1:8377` by default) using purely ephemeral process environment
variables — NEVER mutating `~/.bashrc`, `~/.zshrc`, or existing paid configs.

Automatically:
  - Discovers saved compute-barter API keys from `~/.space_ai/config.json` or `~/.longinus_node.json`
  - Auto-starts an ephemeral local `SpaceLocalProxy` if one isn't already running
  - Injects `X-Space-Client: coding-agent` (2.0x Season 0 demand points) + routing headers
  - Provides rate-limit fallback guidance when switching away from rate-limited subscriptions
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli import (
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    load_space_config,
    mask_secret,
    resolve_barter_key,
    resolve_session_token,
    resolve_upstreams,
)
from cli.proxy import ProxyConfig, SpaceLocalProxy, check_proxy_health, push_proxy_policy

AGENT_SPECS: dict[str, dict[str, Any]] = {
    "claude": {
        "binary": "claude",
        "protocol": "anthropic",
        "display_name": "Claude Code",
        "install_hint": "npm install -g @anthropic-ai/claude-code",
        "rate_limit_note": (
            "Rate-limit fallback active: Claude Code is routed through Space AI with ephemeral env vars "
            "(your paid Anthropic login & ~/.bashrc remain untouched). If you hit a 402/429 balance cap, "
            "run `space up --auto-pause` to earn free barter credits or `space session create --max-usdc 5`."
        ),
    },
    "codex": {
        "binary": "codex",
        "protocol": "openai",
        "display_name": "OpenAI Codex CLI",
        "install_hint": "npm install -g @openai/codex",
        "rate_limit_note": (
            "Rate-limit fallback active: Codex CLI is routed through Space AI with ephemeral OpenAI env vars "
            "(your existing OpenAI config & ~/.bashrc remain untouched). Earn more barter credits with "
            "`space up --auto-pause` or cap spend with `space session create --max-usdc 5`."
        ),
    },
    "opencode": {
        "binary": "opencode",
        "protocol": "openai",
        "display_name": "OpenCode",
        "install_hint": "curl -fsSL https://opencode.ai/install | bash",
        "rate_limit_note": (
            "Rate-limit fallback active: OpenCode is routed through Space AI with ephemeral OpenAI-compatible env vars "
            "(your existing provider configs & ~/.bashrc remain untouched)."
        ),
    },
}


@dataclass
class LaunchOptions:
    agent: str
    agent_args: list[str] = field(default_factory=list)
    coordinator: str | None = None
    proxy_host: str = DEFAULT_PROXY_HOST
    proxy_port: int = DEFAULT_PROXY_PORT
    direct: bool = False
    auto_start_proxy: bool = True
    route: str | None = None
    privacy: str | None = None
    region: str | None = None
    private_node: str | None = None
    private_pool_secret: str | None = None
    model: str | None = None
    session: str | None = None
    api_key: str | None = None
    dry_run: bool = False
    dry_run_format: str = "json"  # "json" or "shell"
    home_dir: Path | str | None = None


@dataclass
class LaunchPlan:
    agent: str
    display_name: str
    protocol: str
    binary: str
    binary_path: str | None
    command: list[str]
    env_overrides: dict[str, str]
    proxy_url: str
    direct_mode: bool
    upstreams: list[str]
    routing: dict[str, str | None]
    barter_key_source: str
    barter_key_masked: str | None
    rate_limit_note: str

    def shell_command(self, mask_secrets: bool = False) -> str:
        env_parts: list[str] = []
        for k, v in self.env_overrides.items():
            shown = v
            if mask_secrets and ("KEY" in k or "TOKEN" in k) and v not in ("space-local-proxy",):
                shown = mask_secret(v) or v
            env_parts.append(f"{k}={shlex.quote(shown)}")
        cmd_str = " ".join(shlex.quote(a) for a in self.command)
        return f"{' '.join(env_parts)} {cmd_str}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "display_name": self.display_name,
            "protocol": self.protocol,
            "binary": self.binary,
            "binary_found": self.binary_path is not None,
            "binary_path": self.binary_path,
            "command": self.command,
            "env": self.env_overrides,
            "shell": self.shell_command(mask_secrets=False),
            "proxy_url": self.proxy_url,
            "direct_mode": self.direct_mode,
            "upstreams": self.upstreams,
            "routing": self.routing,
            "barter_key": {
                "configured": self.barter_key_masked is not None,
                "masked": self.barter_key_masked,
                "source": self.barter_key_source,
            },
            "config_mutated": False,
            "rate_limit_note": self.rate_limit_note,
        }


def build_launch_plan(options: LaunchOptions) -> LaunchPlan:
    """Resolve ephemeral env vars, proxy target, and command without mutating any user config."""
    agent_key = options.agent.strip().lower()
    if agent_key not in AGENT_SPECS:
        raise ValueError(f"Unsupported coding agent '{options.agent}'. Supported: {', '.join(AGENT_SPECS)}")

    spec = AGENT_SPECS[agent_key]
    space_cfg = load_space_config(options.home_dir)
    upstreams = resolve_upstreams(
        [options.coordinator] if options.coordinator else None,
        home_dir=options.home_dir,
    )

    route = (options.route or str(space_cfg.get("route") or "fastest")).strip()
    privacy = (options.privacy or str(space_cfg.get("privacy") or "standard")).strip()
    region = (options.region if options.region is not None else str(space_cfg.get("region") or "")).strip()
    private_node = (
        options.private_node if options.private_node is not None else str(space_cfg.get("private_node") or "")
    ).strip()
    private_pool_secret = (
        options.private_pool_secret
        if options.private_pool_secret is not None
        else str(space_cfg.get("private_pool_secret") or "")
    ).strip()
    session = resolve_session_token(options.session, home_dir=options.home_dir)
    model = (options.model or str(space_cfg.get("model") or "")).strip()

    # Air-gapped local-only mode MUST route through local proxy (never --direct to a remote coordinator)
    if privacy == "local-only" and options.direct:
        from cli.e2ee import is_loopback_url

        if not upstreams or not is_loopback_url(upstreams[0]):
            raise ValueError(
                "Refusing `--direct` with `--privacy local-only` against a non-loopback upstream. "
                "Omit `--direct` or pass `--coordinator http://127.0.0.1:<port>`."
            )

    barter_key, key_source = resolve_barter_key(explicit_key=options.api_key, home_dir=options.home_dir)
    # Use saved barter key if present; otherwise a clean local proxy placeholder so proxy injects or routes cleanly.
    effective_token = barter_key or (session if options.direct and session else "space-local-proxy")

    proxy_base = f"http://{options.proxy_host}:{int(options.proxy_port)}"
    target_base = upstreams[0].rstrip("/") if options.direct else proxy_base

    env_overrides: dict[str, str] = {
        "SPACE_PROXY_URL": proxy_base,
        "SPACE_CLIENT": "coding-agent",
        "SPACE_ROUTE": route,
        "SPACE_PRIVACY": privacy,
    }
    if region:
        env_overrides["SPACE_REGION"] = region
    if session:
        env_overrides["SPACE_SESSION"] = session

    if spec["protocol"] == "anthropic":
        env_overrides["ANTHROPIC_BASE_URL"] = target_base
        env_overrides["ANTHROPIC_AUTH_TOKEN"] = effective_token
        env_overrides["ANTHROPIC_API_KEY"] = effective_token
        if model:
            env_overrides["ANTHROPIC_MODEL"] = model
    else:
        openai_base = target_base if target_base.endswith("/v1") else f"{target_base}/v1"
        env_overrides["OPENAI_BASE_URL"] = openai_base
        env_overrides["OPENAI_API_KEY"] = effective_token
        if agent_key == "opencode":
            env_overrides["OPENAI_API_BASE"] = openai_base
            if model:
                env_overrides["OPENCODE_MODEL"] = model
        if model:
            env_overrides["OPENAI_MODEL"] = model

    binary = spec["binary"]
    binary_path = shutil.which(binary)
    command = [binary, *options.agent_args]

    return LaunchPlan(
        agent=agent_key,
        display_name=spec["display_name"],
        protocol=spec["protocol"],
        binary=binary,
        binary_path=binary_path,
        command=command,
        env_overrides=env_overrides,
        proxy_url=target_base,
        direct_mode=bool(options.direct),
        upstreams=upstreams,
        routing={
            "route": route,
            "privacy": privacy,
            "region": region or None,
            "private_node": private_node or None,
            "private_pool_configured": bool(private_pool_secret),
            "model": model or None,
            "session": session,
            "client": "coding-agent",
        },
        barter_key_source=key_source,
        barter_key_masked=mask_secret(barter_key),
        rate_limit_note=spec["rate_limit_note"],
    )


def format_dry_run(plan: LaunchPlan, fmt: str = "json") -> str:
    if fmt == "shell":
        lines = [
            f"# {plan.display_name} via Space AI ({plan.proxy_url})",
            f"# {plan.rate_limit_note}",
            plan.shell_command(mask_secrets=False),
        ]
        return "\n".join(lines)
    return json.dumps(plan.to_dict(), indent=2)


def launch_coding_agent(options: LaunchOptions) -> int:
    """Execute (or dry-run) a coding agent with ephemeral Space AI environment variables."""
    plan = build_launch_plan(options)

    if options.dry_run:
        print(format_dry_run(plan, fmt=options.dry_run_format))
        return 0

    spec = AGENT_SPECS[plan.agent]
    if plan.binary_path is None:
        print(
            f"[SPACE] Could not find `{plan.binary}` in PATH.\n"
            f"        Install {plan.display_name} with:  {spec['install_hint']}\n"
            f"        Tip: preview the exact env/command anytime with `space {plan.agent} --dry-run`.",
            file=sys.stderr,
        )
        return 127

    ephemeral_proxy: SpaceLocalProxy | None = None
    if not options.direct:
        health = check_proxy_health(options.proxy_host, options.proxy_port, timeout=0.8)
        policy_updates: dict[str, Any] = {
            "route": plan.routing["route"],
            "privacy": plan.routing["privacy"],
            "region": plan.routing["region"] or "",
            "private_node": plan.routing.get("private_node") or "",
            "private_pool_secret": options.private_pool_secret or "",
            "client_tag": "coding-agent",
        }
        if plan.routing["session"]:
            policy_updates["session_token"] = plan.routing["session"]
        if plan.routing["model"]:
            policy_updates["default_model"] = plan.routing["model"]

        if health is not None:
            # Push launcher routing policy to the already-running local proxy
            push_proxy_policy(policy_updates, options.proxy_host, options.proxy_port)
        elif options.auto_start_proxy:
            proxy_cfg = ProxyConfig(
                host=options.proxy_host,
                port=options.proxy_port,
                upstreams=plan.upstreams,
                route=plan.routing["route"] or "fastest",
                privacy=plan.routing["privacy"] or "standard",
                region=plan.routing["region"] or "",
                private_node=plan.routing.get("private_node") or "",
                private_pool_secret=options.private_pool_secret or "",
                session_token=plan.routing["session"],
                client_tag="coding-agent",
                default_model=plan.routing["model"] or "",
                api_key=options.api_key,
                home_dir=options.home_dir,
                quiet=True,
            )
            ephemeral_proxy = SpaceLocalProxy(proxy_cfg)
            ephemeral_proxy.start(background=True)
            print(
                f"[SPACE] Auto-started local buyer proxy on {ephemeral_proxy.base_url} "
                f"(upstreams: {', '.join(plan.upstreams)})",
                file=sys.stderr,
            )
        else:
            print(
                f"[SPACE] Warning: local proxy not reachable at {plan.proxy_url} and --no-auto-start was set.",
                file=sys.stderr,
            )

    print(f"[SPACE] Launching {plan.display_name} -> {plan.proxy_url}", file=sys.stderr)
    print(f"[SPACE] {plan.rate_limit_note}", file=sys.stderr)
    if plan.barter_key_masked:
        print(
            f"[SPACE] Barter key auto-loaded ({plan.barter_key_masked} from {plan.barter_key_source}) — zero copy-paste.",
            file=sys.stderr,
        )
    elif plan.routing["session"]:
        print(f"[SPACE] Active USDC session budget: {mask_secret(plan.routing['session'])}", file=sys.stderr)

    child_env = dict(os.environ)
    child_env.update(plan.env_overrides)
    try:
        proc = subprocess.run([plan.binary_path, *options.agent_args], env=child_env)
        return int(proc.returncode)
    except KeyboardInterrupt:
        return 130
    finally:
        if ephemeral_proxy is not None:
            ephemeral_proxy.stop()


__all__ = [
    "AGENT_SPECS",
    "LaunchOptions",
    "LaunchPlan",
    "build_launch_plan",
    "format_dry_run",
    "launch_coding_agent",
]
