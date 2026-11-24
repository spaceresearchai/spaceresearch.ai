# Security Policy

## Scope & Transparency Commitment

Every file executed on a user's machine by the Space Research CLI (`space`) or GPU/compute worker (`join.py`) is published in plain source in this repository:

- `cli/` (`__init__.py`, `adapters.py`, `launchers.py`, `proxy.py`, `space_cli.py`)
- `scripts/space`
- `scripts/join.py`
- `scripts/install-space.sh`
- `scripts/join-gpu.sh`
- `Dockerfile.worker`

### Client Guarantees

1. **No privilege escalation:** Client installers and runtime scripts run strictly as your unprivileged user and never invoke `sudo`.
2. **No hidden shell modifications:** `install-space.sh` installs only to `~/.local/bin/space` and `~/.space_ai/app` (both overridable via `SPACE_BIN_DIR` / `SPACE_INSTALL_DIR`) and does not edit shell startup files.
3. **Bundle integrity verification:** `install-space.sh` verifies the SHA-256 digest of `space-cli.tar.gz` against `space-cli.sha256` and rejects tar path-traversal entries before extracting.
4. **Optional container isolation:** GPU contributors can run `Dockerfile.worker` with `--read-only` and an isolated `/data` volume so the worker has zero visibility into host files.
5. **Local-only proxy binding:** `space` proxy binds to loopback (`127.0.0.1`) by default.

### Verification

Before running any downloaded script, verify it against this repository:

```bash
sha256sum -c space-cli.sha256
python3 -m unittest tests/test_space_cli_proxy.py -v
```

## Reporting a Vulnerability

If you discover a security issue in the CLI, local proxy, worker client, installers, or `spaceresearch.ai` infrastructure, please report it responsibly:

- **GitHub Security Advisories:** Open a private vulnerability report in this repository's **Security → Advisories** tab.
- **Email:** `engineering@spaceresearch.ai`

Please do not disclose exploitable vulnerabilities publicly before we have had a reasonable opportunity to investigate and release a fix.
