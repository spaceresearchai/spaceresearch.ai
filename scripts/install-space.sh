#!/usr/bin/env sh
# Space AI CLI — Zero-Root One-Line Installer
# Installs the `space` CLI cleanly to ~/.local/bin/space (zero sudo, zero shell mutation).
#
# Usage:
#   sh scripts/install-space.sh
#   curl -fsSL https://spaceresearch.ai/install-space.sh | sh
set -eu

BIN_DIR="${SPACE_BIN_DIR:-$HOME/.local/bin}"
INSTALL_DIR="${SPACE_INSTALL_DIR:-$HOME/.space_ai/app}"
BUNDLE_URL="${SPACE_BUNDLE_URL:-https://spaceresearch.ai/space-cli.tar.gz}"
SHA_URL="${SPACE_SHA_URL:-https://spaceresearch.ai/space-cli.sha256}"
REPO_URL="${SPACE_REPO_URL:-}"
COPY_MODE=0

for arg in "$@"; do
  case "$arg" in
    --copy) COPY_MODE=1 ;;
    --help|-h)
      printf 'Usage: %s [--copy]\n' "$0"
      printf 'Installs `space` CLI to %s/space without root.\n' "$BIN_DIR"
      exit 0
      ;;
  esac
done

say() { printf '[SPACE-INSTALL] %s\n' "$*"; }

# Verify Python 3.10+ is available
PY_BIN=""
for cand in python3 python3.13 python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY_BIN=$(command -v "$cand")
      break
    fi
  fi
done
if [ -z "$PY_BIN" ]; then
  say "Error: Python 3.10+ is required. Install python3 and re-run."
  exit 1
fi

mkdir -p "$BIN_DIR"

# Detect local repository checkout (when run from repo or SPACE_SOURCE_DIR is set)
SOURCE_DIR="${SPACE_SOURCE_DIR:-}"
if [ -z "$SOURCE_DIR" ]; then
  SCRIPT_PATH="$0"
  if [ -f "$SCRIPT_PATH" ]; then
    CAND_DIR=$(cd "$(dirname "$SCRIPT_PATH")/.." 2>/dev/null && pwd || true)
    if [ -n "$CAND_DIR" ] && [ -f "$CAND_DIR/cli/space_cli.py" ] && [ -f "$CAND_DIR/scripts/space" ]; then
      SOURCE_DIR="$CAND_DIR"
    fi
  fi
fi

if [ -n "$SOURCE_DIR" ] && [ -f "$SOURCE_DIR/cli/space_cli.py" ]; then
  if [ "$COPY_MODE" -eq 1 ]; then
    say "Copying Space CLI package from $SOURCE_DIR -> $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR/cli" "$INSTALL_DIR/scripts/space"
    mkdir -p "$INSTALL_DIR/scripts"
    cp -R "$SOURCE_DIR/cli" "$INSTALL_DIR/cli"
    cp "$SOURCE_DIR/scripts/space" "$INSTALL_DIR/scripts/space"
    if [ -f "$SOURCE_DIR/scripts/join.py" ]; then
      cp "$SOURCE_DIR/scripts/join.py" "$INSTALL_DIR/scripts/join.py"
    fi
    chmod +x "$INSTALL_DIR/scripts/space"
    TARGET_EXEC="$INSTALL_DIR/scripts/space"
  else
    chmod +x "$SOURCE_DIR/scripts/space"
    TARGET_EXEC="$SOURCE_DIR/scripts/space"
  fi
else
  say "Downloading Space AI CLI bundle into $INSTALL_DIR..."
  TMP_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t space-cli)
  trap 'rm -rf "$TMP_DIR"' EXIT INT TERM
  ARCHIVE="$TMP_DIR/space-cli.tar.gz"
  SHA_FILE="$TMP_DIR/space-cli.sha256"

  if [ -n "$REPO_URL" ] && command -v git >/dev/null 2>&1; then
    if [ -d "$INSTALL_DIR/.git" ]; then
      git -C "$INSTALL_DIR" pull --ff-only -q || true
    else
      rm -rf "$INSTALL_DIR"
      git clone --depth 1 -q "$REPO_URL" "$INSTALL_DIR"
    fi
  else
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "$BUNDLE_URL" -o "$ARCHIVE"
      curl -fsSL "$SHA_URL" -o "$SHA_FILE" 2>/dev/null || true
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "$ARCHIVE" "$BUNDLE_URL"
      wget -qO "$SHA_FILE" "$SHA_URL" 2>/dev/null || true
    else
      "$PY_BIN" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$BUNDLE_URL" "$ARCHIVE"
      "$PY_BIN" -c 'import sys, urllib.request
try:
  urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
except Exception:
  pass' "$SHA_URL" "$SHA_FILE"
    fi

    if [ -s "$SHA_FILE" ]; then
      EXPECTED_SHA=$(awk '{print $1}' "$SHA_FILE" | tr -d '\r\n ')
      ACTUAL_SHA=$("$PY_BIN" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ARCHIVE")
      if [ -n "$EXPECTED_SHA" ] && [ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]; then
        say "Error: SHA-256 verification failed for space-cli.tar.gz (expected $EXPECTED_SHA, got $ACTUAL_SHA)."
        exit 1
      fi
      say "Verified bundle SHA-256: $(printf '%.16s' "$ACTUAL_SHA")..."
    fi

    rm -rf "$INSTALL_DIR/cli" "$INSTALL_DIR/scripts/space"
    mkdir -p "$INSTALL_DIR"
    "$PY_BIN" -c 'import os, sys, tarfile
arc, dest = sys.argv[1], os.path.abspath(sys.argv[2])
with tarfile.open(arc, "r:gz") as tf:
  for m in tf.getmembers():
    target = os.path.abspath(os.path.join(dest, m.name))
    if not (target == dest or target.startswith(dest + os.sep)):
      raise SystemExit("Unsafe tar member: " + m.name)
  tf.extractall(dest)
' "$ARCHIVE" "$INSTALL_DIR"
  fi
  chmod +x "$INSTALL_DIR/scripts/space"
  TARGET_EXEC="$INSTALL_DIR/scripts/space"
fi

# Install thin zero-dependency wrapper or symlink at ~/.local/bin/space
ln -sf "$TARGET_EXEC" "$BIN_DIR/space"
chmod +x "$BIN_DIR/space"

VER=$("$BIN_DIR/space" --version 2>&1 || true)
say "Installed $VER -> $BIN_DIR/space"

case ":${PATH:-}:" in
  *":$BIN_DIR:"*)
    say "Ready! Try: space adapters  |  space claude --dry-run  |  space up --auto-pause"
    ;;
  *)
    say "Note: $BIN_DIR is not currently on your PATH."
    say "Add it for this shell with:  export PATH=\"$BIN_DIR:\$PATH\""
    say "(This installer never edits ~/.bashrc or ~/.zshrc automatically.)"
    ;;
esac
