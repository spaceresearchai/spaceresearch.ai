#!/usr/bin/env sh
# Space Research — 1-Line NVIDIA GPU & Compute-Barter Worker Installer (Linux / WSL2)
# Uses prebuilt portable CUDA llama-server (FlashAttention + Q8 KV + speculative decoding) — zero PyTorch bloat!
# Usage:
#   curl -fsSL https://spaceresearch.ai/join-gpu.sh | sh -s -- --name my-gpu --auto-pause
set -eu

SPACE_HOME="${SPACE_HOME:-${LONGINUS_HOME:-$HOME/.space_research}}"
VENV_DIR="$SPACE_HOME/venv"
JOIN_URL="${SPACE_JOIN_URL:-${LONGINUS_JOIN_URL:-https://spaceresearch.ai/join.py}}"
JOIN_FILE="$SPACE_HOME/join.py"

say() { printf '[SPACE-GPU] %s\n' "$*"; }

mkdir -p "$SPACE_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1 || true)
  say "Detected NVIDIA GPU: ${GPU_LINE:-unknown}"
else
  say "Warning: nvidia-smi not found; will fall back to CPU mode unless CUDA is present."
fi

PY_BIN=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    PY_BIN=$(command -v "$cand")
    break
  fi
done
if [ -z "$PY_BIN" ]; then
  say "Error: Python 3.10+ is required. Install python3 and re-run."
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  say "Creating lightweight isolated environment at $VENV_DIR..."
  "$PY_BIN" -m venv "$VENV_DIR"
fi

say "Downloading latest auditable worker client ($JOIN_URL)..."
curl -fsSL "$JOIN_URL" -o "$JOIN_FILE"
if command -v sha256sum >/dev/null 2>&1; then
  say "SHA-256: $(sha256sum "$JOIN_FILE" | cut -d' ' -f1)"
fi

"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q cryptography

say "Launching Space Research NVIDIA CUDA worker (portable llama-server · Ctrl+C to stop anytime)..."
exec "$VENV_DIR/bin/python" "$JOIN_FILE" --install "$@"
