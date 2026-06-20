#!/usr/bin/env bash
# OllamAGI — Fedora setup script
# Run once on a fresh Fedora machine to install all prerequisites.
set -euo pipefail

echo "OllamAGI Setup — Fedora"
echo "═══════════════════════"

# ── System packages ──────────────────────────────────────────────────────────
echo "[1/5] Installing system packages…"
sudo dnf install -y python3 python3-pip git curl docker docker-compose

# ── Docker ───────────────────────────────────────────────────────────────────
echo "[2/5] Enabling Docker…"
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
echo "      NOTE: Log out and back in for Docker group to take effect."

# ── Ollama ───────────────────────────────────────────────────────────────────
echo "[3/5] Installing Ollama…"
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.ai/install.sh | sh
else
  echo "      Ollama already installed."
fi
sudo systemctl enable --now ollama

# ── Python deps ──────────────────────────────────────────────────────────────
echo "[4/5] Installing Python dependencies…"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip3 install --user -r "$SCRIPT_DIR/../requirements.txt"

# ── SSH key for agents ───────────────────────────────────────────────────────
echo "[5/5] Setting up agent SSH key…"
KEY="$HOME/.ssh/ollamagi_agent"
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "ollamagi-agent"
  cat "${KEY}.pub" >> "$HOME/.ssh/authorized_keys"
  chmod 600 "$HOME/.ssh/authorized_keys"
  echo "      SSH key created: $KEY"
else
  echo "      SSH key already exists: $KEY"
fi

# ── Pull recommended Ollama models ───────────────────────────────────────────
echo ""
echo "Recommended: pull at least one model before starting."
echo "  ollama pull vaultbox/qwen3.5-uncensored:27b  # all agent roles"
echo "  ollama pull mxbai-embed-large    # semantic search (optional)"
echo ""
echo "Setup complete. Start OllamAGI with:"
echo "  cp .env.example .env && nano .env"
echo "  python3 ollamagi.py serve"
echo ""
