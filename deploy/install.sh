#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WhatsApp Family Assistant Bot — one-shot Linux installer / health check.
#
#   sudo ./deploy/install.sh              full install (apt, venv, pip, npm,
#                                         ollama, systemd units)
#   ./deploy/install.sh --check           diagnose an existing install (no
#                                         changes, safe to run anytime)
#   sudo ./deploy/install.sh --no-ollama  skip Ollama install + model pull
#
# Tested target: Debian/Ubuntu (apt). On other distros install ffmpeg,
# python3.12+, nodejs 18+ yourself and re-run with --no-ollama.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
SERVICE_USER="${SERVICE_USER:-familybot}"
WITH_OLLAMA=1
CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --check)      CHECK_ONLY=1 ;;
    --no-ollama)  WITH_OLLAMA=0 ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

ok()   { printf '  \033[32m✔\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✘\033[0m %s\n' "$1"; FAILED=1; }
FAILED=0

need_cmd() {
  if command -v "$1" >/dev/null 2>&1; then ok "$1 ($(command -v "$1"))"
  else fail "$1 missing${2:+ — $2}"; fi
}

# ---------------------------------------------------------------------------
if [[ $CHECK_ONLY -eq 1 ]]; then
  echo "== environment check =="
  need_cmd python3 "install python3.11+"
  need_cmd ffmpeg   "sudo apt install -y ffmpeg"
  need_cmd node     "install Node.js 18+"
  need_cmd npm
  need_cmd ollama   "curl -fsSL https://ollama.com/install.sh | sh"
  need_cmd systemctl

  echo "== python environment =="
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    ok "venv at $VENV_DIR"
    "$VENV_DIR/bin/python" - <<'PY' || FAILED=1
import importlib.util as u
for mod in ("fastapi", "uvicorn", "httpx", "pydantic", "faster_whisper", "pytest"):
    print(("  \033[32m✔\033[0m " if u.find_spec(mod) else "  \033[31m✘\033[0m ") + mod)
    if not u.find_spec(mod):
        raise SystemExit(1)
PY
  else
    fail "venv missing — run sudo ./deploy/install.sh"
  fi

  echo "== bridge =="
  if [[ -d "$REPO_DIR/bridge/node_modules" ]]; then
    ok "bridge/node_modules present"
  else
    fail "bridge deps missing — (cd bridge && npm install)"
  fi

  echo "== services =="
  for svc in family-bot family-bridge ollama; do
    if systemctl is-enabled "$svc" >/dev/null 2>&1; then ok "$svc enabled"
    else warn "$svc not enabled"; fi
    if systemctl is-active "$svc" >/dev/null 2>&1; then ok "$svc running"
    else warn "$svc not running"; fi
  done

  echo "== endpoints =="
  if curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    ok "backend /health reachable"
    curl -fsS -m 3 http://127.0.0.1:8000/health | head -c 400; echo
  else
    warn "backend not answering on :8000"
  fi
  if curl -fsS -m 3 http://127.0.0.1:8790/bridge/health >/dev/null 2>&1; then
    ok "bridge /bridge/health reachable"
  else
    warn "bridge not answering on :8790"
  fi
  if curl -fsS -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ok "ollama /api/tags reachable"
  else
    warn "ollama not answering on :11434"
  fi

  echo "== config =="
  if [[ -f "$REPO_DIR/.env" ]]; then ok ".env present"
  else warn ".env missing (defaults in use — see deploy/.env.example)"; fi

  [[ $FAILED -eq 0 ]] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
  exit $FAILED
fi

# ------------------------------- full install ------------------------------
echo "== 1/7 system packages (needs sudo) =="
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y ffmpeg python3 python3-venv python3-pip curl
else
  warn "apt-get not found — make sure ffmpeg + python3 + curl are installed"
fi

echo "== 2/7 service user =="
if id "$SERVICE_USER" >/dev/null 2>&1; then ok "$SERVICE_USER exists"
else sudo useradd --system --home "$REPO_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
     ok "created $SERVICE_USER"; fi

echo "== 3/7 python venv + deps =="
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
if ! "$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt" -q; then
  warn "full requirements failed (faster-whisper can be picky) — installing core deps"
  "$VENV_DIR/bin/pip" install 'fastapi>=0.115' 'uvicorn[standard]>=0.30' \
      'httpx>=0.27' 'pydantic>=2.7' pytest -q
fi
ok "python deps installed"

echo "== 4/7 bridge deps =="
(cd "$REPO_DIR/bridge" && npm install --omit=dev --no-audit --no-fund)
ok "node deps installed"

echo "== 5/7 environment file =="
if [[ ! -f "$REPO_DIR/.env" ]]; then
  cp "$REPO_DIR/deploy/.env.example" "$REPO_DIR/.env"
  # Generate a random shared secret so bridge/backend can't be spoofed.
  SECRET="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
  sed -i "s|BRIDGE_SHARED_SECRET=.*|BRIDGE_SHARED_SECRET=$SECRET|" "$REPO_DIR/.env"
  chmod 600 "$REPO_DIR/.env"
  ok ".env created (random BRIDGE_SHARED_SECRET) — EDIT THE WHITELIST NOW"
else
  ok ".env already present"
fi
mkdir -p "$REPO_DIR/family_whatsapp_bot/media" /tmp/family-bot-media
chown -R "$SERVICE_USER:$SERVICE_USER" "$REPO_DIR" /tmp/family-bot-media 2>/dev/null || true

echo "== 6/7 ollama + model =="
if [[ $WITH_OLLAMA -eq 1 ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  ok "ollama installed"
  MODEL="$(grep -E '^OLLAMA_MODEL=' "$REPO_DIR/.env" 2>/dev/null | cut -d= -f2 || true)"
  MODEL="${MODEL:-llama3.2-vision}"
  echo "  pulling $MODEL (can take a while, ~7GB)…"
  ollama pull "$MODEL" || warn "model pull failed — run later: ollama pull $MODEL"
else
  warn "skipped ollama (--no-ollama)"
fi

echo "== 7/7 systemd units =="
if command -v systemctl >/dev/null 2>&1; then
  [[ $WITH_OLLAMA -eq 1 ]] && sudo cp "$REPO_DIR/deploy/ollama.service" /etc/systemd/system/
  sudo cp "$REPO_DIR/deploy/family-bot.service" /etc/systemd/system/
  sudo cp "$REPO_DIR/deploy/family-bridge.service" /etc/systemd/system/
  sudo systemctl daemon-reload
  [[ $WITH_OLLAMA -eq 1 ]] && sudo systemctl enable --now ollama
  sudo systemctl enable --now family-bot
  sudo systemctl enable --now family-bridge
  ok "services enabled and started"
else
  warn "systemctl not found — start manually: see README"
fi

echo
echo "Install complete. Next steps:"
echo "  1. Edit $REPO_DIR/.env  (WHITELISTED_NUMBERS, secrets)"
echo "  2. Scan the QR code:  journalctl -u family-bridge -f"
echo "  3. Verify:            $REPO_DIR/deploy/install.sh --check"
