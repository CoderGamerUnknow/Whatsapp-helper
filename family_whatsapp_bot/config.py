"""Central configuration for the WhatsApp Family Assistant Bot.

Everything the rest of the app needs lives here: the hard whitelist of the
three family numbers, user IDs, Ollama endpoints, bridge endpoints and
database location. Values can be overridden through environment variables or
a local ``.env`` file (optional — the bot runs fine without one).

Design notes
------------
* The whitelist is intentionally *hardcoded* (spec requirement). It can still
  be overridden by an operator via the ``WHITELISTED_NUMBERS`` env var
  (comma separated) without touching code.
* Exactly three numbers are allowed and they map, in order, to the user IDs
  ``dad``, ``mom`` and ``user`` — every DB row is tagged with that ID.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loading (optional dependency — python-dotenv if installed, else a tiny
# built-in parser so the project has zero hard dependencies beyond FastAPI).
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from *path* into os.environ (without overriding
    variables that are already set in the real environment)."""
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key:
                os.environ.setdefault(key, value)
    except OSError:  # pragma: no cover - unreadable .env should never crash boot
        pass


_load_env_file(_ENV_PATH)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# 1. SECURITY — whitelist of exactly 3 family members (CRITICAL, per spec)
# ---------------------------------------------------------------------------
WHITELISTED_NUMBERS: list[str] = [
    "+12345678901",
    "+12345678902",
    "+12345678903",
]

# Operator override: WHITELISTED_NUMBERS="+15550001111,+15550002222,+15550003333"
_env_whitelist = _env("WHITELISTED_NUMBERS", "")
if _env_whitelist:
    WHITELISTED_NUMBERS = [n.strip() for n in _env_whitelist.split(",") if n.strip()]

# In-order mapping number -> user id used to tag every DB action.
USER_IDS: tuple[str, ...] = ("dad", "mom", "user")
PHONE_TO_USER: dict[str, str] = {
    phone: USER_IDS[i] for i, phone in enumerate(WHITELISTED_NUMBERS) if i < len(USER_IDS)
}
# Reverse map, for replying to the right number.
USER_TO_PHONE: dict[str, str] = {user: phone for phone, user in PHONE_TO_USER.items()}


def resolve_user(phone_number: str | None) -> str | None:
    """Return the user id ('dad' | 'mom' | 'user') for a sender number.

    Returns ``None`` for anything not on the whitelist — callers must drop
    the message in that case. Numbers are compared on their digits only, so
    '+1 234-567-8901' and '12345678901' both match the same entry.
    """
    if not phone_number:
        return None
    digits = "".join(ch for ch in str(phone_number) if ch.isdigit())
    if not digits:
        return None
    for phone, user_id in PHONE_TO_USER.items():
        allowed = "".join(ch for ch in phone if ch.isdigit())
        if digits == allowed or digits.endswith(allowed) or allowed.endswith(digits):
            return user_id
    return None


def is_whitelisted(phone_number: str | None) -> bool:
    """Convenience boolean wrapper around :func:`resolve_user`."""
    return resolve_user(phone_number) is not None


# ---------------------------------------------------------------------------
# 2. Ollama — ONE multimodal model handles both text and vision (VRAM fix).
#    'llama3.2-vision' does text + image in a single loaded model; swap to
#    'qwen2.5-vl' via env if preferred. Text-only alternatives: qwen2.5:7b /
#    llama3.1:8b (see README for the split-model opt-in).
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "llama3.2-vision")
# Keep the model resident in VRAM between calls so it is not re-loaded per message.
OLLAMA_KEEP_ALIVE: str = _env("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_TIMEOUT_SECONDS: float = float(_env("OLLAMA_TIMEOUT_SECONDS", "120"))
# Short probe timeout used by /health so the endpoint answers instantly.
OLLAMA_PROBE_TIMEOUT_SECONDS: float = float(_env("OLLAMA_PROBE_TIMEOUT_SECONDS", "2"))

# ---------------------------------------------------------------------------
# 3. WhatsApp bridge (Node.js / Baileys) — single integration path.
#    The bridge POSTs inbound messages to BACKEND_INBOUND_URL and the backend
#    POSTs outbound replies back to the bridge (BRIDGE_REPLY_URL).
# ---------------------------------------------------------------------------
BACKEND_HOST: str = _env("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT: int = int(_env("BACKEND_PORT", "8000"))
BACKEND_INBOUND_URL: str = _env(
    "BACKEND_INBOUND_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/incoming"
)
BRIDGE_HOST: str = _env("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT: int = int(_env("BRIDGE_PORT", "8790"))
BRIDGE_REPLY_URL: str = _env(
    "BRIDGE_REPLY_URL", f"http://{BRIDGE_HOST}:{BRIDGE_PORT}/bridge/send"
)
BRIDGE_TIMEOUT_SECONDS: float = float(_env("BRIDGE_TIMEOUT_SECONDS", "10"))
# Token shared between bridge and backend so only our own bridge can call in.
BRIDGE_SHARED_SECRET: str = _env("BRIDGE_SHARED_SECRET", "family-bridge-secret")

# ---------------------------------------------------------------------------
# 4. Transcription (faster-whisper) + media handling
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE: str = _env("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE: str = _env("WHISPER_DEVICE", "auto")          # auto|cpu|cuda
WHISPER_COMPUTE_TYPE: str = _env("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE: str | None = _env("WHISPER_LANGUAGE", "") or None
MEDIA_DIR: Path = Path(_env("MEDIA_DIR", str(Path(__file__).resolve().parent / "media")))

# ---------------------------------------------------------------------------
# 5. SQLite database
# ---------------------------------------------------------------------------
DB_PATH: Path = Path(_env("DB_PATH", str(Path(__file__).resolve().parent / "family_hub.db")))

# ---------------------------------------------------------------------------
# 6. Context memory / behaviour
# ---------------------------------------------------------------------------
# Sliding window: only the last N messages per user are fed back to the LLM
# (prevents context-window bloat — review fix #6).
CONTEXT_WINDOW_SIZE: int = int(_env("CONTEXT_WINDOW_SIZE", "8"))
# Expense confirmation flow (review fix #5): a staged entry lives this long.
PENDING_CONFIRMATION_TTL_SECONDS: int = int(_env("PENDING_CONFIRMATION_TTL_SECONDS", "300"))
# Prices are formatted with this symbol in WhatsApp replies.
CURRENCY_SYMBOL: str = _env("CURRENCY_SYMBOL", "$")

# Which LLM roles to expose (all six intents are always available).
SUPPORTED_INTENTS: tuple[str, ...] = (
    "budget_log",
    "budget_query",
    "doc_summary",
    "tech_help",
    "recipe",
    "general_chat",
)

# Allowed spending categories (spec §6) + common synonyms normalised onto them.
ALLOWED_CATEGORIES: tuple[str, ...] = ("Groceries", "Utilities", "Gas", "Food", "Misc")
CATEGORY_SYNONYMS: dict[str, str] = {
    "grocery": "Groceries",
    "groceries": "Groceries",
    "supermarket": "Groceries",
    "market": "Groceries",
    "food store": "Groceries",
    "utilities": "Utilities",
    "utility": "Utilities",
    "electric": "Utilities",
    "electricity": "Utilities",
    "power": "Utilities",
    "water": "Utilities",
    "internet": "Utilities",
    "bill": "Utilities",
    "bills": "Utilities",
    "gas": "Gas",
    "fuel": "Gas",
    "petrol": "Gas",
    "diesel": "Gas",
    "gasoline": "Gas",
    "food": "Food",
    "restaurant": "Food",
    "restaurants": "Food",
    "dining": "Food",
    "takeout": "Food",
    "take-out": "Food",
    "misc": "Misc",
    "other": "Misc",
    "miscellaneous": "Misc",
}


def normalize_category(raw: str | None) -> str:
    """Map free-form category text onto one of ALLOWED_CATEGORIES."""
    if not raw:
        return "Misc"
    key = str(raw).strip().lower()
    if key in {c.lower() for c in ALLOWED_CATEGORIES}:
        return key.capitalize() if key != "misc" else "Misc"
    if key in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[key]
    for syn, canon in CATEGORY_SYNONYMS.items():
        if syn in key:
            return canon
    return "Misc"


def guess_category_from_text(text: str) -> str:
    """Best-effort category when neither the model nor the caller supplied a
    usable one: scan the raw message for the longest matching synonym.

    Longest-first so 'electric bill' beats 'bill' → Utilities, and
    'gas station food' prefers the most specific signal available.
    """
    lowered = (text or "").lower()
    if not lowered:
        return "Misc"
    best_key = ""
    best_canon = "Misc"
    for syn, canon in sorted(CATEGORY_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if syn in lowered and len(syn) > len(best_key):
            best_key, best_canon = syn, canon
    return best_canon
