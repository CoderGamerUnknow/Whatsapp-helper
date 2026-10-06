"""Ollama API client: prompt formatting, strict JSON parsing, expense validation.

Design (review fixes #3 and #5):
- ONE multimodal model (``llama3.2-vision`` by default) handles both text and
  image tasks, so Ollama never thrashes between a text model and a vision
  model in VRAM. All calls also run through the server's single worker queue,
  which serialises access to the model.
- Financial extraction is NOT trusted blindly: the model's JSON is parsed into
  a strict Pydantic model, the amount is re-extracted from the raw user text
  with regex, and the two must agree (within a small tolerance) before an
  INSERT is allowed. When they disagree — or the message has no explicit
  amount — a confirmation is staged ("Did you mean $12.50 for Groceries?")
  and the expense only lands in SQLite after the user replies "yes".

Endpoint: POST {OLLAMA_BASE_URL}/api/chat (stream=false, format=json).
"""

from __future__ import annotations

import json
import logging
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import config, prompts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strict schemas (review fix #5)
# ---------------------------------------------------------------------------
class ExpenseData(BaseModel):
    """Validated expense payload. Rejects nonsense before any SQL runs."""

    amount: Decimal = Field(gt=0, le=1_000_000)
    category: str = "Misc"
    description: str = ""

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        if isinstance(v, Decimal):
            return v
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            cleaned = v.strip().replace("$", "").replace(",", "")
            try:
                return Decimal(cleaned)
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"invalid amount: {v!r}") from exc
        raise ValueError(f"invalid amount type: {type(v).__name__}")

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        return config.normalize_category(v)

    @field_validator("description")
    @classmethod
    def _trim_description(cls, v: str) -> str:
        return (v or "").strip()[:120]


class ModelResponse(BaseModel):
    """The JSON contract Ollama must return (spec §6)."""

    intent: str
    data: dict[str, Any] = Field(default_factory=dict)
    whatsapp_reply: str = ""

    @field_validator("intent")
    @classmethod
    def _known_intent(cls, v: str) -> str:
        v = (v or "").strip().lower()
        return v if v in config.SUPPORTED_INTENTS else "general_chat"

    def expense(self) -> ExpenseData | None:
        """Parse data into ExpenseData; None when absent/invalid."""
        if not self.data:
            return None
        try:
            return ExpenseData.model_validate(self.data)
        except ValidationError:
            return None


# ---------------------------------------------------------------------------
# Regex amount extraction from the RAW message (ground truth for money)
# ---------------------------------------------------------------------------
_AMOUNT_PATTERNS = [
    # $12.50 / 12.50$ / €7 / £3.20 with optional thousands separators
    re.compile(r"(?<!\d)(?:[$€£¥]\s*)?(\d{1,3}(?:[,\.]\d{3})+(?:[,\.]\d{1,2})?|\d+[,\.]\d{1,2})(?!\d)"),
    # bare integers like "spent 40 on groceries"
    re.compile(r"(?<!\d)(\d{1,6})(?!\d)"),
]


def extract_amount_from_text(text: str) -> Decimal | None:
    """Pull the most plausible money amount out of free text.

    Handles "$12.50", "12,50" (comma decimal), "1,234.56" (thousands sep)
    and bare "40". Returns None when the text carries no amount at all.
    """
    if not text:
        return None
    for pattern in _AMOUNT_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group(1)
        candidate = _normalize_numeric(raw)
        if candidate is not None:
            return candidate
    return None


def _normalize_numeric(raw: str) -> Decimal | None:
    """Normalise a numeric token to a Decimal amount.

    Rules:
      "1,234.56" -> 1234.56   (comma = thousands)
      "12,50"    -> 12.50      (comma = decimal, exactly 2 digits after)
      "12.50"    -> 12.50
      "40"       -> 40.00
    """
    if not raw:
        return None
    s = raw.strip()
    try:
        if "," in s and "." in s:
            # Both separators: the last one is the decimal point.
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            tail = s.rsplit(",", 1)[1]
            s = s.replace(",", ".") if len(tail) in (1, 2) else s.replace(",", "")
        return Decimal(s).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def amounts_agree(a: Decimal, b: Decimal, tol: Decimal = Decimal("0.01")) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Pending confirmations (staged until the user answers yes/no)
# ---------------------------------------------------------------------------
class PendingConfirmation:
    __slots__ = ("user_id", "amount", "category", "description", "raw_text", "created_at")

    def __init__(self, user_id: str, amount: Decimal, category: str,
                 description: str, raw_text: str) -> None:
        self.user_id = user_id
        self.amount = amount
        self.category = category
        self.description = description
        self.raw_text = raw_text
        self.created_at = time.monotonic()

    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > config.PENDING_CONFIRMATION_TTL_SECONDS

    def question(self) -> str:
        cur = config.CURRENCY_SYMBOL
        return (
            "*Please confirm this expense*\n"
            f"Did you mean *{cur}{self.amount:.2f}* for _{self.category}_ "
            f"— {self.description or 'no description'}?\n"
            "Reply *yes* to save it or *no* to cancel."
        )


# ---------------------------------------------------------------------------
# Ollama engine
# ---------------------------------------------------------------------------
class OllamaEngine:
    """Talks to the local Ollama server; degrades to deterministic fallback."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self._client = httpx.Client(timeout=config.OLLAMA_TIMEOUT_SECONDS)

    # ------------------------------------------------------------- health
    def is_reachable(self) -> bool:
        try:
            r = self.client_get("/api/tags")
            return bool(r is not None)
        except Exception:
            return False

    def client_get(self, path: str) -> dict[str, Any] | None:
        try:
            with httpx.Client(timeout=config.OLLAMA_PROBE_TIMEOUT_SECONDS) as c:
                resp = c.get(self.base_url + path)
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return None

    def model_loaded(self) -> bool:
        """True when *self.model* is already resident (helps /health)."""
        data = self.client_get("/api/tags")
        if not data:
            return False
        return any(m.get("name") == self.model for m in data.get("models", []))

    # ------------------------------------------------------------ internal
    def _chat(self, messages: list[dict[str, Any]], *, want_json: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }
        if want_json:
            payload["format"] = "json"
        with self._client as c:
            resp = c.post(self.base_url + "/api/chat", json=payload)
            resp.raise_for_status()
            body = resp.json()
        return str(body.get("message", {}).get("content", ""))

    @staticmethod
    def parse_json_reply(raw: str) -> dict[str, Any] | None:
        """Extract a JSON object from a model reply, tolerating code fences
        and stray prose around the object."""
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        # Find the outermost {...} block.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    # ------------------------------------------------------------ public API
    def classify(
        self,
        user_text: str,
        *,
        user_id: str = "user",
        history: list[dict[str, str]] | None = None,
        summary: dict[str, Any] | None = None,
        user_totals: dict[str, float] | None = None,
        image_b64: str | None = None,
    ) -> ModelResponse:
        """Run the full classify+reply step. Never raises: on any failure the
        deterministic fallback classifier (prompts.fallback_*) answers."""
        system = prompts.build_system_prompt(
            user_id, has_image=image_b64 is not None, summary=summary, user_totals=user_totals
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        # Sliding-window context (review fix #6): only recent turns.
        for turn in (history or [])[-config.CONTEXT_WINDOW_SIZE:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])})

        user_msg: dict[str, Any] = {"role": "user", "content": user_text or "(see attached image)"}
        if image_b64:
            user_msg["images"] = [image_b64]
        messages.append(user_msg)

        if not self.is_reachable():
            logger.warning("Ollama unreachable — using fallback classifier")
            return self._fallback(user_text, summary)

        raw = ""
        for attempt in (1, 2):  # one retry on transient errors
            try:
                raw = self._chat(messages)
                break
            except Exception as exc:
                logger.error("Ollama call failed (attempt %d): %s", attempt, exc)
                if attempt == 2:
                    return self._fallback(user_text, summary)
                time.sleep(0.5)

        parsed = self.parse_json_reply(raw)
        if parsed is None:
            logger.error("Ollama returned non-JSON: %.200s", raw)
            return self._fallback(user_text, summary)

        try:
            response = ModelResponse.model_validate(parsed)
        except ValidationError as exc:
            logger.error("Ollama JSON failed validation: %s", exc)
            return self._fallback(user_text, summary)

        if not response.whatsapp_reply.strip():
            fb = self._fallback(user_text, summary)
            response.whatsapp_reply = fb.whatsapp_reply
        return response

    def _fallback(self, user_text: str, summary: dict[str, Any] | None) -> ModelResponse:
        intent = prompts.fallback_classify(user_text)
        reply = prompts.fallback_reply(intent, user_text, summary)
        data: dict[str, Any] = {}
        if intent == "budget_log":
            amt = extract_amount_from_text(user_text)
            if amt is not None:
                data = {
                    "amount": float(amt),
                    "category": _guess_category(user_text),
                    "description": clean_description(user_text),
                }
        return ModelResponse(intent=intent, data=data, whatsapp_reply=reply)

    # ------------------------------------------------- expense safety gate
    def validate_expense(
        self,
        raw_text: str,
        response: ModelResponse,
    ) -> tuple[ExpenseData | None, PendingConfirmation | None, str]:
        """Cross-check model extraction against the raw message (review fix #5).

        Returns ``(expense, pending, reply)``:
          - expense set, pending None  -> safe to INSERT immediately
          - expense None, pending set  -> ask the user to confirm first
          - both None                  -> reply explains what is missing
        """
        expense = response.expense()
        regex_amount = extract_amount_from_text(raw_text)

        if expense is None:
            if regex_amount is None:
                return None, None, (
                    "*Budget log*\nI couldn't spot an amount in that message. "
                    "Please resend like: _spent 24.50 on Groceries — milk_"
                )
            # Model failed but the message itself has a clear amount.
            candidate = ExpenseData(
                amount=regex_amount,
                category=_guess_category(raw_text),
                description=clean_description(raw_text),
            )
            pending = PendingConfirmation(
                "pending", candidate.amount, candidate.category, candidate.description, raw_text
            )
            return None, pending, pending.question()

        # Model produced a payload — compare against raw text.
        if expense.category == "Misc":
            # Model gave no usable category: infer one from the message.
            expense = expense.model_copy(
                update={"category": config.guess_category_from_text(raw_text)}
            )
        if regex_amount is None:
            # No amount in the message: the model invented it. Confirm first.
            pending = PendingConfirmation(
                "pending", expense.amount, expense.category, expense.description, raw_text
            )
            return None, pending, pending.question()

        if amounts_agree(expense.amount, regex_amount):
            # Agrees with the message (also fixes "$12.50 parsed as 1250").
            expense = expense.model_copy(update={"amount": regex_amount})
            return expense, None, ""

        # Disagrees -> never write automatically; ask.
        mismatch = ExpenseData.model_validate(
            {**expense.model_dump(), "amount": regex_amount}
        )
        pending = PendingConfirmation(
            "pending", mismatch.amount, mismatch.category, mismatch.description, raw_text
        )
        return None, pending, pending.question()


def _guess_category(text: str) -> str:
    return config.guess_category_from_text(text)


# Leading verbs/filler stripped when deriving an item description from raw text.
_DESC_PREFIX = re.compile(
    r"^(?:i\s+)?(?:we\s+)?(?:just\s+)?(?:spent|paid|bought|purchased|log(?:ged)?|"
    r"record(?:ed)?|cost|price(?:d)?|was)\b[:\s]*",
    re.IGNORECASE,
)
_DESC_AMOUNT = re.compile(r"[$€£¥]\s*\d[\d,.]*|\b\d+[.,]\d{2}\b|\b\d+\b")
_DESC_CONNECTOR = re.compile(r"^\s*(?:on|for|at|to)\b[:\s]*", re.IGNORECASE)


def clean_description(text: str, limit: int = 70) -> str:
    """Derive a short item description from a raw message.

    "spent $24.50 on groceries, milk and eggs" -> "groceries, milk and eggs"
    """
    s = (text or "").strip()
    s = _DESC_PREFIX.sub("", s)
    s = _DESC_AMOUNT.sub("", s)
    s = _DESC_CONNECTOR.sub("", s.strip(" ,.-"))
    s = re.sub(r"\s+", " ", s).strip(" ,.-")
    return s[:limit] or (text or "").strip()[:limit]
