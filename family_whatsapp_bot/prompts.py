"""Master system prompts for intent classification and every reply the bot sends.

The contract with Ollama is strict: every model response must be a single JSON
object of the shape::

    {
      "intent": "<budget_log|budget_query|doc_summary|tech_help|recipe|general_chat>",
      "data": {"amount": 0.0, "category": "", "description": ""},
      "whatsapp_reply": "<reply already formatted with WhatsApp markdown>"
    }

`build_system_prompt()` assembles that contract plus the family persona, the
whitelisted member speaking, available spending totals (grounding for
budget answers) and per-intent rules. A deterministic fallback classifier in
`ollama_engine.py` mirrors the same intent names so the bot keeps working
when Ollama is down.
"""

from __future__ import annotations

import json
from typing import Any

from . import config

JSON_CONTRACT: dict[str, Any] = {
    "intent": "<intent_name>",
    "data": {"amount": 0.0, "category": "", "description": ""},
    "whatsapp_reply": "<Formatted reply using WhatsApp markdown: *bold*, _italics_>",
}

INTENT_RULES: dict[str, str] = {
    "budget_log": (
        "The message reports a purchase or expense (amount + what it was for). "
        "Extract amount (float, no currency symbol), category (one of "
        f"{', '.join(config.ALLOWED_CATEGORIES)}) and a short item description. "
        "If the amount is missing, ask for it in whatsapp_reply and set intent "
        "to budget_log with amount 0.0."
    ),
    "budget_query": (
        "The message asks about spending: totals per category, per person, or "
        "the combined household total / remaining budget. Use the GROUNDING "
        "data below for all numbers — never invent amounts."
    ),
    "doc_summary": (
        "The message contains (or attaches) a bill, letter or official document. "
        "Reply with EXACTLY three WhatsApp bullets, in this order and wording: "
        "• *Summary:* …, • *Action Required:* …, • *Due Date:* … "
        "(use 'Unknown' when a field is not present in the document)."
    ),
    "tech_help": (
        "Home tech troubleshooting (TV, router, Wi-Fi, printer, phone, smart "
        "home). Give calm, conversational step-by-step help: at most 1–2 short "
        "numbered steps at a time, then ask how it went. Use prior turns of "
        "this conversation to avoid repeating steps already tried."
    ),
    "recipe": (
        "Quick meal recommendation from the ingredients the user lists. Reply "
        "with the dish name in *bold*, a short ingredient list and 3–5 numbered "
        "steps, total time under ~30 minutes. Only use plausible ingredients."
    ),
    "general_chat": (
        "Anything else: casual conversation, greetings, questions about the "
        "bot. Be warm and brief."
    ),
}

MARKDOWN_RULES = """\
WHATSAPP MARKDOWN (enforce in every reply):
- *bold* for headings/key values, _italics_ for asides — no ** or __.
- Numbered lists as "1. ", bullets as "• " (never "- " or "* " at line start).
- No markdown headers, code fences or tables.
- Keep replies short: max ~8 lines unless the user asks for detail.
- Use the family member's name/tag ({user_label}) and the local currency symbol ({currency})."""


def _intent_catalog() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in INTENT_RULES.items())


def grounding_block(summary: dict[str, Any] | None, user_totals: dict[str, float] | None) -> str:
    """Deterministic spending facts the model must quote verbatim (fix #5)."""
    if not summary:
        return "GROUNDING: no expenses logged yet."
    lines = [
        "GROUNDING (use these exact numbers, never invent others):",
        f"- Combined household total: {summary.get('household_total', 0.0):.2f}",
        f"- This speaker's total: {summary.get('user_total', 0.0):.2f}",
        "- By category: " + (
            json.dumps(summary.get("by_category", {})) or "{}"
        ),
    ]
    if user_totals:
        lines.append("- Per member: " + json.dumps(user_totals))
    return "\n".join(lines)


def build_system_prompt(
    user_id: str,
    has_image: bool = False,
    summary: dict[str, Any] | None = None,
    user_totals: dict[str, float] | None = None,
) -> str:
    """Assemble the master system prompt for one incoming message."""
    user_label = user_id.capitalize() if user_id != "user" else "You"
    intents = ", ".join(config.SUPPORTED_INTENTS)

    image_note = (
        "\nThe user ATTACHED A PHOTO of a document. Extract text yourself and "
        "treat it as doc_summary unless the user clearly asked otherwise."
        if has_image
        else ""
    )

    return f"""You are the Family Assistant inside a private WhatsApp family group.
Allowed speakers (whitelist): dad, mom, user. You are talking to: {user_id}.
{image_note}
INTENTS: {intents}
{_intent_catalog()}

RESPONSE CONTRACT — output ONE valid JSON object and nothing else (no prose, no fences):
{json.dumps(JSON_CONTRACT, indent=2)}

- "intent" must be one of the six intent names above.
- "data.amount" is a float > 0 for budget_log (0.0 otherwise); "data.category"
  must be one of {list(config.ALLOWED_CATEGORIES)}; "data.description" is a
  3–7 word item description.
- "whatsapp_reply" is the complete message the user receives on WhatsApp.

{MARKDOWN_RULES.format(user_label=user_label, currency=config.CURRENCY_SYMBOL)}

{grounding_block(summary, user_totals)}
"""


# ---------------------------------------------------------------------------
# Fallback (Ollama-down) intent hints — deterministic keyword classifier used
# by ollama_engine.OllamaEngine when the local model is unreachable.
# ---------------------------------------------------------------------------
FALLBACK_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("budget_log", ("spent", "paid", "bought", "cost", "log ", "expense", "purchase", "groceries for", "recharge", "bill of")),
    ("budget_query", ("total", "how much", "budget", "left", "spent so far", "summary of", "spending", "remaining")),
    ("doc_summary", ("summarize", "summarise", "summary of this", "invoice", "letter", "due date", "official", "statement")),
    ("tech_help", ("wifi", "wi-fi", "router", "internet", "printer", "tv", "television", "bluetooth", "not working", "won't", "cannot", "can't", "setup", "connect", "password", "remote")),
    ("recipe", ("recipe", "cook", "meal", "dinner", "lunch", "ingredients", "make with", "hungry", "eat")),
]

# Question-shaped spend queries outrank the plain spend verbs in budget_log
# ("how much have we spent this month?" must NOT classify as budget_log).
_QUERY_KEYWORDS: tuple[str, ...] = (
    "how much", "total spent", "our total", "budget", "remaining",
    "spent so far", "summary of", "spending breakdown", "left to spend",
    "breakdown", "spent this", "spent in",
)


def fallback_classify(text: str) -> str:
    """Keyword-based intent classifier used when Ollama is unreachable."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return "general_chat"
    # 1) Spending *questions* first.
    if any(k in lowered for k in _QUERY_KEYWORDS):
        return "budget_query"
    # 2) An explicit amount with a spend verb is a strong budget_log signal.
    has_amount = any(ch.isdigit() for ch in lowered)
    spend_verbs = ("spent", "paid", "bought", "cost")
    if has_amount and any(v in lowered for v in spend_verbs):
        return "budget_log"
    # 3) Remaining keyword table.
    for intent, keywords in FALLBACK_INTENT_KEYWORDS:
        if any(k in lowered for k in keywords):
            return intent
    return "general_chat"


def fallback_reply(intent: str, text: str, summary: dict[str, Any] | None = None) -> str:
    """Deterministic WhatsApp-markdown reply used when Ollama is unreachable."""
    currency = config.CURRENCY_SYMBOL
    if intent == "budget_log":
        return (
            "*Budget log* (offline mode)\n"
            "I couldn't reach the local LLM, so I couldn't extract the amount "
            "automatically. Please resend in the form:\n"
            "_spent 24.50 on Groceries — milk, eggs_"
        )
    if intent == "budget_query":
        total = summary.get("household_total", 0.0) if summary else 0.0
        mine = summary.get("user_total", 0.0) if summary else 0.0
        cats = summary.get("by_category", {}) if summary else {}
        lines = ["*Spending summary*", f"• Household total: *{currency}{total:.2f}*",
                 f"• Yours: *{currency}{mine:.2f}*"]
        for cat, amt in list(cats.items())[:5]:
            lines.append(f"• {cat}: {currency}{amt:.2f}")
        lines.append("_Live totals from family_hub.db_")
        return "\n".join(lines)
    if intent == "doc_summary":
        return (
            "*Document summary*\n"
            "• *Summary:* I can't read the document while the local LLM is offline.\n"
            "• *Action Required:* Restart Ollama, then resend the photo.\n"
            "• *Due Date:* Unknown"
        )
    if intent == "tech_help":
        return (
            "*Tech help* (offline mode)\n"
            "1. Power-cycle the device: unplug 10 seconds, plug back in.\n"
            "2. Tell me what happened and I'll continue once Ollama is back."
        )
    if intent == "recipe":
        return (
            "*Quick recipe* (offline mode)\n"
            "• *Pasta:* garlic, oil, chilli, cheese.\n"
            "1. Boil pasta. 2. Fry garlic. 3. Toss and serve."
        )
    return (
        "*Family Assistant*\n"
        "The local LLM is offline — I'm running in fallback mode. "
        "Try: _spent 24.50 on Groceries_, _budget totals_, or _recipe with eggs_."
    )
