"""Expense guard (fix #5): regex extraction + Pydantic validation + confirm."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from family_whatsapp_bot.ollama_engine import (
    ExpenseData,
    ModelResponse,
    OllamaEngine,
    extract_amount_from_text,
)


# ------------------------------------------------------- amount extraction
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I spent $12.50 on groceries", Decimal("12.50")),
        ("paid 12,50 for bread", Decimal("12.50")),           # comma decimal
        ("the TV cost 1,234.56", Decimal("1234.56")),         # thousands sep
        ("bought 40 of gas", Decimal("40.00")),               # bare integer
        ("€7 coffee", Decimal("7.00")),
        ("no money mentioned here", None),
        ("", None),
    ],
)
def test_extract_amount(text, expected):
    assert extract_amount_from_text(text) == expected


def test_amount_not_misparsed_as_thousands():
    """$12.50 must NEVER become 1250."""
    assert extract_amount_from_text("spent $12.50 on milk") == Decimal("12.50")


# ------------------------------------------------------- Pydantic contract
def test_expense_data_accepts_valid():
    e = ExpenseData(amount="12.50", category="groceries", description=" milk ")
    assert e.amount == Decimal("12.50")
    assert e.category == "Groceries"       # normalised
    assert e.description == "milk"


def test_expense_data_rejects_bad_amount():
    with pytest.raises(ValidationError):
        ExpenseData(amount="twelve dollars", category="Food", description="x")


def test_expense_data_rejects_zero_and_negative():
    with pytest.raises(ValidationError):
        ExpenseData(amount=0, category="Food", description="x")
    with pytest.raises(ValidationError):
        ExpenseData(amount=-5, category="Food", description="x")


def test_expense_data_rejects_unknown_category_by_mapping_to_misc():
    e = ExpenseData(amount=3, category="banana", description="x")
    assert e.category == "Misc"


def test_model_response_intent_whitelisted():
    r = ModelResponse.model_validate({"intent": "HACK_ATTEMPT", "data": {}, "whatsapp_reply": "hi"})
    assert r.intent == "general_chat"


def test_model_response_parses_expense():
    r = ModelResponse.model_validate(
        {"intent": "budget_log",
         "data": {"amount": 9.99, "category": "Gas", "description": "fill up"},
         "whatsapp_reply": "ok"}
    )
    e = r.expense()
    assert e is not None and e.amount == Decimal("9.99") and e.category == "Gas"


def test_model_response_expense_none_on_garbage_data():
    r = ModelResponse.model_validate(
        {"intent": "budget_log", "data": {"amount": "a lot"}, "whatsapp_reply": "ok"}
    )
    assert r.expense() is None


# ------------------------------------------- agreement / confirmation gate
@pytest.fixture()
def engine() -> OllamaEngine:
    return OllamaEngine(base_url="http://127.0.0.1:1", model="test-model")


def _response(amount, category="Groceries", description="milk"):
    return ModelResponse(
        intent="budget_log",
        data={"amount": amount, "category": category, "description": description},
        whatsapp_reply="ok",
    )


def test_matching_amounts_insert_without_confirmation(engine):
    expense, pending, reply = engine.validate_expense(
        "spent $12.50 on groceries", _response(12.50)
    )
    assert pending is None and reply == ""
    assert expense is not None
    assert expense.amount == Decimal("12.50")
    assert expense.category == "Groceries"


def test_mismatch_triggers_confirmation_not_insert(engine):
    """Model says 1250 but the message says 12.50 → confirm first."""
    expense, pending, reply = engine.validate_expense(
        "spent $12.50 on groceries", _response(1250)
    )
    assert expense is None
    assert pending is not None
    assert pending.amount == Decimal("12.50")      # message wins
    assert "confirm" in reply.lower()


def test_model_invented_amount_triggers_confirmation(engine):
    """No amount anywhere in the message → never insert blindly."""
    expense, pending, reply = engine.validate_expense(
        "bought the usual groceries", _response(45.00)
    )
    assert expense is None and pending is not None
    assert pending.amount == Decimal("45.00")


def test_missing_amount_asks_user(engine):
    expense, pending, reply = engine.validate_expense(
        "bought the usual groceries", ModelResponse(
            intent="budget_log", data={}, whatsapp_reply=""
        )
    )
    assert expense is None and pending is None
    assert "amount" in reply.lower()


def test_model_json_failed_but_message_clear_stages_confirm(engine):
    """Model returned unparseable data but the raw text has an amount."""
    expense, pending, reply = engine.validate_expense(
        "spent 24.50 on Groceries", ModelResponse(
            intent="budget_log", data={"amount": "n/a"}, whatsapp_reply="x"
        )
    )
    assert expense is None
    assert pending is not None
    assert pending.amount == Decimal("24.50")
    assert pending.category == "Groceries"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("spent $24.50 on groceries, milk and eggs", "groceries, milk and eggs"),
        ("I paid 15.75 for gas fill-up", "gas fill-up"),
        ("bought bread", "bread"),
        ("log 9 coffee", "coffee"),
    ],
)
def test_clean_description(text, expected):
    from family_whatsapp_bot.ollama_engine import clean_description

    assert clean_description(text) == expected


def test_pending_confirmation_expiry_and_question(engine):
    from family_whatsapp_bot import config
    from family_whatsapp_bot.ollama_engine import PendingConfirmation

    p = PendingConfirmation("dad", Decimal("12.50"), "Groceries", "milk", "spent 12.50")
    assert not p.expired()
    assert "12.50" in p.question() and "yes" in p.question().lower()
    p.created_at -= config.PENDING_CONFIRMATION_TTL_SECONDS + 1
    assert p.expired()
