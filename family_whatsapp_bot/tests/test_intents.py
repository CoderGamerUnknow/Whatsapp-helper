"""Intent classification (spec §6) + WhatsApp markdown output rules."""

from __future__ import annotations

import json

import pytest

from family_whatsapp_bot import config, prompts
from family_whatsapp_bot.ollama_engine import OllamaEngine


# -------------------------------------------------------- fallback classify
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("spent 45 dollars on groceries", "budget_log"),
        ("I paid 1200 for electricity", "budget_log"),
        ("how much have we spent this month?", "budget_query"),
        ("what's our total budget left?", "budget_query"),
        ("summarize this electricity bill", "doc_summary"),
        ("the wifi is not working", "tech_help"),
        ("my printer won't connect", "tech_help"),
        ("give me a recipe with eggs and spinach", "recipe"),
        ("what can I cook with chicken?", "recipe"),
        ("hello there", "general_chat"),
        ("", "general_chat"),
    ],
)
def test_fallback_classify(text, expected):
    assert prompts.fallback_classify(text) == expected


def test_fallback_replies_are_non_empty():
    summary = {"household_total": 42.0, "user_total": 20.0,
               "by_category": {"Groceries": 42.0}}
    for intent in config.SUPPORTED_INTENTS:
        reply = prompts.fallback_reply(intent, "sample text", summary)
        assert reply.strip(), intent


def test_fallback_budget_query_contains_amounts():
    summary = {"household_total": 42.5, "user_total": 20.0, "by_category": {"Gas": 5.0}}
    reply = prompts.fallback_reply("budget_query", "totals?", summary)
    assert "42.50" in reply and "20.00" in reply


# ----------------------------------------------------------- system prompt
def test_system_prompt_contains_json_contract():
    prompt = prompts.build_system_prompt("dad")
    assert "whatsapp_reply" in prompt
    assert "budget_log" in prompt
    assert "intent" in prompt
    for intent in config.SUPPORTED_INTENTS:
        assert intent in prompt


def test_system_prompt_grounding_block():
    prompt = prompts.build_system_prompt(
        "mom",
        summary={"household_total": 99.95, "user_total": 10.0, "by_category": {"Food": 99.95}},
        user_totals={"dad": 89.95, "mom": 10.0, "user": 0.0},
    )
    assert "99.95" in prompt
    assert "GROUNDING" in prompt


def test_system_prompt_image_flag():
    assert "ATTACHED A PHOTO" in prompts.build_system_prompt("dad", has_image=True)
    assert "ATTACHED A PHOTO" not in prompts.build_system_prompt("dad", has_image=False)


# ------------------------------------------------------- markdown contract
def test_fallback_doc_summary_has_three_bullet_labels():
    reply = prompts.fallback_reply("doc_summary", "bill text")
    for label in ("*Summary:*", "*Action Required:*", "*Due Date:*"):
        assert label in reply


def test_fallback_replies_use_whatsapp_markdown_only():
    """No ** bold, no __italic__, no markdown headers or code fences."""
    summary = {"household_total": 1.0, "user_total": 1.0, "by_category": {}}
    for intent in config.SUPPORTED_INTENTS:
        reply = prompts.fallback_reply(intent, "x", summary)
        assert "**" not in reply, intent
        assert "__" not in reply, intent
        assert "```" not in reply, intent
        assert not any(line.startswith("#") for line in reply.splitlines()), intent


# ------------------------------------------------------- JSON reply parsing
@pytest.mark.parametrize(
    "raw",
    [
        '{"intent": "recipe", "data": {}, "whatsapp_reply": "hi"}',
        '```json\n{"intent": "recipe", "data": {}, "whatsapp_reply": "hi"}\n```',
        'Sure! Here is the JSON:\n{"intent": "recipe", "data": {}, "whatsapp_reply": "hi"}\nHope that helps',
    ],
)
def test_parse_json_reply_tolerant(raw):
    parsed = OllamaEngine.parse_json_reply(raw)
    assert parsed is not None
    assert parsed["intent"] == "recipe"


def test_parse_json_reply_rejects_garbage():
    assert OllamaEngine.parse_json_reply("not json at all") is None
    assert OllamaEngine.parse_json_reply("") is None
    assert OllamaEngine.parse_json_reply("[1, 2, 3]") is None


def test_json_contract_shape_stable():
    """The documented contract keys must never drift."""
    keys = set(prompts.JSON_CONTRACT)
    assert keys == {"intent", "data", "whatsapp_reply"}
    assert set(prompts.JSON_CONTRACT["data"]) == {"amount", "category", "description"}
    json.dumps(prompts.JSON_CONTRACT)  # must stay serialisable
