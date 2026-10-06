"""End-to-end pipeline tests through /api/simulate (offline engine).

Covers the four core intents (spec §4): budget_log, budget_query/summary,
doc_summary, tech_help, recipe — plus the confirmation gate, voice notes and
per-user context memory.
"""

from __future__ import annotations

import time
from pathlib import Path

from family_whatsapp_bot import config
from family_whatsapp_bot.tests.conftest import DAD, MOM, USER as USER_PHONE


def simulate(client, payload: dict) -> dict:
    resp = client.post("/api/simulate", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------ health
def test_health_shape(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["whitelist"] == config.WHITELISTED_NUMBERS
    assert body["users"] == config.PHONE_TO_USER
    assert "model_loaded" in body["ollama"]
    assert "faster_whisper" in body["whisper"]
    assert body["queue_depth"] == 0


# -------------------------------------------------------------- budget_log
def test_budget_log_writes_expense_and_totals(client, db):
    out = simulate(client, {"from": DAD, "type": "text",
                            "text": "spent $24.50 on groceries — milk, eggs"})
    assert out["status"] == "ok"
    assert out["intent"] == "budget_log"
    # Offline engine's regex extraction agrees with the message → direct insert.
    assert "*Expense logged*" in out["reply"]
    assert db.count_table("expenses") == 1

    row = db.get_expenses()[0]
    assert row["user_id"] == "dad"
    assert row["amount"] == 24.50
    assert row["category"] == "Groceries"
    # reply carries individual + combined totals (spec §7.5)
    assert "24.50" in out["reply"]


def test_budget_log_immediate_insert_without_confirmation(client, db):
    """Model output that MATCHES the raw text needs no confirmation (fix #5)."""
    out = simulate(client, {"from": MOM, "type": "text",
                            "text": "I paid $9.99 for gas fill-up"})
    assert "*Expense logged*" in out["reply"]
    assert "confirm" not in out["reply"].lower()
    rows = db.get_expenses()
    assert len(rows) == 1
    assert rows[0]["amount"] == 9.99
    assert rows[0]["user_id"] == "mom"
    assert rows[0]["category"] == "Gas"


def _hallucination_engine():
    """Engine that always claims $1250 regardless of the message → forces the
    confirmation gate (the model invented/misread the amount)."""
    from family_whatsapp_bot.ollama_engine import ModelResponse
    from family_whatsapp_bot.tests.conftest import ScriptedEngine

    return ScriptedEngine(ModelResponse(
        intent="budget_log",
        data={"amount": 1250.0, "category": "Groceries", "description": "milk"},
        whatsapp_reply="logged it",
    ))


def test_confirmation_required_when_model_disagrees(client_factory, db):
    """Model says 1250, message says 12.50 → staged, nothing saved yet."""
    c = client_factory(_hallucination_engine())
    out = simulate(c, {"from": DAD, "type": "text", "text": "spent $12.50 on groceries"})
    assert "Please confirm" in out["reply"]
    assert "12.50" in out["reply"]
    assert db.count_table("expenses") == 0


def test_confirmation_accepted_commits_message_amount(client_factory, db):
    """'yes' saves the MESSAGE's 12.50 — never the model's 1250 (fix #5)."""
    c = client_factory(_hallucination_engine())
    simulate(c, {"from": DAD, "type": "text", "text": "spent $12.50 on groceries"})
    out = simulate(c, {"from": DAD, "type": "text", "text": "yes"})
    assert "*Expense logged*" in out["reply"]
    rows = db.get_expenses()
    assert len(rows) == 1
    assert rows[0]["amount"] == 12.50
    assert rows[0]["user_id"] == "dad"


def test_confirmation_rejected_saves_nothing(client_factory, db):
    """A hallucinated amount never lands in SQLite without a 'yes'."""
    c = client_factory(_hallucination_engine())
    simulate(c, {"from": DAD, "type": "text", "text": "spent $12.50 on groceries"})
    out = simulate(c, {"from": DAD, "type": "text", "text": "no"})
    assert "discarded" in out["reply"].lower()
    assert db.count_table("expenses") == 0


def test_confirmation_per_user_isolated(client_factory, db):
    """Dad's pending confirmation must not hijack Mum's message."""
    from family_whatsapp_bot.ollama_engine import ModelResponse
    from family_whatsapp_bot.tests.conftest import ScriptedEngine

    # Mum's engine behaves honestly; dad's invents an amount.
    honest = ScriptedEngine(ModelResponse(
        intent="general_chat", data={}, whatsapp_reply="hi mum",
    ))
    c = client_factory(_hallucination_engine())
    simulate(c, {"from": DAD, "type": "text", "text": "spent $12.50 on groceries"})
    assert db.count_table("expenses") == 0

    # Mum says yes — must NOT commit dad's staged entry.
    c2 = client_factory(honest)
    out = simulate(c2, {"from": MOM, "type": "text", "text": "yes"})
    assert out["user"] == "mom"
    assert db.count_table("expenses") == 0

    # Dad confirms his own → only then does it land, tagged 'dad'.
    simulate(c, {"from": DAD, "type": "text", "text": "yes"})
    rows = db.get_expenses()
    assert len(rows) == 1 and rows[0]["user_id"] == "dad"


# -------------------------------------------------------------- budget_query
def test_budget_query_reports_db_totals(client, db):
    db.log_expense("dad", 30.0, "Groceries", "shop")
    db.log_expense("mom", 20.0, "Utilities", "power")
    out = simulate(client, {"from": USER_PHONE, "type": "text",
                            "text": "how much have we spent in total?"})
    assert out["intent"] == "budget_query"
    reply = out["reply"]
    assert "50.00" in reply       # combined household
    assert "30.00" in reply or "20.00" in reply


# ---------------------------------------------------------------- doc_summary
def test_doc_summary_three_bullets(client):
    out = simulate(client, {"from": DAD, "type": "text",
                            "text": "summarize this electricity bill due 15 Oct"})
    assert out["intent"] == "doc_summary"
    for label in ("*Summary:*", "*Action Required:*", "*Due Date:*"):
        assert label in out["reply"], f"missing {label} in: {out['reply']}"


# ------------------------------------------------------------------ tech_help
def test_tech_help_offline_steps(client):
    out = simulate(client, {"from": MOM, "type": "text", "text": "the wifi is not working"})
    assert out["intent"] == "tech_help"
    assert "1." in out["reply"]


# -------------------------------------------------------------------- recipe
def test_recipe_offline_reply(client):
    out = simulate(client, {"from": MOM, "type": "text",
                            "text": "give me a recipe with eggs and spinach"})
    assert out["intent"] == "recipe"
    assert "*" in out["reply"]  # bold dish name or heading


# --------------------------------------------------------------- general chat
def test_general_chat_fallback(client):
    out = simulate(client, {"from": DAD, "type": "text", "text": "good morning!"})
    assert out["intent"] == "general_chat"
    assert out["reply"].strip()


# ------------------------------------------------------------- conversation
def test_conversation_history_saved_per_number(client, db):
    simulate(client, {"from": DAD, "type": "text", "text": "good morning!"})
    simulate(client, {"from": DAD, "type": "text", "text": "another one"})
    rows = db.get_chat_history(DAD)
    assert len(rows) >= 2
    # sliding window cap
    assert len(rows) <= config.CONTEXT_WINDOW_SIZE


def test_history_isolated_between_users(client, db):
    simulate(client, {"from": DAD, "type": "text", "text": "dad speaking"})
    mom_rows = db.get_chat_history(MOM)
    assert all("dad speaking" not in r["content"] for r in mom_rows)


def test_context_memory_feeds_next_turn(client, db):
    """Last turns are stored and returned oldest-first (tech_help memory)."""
    simulate(client, {"from": DAD, "type": "text", "text": "my printer won't print"})
    simulate(client, {"from": DAD, "type": "text", "text": "the wifi is not working"})
    ctx = db.get_context_messages(DAD)
    assert any("printer" in m["content"] for m in ctx)
    assert ctx[0]["content"].endswith("print") or "printer" in ctx[0]["content"]


# ----------------------------------------------------------------- voice note
def test_voice_note_missing_file_graceful(client):
    out = simulate(client, {"from": DAD, "type": "audio",
                            "media_path": "/nonexistent/voice.ogg"})
    assert "transcribe" in out["reply"].lower() or "voice" in out["reply"].lower()


def test_voice_note_transcribes_then_logs(client, db, monkeypatch, tmp_path):
    """Fake the Whisper call → full audio pipeline executes."""
    from family_whatsapp_bot import server as server_mod

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"OggS fake")

    monkeypatch.setattr(server_mod.transcriber, "transcribe",
                        lambda p: "spent 7 dollars on gas")
    out = simulate(client, {"from": DAD, "type": "audio", "media_path": str(audio)})
    assert out["intent"] == "budget_log"
    # confirmation gate then commit
    if "Please confirm" in out["reply"]:
        out = simulate(client, {"from": DAD, "type": "text", "text": "yes"})
    assert db.count_table("expenses") == 1
    assert db.get_expenses()[0]["amount"] == 7.0


def test_audio_without_media_path_rejected_gracefully(client):
    out = simulate(client, {"from": DAD, "type": "audio", "text": ""})
    assert out["reply"]  # some human-readable reply, never a crash


# ------------------------------------------------------------ async webhook
def test_incoming_queues_and_worker_processes(client, db):
    """POST /api/incoming returns instantly; worker later persists the turn."""
    resp = client.post("/api/incoming",
                       json={"from": DAD, "type": "text", "text": "hi bot"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    assert resp.json()["queued_ms"] < 1000   # instant ACK (fix #1)

    deadline = time.time() + 10
    while time.time() < deadline and db.count_table("conversation_history") == 0:
        time.sleep(0.05)
    assert db.count_table("conversation_history") > 0


# ------------------------------------------------------------- image intake
def test_image_without_file_graceful(client):
    out = simulate(client, {"from": DAD, "type": "image",
                            "media_path": str(Path("/missing/doc.jpg")),
                            "caption": "summarize this bill"})
    assert out["reply"]


def test_empty_text_message_replies_help(client):
    out = simulate(client, {"from": DAD, "type": "text", "text": "   "})
    assert "didn't get any content" in out["reply"]
