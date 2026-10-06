"""db_manager tests: schema, CRUD, user tagging, totals, WAL concurrency."""

from __future__ import annotations

import threading

import pytest

from family_whatsapp_bot import config


def test_all_four_tables_exist(db):
    for table in ("users", "expenses", "reminders", "conversation_history"):
        assert db.count_table(table) >= 0  # raises if the table is missing


def test_users_seeded_from_whitelist(db):
    for phone, user_id in config.PHONE_TO_USER.items():
        row = db.get_user_by_phone(phone)
        assert row is not None, f"{phone} missing from users table"
        assert row["name"] == user_id


def test_log_expense_tags_user(db):
    db.log_expense("dad", 12.50, "Groceries", "milk and eggs")
    rows = db.get_expenses()
    assert len(rows) == 1
    assert rows[0]["user_id"] == "dad"
    assert rows[0]["amount"] == 12.50
    assert rows[0]["category"] == "Groceries"
    assert rows[0]["description"] == "milk and eggs"


def test_log_expense_normalizes_category(db):
    db.log_expense("mom", 5.0, "fuel")   # synonym -> Gas
    db.log_expense("mom", 5.0, "nonsense-category")
    cats = {r["category"] for r in db.get_expenses()}
    assert cats == {"Gas", "Misc"}


def test_log_expense_rejects_non_positive(db):
    with pytest.raises(ValueError):
        db.log_expense("dad", -5.0, "Misc")
    with pytest.raises(ValueError):
        db.log_expense("dad", 0.0, "Misc")


def test_category_summary_per_user_and_household(db):
    db.log_expense("dad", 30.0, "Groceries", "weekly shop")
    db.log_expense("dad", 20.0, "Utilities", "electricity")
    db.log_expense("mom", 10.0, "Groceries", "milk")
    db.log_expense("user", 5.0, "Gas", "fill-up")

    dad = db.get_category_summary("dad")
    assert dad["user_total"] == 50.0
    assert dad["by_category"] == {"Groceries": 30.0, "Utilities": 20.0}
    assert dad["household_total"] == 65.0

    household = db.get_category_summary()
    assert household["household_total"] == 65.0
    assert household["by_category"]["Groceries"] == 40.0

    totals = db.get_user_totals()
    assert totals == {"dad": 50.0, "mom": 10.0, "user": 5.0}


def test_reminder_crud(db):
    rid = db.add_reminder("mom", "pay electricity", "2026-10-15")
    pending = db.get_reminders(user_id="mom")
    assert len(pending) == 1
    assert pending[0]["task"] == "pay electricity"
    assert pending[0]["status"] == "pending"

    assert db.complete_reminder(rid) is True
    assert db.get_reminders(user_id="mom") == []
    assert len(db.get_reminders(user_id="mom", status="done")) == 1


def test_chat_history_roundtrip(db):
    db.save_chat_history("+12345678901", "user", "first")
    db.save_chat_history("+12345678901", "assistant", "second")
    db.save_chat_history("+12345678902", "user", "other person")

    rows = db.get_chat_history("+12345678901")
    assert len(rows) == 2
    assert {r["role"] for r in rows} == {"user", "assistant"}


def test_sliding_window_limits_context(db):
    """fix #6 — only the last CONTEXT_WINDOW_SIZE messages feed the LLM."""
    for i in range(20):
        db.save_chat_history("+12345678901", "user", f"msg {i}")
    rows = db.get_chat_history("+12345678901")
    assert len(rows) == config.CONTEXT_WINDOW_SIZE
    # Newest first from get_chat_history…
    assert rows[0]["content"] == "msg 19"
    # …and oldest-first for prompt building.
    ctx = db.get_context_messages("+12345678901")
    assert ctx[0]["content"] == f"msg {20 - config.CONTEXT_WINDOW_SIZE}"
    assert ctx[-1]["content"] == "msg 19"
    assert all(m["role"] == "user" for m in ctx)


def test_concurrent_writes_do_not_raise_locked(db):
    """fix #4 — WAL + busy_timeout: parallel writers must never see
    'database is locked'."""
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def writer(idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(10):
                db.log_expense("dad", 1.0 + idx, "Misc", f"t{idx}-{i}")
        except Exception as exc:  # noqa: BLE001 - collected for assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent writes failed: {errors}"
    assert db.count_table("expenses") == 80
    assert db.get_category_summary("dad")["user_total"] > 0
