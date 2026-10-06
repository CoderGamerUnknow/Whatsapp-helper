"""Whitelist enforcement (spec §3): only the 3 family numbers get through."""

from __future__ import annotations

from family_whatsapp_bot import config
from family_whatsapp_bot.tests.conftest import DAD, MOM, STRANGER, USER, from_stranger


# --------------------------------------------------------------------- unit
def test_exactly_three_whitelisted_numbers():
    assert len(config.WHITELISTED_NUMBERS) == 3


def test_phone_to_user_mapping_order():
    assert config.PHONE_TO_USER == {DAD: "dad", MOM: "mom", USER: "user"}


def test_resolve_user_accepts_all_three():
    assert config.resolve_user(DAD) == "dad"
    assert config.resolve_user(MOM) == "mom"
    assert config.resolve_user(USER) == "user"


def test_resolve_user_tolerates_formatting():
    assert config.resolve_user("+1 234-567-8901") == "dad"
    assert config.resolve_user("12345678902") == "mom"


def test_resolve_user_rejects_strangers():
    assert config.resolve_user(STRANGER) is None
    assert config.resolve_user("+12345678904") is None
    assert config.resolve_user("") is None
    assert config.resolve_user(None) is None
    assert config.resolve_user("not-a-number") is None


# ------------------------------------------------------- webhook integration
def test_incoming_from_stranger_is_dropped(client):
    resp = client.post("/api/incoming", json=from_stranger())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ignored"
    assert body["reason"] == "not_whitelisted"
    assert "user" not in body


def test_incoming_from_family_is_accepted(client):
    resp = client.post("/api/incoming", json={"from": DAD, "type": "text", "text": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["user"] == "dad"


def test_stranger_never_reaches_queue(client):
    """A dropped message must not enqueue any work (queue stays empty)."""
    resp = client.post("/api/incoming", json=from_stranger())
    assert resp.json()["status"] == "ignored"
    health = client.get("/health").json()
    assert health["queue_depth"] == 0


def test_simulate_also_enforces_whitelist(client):
    resp = client.post("/api/simulate", json=from_stranger(text="log 10 on food"))
    assert resp.json()["status"] == "ignored"
    assert client.get("/api/expenses").json()["expenses"] == []
