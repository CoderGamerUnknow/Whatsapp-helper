"""Shared pytest fixtures: isolated temp DB, offline LLM engine, FastAPI client.

Every test gets a fresh SQLite file under tmp_path and an engine pointed at a
dead Ollama URL so the suite is fully deterministic/offline — the fallback
classifier (and the Pydantic/regex expense guard) do the work.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from family_whatsapp_bot import config
from family_whatsapp_bot.db_manager import FamilyDatabase
from family_whatsapp_bot.ollama_engine import OllamaEngine
from family_whatsapp_bot.server import create_app

# Handy constants for tests (valid whitelist + one intruder).
DAD = "+12345678901"
MOM = "+12345678902"
USER = "+12345678903"
STRANGER = "+15550009999"


def from_dad(**kw):
    payload = {"from": DAD, "type": "text", "text": "hello"}
    payload.update(kw)
    return payload


def from_mom(**kw):
    payload = {"from": MOM, "type": "text", "text": "hello"}
    payload.update(kw)
    return payload


def from_user(**kw):
    payload = {"from": USER, "type": "text", "text": "hello"}
    payload.update(kw)
    return payload


def from_stranger(**kw):
    payload = {"from": STRANGER, "type": "text", "text": "hello"}
    payload.update(kw)
    return payload


@pytest.fixture(autouse=True)
def _clear_pending_confirmations():
    """Pending expense confirmations are module-level state — reset per test
    so tests can't answer each other's staged entries."""
    from family_whatsapp_bot import server as server_mod
    server_mod._pending_confirmations.clear()
    yield
    server_mod._pending_confirmations.clear()


@pytest.fixture()
def db(tmp_path) -> FamilyDatabase:
    """Isolated SQLite database per test (WAL enabled by db_manager)."""
    database = FamilyDatabase(tmp_path / "family_hub.db")
    yield database
    database.close()


@pytest.fixture()
def offline_engine() -> OllamaEngine:
    """Engine pointing at a dead port → deterministic fallback behaviour."""
    return OllamaEngine(base_url="http://127.0.0.1:1", model="test-model")


@pytest.fixture()
def client(db, offline_engine) -> TestClient:
    app = create_app(db=db, engine=offline_engine)
    with TestClient(app) as c:
        yield c


class ScriptedEngine(OllamaEngine):
    """Engine that ignores Ollama and returns a scripted ModelResponse —
    simulates a hallucinating model for confirmation-flow tests.

    *response* may be a ModelResponse or a callable(user_text) -> ModelResponse.
    """

    def __init__(self, response) -> None:
        super().__init__(base_url="http://127.0.0.1:1", model="scripted")
        self.response = response
        self.calls = 0

    def classify(self, user_text: str, **kw):  # noqa: D401
        self.calls += 1
        if callable(self.response):
            return self.response(user_text)
        return self.response


@pytest.fixture()
def client_factory(db):
    """Build extra TestClients (e.g. with a scripted engine) against the
    same test database; all are shut down when the test ends."""
    opened: list[TestClient] = []

    def _make(engine=None):
        eng = engine or OllamaEngine(base_url="http://127.0.0.1:1", model="test-model")
        c = TestClient(create_app(db=db, engine=eng))
        c.__enter__()  # start lifespan/worker
        opened.append(c)
        return c

    yield _make
    for c in opened:
        c.__exit__(None, None, None)
