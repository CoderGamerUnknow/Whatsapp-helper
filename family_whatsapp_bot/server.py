"""FastAPI entry point: webhook/listener, whitelist gate, async worker queue.

Flow (review fix #1 — Meta's 5s webhook deadline vs. 5-15s local inference):

    bridge (Baileys) ──POST /api/incoming──▶ backend
                                             │  whitelist check + validate
                                             │  → 202 Accepted  (instant, <100ms)
                                             ▼
                                      asyncio.Queue
                                             │  single background worker
                                             │  (serialises all Ollama calls → fix #3)
                                             ▼
              transcribe? → classify → validate expense → SQLite → reply
                                             │
                    POST /bridge/send ◀──────┘   (delivered by the Node bridge)

Endpoints
---------
GET  /health          — Ollama / whisper / db / queue diagnostics
POST /api/incoming    — the webhook the bridge posts to (202 + queue)
POST /api/simulate    — local dev/test: same pipeline, returns reply inline
GET  /api/expenses    — quick look at logged expenses (debugging)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import config, db_manager, prompts, transcriber
from .db_manager import FamilyDatabase, get_database
from .ollama_engine import OllamaEngine, PendingConfirmation

logger = logging.getLogger(__name__)

GENERIC_ERROR = (
    "*Something went wrong* ⚠️\n"
    "I couldn't process that just now — please try again in a moment."
)

# Per-user staged expense confirmations (review fix #5).
_pending_confirmations: dict[str, PendingConfirmation] = {}


# ---------------------------------------------------------------------------
# Inbound payload model (what the Node bridge posts)
# ---------------------------------------------------------------------------
class IncomingMessage(BaseModel):
    from_: str = Field(alias="from", min_length=1)
    type: str = "text"  # text | audio | image | document
    text: str = ""
    media_path: str | None = None
    message_id: str | None = None
    caption: str | None = None

    model_config = {"populate_by_name": True}


class SendPayload(BaseModel):
    to: str
    reply: str


# ---------------------------------------------------------------------------
# App factory with lifespan-managed worker
# ---------------------------------------------------------------------------
def create_app(
    db: FamilyDatabase | None = None,
    engine: OllamaEngine | None = None,
) -> FastAPI:
    database = db or get_database()
    llm = engine or OllamaEngine()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.queue = queue
        app.state.db = database
        app.state.engine = llm
        worker = asyncio.create_task(_worker_loop(app))
        app.state.worker = worker
        logger.info("backend started (model=%s, db=%s)", llm.model, database.db_path)
        try:
            yield
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            logger.info("backend stopped")

    app = FastAPI(title="WhatsApp Family Assistant Bot", version="1.0.0", lifespan=lifespan)

    # ------------------------------------------------------------------ health
    @app.get("/health")
    async def health() -> dict[str, Any]:
        ollama_ok = llm.is_reachable()
        return {
            "status": "ok" if ollama_ok else "degraded",
            "whitelist": list(config.WHITELISTED_NUMBERS),
            "users": dict(config.PHONE_TO_USER),
            "ollama": {
                "reachable": ollama_ok,
                "url": llm.base_url,
                "model": llm.model,
                "model_loaded": llm.model_loaded() if ollama_ok else False,
            },
            "whisper": transcriber.status(),
            "queue_depth": queue.qsize(),
            "db_path": str(database.db_path),
        }

    # ---------------------------------------------------------- inbound webhook
    @app.post("/api/incoming")
    async def incoming(message: IncomingMessage, request: Request) -> dict[str, Any]:
        started = time.perf_counter()
        user_id = config.resolve_user(message.from_)

        # ---- SECURITY: drop anything not on the 3-number whitelist (spec §3).
        if user_id is None:
            logger.warning(
                "DROPPED non-whitelisted sender %s (%s)",
                message.from_[:6] + "…" + message.from_[-3:], message.type,
            )
            # Acknowledge so the bridge doesn't retry; the message is discarded.
            return {"status": "ignored", "reason": "not_whitelisted"}

        job = {
            "user_id": user_id,
            "phone": message.from_,
            "type": message.type,
            "text": (message.text or message.caption or "").strip(),
            "media_path": message.media_path,
            "enqueued_at": time.perf_counter(),
        }
        try:
            queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.error("queue full — rejecting message from %s", user_id)
            raise HTTPException(status_code=503, detail="queue full") from None

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("202 accepted for %s in %.1fms", user_id, elapsed_ms)
        # ---- instant acknowledgement (review fix #1)
        return {"status": "accepted", "user": user_id, "queued_ms": round(elapsed_ms, 1)}

    # -------------------------------------------------- local dev/test endpoint
    @app.post("/api/simulate")
    async def simulate(message: IncomingMessage) -> dict[str, Any]:
        """Runs the same pipeline synchronously and returns the reply — handy
        for curl-based testing without the Node bridge."""
        user_id = config.resolve_user(message.from_)
        if user_id is None:
            return {"status": "ignored", "reason": "not_whitelisted"}
        job = {
            "user_id": user_id,
            "phone": message.from_,
            "type": message.type,
            "text": (message.text or message.caption or "").strip(),
            "media_path": message.media_path,
            "enqueued_at": time.perf_counter(),
        }
        result = await _process_job(app, job)
        return {"status": "ok", "user": user_id, **result}

    # ------------------------------------------------------------ debug listing
    @app.get("/api/expenses")
    async def list_expenses(limit: int = 20) -> dict[str, Any]:
        return {"expenses": database.get_expenses(limit=min(limit, 100))}

    # --------------------------------------------------------- background worker
    async def _worker_loop(app: FastAPI) -> None:
        """Single worker: dequeues jobs, runs the pipeline, delivers replies.

        Being single-threaded by design, it serialises all Ollama calls so the
        one multimodal model is never asked to swap weights mid-request.
        """
        while True:
            job = await queue.get()
            try:
                result = await _process_job(app, job)
                await _deliver(app, job["phone"], result["reply"])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker: pipeline failed for %s", job.get("user_id"))
                try:
                    await _deliver(app, job["phone"], GENERIC_ERROR)
                except Exception:
                    logger.exception("worker: failed to deliver error reply")
            finally:
                queue.task_done()

    async def _process_job(app: FastAPI, job: dict[str, Any]) -> dict[str, Any]:
        user_id: str = job["user_id"]
        phone: str = job["phone"]
        text: str = job["text"]
        image_b64: str | None = None

        # ---- voice note → text (Whisper) before anything else
        if job["type"] == "audio":
            media = job.get("media_path")
            if not media:
                return {"reply": "*Voice note* ⚠️\nThe audio file was missing."}
            loop = asyncio.get_running_loop()
            transcript = await loop.run_in_executor(None, transcriber.transcribe, media)
            if not transcript:
                return {
                    "reply": (
                        "*Voice note* ⚠️\n"
                        "I couldn't transcribe that (audio unclear or Whisper "
                        "isn't installed). Please type your message instead."
                    )
                }
            text = transcript
            logger.info("voice note from %s transcribed: %.80s", user_id, transcript)

        # ---- image → base64 for the vision model
        if job["type"] == "image" and job.get("media_path"):
            image_b64 = _image_to_base64(job["media_path"])

        if not text and not image_b64:
            return {"reply": "*Family Assistant*\nI didn't get any content — send a message, voice note or photo."}

        # ---- confirmations take priority (review fix #5)
        pending = _pending_confirmations.get(user_id)
        if pending is not None and not pending.expired():
            answer = text.strip().lower()
            if answer in {"yes", "y", "yep", "yeah", "confirm", "ok", "sure", "correct"}:
                _pending_confirmations.pop(user_id, None)
                _commit_expense(app, user_id, pending)
                reply = _expense_committed_reply(app, user_id, pending)
                database_save_turns(app, phone, text, reply)
                return {"reply": reply, "intent": "budget_log"}
            if answer in {"no", "n", "nope", "cancel", "wrong"}:
                _pending_confirmations.pop(user_id, None)
                reply = "*Budget log*\nOK — I discarded that entry. Nothing was saved."
                database_save_turns(app, phone, text, reply)
                return {"reply": reply, "intent": "budget_log"}
            # Some other message — drop the stale confirmation and continue.
            _pending_confirmations.pop(user_id, None)

        db: FamilyDatabase = app.state.db
        engine: OllamaEngine = app.state.engine

        summary = db.get_category_summary(user_id)
        user_totals = db.get_user_totals()
        history = db.get_context_messages(phone)

        # ---- LLM classify + reply (fallback classifier if Ollama is down)
        response = await asyncio.to_thread(
            engine.classify,
            text,
            user_id=user_id,
            history=history,
            summary=summary,
            user_totals=user_totals,
            image_b64=image_b64,
        )
        reply = response.whatsapp_reply.strip() or prompts.fallback_reply(response.intent, text, summary)

        # ---- budget_log: safety gate before any INSERT (review fix #5)
        if response.intent == "budget_log":
            expense, pending_confirmation, gate_reply = engine.validate_expense(text, response)
            if pending_confirmation is not None:
                pending_confirmation.user_id = user_id
                _pending_confirmations[user_id] = pending_confirmation
                reply = gate_reply
            elif expense is not None:
                _commit_expense(app, user_id, expense)
                reply = _expense_committed_reply(app, user_id, expense)
            else:
                reply = gate_reply or reply

        # ---- budget_query: ground the numbers in SQLite, never the model
        elif response.intent == "budget_query":
            household = db.get_category_summary()
            own = db.get_category_summary(user_id)
            reply = _budget_query_reply(user_id, household, own)

        database_save_turns(app, phone, text, reply)
        return {"reply": reply, "intent": response.intent}

    async def _deliver(app: FastAPI, phone: str, reply: str) -> None:
        """POST the reply to the Node bridge, which sends it on WhatsApp."""
        payload = {"to": phone, "reply": reply}
        try:
            async with httpx.AsyncClient(timeout=config.BRIDGE_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    config.BRIDGE_REPLY_URL,
                    json=payload,
                    headers={"X-Bridge-Secret": config.BRIDGE_SHARED_SECRET},
                )
                resp.raise_for_status()
            logger.info("delivered reply to %s (%d chars)", phone[:6] + "…", len(reply))
        except Exception as exc:
            # Fall back to a direct send attempt only through the bridge; if the
            # bridge is down the reply is logged so nothing silently vanishes.
            logger.error("bridge delivery failed for %s: %s", phone[:6] + "…", exc)

    return app


# ---------------------------------------------------------------------------
# Helpers (module-level so tests can import them directly)
# ---------------------------------------------------------------------------
def _image_to_base64(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        logger.error("image not found: %s", path)
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _commit_expense(app: FastAPI, user_id: str, expense_or_pending: Any) -> None:
    db: FamilyDatabase = app.state.db
    db.log_expense(
        user_id=user_id,
        amount=expense_or_pending.amount,
        category=expense_or_pending.category,
        description=expense_or_pending.description,
    )


def _expense_committed_reply(
    app: FastAPI, user_id: str, expense: Any
) -> str:
    """Confirmation + fresh individual/combined totals (spec §7.5)."""
    db: FamilyDatabase = app.state.db
    cur = config.CURRENCY_SYMBOL
    summary = db.get_category_summary(user_id)
    totals = db.get_user_totals()
    lines = [
        "*Expense logged* ✅",
        f"• Amount: *{cur}{float(expense.amount):.2f}*",
        f"• Category: _{expense.category}_",
        f"• Item: {expense.description or '—'}",
        "",
        "*Household totals*",
        f"• Yours: *{cur}{summary['user_total']:.2f}*",
        f"• Combined: *{cur}{summary['household_total']:.2f}*",
    ]
    for uid in config.USER_IDS:
        if uid != user_id:
            lines.append(f"• {uid.capitalize()}: {cur}{totals.get(uid, 0.0):.2f}")
    return "\n".join(lines)


def _budget_query_reply(
    user_id: str,
    household: dict[str, Any],
    own: dict[str, Any],
) -> str:
    cur = config.CURRENCY_SYMBOL
    lines = [
        "*Spending summary* 📊",
        f"• Your total: *{cur}{own['user_total']:.2f}*",
        f"• Household combined: *{cur}{household['household_total']:.2f}*",
    ]
    cats = household.get("by_category", {})
    if cats:
        lines.append("• By category:")
        for cat, amt in sorted(cats.items(), key=lambda kv: -kv[1]):
            lines.append(f"    – {cat}: {cur}{amt:.2f}")
    else:
        lines.append("• No expenses logged yet.")
    return "\n".join(lines)


def database_save_turns(app: FastAPI, phone: str, user_text: str, reply: str) -> None:
    """Persist the turn pair into conversation_history (per-number context)."""
    db: FamilyDatabase = app.state.db
    try:
        if user_text:
            db.save_chat_history(phone, "user", user_text)
        db.save_chat_history(phone, "assistant", reply)
    except Exception:
        logger.exception("failed to persist chat history")


# ---------------------------------------------------------------------------
# Uvicorn entrypoint:  python -m family_whatsapp_bot.server
# ---------------------------------------------------------------------------
app = create_app()


def main() -> None:  # pragma: no cover - manual entrypoint
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "family_whatsapp_bot.server:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
