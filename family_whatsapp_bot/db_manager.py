"""SQLite manager for ``family_hub.db``.

Tables (spec §5):
    users                 — id, phone_number (UNIQUE), name
    expenses              — id, user_id, amount, category, description, created_at
    reminders             — id, user_id, task, due_date, status
    conversation_history  — id, phone_number, role, content, timestamp

Concurrency (review fix #4): every connection enables WAL, a busy timeout and
synchronous=NORMAL so concurrent writers (Dad logging an expense while Mum
queries totals) queue briefly instead of throwing ``database is locked``.
All write transactions additionally go through a process-wide re-entrant lock,
which makes SQLite safe to share across the FastAPI worker thread, the asyncio
queue consumer and any test threads.

Every expense / reminder / history row is tagged with the sender's user id
('dad' | 'mom' | 'user') resolved from the whitelist (spec §3).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from . import config

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY,
    phone_number TEXT UNIQUE NOT NULL,
    name         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS expenses (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL,
    amount      REAL NOT NULL,
    category    TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reminders (
    id        INTEGER PRIMARY KEY,
    user_id   TEXT NOT NULL,
    task      TEXT NOT NULL,
    due_date  TEXT,
    status    TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id           INTEGER PRIMARY KEY,
    phone_number TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_expenses_user_created  ON expenses (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_expenses_created       ON expenses (created_at);
CREATE INDEX IF NOT EXISTS idx_history_phone_time     ON conversation_history (phone_number, timestamp);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FamilyDatabase:
    """Thread-safe wrapper around a single SQLite connection (WAL mode)."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else Path(config.DB_PATH)
        self._write_lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # shared across worker thread + request threads
            isolation_level=None,     # autocommit; we control transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._conn.executescript(_SCHEMA)
        self._seed_users()

    # ------------------------------------------------------------------ setup
    def _apply_pragmas(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")       # review fix #4
        cur.execute("PRAGMA busy_timeout=5000")      # review fix #4
        cur.execute("PRAGMA synchronous=NORMAL")     # safe with WAL, much faster
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    def _seed_users(self) -> None:
        """Insert/refresh the three whitelisted members (id-tagging source)."""
        with self._write_lock:
            for phone, user_id in config.PHONE_TO_USER.items():
                self._conn.execute(
                    "INSERT INTO users (phone_number, name) VALUES (?, ?) "
                    "ON CONFLICT(phone_number) DO UPDATE SET name=excluded.name",
                    (phone, user_id),
                )

    def close(self) -> None:
        with self._write_lock:
            self._conn.close()

    # ------------------------------------------------------------- primitives
    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            return self._conn.execute(sql, tuple(params))

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        # Reads are safe under WAL alongside a writer; the lock is not needed
        # for correctness but keeps row iteration stable on a shared connection.
        with self._write_lock:
            return list(self._conn.execute(sql, tuple(params)))

    # ------------------------------------------------------------------ users
    def get_user_by_phone(self, phone_number: str) -> sqlite3.Row | None:
        rows = self._query("SELECT * FROM users WHERE phone_number = ?", (phone_number,))
        return rows[0] if rows else None

    def get_user_name(self, user_id: str) -> str:
        rows = self._query("SELECT name FROM users WHERE name = ?", (user_id,))
        if rows:
            return str(rows[0]["name"])
        return user_id

    # --------------------------------------------------------------- expenses
    def log_expense(
        self,
        user_id: str,
        amount: float | Decimal,
        category: str,
        description: str = "",
    ) -> int:
        """Insert one expense, tagged with the acting family member's id."""
        value = float(amount)
        if value <= 0:
            raise ValueError(f"expense amount must be positive, got {value!r}")
        category = config.normalize_category(category)
        cur = self._execute(
            "INSERT INTO expenses (user_id, amount, category, description, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, value, category, description, _utc_now().isoformat(sep=" ", timespec="seconds")),
        )
        logger.info("expense logged: user=%s amount=%.2f category=%s", user_id, value, category)
        return int(cur.lastrowid or 0)

    def get_expenses(
        self,
        user_id: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM expenses"
        clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if since:
            clauses.append("created_at >= ?")
            params.append(since)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._query(sql, params)]

    def get_category_summary(
        self,
        user_id: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Totals by category and combined household total.

        With *user_id* the summary covers that member only (their spending);
        the ``household_total`` key always covers the whole family so replies
        can show individual + combined totals (spec §7.5).
        """
        clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if since:
            clauses.append("created_at >= ?")
            params.append(since)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        by_category: dict[str, float] = {}
        for row in self._query(
            "SELECT category, SUM(amount) AS total FROM expenses" + where + " GROUP BY category",
            params,
        ):
            by_category[str(row["category"])] = round(float(row["total"] or 0), 2)

        per_user_total = 0.0
        if user_id:
            rows = self._query(
                "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ?"
                + (" AND created_at >= ?" if since else ""),
                [user_id] + ([since] if since else []),
            )
            per_user_total = round(float(rows[0]["total"] or 0), 2)

        house_rows = self._query(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses"
            + (" WHERE created_at >= ?" if since else ""),
            [since] if since else [],
        )
        household_total = round(float(house_rows[0]["total"] or 0), 2)

        return {
            "user_id": user_id,
            "by_category": by_category,
            "user_total": per_user_total,
            "household_total": household_total,
        }

    def get_user_totals(self, since: str | None = None) -> dict[str, float]:
        """Combined household split per member: {'dad': x, 'mom': y, 'user': z}."""
        sql = "SELECT user_id, SUM(amount) AS total FROM expenses"
        if since:
            sql += " WHERE created_at >= ?"
        sql += " GROUP BY user_id"
        totals = {uid: 0.0 for uid in config.USER_IDS}
        for row in self._query(sql, [since] if since else []):
            totals[str(row["user_id"])] = round(float(row["total"] or 0), 2)
        return totals

    # -------------------------------------------------------------- reminders
    def add_reminder(
        self, user_id: str, task: str, due_date: str | None = None, status: str = "pending"
    ) -> int:
        cur = self._execute(
            "INSERT INTO reminders (user_id, task, due_date, status) VALUES (?, ?, ?, ?)",
            (user_id, task, due_date, status),
        )
        return int(cur.lastrowid or 0)

    def get_reminders(
        self, user_id: str | None = None, status: str | None = "pending"
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reminders"
        clauses, params = [], []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY due_date IS NULL, due_date, id"
        return [dict(r) for r in self._query(sql, params)]

    def complete_reminder(self, reminder_id: int) -> bool:
        cur = self._execute(
            "UPDATE reminders SET status = 'done' WHERE id = ?", (reminder_id,)
        )
        return cur.rowcount > 0

    # ------------------------------------------------------- conversation log
    def save_chat_history(self, phone_number: str, role: str, content: str) -> int:
        """Persist one chat turn (role: 'user' | 'assistant')."""
        cur = self._execute(
            "INSERT INTO conversation_history (phone_number, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (phone_number, role, content, _utc_now().isoformat(sep=" ", timespec="seconds")),
        )
        return int(cur.lastrowid or 0)

    def get_chat_history(self, phone_number: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Most recent turns, newest first (apply your own reversal if needed).

        *limit* defaults to ``CONTEXT_WINDOW_SIZE`` — the sliding window from
        review fix #6 — so prompt building never re-sends the full history.
        """
        if limit is None:
            limit = config.CONTEXT_WINDOW_SIZE
        rows = self._query(
            "SELECT * FROM conversation_history WHERE phone_number = ? "
            "ORDER BY timestamp DESC, id DESC LIMIT ?",
            (phone_number, limit),
        )
        return [dict(r) for r in rows]

    def get_context_messages(self, phone_number: str, limit: int | None = None) -> list[dict[str, str]]:
        """Sliding-window history formatted as Ollama messages (oldest first)."""
        history = list(reversed(self.get_chat_history(phone_number, limit)))
        return [{"role": str(r["role"]), "content": str(r["content"])} for r in history]

    def count_table(self, table: str) -> int:
        """Small helper used by health checks/tests."""
        if table not in {"users", "expenses", "reminders", "conversation_history"}:
            raise ValueError(f"unknown table: {table}")
        rows = self._query(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608 - whitelisted table
        return int(rows[0]["n"])


# --------------------------------------------------------------- module API
_default_db: FamilyDatabase | None = None
_default_lock = threading.Lock()


def get_database(db_path: Path | str | None = None) -> FamilyDatabase:
    """Process-wide singleton (tests can create isolated FamilyDatabase objects)."""
    global _default_db
    with _default_lock:
        if _default_db is None or (db_path is not None and _default_db.db_path != Path(db_path)):
            _default_db = FamilyDatabase(db_path)
        return _default_db


def init_db(db_path: Path | str | None = None) -> FamilyDatabase:
    """Create/upgrade the schema and return the shared connection."""
    return get_database(db_path)
