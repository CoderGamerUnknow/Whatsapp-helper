# Knowledge — WhatsApp Family Assistant Bot

Local, 24/7 family assistant: WhatsApp messages/voice notes/photos from **3
whitelisted numbers** → intent routing via a **local Ollama LLM** → **SQLite**
ledger + replies in WhatsApp markdown. No cloud AI, no Meta Cloud API.

## Architecture

```
 WhatsApp (Web)
      ▲  ▼
 ┌─────────────────────┐   POST /api/incoming    ┌──────────────────────────┐
 │ bridge/index.js     │ ───────────────────────▶│ server.py (FastAPI)      │
 │ Baileys session     │  {from,type,text|media} │  ① whitelist gate        │
 │ QR login (./auth)   │                         │  ② 202 ACK  (<100 ms)    │
 │ downloads voice/img │                         │  ③ asyncio.Queue         │
 │                     │◀─────────────────────── │  ④ single worker         │
 │ POST /bridge/send   │   {to, reply}           │     ├ transcriber.py     │
 └─────────────────────┘                         │     ├ ollama_engine.py   │
                                                 │     └ db_manager.py      │
                                                 └──────────┬───────────────┘
                                                            │ /api/chat
                                                 ┌──────────▼───────────────┐
                                                 │ Ollama  :11434           │
                                                 │ llama3.2-vision (1 model)│
                                                 └──────────────────────────┘
                                                 family_hub.db (SQLite, WAL)
```

**Why the queue:** Meta's webhook deadline is 5 s; local LLM + Whisper take
5–15 s. `/api/incoming` validates + enqueues + answers 202 instantly; one
background worker does the slow work and calls the bridge back. The single
worker also serialises all Ollama calls → the one multimodal model is never
asked to swap weights mid-request (VRAM thrash fix).

## Message pipeline (worker)

1. `audio` → `transcriber.transcribe()` (ffmpeg → 16 kHz WAV → faster-whisper).
2. `image` → base64 → attached to the Ollama user message (vision).
3. Pending **confirmation** check: if this user owes a `yes/no` for a staged
   expense, that answer wins (nothing else is classified).
4. `ollama_engine.classify()` → `ModelResponse{intent,data,whatsapp_reply}`
   (strict JSON; deterministic fallback classifier if Ollama is down).
5. Intent dispatch:
   - `budget_log` → **expense safety gate** (below) → `log_expense()`.
   - `budget_query` → reply rebuilt from SQLite aggregates (never model numbers).
   - others → model/fallback reply passed through.
6. Save turn pair to `conversation_history`, deliver reply via bridge.

## Intent contract (JSON)

```json
{"intent": "budget_log|budget_query|doc_summary|tech_help|recipe|general_chat",
 "data": {"amount": 0.0, "category": "", "description": ""},
 "whatsapp_reply": "uses *bold*, _italics_, • bullets, 1. steps"}
```

- `doc_summary` reply must be exactly 3 bullets:
  `• *Summary:*` `• *Action Required:*` `• *Due Date:*`
- Allowed categories: `Groceries, Utilities, Gas, Food, Misc`
  (`config.normalize_category()` maps synonyms).

## Expense safety gate (money never hallucinates)

`ollama_engine.validate_expense()`:

1. Parse model `data` into Pydantic `ExpenseData` (amount `Decimal > 0`,
   category enum, description trimmed). Bad parse → treat as "no model amount".
2. Re-extract the amount from the **raw user text** with regex
   (`$12.50`, `12,50`, `1,234.56`, bare `40` all handled).
3. Model amount **agrees** with the message → INSERT immediately.
4. Disagreement, or an amount the model invented (not in the message) →
   **stage a confirmation** ("Did you mean *$12.50* for _Groceries_?"),
   user replies `yes`/`no`. TTL: `PENDING_CONFIRMATION_TTL_SECONDS`.
5. Only staged values are ever written — the message text wins on conflict.

Replies always show **fresh DB aggregates** (individual + combined household).

## Database (`family_hub.db`, WAL + `busy_timeout=5000`)

| table | key columns | notes |
|---|---|---|
| `users` | phone_number UNIQUE, name | seeded from whitelist = `dad/mom/user` |
| `expenses` | user_id, amount, category, description, created_at | every row tagged |
| `reminders` | user_id, task, due_date, status | pending/done |
| `conversation_history` | phone_number, role, content, timestamp | full audit log |

Prompt context uses a **sliding window** of the last `CONTEXT_WINDOW_SIZE`
(8) messages per user (`get_context_messages()`), oldest-first — the table
keeps everything, the model only sees the window.

## Environment variables

See `deploy/.env.example`. Critical ones:

- `WHITELISTED_NUMBERS` — exactly 3, comma separated, order = dad, mom, user
- `OLLAMA_MODEL` — default `llama3.2-vision` (text + images in ONE model;
  `qwen2.5-vl` also works). Split text/vision models only with ≥16 GB VRAM:
  run one `ollama serve`, pin `OLLAMA_KEEP_ALIVE`, and expect a model swap
  per image message.
- `BRIDGE_SHARED_SECRET` — must match in backend `.env` and bridge env;
  bridge rejects `/bridge/send` without it.
- `DB_PATH`, `CONTEXT_WINDOW_SIZE`, `WHISPER_MODEL_SIZE`, `CURRENCY_SYMBOL`

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | ollama/whisper/db/queue diagnostics |
| POST | `/api/incoming` | bridge webhook — 202 + queue; `ignored` for strangers |
| POST | `/api/simulate` | same pipeline, synchronous reply (curl testing) |
| GET | `/api/expenses` | debug listing |
| GET | `/bridge/health`, POST | `/bridge/send` | bridge side (port 8790) |

## Operational facts / troubleshooting

- **Ollama down** → fallback classifier answers (offline templates); `/health`
  reports `degraded`. Nothing crashes.
- **Whisper unavailable** → voice notes get a polite "type it instead" reply.
- **DB locked errors** → shouldn't happen (WAL+timeout+writer lock); check
  for a second process writing to the same file.
- **Bridge logged out** → delete `bridge/auth`, re-scan QR.
- **Logs**: `journalctl -u family-bot -f`, `journalctl -u family-bridge -f`.
- **Tests**: `python -m pytest family_whatsapp_bot/tests -q` (offline, temp DBs).

## Future: migrating to Meta WhatsApp Cloud API

Not used today (single integration path = Baileys). If you ever switch:
- webhook verification + `META_ACCESS_TOKEN` — use a **permanent System User
  token** from Meta Business Settings ▸ System Users; dashboard tokens expire
  in 24 h and would silently kill the bot on day two.
- Keep the same `/api/incoming` payload shape so only the transport changes.
