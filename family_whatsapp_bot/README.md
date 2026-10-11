# WhatsApp Family Assistant Bot 🏠

A **local, 24/7 family assistant** for WhatsApp: expense tracking, bill
summaries, tech help and recipes — powered by a **local Ollama LLM**,
**local Whisper transcription** and **SQLite**. Nothing leaves your machine.

- **Whitelist:** only 3 numbers (Dad, Mom, You) — everyone else is dropped.
- **Intents:** `budget_log` · `budget_query` · `doc_summary` · `tech_help` · `recipe` · `general_chat`
- **Voice notes:** transcribed locally (faster-whisper) before they reach the LLM
- **Photos:** read by the same local vision model (bills/letters)
- **Money is guarded:** model output is validated (Pydantic + regex) and
  confirmed by the user before anything is written to the ledger
- **One integration path:** a local Node.js **Baileys** bridge (WhatsApp Web,
  QR login) — no Meta Cloud API, no tunnels, no tokens

```
WhatsApp ⇄ bridge/ (Baileys, :8790) ⇄ server.py (FastAPI, :8000) ⇄ Ollama (:11434)
                                                     │
                                              family_hub.db (SQLite/WAL)
```

## Requirements

- Python 3.10+ · Node.js 18+ · `ffmpeg` · Ollama (https://ollama.com)
- Linux recommended (systemd units included). Windows/macOS work for dev.

## Quick start (Linux, one shot)

All commands below run from the **repository root** (the folder containing
`family_whatsapp_bot/`, `bridge/` and `deploy/`).

```bash
sudo ./deploy/install.sh            # apt deps, venv, pip, npm, ollama, systemd
nano .env                           # ← set YOUR 3 numbers in WHITELISTED_NUMBERS
journalctl -u family-bridge -f      # scan the QR code with WhatsApp
./deploy/install.sh --check         # verify everything is green
```

> `config.py` reads `.env` from the repository root *or* from
> `family_whatsapp_bot/.env` (the latter wins) — either location works.

## Quick start (manual / dev)

```bash
# 1. backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r family_whatsapp_bot/requirements.txt
cp deploy/.env.example .env         # edit WHITELISTED_NUMBERS!

# 2. LLM (one multimodal model for text + images)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2-vision

# 3. run backend
uvicorn family_whatsapp_bot.server:app --host 127.0.0.1 --port 8000

# 4. run the bridge (new terminal) — scan the QR it prints
cd bridge && npm install && npm start
```

Windows PowerShell users: same steps, but `.\.venv\Scripts\activate` and run
`python -m family_whatsapp_bot.server` instead of the uvicorn line.

## Try it (curl, no WhatsApp needed)

```bash
# health — shows whitelist, Ollama, whisper, queue depth
curl -s localhost:8000/health | python -m json.tool

# budget_log (from Dad's whitelisted number)
curl -s localhost:8000/api/simulate -H 'content-type: application/json' \
  -d '{"from":"+12345678901","type":"text","text":"spent $24.50 on groceries — milk, eggs"}'

# budget_query — combined household + per-category totals from SQLite
curl -s localhost:8000/api/simulate -H 'content-type: application/json' \
  -d '{"from":"+12345678902","type":"text","text":"how much have we spent in total?"}'

# doc_summary — always 3 bullets: Summary / Action Required / Due Date
curl -s localhost:8000/api/simulate -H 'content-type: application/json' \
  -d '{"from":"+12345678901","type":"text","text":"summarize this electricity bill due 15 Oct"}'

# tech_help & recipe
curl -s localhost:8000/api/simulate -H 'content-type: application/json' \
  -d '{"from":"+12345678903","type":"text","text":"the wifi is not working"}'
curl -s localhost:8000/api/simulate -H 'content-type: application/json' \
  -d '{"from":"+12345678903","type":"text","text":"give me a recipe with eggs and spinach"}'

# a STRANGER is dropped instantly (and nothing is stored)
curl -s localhost:8000/api/incoming -H 'content-type: application/json' \
  -d '{"from":"+15550009999","type":"text","text":"let me in"}'
# → {"status":"ignored","reason":"not_whitelisted"}
```

Expenses in the ledger:

```bash
curl -s localhost:8000/api/expenses | python -m json.tool
```

## How a message flows

1. Bridge receives it → drops strangers locally → `POST /api/incoming`.
2. Backend whitelists again (authoritative), answers **202 in <100 ms**,
   enqueues. *(Meta's 5 s webhook rule is honored by construction.)*
3. Single worker: voice note → Whisper text; photo → base64 for the vision
   model; then Ollama classifies intent + drafts the reply (JSON contract).
4. `budget_log` goes through the **safety gate**: Pydantic validation + regex
   cross-check against your actual message. Match → saved. Mismatch/invented
   amount → *"Did you mean $12.50 for Groceries?"* → saved only after `yes`.
5. Reply is grounded in SQLite totals (individual **and** combined household),
   formatted in WhatsApp markdown (`*bold*`, `_italics_`, `•`, `1.`), and
   delivered through the bridge.

## Tests

```bash
pip install -r family_whatsapp_bot/requirements.txt
python -m pytest family_whatsapp_bot/tests -q      # 95 tests, fully offline
node  --check bridge/index.js                      # bridge syntax
bash  -n deploy/install.sh                         # installer syntax
```

The suite uses temp databases and a dead Ollama URL — no model downloads,
deterministic on CI.

## Configuration

Everything lives in `.env` (template: [deploy/.env.example](../deploy/.env.example)):

| var | default | meaning |
|---|---|---|
| `WHITELISTED_NUMBERS` | placeholder trio | **the only senders allowed**, order = dad, mom, user |
| `OLLAMA_MODEL` | `llama3.2-vision` | one multimodal model for text + images |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API |
| `BRIDGE_SHARED_SECRET` | placeholder | auth between bridge ↔ backend |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`→`medium`, bigger = slower + better |
| `CONTEXT_WINDOW_SIZE` | `8` | sliding window fed to the LLM |
| `CURRENCY_SYMBOL` | `$` | shown in totals |
| `DB_PATH` | `family_whatsapp_bot/family_hub.db` | SQLite location |

**Model split (optional, needs ≥16 GB VRAM):** the default single
`llama3.2-vision` avoids model-swap stalls. If you *do* want a text specialist,
`ollama pull qwen2.5:7b`, set `OLLAMA_MODEL=qwen2.5:7b`, and expect an extra
load/unlock hop whenever a photo arrives.

## Troubleshooting

| symptom | fix |
|---|---|
| `/health` shows `degraded` | Ollama not running: `sudo systemctl start ollama` |
| replies say "offline mode" | model missing: `ollama pull llama3.2-vision` |
| voice notes get "couldn't transcribe" | `pip install faster-whisper` + `apt install ffmpeg` |
| bridge prints a QR every restart | session deleted — rescan; don't wipe `bridge/auth/` |
| strangers reach the bot | check `WHITELISTED_NUMBERS` in `.env`, restart `family-bot` |
| `database is locked` | another process holds the DB; WAL+busy_timeout should prevent this — check for a stray worker |
| bridge can't reach backend | both must share `BRIDGE_SHARED_SECRET`; check ports 8000/8790 |

## Repository layout

```
family_whatsapp_bot/     this package: config, db, transcriber, engine,
                         server, prompts, knowledge.md, README.md,
                         requirements.txt
family_whatsapp_bot/tests/   95 offline tests (pytest)
bridge/                  Node.js Baileys bridge (QR session, media download, reply send)
deploy/                  install.sh, systemd units, .env.example
knowledge.md             architecture cheat sheet (in this package)
```

## Security notes

- Whitelist enforced **twice** (bridge pre-filter + backend authoritative).
- Bridge callback requires `X-Bridge-Secret`; both ports bind `127.0.0.1` by
  default — nothing is exposed to the internet.
- The WhatsApp Web session (`bridge/auth/`) is a credential: keep it out of
  git (already in `.gitignore`) and off shared machines.
- No message content ever leaves the machine: LLM, STT and DB are all local.
