/**
 * WhatsApp Web bridge (Baileys) for the Family Assistant Bot.
 *
 * Responsibilities (single integration path — no Meta Cloud API):
 *   1. Hold a WhatsApp Web session (QR-code login, stored under ./auth).
 *   2. Receive inbound messages: text, voice notes (.ogg), images.
 *   3. Forward them to the Python backend (POST /api/incoming) — the backend
 *      answers 202 immediately and processes in its async worker queue.
 *   4. Expose a tiny HTTP server (POST /bridge/send) that the backend calls
 *      when a reply is ready; we push that reply to WhatsApp from here.
 *
 * Security: the backend is the authoritative whitelist (it drops unknown
 * senders). We pre-filter here too so we never even download media for
 * strangers, and inbound calls to /bridge/send require the shared secret.
 *
 * Env: BACKEND_URL, BRIDGE_PORT, BRIDGE_SHARED_SECRET, WHITELISTED_NUMBERS,
 *      MEDIA_DIR, AUTH_DIR
 */

'use strict';

const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');

const makeWASocket = require('@whiskeysockets/baileys').default;
const {
  useMultiFileAuthState,
  downloadMediaMessage,
  DisconnectReason,
  fetchLatestBaileysVersion,
} = require('@whiskeysockets/baileys');
const pino = require('pino');
const qrcode = require('qrcode-terminal');

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
const BRIDGE_PORT = Number(process.env.BRIDGE_PORT || 8790);
const BRIDGE_HOST = process.env.BRIDGE_HOST || '127.0.0.1';
const SHARED_SECRET = process.env.BRIDGE_SHARED_SECRET || 'family-bridge-secret';
const MEDIA_DIR = process.env.MEDIA_DIR || path.join(os.tmpdir(), 'family-bot-media');
const AUTH_DIR = process.env.AUTH_DIR || path.join(__dirname, 'auth');

// Defense in depth: the backend enforces the whitelist too.
const WHITELIST = (process.env.WHITELISTED_NUMBERS || '+12345678901,+12345678902,+12345678903')
  .split(',')
  .map((n) => n.replace(/\D/g, ''))
  .filter(Boolean);

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

fs.mkdirSync(MEDIA_DIR, { recursive: true });

let sock = null;
let starting = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
/** '+1 234-567-8901' | '12345678901:12@s.whatsapp.net' -> digits only */
function digitsOf(value) {
  return String(value || '').split('@')[0].split(':')[0].replace(/\D/g, '');
}

function isWhitelisted(jidOrNumber) {
  const d = digitsOf(jidOrNumber);
  if (!d) return false;
  return WHITELIST.some((allowed) =>
    d === allowed || d.endsWith(allowed) || allowed.endsWith(d)
  );
}

async function postJson(url, payload, { secret = false, retries = 3 } = {}) {
  const headers = { 'content-type': 'application/json' };
  if (secret) headers['x-bridge-secret'] = SHARED_SECRET;
  let lastErr = null;
  for (let attempt = 1; attempt <= retries; attempt += 1) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
      return await res.json().catch(() => ({}));
    } catch (err) {
      lastErr = err;
      if (attempt < retries) await new Promise((r) => setTimeout(r, 500 * attempt));
    }
  }
  throw lastErr;
}

/** Persist a media buffer to disk; returns the absolute path. */
function saveMedia(buffer, ext) {
  const file = path.join(MEDIA_DIR, `${Date.now()}-${Math.random().toString(36).slice(2, 8)}${ext}`);
  fs.writeFileSync(file, buffer);
  return file;
}

// ---------------------------------------------------------------------------
// Inbound: WhatsApp -> backend
// ---------------------------------------------------------------------------
async function forward(message) {
  try {
    const result = await postJson(`${BACKEND_URL}/api/incoming`, message, { retries: 3 });
    logger.info({ user: result.user || 'ignored', status: result.status }, 'forwarded');
  } catch (err) {
    logger.error({ err: String(err) }, 'forward to backend failed');
  }
}

async function handleIncoming(msg) {
  const jid = msg.key.remoteJid;
  if (!jid || msg.key.fromMe) return;
  if (jid === 'status@broadcast' || jid.endsWith('@g.us')) return; // groups & status ignored
  if (!isWhitelisted(jid)) {
    logger.warn({ jid: digitsOf(jid).slice(-4) }, 'dropped non-whitelisted sender');
    return; // unknown senders never reach the backend or media download
  }

  const from = digitsOf(jid);
  const message = msg.message;
  if (!message) return;

  // Text (including extended text / captions)
  const text =
    message.conversation ||
    message.extendedTextMessage?.text ||
    message.imageMessage?.caption ||
    message.videoMessage?.caption ||
    '';

  if (message.audioMessage) {
    // Voice note → download once, let the backend transcribe with Whisper.
    try {
      const buffer = await downloadMediaMessage(msg, 'buffer', {}, {
        logger,
        reuploadRequest: sock.updateMediaMessage,
      });
      const ext = message.audioMessage.mimetype?.includes('ogg') ? '.ogg' : '.audio';
      const mediaPath = saveMedia(Buffer.from(buffer), ext);
      await forward({ from, type: 'audio', media_path: mediaPath, message_id: msg.key.id });
    } catch (err) {
      logger.error({ err: String(err) }, 'voice note download failed');
    }
    return;
  }

  if (message.imageMessage) {
    try {
      const buffer = await downloadMediaMessage(msg, 'buffer', {}, {
        logger,
        reuploadRequest: sock.updateMediaMessage,
      });
      const mediaPath = saveMedia(Buffer.from(buffer), '.jpg');
      await forward({
        from, type: 'image', media_path: mediaPath, caption: text || '',
        message_id: msg.key.id,
      });
    } catch (err) {
      logger.error({ err: String(err) }, 'image download failed');
    }
    return;
  }

  if (text.trim()) {
    await forward({ from, type: 'text', text: text.trim(), message_id: msg.key.id });
  }
  // Other message types (video, stickers, documents…) are silently skipped.
}

// ---------------------------------------------------------------------------
// Outbound: backend -> WhatsApp
// ---------------------------------------------------------------------------
async function sendReply(to, reply) {
  if (!sock) throw new Error('WhatsApp session not connected');
  const jid = `${digitsOf(to)}@s.whatsapp.net`;
  await sock.sendMessage(jid, { text: reply });
}

// Tiny HTTP server the backend calls back into.
function startCallbackServer() {
  const server = http.createServer((req, res) => {
    const sendStatus = (code, obj) => {
      const body = JSON.stringify(obj);
      res.writeHead(code, { 'content-type': 'application/json' });
      res.end(body);
    };

    if (req.method === 'GET' && req.url === '/bridge/health') {
      return sendStatus(200, { ok: true, connected: Boolean(sock) });
    }

    if (req.method === 'POST' && req.url === '/bridge/send') {
      if (req.headers['x-bridge-secret'] !== SHARED_SECRET) {
        return sendStatus(403, { error: 'bad shared secret' });
      }
      let raw = '';
      req.on('data', (chunk) => {
        raw += chunk;
        if (raw.length > 1_000_000) req.destroy();
      });
      req.on('end', async () => {
        try {
          const { to, reply } = JSON.parse(raw || '{}');
          if (!to || !reply) return sendStatus(400, { error: 'to and reply are required' });
          await sendReply(to, reply);
          return sendStatus(200, { ok: true });
        } catch (err) {
          logger.error({ err: String(err) }, 'sendReply failed');
          return sendStatus(500, { error: String(err) });
        }
      });
      return undefined;
    }

    return sendStatus(404, { error: 'not found' });
  });

  server.listen(BRIDGE_PORT, BRIDGE_HOST, () => {
    logger.info(`bridge callback server on http://${BRIDGE_HOST}:${BRIDGE_PORT}`);
  });
  return server;
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
async function startSocket() {
  if (starting) return;
  starting = true;
  try {
    const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
    const { version } = await fetchLatestBaileysVersion().catch(() => ({ version: undefined }));

    sock = makeWASocket({
      version,
      auth: state,
      logger: pino({ level: 'silent' }),
      printQRInTerminal: false,
      browser: ['Family-Assistant-Bot', 'Chrome', '1.0.0'],
      markOnlineOnConnect: false,
    });

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('connection.update', (update) => {
      const { connection, lastDisconnect, qr } = update;
      if (qr) {
        logger.info('Scan this QR with WhatsApp (Linked Devices):');
        qrcode.generate(qr, { small: true });
      }
      if (connection === 'open') {
        logger.info('WhatsApp session connected');
      }
      if (connection === 'close') {
        const statusCode = lastDisconnect?.error?.output?.statusCode;
        const loggedOut = statusCode === DisconnectReason.loggedOut;
        logger.warn({ statusCode, loggedOut }, 'connection closed');
        sock = null;
        if (loggedOut) {
          logger.error('Logged out — delete bridge/auth and re-scan the QR code.');
          process.exit(1);
        }
        starting = false;
        setTimeout(() => startSocket(), 3000); // auto-reconnect
      }
    });

    sock.ev.on('messages.upsert', async ({ messages, type }) => {
      if (type !== 'notify') return;
      for (const msg of messages) {
        try {
          await handleIncoming(msg);
        } catch (err) {
          logger.error({ err: String(err) }, 'message handling failed');
        }
      }
    });
  } catch (err) {
    logger.error({ err: String(err) }, 'socket start failed');
    starting = false;
    setTimeout(() => startSocket(), 5000);
  } finally {
    starting = false;
  }
}

// ---------------------------------------------------------------------------
startCallbackServer();
startSocket();
logger.info(`bridge started (backend=${BACKEND_URL}, whitelist=${WHITELIST.length} numbers)`);
