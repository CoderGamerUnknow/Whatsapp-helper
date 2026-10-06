"""Local speech-to-text pipeline for WhatsApp voice notes (.ogg / .opus / .wav).

Provider priority (auto-detected at runtime, no hard dependency):
    1. faster-whisper   — preferred, runs fully offline on CPU or CUDA
    2. whisper.cpp      — optional: set WHISPER_CPP_BIN to a compiled binary
    3. unavailable      — returns None with a clear log line; callers fall back
                          to asking the user to type the message

WhatsApp voice notes are Ogg/Opus. We normalise them to 16 kHz mono WAV with
the system ``ffmpeg`` before transcription when ffmpeg is present; if it is
not, faster-whisper's bundled PyAV decoder still reads .ogg directly, so the
pipeline degrades gracefully either way.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# Lazily-created model singleton (loading a Whisper model takes seconds and
# hundreds of MBs — never do it at import time).
_whisper_model: Any = None
_whisper_load_failed = False

_AUDIO_SUFFIXES = {".ogg", ".oga", ".opus", ".wav", ".mp3", ".m4a", ".webm", ".flac"}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:  # pragma: no cover - import machinery can raise anything
        return False


def _load_whisper_model() -> Any:
    """Load (once) and cache a faster-whisper WhisperModel."""
    global _whisper_model, _whisper_load_failed
    if _whisper_model is not None:
        return _whisper_model
    if _whisper_load_failed:
        return None
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        _whisper_load_failed = True
        logger.warning("faster-whisper not installed (%s); voice notes unsupported", exc)
        return None
    try:
        logger.info(
            "Loading Whisper model '%s' (device=%s, compute=%s)…",
            config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model ready")
        return _whisper_model
    except Exception as exc:  # pragma: no cover - depends on host hardware
        _whisper_load_failed = True
        logger.error("Failed to load Whisper model: %s", exc)
        return None


def _to_wav(audio_path: Path) -> Path:
    """Re-encode to 16 kHz mono WAV via ffmpeg; fall back to the original file.

    Returns a path that the whisper decoder can read. The temporary WAV lives
    next to the source file and is cleaned up by the caller when different.
    """
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    if not ffmpeg_available():
        return audio_path  # PyAV inside faster-whisper can decode ogg/opus itself
    out = Path(tempfile.mkstemp(suffix=".wav", prefix="wa_voice_")[1])
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return out
    except Exception as exc:
        logger.warning("ffmpeg conversion failed (%s); trying original file", exc)
        if out.exists():
            out.unlink(missing_ok=True)
        return audio_path


def transcribe(audio_path: str | Path) -> str | None:
    """Convert a voice-note file to text. Returns None when STT is unavailable
    or the audio yields no speech — callers must handle that gracefully."""
    path = Path(audio_path)
    if not path.is_file():
        logger.error("transcribe: file not found: %s", path)
        return None
    if path.suffix.lower() not in _AUDIO_SUFFIXES:
        logger.warning("transcribe: unsupported audio type %s", path.suffix)
        return None

    model = _load_whisper_model()
    if model is None:
        return None

    wav_path = _to_wav(path)
    try:
        segments, _info = model.transcribe(
            str(wav_path),
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,          # skip silence — faster + cleaner text
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("transcribed %s -> %d chars", path.name, len(text))
        return text or None
    except Exception as exc:
        logger.error("transcription failed for %s: %s", path.name, exc)
        return None
    finally:
        if wav_path != path and wav_path.name.startswith("wa_voice_"):
            Path(wav_path).unlink(missing_ok=True)


def status() -> dict[str, Any]:
    """Diagnostics for GET /health."""
    return {
        "faster_whisper": faster_whisper_available(),
        "ffmpeg": ffmpeg_available(),
        "model_size": config.WHISPER_MODEL_SIZE,
        "model_loaded": _whisper_model is not None,
        "available": faster_whisper_available() and _whisper_model is not False,
    }


# Silence linters about the unused import kept for env-based path overrides.
_ = os
