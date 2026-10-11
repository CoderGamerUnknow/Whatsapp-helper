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

Locked-down hosts
-----------------
``faster_whisper`` imports PyAV (``av``) at package scope, but only *uses* it
inside ``decode_audio()``. Some managed Windows machines run an Application
Control policy that blocks av's native ``_core`` DLL, which makes the whole
package unimportable even though the Whisper engine itself would work. In that
case we install an inert ``av`` placeholder (never called — we never hand the
model a file path) and decode the WAV ourselves with the stdlib ``wave`` module
plus numpy, passing raw samples to ``WhisperModel.transcribe()`` instead.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# Lazily-created model singleton (loading a Whisper model takes seconds and
# hundreds of MBs — never do it at import time).
_whisper_model: Any = None
_whisper_load_failed = False
# Cached result of the availability probe. A failing native import is retried
# on every call and costs ~1s (DLL search + loader), which would make /health
# slow; the answer cannot change while the process lives.
_faster_whisper_available: bool | None = None
# Cached PyAV probe (a blocked DLL stays blocked for the life of the process).
_pyav_available: bool | None = None

_AUDIO_SUFFIXES = {".ogg", ".oga", ".opus", ".wav", ".mp3", ".m4a", ".webm", ".flac"}
# Sample rate faster-whisper expects; we resample to it with ffmpeg.
_TARGET_RATE = 16000


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def pyav_available() -> bool:
    """Whether PyAV imports. When it does not, we insert an inert placeholder
    so ``faster_whisper`` can still be imported, and decode audio ourselves."""
    global _pyav_available
    if _pyav_available is None:
        try:
            import av  # noqa: F401
            _pyav_available = True
        except Exception as exc:
            # An Application Control policy can block av's native _core DLL.
            logger.warning(
                "PyAV unavailable (%s); decoding WAV via the stdlib instead", exc
            )
            sys.modules.setdefault("av", types.ModuleType("av"))
            _pyav_available = False
    return _pyav_available


def faster_whisper_available() -> bool:
    """Whether faster-whisper can be used at all (with or without PyAV).

    Probed once and cached — see ``_faster_whisper_available``.
    """
    global _faster_whisper_available
    if _faster_whisper_available is None:
        pyav_available()  # make sure a blocked PyAV cannot break the import
        try:
            import faster_whisper  # noqa: F401
            _faster_whisper_available = True
        except Exception:  # pragma: no cover - import machinery can raise anything
            _faster_whisper_available = False
    return _faster_whisper_available


def _load_whisper_model() -> Any:
    """Load (once) and cache a faster-whisper WhisperModel."""
    global _whisper_model, _whisper_load_failed
    if _whisper_model is not None:
        return _whisper_model
    if _whisper_load_failed:
        return None
    if not faster_whisper_available():
        _whisper_load_failed = True
        logger.warning("faster-whisper not installed; voice notes unsupported")
        return None
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - depends on the host
        _whisper_load_failed = True
        logger.warning("faster-whisper unusable (%s); voice notes unsupported", exc)
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


def _to_wav(audio_path: Path, force_reencode: bool = False) -> Path:
    """Re-encode to 16 kHz mono WAV via ffmpeg; fall back to the original file.

    ``force_reencode`` is used when PyAV is blocked: we cannot rely on the
    decoder reading the source file, so even a .wav input is normalised to the
    exact rate/channels we can parse with the stdlib.
    """
    if audio_path.suffix.lower() == ".wav" and not force_reencode:
        return audio_path
    if not ffmpeg_available():
        return audio_path  # PyAV inside faster-whisper can decode ogg/opus itself
    fd, out_name = tempfile.mkstemp(suffix=".wav", prefix="wa_voice_")
    os.close(fd)  # close the handle now, else the file can't be unlinked later
    out = Path(out_name)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(audio_path),
        "-ar", str(_TARGET_RATE), "-ac", "1", "-c:a", "pcm_s16le",
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


def _read_wav_samples(wav_path: Path) -> Any:
    """Read a 16 kHz mono 16-bit PCM WAV into a float32 numpy array.

    Returns ``None`` for any other encoding — the caller then declines the
    voice note instead of guessing. Implemented with the stdlib so it works
    with no PyAV, no ffmpeg and no system audio libraries.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy ships with faster-whisper
        logger.error("numpy unavailable; cannot decode WAV samples")
        return None
    try:
        with wave.open(str(wav_path), "rb") as fh:
            if (
                fh.getsampwidth() != 2
                or fh.getnchannels() != 1
                or fh.getframerate() != _TARGET_RATE
            ):
                logger.warning(
                    "unexpected WAV format (rate=%d ch=%d width=%d) for %s",
                    fh.getframerate(), fh.getnchannels(), fh.getsampwidth(), wav_path.name,
                )
                return None
            raw = fh.readframes(fh.getnframes())
    except Exception as exc:
        logger.warning("stdlib WAV decode failed for %s: %s", wav_path.name, exc)
        return None
    if not raw:
        return None
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0


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

    have_pyav = _pyav_available
    wav_path = _to_wav(path, force_reencode=not have_pyav)
    try:
        # Without PyAV we must hand the model raw samples: decode_audio() is not
        # usable, so a non-normalised file means we cannot read it at all.
        audio: Any = str(wav_path)
        if not have_pyav:
            samples = _read_wav_samples(wav_path)
            if samples is None:
                logger.warning(
                    "PyAV blocked and %s is not 16 kHz mono PCM; skipping voice note",
                    path.name,
                )
                return None
            audio = samples

        segments, _info = model.transcribe(
            audio,
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
            try:
                Path(wav_path).unlink(missing_ok=True)
            except OSError:
                pass  # temp cleanup must never fail a successful transcription


def status() -> dict[str, Any]:
    """Diagnostics for GET /health."""
    return {
        "faster_whisper": faster_whisper_available(),
        "pyav": _pyav_available if _pyav_available is not None else pyav_available(),
        "ffmpeg": ffmpeg_available(),
        "model_size": config.WHISPER_MODEL_SIZE,
        "model_loaded": _whisper_model is not None,
        "available": faster_whisper_available(),
    }
