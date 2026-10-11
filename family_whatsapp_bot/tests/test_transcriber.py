"""transcriber.py: media pipeline, ffmpeg conversion, graceful degradation.

These tests never require a working Whisper model: missing files, wrong
suffixes, and an unavailable model must all return ``None`` (never raise) so
the server can answer with its polite "type it instead" reply. The ffmpeg
conversion path IS exercised for real when ffmpeg is on PATH.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from family_whatsapp_bot import transcriber


def test_missing_file_returns_none(tmp_path):
    assert transcriber.transcribe(tmp_path / "nope.ogg") is None


def test_unsupported_suffix_returns_none(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not audio")
    assert transcriber.transcribe(bad) is None


def test_status_shape():
    s = transcriber.status()
    assert set(s) == {
        "faster_whisper", "pyav", "ffmpeg", "model_size", "model_loaded", "available"
    }
    assert isinstance(s["faster_whisper"], bool)
    assert isinstance(s["pyav"], bool)
    assert isinstance(s["ffmpeg"], bool)


def test_wav_passthrough(tmp_path):
    """A .wav input is used as-is — no ffmpeg round trip."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfake")
    assert transcriber._to_wav(wav) == wav


@pytest.mark.skipif(not transcriber.ffmpeg_available(), reason="ffmpeg not installed")
def test_ffmpeg_converts_ogg_to_wav(tmp_path):
    """Real ffmpeg run: sine tone .ogg -> 16 kHz mono PCM WAV."""
    ogg = tmp_path / "tone.ogg"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(ogg)],
        check=True, capture_output=True, timeout=60,
    )
    assert ogg.is_file() and ogg.stat().st_size > 0

    wav = transcriber._to_wav(ogg)
    try:
        assert wav != ogg                     # a new file was produced
        assert wav.is_file() and wav.stat().st_size > 0
        with open(wav, "rb") as fh:
            assert fh.read(4) == b"RIFF"       # valid WAV container
        # 16 kHz mono was requested — verify via ffprobe when present.
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", str(wav)],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            rate, channels = probe.stdout.strip().split(",")[:2]
            assert (rate, channels) == ("16000", "1")
    finally:
        Path(wav).unlink(missing_ok=True)


def test_transcribe_returns_none_when_model_unavailable(tmp_path, monkeypatch):
    """If the Whisper model can't load (missing dep, blocked DLL, OOM),
    transcribe must return None — never raise."""
    monkeypatch.setattr(transcriber, "_load_whisper_model", lambda: None)
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"OggS-not-really")
    assert transcriber.transcribe(audio) is None


def test_pyav_probe_leaves_av_importable(monkeypatch):
    """Whatever the host policy decides, the probe must leave a module named
    'av' in sys.modules — real PyAV, or the inert placeholder we substitute so
    faster-whisper can still be imported."""
    monkeypatch.setattr(transcriber, "_pyav_available", None)
    result = transcriber.pyav_available()
    assert isinstance(result, bool)
    assert "av" in sys.modules
    assert transcriber._pyav_available is result


def test_read_wav_samples_decodes_pcm16(monkeypatch, tmp_path):
    """A 16 kHz mono PCM16 WAV becomes float32 samples in [-1, 1] — the exact
    input faster-whisper accepts when PyAV is unavailable."""
    import array
    import math
    import wave as wave_mod

    n = 16000
    frames = array.array("h", (int(20000 * math.sin(i / 20)) for i in range(n)))
    wav = tmp_path / "tone.wav"
    with wave_mod.open(str(wav), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(frames.tobytes())

    samples = transcriber._read_wav_samples(wav)
    assert samples is not None
    assert samples.dtype.name == "float32"
    assert len(samples) == n
    assert float(samples.max()) <= 1.0 and float(samples.min()) >= -1.0
    assert float(samples.max()) > 0.5          # the tone survived the decode


def test_read_wav_samples_rejects_wrong_format(tmp_path):
    """8 kHz stereo is not what Whisper wants: decline instead of guessing."""
    import array
    import wave as wave_mod

    wav = tmp_path / "stereo.wav"
    with wave_mod.open(str(wav), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(8000)
        fh.writeframes(array.array("h", [0] * 800).tobytes())
    assert transcriber._read_wav_samples(wav) is None


def test_missing_pyav_skips_voice_note_instead_of_crashing(tmp_path, monkeypatch):
    """PyAV blocked + undecodable input => None (server then asks the user to
    type), never an exception."""
    monkeypatch.setattr(transcriber, "_whisper_model", object())
    monkeypatch.setattr(transcriber, "_pyav_available", False)
    monkeypatch.setattr(transcriber, "_read_wav_samples", lambda p: None)
    audio = tmp_path / "broken.wav"
    audio.write_bytes(b"not a wav")
    assert transcriber.transcribe(audio) is None


@pytest.mark.skipif(not transcriber.ffmpeg_available(), reason="ffmpeg not installed")
def test_to_wav_force_reencode_normalises_wav_input(tmp_path):
    """With PyAV blocked a .wav is normalised too, so the stdlib can always
    parse it (rate/channels are enforced, not assumed)."""
    import array
    import wave as wave_mod

    src = tmp_path / "48k_stereo.wav"
    with wave_mod.open(str(src), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(48000)
        fh.writeframes(array.array("h", [0] * 4800).tobytes())

    assert transcriber._to_wav(src) == src                    # untouched by default
    out = transcriber._to_wav(src, force_reencode=True)
    try:
        assert out != src and out.name.startswith("wa_voice_")
        with wave_mod.open(str(out), "rb") as fh:
            assert (fh.getframerate(), fh.getnchannels(), fh.getsampwidth()) == (16000, 1, 2)
        assert transcriber._read_wav_samples(out) is not None
    finally:
        Path(out).unlink(missing_ok=True)


def test_availability_probe_is_cached(monkeypatch):
    """The native-import probe runs once: a cached answer is returned as-is,
    however the host blocks the DLL later. Keeps /health fast and honest."""
    monkeypatch.setattr(transcriber, "_faster_whisper_available", True)
    assert transcriber.faster_whisper_available() is True
    monkeypatch.setattr(transcriber, "_faster_whisper_available", False)
    assert transcriber.faster_whisper_available() is False


def test_failed_model_load_is_cached(monkeypatch):
    """A model that fails to load is not retried on every voice note."""
    monkeypatch.setattr(transcriber, "_whisper_model", None)
    monkeypatch.setattr(transcriber, "_whisper_load_failed", True)
    assert transcriber._load_whisper_model() is None
