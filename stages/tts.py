# stages/tts.py
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from elevenlabs.client import ElevenLabs

import config
logger = logging.getLogger(__name__)

# ── Gender detection ─────────────────────────────────────────────────────────

def _extract_audio_numpy(
    video_path: Path, start_sec: float = 0.0, duration: float = 10.0
) -> tuple[np.ndarray, int]:
    """Extract a mono audio segment from video as a float32 numpy array via ffmpeg."""
    sr = 16000
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(duration),
            "-ac", "1", "-ar", str(sr),
            "-f", "s16le", "-vn",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def _classify_gender(audio: np.ndarray, sr: int) -> str:
    """Return 'male' or 'female' from pitch via autocorrelation.

    Human speech pitch ranges: male ~60–165 Hz, female ~165–400 Hz.
    Threshold at 165 Hz is the standard boundary used in phonetics research.
    """
    frame_len = int(sr * 0.04)   # 40 ms frames
    hop_len   = frame_len // 2
    min_lag   = int(sr / 400)    # 400 Hz ceiling
    max_lag   = int(sr / 60)     # 60 Hz floor

    f0_values: list[float] = []
    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start : start + frame_len].copy()
        frame -= frame.mean()
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < 0.01:          # silence / near-silence — skip
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        corr = corr / (corr[0] + 1e-10)
        if max_lag >= len(corr):
            continue
        peak_idx = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        if corr[peak_idx] > 0.4:   # voiced-frame threshold
            f0_values.append(sr / peak_idx)

    if not f0_values:
        return "female"   # safe default for beauty/cosmetics content

    median_f0 = float(np.median(f0_values))
    gender = "male" if median_f0 < 165.0 else "female"
    logger.info("TTS: gender detection — median F0 = %.1f Hz → %s voice", median_f0, gender)
    return gender


def _extract_vocals_audio_numpy(
    video_path: Path,
    cache_dir: Path,
    start_sec: float = 0.0,
    duration: float = 10.0,
) -> tuple[np.ndarray, int] | None:
    """Extract vocals-only audio for a clip via Demucs.  Cached as vocals_clip.wav.

    BGM bass contaminates autocorrelation and causes pitch halving (a 200 Hz
    female F0 gets misread as 100 Hz, classified male).  Removing BGM first
    fixes this.  Returns (audio, sr) or None on failure.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocals_clip = cache_dir / "vocals_clip.wav"

    if not vocals_clip.exists():
        raw_clip = cache_dir / "_raw_clip.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", str(start_sec),
                 "-i", str(video_path),
                 "-t", str(duration),
                 "-ac", "2", "-ar", "44100", "-vn",
                 str(raw_clip)],
                check=True, capture_output=True,
            )
        except Exception as exc:
            logger.warning("Gender: clip extraction failed: %s", exc)
            return None

        demucs_out = cache_dir / "_demucs_clip"
        try:
            subprocess.run(
                [sys.executable, "-m", "demucs",
                 "--two-stems=vocals", "-o", str(demucs_out), str(raw_clip)],
                check=True, capture_output=True,
            )
            vocals_files = list(demucs_out.rglob("vocals.wav"))
            if not vocals_files:
                logger.warning("Gender: Demucs produced no vocals.wav")
                return None
            shutil.move(str(vocals_files[0]), str(vocals_clip))
        except Exception as exc:
            logger.warning("Gender: Demucs separation failed: %s", exc)
            return None
        finally:
            raw_clip.unlink(missing_ok=True)
            shutil.rmtree(demucs_out, ignore_errors=True)

    sr = 16000
    try:
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(vocals_clip),
             "-ac", "1", "-ar", str(sr),
             "-f", "s16le", "-vn",
             "pipe:1"],
            capture_output=True, check=True,
        )
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sr
    except Exception as exc:
        logger.warning("Gender: vocals readback failed: %s", exc)
        return None


def detect_speaker_gender(
    video_path: Path,
    speech_segments: list[dict] | None = None,
    cache_dir: Path | None = None,
) -> str:
    """Detect speaker gender from the video audio.

    If speech_segments is provided, analysis starts at the first segment's
    start_sec so we analyse actual speech rather than intro music/silence.
    If cache_dir is provided, vocals are isolated via Demucs first, which
    avoids BGM-induced pitch halving on raw audio.  Returns 'male' or
    'female'.  Falls back to 'female' on any error.
    """
    start_sec = 0.0
    if speech_segments:
        start_sec = float(speech_segments[0].get("start_sec", 0.0))

    audio_data = None
    if cache_dir is not None:
        audio_data = _extract_vocals_audio_numpy(
            video_path, cache_dir, start_sec=start_sec, duration=10.0,
        )
    if audio_data is None:
        try:
            audio_data = _extract_audio_numpy(video_path, start_sec=start_sec, duration=10.0)
        except Exception as exc:
            logger.warning("TTS: gender detection failed (%s) — defaulting to female voice", exc)
            return "female"

    audio, sr = audio_data
    return _classify_gender(audio, sr)


# ── Voice selection ───────────────────────────────────────────────────────────

def _resolve_voice_id(
    video_path: Path | None,
    speech_segments: list[dict] | None = None,
    cache_dir: Path | None = None,
) -> str:
    """Return the ElevenLabs voice ID to use.

    Priority:
    1. ELEVENLABS_VOICE_ID env var (manual override)
    2. Auto-detected gender from video audio
    3. Female voice as fallback
    """
    if config.ELEVENLABS_VOICE_ID:
        return config.ELEVENLABS_VOICE_ID

    if video_path and video_path.exists():
        gender = detect_speaker_gender(video_path, speech_segments, cache_dir=cache_dir)
        return (
            config.ELEVENLABS_VOICE_ID_MALE
            if gender == "male"
            else config.ELEVENLABS_VOICE_ID_FEMALE
        )

    return config.ELEVENLABS_VOICE_ID_FEMALE


# ── Audio utilities ───────────────────────────────────────────────────────────

def _wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path)) as w:
        return w.getnframes() / w.getframerate()


# ── Synthesis helpers ─────────────────────────────────────────────────────────

def _synthesize(text: str, client: ElevenLabs, voice_id: str, path: Path) -> None:
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=config.ELEVENLABS_MODEL,
    )
    with open(str(path), "wb") as f:
        for chunk in audio:
            f.write(chunk)


def _mp3_path_for(text: str, tts_dir: Path) -> Path:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return tts_dir / f"{digest}.mp3"


def clip_path_for(text: str, tts_dir: Path) -> Path:
    """Return the WAV path for a given text in tts_dir.

    ElevenLabs outputs MP3 which is converted to 48kHz stereo PCM WAV using
    ffmpeg for Premiere Pro compatibility.
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return tts_dir / f"{digest}.wav"


def _convert_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert mp3_path to 48kHz stereo PCM WAV using ffmpeg.

    Returns True on success, False on failure (wav_path may not exist if False).
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp3_path),
                "-ar", "48000",
                "-ac", "2",
                "-acodec", "pcm_s16le",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except Exception as exc:
        logger.warning("TTS: WAV conversion failed for %s: %s", mp3_path.name, exc)
        return False


def _synthesize_texts(
    texts: list[str],
    tts_dir: Path,
    voice_id: str,
) -> dict[str, Path]:
    """Synthesize a list of Japanese texts into WAV files. Returns {text: wav_path}."""
    tts_dir.mkdir(exist_ok=True)
    client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    seen: dict[str, Path] = {}

    for text in texts:
        if not text or text in seen:
            continue

        wav_path = clip_path_for(text, tts_dir)
        seen[text] = wav_path

        if not wav_path.exists():
            mp3_path = _mp3_path_for(text, tts_dir)
            logger.info("TTS: synthesising → %s", text[:60])
            try:
                if not mp3_path.exists():
                    _synthesize(text, client, voice_id, mp3_path)
                if not _convert_to_wav(mp3_path, wav_path):
                    raise RuntimeError("WAV conversion failed")
            except Exception as exc:
                logger.warning("TTS: clip failed for '%s': %s", text[:40], exc)
                del seen[text]

    logger.info("TTS: %d unique audio clips ready", len(seen))
    return seen


# ── Public API ────────────────────────────────────────────────────────────────

def run_speech(
    speech_segments: list[dict],
    checkpoint_dir: Path,
    video_path: Path | None = None,
) -> dict[str, Path]:
    """Generate Japanese TTS WAV files from translated speech segments.

    Returns {text_ja: wav_path} for every speech segment with non-empty text_ja.
    If video_path is provided, speaker gender is auto-detected from the first
    speech segment's audio and the matching ElevenLabs voice is used.
    """
    tts_dir  = checkpoint_dir / "tts"
    texts    = [s["text_ja"] for s in speech_segments if s.get("text_ja")]
    voice_id = _resolve_voice_id(video_path, speech_segments, cache_dir=checkpoint_dir / "_gender")
    tts_map  = _synthesize_texts(texts, tts_dir, voice_id)

    # TTS plays at natural Japanese cadence. The editor adjusts clip speed in
    # Premiere if a clip overruns its speech window — fitting via atempo
    # produced rushed, unnatural narration.

    return tts_map


def generate_tts_map(
    overlays: list,
    checkpoint_dir: Path,
    video_path: Path | None = None,
) -> dict[str, Path]:
    """Generate Japanese TTS WAV files from translated text overlays.

    Returns {text_ja: wav_path} for overlays with non-empty text_ja.
    Used by the web pipeline (no GVI speech segments available).
    """
    tts_dir = checkpoint_dir / "tts"
    texts = [o.text_ja for o in overlays if o.text_ja]
    voice_id = _resolve_voice_id(video_path)
    return _synthesize_texts(texts, tts_dir, voice_id)


def extract_bgm(video_path: Path, out_dir: Path) -> Path | None:
    """Separate background music from speech using Demucs source separation.

    Extracts full-quality stereo audio from the video, runs Demucs to remove
    the vocal track, and converts the result to 48kHz stereo PCM WAV for
    Premiere Pro.  The output is cached as out_dir/bgm.wav.

    Returns the path to bgm.wav, or None if Demucs is unavailable or fails.
    """
    out_dir.mkdir(exist_ok=True)
    bgm_path = out_dir / "bgm.wav"
    if bgm_path.exists():
        return bgm_path

    # Extract full-quality stereo audio (44.1kHz for Demucs; higher than the
    # 16kHz mono we use for Whisper, so we do this separately).
    raw_audio = out_dir / "_raw_audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-ac", "2", "-ar", "44100", "-vn", str(raw_audio)],
            check=True, capture_output=True,
        )
    except Exception as exc:
        logger.warning("BGM: audio extraction failed: %s", exc)
        raw_audio.unlink(missing_ok=True)
        return None

    demucs_out = out_dir / "_demucs"
    try:
        subprocess.run(
            [sys.executable, "-m", "demucs",
             "--two-stems=vocals", "-o", str(demucs_out), str(raw_audio)],
            check=True, capture_output=True,
        )
        no_vocals_list = list(demucs_out.rglob("no_vocals.wav"))
        if not no_vocals_list:
            logger.warning("BGM: no_vocals.wav not found in Demucs output")
            return None

        # Convert Demucs output (44.1kHz) to 48kHz stereo PCM WAV for Premiere Pro
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(no_vocals_list[0]),
                "-ar", "48000",
                "-ac", "2",
                "-acodec", "pcm_s16le",
                str(bgm_path),
            ],
            check=True, capture_output=True,
        )
        logger.info("BGM: extracted background music → %s (%.0f s)",
                    bgm_path.name, _wav_duration(bgm_path))
        return bgm_path

    except Exception as exc:
        logger.warning("BGM: extraction failed: %s", exc)
        bgm_path.unlink(missing_ok=True)
        return None
    finally:
        raw_audio.unlink(missing_ok=True)
        shutil.rmtree(demucs_out, ignore_errors=True)
