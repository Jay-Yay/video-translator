"""Regenerate TTS for all videos: Whisper re-transcription → Claude translation
→ ElevenLabs TTS (gender-detected) → rebuilt .prproj.

Run from the project root:
    python regenerate_tts.py
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import whisper

import config
from models.text_overlay import TextOverlay
from stages import tts, export_prproj
from stages.translate import translate_speech

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("regenerate_tts")

WHISPER_MODEL = "large-v3"

# Beauty/cosmetics vocabulary primes Whisper to recognise domain-specific Korean.
WHISPER_INITIAL_PROMPT = (
    "뷰티 스킨케어 경락 마사지 리프팅 사각턱 브이라인 V세라 중주파 "
    "찐 극복템 갓성비 비포 애프터 효과 피부 화장품 안면 비대칭"
)

WHISPER_MIN_CONFIDENCE = -0.5    # avg_logprob below this = hallucinated / no speech
DEMUCS_CONFIDENCE_THRESHOLD = -0.40  # retry with vocal isolation below this


def extract_audio(video_path: Path, out_wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-vn",
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )


def _isolate_vocals(audio_path: Path) -> tuple[Path, Path | None]:
    """Separate vocals from BGM using Demucs.

    Returns (vocals_wav_path, cleanup_dir). Caller must shutil.rmtree(cleanup_dir).
    Falls back to (audio_path, None) if Demucs is unavailable or fails.
    """
    demucs_out = audio_path.parent / f"demucs_{audio_path.stem}"
    try:
        subprocess.run(
            [sys.executable, "-m", "demucs",
             "--two-stems=vocals", "-o", str(demucs_out), str(audio_path)],
            check=True, capture_output=True,
        )
        vocals_list = list(demucs_out.rglob("vocals.wav"))
        if vocals_list:
            logger.info("Vocal isolation: BGM separated from voice")
            return vocals_list[0], demucs_out
    except Exception as exc:
        logger.debug("Demucs unavailable (%s)", exc)
        if demucs_out.exists():
            shutil.rmtree(demucs_out, ignore_errors=True)
    return audio_path, None


def transcribe_korean(audio_path: Path, model: whisper.Whisper) -> list[dict]:
    result = model.transcribe(
        str(audio_path),
        language="ko",
        task="transcribe",
        initial_prompt=WHISPER_INITIAL_PROMPT,
    )
    segments = [
        {
            "text_ko": seg["text"].strip(),
            "text_ja": "",
            "start_sec": float(seg["start"]),
            "end_sec":   float(seg["end"]),
            "confidence": float(seg.get("avg_logprob", 0.0)),
        }
        for seg in result["segments"] if seg["text"].strip()
    ]
    if not segments:
        return []
    merged = " ".join(s["text_ko"] for s in segments)
    return [{
        "text_ko":    merged,
        "text_ja":    "",
        "start_sec":  segments[0]["start_sec"],
        "end_sec":    segments[-1]["end_sec"],
        "confidence": sum(s["confidence"] for s in segments) / len(segments),
    }]


def process_video(stem: str, model: whisper.Whisper) -> bool:
    video_path     = config.VIDEOS_OUTPUT_DIR / stem / f"{stem}.mp4"
    checkpoint_dir = config.CHECKPOINTS_DIR / stem

    if not video_path.exists():
        logger.warning("[%s] Video not found — skipping", stem)
        return False

    # Respect GVI's speech detection flag — if it says no Korean speech, skip.
    speech_flag = checkpoint_dir / "has_korean_speech.json"
    if speech_flag.exists():
        has_speech = json.loads(speech_flag.read_text(encoding="utf-8"))
        if not has_speech:
            logger.info("[%s] has_korean_speech=false — skipping TTS", stem)
            return False

    # ── Step 1: Transcribe with Whisper ───────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = Path(f.name)
    demucs_dir: Path | None = None
    try:
        extract_audio(video_path, audio_path)
        segments = transcribe_korean(audio_path, model)

        if segments and segments[0]["confidence"] < DEMUCS_CONFIDENCE_THRESHOLD:
            logger.info(
                "[%s] Low confidence %.2f — retrying with vocal isolation…",
                stem, segments[0]["confidence"],
            )
            clean_path, demucs_dir = _isolate_vocals(audio_path)
            if clean_path != audio_path:
                retry = transcribe_korean(clean_path, model)
                if retry and retry[0]["confidence"] > segments[0]["confidence"]:
                    logger.info(
                        "[%s] Vocal isolation improved confidence %.2f → %.2f",
                        stem, segments[0]["confidence"], retry[0]["confidence"],
                    )
                    segments = retry
    finally:
        audio_path.unlink(missing_ok=True)
        if demucs_dir:
            shutil.rmtree(demucs_dir, ignore_errors=True)

    if not segments:
        logger.info("[%s] Whisper found no Korean speech — skipping", stem)
        return False

    if segments[0]["confidence"] < WHISPER_MIN_CONFIDENCE:
        logger.info(
            "[%s] Whisper confidence %.2f below threshold — likely no real speech, skipping",
            stem, segments[0]["confidence"],
        )
        return False

    logger.info("[%s] Korean: %s…", stem, segments[0]["text_ko"][:80])
    (checkpoint_dir / "speech_segments_whisper.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Step 2: Translate with Claude ────────────────────────────────────
    translated = translate_speech(segments, checkpoint_dir)
    if not translated or not translated[0].get("text_ja"):
        logger.warning("[%s] Translation empty — skipping", stem)
        return False

    logger.info("[%s] Japanese: %s…", stem, translated[0]["text_ja"][:80])
    (checkpoint_dir / "speech_translations.json").write_text(
        json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Step 3: Regenerate TTS ───────────────────────────────────────────
    tts_dir = checkpoint_dir / "tts"
    if tts_dir.exists():
        shutil.rmtree(tts_dir)

    tts_map = tts.run_speech(translated, checkpoint_dir, video_path=video_path)
    if not tts_map:
        logger.warning("[%s] TTS generated 0 clips — skipping prproj", stem)
        return False

    # ── Step 4: Rebuild .prproj ───────────────────────────────────────────
    overlays_path = checkpoint_dir / "translations.json"
    overlays = (
        [TextOverlay.from_dict(d) for d in json.loads(overlays_path.read_text(encoding="utf-8"))]
        if overlays_path.exists() else []
    )
    export_prproj.run(
        video_path, overlays, tts_map, config.VIDEOS_OUTPUT_DIR,
        speech_segments=translated,
    )
    logger.info("[%s] Done — %d TTS clip(s), prproj rebuilt", stem, len(tts_map))
    return True


def main() -> None:
    stems = sorted(d.name for d in config.CHECKPOINTS_DIR.iterdir() if d.is_dir())
    if not stems:
        sys.exit("No checkpoints found.")

    logger.info("Loading Whisper model '%s'…", WHISPER_MODEL)
    model = whisper.load_model(WHISPER_MODEL)

    ok = skip = 0
    for stem in stems:
        logger.info("── %s ──", stem)
        if process_video(stem, model):
            ok += 1
        else:
            skip += 1

    logger.info("Finished: %d processed, %d skipped", ok, skip)


if __name__ == "__main__":
    main()
