"""Fix TTS for a single video: re-transcribe Korean audio with Whisper,
re-translate with Claude, then regenerate TTS.

Usage:
    python fix_video_tts.py <video_stem>

Example:
    python fix_video_tts.py "vs_my0320_1_브이쎄라알기전"
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
logger = logging.getLogger("fix_video_tts")

WHISPER_MODEL = "large-v3"

# Beauty/cosmetics vocabulary primes Whisper to recognise domain-specific Korean.
WHISPER_INITIAL_PROMPT = (
    "뷰티 스킨케어 경락 마사지 리프팅 사각턱 브이라인 V세라 중주파 "
    "찐 극복템 갓성비 비포 애프터 효과 피부 화장품 안면 비대칭"
)

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
    """Transcribe Korean audio with Whisper. Returns list of merged segment dicts."""
    logger.info("Transcribing %s…", audio_path.name)
    result = model.transcribe(
        str(audio_path),
        language="ko",
        task="transcribe",
        initial_prompt=WHISPER_INITIAL_PROMPT,
    )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if text:
            segments.append({
                "text_ko": text,
                "text_ja": "",
                "start_sec": float(seg["start"]),
                "end_sec":   float(seg["end"]),
                "confidence": float(seg.get("avg_logprob", 0.0)),
            })

    if segments:
        merged_ko = " ".join(s["text_ko"] for s in segments)
        segments = [{
            "text_ko":    merged_ko,
            "text_ja":    "",
            "start_sec":  segments[0]["start_sec"],
            "end_sec":    segments[-1]["end_sec"],
            "confidence": sum(s["confidence"] for s in segments) / len(segments),
        }]

    return segments


def main() -> None:
    args = sys.argv[1:]
    skip_whisper = "--skip-whisper" in args
    stems = [a for a in args if not a.startswith("--")]
    if not stems:
        sys.exit("Usage: python fix_video_tts.py [--skip-whisper] <video_stem>")

    stem = stems[0]
    video_path = config.VIDEOS_OUTPUT_DIR / stem / f"{stem}.mp4"
    checkpoint_dir = config.CHECKPOINTS_DIR / stem

    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    if not checkpoint_dir.exists():
        sys.exit(f"Checkpoint dir not found: {checkpoint_dir}")

    # ── Step 1: Transcribe Korean audio with Whisper ──────────────────────
    whisper_json = checkpoint_dir / "speech_segments_whisper.json"
    if skip_whisper:
        if not whisper_json.exists():
            sys.exit(f"--skip-whisper requested but {whisper_json} not found")
        segments = json.loads(whisper_json.read_text(encoding="utf-8"))
        logger.info("Step 1: using existing transcript (--skip-whisper)")
        logger.info("Korean transcript: %s", segments[0]["text_ko"][:120])
    else:
        logger.info("Step 1: re-transcribing Korean audio with Whisper…")
        logger.info("Loading Whisper model '%s'…", WHISPER_MODEL)
        model = whisper.load_model(WHISPER_MODEL)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = Path(f.name)
        demucs_dir: Path | None = None
        try:
            extract_audio(video_path, audio_path)
            segments = transcribe_korean(audio_path, model)

            # If confidence is low (likely BGM interference), retry with vocal isolation.
            if segments and segments[0]["confidence"] < DEMUCS_CONFIDENCE_THRESHOLD:
                logger.info(
                    "Low Whisper confidence %.2f — retrying with vocal isolation…",
                    segments[0]["confidence"],
                )
                clean_path, demucs_dir = _isolate_vocals(audio_path)
                if clean_path != audio_path:
                    retry = transcribe_korean(clean_path, model)
                    if retry and retry[0]["confidence"] > segments[0]["confidence"]:
                        logger.info(
                            "Vocal isolation improved confidence %.2f → %.2f",
                            segments[0]["confidence"], retry[0]["confidence"],
                        )
                        segments = retry
        finally:
            audio_path.unlink(missing_ok=True)
            if demucs_dir:
                shutil.rmtree(demucs_dir, ignore_errors=True)

        if not segments:
            sys.exit("Whisper returned no segments — check the audio.")

        logger.info("Korean transcript: %s", segments[0]["text_ko"][:120])

        # Save corrected Korean transcript
        whisper_json.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── Step 2: Translate with Claude ─────────────────────────────────────
    logger.info("Step 2: translating with Claude…")
    translated = translate_speech(segments, checkpoint_dir)
    if not translated or not translated[0].get("text_ja"):
        sys.exit("Translation returned empty — check Claude API key.")

    logger.info("Japanese translation: %s", translated[0]["text_ja"][:120])

    # Overwrite speech_translations.json
    (checkpoint_dir / "speech_translations.json").write_text(
        json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Step 3: Clear old TTS cache and regenerate ────────────────────────
    logger.info("Step 3: regenerating TTS…")
    tts_dir = checkpoint_dir / "tts"
    if tts_dir.exists():
        shutil.rmtree(tts_dir)

    tts_map = tts.run_speech(translated, checkpoint_dir, video_path=video_path)
    logger.info("Generated %d TTS clip(s)", len(tts_map))
    for text, wav_path in tts_map.items():
        logger.info("  → %s  (%s)", wav_path.name, text[:60])

    # ��─ Step 4: Rebuild .prproj to reference the new TTS file ────────────
    logger.info("Step 4: rebuilding Premiere Pro project…")
    overlays_path = checkpoint_dir / "translations.json"
    overlays = (
        [TextOverlay.from_dict(d) for d in json.loads(overlays_path.read_text(encoding="utf-8"))]
        if overlays_path.exists() else []
    )
    prproj_path = export_prproj.run(
        video_path, overlays, tts_map, config.VIDEOS_OUTPUT_DIR,
        speech_segments=translated,
    )
    logger.info("Done → %s", prproj_path)


if __name__ == "__main__":
    main()
