# translate_video_gvi.py
# Pipeline variant that uses Google Video Intelligence API for text detection.
# Detection is faster, higher quality, and returns pre-deduplicated results.
#
# Prerequisites:
#   1. Enable the Video Intelligence API in your Google Cloud project.
#   2. Create a service account key and set:
#        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
#   3. pip install google-cloud-videointelligence
#
# Usage:
#   python translate_video_gvi.py               # full run
#   python translate_video_gvi.py --skip-detect # resume from existing detections.json
#   python translate_video_gvi.py --force       # re-run all stages
from __future__ import annotations
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import config
from models.text_overlay import TextOverlay
from stages import translate, tts, export_prproj
from stages import detect_gvi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("translate_video_gvi")


def _startup_check(need_gvi: bool = True) -> None:
    if not config.ANTHROPIC_API_KEY:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is set but empty")

    if need_gvi:
        # Verify Google credentials are reachable (only needed for Stage 1 detection)
        try:
            import google.auth
            google.auth.default()
        except Exception as exc:
            sys.exit(
                f"ERROR: Google credentials not found: {exc}\n"
                "Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json"
            )

    config.VIDEOS_INPUT_DIR.mkdir(exist_ok=True)
    config.VIDEOS_OUTPUT_DIR.mkdir(exist_ok=True)
    config.CHECKPOINTS_DIR.mkdir(exist_ok=True)


def _load_overlays(path: Path) -> list[TextOverlay]:
    return [TextOverlay.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def process_video(video_path: Path, args: argparse.Namespace) -> None:
    stem = video_path.stem
    checkpoint_dir = config.CHECKPOINTS_DIR / stem
    checkpoint_dir.mkdir(exist_ok=True)

    det_path = checkpoint_dir / "detections.json"
    trans_path = checkpoint_dir / "translations.json"

    # ── Stage 1: Detect via GVI ──────────────────────────────────────────────
    # GVI returns pre-deduplicated results — no separate deduplication step needed.
    # Also detects whether the video contains Korean speech audio.
    speech_flag_path = checkpoint_dir / "has_korean_speech.json"
    has_korean_speech = True  # default: assume speech exists (safe fallback)
    if not args.skip_detect:
        if not det_path.exists() or args.force:
            logger.info("[%s] Stage 1 (GVI): detecting Korean text…", stem)
            overlays, has_korean_speech = detect_gvi.run(video_path, checkpoint_dir)
        else:
            overlays = _load_overlays(det_path)
            if speech_flag_path.exists():
                has_korean_speech = json.loads(speech_flag_path.read_text(encoding="utf-8"))
            logger.info("[%s] Stage 1: loaded %d detections from checkpoint", stem, len(overlays))
    else:
        if not det_path.exists():
            logger.warning("[%s] --skip-detect: detections.json missing — skipping video", stem)
            return
        overlays = _load_overlays(det_path)
        if speech_flag_path.exists():
            has_korean_speech = json.loads(speech_flag_path.read_text(encoding="utf-8"))
        logger.info("[%s] --skip-detect: loaded %d overlays", stem, len(overlays))

    if not overlays:
        logger.info("[%s] No Korean text found — moving original to videos-done/%s/", stem, stem)
        dest_dir = config.VIDEOS_OUTPUT_DIR / stem
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(video_path), str(dest_dir / video_path.name))
        return

    # ── Stage 2: Translate ───────────────────────────────────────────────────
    if not args.skip_translate:
        if not trans_path.exists() or args.force:
            logger.info("[%s] Stage 2: translating with Claude…", stem)
            translated = translate.run(overlays, checkpoint_dir)
        else:
            translated = _load_overlays(trans_path)
            logger.info("[%s] Stage 2: loaded %d translations from checkpoint", stem, len(translated))
    else:
        if not trans_path.exists():
            logger.warning("[%s] --skip-translate: translations.json missing — skipping video", stem)
            return
        translated = _load_overlays(trans_path)
        logger.info("[%s] --skip-translate: loaded %d overlays", stem, len(translated))

    if all(not o.text_ja for o in translated):
        logger.warning("[%s] All translations failed — moving original, skipping prproj", stem)
        dest_dir = config.VIDEOS_OUTPUT_DIR / stem
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(video_path), str(dest_dir / video_path.name))
        return

    # ── Stage 2b: Translate speech segments (for TTS) ──────────────────────
    speech_segments: list[dict] = []
    tts_map: dict[str, Path] = {}
    if has_korean_speech:
        speech_seg_path = checkpoint_dir / "speech_segments.json"
        speech_trans_path = checkpoint_dir / "speech_translations.json"

        if speech_seg_path.exists():
            raw_segments = json.loads(speech_seg_path.read_text(encoding="utf-8"))
        else:
            raw_segments = []

        if raw_segments:
            if not speech_trans_path.exists() or args.force:
                logger.info("[%s] Stage 2b: translating %d speech segments…", stem, len(raw_segments))
                speech_segments = translate.translate_speech(raw_segments, checkpoint_dir)
            else:
                speech_segments = json.loads(speech_trans_path.read_text(encoding="utf-8"))
                logger.info("[%s] Stage 2b: loaded %d speech translations from checkpoint", stem, len(speech_segments))

    # ── Stage 3: TTS from speech transcription (not overlay text) ────────
    if has_korean_speech and speech_segments:
        logger.info("[%s] Stage 3: generating TTS from speech transcription…", stem)
        tts_map = tts.run_speech(speech_segments, checkpoint_dir, video_path=video_path)
    elif has_korean_speech:
        logger.info("[%s] Stage 3: Korean speech detected but no segments transcribed — skipping TTS", stem)
    else:
        logger.info("[%s] Stage 3: skipped TTS — no Korean speech detected in audio", stem)

    # ── Stage 4: Export .prproj ───────────────────────────────────────────────
    logger.info("[%s] Stage 4: assembling Premiere Pro project…", stem)
    prproj_path = export_prproj.run(
        video_path, translated, tts_map, config.VIDEOS_OUTPUT_DIR,
        speech_segments=speech_segments,
    )
    logger.info("✓ [%s] complete → %s", stem, prproj_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate Korean on-screen text to Japanese using Google Video Intelligence API."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages ignoring checkpoints")
    parser.add_argument("--skip-detect", dest="skip_detect", action="store_true",
                        help="Skip Stage 1 (use existing detections.json)")
    parser.add_argument("--skip-translate", dest="skip_translate", action="store_true",
                        help="Skip Stage 2 (use existing translations.json)")
    args = parser.parse_args()

    _startup_check(need_gvi=not args.skip_detect)

    videos = sorted(config.VIDEOS_INPUT_DIR.glob("*.mp4"))
    if not videos:
        logger.info("No .mp4 files found in %s", config.VIDEOS_INPUT_DIR)
        return

    logger.info("Found %d video(s) to process", len(videos))
    for video_path in videos:
        try:
            process_video(video_path, args)
        except Exception as exc:
            logger.error(
                "[%s] FAILED: %s — skipping, video stays in input folder",
                video_path.stem, exc, exc_info=True,
            )


if __name__ == "__main__":
    main()
