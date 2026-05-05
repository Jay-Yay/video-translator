"""Orchestration stage for exporting video projects with Premiere Pro integration.

Handles the output folder layout for a single video and calls prproj_builder.build_prproj.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import config
from models.text_overlay import TextOverlay
from stages import prproj_builder, tts as tts_stage

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = config.ROOT_DIR / "templates" / "base.prproj"


def run(
    video_path: Path,
    overlays: list[TextOverlay],
    tts_map: dict[str, Path],
    output_dir: Path,
    speech_segments: list[dict] | None = None,
    relative_paths: bool = False,
) -> Path:
    """Assemble the Premiere Pro project for one video.

    Creates:
        output_dir/{stem}/
            {stem}.mp4          ← original video (moved)
            tts/*.wav           ← TTS clips (copied)
            {stem}.prproj       ← Premiere Pro 2026 project

    Returns the path to the generated .prproj file.
    """
    stem = video_path.stem
    project_dir = output_dir / stem
    project_dir.mkdir(parents=True, exist_ok=True)

    # Move original video into project folder
    dest_video = project_dir / video_path.name
    shutil.move(str(video_path), str(dest_video))
    logger.info("[%s] Moved original video → %s", stem, dest_video)

    # Copy TTS clips into project folder (keep them next to .prproj for portability)
    tts_dest_dir = project_dir / "tts"
    tts_dest_dir.mkdir(exist_ok=True)
    dest_tts_map: dict[str, Path] = {}
    for text_ja, src_path in tts_map.items():
        dest = tts_dest_dir / src_path.name
        shutil.copy2(str(src_path), str(dest))
        dest_tts_map[text_ja] = dest
    logger.info("[%s] Copied %d TTS clip(s) → %s", stem, len(dest_tts_map), tts_dest_dir)

    # Extract BGM (background music with vocals removed) for A3 track.
    # Runs Demucs on the video audio; cached as tts/bgm.wav so reruns are instant.
    bgm_path: Path | None = None
    if dest_tts_map:
        bgm_path = tts_stage.extract_bgm(dest_video, tts_dest_dir)
        if bgm_path:
            logger.info("[%s] BGM extracted → %s", stem, bgm_path.name)
        else:
            logger.info("[%s] BGM extraction skipped or failed — A3 will be empty", stem)

    # Build .prproj
    prproj_path = project_dir / f"{stem}.prproj"
    prproj_builder.build_prproj(
        _TEMPLATE_PATH, dest_video, overlays, dest_tts_map, prproj_path,
        speech_segments=speech_segments,
        bgm_path=bgm_path,
        relative_paths=relative_paths,
    )
    logger.info("[%s] Generated %s", stem, prproj_path)

    return prproj_path
