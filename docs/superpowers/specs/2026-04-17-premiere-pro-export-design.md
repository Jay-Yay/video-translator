# Premiere Pro Export — Design Spec

**Date:** 2026-04-17
**Status:** Approved
**Target:** Premiere Pro 2026

---

## Overview

Replace the current burned-in `.mp4` output with a Premiere Pro 2026 project file (`.prproj`) that contains the original video, positioned Japanese text overlay clips (Essential Graphics, editable), and Japanese TTS audio clips on a dedicated track. The Korean audio is muted but preserved.

---

## Goals

- Output a `.prproj` file that opens in PP2026 with Japanese text + audio fully set up
- Text overlays must be editable inside Premiere (not burned-in PNG)
- Original video is left untouched as the source media
- No `{stem}_ja.mp4` is produced

## Non-Goals

- Voiceover replacement (Korean audio is muted, not removed)
- Premiere project packaging / media duplication
- Support for Premiere versions other than 2025

---

## Architecture

### Pipeline Change

**Before:**
```
detect_gvi → translate → render (burn-in) → tts (audio mix) → {stem}_ja.mp4
```

**After:**
```
detect_gvi → translate → tts (mp3 only) → export_prproj → {stem}.prproj
```

### File Changes

| File | Change |
|------|--------|
| `stages/render.py` | No longer called — kept for reference but removed from pipeline |
| `stages/tts.py` | Stripped to mp3 generation only (remove video write logic) |
| `stages/export_prproj.py` | **New** — assembles `.prproj` from all inputs |
| `stages/prproj_builder.py` | **New** — low-level XML manipulation (zlib decompress → inject → recompress) |
| `templates/base.prproj` | **New** — minimal PP2026 project with empty sequence + one sample text clip |
| `translate_video_gvi.py` | Remove render/tts calls, add export_prproj call, remove `_ja.mp4` output path |

### Output Folder Structure

```
videos-done/
└── {stem}/
    ├── {stem}.prproj       ← Premiere Pro 2026 project file
    ├── {stem}.mp4          ← original video (source media, moved from videos-to-be-done)
    └── tts/
        ├── {hash}.mp3      ← Japanese TTS clips (copied from checkpoints)
        └── ...
```

---

## Component Design

### `stages/tts.py` (modified)

Remove all MoviePy video-write logic. New signature:

```python
def run(overlays: list[TextOverlay], checkpoint_dir: Path) -> dict[str, Path]:
    """Generate Japanese TTS mp3 files for all unique text_ja.
    Returns: {text_ja: mp3_path} for overlays that have audio.
    """
```

- Unchanged: deduplication by `text_ja`, MD5-hash filenames, async Edge TTS synthesis, checkpoint caching
- Removed: loading rendered video, CompositeAudioClip, VideoFileClip write

### `stages/prproj_builder.py` (new)

Low-level XML builder:

```python
def load_template(template_path: Path) -> ET.Element
    """zlib-decompress base.prproj → parse XML → return root element."""

def clone_text_clip(template_clip_el: ET.Element, overlay: TextOverlay,
                    video_w: int, video_h: int, fps: float) -> ET.Element
    """Clone the sample text clip element, then set:
    - text content → overlay.text_ja
    - position → bbox center converted to PP coordinate space
    - in/out points → start_sec / end_sec in PP ticks (1 tick = 1/254016000 s)
    """

def build_prproj(template_path: Path, video_path: Path,
                 overlays: list[TextOverlay], tts_map: dict[str, Path],
                 output_path: Path) -> None
    """Full assembly:
    1. load_template
    2. Probe video_path with cv2 for (video_w, video_h, fps)
    3. Update sequence settings in XML to match probed resolution + FPS
    4. Inject V1 video clip (full duration, A1 muted)
    5. For each overlay: clone_text_clip → insert on V2
    6. For each tts_map entry: build audio clip → insert on A2 at start_sec
    7. zlib-compress → write output_path
    """
```

**Coordinate conversion (bbox → Premiere):**
```
center_x_px = bbox_x + bbox_w / 2
center_y_px = bbox_y + bbox_h / 2
pp_x = center_x_px - video_w / 2          # origin at frame center
pp_y = -(center_y_px - video_h / 2)       # y-axis inverted in PP
```

**Timecode conversion:**
```
pp_ticks = int(time_sec * 254016000)
```

**Text clip defaults (editable in Premiere after import):**
- Font: NotoSansJP-Bold
- Color: white
- Background: black semi-transparent box
- Size: auto-fit to bbox dimensions (set via PP's text size parameter)

### `stages/export_prproj.py` (new)

Orchestration:

```python
def run(video_path: Path, overlays: list[TextOverlay],
        tts_map: dict[str, Path], output_dir: Path) -> Path:
    """
    1. Create output_dir/{stem}/ folder
    2. Move video_path → output_dir/{stem}/{stem}.mp4
    3. Copy tts mp3s → output_dir/{stem}/tts/
    4. Call prproj_builder.build_prproj(...)
    5. Return path to .prproj
    """
```

### `translate_video_gvi.py` (modified)

```python
# Stage 3: TTS (mp3 generation only)
tts_map = tts.run(translated, checkpoint_dir)

# Stage 4: Export .prproj
output_dir = config.VIDEOS_OUTPUT_DIR
export_prproj.run(video_path, translated, tts_map, output_dir)
```

Remove:
- `output_path = config.VIDEOS_OUTPUT_DIR / f"{stem}_ja.mp4"`
- `render.run(...)`
- old `tts.run(overlays, rendered_video_path, output_path, checkpoint_dir)`
- `shutil.move` of video (now handled inside export_prproj)

### `templates/base.prproj`

Manually created in Premiere Pro 2026 once:
1. New project → new sequence (1080p, 29.97fps or match most common source)
2. Add one Essential Graphics text clip on V2 (any text, any position)
3. Save → this file becomes `templates/base.prproj`
4. Python uses this as the XML schema reference for text clip structure

The template is committed to the repo. If PP format changes, update the template.

---

## Sequence Track Layout

| Track | Content | Notes |
|-------|---------|-------|
| V1 | `{stem}.mp4` — full duration | source video |
| V2 | Essential Graphics text clips | one per overlay, positioned + timed |
| A1 | Original audio from `{stem}.mp4` | **muted** (volume = 0) |
| A2 | TTS `.mp3` clips | one per overlay at `start_sec` |

---

## Error Handling

| Case | Behaviour |
|------|-----------|
| `text_ja` is empty for an overlay | Skip text clip and TTS clip for that overlay |
| TTS synthesis failed for an overlay | Skip A2 clip; V2 text clip still added |
| Template file missing | `sys.exit` with clear message pointing to creation instructions |
| Video probe fails (no FPS/dimensions) | Fall back to 1920×1080 / 29.97; log warning |

---

## Dependencies

No new Python packages required beyond what already exists:
- `xml.etree.ElementTree` (stdlib) — XML parsing/generation
- `zlib` (stdlib) — `.prproj` compress/decompress
- `moviepy` — removed from tts.py; no longer needed for audio mixing
- `edge_tts` — unchanged

---

## Testing

- `tests/test_export_prproj.py` — unit tests for coordinate conversion, timecode conversion, XML injection
- Manual smoke test: open generated `.prproj` in PP2026, verify V1/V2/A1/A2 track layout, edit a text clip
