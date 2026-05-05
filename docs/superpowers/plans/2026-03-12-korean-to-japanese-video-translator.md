# Korean-to-Japanese Video Translator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staged Python pipeline that detects Korean text in `.mp4` videos via PaddleOCR, translates it to Japanese using Claude, and composites solid-color overlay boxes onto the video using MoviePy.

**Architecture:** Three-stage pipeline with JSON checkpoints per video. Stage 1 (detect) uses PySceneDetect + PaddleOCR to produce `raw_detections.json`. Stage 1.5 (deduplicate) merges spatial/temporal duplicates into `detections.json`. Stage 2 (translate) sends a single batch request to Claude and produces `translations.json`. Stage 3 (render) composites PIL-drawn overlay clips via MoviePy.

**Tech Stack:** PaddleOCR ≥2.7, PySceneDetect ≥0.6, MoviePy ≥1.0<2.0, Pillow ≥10.0, anthropic ≥0.25, opencv-python ≥4.8, pytest, Python 3.9+

**Spec:** `docs/superpowers/specs/2026-03-12-korean-to-japanese-video-translator-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Create | All pip dependencies with version constraints |
| `config.py` | Create | All configurable constants; API key loading |
| `models/__init__.py` | Create | Package marker |
| `models/text_overlay.py` | Create | `TextOverlay` dataclass + `from_dict` / `to_dict` |
| `utils/__init__.py` | Create | Package marker |
| `utils/font_scaler.py` | Create | `fit_font_size`, `wrap_text`, `_measure_text` |
| `stages/__init__.py` | Create | Package marker |
| `stages/detect.py` | Create | Scene detection + PaddleOCR → `raw_detections.json` |
| `stages/deduplicate.py` | Create | IoU merge + temporal grouping → `detections.json` |
| `stages/translate.py` | Create | Claude batch translation → `translations.json` |
| `stages/render.py` | Create | MoviePy compositing → `<stem>_ja.mp4` |
| `translate_video.py` | Create | CLI entry point; orchestrates all stages |
| `tests/conftest.py` | Create | Shared pytest fixtures |
| `tests/test_models.py` | Create | TextOverlay unit tests |
| `tests/test_font_scaler.py` | Create | Font scaling unit tests (mocked PIL) |
| `tests/test_deduplicate.py` | Create | IoU + deduplication algorithm unit tests |
| `tests/test_translate.py` | Create | Translation unit tests (mocked Anthropic) |
| `tests/test_render.py` | Create | Render unit tests (mocked MoviePy + PIL) |
| `tests/test_detect.py` | Create | Detect helper function unit tests |
| `tests/test_translate_video.py` | Create | Orchestrator unit tests (mocked stages) |
| `fonts/.gitignore` | Create | Ignore `*.otf` binaries; keeps fonts/ tracked via `.gitkeep` |
| `fonts/.gitkeep` | Create | Track empty fonts/ directory in git |
| `videos-to-be-done/.gitkeep` | Create | Track empty input directory in git |
| `videos-done/.gitkeep` | Create | Track empty output directory in git |
| `checkpoints/.gitkeep` | Create | Track empty checkpoints directory in git |

---

## Chunk 1: Project Scaffold

**Files:** `requirements.txt`, `config.py`, package `__init__.py` files, `tests/conftest.py`

---

### Task 1: Create folder structure and `requirements.txt`

**Files:**
- Create: `requirements.txt`
- Create: `models/__init__.py`
- Create: `utils/__init__.py`
- Create: `stages/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Move input videos to correct folder**

```bash
# Run from video-translator/ root
mv pre_2_1.mp4 videos-to-be-done/ 2>/dev/null || true
mv pre_2_2.mp4 videos-to-be-done/ 2>/dev/null || true
ls videos-to-be-done/
```

Expected: `pre_2_1.mp4` and `pre_2_2.mp4` listed

- [ ] **Step 2: Create `requirements.txt`**

```
paddlepaddle>=2.5
paddleocr>=2.7
scenedetect[opencv]>=0.6
moviepy>=1.0,<2.0
anthropic>=0.25
Pillow>=10.0
numpy>=1.24
opencv-python>=4.8
pytest>=7.0
pytest-mock>=3.0
```

- [ ] **Step 3: Create package `__init__.py` files**

Create empty files at: `models/__init__.py`, `utils/__init__.py`, `stages/__init__.py`, `tests/__init__.py`

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error. Note: PaddleOCR will download the Korean language model (~2–4GB) on first use — not during install.

- [ ] **Step 5: Commit scaffold**

```bash
git add requirements.txt models/__init__.py utils/__init__.py stages/__init__.py tests/__init__.py
git commit -m "feat: scaffold project structure and requirements"
```

---

### Task 2: Create `config.py`

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write `config.py`**

```python
# config.py
import os
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).parent
VIDEOS_INPUT_DIR  = ROOT_DIR / "videos-to-be-done"
VIDEOS_OUTPUT_DIR = ROOT_DIR / "videos-done"
CHECKPOINTS_DIR   = ROOT_DIR / "checkpoints"
FONT_PATH         = ROOT_DIR / "fonts" / "NotoSansJP-Bold.otf"

# ── API Keys ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]  # raises KeyError immediately if unset

# ── OCR / Scene Detection ───────────────────────────────────────────────
SCENE_THRESHOLD          = 27.0
MIN_SCENE_LEN            = 15
OCR_CONFIDENCE_THRESHOLD = 0.6

# ── Deduplication ───────────────────────────────────────────────────────
BBOX_IOU_THRESHOLD    = 0.5
TEXT_SIMILARITY_RATIO = 0.85
TEMPORAL_GAP_SEC      = 0.5

# ── Translation ─────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-opus-4-6"
MAX_TOKENS   = 1024

# ── Overlay Styling ─────────────────────────────────────────────────────
BOX_COLOR   = (0, 0, 0)        # RGB black background
TEXT_COLOR  = (255, 255, 255)  # RGB white text
BOX_PADDING = 4

# ── Font Sizing ─────────────────────────────────────────────────────────
FONT_SIZE_MIN            = 8
FONT_SIZE_MAX            = 72
FONT_SIZE_WRAP_THRESHOLD = 12

# ── Output Encoding ─────────────────────────────────────────────────────
OUTPUT_CODEC       = "libx264"
OUTPUT_AUDIO_CODEC = "aac"
OUTPUT_FPS         = None  # None → match source FPS
```

- [ ] **Step 2: Verify config imports cleanly**

```bash
python -c "import config; print(config.ROOT_DIR)"
```

Expected: prints the `video-translator/` directory path with no errors

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config module with all constants"
```

---

### Task 3: Create `tests/conftest.py`

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
# tests/conftest.py
# Imports are inside fixtures (lazy) so conftest loads before models/ is implemented.
import pytest


@pytest.fixture
def sample_overlay():
    from models.text_overlay import TextOverlay
    return TextOverlay(
        text_ko="촉촉한 수분감",
        text_ja="",
        bbox=(100, 200, 300, 50),
        start_sec=1.0,
        end_sec=4.0,
        confidence=0.95,
    )


@pytest.fixture
def translated_overlay():
    from models.text_overlay import TextOverlay
    return TextOverlay(
        text_ko="촉촉한 수분감",
        text_ja="うるつやもちもち肌",
        bbox=(100, 200, 300, 50),
        start_sec=1.0,
        end_sec=4.0,
        confidence=0.95,
    )


@pytest.fixture
def sample_overlays():
    from models.text_overlay import TextOverlay
    return [
        TextOverlay("텍스트A", "", (10, 10, 200, 40), 0.0, 2.0, 0.9),
        TextOverlay("텍스트B", "", (10, 60, 200, 40), 3.0, 5.0, 0.85),
    ]
```

- [ ] **Step 2: Verify pytest can collect with no errors**

```bash
python -m pytest tests/ --collect-only 2>&1 | head -20
```

Expected: no import errors (zero tests collected is fine — `models/text_overlay.py` is not yet created, but conftest uses lazy imports so collection succeeds)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add pytest conftest with shared fixtures"
```

---

## Chunk 2: Data Model

**Files:** `models/text_overlay.py`, `tests/test_models.py`

---

### Task 4: Implement `TextOverlay` dataclass

**Files:**
- Create: `models/text_overlay.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import json
import pytest
from models.text_overlay import TextOverlay


def test_bbox_is_tuple(sample_overlay):
    assert isinstance(sample_overlay.bbox, tuple)


def test_from_dict_restores_bbox_as_tuple():
    d = {
        "text_ko": "안녕",
        "text_ja": "",
        "bbox": [10, 20, 100, 30],  # JSON gives list
        "start_sec": 0.0,
        "end_sec": 2.0,
        "confidence": 0.9,
    }
    overlay = TextOverlay.from_dict(d)
    assert isinstance(overlay.bbox, tuple)
    assert overlay.bbox == (10, 20, 100, 30)


def test_round_trip_preserves_all_fields(sample_overlay):
    restored = TextOverlay.from_dict(sample_overlay.to_dict())
    assert restored.text_ko == sample_overlay.text_ko
    assert restored.text_ja == sample_overlay.text_ja
    assert restored.bbox == sample_overlay.bbox
    assert restored.start_sec == sample_overlay.start_sec
    assert restored.end_sec == sample_overlay.end_sec
    assert restored.confidence == sample_overlay.confidence


def test_json_serialization_round_trip(sample_overlay):
    json_str = json.dumps(sample_overlay.to_dict())
    restored = TextOverlay.from_dict(json.loads(json_str))
    assert restored.bbox == sample_overlay.bbox
    assert isinstance(restored.bbox, tuple)


def test_text_ja_defaults_to_empty():
    o = TextOverlay("한국어", "", (0, 0, 100, 30), 0.0, 1.0, 0.8)
    assert o.text_ja == ""
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_models.py -v
```

Expected: `ImportError` — `TextOverlay` not yet implemented

- [ ] **Step 3: Implement `models/text_overlay.py`**

```python
# models/text_overlay.py
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class TextOverlay:
    text_ko: str
    text_ja: str
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    start_sec: float
    end_sec: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TextOverlay:
        return cls(
            text_ko=d["text_ko"],
            text_ja=d["text_ja"],
            bbox=tuple(int(v) for v in d["bbox"]),  # JSON array → tuple[int,...]
            start_sec=float(d["start_sec"]),
            end_sec=float(d["end_sec"]),
            confidence=float(d["confidence"]),
        )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_models.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/text_overlay.py tests/test_models.py
git commit -m "feat: implement TextOverlay dataclass with JSON round-trip"
```

---

## Chunk 3: Font Scaler Utility

**Files:** `utils/font_scaler.py`, `tests/test_font_scaler.py`

---

### Task 5: Implement `utils/font_scaler.py`

**Files:**
- Create: `utils/font_scaler.py`
- Test: `tests/test_font_scaler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_font_scaler.py
from unittest.mock import MagicMock, patch
import pytest
import config
from utils.font_scaler import fit_font_size, wrap_text


@patch("utils.font_scaler.ImageFont.truetype")
@patch("utils.font_scaler._measure_text")
def test_fit_font_size_returns_largest_fitting(mock_measure, mock_truetype):
    """Returns FONT_SIZE_MAX when text fits at that size."""
    mock_measure.return_value = (50, 20)
    mock_font = MagicMock()
    mock_truetype.return_value = mock_font

    size, font = fit_font_size("テスト", bbox_w=200, bbox_h=60, font_path="fake.otf", padding=4)

    assert size == config.FONT_SIZE_MAX
    assert font is mock_font


@patch("utils.font_scaler.ImageFont.truetype")
@patch("utils.font_scaler._measure_text")
def test_fit_font_size_shrinks_until_text_fits(mock_measure, mock_truetype):
    """Iterates down until text fits; returns that size."""
    mock_font = MagicMock()
    mock_truetype.return_value = mock_font
    call_count = [0]

    def measure_side_effect(text, font):
        call_count[0] += 1
        return (9999, 9999) if call_count[0] <= 5 else (50, 20)

    mock_measure.side_effect = measure_side_effect

    size, _ = fit_font_size("テスト", bbox_w=200, bbox_h=60, font_path="fake.otf", padding=4)

    assert size == config.FONT_SIZE_MAX - 5


@patch("utils.font_scaler.ImageFont.truetype")
@patch("utils.font_scaler._measure_text")
def test_fit_font_size_returns_min_when_nothing_fits(mock_measure, mock_truetype):
    """Returns FONT_SIZE_MIN as last resort even if it overflows."""
    mock_measure.return_value = (9999, 9999)
    mock_truetype.return_value = MagicMock()

    size, _ = fit_font_size("テスト", bbox_w=10, bbox_h=10, font_path="fake.otf", padding=4)

    assert size == config.FONT_SIZE_MIN


@patch("utils.font_scaler.fit_font_size")
def test_wrap_text_returns_single_line_when_fits(mock_fit):
    """No wrapping when single-line font size >= FONT_SIZE_WRAP_THRESHOLD."""
    mock_font = MagicMock()
    mock_fit.return_value = (config.FONT_SIZE_WRAP_THRESHOLD + 2, mock_font)

    text, size, _ = wrap_text("テスト", 200, 50, "fake.otf", 4)

    assert text == "テスト"
    assert size == config.FONT_SIZE_WRAP_THRESHOLD + 2
    mock_fit.assert_called_once()


@patch("utils.font_scaler.fit_font_size")
def test_wrap_text_wraps_at_midpoint_when_small(mock_fit):
    """Splits at character midpoint when single-line size is below threshold."""
    mock_font = MagicMock()
    mock_fit.side_effect = [
        (config.FONT_SIZE_WRAP_THRESHOLD - 2, mock_font),  # single-line: too small
        (config.FONT_SIZE_WRAP_THRESHOLD + 1, mock_font),  # 2-line: bigger
    ]

    result_text, size, _ = wrap_text("テストテスト", 200, 50, "fake.otf", 4)

    assert "\n" in result_text
    assert result_text == "テスト\nテスト"
    assert size == config.FONT_SIZE_WRAP_THRESHOLD + 1


@patch("utils.font_scaler.fit_font_size")
def test_wrap_text_returns_original_when_wrap_is_worse(mock_fit):
    """Returns original text if wrapping produces a smaller font size."""
    mock_font = MagicMock()
    single_size = config.FONT_SIZE_WRAP_THRESHOLD - 1
    mock_fit.side_effect = [
        (single_size, mock_font),
        (single_size - 2, mock_font),
    ]

    result_text, size, _ = wrap_text("テストテスト", 50, 30, "fake.otf", 4)

    assert result_text == "テストテスト"
    assert size == single_size


@patch("utils.font_scaler.fit_font_size")
def test_wrap_text_returns_original_on_equal_size_tie(mock_fit):
    """Returns original (no newline) when wrap produces same font size as single-line."""
    mock_font = MagicMock()
    same_size = config.FONT_SIZE_WRAP_THRESHOLD - 1
    mock_fit.side_effect = [
        (same_size, mock_font),  # single-line
        (same_size, mock_font),  # 2-line tie — should NOT prefer wrapped
    ]

    result_text, size, _ = wrap_text("テストテスト", 100, 40, "fake.otf", 4)

    assert "\n" not in result_text   # original returned, no wrap
    assert size == same_size


@patch("utils.font_scaler.fit_font_size")
def test_wrap_text_single_char_no_wrap(mock_fit):
    """Single-character text is never wrapped (mid=0 would produce leading newline)."""
    mock_font = MagicMock()
    mock_fit.return_value = (config.FONT_SIZE_WRAP_THRESHOLD - 2, mock_font)

    result_text, _, _ = wrap_text("テ", 50, 30, "fake.otf", 4)

    assert result_text == "テ"
    assert "\n" not in result_text
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_font_scaler.py -v
```

Expected: `ImportError` — `font_scaler` not yet implemented

- [ ] **Step 3: Implement `utils/font_scaler.py`**

```python
# utils/font_scaler.py
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config


def _measure_text(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Measure bounding box of text string, handling multiline via ImageDraw.textbbox.

    Uses the full multi-line string — not per-line — so combined height including
    line spacing is correctly captured.
    """
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def fit_font_size(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: str | Path,
    padding: int,
) -> tuple[int, ImageFont.FreeTypeFont]:
    """Return (size, font) — the largest size where text fits within bbox minus padding.

    'font' is a PIL.ImageFont.FreeTypeFont (truetype) instance.
    Falls back to FONT_SIZE_MIN if nothing fits.
    """
    usable_w = bbox_w - padding * 2
    usable_h = bbox_h - padding * 2

    for size in range(config.FONT_SIZE_MAX, config.FONT_SIZE_MIN - 1, -1):
        font = ImageFont.truetype(str(font_path), size)
        text_w, text_h = _measure_text(text, font)
        if text_w <= usable_w and text_h <= usable_h:
            return size, font

    font = ImageFont.truetype(str(font_path), config.FONT_SIZE_MIN)
    return config.FONT_SIZE_MIN, font


def wrap_text(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: str | Path,
    padding: int,
) -> tuple[str, int, ImageFont.FreeTypeFont]:
    """Return (wrapped_text, size, font) fitting within bbox.

    If single-line size >= FONT_SIZE_WRAP_THRESHOLD, returns text unchanged.
    Otherwise tries a 2-line split at character midpoint.

    IMPORTANT: callers must render the returned wrapped_text, not the original
    input — it may contain a newline character.
    """
    single_size, single_font = fit_font_size(text, bbox_w, bbox_h, font_path, padding)

    if single_size >= config.FONT_SIZE_WRAP_THRESHOLD:
        return text, single_size, single_font

    # Guard: single-char text — mid=0 would produce a leading newline
    if len(text) <= 1:
        return text, single_size, single_font

    mid = len(text) // 2
    wrapped = text[:mid] + "\n" + text[mid:]
    wrap_size, wrap_font = fit_font_size(wrapped, bbox_w, bbox_h, font_path, padding)

    # Strictly greater: on tie, prefer single-line (no unnecessary newline)
    if wrap_size > single_size:
        return wrapped, wrap_size, wrap_font

    return text, single_size, single_font
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_font_scaler.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/font_scaler.py tests/test_font_scaler.py
git commit -m "feat: implement font scaler utility with wrap_text"
```

---

## Chunk 4: Stage 1 — Detection

**Files:** `stages/detect.py`, `tests/test_detect.py`

---

### Task 6: Implement detection helper functions and `stages/detect.py`

**Files:**
- Create: `stages/detect.py`
- Test: `tests/test_detect.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_detect.py
import pytest
from stages.detect import polygon_to_rect, contains_hangul, filter_detections
from models.text_overlay import TextOverlay


# ── polygon_to_rect ──────────────────────────────────────────────────────

def test_polygon_to_rect_axis_aligned():
    pts = [[10, 20], [110, 20], [110, 70], [10, 70]]
    assert polygon_to_rect(pts) == (10, 20, 100, 50)


def test_polygon_to_rect_rotated_polygon():
    pts = [[15, 20], [110, 18], [112, 72], [8, 74]]
    x, y, w, h = polygon_to_rect(pts)
    assert x == 8
    assert y == 18
    assert w == 112 - 8
    assert h == 74 - 18


# ── contains_hangul ──────────────────────────────────────────────────────

def test_hangul_syllables_detected():
    assert contains_hangul("촉촉한") is True


def test_hangul_jamo_detected():
    assert contains_hangul("\u1100\u1101") is True


def test_hangul_compatibility_jamo_detected():
    assert contains_hangul("\u3131\u3132") is True


def test_latin_text_rejected():
    assert contains_hangul("Hello World") is False


def test_mixed_text_with_hangul_passes():
    assert contains_hangul("ABC 촉촉한 123") is True


def test_empty_string_rejected():
    assert contains_hangul("") is False


# ── filter_detections ────────────────────────────────────────────────────

def test_filter_drops_low_confidence():
    o = TextOverlay("촉촉", "", (0, 0, 100, 30), 0.0, 1.0, 0.5)
    assert filter_detections([o], confidence_threshold=0.6) == []


def test_filter_keeps_sufficient_confidence():
    o = TextOverlay("촉촉", "", (0, 0, 100, 30), 0.0, 1.0, 0.7)
    assert len(filter_detections([o], confidence_threshold=0.6)) == 1


def test_filter_drops_non_korean():
    o = TextOverlay("Hello", "", (0, 0, 100, 30), 0.0, 1.0, 0.9)
    assert filter_detections([o], confidence_threshold=0.6) == []


def test_filter_keeps_mixed_korean():
    o = TextOverlay("ABC 안녕", "", (0, 0, 100, 30), 0.0, 1.0, 0.9)
    assert len(filter_detections([o], confidence_threshold=0.6)) == 1
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_detect.py -v
```

Expected: `ImportError` — `detect.py` not yet implemented

- [ ] **Step 3: Implement `stages/detect.py`**

```python
# stages/detect.py
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from paddleocr import PaddleOCR

import config
from models.text_overlay import TextOverlay

logger = logging.getLogger(__name__)


def polygon_to_rect(polygon_pts: list[list[float]]) -> tuple[int, int, int, int]:
    """Convert PaddleOCR 4-point polygon to axis-aligned (x, y, w, h) bounding rect."""
    xs = [pt[0] for pt in polygon_pts]
    ys = [pt[1] for pt in polygon_pts]
    x = int(min(xs))
    y = int(min(ys))
    w = int(max(xs)) - x
    h = int(max(ys)) - y
    return x, y, w, h


def contains_hangul(text: str) -> bool:
    """Return True if text contains at least one Hangul character (any Unicode block)."""
    for ch in text:
        cp = ord(ch)
        if (
            0xAC00 <= cp <= 0xD7A3   # Hangul Syllables
            or 0x1100 <= cp <= 0x11FF  # Hangul Jamo
            or 0x3130 <= cp <= 0x318F  # Hangul Compatibility Jamo
        ):
            return True
    return False


def filter_detections(
    overlays: list[TextOverlay],
    confidence_threshold: float,
) -> list[TextOverlay]:
    """Drop overlays below confidence threshold or containing no Hangul."""
    return [
        o for o in overlays
        if o.confidence >= confidence_threshold and contains_hangul(o.text_ko)
    ]


def _extract_keyframes(video_path: Path) -> list[tuple[float, float, Any]]:
    """Run PySceneDetect; return (start_sec, end_sec, frame_ndarray) per scene."""
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=config.SCENE_THRESHOLD,
            min_scene_len=config.MIN_SCENE_LEN,
        )
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    results = []

    for start_tc, end_tc in scene_list:
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        mid_frame = int(((start_sec + end_sec) / 2) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret:
            results.append((start_sec, end_sec, frame))

    cap.release()
    return results


def run(video_path: Path, checkpoint_dir: Path) -> list[TextOverlay]:
    """Stage 1: detect Korean text. Writes raw_detections.json."""
    ocr = PaddleOCR(use_angle_cls=True, lang="korean", show_log=False)
    keyframes = _extract_keyframes(video_path)
    overlays: list[TextOverlay] = []

    for start_sec, end_sec, frame in keyframes:
        results = ocr.ocr(frame, cls=True)
        if not results or not results[0]:
            continue
        for line in results[0]:
            polygon_pts, (text, confidence) = line
            overlay = TextOverlay(
                text_ko=text,
                text_ja="",
                bbox=polygon_to_rect(polygon_pts),
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=float(confidence),
            )
            overlays.append(overlay)

    filtered = filter_detections(overlays, config.OCR_CONFIDENCE_THRESHOLD)
    out_path = checkpoint_dir / "raw_detections.json"
    out_path.write_text(
        json.dumps([o.to_dict() for o in filtered], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Stage 1: wrote %d detections to %s", len(filtered), out_path)
    return filtered
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_detect.py -v
```

Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add stages/detect.py tests/test_detect.py
git commit -m "feat: implement detect stage with OCR and Hangul filter"
```

---

## Chunk 5: Stage 1.5 — Deduplication

**Files:** `stages/deduplicate.py`, `tests/test_deduplicate.py`

---

### Task 7: Implement deduplication algorithm

**Files:**
- Create: `stages/deduplicate.py`
- Test: `tests/test_deduplicate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_deduplicate.py
import json
import pytest
from models.text_overlay import TextOverlay
from stages.deduplicate import compute_iou, pass1_spatial_merge, pass2_temporal_group, run


def _o(text, bbox, start, end, conf=0.9):
    return TextOverlay(text, "", bbox, start, end, conf)


# ── compute_iou ──────────────────────────────────────────────────────────

def test_iou_identical_boxes():
    assert compute_iou((0, 0, 100, 50), (0, 0, 100, 50)) == pytest.approx(1.0)


def test_iou_non_overlapping():
    assert compute_iou((0, 0, 50, 50), (100, 0, 50, 50)) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # A=(0,0,100,100) B=(50,0,100,100): intersection=50*100=5000, union=15000
    assert compute_iou((0, 0, 100, 100), (50, 0, 100, 100)) == pytest.approx(5000 / 15000)


def test_iou_zero_area_returns_zero():
    assert compute_iou((0, 0, 0, 0), (0, 0, 100, 50)) == pytest.approx(0.0)


# ── pass1_spatial_merge ──────────────────────────────────────────────────

def test_pass1_merges_identical_consecutive():
    a = _o("텍스트", (10, 10, 200, 40), 0.0, 1.0)
    b = _o("텍스트", (10, 10, 200, 40), 1.0, 2.0)
    result = pass1_spatial_merge([a, b])
    assert len(result) == 1
    assert result[0].end_sec == pytest.approx(2.0)


def test_pass1_stops_at_chain_break():
    a = _o("텍스트A", (10, 10, 200, 40), 0.0, 1.0)
    b = _o("다른텍스트", (500, 500, 50, 20), 1.0, 2.0)  # fails both thresholds
    c = _o("텍스트A", (10, 10, 200, 40), 2.0, 3.0)
    result = pass1_spatial_merge([a, b, c])
    assert len(result) == 3  # c not merged with a because chain broke at b


def test_pass1_extends_end_sec_across_chain():
    a = _o("같은", (0, 0, 100, 30), 0.0, 1.0)
    b = _o("같은", (0, 0, 100, 30), 1.0, 2.0)
    c = _o("같은", (0, 0, 100, 30), 2.0, 3.5)
    result = pass1_spatial_merge([a, b, c])
    assert len(result) == 1
    assert result[0].end_sec == pytest.approx(3.5)


def test_pass1_preserves_unrelated_entries():
    a = _o("A", (0, 0, 100, 30), 0.0, 1.0)
    b = _o("B", (300, 300, 100, 30), 1.0, 2.0)
    assert len(pass1_spatial_merge([a, b])) == 2


def test_pass1_skips_removed_continues_chain():
    """Removed (already-merged) entries are skipped in the forward scan.
    This is correct: 'do not skip non-matching' ≠ 'do not skip already-merged'.
    """
    a = _o("같은", (0, 0, 100, 30), 0.0, 1.0)
    b = _o("같은", (0, 0, 100, 30), 1.0, 2.0)   # merges into a
    c = _o("같은", (0, 0, 100, 30), 2.0, 3.0)   # should also merge into a (b is removed)
    result = pass1_spatial_merge([a, b, c])
    assert len(result) == 1
    assert result[0].end_sec == pytest.approx(3.0)


def test_pass1_chain_breaks_on_iou_failure_alone():
    """Chain breaks when IoU fails even if text is similar."""
    a = _o("같은", (0, 0, 100, 30), 0.0, 1.0)
    b = _o("같은", (500, 500, 100, 30), 1.0, 2.0)   # IoU = 0, text similar
    c = _o("같은", (0, 0, 100, 30), 2.0, 3.0)        # same bbox as a, but chain broke
    result = pass1_spatial_merge([a, b, c])
    assert len(result) == 3   # chain broke at b; c not merged with a


def test_pass1_chain_breaks_on_text_failure_alone():
    """Chain breaks when text dissimilar even if IoU passes."""
    a = _o("텍스트AAA", (0, 0, 100, 30), 0.0, 1.0)
    b = _o("완전달라요", (0, 0, 100, 30), 1.0, 2.0)   # same bbox, different text
    result = pass1_spatial_merge([a, b])
    assert len(result) == 2


# ── pass2_temporal_group ─────────────────────────────────────────────────

def test_pass2_merges_adjacent_temporal():
    a = _o("같은", (0, 0, 100, 30), 0.0, 1.0, conf=0.8)
    b = _o("같은", (5, 5, 100, 30), 1.3, 2.5, conf=0.95)  # gap 0.3 < 0.5
    result = pass2_temporal_group([a, b])
    assert len(result) == 1
    assert result[0].start_sec == pytest.approx(0.0)
    assert result[0].end_sec == pytest.approx(2.5)


def test_pass2_uses_highest_confidence_bbox():
    a = _o("같은", (0, 0, 100, 30), 0.0, 1.0, conf=0.8)
    b = _o("같은", (5, 5, 110, 35), 1.3, 2.5, conf=0.95)
    result = pass2_temporal_group([a, b])
    assert result[0].bbox == (5, 5, 110, 35)


def test_pass2_does_not_merge_large_gap():
    a = _o("같은", (0, 0, 100, 30), 0.0, 1.0)
    b = _o("같은", (0, 0, 100, 30), 2.0, 3.0)  # gap 1.0 > 0.5
    assert len(pass2_temporal_group([a, b])) == 2


def test_pass2_sorts_output_by_start_sec():
    a = _o("B", (100, 0, 50, 30), 3.0, 4.0)
    b = _o("A", (0, 0, 50, 30), 0.0, 1.0)
    result = pass2_temporal_group([a, b])
    assert result[0].start_sec < result[1].start_sec


def test_run_empty_list_produces_empty_json(tmp_path):
    run([], tmp_path)
    out = json.loads((tmp_path / "detections.json").read_text())
    assert out == []
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_deduplicate.py -v
```

Expected: `ImportError` — `deduplicate.py` not yet implemented

- [ ] **Step 3: Implement `stages/deduplicate.py`**

```python
# stages/deduplicate.py
from __future__ import annotations
import difflib
import json
import logging
from pathlib import Path

import config
from models.text_overlay import TextOverlay

logger = logging.getLogger(__name__)


def compute_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection over Union for two (x, y, w, h) bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    # Both dimension clamps applied independently before multiplying
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _text_similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() > config.TEXT_SIMILARITY_RATIO


def pass1_spatial_merge(entries: list[TextOverlay]) -> list[TextOverlay]:
    """Greedy chain merge: scan forward while IoU and text similarity both pass.
    Chain stops on first *non-matching* entry — already-removed (merged) entries
    are skipped transparently, as they are no longer candidates.
    """
    sorted_entries = sorted(entries, key=lambda o: o.start_sec)
    removed: set[int] = set()

    for i in range(len(sorted_entries)):
        if i in removed:
            continue
        j = i + 1
        while j < len(sorted_entries):
            if j in removed:
                # Skip already-merged entries — NOT a chain break
                j += 1
                continue
            a = sorted_entries[i]
            b = sorted_entries[j]
            if (
                compute_iou(a.bbox, b.bbox) > config.BBOX_IOU_THRESHOLD
                and _text_similar(a.text_ko, b.text_ko)
            ):
                sorted_entries[i] = TextOverlay(
                    text_ko=a.text_ko,
                    text_ja=a.text_ja,
                    bbox=a.bbox,
                    start_sec=a.start_sec,
                    end_sec=max(a.end_sec, b.end_sec),
                    confidence=max(a.confidence, b.confidence),
                )
                removed.add(j)
                j += 1
            else:
                break  # chain breaks — do not skip

    return [e for i, e in enumerate(sorted_entries) if i not in removed]


def pass2_temporal_group(entries: list[TextOverlay]) -> list[TextOverlay]:
    """Group by text_ko; merge temporally adjacent entries within TEMPORAL_GAP_SEC.
    Uses highest-confidence bbox for the merged entry.
    """
    groups: dict[str, list[TextOverlay]] = {}
    for o in entries:
        groups.setdefault(o.text_ko.strip(), []).append(o)

    merged: list[TextOverlay] = []
    for group in groups.values():
        group_sorted = sorted(group, key=lambda o: o.start_sec)
        current = group_sorted[0]

        for nxt in group_sorted[1:]:
            if nxt.start_sec - current.end_sec < config.TEMPORAL_GAP_SEC:
                best_bbox = nxt.bbox if nxt.confidence >= current.confidence else current.bbox
                current = TextOverlay(
                    text_ko=current.text_ko,
                    text_ja=current.text_ja,
                    bbox=best_bbox,
                    start_sec=min(current.start_sec, nxt.start_sec),
                    end_sec=max(current.end_sec, nxt.end_sec),
                    confidence=max(current.confidence, nxt.confidence),
                )
            else:
                merged.append(current)
                current = nxt

        merged.append(current)

    return sorted(merged, key=lambda o: o.start_sec)


def run(detections: list[TextOverlay], checkpoint_dir: Path) -> list[TextOverlay]:
    """Stage 1.5: deduplicate detections. Writes detections.json."""
    after_pass1 = pass1_spatial_merge(detections)
    after_pass2 = pass2_temporal_group(after_pass1)

    out_path = checkpoint_dir / "detections.json"
    out_path.write_text(
        json.dumps([o.to_dict() for o in after_pass2], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Stage 1.5: %d → %d overlays after dedup, wrote %s",
        len(detections), len(after_pass2), out_path,
    )
    return after_pass2
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_deduplicate.py -v
```

Expected: all 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add stages/deduplicate.py tests/test_deduplicate.py
git commit -m "feat: implement deduplication stage with IoU merge and temporal grouping"
```

---

## Chunk 6: Stage 2 — Translation

**Files:** `stages/translate.py`, `tests/test_translate.py`

---

### Task 8: Implement Claude batch translation

**Files:**
- Create: `stages/translate.py`
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_translate.py
import json
import pytest
from unittest.mock import MagicMock, patch
from models.text_overlay import TextOverlay
from stages.translate import build_batch_payload, parse_response, translate_batch


def _o(text_ko, text_ja=""):
    return TextOverlay(text_ko, text_ja, (0, 0, 100, 30), 0.0, 1.0, 0.9)


# ── build_batch_payload ──────────────────────────────────────────────────

def test_build_payload_deduplicates_keys():
    overlays = [_o("안녕"), _o("안녕"), _o("다른텍스트")]
    payload = build_batch_payload(overlays)
    assert list(payload.keys()) == ["안녕", "다른텍스트"]


def test_build_payload_values_are_empty_strings():
    payload = build_batch_payload([_o("안녕")])
    assert payload["안녕"] == ""


def test_build_payload_empty_list():
    assert build_batch_payload([]) == {}


# ── parse_response ───────────────────────────────────────────────────────

def test_parse_response_maps_translations():
    overlays = [_o("안녕"), _o("반가워요")]
    result = parse_response(overlays, {"안녕": "こんにちは", "반가워요": "よろしく"})
    assert result[0].text_ja == "こんにちは"
    assert result[1].text_ja == "よろしく"


def test_parse_response_leaves_missing_keys_empty():
    overlays = [_o("안녕"), _o("누락됨")]
    result = parse_response(overlays, {"안녕": "こんにちは"})
    assert result[1].text_ja == ""


def test_parse_response_maps_duplicate_overlays():
    overlays = [_o("안녕"), _o("안녕")]
    result = parse_response(overlays, {"안녕": "こんにちは"})
    assert result[0].text_ja == result[1].text_ja == "こんにちは"


# ── translate_batch ──────────────────────────────────────────────────────

def test_translate_batch_returns_translations():
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [
        MagicMock(text=json.dumps({"안녕": "こんにちは"}))
    ]
    result = translate_batch([_o("안녕")], mock_client)
    assert result[0].text_ja == "こんにちは"


@patch("stages.translate.time.sleep")
def test_translate_batch_retries_on_bad_json(mock_sleep):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text="garbage")]),
        MagicMock(content=[MagicMock(text="garbage")]),
        MagicMock(content=[MagicMock(text=json.dumps({"안녕": "こんにちは"}))]),
    ]
    result = translate_batch([_o("안녕")], mock_client)
    assert mock_client.messages.create.call_count == 3   # exactly 3 total attempts
    assert result[0].text_ja == "こんにちは"
    assert mock_sleep.call_count == 2   # sleeps before attempt 2 and 3


@patch("stages.translate.time.sleep")
def test_translate_batch_returns_empty_after_all_retries_fail(mock_sleep):
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="always garbage")]
    # Start with a non-empty text_ja to prove it gets reset to "" on failure
    overlay = TextOverlay("안녕", "old_value", (0, 0, 100, 30), 0.0, 1.0, 0.9)
    result = translate_batch([overlay], mock_client)
    # All 3 attempts fail → _call_claude returns None → overlays returned unchanged
    # text_ja was "old_value" going in; the unmodified original is returned
    assert result[0].text_ja == "old_value"


@patch("stages.translate.time.sleep")
def test_translate_batch_fallback_per_string_on_missing_key(mock_sleep):
    """Batch returns partial result; individual fallback fills the missing key."""
    mock_client = MagicMock()
    # Batch call returns translation for only one of two strings
    batch_response = json.dumps({"안녕": "こんにちは"})
    # Individual fallback for missing "반가워요"
    fallback_response = json.dumps({"반가워요": "よろしく"})
    mock_client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text=batch_response)]),
        MagicMock(content=[MagicMock(text=fallback_response)]),
    ]
    overlays = [_o("안녕"), _o("반가워요")]
    result = translate_batch(overlays, mock_client)
    assert result[0].text_ja == "こんにちは"
    assert result[1].text_ja == "よろしく"
    assert mock_client.messages.create.call_count == 2   # 1 batch + 1 individual


def test_translate_batch_empty_input():
    assert translate_batch([], MagicMock()) == []
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_translate.py -v
```

Expected: `ImportError` — `translate.py` not yet implemented

- [ ] **Step 3: Implement `stages/translate.py`**

```python
# stages/translate.py
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import anthropic

import config
from models.text_overlay import TextOverlay

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a professional Japanese copywriter specialising in beauty and cosmetics "
    "advertising targeting Japanese girls aged 18–25. Translate each Korean phrase into "
    "natural, trendy Japanese suitable for a beauty ad campaign. Actively use contemporary "
    "beauty slang popular among late-teen to mid-twenties Japanese women — words like "
    "'うるつや', 'ぷるぷる', 'もちもち', 'つるつる', 'バズり', 'エモい', 'ガチ', "
    "'めちゃ盛れ', 'スキンケア沼' etc. where contextually appropriate. Keep translations "
    "concise — they must fit within the same space as the original text. "
    "Respond ONLY with a JSON object mapping each Korean input to its Japanese translation. "
    "No explanations."
)

_RETRY_DELAYS = [2, 4]  # 3 total attempts: 1 initial + 2 retries (spec: "up to 3 attempts")


def build_batch_payload(overlays: list[TextOverlay]) -> dict[str, str]:
    """Return {text_ko: ""} for all unique Korean strings, preserving insertion order."""
    seen: dict[str, str] = {}
    for o in overlays:
        if o.text_ko not in seen:
            seen[o.text_ko] = ""
    return seen


def parse_response(
    overlays: list[TextOverlay],
    translations: dict[str, str],
) -> list[TextOverlay]:
    """Map translation dict back onto overlays. Missing keys leave text_ja=""."""
    return [
        TextOverlay(
            text_ko=o.text_ko,
            text_ja=translations.get(o.text_ko, ""),
            bbox=o.bbox,
            start_sec=o.start_sec,
            end_sec=o.end_sec,
            confidence=o.confidence,
        )
        for o in overlays
    ]


def _call_claude(
    client: anthropic.Anthropic,
    payload: dict[str, str],
) -> dict[str, str] | None:
    """Up to 3 total attempts (1 initial + 2 retries). Returns parsed dict or None."""
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):  # [0, 2, 4] → 3 iterations
        if delay:
            time.sleep(delay)
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                }],
            )
            result = json.loads(response.content[0].text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Claude response parse failed attempt %d: %s", attempt + 1, exc)
        except anthropic.APIError as exc:
            logger.warning("Claude API error attempt %d: %s", attempt + 1, exc)
    return None


def translate_batch(
    overlays: list[TextOverlay],
    client: anthropic.Anthropic,
) -> list[TextOverlay]:
    """Translate all overlays in one batch; fall back per-string for missing keys."""
    if not overlays:
        return []

    payload = build_batch_payload(overlays)
    result = _call_claude(client, payload)

    if result is None:
        logger.error("All batch retries failed — all text_ja will be empty")
        return overlays

    # Individual fallback for any keys missing or empty in batch response
    missing = {k for k in payload if not result.get(k)}
    for text_ko in missing:
        individual = _call_claude(client, {text_ko: ""})
        if individual and individual.get(text_ko):
            result[text_ko] = individual[text_ko]
        else:
            logger.warning("Individual translation failed for: %s", text_ko)

    return parse_response(overlays, result)


def run(detections: list[TextOverlay], checkpoint_dir: Path) -> list[TextOverlay]:
    """Stage 2: translate Korean text via Claude. Writes translations.json."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    translated = translate_batch(detections, client)

    out_path = checkpoint_dir / "translations.json"
    out_path.write_text(
        json.dumps([o.to_dict() for o in translated], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Stage 2: wrote %d translations to %s", len(translated), out_path)
    return translated
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_translate.py -v
```

Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add stages/translate.py tests/test_translate.py
git commit -m "feat: implement translate stage with Claude batch API and retry"
```

---

## Chunk 7: Stage 3 — Render

**Files:** `stages/render.py`, `tests/test_render.py`

---

### Task 9: Implement MoviePy render stage

**Files:**
- Create: `stages/render.py`
- Test: `tests/test_render.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_render.py
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

import config
from models.text_overlay import TextOverlay
from stages.render import create_overlay_image, build_overlay_clip, run


def _o(text_ja, bbox=(100, 200, 300, 50)):
    return TextOverlay("한국어", text_ja, bbox, 1.0, 4.0, 0.9)


# ── create_overlay_image ─────────────────────────────────────────────────

@patch("stages.render.ImageDraw.Draw")
@patch("stages.render.wrap_text")
def test_overlay_image_correct_size(mock_wrap, mock_draw_cls):
    mock_wrap.return_value = ("テスト", 16, MagicMock())
    mock_draw = MagicMock()
    mock_draw.textbbox.return_value = (0, 0, 80, 20)
    mock_draw_cls.return_value = mock_draw
    img = create_overlay_image(_o("テスト", bbox=(100, 200, 300, 50)))
    assert img.size == (300, 50)


@patch("stages.render.ImageDraw.Draw")
@patch("stages.render.wrap_text")
def test_overlay_image_background_is_box_color(mock_wrap, mock_draw_cls):
    mock_wrap.return_value = ("テスト", 16, MagicMock())
    mock_draw = MagicMock()
    mock_draw.textbbox.return_value = (0, 0, 80, 20)
    mock_draw_cls.return_value = mock_draw
    img = create_overlay_image(_o("テスト", bbox=(0, 0, 200, 60)))
    # Image is RGBA — check first 3 channels only
    assert img.getpixel((0, 0))[:3] == config.BOX_COLOR
    assert img.mode == "RGBA"


# ── build_overlay_clip ───────────────────────────────────────────────────

@patch("stages.render.create_overlay_image")
def test_build_clip_skips_empty_text_ja(mock_create):
    result = build_overlay_clip(_o(""))
    mock_create.assert_not_called()
    assert result is None


@patch("stages.render.CompositeVideoClip")
@patch("stages.render.VideoFileClip")
def test_run_composites_and_writes(mock_vfc_cls, mock_composite_cls):
    """run() loads video, composites overlays, writes output with correct codec args."""
    mock_video = MagicMock()
    mock_video.fps = 30.0
    mock_vfc_cls.return_value = mock_video
    mock_composite = MagicMock()
    mock_composite_cls.return_value = mock_composite

    overlay = TextOverlay("한국어", "テスト", (0, 0, 100, 30), 1.0, 3.0, 0.9)

    with patch("stages.render.build_overlay_clip") as mock_build:
        mock_clip = MagicMock()
        mock_build.return_value = mock_clip
        run(Path("input.mp4"), [overlay], Path("output.mp4"))

    mock_composite.write_videofile.assert_called_once()
    args, kwargs = mock_composite.write_videofile.call_args
    assert args[0] == str(Path("output.mp4"))
    assert kwargs["codec"] == config.OUTPUT_CODEC
    assert kwargs["audio_codec"] == config.OUTPUT_AUDIO_CODEC
    assert kwargs["fps"] == mock_video.fps
    mock_composite.close.assert_called_once()
    mock_video.close.assert_called_once()


@patch("stages.render.CompositeVideoClip")
@patch("stages.render.VideoFileClip")
def test_run_still_writes_when_all_clips_empty(mock_vfc_cls, mock_composite_cls):
    """run() still calls write_videofile even when all overlays have empty text_ja."""
    mock_video = MagicMock()
    mock_video.fps = 30.0
    mock_vfc_cls.return_value = mock_video
    mock_composite = MagicMock()
    mock_composite_cls.return_value = mock_composite

    overlay = TextOverlay("한국어", "", (0, 0, 100, 30), 1.0, 3.0, 0.9)
    run(Path("input.mp4"), [overlay], Path("output.mp4"))

    mock_composite.write_videofile.assert_called_once()


@patch("stages.render.ImageClip")
@patch("stages.render.create_overlay_image")
def test_build_clip_sets_position_start_end(mock_create, mock_clip_cls):
    mock_img = MagicMock(spec=Image.Image)
    mock_img.size = (300, 50)
    mock_create.return_value = mock_img

    mock_clip = MagicMock()
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_start.return_value = mock_clip
    mock_clip.set_end.return_value = mock_clip
    mock_clip_cls.return_value = mock_clip

    overlay = _o("テスト", bbox=(100, 200, 300, 50))
    result = build_overlay_clip(overlay)

    assert result is not None
    # ImageClip must receive a numpy array derived from the PIL image
    clip_call_arg = mock_clip_cls.call_args[0][0]
    assert isinstance(clip_call_arg, np.ndarray)
    mock_clip.set_position.assert_called_once_with((100, 200))
    mock_clip.set_start.assert_called_once_with(overlay.start_sec)
    mock_clip.set_end.assert_called_once_with(overlay.end_sec)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_render.py -v
```

Expected: `ImportError` — `render.py` not yet implemented

- [ ] **Step 3: Implement `stages/render.py`**

```python
# stages/render.py
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

import config
from models.text_overlay import TextOverlay
from utils.font_scaler import wrap_text

logger = logging.getLogger(__name__)


def create_overlay_image(overlay: TextOverlay) -> Image.Image:
    """Draw a solid BOX_COLOR rectangle with centered Japanese text. Returns RGBA image."""
    _, _, w, h = overlay.bbox
    box_color_rgba = (*config.BOX_COLOR, 255)
    text_color_rgba = (*config.TEXT_COLOR, 255)
    img = Image.new("RGBA", (w, h), box_color_rgba)
    draw = ImageDraw.Draw(img)

    # wrap_text returns (wrapped_text, size, font) — must use wrapped_text for render
    wrapped_text, _, font = wrap_text(
        overlay.text_ja, w, h, config.FONT_PATH, config.BOX_PADDING
    )

    bb = draw.textbbox((0, 0), wrapped_text, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]
    x = (w - text_w) // 2
    y = (h - text_h) // 2

    draw.text((x, y), wrapped_text, font=font, fill=text_color_rgba)
    return img


def build_overlay_clip(overlay: TextOverlay) -> ImageClip | None:
    """Return a timed ImageClip positioned at bbox origin, or None if text_ja is empty."""
    if not overlay.text_ja:
        return None

    x, y, _, _ = overlay.bbox
    img = create_overlay_image(overlay)

    clip = (
        ImageClip(np.array(img))
        .set_position((x, y))
        .set_start(overlay.start_sec)
        .set_end(overlay.end_sec)
    )
    return clip


def run(
    video_path: Path,
    overlays: list[TextOverlay],
    output_path: Path,
) -> None:
    """Stage 3: composite overlays onto video and write H.264 output."""
    original = VideoFileClip(str(video_path))

    active_clips = [c for o in overlays if (c := build_overlay_clip(o)) is not None]

    if not active_clips:
        logger.warning("No overlay clips to render — re-encoding original only")

    composite = CompositeVideoClip([original, *active_clips])
    fps = config.OUTPUT_FPS or original.fps

    composite.write_videofile(
        str(output_path),
        codec=config.OUTPUT_CODEC,
        audio_codec=config.OUTPUT_AUDIO_CODEC,
        fps=fps,
        logger="bar",
    )
    composite.close()
    original.close()
    logger.info("Stage 3: wrote %s", output_path)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_render.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add stages/render.py tests/test_render.py
git commit -m "feat: implement render stage with MoviePy compositing"
```

---

## Chunk 8: Main Entry Point

**Files:** `translate_video.py`, `tests/test_translate_video.py`

---

### Task 10: Implement CLI orchestrator

**Files:**
- Create: `translate_video.py`
- Test: `tests/test_translate_video.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_translate_video.py
from __future__ import annotations
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config


def _make_args(force=False, skip_detect=False, skip_translate=False):
    return argparse.Namespace(
        force=force, skip_detect=skip_detect, skip_translate=skip_translate
    )


class TestStartupCheck:
    def test_exits_when_font_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "FONT_PATH", tmp_path / "missing.otf")
        monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "key")
        from translate_video import _startup_check
        with pytest.raises(SystemExit):
            _startup_check()

    def test_exits_when_api_key_empty(self, tmp_path, monkeypatch):
        font = tmp_path / "font.otf"
        font.touch()
        monkeypatch.setattr(config, "FONT_PATH", font)
        monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
        monkeypatch.setattr(config, "VIDEOS_INPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "VIDEOS_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path)
        from translate_video import _startup_check
        with pytest.raises(SystemExit, match="empty"):
            _startup_check()

    def test_passes_when_font_and_key_present(self, tmp_path, monkeypatch):
        font = tmp_path / "font.otf"
        font.touch()
        monkeypatch.setattr(config, "FONT_PATH", font)
        monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "key")
        monkeypatch.setattr(config, "VIDEOS_INPUT_DIR", tmp_path / "in")
        monkeypatch.setattr(config, "VIDEOS_OUTPUT_DIR", tmp_path / "out")
        monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "ckpt")
        from translate_video import _startup_check
        _startup_check()  # should not raise


class TestProcessVideo:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(config, "VIDEOS_OUTPUT_DIR", tmp_path / "done")
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "done").mkdir()
        video = tmp_path / "test.mp4"
        video.touch()
        return video

    def test_moves_original_when_no_detections(self, tmp_path, monkeypatch):
        video = self._setup(tmp_path, monkeypatch)
        with patch("translate_video.detect.run", return_value=[]), \
             patch("translate_video.deduplicate.run", return_value=[]):
            from translate_video import process_video
            process_video(video, _make_args())
        assert (tmp_path / "done" / "test.mp4").exists()
        assert not video.exists()

    def test_moves_original_when_all_translations_empty(self, tmp_path, monkeypatch):
        from models.text_overlay import TextOverlay
        overlay = TextOverlay("한국어", "", (0, 0, 50, 20), 0.0, 1.0, 0.9)
        video = self._setup(tmp_path, monkeypatch)
        with patch("translate_video.detect.run", return_value=[overlay]), \
             patch("translate_video.deduplicate.run", return_value=[overlay]), \
             patch("translate_video.translate.run", return_value=[overlay]), \
             patch("translate_video.render.run") as mock_render:
            from translate_video import process_video
            process_video(video, _make_args())
        mock_render.assert_not_called()
        assert (tmp_path / "done" / "test.mp4").exists()

    def test_skip_detect_returns_early_when_no_checkpoint(self, tmp_path, monkeypatch):
        video = self._setup(tmp_path, monkeypatch)
        with patch("translate_video.detect.run") as mock_detect:
            from translate_video import process_video
            process_video(video, _make_args(skip_detect=True))
        mock_detect.assert_not_called()
        assert video.exists()  # video stays in place — not moved

    def test_skip_translate_returns_early_when_no_checkpoint(self, tmp_path, monkeypatch):
        from models.text_overlay import TextOverlay
        overlay = TextOverlay("한국어", "", (0, 0, 50, 20), 0.0, 1.0, 0.9)
        video = self._setup(tmp_path, monkeypatch)
        # Create detections.json so --skip-detect path succeeds, but no translations.json
        ckpt = tmp_path / "checkpoints" / "test"
        ckpt.mkdir()
        import json
        (ckpt / "detections.json").write_text(
            json.dumps([overlay.to_dict()]), encoding="utf-8"
        )
        with patch("translate_video.render.run") as mock_render:
            from translate_video import process_video
            process_video(video, _make_args(skip_detect=True, skip_translate=True))
        mock_render.assert_not_called()
        assert video.exists()  # video stays in place — not moved
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_translate_video.py -v
```

Expected: ImportError — `translate_video` not yet implemented

- [ ] **Step 3: Implement `translate_video.py`**

```python
# translate_video.py
from __future__ import annotations
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import config
from models.text_overlay import TextOverlay
from stages import detect, deduplicate, translate, render

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("translate_video")


def _startup_check() -> None:
    if not config.FONT_PATH.exists():
        sys.exit(
            f"ERROR: Font not found at {config.FONT_PATH}\n"
            "Download NotoSansJP-Bold.otf from:\n"
            "  https://fonts.google.com/noto/specimen/Noto+Sans+JP\n"
            f"and place it in: {config.FONT_PATH.parent}"
        )
    # config.ANTHROPIC_API_KEY uses os.environ[] — KeyError raised at import if unset.
    # Belt-and-suspenders guard for empty-string edge cases.
    if not config.ANTHROPIC_API_KEY:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is set but empty")
    config.VIDEOS_INPUT_DIR.mkdir(exist_ok=True)
    config.VIDEOS_OUTPUT_DIR.mkdir(exist_ok=True)
    config.CHECKPOINTS_DIR.mkdir(exist_ok=True)


def _load_overlays(path: Path) -> list[TextOverlay]:
    return [TextOverlay.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def process_video(video_path: Path, args: argparse.Namespace) -> None:
    stem = video_path.stem
    checkpoint_dir = config.CHECKPOINTS_DIR / stem
    checkpoint_dir.mkdir(exist_ok=True)

    raw_path   = checkpoint_dir / "raw_detections.json"
    det_path   = checkpoint_dir / "detections.json"
    trans_path = checkpoint_dir / "translations.json"

    # ── Stage 1 + 1.5: Detect ───────────────────────────────────────────
    # --skip-* flags take precedence over --force for their respective stages
    if not args.skip_detect:
        if not raw_path.exists() or args.force:
            logger.info("[%s] Stage 1: detecting Korean text...", stem)
            raw_overlays = detect.run(video_path, checkpoint_dir)
        else:
            raw_overlays = _load_overlays(raw_path)
            logger.info("[%s] Stage 1: loaded %d detections from checkpoint", stem, len(raw_overlays))

        if not det_path.exists() or args.force:
            logger.info("[%s] Stage 1.5: deduplicating...", stem)
            overlays = deduplicate.run(raw_overlays, checkpoint_dir)
        else:
            overlays = _load_overlays(det_path)
            logger.info("[%s] Stage 1.5: loaded %d from checkpoint", stem, len(overlays))
    else:
        # --skip-detect skips BOTH Stage 1 and Stage 1.5
        if not det_path.exists():
            logger.warning("[%s] --skip-detect: detections.json missing — skipping video", stem)
            return
        overlays = _load_overlays(det_path)
        logger.info("[%s] --skip-detect: loaded %d overlays", stem, len(overlays))

    if not overlays:
        logger.info("[%s] No Korean text found — moving original to videos-done/", stem)
        shutil.move(str(video_path), str(config.VIDEOS_OUTPUT_DIR / video_path.name))
        return

    # ── Stage 2: Translate ───────────────────────────────────────────────
    if not args.skip_translate:
        if not trans_path.exists() or args.force:
            logger.info("[%s] Stage 2: translating with Claude...", stem)
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
        logger.warning("[%s] All translations failed — moving original, skipping render", stem)
        shutil.move(str(video_path), str(config.VIDEOS_OUTPUT_DIR / video_path.name))
        return

    # ── Stage 3: Render ──────────────────────────────────────────────────
    output_path = config.VIDEOS_OUTPUT_DIR / f"{stem}_ja.mp4"
    logger.info("[%s] Stage 3: rendering → %s", stem, output_path)
    render.run(video_path, translated, output_path)

    shutil.move(str(video_path), str(config.VIDEOS_OUTPUT_DIR / video_path.name))
    logger.info("✓ [%s] complete → %s", stem, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate Korean on-screen text in .mp4 videos to Japanese."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages ignoring checkpoints")
    parser.add_argument("--skip-detect", dest="skip_detect", action="store_true",
                        help="Skip Stage 1+1.5 (use existing detections.json)")
    parser.add_argument("--skip-translate", dest="skip_translate", action="store_true",
                        help="Skip Stage 2 (use existing translations.json)")
    args = parser.parse_args()

    _startup_check()

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
```

- [ ] **Step 4: Run full test suite — expect PASS**

```bash
python -m pytest tests/ -v
```

Expected: all 65 tests across all modules PASS (5 models + 8 font_scaler + 12 detect + 16 deduplicate + 11 translate + 6 render + 7 translate_video)

- [ ] **Step 5: Verify CLI help text**

```bash
python translate_video.py --help
```

Expected: usage text with `--force`, `--skip-detect`, `--skip-translate` listed

- [ ] **Step 6: Commit**

```bash
git add translate_video.py tests/test_translate_video.py
git commit -m "feat: implement main CLI orchestrator with tests"
```

---

## Chunk 9: End-to-End Smoke Test

### Task 11: Download font and run pipeline on real video

- [ ] **Step 1: Download font and add `.gitignore`**

Go to https://fonts.google.com/noto/specimen/Noto+Sans+JP → click "Download family" → unzip → find `NotoSansJP-Bold.otf` → place at `fonts/NotoSansJP-Bold.otf`

```bash
echo "*.otf" > fonts/.gitignore
```

This prevents the binary font file from being accidentally committed while still allowing `fonts/.gitkeep` to be tracked.

- [ ] **Step 2: Set API key**

```bash
export ANTHROPIC_API_KEY=your_key_here
```

- [ ] **Step 3: Confirm videos are in input folder**

```bash
ls videos-to-be-done/
```

Expected: `pre_2_1.mp4` and `pre_2_2.mp4`

- [ ] **Step 4: Run pipeline on one video first**

```bash
# Temporarily move pre_2_2.mp4 aside so only pre_2_1.mp4 processes
mv videos-to-be-done/pre_2_2.mp4 .
python translate_video.py
mv pre_2_2.mp4 videos-to-be-done/
```

Expected log lines (timestamps omitted):
```
[INFO] translate_video: Found 1 video(s) to process
[INFO] translate_video: [pre_2_1] Stage 1: detecting Korean text...
[INFO] translate_video: [pre_2_1] Stage 1.5: deduplicating...
[INFO] translate_video: [pre_2_1] Stage 2: translating with Claude...
[INFO] translate_video: [pre_2_1] Stage 3: rendering → videos-done/pre_2_1_ja.mp4
[INFO] translate_video: ✓ [pre_2_1] complete → videos-done/pre_2_1_ja.mp4
```

- [ ] **Step 5: Verify output files exist**

```bash
ls videos-done/
```

Expected: `pre_2_1_ja.mp4` and `pre_2_1.mp4` both present (translated video + moved original)

- [ ] **Step 6: Spot-check the output video**

Open `videos-done/pre_2_1_ja.mp4` and verify:
- Korean text replaced with black box + white Japanese text
- Japanese text uses natural beauty-ad tone with slang
- Audio plays correctly and is unaltered
- Video duration matches original

- [ ] **Step 7: Test no-Korean-text edge case**

Create a short video with no Korean text (e.g., a plain English or silent video) and verify the pipeline handles it gracefully:

```bash
# Create a 2-second blank test video (requires ffmpeg)
ffmpeg -f lavfi -i color=c=black:s=640x480:d=2 -c:v libx264 videos-to-be-done/no_korean.mp4
python translate_video.py
```

Expected: `no_korean.mp4` moved to `videos-done/no_korean.mp4` (original, unmodified) with log line:
```
[INFO] translate_video: [no_korean] No Korean text found — moving original to videos-done/
```

```bash
ls videos-done/no_korean.mp4
```

Expected: file exists (original moved, no `_ja.mp4` created)

```bash
# Clean up test file
rm -f videos-done/no_korean.mp4
```

- [ ] **Step 8: Test `--force` re-runs all stages on an already-processed video**

Move `pre_2_1.mp4` back from `videos-done/` and run with `--force`:

```bash
mv videos-done/pre_2_1.mp4 videos-to-be-done/
mv videos-to-be-done/pre_2_2.mp4 .
python translate_video.py --force
mv pre_2_2.mp4 videos-to-be-done/
```

Expected: all three stage log lines appear for `pre_2_1` (not "loaded from checkpoint"):
```
[INFO] translate_video: [pre_2_1] Stage 1: detecting Korean text...
[INFO] translate_video: [pre_2_1] Stage 1.5: deduplicating...
[INFO] translate_video: [pre_2_1] Stage 2: translating with Claude...
```

- [ ] **Step 9: Process second video**

```bash
python translate_video.py
```

Expected: `pre_2_2_ja.mp4` written to `videos-done/`

- [ ] **Step 10: Add gitkeep files and final commit**

```bash
touch fonts/.gitkeep videos-to-be-done/.gitkeep videos-done/.gitkeep checkpoints/.gitkeep
git add fonts/.gitignore fonts/.gitkeep videos-to-be-done/.gitkeep videos-done/.gitkeep checkpoints/.gitkeep
git commit -m "feat: complete Korean-to-Japanese video translator pipeline"
```

---

## Quick Reference

```bash
# Full pipeline (processes all .mp4 in videos-to-be-done/)
python translate_video.py

# Resume after OCR crash — skip heavy PaddleOCR re-run
python translate_video.py --skip-detect

# Re-translate only (tweak prompt, keep same detections)
python translate_video.py --skip-detect

# Re-render only (tweak font/color, keep same translations)
python translate_video.py --skip-detect --skip-translate

# Full re-run from scratch (ignore all checkpoints)
python translate_video.py --force

# Run all unit tests
python -m pytest tests/ -v
```
