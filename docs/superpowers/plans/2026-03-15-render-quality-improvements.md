# Render Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 visual quality issues in the Korean→Japanese video overlay pipeline to match the reference video's look: proper brand name rendering, no duplicate text, full-width subtitle bars, and larger readable fonts.

**Architecture:** Three-layer fix — (1) GVI post-processing removes annotation artifacts before translation, (2) translation prompts are updated to romanize brand names, (3) the render stage expands overlays to full video width and enforces minimum font sizes. Each layer is independent and testable.

**Tech Stack:** Python 3.11, Pillow, MoviePy, anthropic SDK, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `stages/gvi_postprocess.py` | **CREATE** | Remove repeated GVI phrases, filter oversized overlays, suppress sub-fragments of title card |
| `stages/detect_gvi.py` | **MODIFY** | Call `gvi_postprocess.run()` after building overlay list |
| `stages/translate.py` | **MODIFY** | Update prompts: brand romanization + phrase dedup instruction |
| `stages/render.py` | **MODIFY** | `compute_render_box()` (full-width expansion), `create_overlay_image()` takes explicit dims, parenthetical style |
| `utils/font_scaler.py` | **MODIFY** | Add `min_font_size` param to `fit_font_size` and `wrap_text` |
| `config.py` | **MODIFY** | Add zone fractions, min font sizes, max area fraction constants |
| `tests/test_gvi_postprocess.py` | **CREATE** | Unit tests for all three post-processing functions |
| `tests/test_render.py` | **MODIFY** | Tests for `compute_render_box`, updated `create_overlay_image` signature |
| `tests/test_font_scaler.py` | **MODIFY** | Tests for `min_font_size` parameter |

---

## Chunk 1: GVI Post-processing

Fixes issues: #4 (duplicate phrases), #6 (oversized overlays), #10 (sub-fragment title overlap).

### Task 1: `stages/gvi_postprocess.py` — repeated phrase removal

**Files:**
- Create: `stages/gvi_postprocess.py`
- Create: `tests/test_gvi_postprocess.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gvi_postprocess.py
import pytest
from stages.gvi_postprocess import remove_repeated_phrases


def test_remove_no_repetition():
    assert remove_repeated_phrases("쓰기 전에는") == "쓰기 전에는"


def test_remove_leading_bigram_repeat():
    # "관리 전 관리 전 1달 후 1달 후" → "관리 전 1달 후"
    result = remove_repeated_phrases("관리 전 관리 전 1달 후 1달 후")
    assert result == "관리 전 1달 후"


def test_remove_leading_phrase_repeat():
    # "다만 효과가 좋아서 그런지 다만 효과가 좋아서 그런지 비싸게 파는데"
    result = remove_repeated_phrases(
        "다만 효과가 좋아서 그런지 다만 효과가 좋아서 그런지 비싸게 파는데"
    )
    assert result == "다만 효과가 좋아서 그런지 비싸게 파는데"


def test_remove_link_phrase_repeat():
    result = remove_repeated_phrases(
        "제가 찾은 구매처 링크 제가 찾은 구매처 링크 공유해놓을게요!"
    )
    assert result == "제가 찾은 구매처 링크 공유해놓을게요!"


def test_remove_single_word_repeat():
    # "전에는 전에는 쓰기?" → "전에는 쓰기?"
    result = remove_repeated_phrases("전에는 전에는 쓰기?")
    assert result == "전에는 쓰기?"


def test_short_text_unchanged():
    assert remove_repeated_phrases("안녕") == "안녕"
    assert remove_repeated_phrases("안녕 하세요") == "안녕 하세요"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/jerryhong/Documents/video-translator
pytest tests/test_gvi_postprocess.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `remove_repeated_phrases`**

```python
# stages/gvi_postprocess.py
from __future__ import annotations
import logging
from pathlib import Path

import cv2

import config
from models.text_overlay import TextOverlay

logger = logging.getLogger(__name__)


def remove_repeated_phrases(text: str) -> str:
    """Remove verbatim repeated phrases caused by GVI annotation artifacts.

    GVI sometimes concatenates duplicate OCR readings of the same word/phrase
    within one annotation segment. E.g.:
      "관리 전 관리 전 1달 후 1달 후" → "관리 전 1달 후"
      "다만 효과가 그런지 다만 효과가 그런지 비싸게" → "다만 효과가 그런지 비싸게"

    Algorithm: find the shortest N-gram (N≥1) that appears at two or more
    positions in the token list; remove the second occurrence; recurse.
    """
    words = text.split()
    n = len(words)
    if n <= 2:
        return text

    for k in range(n // 2, 0, -1):
        for start in range(0, n - k):
            phrase = words[start : start + k]
            for start2 in range(start + 1, n - k + 1):
                if words[start2 : start2 + k] == phrase:
                    new_words = words[:start2] + words[start2 + k :]
                    return remove_repeated_phrases(" ".join(new_words))
    return text
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_gvi_postprocess.py::test_remove_no_repetition \
       tests/test_gvi_postprocess.py::test_remove_leading_bigram_repeat \
       tests/test_gvi_postprocess.py::test_remove_leading_phrase_repeat \
       tests/test_gvi_postprocess.py::test_remove_link_phrase_repeat \
       tests/test_gvi_postprocess.py::test_remove_single_word_repeat \
       tests/test_gvi_postprocess.py::test_short_text_unchanged -v
```

Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add stages/gvi_postprocess.py tests/test_gvi_postprocess.py
git commit -m "feat: add GVI post-processor with repeated-phrase removal"
```

---

### Task 2: Oversized overlay filter + sub-fragment suppressor

**Files:**
- Modify: `stages/gvi_postprocess.py`
- Modify: `tests/test_gvi_postprocess.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_gvi_postprocess.py
from stages.gvi_postprocess import filter_large_overlays, suppress_subfragments
from models.text_overlay import TextOverlay


def _ov(text_ko, bbox, start=0.0, end=5.0, conf=0.99):
    return TextOverlay(text_ko, "", bbox, start, end, conf)


# --- filter_large_overlays ---

def test_filter_keeps_small_overlay():
    # 100x50 = 5000px on 1000x1000 frame = 0.5% — keep
    o = _ov("작은", (0, 0, 100, 50))
    result = filter_large_overlays([o], frame_w=1000, frame_h=1000)
    assert len(result) == 1


def test_filter_removes_oversized_overlay():
    # 800x300 = 240000px on 1080x1920 = 11.6% — remove (threshold 10%)
    o = _ov("광고", (99, 1108, 800, 300))
    result = filter_large_overlays([o], frame_w=1080, frame_h=1920)
    assert len(result) == 0


def test_filter_keeps_borderline_overlay():
    # Area exactly at threshold — keep (exclusive)
    threshold = config.OVERLAY_MAX_AREA_FRACTION
    frame_w, frame_h = 1080, 1920
    max_area = int(frame_w * frame_h * threshold) - 1
    w = 100
    h = max_area // w
    o = _ov("경계", (0, 0, w, h))
    result = filter_large_overlays([o], frame_w=frame_w, frame_h=frame_h)
    assert len(result) == 1


# --- suppress_subfragments ---

def test_suppress_removes_spatially_contained_subfragment():
    # Main title covers t=0-4.67, sub-fragment covers t=1.33-2.0 with same bbox
    main = _ov("브이쎄라를 끊을 수 없는 이유", (262, 248, 487, 170), start=0.0, end=4.67)
    frag = _ov("끊을 수 없는 이유", (264, 326, 477, 90), start=1.33, end=2.0)
    result = suppress_subfragments([main, frag])
    texts = [o.text_ko for o in result]
    assert "브이쎄라를 끊을 수 없는 이유" in texts
    assert "끊을 수 없는 이유" not in texts


def test_suppress_keeps_non_overlapping_overlay():
    # Subtitle at bottom doesn't overlap title at top
    title = _ov("제목", (262, 248, 487, 170), start=0.0, end=4.67)
    subtitle = _ov("자막", (105, 1300, 870, 80), start=0.0, end=4.67)
    result = suppress_subfragments([title, subtitle])
    assert len(result) == 2


def test_suppress_keeps_non_overlapping_time():
    # Two overlays at same position but different time ranges — both kept
    a = _ov("텍스트A", (262, 248, 487, 170), start=0.0, end=2.0)
    b = _ov("텍스트B", (262, 248, 487, 170), start=5.0, end=8.0)
    result = suppress_subfragments([a, b])
    assert len(result) == 2
```

- [ ] **Step 2: Run tests — expect fail**

```bash
pytest tests/test_gvi_postprocess.py -k "filter or suppress" -v 2>&1 | head -20
```

Expected: `ImportError` for the new functions.

- [ ] **Step 3: Implement both functions in `gvi_postprocess.py`**

```python
def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def filter_large_overlays(
    overlays: list[TextOverlay],
    frame_w: int,
    frame_h: int,
) -> list[TextOverlay]:
    """Remove overlays whose bbox area exceeds OVERLAY_MAX_AREA_FRACTION of the frame.

    These are typically embedded product-card or ad graphics detected as text,
    not spoken-word subtitles. They produce enormous black rectangles that
    block the video.
    """
    max_area = frame_w * frame_h * config.OVERLAY_MAX_AREA_FRACTION
    return [o for o in overlays if o.bbox[2] * o.bbox[3] < max_area]


def suppress_subfragments(overlays: list[TextOverlay]) -> list[TextOverlay]:
    """Remove overlays that are spatial+temporal sub-fragments of a larger overlay.

    When GVI produces both a full title ("브이쎄라를 끊을 수 없는 이유", t=0-4.67s)
    and shorter sub-segments of it ("끊을 수 없는 이유", t=1.33-2.0s) with heavily
    overlapping bounding boxes, the sub-segments cause duplicate overlays on the
    title card. This function keeps only the longest-duration overlay at each
    spatial position.
    """
    # Sort longest duration first
    by_duration = sorted(
        range(len(overlays)),
        key=lambda i: overlays[i].end_sec - overlays[i].start_sec,
        reverse=True,
    )
    suppressed: set[int] = set()

    for pos, i in enumerate(by_duration):
        if i in suppressed:
            continue
        a = overlays[i]
        for j in by_duration[pos + 1 :]:
            if j in suppressed:
                continue
            b = overlays[j]
            # B's time range must be fully within A's (with 0.2s tolerance)
            time_contained = (
                b.start_sec >= a.start_sec - 0.2
                and b.end_sec <= a.end_sec + 0.2
            )
            if time_contained and _iou(a.bbox, b.bbox) > 0.3:
                suppressed.add(j)

    return [o for i, o in enumerate(overlays) if i not in suppressed]


def run(
    overlays: list[TextOverlay],
    frame_w: int,
    frame_h: int,
) -> list[TextOverlay]:
    """Apply all GVI post-processing passes in order."""
    # Pass 1: fix repeated phrases within each annotation
    cleaned = [
        TextOverlay(
            text_ko=remove_repeated_phrases(o.text_ko),
            text_ja=o.text_ja,
            bbox=o.bbox,
            start_sec=o.start_sec,
            end_sec=o.end_sec,
            confidence=o.confidence,
        )
        for o in overlays
    ]
    # Pass 2: drop oversized ad/graphic overlays
    cleaned = filter_large_overlays(cleaned, frame_w, frame_h)
    # Pass 3: suppress sub-fragments of title cards
    cleaned = suppress_subfragments(cleaned)
    logger.info(
        "GVI post-process: %d → %d overlays", len(overlays), len(cleaned)
    )
    return cleaned
```

- [ ] **Step 4: Run all gvi_postprocess tests**

```bash
pytest tests/test_gvi_postprocess.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add stages/gvi_postprocess.py tests/test_gvi_postprocess.py
git commit -m "feat: add oversized overlay filter and sub-fragment suppressor"
```

---

### Task 3: Wire `gvi_postprocess.run()` into `detect_gvi.py`

**Files:**
- Modify: `stages/detect_gvi.py`

- [ ] **Step 1: Add import and call in `detect_gvi.run()`**

In `stages/detect_gvi.py`, after the `overlays.sort(...)` line and before writing to `out_path`, add:

```python
from stages import gvi_postprocess  # add at top of file

# Inside run(), after overlays.sort():
overlays = gvi_postprocess.run(overlays, width, height)
```

Full relevant section of `run()` after the change:
```python
    overlays.sort(key=lambda o: o.start_sec)
    overlays = gvi_postprocess.run(overlays, width, height)

    out_path = checkpoint_dir / "detections.json"
    out_path.write_text(...)
```

- [ ] **Step 2: Run existing tests to confirm no breakage**

```bash
pytest tests/ -v --ignore=tests/test_detect.py 2>&1 | tail -20
```

Expected: all existing tests PASS (detect_gvi isn't unit-tested directly, so nothing breaks).

- [ ] **Step 3: Commit**

```bash
git add stages/detect_gvi.py
git commit -m "feat: wire GVI post-processor into detect_gvi pipeline"
```

---

## Chunk 2: Translation Prompt Fixes

Fixes issues: #1 (brand name □□), #4 (residual repeated phrases via Claude correction).

### Task 4: Update translation prompts

**Files:**
- Modify: `stages/translate.py`
- Modify: `tests/test_translate.py`

- [ ] **Step 1: Check existing translate tests**

```bash
pytest tests/test_translate.py -v 2>&1 | tail -20
```

- [ ] **Step 2: Update `_CORRECT_SYSTEM_PROMPT` in `translate.py`**

Replace the existing `_CORRECT_SYSTEM_PROMPT` string with:

```python
_CORRECT_SYSTEM_PROMPT = (
    "You are a Korean proofreader. You will receive a JSON object mapping IDs to Korean text "
    "strings extracted via OCR from a video. Fix the following issues in each string:\n"
    "1. OCR typos, missing spaces, or broken syllables — correct to natural Korean.\n"
    "2. Repeated phrases caused by OCR artifacts — if the same word or phrase appears "
    "twice consecutively or the text looks like two OCR readings concatenated "
    "(e.g. '쓰기 전에는 전에는 쓰기?'), keep only one clean occurrence.\n"
    "3. Brand names: '브이쎄라' is the brand 'V-THERA' — if it appears, correct any "
    "misspelling of it but leave it as Korean (the translator will romanize it).\n"
    "If a string is already correct, return it unchanged. "
    "Respond ONLY with a JSON object mapping the same IDs to the corrected strings. "
    "No explanations."
)
```

- [ ] **Step 3: Update `_SYSTEM_PROMPT` in `translate.py`**

Replace the existing `_SYSTEM_PROMPT` string with:

```python
_SYSTEM_PROMPT = (
    "You are a professional Japanese copywriter specialising in beauty and cosmetics "
    "advertising targeting Japanese girls aged 18–25. Translate each Korean phrase into "
    "natural, trendy Japanese suitable for a beauty ad campaign. Actively use contemporary "
    "beauty slang popular among late-teen to mid-twenties Japanese women — words like "
    "'うるつや', 'ぷるぷる', 'もちもち', 'つるつる', 'バズり', 'エモい', 'ガチ', "
    "'めちゃ盛れ', 'スキンケア沼' etc. where contextually appropriate. Keep translations "
    "concise — they must fit within the same space as the original text.\n"
    "IMPORTANT: Korean brand names must be romanized — '브이쎄라' → 'V-THERA', "
    "'마미케어' → 'Mummy Care'. Do NOT leave Korean characters in the Japanese output; "
    "the font cannot render Hangul.\n"
    "Respond ONLY with a JSON object mapping each Korean input to its Japanese translation. "
    "No explanations."
)
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_translate.py -v
```

Expected: all PASS (prompt constants are not directly tested — tests mock the API call).

- [ ] **Step 5: Commit**

```bash
git add stages/translate.py
git commit -m "fix: romanize Korean brand names and remove repeated phrases in translation prompts"
```

---

## Chunk 3: Render Overhaul

Fixes issues: #2 (solid black box), #3 (title not full-width), #5 (font too small), #7 (split-screen labels), #8 (parenthetical style), #9 (line wrapping).

### Task 5: Add config constants

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add new constants to `config.py`**

After the `# ── Overlay Styling ──` block, add:

```python
# ── Render Zones ─────────────────────────────────────────────────────────
# Vertical fraction thresholds for title vs subtitle zones (0.0 = top, 1.0 = bottom)
TITLE_Y_FRACTION    = 0.38   # center y < this → title zone (expand full width)
SUBTITLE_Y_FRACTION = 0.62   # center y > this → subtitle zone (expand full width)

# Minimum rendered font sizes per zone
TITLE_MIN_FONT_SIZE    = 46
SUBTITLE_MIN_FONT_SIZE = 40
MID_MIN_FONT_SIZE      = 28

# Maximum fraction of frame area a single overlay may occupy (filters ad graphics)
OVERLAY_MAX_AREA_FRACTION = 0.10
```

- [ ] **Step 2: Run tests to confirm nothing breaks**

```bash
pytest tests/ -v 2>&1 | tail -10
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "config: add render zone fractions, min font sizes, max area fraction"
```

---

### Task 6: `fit_font_size` min-size floor

**Files:**
- Modify: `utils/font_scaler.py`
- Modify: `tests/test_font_scaler.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_font_scaler.py

@patch("utils.font_scaler.ImageFont.truetype")
@patch("utils.font_scaler._measure_text")
def test_fit_font_size_respects_min_size_floor(mock_measure, mock_truetype):
    """When min_size=40 is given, never returns a size below 40."""
    mock_measure.return_value = (9999, 9999)  # nothing fits
    mock_truetype.return_value = MagicMock()

    size, _ = fit_font_size(
        "長いテキスト", bbox_w=10, bbox_h=10, font_path="fake.otf", padding=4,
        min_size=40,
    )
    assert size == 40
```

- [ ] **Step 2: Run — expect fail**

```bash
pytest tests/test_font_scaler.py::test_fit_font_size_respects_min_size_floor -v
```

Expected: `TypeError` (unexpected keyword argument `min_size`).

- [ ] **Step 3: Add `min_size` param to `fit_font_size` and `wrap_text`**

In `utils/font_scaler.py`, modify `fit_font_size`:

```python
def fit_font_size(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: str | Path,
    padding: int,
    min_size: int | None = None,
) -> tuple[int, ImageFont.FreeTypeFont]:
    """Return (size, font) — the largest size where text fits within bbox minus padding.
    Never returns a size below min_size (defaults to config.FONT_SIZE_MIN).
    """
    floor = min_size if min_size is not None else config.FONT_SIZE_MIN
    usable_w = bbox_w - padding * 2
    usable_h = bbox_h - padding * 2

    for size in range(config.FONT_SIZE_MAX, floor - 1, -1):
        font = ImageFont.truetype(str(font_path), size)
        text_w, text_h = _measure_text(text, font)
        if text_w <= usable_w and text_h <= usable_h:
            return size, font

    font = ImageFont.truetype(str(font_path), floor)
    return floor, font
```

Also update `wrap_text` signature to pass `min_size` through:

```python
def wrap_text(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: str | Path,
    padding: int,
    min_size: int | None = None,
) -> tuple[str, int, ImageFont.FreeTypeFont]:
    """Return (wrapped_text, size, font) fitting within bbox.

    IMPORTANT: callers must render the returned wrapped_text, not the original input.
    min_size sets the minimum font size floor (never shrinks below this).
    """
    single_size, single_font = fit_font_size(
        text, bbox_w, bbox_h, font_path, padding, min_size=min_size
    )

    if single_size >= config.FONT_SIZE_WRAP_THRESHOLD:
        return text, single_size, single_font

    if len(text) <= 1:
        return text, single_size, single_font

    mid = len(text) // 2
    wrapped = text[:mid] + "\n" + text[mid:]
    wrap_size, wrap_font = fit_font_size(
        wrapped, bbox_w, bbox_h, font_path, padding, min_size=min_size
    )

    if wrap_size > single_size:
        return wrapped, wrap_size, wrap_font

    return text, single_size, single_font
```

- [ ] **Step 4: Run font_scaler tests**

```bash
pytest tests/test_font_scaler.py -v
```

Expected: all PASS (existing tests also pass because `min_size=None` defaults to old behavior).

- [ ] **Step 5: Commit**

```bash
git add utils/font_scaler.py tests/test_font_scaler.py
git commit -m "feat: add min_size floor to fit_font_size and wrap_text"
```

---

### Task 7: `compute_render_box` — full-width expansion

**Files:**
- Modify: `stages/render.py`
- Modify: `tests/test_render.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_render.py
from stages.render import compute_render_box

VIDEO_W, VIDEO_H = 1080, 1920


def test_compute_render_box_title_zone_expands_full_width():
    """Title zone (y_center < TITLE_Y_FRACTION) → render_x=0, render_w=video_w."""
    # y=100, h=200 → center y=200, center fraction=200/1920≈0.104 < 0.38
    o = TextOverlay("타이틀", "", (262, 100, 487, 200), 0.0, 4.0, 0.99)
    rx, ry, rw, rh, min_font = compute_render_box(o, VIDEO_W, VIDEO_H)
    assert rx == 0
    assert rw == VIDEO_W
    assert min_font == config.TITLE_MIN_FONT_SIZE


def test_compute_render_box_subtitle_zone_expands_full_width():
    """Subtitle zone (y_center > SUBTITLE_Y_FRACTION) → render_x=0, render_w=video_w."""
    # y=1300, h=80 → center y=1340, fraction≈0.698 > 0.62
    o = TextOverlay("자막", "", (105, 1300, 870, 80), 0.0, 4.0, 0.99)
    rx, ry, rw, rh, min_font = compute_render_box(o, VIDEO_W, VIDEO_H)
    assert rx == 0
    assert rw == VIDEO_W
    assert min_font == config.SUBTITLE_MIN_FONT_SIZE


def test_compute_render_box_mid_zone_keeps_original_width():
    """Mid zone overlay keeps original x and w."""
    # y=800, h=100 → center y=850, fraction≈0.443 (between 0.38 and 0.62)
    o = TextOverlay("중간", "", (200, 800, 400, 100), 0.0, 4.0, 0.99)
    rx, ry, rw, rh, min_font = compute_render_box(o, VIDEO_W, VIDEO_H)
    assert rx == 200
    assert rw == 400
    assert min_font == config.MID_MIN_FONT_SIZE


def test_compute_render_box_height_at_least_min_font_plus_padding():
    """Render height is at least min_font + 4*padding to fit text."""
    o = TextOverlay("타이틀", "", (262, 248, 487, 10), 0.0, 4.0, 0.99)  # h=10 very small
    _, _, _, rh, min_font = compute_render_box(o, VIDEO_W, VIDEO_H)
    assert rh >= min_font + config.BOX_PADDING * 4


def test_compute_render_box_clips_to_frame_bottom():
    """If y + render_h would exceed video_h, y is moved up."""
    # y near bottom
    o = TextOverlay("하단", "", (0, 1910, 500, 80), 0.0, 2.0, 0.99)
    rx, ry, rw, rh, _ = compute_render_box(o, VIDEO_W, VIDEO_H)
    assert ry + rh <= VIDEO_H
```

- [ ] **Step 2: Run — expect fail**

```bash
pytest tests/test_render.py -k "compute_render_box" -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `compute_render_box` in `render.py`**

Add this function to `stages/render.py` (before `create_overlay_image`):

```python
def compute_render_box(
    overlay: TextOverlay,
    video_w: int,
    video_h: int,
) -> tuple[int, int, int, int, int]:
    """Return (x, y, w, h, min_font) for the rendered overlay.

    Title zone (upper) and subtitle zone (lower) expand to full video width
    so the text reads like a proper banner rather than a bbox-constrained snippet.
    Mid-frame overlays keep their original dimensions.

    Returns a 5-tuple: (render_x, render_y, render_w, render_h, min_font_size)
    """
    src_x, src_y, src_w, src_h = overlay.bbox
    y_center_frac = (src_y + src_h / 2) / video_h

    if y_center_frac < config.TITLE_Y_FRACTION:
        render_x = 0
        render_w = video_w
        min_font = config.TITLE_MIN_FONT_SIZE
    elif y_center_frac > config.SUBTITLE_Y_FRACTION:
        render_x = 0
        render_w = video_w
        min_font = config.SUBTITLE_MIN_FONT_SIZE
    else:
        render_x = src_x
        render_w = src_w
        min_font = config.MID_MIN_FONT_SIZE

    render_h = max(src_h, min_font + config.BOX_PADDING * 4)
    render_y = src_y

    # Clip to frame bounds
    if render_y + render_h > video_h:
        render_y = video_h - render_h

    return render_x, render_y, render_w, render_h, min_font
```

- [ ] **Step 4: Run compute_render_box tests**

```bash
pytest tests/test_render.py -k "compute_render_box" -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add stages/render.py tests/test_render.py
git commit -m "feat: add compute_render_box for full-width title/subtitle expansion"
```

---

### Task 8: Update `create_overlay_image` and `build_overlay_clip` to use new dimensions

**Files:**
- Modify: `stages/render.py`
- Modify: `tests/test_render.py`

- [ ] **Step 1: Write updated tests**

```python
# Add/update in tests/test_render.py

@patch("stages.render.ImageDraw.Draw")
@patch("stages.render.wrap_text")
def test_overlay_image_uses_render_dimensions(mock_wrap, mock_draw_cls):
    """create_overlay_image uses the provided render_w/render_h, not overlay bbox."""
    mock_wrap.return_value = ("テスト", 46, MagicMock())
    mock_draw = MagicMock()
    mock_draw.textbbox.return_value = (0, 0, 500, 50)
    mock_draw_cls.return_value = mock_draw

    o = TextOverlay("타이틀", "テスト", (262, 248, 487, 170), 0.0, 4.0, 0.99)
    img = create_overlay_image(o, render_w=1080, render_h=200, min_font=46)
    assert img.size == (1080, 200)


@patch("stages.render.compute_render_box")
@patch("stages.render.create_overlay_image")
@patch("stages.render.ImageClip")
def test_build_clip_uses_computed_render_box(mock_clip_cls, mock_create, mock_crb):
    """build_overlay_clip positions the clip at compute_render_box (rx, ry)."""
    from PIL import Image as PILImage
    mock_crb.return_value = (0, 248, 1080, 200, 46)
    mock_create.return_value = PILImage.new("RGBA", (1080, 200))
    mock_clip = MagicMock()
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_start.return_value = mock_clip
    mock_clip.set_end.return_value = mock_clip
    mock_clip_cls.return_value = mock_clip

    o = TextOverlay("타이틀", "テスト", (262, 248, 487, 170), 0.0, 4.0, 0.99)
    build_overlay_clip(o, video_w=1080, video_h=1920)

    mock_clip.set_position.assert_called_once_with((0, 248))


@patch("stages.render.build_overlay_clip")
@patch("stages.render.CompositeVideoClip")
@patch("stages.render.VideoFileClip")
def test_run_passes_video_dims_to_build_clip(mock_vfc_cls, mock_composite_cls, mock_build):
    """run() passes video width/height to build_overlay_clip."""
    mock_video = MagicMock()
    mock_video.fps = 30.0
    mock_video.size = (1080, 1920)
    mock_vfc_cls.return_value = mock_video
    mock_composite_cls.return_value = MagicMock()
    mock_build.return_value = None

    overlay = TextOverlay("한국어", "テスト", (0, 0, 100, 30), 1.0, 3.0, 0.9)
    run(Path("input.mp4"), [overlay], Path("output.mp4"))

    mock_build.assert_called_once_with(overlay, video_w=1080, video_h=1920)
```

- [ ] **Step 2: Run — expect fail**

```bash
pytest tests/test_render.py -k "render_dimensions or computed_render or video_dims" -v
```

- [ ] **Step 3: Rewrite `create_overlay_image`, `build_overlay_clip`, `run` in `render.py`**

```python
def create_overlay_image(
    overlay: TextOverlay,
    render_w: int,
    render_h: int,
    min_font: int,
) -> Image.Image:
    """Draw overlay: solid BOX_COLOR bg with centered Japanese text. Returns RGBA image."""
    # Parenthetical text (e.g. "(顔も小さくなった気がします！)") uses smaller font
    effective_min = (
        max(config.FONT_SIZE_MIN, int(min_font * 0.7))
        if overlay.text_ja.startswith("(")
        else min_font
    )

    box_color_rgba = (*config.BOX_COLOR, 255)
    text_color_rgba = (*config.TEXT_COLOR, 255)
    img = Image.new("RGBA", (render_w, render_h), box_color_rgba)
    draw = ImageDraw.Draw(img)

    wrapped_text, _, font = wrap_text(
        overlay.text_ja, render_w, render_h, config.FONT_PATH,
        config.BOX_PADDING, min_size=effective_min,
    )

    bb = draw.textbbox((0, 0), wrapped_text, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]
    x = (render_w - text_w) // 2
    y = (render_h - text_h) // 2

    draw.text((x, y), wrapped_text, font=font, fill=text_color_rgba)
    return img


def build_overlay_clip(
    overlay: TextOverlay,
    video_w: int,
    video_h: int,
) -> ImageClip | None:
    """Return a timed ImageClip at the computed render position, or None if text_ja is empty."""
    if not overlay.text_ja:
        return None

    rx, ry, rw, rh, min_font = compute_render_box(overlay, video_w, video_h)
    img = create_overlay_image(overlay, rw, rh, min_font)

    clip = (
        ImageClip(np.array(img))
        .set_position((rx, ry))
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
    video_w, video_h = original.size  # (width, height)

    active_clips = [
        c
        for o in overlays
        if (c := build_overlay_clip(o, video_w=video_w, video_h=video_h)) is not None
    ]

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

- [ ] **Step 4: Update old render tests that assumed old signatures**

The existing test `test_overlay_image_correct_size` calls `create_overlay_image(_o("テスト", bbox=(100, 200, 300, 50)))`. Update it to use the new signature:

```python
def test_overlay_image_correct_size(mock_wrap, mock_draw_cls):
    mock_wrap.return_value = ("テスト", 16, MagicMock())
    mock_draw = MagicMock()
    mock_draw.textbbox.return_value = (0, 0, 80, 20)
    mock_draw_cls.return_value = mock_draw
    img = create_overlay_image(_o("テスト", bbox=(100, 200, 300, 50)), render_w=300, render_h=50, min_font=16)
    assert img.size == (300, 50)
```

Also update `test_overlay_image_background_is_box_color`:
```python
def test_overlay_image_background_is_box_color(mock_wrap, mock_draw_cls):
    mock_wrap.return_value = ("テスト", 16, MagicMock())
    mock_draw = MagicMock()
    mock_draw.textbbox.return_value = (0, 0, 80, 20)
    mock_draw_cls.return_value = mock_draw
    img = create_overlay_image(_o("テスト", bbox=(0, 0, 200, 60)), render_w=200, render_h=60, min_font=16)
    assert img.getpixel((0, 0))[:3] == config.BOX_COLOR
    assert img.mode == "RGBA"
```

Also update `test_build_clip_sets_position_start_end` to pass video dims:
```python
def test_build_clip_sets_position_start_end(mock_create, mock_clip_cls):
    mock_img = Image.new("RGBA", (300, 50), (0, 0, 0, 255))
    mock_create.return_value = mock_img
    mock_clip = MagicMock()
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_start.return_value = mock_clip
    mock_clip.set_end.return_value = mock_clip
    mock_clip_cls.return_value = mock_clip

    overlay = _o("テスト", bbox=(100, 960, 300, 50))  # mid zone: y_center=985/1920=0.51
    with patch("stages.render.compute_render_box") as mock_crb:
        mock_crb.return_value = (100, 960, 300, 50, 28)
        result = build_overlay_clip(overlay, video_w=1080, video_h=1920)

    assert result is not None
    mock_clip.set_position.assert_called_once_with((100, 960))
```

Also update `test_build_clip_skips_empty_text_ja`:
```python
def test_build_clip_skips_empty_text_ja(mock_create):
    result = build_overlay_clip(_o(""), video_w=1080, video_h=1920)
    mock_create.assert_not_called()
    assert result is None
```

- [ ] **Step 5: Run all render tests**

```bash
pytest tests/test_render.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add stages/render.py tests/test_render.py
git commit -m "feat: full-width overlay expansion with per-zone min font sizes"
```

---

## Chunk 4: End-to-End Verification

### Task 9: Delete stale checkpoints and run full pipeline

- [ ] **Step 1: Move video back to input, delete stale checkpoints**

```bash
cp /Users/jerryhong/Documents/video-translator/videos-done/pre_2_1.mp4 \
   /Users/jerryhong/Documents/video-translator/videos-to-be-done/
rm -rf /Users/jerryhong/Documents/video-translator/checkpoints/pre_2_1/
```

- [ ] **Step 2: Run full pipeline**

```bash
GOOGLE_APPLICATION_CREDENTIALS=/Users/jerryhong/Documents/video-translator/key.json \
python translate_video_gvi.py 2>&1 | tee /tmp/pipeline_run.log
```

Expected: completes with `✓ [pre_2_1] complete`.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 4: Extract and visually verify key frames**

```bash
python -c "
import cv2, os
cap = cv2.VideoCapture('videos-done/pre_2_1_ja.mp4')
os.makedirs('/tmp/verify_frames', exist_ok=True)
for t in [0.5, 2.0, 4.7, 6.5, 8.5, 11.0, 16.5, 21.5]:
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
    ret, frame = cap.read()
    if ret: cv2.imwrite(f'/tmp/verify_frames/t{t}.jpg', frame)
cap.release()
print('frames extracted')
"
```

Visually verify each frame against the reference checklist:
- [ ] t=0.5: Title spans full width, no □□ characters
- [ ] t=4.7: Labels show "ケア前 1ヶ月後" (single clean line) or better
- [ ] t=6.5: Clean subtitle, no duplicated words
- [ ] t=8.5: Subtitle font visibly larger than before
- [ ] t=16.5: No massive black box; smaller subtitle only
- [ ] t=21.5: Full-width bottom bar with readable font

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete render quality improvements — full-width overlays, brand name fix, GVI dedup"
```
