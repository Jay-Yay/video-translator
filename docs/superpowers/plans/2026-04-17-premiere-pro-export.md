# Premiere Pro Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the burned-in `_ja.mp4` output with a Premiere Pro 2026 `.prproj` project file that contains the original video on V1, editable Essential Graphics Japanese text clips on V2, muted Korean audio on A1, and Japanese TTS clips on A2.

**Architecture:** A new `stages/prproj_builder.py` handles all `.prproj` XML manipulation (zlib decompress → inject clips → recompress) using a manually-created `templates/base.prproj` as the schema reference. `stages/tts.py` is stripped to mp3-generation-only (returns `dict[str, Path]`). A new `stages/export_prproj.py` orchestrates file layout and calls the builder. `translate_video_gvi.py` is rewired to call the new stages.

**Tech Stack:** `xml.etree.ElementTree` (stdlib), `zlib` (stdlib), `cv2` (already installed), `edge_tts` (already installed), `pytest`, `pytest-mock`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `stages/tts.py` | Strip video-write logic; return `dict[str, Path]` |
| Create | `stages/prproj_builder.py` | zlib decompress/compress, XML clone/inject, coordinate+timecode math |
| Create | `stages/export_prproj.py` | Orchestrate folder layout + call builder |
| Modify | `translate_video_gvi.py` | Rewire stage calls; remove `_ja.mp4` output path |
| Create | `templates/base.prproj` | Manual: minimal PP2026 project (one sequence, one text clip on V2) |
| Create | `scripts/inspect_template.py` | One-time helper: decompress template → pretty-print XML for inspection |
| Create | `tests/test_tts_mp3.py` | Unit tests for modified tts.run |
| Create | `tests/test_prproj_builder.py` | Unit tests for math utils + XML manipulation |
| Create | `tests/test_export_prproj.py` | Unit tests for export_prproj.run |

---

## Task 1: Modify `stages/tts.py` — strip video logic, return mp3 dict

**Files:**
- Modify: `stages/tts.py`
- Create: `tests/test_tts_mp3.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tts_mp3.py
import hashlib
from pathlib import Path
from models.text_overlay import TextOverlay
from stages import tts


def _overlay(text_ja: str) -> TextOverlay:
    return TextOverlay(text_ko="안녕", text_ja=text_ja,
                       bbox=(0, 0, 100, 50), start_sec=0.0,
                       end_sec=2.0, confidence=0.9)


def _make_cached_mp3(text_ja: str, tts_dir: Path) -> Path:
    """Pre-create expected mp3 file so synthesis is skipped."""
    digest = hashlib.md5(text_ja.encode()).hexdigest()[:12]
    path = tts_dir / f"{digest}.mp3"
    path.write_bytes(b"fake-mp3")
    return path


def test_run_returns_dict_of_mp3_paths(tmp_path):
    tts_dir = tmp_path / "tts"
    tts_dir.mkdir()
    _make_cached_mp3("こんにちは", tts_dir)
    overlays = [_overlay("こんにちは")]
    result = tts.run(overlays, tmp_path)
    assert isinstance(result, dict)
    assert "こんにちは" in result
    assert result["こんにちは"].suffix == ".mp3"
    assert result["こんにちは"].exists()


def test_run_deduplicates_identical_text_ja(tmp_path):
    tts_dir = tmp_path / "tts"
    tts_dir.mkdir()
    _make_cached_mp3("こんにちは", tts_dir)
    overlays = [_overlay("こんにちは"), _overlay("こんにちは")]
    result = tts.run(overlays, tmp_path)
    assert list(result.keys()) == ["こんにちは"]


def test_run_skips_empty_text_ja(tmp_path):
    overlays = [_overlay("")]
    result = tts.run(overlays, tmp_path)
    assert result == {}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/jerryhong/Documents/video-translator
pytest tests/test_tts_mp3.py -v
```
Expected: `FAILED` — `TypeError` because `tts.run` currently takes 4 args and returns `None`.

- [ ] **Step 3: Rewrite `stages/tts.py`**

Replace the entire file:

```python
# stages/tts.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import edge_tts

import config
from models.text_overlay import TextOverlay

logger = logging.getLogger(__name__)


async def _synthesize(text: str, voice: str, path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(path))


def _clip_path_for(text: str, tts_dir: Path) -> Path:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return tts_dir / f"{digest}.mp3"


def run(overlays: list[TextOverlay], checkpoint_dir: Path) -> dict[str, Path]:
    """Generate Japanese TTS mp3 files for all unique text_ja.

    Returns {text_ja: mp3_path} for every overlay that has non-empty text_ja.
    Files are cached in checkpoint_dir/tts/ and reused across runs.
    """
    tts_dir = checkpoint_dir / "tts"
    tts_dir.mkdir(exist_ok=True)

    seen: dict[str, Path] = {}

    for overlay in overlays:
        if not overlay.text_ja or overlay.text_ja in seen:
            continue

        clip_path = _clip_path_for(overlay.text_ja, tts_dir)
        seen[overlay.text_ja] = clip_path

        if not clip_path.exists():
            logger.info("TTS: synthesising → %s", overlay.text_ja[:60])
            try:
                asyncio.run(_synthesize(overlay.text_ja, config.TTS_VOICE, clip_path))
            except Exception as exc:
                logger.warning("TTS: synthesis failed for '%s': %s",
                               overlay.text_ja[:40], exc)
                del seen[overlay.text_ja]

    logger.info("TTS: %d unique audio clips ready", len(seen))
    return seen
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tts_mp3.py -v
```
Expected: all 3 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add stages/tts.py tests/test_tts_mp3.py
git commit -m "feat: tts.run returns mp3 dict, strips video-write logic"
```

---

## Task 2: Create `stages/prproj_builder.py` — math utilities

**Files:**
- Create: `stages/prproj_builder.py`
- Create: `tests/test_prproj_builder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prproj_builder.py
import pytest
from stages import prproj_builder


# ── bbox_to_pp_position ──────────────────────────────────────────────────

def test_center_of_frame_maps_to_origin():
    # bbox centered at (960, 540) in 1920x1080 → PP (0.0, 0.0)
    pp_x, pp_y = prproj_builder.bbox_to_pp_position((860, 490, 200, 100), 1920, 1080)
    assert pp_x == pytest.approx(0.0)
    assert pp_y == pytest.approx(0.0)


def test_top_left_corner_bbox():
    # center at (50, 25) in 1920x1080 → PP (-910, +515)
    pp_x, pp_y = prproj_builder.bbox_to_pp_position((0, 0, 100, 50), 1920, 1080)
    assert pp_x == pytest.approx(-910.0)
    assert pp_y == pytest.approx(515.0)


def test_bottom_right_corner_bbox():
    # center at (1870, 1055) in 1920x1080 → PP (+910, -515)
    pp_x, pp_y = prproj_builder.bbox_to_pp_position((1820, 1030, 100, 50), 1920, 1080)
    assert pp_x == pytest.approx(910.0)
    assert pp_y == pytest.approx(-515.0)


# ── sec_to_ticks ─────────────────────────────────────────────────────────

def test_one_second_is_254016000_ticks():
    assert prproj_builder.sec_to_ticks(1.0) == 254016000


def test_zero_seconds():
    assert prproj_builder.sec_to_ticks(0.0) == 0


def test_fractional_seconds():
    assert prproj_builder.sec_to_ticks(2.5) == 635040000
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_prproj_builder.py -v
```
Expected: `ERROR` — `ModuleNotFoundError: No module named 'stages.prproj_builder'`.

- [ ] **Step 3: Create `stages/prproj_builder.py` with math utilities only**

```python
# stages/prproj_builder.py
from __future__ import annotations
import logging
import uuid
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import cv2

import config
from models.text_overlay import TextOverlay

logger = logging.getLogger(__name__)

_PP_TICKS_PER_SEC = 254016000


def bbox_to_pp_position(
    bbox: tuple[int, int, int, int],
    video_w: int,
    video_h: int,
) -> tuple[float, float]:
    """Convert pixel bbox to Premiere Pro sequence coordinates.

    PP origin is at frame center; y-axis points up (inverted vs pixel coords).
    Returns (pp_x, pp_y) as floats representing the bbox center point.
    """
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    pp_x = center_x - video_w / 2
    pp_y = -(center_y - video_h / 2)
    return pp_x, pp_y


def sec_to_ticks(sec: float) -> int:
    """Convert seconds to Premiere Pro internal ticks (1 tick = 1/254016000 s)."""
    return int(sec * _PP_TICKS_PER_SEC)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prproj_builder.py -v
```
Expected: all 6 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add stages/prproj_builder.py tests/test_prproj_builder.py
git commit -m "feat: add prproj_builder with coordinate + timecode math"
```

---

## Task 3: Create `templates/base.prproj` (manual) + inspect XML structure

This task is a manual prerequisite. No automated tests.

- [ ] **Step 1: Create the template in Premiere Pro 2026**

1. Open Premiere Pro 2026 → New Project → save as `base` anywhere
2. New Sequence → `1080p` preset (1920×1080, 29.97fps) → name it `base_sequence`
3. In the **Essential Graphics** panel (Window → Essential Graphics):
   - Click **New Layer → Text**
   - Type any placeholder text (e.g. `PLACEHOLDER`)
   - Position it somewhere visible on the canvas
4. The text clip should appear on **V2** in the timeline
5. `File → Save` → save the project file to:
   `{repo}/templates/base.prproj`

- [ ] **Step 2: Create the inspection helper script**

```python
# scripts/inspect_template.py
"""Decompress base.prproj and pretty-print XML for structure inspection."""
import zlib
import xml.dom.minidom
from pathlib import Path

data = (Path(__file__).parent.parent / "templates" / "base.prproj").read_bytes()

# Try standard zlib, then raw deflate, then gzip
xml_bytes = None
for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 16):
    try:
        xml_bytes = zlib.decompress(data, wbits)
        break
    except zlib.error:
        continue

if xml_bytes is None:
    raise RuntimeError("Cannot decompress base.prproj — unknown compression")

out = Path(__file__).parent.parent / "templates" / "base_pretty.xml"
out.write_text(
    xml.dom.minidom.parseString(xml_bytes).toprettyxml(indent="  "),
    encoding="utf-8",
)
print(f"Written: {out}")
print(f"Decompressed size: {len(xml_bytes):,} bytes")
```

- [ ] **Step 3: Run the inspection script**

```bash
python scripts/inspect_template.py
```
Expected output:
```
Written: .../templates/base_pretty.xml
Decompressed size: ~50,000+ bytes
```

- [ ] **Step 4: Inspect `templates/base_pretty.xml` and record these values**

Open `templates/base_pretty.xml` in a text editor and find:

```
# Record these element paths — you will use them in Task 4 and 5:
A) Root compression wbits that worked (note which of MAX_WBITS / -MAX_WBITS / MAX_WBITS|16)
B) XPath to sequence <FrameWidth> and <FrameHeight> elements
C) XPath to sequence timebase element (frames per second numerator/denominator)
D) The ObjectID of the text clip on V2 (search for "PLACEHOLDER" to find it)
E) The full element subtree of that text clip (copy to a scratch file)
F) The XML element/attribute that stores the text string content
G) The XML attribute that stores position X and Y
H) The XML attributes for clip start tick and end tick
I) The ObjectID of the V2 video track
J) The ObjectID of the A1 audio track (to mute it)
K) The ObjectID of the A2 audio track (for TTS clips)
```

- [ ] **Step 5: Commit template + scripts**

```bash
git add templates/base.prproj scripts/inspect_template.py
git commit -m "feat: add base.prproj template + inspect_template.py helper"
```

> **Note:** `templates/base_pretty.xml` is a derived file — do NOT commit it. Add it to `.gitignore`.

---

## Task 4: `prproj_builder.py` — template load/save utilities

**Files:**
- Modify: `stages/prproj_builder.py`
- Modify: `tests/test_prproj_builder.py`

> **Before coding:** confirm the wbits value found in Task 3 Step 4 (A).
> Use that value in `_decompress()` below. Default shown is `zlib.MAX_WBITS`.

- [ ] **Step 1: Add tests for load/save round-trip**

Append to `tests/test_prproj_builder.py`:

```python
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path


def _make_fake_prproj(tmp_path: Path, xml_content: str) -> Path:
    """Create a minimal zlib-compressed fake .prproj for testing."""
    raw = xml_content.encode("utf-8")
    compressed = zlib.compress(raw)
    path = tmp_path / "fake.prproj"
    path.write_bytes(compressed)
    return path


MINIMAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PremiereData Version="3">
  <Project ObjectID="1">
    <Sequence ObjectID="2">
      <Settings>
        <FrameWidth>1920</FrameWidth>
        <FrameHeight>1080</FrameHeight>
        <FrameRate>254016000</FrameRate>
      </Settings>
    </Sequence>
  </Project>
</PremiereData>"""


def test_load_template_returns_element_tree_root(tmp_path):
    prproj_path = _make_fake_prproj(tmp_path, MINIMAL_XML)
    root = prproj_builder.load_template(prproj_path)
    assert root.tag == "PremiereData"


def test_save_prproj_round_trips(tmp_path):
    prproj_path = _make_fake_prproj(tmp_path, MINIMAL_XML)
    root = prproj_builder.load_template(prproj_path)
    out = tmp_path / "out.prproj"
    prproj_builder.save_prproj(root, out)
    # Reload and verify content survives round-trip
    root2 = prproj_builder.load_template(out)
    assert root2.find(".//FrameWidth").text == "1920"
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_prproj_builder.py::test_load_template_returns_element_tree_root \
       tests/test_prproj_builder.py::test_save_prproj_round_trips -v
```
Expected: `FAILED` — `AttributeError: module 'stages.prproj_builder' has no attribute 'load_template'`.

- [ ] **Step 3: Add `load_template` and `save_prproj` to `stages/prproj_builder.py`**

```python
# Add after the existing math functions:

def _decompress(data: bytes) -> bytes:
    """Try standard zlib, raw deflate, gzip — return first that succeeds."""
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 16):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    raise ValueError("Cannot decompress .prproj — unknown compression format")


def load_template(template_path: Path) -> ET.Element:
    """Decompress .prproj and return parsed XML root element."""
    data = template_path.read_bytes()
    xml_bytes = _decompress(data)
    return ET.fromstring(xml_bytes)


def save_prproj(root: ET.Element, output_path: Path) -> None:
    """Serialize root to XML, zlib-compress, and write to output_path."""
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output_path.write_bytes(zlib.compress(xml_bytes))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prproj_builder.py -v
```
Expected: all 8 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add stages/prproj_builder.py tests/test_prproj_builder.py
git commit -m "feat: add prproj load/save round-trip utilities"
```

---

## Task 5: `prproj_builder.py` — update sequence settings + unique ObjectID

**Files:**
- Modify: `stages/prproj_builder.py`
- Modify: `tests/test_prproj_builder.py`

> **Before coding:** confirm the XPath to FrameWidth, FrameHeight, and timebase elements from Task 3 Step 4 (B, C). The XPaths below use the element names found in `base_pretty.xml`; adjust if different.

- [ ] **Step 1: Add tests**

Append to `tests/test_prproj_builder.py`:

```python
def test_update_sequence_settings_sets_dimensions(tmp_path):
    prproj_path = _make_fake_prproj(tmp_path, MINIMAL_XML)
    root = prproj_builder.load_template(prproj_path)
    prproj_builder.update_sequence_settings(root, video_w=3840, video_h=2160, fps=25.0)
    assert root.find(".//FrameWidth").text == "3840"
    assert root.find(".//FrameHeight").text == "2160"


def test_new_object_id_is_unique():
    ids = {prproj_builder.new_object_id() for _ in range(100)}
    assert len(ids) == 100  # all unique
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_prproj_builder.py::test_update_sequence_settings_sets_dimensions \
       tests/test_prproj_builder.py::test_new_object_id_is_unique -v
```
Expected: `FAILED`.

- [ ] **Step 3: Implement in `stages/prproj_builder.py`**

```python
def new_object_id() -> str:
    """Generate a unique ObjectID string in Premiere's format."""
    return str(uuid.uuid4()).upper()


def update_sequence_settings(
    root: ET.Element, video_w: int, video_h: int, fps: float
) -> None:
    """Update sequence resolution and frame rate in-place.

    XPaths below match the element names in base_pretty.xml (Task 3).
    If your template uses different paths, update these two lines.
    """
    fw = root.find(".//FrameWidth")
    fh = root.find(".//FrameHeight")
    fr = root.find(".//FrameRate")     # stores ticks-per-frame

    if fw is not None:
        fw.text = str(video_w)
    if fh is not None:
        fh.text = str(video_h)
    if fr is not None:
        # ticks-per-frame = ticks-per-second / fps
        fr.text = str(int(_PP_TICKS_PER_SEC / fps))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prproj_builder.py -v
```
Expected: all 10 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add stages/prproj_builder.py tests/test_prproj_builder.py
git commit -m "feat: add sequence settings update + unique ObjectID generator"
```

---

## Task 6: `prproj_builder.py` — clone and inject text clips on V2

**Files:**
- Modify: `stages/prproj_builder.py`
- Modify: `tests/test_prproj_builder.py`

> **Before coding:** use the values from Task 3 Step 4 (D–H, I).
> The element names in the fixture XML below must match what you found.
> `TEXT_CLIP_XML` is a simplified stand-in for the real clip subtree from `base_pretty.xml`.
> Replace element names (`<TextContent>`, `<PosX>`, `<ClipStart>`, `<ClipEnd>`) with the real ones from your inspection.

- [ ] **Step 1: Add tests**

Append to `tests/test_prproj_builder.py`:

```python
# Minimal text clip XML — replace element names with real ones from base_pretty.xml
TEXT_CLIP_XML = """<ClipTrackItem ObjectID="99">
  <TextContent>PLACEHOLDER</TextContent>
  <PosX>0.0</PosX>
  <PosY>0.0</PosY>
  <ClipStart>0</ClipStart>
  <ClipEnd>254016000</ClipEnd>
</ClipTrackItem>"""

# Minimal V2 track XML
TRACK_WITH_TEXT_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<PremiereData Version="3">
  <Project ObjectID="1">
    <Sequence ObjectID="2">
      <Settings>
        <FrameWidth>1920</FrameWidth>
        <FrameHeight>1080</FrameHeight>
        <FrameRate>8467200</FrameRate>
      </Settings>
      <VideoTracks>
        <Track ObjectID="10">
          <TrackItems>{TEXT_CLIP_XML}</TrackItems>
        </Track>
      </VideoTracks>
    </Sequence>
  </Project>
</PremiereData>"""

from models.text_overlay import TextOverlay


def _make_prproj_with_text_clip(tmp_path: Path) -> Path:
    raw = TRACK_WITH_TEXT_XML.encode("utf-8")
    path = tmp_path / "with_text.prproj"
    path.write_bytes(zlib.compress(raw))
    return path


def _overlay_at(start: float, end: float, text_ja: str = "テスト") -> TextOverlay:
    return TextOverlay(
        text_ko="테스트", text_ja=text_ja,
        bbox=(100, 200, 400, 80), start_sec=start, end_sec=end,
        confidence=0.9,
    )


def test_clone_text_clip_sets_text_content(tmp_path):
    path = _make_prproj_with_text_clip(tmp_path)
    root = prproj_builder.load_template(path)
    template_clip = root.find(".//ClipTrackItem[@ObjectID='99']")
    overlay = _overlay_at(1.0, 3.0, text_ja="こんにちは")
    cloned = prproj_builder.clone_text_clip(template_clip, overlay, 1920, 1080)
    assert cloned.find("TextContent").text == "こんにちは"


def test_clone_text_clip_sets_timecodes(tmp_path):
    path = _make_prproj_with_text_clip(tmp_path)
    root = prproj_builder.load_template(path)
    template_clip = root.find(".//ClipTrackItem[@ObjectID='99']")
    overlay = _overlay_at(2.0, 5.0)
    cloned = prproj_builder.clone_text_clip(template_clip, overlay, 1920, 1080)
    assert cloned.find("ClipStart").text == str(2 * 254016000)
    assert cloned.find("ClipEnd").text == str(5 * 254016000)


def test_clone_text_clip_gets_new_object_id(tmp_path):
    path = _make_prproj_with_text_clip(tmp_path)
    root = prproj_builder.load_template(path)
    template_clip = root.find(".//ClipTrackItem[@ObjectID='99']")
    overlay = _overlay_at(0.0, 1.0)
    cloned = prproj_builder.clone_text_clip(template_clip, overlay, 1920, 1080)
    assert cloned.get("ObjectID") != "99"
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_prproj_builder.py::test_clone_text_clip_sets_text_content \
       tests/test_prproj_builder.py::test_clone_text_clip_sets_timecodes \
       tests/test_prproj_builder.py::test_clone_text_clip_gets_new_object_id -v
```
Expected: `FAILED`.

- [ ] **Step 3: Implement `clone_text_clip` in `stages/prproj_builder.py`**

```python
import copy

# ─── IMPORTANT: Replace these element-name constants with the real values ───
# found by inspecting templates/base_pretty.xml in Task 3 Step 4 (F, G, H).
_TEXT_CONTENT_TAG = "TextContent"   # element holding the text string
_POS_X_TAG        = "PosX"         # element holding PP x-coordinate
_POS_Y_TAG        = "PosY"         # element holding PP y-coordinate
_CLIP_START_TAG   = "ClipStart"    # element holding start tick
_CLIP_END_TAG     = "ClipEnd"      # element holding end tick
# ────────────────────────────────────────────────────────────────────────────


def clone_text_clip(
    template_clip: ET.Element,
    overlay: TextOverlay,
    video_w: int,
    video_h: int,
) -> ET.Element:
    """Deep-clone the template text clip element and configure for this overlay."""
    cloned = copy.deepcopy(template_clip)
    cloned.set("ObjectID", new_object_id())

    pp_x, pp_y = bbox_to_pp_position(overlay.bbox, video_w, video_h)

    _set_text(cloned, _TEXT_CONTENT_TAG, overlay.text_ja)
    _set_text(cloned, _POS_X_TAG, str(pp_x))
    _set_text(cloned, _POS_Y_TAG, str(pp_y))
    _set_text(cloned, _CLIP_START_TAG, str(sec_to_ticks(overlay.start_sec)))
    _set_text(cloned, _CLIP_END_TAG, str(sec_to_ticks(overlay.end_sec)))

    return cloned


def _set_text(parent: ET.Element, tag: str, value: str) -> None:
    """Set text on a direct child element, creating it if absent."""
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    el.text = value
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_prproj_builder.py -v
```
Expected: all 13 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add stages/prproj_builder.py tests/test_prproj_builder.py
git commit -m "feat: add clone_text_clip for Essential Graphics injection"
```

---

## Task 7: `prproj_builder.py` — `build_prproj` full assembly

**Files:**
- Modify: `stages/prproj_builder.py`
- Modify: `tests/test_prproj_builder.py`

> **Before coding:** note the ObjectIDs of V2 track, A1 track, A2 track from Task 3 Step 4 (I, J, K).
> The `TEMPLATE_TEXT_CLIP_OBJ_ID` constant below must match the ObjectID of the sample text clip in `base.prproj`.

- [ ] **Step 1: Add integration test**

Append to `tests/test_prproj_builder.py`:

```python
from unittest.mock import patch


def test_build_prproj_creates_output_file(tmp_path):
    """build_prproj writes a valid compressed .prproj file."""
    template_path = _make_prproj_with_text_clip(tmp_path)

    # Fake video file — build_prproj probes it with cv2
    video = tmp_path / "pre_2_1.mp4"
    video.write_bytes(b"fake")

    overlays = [_overlay_at(0.5, 2.0, "こんにちは")]
    tts_map = {}  # no audio in this test
    output = tmp_path / "out.prproj"

    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap.return_value.get.side_effect = lambda prop: {
            3: 1920.0, 4: 1080.0, 5: 29.97,
        }.get(prop, 0.0)
        mock_cap.return_value.isOpened.return_value = True

        prproj_builder.build_prproj(template_path, video, overlays, tts_map, output)

    assert output.exists()
    # Verify it can be loaded back
    root = prproj_builder.load_template(output)
    assert root.tag == "PremiereData"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_prproj_builder.py::test_build_prproj_creates_output_file -v
```
Expected: `FAILED`.

- [ ] **Step 3: Add constants and `build_prproj` to `stages/prproj_builder.py`**

```python
# ─── IMPORTANT: Set these from Task 3 Step 4 (D, I, J, K) ───────────────────
# ObjectID of the sample text clip in base.prproj (search for "PLACEHOLDER")
TEMPLATE_TEXT_CLIP_OBJ_ID = "REPLACE_WITH_REAL_OBJECT_ID"
# ObjectID of the V2 video track (where text clips go)
V2_TRACK_OBJ_ID           = "REPLACE_WITH_REAL_OBJECT_ID"
# ObjectID of the A1 audio track (to mute)
A1_TRACK_OBJ_ID           = "REPLACE_WITH_REAL_OBJECT_ID"
# ObjectID of the A2 audio track (for TTS clips)
A2_TRACK_OBJ_ID           = "REPLACE_WITH_REAL_OBJECT_ID"
# ─────────────────────────────────────────────────────────────────────────────

# Fallback video dimensions if probe fails
_FALLBACK_W, _FALLBACK_H, _FALLBACK_FPS = 1920, 1080, 29.97


def _probe_video(video_path: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s — using fallback dimensions", video_path.name)
        return _FALLBACK_W, _FALLBACK_H, _FALLBACK_FPS
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or _FALLBACK_FPS
    cap.release()
    return (w or _FALLBACK_W), (h or _FALLBACK_H), fps


def _find_track(root: ET.Element, obj_id: str) -> ET.Element | None:
    return root.find(f".//*[@ObjectID='{obj_id}']")


def _get_or_create_track_items(track: ET.Element) -> ET.Element:
    items = track.find("TrackItems")
    if items is None:
        items = ET.SubElement(track, "TrackItems")
    return items


def _mute_track(track: ET.Element) -> None:
    """Set track volume/mute attribute to silence it."""
    # Exact attribute name varies — check base_pretty.xml for the mute/volume element.
    mute = track.find("MuteState")
    if mute is None:
        mute = ET.SubElement(track, "MuteState")
    mute.text = "true"


def build_prproj(
    template_path: Path,
    video_path: Path,
    overlays: list[TextOverlay],
    tts_map: dict[str, Path],
    output_path: Path,
) -> None:
    """Assemble a Premiere Pro project file from template + overlays + TTS clips.

    V1: original video (full duration)
    V2: Essential Graphics text clips (one per overlay with text_ja)
    A1: original audio — muted
    A2: TTS mp3 clips at their start times
    """
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path}\n"
            "Create it in Premiere Pro 2026 (see Task 3 in the implementation plan)."
        )

    root = load_template(template_path)
    video_w, video_h, fps = _probe_video(video_path)
    update_sequence_settings(root, video_w, video_h, fps)

    # ── Text clips on V2 ────────────────────────────────────────────────────
    v2_track = _find_track(root, V2_TRACK_OBJ_ID)
    template_clip = root.find(f".//*[@ObjectID='{TEMPLATE_TEXT_CLIP_OBJ_ID}']")

    if v2_track is not None and template_clip is not None:
        track_items = _get_or_create_track_items(v2_track)
        # Remove the placeholder template clip from the track
        existing = track_items.find(f"ClipTrackItem[@ObjectID='{TEMPLATE_TEXT_CLIP_OBJ_ID}']")
        if existing is not None:
            track_items.remove(existing)

        for overlay in overlays:
            if not overlay.text_ja:
                continue
            clip = clone_text_clip(template_clip, overlay, video_w, video_h)
            track_items.append(clip)
        logger.info("Injected %d text clips on V2", sum(1 for o in overlays if o.text_ja))
    else:
        logger.warning("V2 track or template clip not found — text overlays skipped. "
                       "Check V2_TRACK_OBJ_ID and TEMPLATE_TEXT_CLIP_OBJ_ID constants.")

    # ── Mute A1 ─────────────────────────────────────────────────────────────
    a1_track = _find_track(root, A1_TRACK_OBJ_ID)
    if a1_track is not None:
        _mute_track(a1_track)
    else:
        logger.warning("A1 track not found — Korean audio not muted. Check A1_TRACK_OBJ_ID.")

    # ── TTS clips on A2 ─────────────────────────────────────────────────────
    a2_track = _find_track(root, A2_TRACK_OBJ_ID)
    if a2_track is not None and tts_map:
        a2_items = _get_or_create_track_items(a2_track)
        placed = 0
        for overlay in overlays:
            if not overlay.text_ja or overlay.text_ja not in tts_map:
                continue
            audio_clip = _build_audio_clip(tts_map[overlay.text_ja], overlay)
            a2_items.append(audio_clip)
            placed += 1
        logger.info("Placed %d TTS clips on A2", placed)
    else:
        logger.warning("A2 track not found or no TTS map — audio skipped. Check A2_TRACK_OBJ_ID.")

    save_prproj(root, output_path)
    logger.info("Wrote %s", output_path)


def _build_audio_clip(mp3_path: Path, overlay: TextOverlay) -> ET.Element:
    """Build a minimal audio clip element referencing an mp3 file."""
    clip = ET.Element("ClipTrackItem", ObjectID=new_object_id())
    _set_text(clip, "FilePath",   str(mp3_path.resolve()))
    _set_text(clip, "ClipStart",  str(sec_to_ticks(overlay.start_sec)))
    _set_text(clip, "ClipEnd",    str(sec_to_ticks(overlay.end_sec)))
    return clip
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_prproj_builder.py -v
```
Expected: all 14 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add stages/prproj_builder.py tests/test_prproj_builder.py
git commit -m "feat: add build_prproj — full V1/V2/A1/A2 assembly"
```

---

## Task 8: Create `stages/export_prproj.py`

**Files:**
- Create: `stages/export_prproj.py`
- Create: `tests/test_export_prproj.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_export_prproj.py
import shutil
from pathlib import Path
from unittest.mock import patch
from models.text_overlay import TextOverlay
from stages import export_prproj


def _make_video(tmp_path: Path, name: str = "pre_2_1.mp4") -> Path:
    v = tmp_path / "input" / name
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_bytes(b"fake-video")
    return v


def _make_mp3(tmp_path: Path, name: str = "abc123.mp3") -> Path:
    mp3 = tmp_path / "tts" / name
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"fake-audio")
    return mp3


def _overlay() -> TextOverlay:
    return TextOverlay("테스트", "テスト", (0, 0, 100, 50), 0.0, 2.0, 0.9)


def test_run_creates_project_folder(tmp_path):
    video = _make_video(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()

    with patch("stages.export_prproj.prproj_builder.build_prproj"):
        export_prproj.run(video, [], {}, output_dir)

    assert (output_dir / "pre_2_1").is_dir()


def test_run_moves_original_video(tmp_path):
    video = _make_video(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()

    with patch("stages.export_prproj.prproj_builder.build_prproj"):
        export_prproj.run(video, [], {}, output_dir)

    assert (output_dir / "pre_2_1" / "pre_2_1.mp4").exists()
    assert not video.exists()


def test_run_copies_tts_mp3s(tmp_path):
    video = _make_video(tmp_path)
    mp3 = _make_mp3(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()
    tts_map = {"テスト": mp3}

    with patch("stages.export_prproj.prproj_builder.build_prproj"):
        export_prproj.run(video, [], tts_map, output_dir)

    assert (output_dir / "pre_2_1" / "tts" / "abc123.mp3").exists()


def test_run_returns_prproj_path(tmp_path):
    video = _make_video(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()

    with patch("stages.export_prproj.prproj_builder.build_prproj"):
        result = export_prproj.run(video, [], {}, output_dir)

    assert result == output_dir / "pre_2_1" / "pre_2_1.prproj"
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_export_prproj.py -v
```
Expected: `FAILED` — `ModuleNotFoundError`.

- [ ] **Step 3: Create `stages/export_prproj.py`**

```python
# stages/export_prproj.py
from __future__ import annotations
import logging
import shutil
from pathlib import Path

import config
from models.text_overlay import TextOverlay
from stages import prproj_builder

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = config.ROOT_DIR / "templates" / "base.prproj"


def run(
    video_path: Path,
    overlays: list[TextOverlay],
    tts_map: dict[str, Path],
    output_dir: Path,
) -> Path:
    """Assemble the Premiere Pro project for one video.

    Creates:
        output_dir/{stem}/
            {stem}.mp4          ← original video (moved)
            tts/*.mp3           ← TTS clips (copied)
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

    # Copy TTS mp3s into project folder (keep them next to .prproj for portability)
    tts_dest_dir = project_dir / "tts"
    tts_dest_dir.mkdir(exist_ok=True)
    dest_tts_map: dict[str, Path] = {}
    for text_ja, src_path in tts_map.items():
        dest = tts_dest_dir / src_path.name
        shutil.copy2(str(src_path), str(dest))
        dest_tts_map[text_ja] = dest
    logger.info("[%s] Copied %d TTS clip(s) → %s", stem, len(dest_tts_map), tts_dest_dir)

    # Build .prproj
    prproj_path = project_dir / f"{stem}.prproj"
    prproj_builder.build_prproj(
        _TEMPLATE_PATH, dest_video, overlays, dest_tts_map, prproj_path,
    )
    logger.info("[%s] Generated %s", stem, prproj_path)

    return prproj_path
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_export_prproj.py -v
```
Expected: all 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add stages/export_prproj.py tests/test_export_prproj.py
git commit -m "feat: add export_prproj stage — assembles project folder + .prproj"
```

---

## Task 9: Rewire `translate_video_gvi.py`

**Files:**
- Modify: `translate_video_gvi.py`

- [ ] **Step 1: Update the imports at the top of `translate_video_gvi.py`**

```python
# Remove this line:
from stages import translate, render
from stages import detect_gvi

# Replace with:
from stages import translate, tts, export_prproj
from stages import detect_gvi
```

- [ ] **Step 2: Replace `process_video` Stage 3 block**

Find and replace the Stage 3 block in `process_video()`:

```python
# ── REMOVE these lines ───────────────────────────────────────────────────────
    # ── Stage 3: Render ──────────────────────────────────────────────────────
    output_path = config.VIDEOS_OUTPUT_DIR / f"{stem}_ja.mp4"
    logger.info("[%s] Stage 3: rendering → %s", stem, output_path)
    render.run(video_path, translated, output_path)

    shutil.move(str(video_path), str(config.VIDEOS_OUTPUT_DIR / video_path.name))
    logger.info("✓ [%s] complete → %s", stem, output_path)
# ─────────────────────────────────────────────────────────────────────────────

# ── ADD these lines in their place ──────────────────────────────────────────
    # ── Stage 3: TTS (generate mp3s only) ────────────────────────────────────
    logger.info("[%s] Stage 3: generating TTS audio…", stem)
    tts_map = tts.run(translated, checkpoint_dir)

    # ── Stage 4: Export .prproj ───────────────────────────────────────────────
    logger.info("[%s] Stage 4: assembling Premiere Pro project…", stem)
    prproj_path = export_prproj.run(
        video_path, translated, tts_map, config.VIDEOS_OUTPUT_DIR,
    )
    logger.info("✓ [%s] complete → %s", stem, prproj_path)
# ─────────────────────────────────────────────────────────────────────────────
```

- [ ] **Step 3: Also update the "no overlays" early-exit path**

Find this block:

```python
    if not overlays:
        logger.info("[%s] No Korean text found — moving original to videos-done/", stem)
        shutil.move(str(video_path), str(config.VIDEOS_OUTPUT_DIR / video_path.name))
        return
```

Replace with:

```python
    if not overlays:
        logger.info("[%s] No Korean text found — moving original to videos-done/%s/", stem, stem)
        dest_dir = config.VIDEOS_OUTPUT_DIR / stem
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(video_path), str(dest_dir / video_path.name))
        return
```

Do the same for the `all translations failed` early-exit:

```python
    if all(not o.text_ja for o in translated):
        logger.warning("[%s] All translations failed — moving original, skipping render", stem)
        dest_dir = config.VIDEOS_OUTPUT_DIR / stem
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(video_path), str(dest_dir / video_path.name))
        return
```

- [ ] **Step 4: Run all tests to check for regressions**

```bash
pytest tests/ -v --ignore=tests/test_render.py
```
Expected: all tests `PASSED`. (`test_render.py` tests the old render stage which is no longer called — skip it.)

- [ ] **Step 5: Commit**

```bash
git add translate_video_gvi.py
git commit -m "feat: rewire pipeline to tts→export_prproj, remove _ja.mp4 output"
```

---

## Task 10: Fill in real ObjectIDs + smoke test

**Files:**
- Modify: `stages/prproj_builder.py` (fill in constants)
- Modify: `tests/test_prproj_builder.py` (update fixture element names if needed)

- [ ] **Step 1: Fill in the four ObjectID constants**

Using the values recorded in Task 3 Step 4 (D, I, J, K), edit `stages/prproj_builder.py`:

```python
TEMPLATE_TEXT_CLIP_OBJ_ID = "<paste ObjectID of PLACEHOLDER text clip>"
V2_TRACK_OBJ_ID           = "<paste ObjectID of V2 track>"
A1_TRACK_OBJ_ID           = "<paste ObjectID of A1 audio track>"
A2_TRACK_OBJ_ID           = "<paste ObjectID of A2 audio track>"
```

- [ ] **Step 2: Update element-name constants if needed**

If the real element names in `base_pretty.xml` differ from the placeholders (`TextContent`, `PosX`, `PosY`, `ClipStart`, `ClipEnd`), update:

```python
_TEXT_CONTENT_TAG = "<real element name from base_pretty.xml>"
_POS_X_TAG        = "<real element name>"
_POS_Y_TAG        = "<real element name>"
_CLIP_START_TAG   = "<real element name>"
_CLIP_END_TAG     = "<real element name>"
```

Also update the `MINIMAL_XML` and `TEXT_CLIP_XML` fixtures in `tests/test_prproj_builder.py` to match.

- [ ] **Step 3: Run all tests**

```bash
pytest tests/ -v --ignore=tests/test_render.py
```
Expected: all tests `PASSED`.

- [ ] **Step 4: Run pipeline on one real video**

```bash
cd /Users/jerryhong/Documents/video-translator
python translate_video_gvi.py --skip-detect
```
(Uses existing `checkpoints/pre_2_1/translations.json` — skips GVI detection.)

Expected output:
```
videos-done/
└── pre_2_1/
    ├── pre_2_1.prproj
    ├── pre_2_1.mp4
    └── tts/
        └── *.mp3
```

- [ ] **Step 5: Open `.prproj` in Premiere Pro 2026 and verify**

Open `videos-done/pre_2_1/pre_2_1.prproj` in PP2026 and confirm:
- V1: original video plays correctly
- V2: Japanese text clips appear at correct timecodes, text is editable
- A1: Korean audio is muted
- A2: TTS clips appear at their timecodes and play Japanese audio

- [ ] **Step 6: Commit**

```bash
git add stages/prproj_builder.py tests/test_prproj_builder.py
git commit -m "feat: wire real PP2026 ObjectIDs + smoke test verified"
```

---

## Task 11: Run all 4 videos + final commit

- [ ] **Step 1: Run full pipeline on all 4 videos**

```bash
python translate_video_gvi.py
```

Expected: 4 project folders created under `videos-done/`.

- [ ] **Step 2: Spot-check one project in PP2026**

Open each `.prproj`, verify track layout, test text editing on V2 clips.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete premiere pro export pipeline for all 4 videos"
```
