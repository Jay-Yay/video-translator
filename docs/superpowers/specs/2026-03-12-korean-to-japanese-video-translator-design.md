# Korean-to-Japanese Video Translator — Design Spec

**Date:** 2026-03-12
**Status:** In Review

---

## Overview

A staged Python pipeline that automatically detects hard-coded Korean text in `.mp4` advertisement videos, translates it into trendy Japanese beauty-ad copy using the Anthropic Claude API, and composites solid-color text overlay boxes onto the original video — preserving all audio and non-text visuals.

---

## Goals

- Process all `.mp4` files in `videos-to-be-done/` automatically in a single command
- Replace Korean on-screen text with Japanese translations using a solid box overlay
- Preserve original audio track and all background visuals unchanged
- Support resumable execution via per-video JSON checkpoints
- Produce H.264-encoded output in `videos-done/`

## Non-Goals

- Voiceover / audio translation
- Subtitle/SRT file export
- Inpainting or background reconstruction
- Cloud OCR (all OCR runs locally)

---

## Architecture

### File Layout

```
video-translator/
├── videos-to-be-done/           # INPUT: drop .mp4 files here
├── videos-done/                 # OUTPUT: translated .mp4s + moved originals
├── translate_video.py           # main entry point
├── config.py                    # all configurable constants & API key loading
├── stages/
│   ├── detect.py                # Stage 1: scene detection + PaddleOCR → raw_detections.json
│   ├── deduplicate.py           # Stage 1.5: cluster & merge → detections.json
│   ├── translate.py             # Stage 2: Claude batch translation → translations.json
│   └── render.py                # Stage 3: MoviePy overlay compositing
├── models/
│   └── text_overlay.py          # TextOverlay dataclass (shared schema)
├── utils/
│   └── font_scaler.py           # dynamic font sizing algorithm
├── checkpoints/
│   └── <video_stem>/
│       ├── raw_detections.json  # Stage 1 raw output (pre-deduplication)
│       ├── detections.json      # Stage 1.5 output (deduplicated, merged)
│       └── translations.json    # Stage 2 output (text_ja populated)
├── fonts/
│   └── NotoSansJP-Bold.otf      # Download from: https://fonts.google.com/noto/specimen/Noto+Sans+JP
└── requirements.txt
```

### Main Loop (`translate_video.py`)

```
# Startup validation (runs before any video processing)
assert FONT_PATH.exists(), f"Font not found: {FONT_PATH}. Download NotoSansJP-Bold.otf from https://fonts.google.com/noto/specimen/Noto+Sans+JP"

for each .mp4 in videos-to-be-done/:
    checkpoint_dir = checkpoints/<stem>/
    os.makedirs(checkpoint_dir, exist_ok=True)   # create per-video checkpoint dir

    # --skip-* flags take precedence over --force for their respective stages
    if not --skip-detect:
        if raw_detections.json missing or --force:
            run Stage 1 (detect) → raw_detections.json
        if detections.json missing or --force:
            run Stage 1.5 (deduplicate) → detections.json
    else:
        # --skip-detect skips BOTH Stage 1 and Stage 1.5
        # If only raw_detections.json exists (Stage 1 done, Stage 1.5 not), this
        # video will be skipped — omit --skip-detect to let checkpoint logic handle it
        if detections.json missing: log warning, skip this video, continue

    if detections.json is empty (zero overlays):
        log "No Korean text found in <stem>, skipping translation and render"
        move source .mp4 → videos-done/<stem>.mp4  # overwrite silently if exists
        continue

    if not --skip-translate:
        if translations.json missing or --force:
            run Stage 2 (translate) → translations.json
    else:
        if translations.json missing: log warning, skip this video, continue

    if all text_ja fields in translations.json are empty:
        log "All translations failed for <stem>, skipping render (no overlay to apply)"
        move source .mp4 → videos-done/<stem>.mp4  # overwrite silently if exists
        continue

    run Stage 3 (render) → videos-done/<stem>_ja.mp4  # overwrite silently if exists
    move source .mp4 → videos-done/<stem>.mp4         # overwrite silently if exists
    log "✓ <stem> complete"
```

**CLI flags:**
- `--force` — re-run all stages even if checkpoints exist. `--skip-*` flags take precedence over `--force` for their respective stages.
- `--skip-detect` — skip both Stage 1 and Stage 1.5. If `detections.json` is missing for a video, log a warning and skip that video (do not crash).
- `--skip-translate` — skip Stage 2. If `translations.json` is missing for a video, log a warning and skip that video (do not crash).

**Error handling:** Exceptions in any stage for a given video are caught, logged, and execution continues with the next video. Failed videos remain in `videos-to-be-done/` for retry.

---

## Shared Data Model

### `models/text_overlay.py` — `TextOverlay` dataclass

```python
@dataclass
class TextOverlay:
    text_ko: str                    # original Korean text
    text_ja: str                    # translated Japanese (empty string until Stage 2)
    bbox: tuple[int, int, int, int] # (x, y, width, height) in pixels — axis-aligned rect
    start_sec: float                # overlay start time in seconds
    end_sec: float                  # overlay end time in seconds
    confidence: float               # OCR confidence score (retained for debugging/tuning)
```

`translations.json` uses the same `TextOverlay` schema as `detections.json`, with `text_ja` fields populated. Overlays where translation permanently failed retain `text_ja = ""` and are silently skipped during render (no box drawn).

**Serialisation:** Use `dataclasses.asdict()` for serialisation. `bbox` is written as a JSON array and must be explicitly converted back to `tuple[int,int,int,int]` on deserialisation. Implement a `TextOverlay.from_dict(d: dict) -> TextOverlay` classmethod that handles this conversion.

---

## Configuration (`config.py`)

All tuneable constants in one place. Stages import from this module only.

| Setting | Default | Description |
|---|---|---|
| `VIDEOS_INPUT_DIR` | `videos-to-be-done/` | Input folder |
| `VIDEOS_OUTPUT_DIR` | `videos-done/` | Output folder |
| `CHECKPOINTS_DIR` | `checkpoints/` | Per-video checkpoint folder |
| `FONT_PATH` | `fonts/NotoSansJP-Bold.otf` | Japanese font (see font download note) |
| `ANTHROPIC_API_KEY` | `os.environ["ANTHROPIC_API_KEY"]` | Loaded from environment |
| `SCENE_THRESHOLD` | `27.0` | PySceneDetect ContentDetector sensitivity (lower = more scenes) |
| `MIN_SCENE_LEN` | `15` | Minimum frames between scene cuts |
| `OCR_CONFIDENCE_THRESHOLD` | `0.6` | Minimum PaddleOCR confidence to keep a detection |
| `BBOX_IOU_THRESHOLD` | `0.5` | IoU overlap to consider two boxes the same region |
| `TEXT_SIMILARITY_RATIO` | `0.85` | Fuzzy match threshold for same-text grouping |
| `TEMPORAL_GAP_SEC` | `0.5` | Max gap (seconds) to merge temporally adjacent same-text entries |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Anthropic model ID — `claude-opus-4-6` is the Claude Opus 4.6 model, confirmed valid. Always verify the latest available model at docs.anthropic.com/en/docs/about-claude/models before implementation |
| `MAX_TOKENS` | `1024` | Max tokens for Claude response |
| `BOX_COLOR` | `(0, 0, 0)` | Overlay box background (RGB black) |
| `TEXT_COLOR` | `(255, 255, 255)` | Overlay text color (RGB white) |
| `BOX_PADDING` | `4` | Pixels of padding inside bbox |
| `FONT_SIZE_MIN` | `8` | Minimum font size |
| `FONT_SIZE_MAX` | `72` | Maximum font size |
| `FONT_SIZE_WRAP_THRESHOLD` | `12` | Font size below which 2-line wrapping is attempted before further shrinking |
| `OUTPUT_CODEC` | `libx264` | H.264 video encoding |
| `OUTPUT_AUDIO_CODEC` | `aac` | Audio codec for output (preserves original audio track) |
| `OUTPUT_FPS` | `None` | When `None`, uses `original_clip.fps`; set an explicit value to override |

**Font download:** `NotoSansJP-Bold.otf` — download from https://fonts.google.com/noto/specimen/Noto+Sans+JP and place in `fonts/`. A startup check in `translate_video.py` will exit with a clear error if the file is missing.

---

## Stage 1 — Detection (`stages/detect.py`)

**Input:** `.mp4` file path
**Output:** `checkpoints/<stem>/raw_detections.json`

### Pipeline

1. **PySceneDetect `ContentDetector`** splits the video into scenes using `SCENE_THRESHOLD` and `MIN_SCENE_LEN`
2. For each scene, extract the midpoint frame as a keyframe (via OpenCV `cap.read()`)
3. **PaddleOCR** (`lang='korean'`) runs on each keyframe → returns list of `(polygon_pts, text, confidence)` where `polygon_pts` is a list of 4 corner points `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`
4. **Convert PaddleOCR polygon to axis-aligned rect:**
   ```
   x = int(min(pt[0] for pt in polygon_pts))
   y = int(min(pt[1] for pt in polygon_pts))
   w = int(max(pt[0] for pt in polygon_pts)) - x
   h = int(max(pt[1] for pt in polygon_pts)) - y
   bbox = (x, y, w, h)
   ```
   This bounding rect approach safely handles any slight rotation in the detected polygon.
5. Filter: discard detections with `confidence < OCR_CONFIDENCE_THRESHOLD`
6. Filter: discard detections containing no Hangul characters. Check against all three Hangul Unicode blocks:
   - Hangul Syllables: `\uAC00–\uD7A3` (primary block)
   - Hangul Jamo: `\u1100–\u11FF` (decomposed characters, sometimes returned by OCR)
   - Hangul Compatibility Jamo: `\u3130–\u318F`
7. Map each detection to a `TextOverlay(text_ko, bbox, start_sec=scene_start, end_sec=scene_end, text_ja="", confidence=confidence)`
8. Serialise list to `raw_detections.json` using `dataclasses.asdict()`

---

## Stage 1.5 — Deduplication (`stages/deduplicate.py`)

**Input:** `checkpoints/<stem>/raw_detections.json`
**Output:** `checkpoints/<stem>/detections.json` (new file — `raw_detections.json` is preserved unchanged)

### Algorithm

Entries are first sorted by `start_sec` ascending before processing begins.

**Pass 1 — Spatial merge (greedy chain, consecutive-forward):**
- Iterate through sorted entries with index `i`. For each `i`, perform a greedy forward scan: compare `entries[i]` against `entries[i+1]`, `entries[i+2]`, etc., stopping the scan as soon as a candidate pair fails either the IoU or text similarity threshold (i.e., the chain breaks — do not skip over a non-matching entry to find a later matching one).
- For each pair `(a, b)` encountered in the forward scan:
  - Compute `IoU(a.bbox, b.bbox) > BBOX_IOU_THRESHOLD`
  - Compute `SequenceMatcher(None, a.text_ko, b.text_ko).ratio() > TEXT_SIMILARITY_RATIO`
  - If both pass: extend `a.end_sec = max(a.end_sec, b.end_sec)`, mark `b` for removal, continue scanning forward from `b`'s position
  - If either fails: stop the forward scan for entry `i`, advance `i` to the next unmarked entry
- After full pass: remove all marked entries

**Pass 2 — Temporal grouping (same normalised text, video-wide):**
- Group remaining entries by `text_ko.strip()`
- Within each group (sorted by `start_sec`): merge entries where `next.start_sec - current.end_sec < TEMPORAL_GAP_SEC`
  - Merged entry uses: the `bbox` of the highest-`confidence` entry in the group (most reliable OCR position), `start_sec = min(group)`, `end_sec = max(group)`, `confidence = max(group)`
  - Note: using the highest-confidence bbox minimises placement error for text that shifts slightly across scenes due to camera motion
- Collect all groups back into a flat list, sorted by `start_sec`

**IoU formula:**
```
intersection_x1 = max(a.x, b.x)
intersection_y1 = max(a.y, b.y)
intersection_x2 = min(a.x+a.w, b.x+b.w)
intersection_y2 = min(a.y+a.h, b.y+b.h)
# Both dimension clamps applied independently before multiplying —
# a negative dimension multiplied by a positive one would produce nonsensical negative area
intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
union_area = a.w*a.h + b.w*b.h - intersection_area
IoU = intersection_area / union_area  (0.0 if union_area == 0)
```

**Fuzzy matching:** `difflib.SequenceMatcher(None, a.text_ko, b.text_ko).ratio()` (stdlib, no extra dependency)

---

## Stage 2 — Translation (`stages/translate.py`)

**Input:** `checkpoints/<stem>/detections.json`
**Output:** `checkpoints/<stem>/translations.json`

`translations.json` shares the same `TextOverlay` list schema as `detections.json`, with `text_ja` fields populated. Any overlay where translation permanently fails retains `text_ja = ""`.

### API Call Design

Single batch call to Anthropic Claude with all unique `text_ko` strings:

**System prompt:**
> You are a professional Japanese copywriter specialising in beauty and cosmetics advertising targeting Japanese girls aged 18–25. Translate each Korean phrase into natural, trendy Japanese suitable for a beauty ad campaign. Actively use contemporary beauty slang popular among late-teen to mid-twenties Japanese women — words like 'うるつや', 'ぷるぷる', 'もちもち', 'つるつる', 'バズり', 'エモい', 'ガチ', 'めちゃ盛れ', 'スキンケア沼' etc. where contextually appropriate. Keep translations concise — they must fit within the same space as the original text. Respond ONLY with a JSON object mapping each Korean input to its Japanese translation. No explanations.

**User message:** JSON object with empty string values for each unique `text_ko` string. Values are always `""` (empty string placeholders); keys are the unique Korean strings to translate. Example:
```json
{ "촉촉한 수분감": "", "피부 트러블 케어": "", "지금 바로 구매하기": "" }
```

**Expected response:**
```json
{ "촉촉한 수분감": "うるつやもちもち肌", "피부 트러블 케어": "肌トラブルケア", "지금 바로 구매하기": "今すぐGET" }
```

### Error Handling

- Parse response as JSON; if malformed: retry with exponential backoff (delays: 2s, 4s, 8s), up to 3 attempts
- After 3 failed attempts for the full batch: fall back to per-string individual API calls for any missing/untranslated keys
- Individual fallback calls also use the same 3-attempt exponential backoff (delays: 2s, 4s, 8s)
- A string that fails all 3 individual attempts retains `text_ja = ""` (logged as a warning; no overlay rendered for that string)
- Map successful translations back to all `TextOverlay` objects sharing the same `text_ko`

---

## Utility — Font Scaling (`utils/font_scaler.py`)

### `fit_font_size(text, bbox_w, bbox_h, font_path, padding) → (size: int, font: PIL.ImageFont.FreeTypeFont)`

Iterates from `FONT_SIZE_MAX` down to `FONT_SIZE_MIN`, returning the largest size where `text_w ≤ usable_w AND text_h ≤ usable_h` (where `usable_w = bbox_w - padding*2`, `usable_h = bbox_h - padding*2`). Both `size` and `font` are returned — `font` is a `PIL.ImageFont.truetype(font_path, size)` instance. Returns `(FONT_SIZE_MIN, PIL.ImageFont.truetype(font_path, FONT_SIZE_MIN))` as last resort even if it overflows.

### `wrap_text(text, bbox_w, bbox_h, font_path, padding) → (wrapped_text: str, size: int, font: PIL.ImageFont.FreeTypeFont)`

1. Try single-line fit via `fit_font_size`
2. If resulting size ≥ `FONT_SIZE_WRAP_THRESHOLD` (default: 12): return `(text, size, font)` — single-line result
3. Otherwise: split text at character midpoint into 2 lines with `\n` separator (Japanese has no word spaces, so character midpoint is the correct split strategy)
4. Re-run `fit_font_size` on the `\n`-joined 2-line string with `bbox_h` used as-is. Inside `fit_font_size`, height must be measured using `PIL.ImageFont.getbbox()` on the **full multi-line string** (not per-line) so the combined height of both lines plus line spacing is correctly constrained against `usable_h`. Do not measure a single line and halve the allowance — this would incorrectly double the permitted font size.
5. Return whichever (`wrapped_text`, `size`, `font`) produces the larger font size

The first return value `wrapped_text` may differ from the input `text` (a newline may be inserted). Callers must use `wrapped_text` for rendering, not the original input string.

`FONT_SIZE_WRAP_THRESHOLD` is configurable in `config.py` — it represents the minimum "acceptable" single-line font size before wrapping is preferred.

---

## Stage 3 — Render (`stages/render.py`)

**Input:** original `.mp4` + `checkpoints/<stem>/translations.json`
**Output:** `videos-done/<stem>_ja.mp4` (silently overwritten if already exists)

### Compositing Pipeline

1. Load original video as `MoviePy.VideoFileClip` (audio track preserved)
2. For each `TextOverlay` in `translations.json` where `text_ja != ""`:
   a. Create blank RGBA PIL Image sized `(bbox_w, bbox_h)`
   b. Draw filled rectangle using `BOX_COLOR`
   c. Call `wrap_text()` → returns `(wrapped_text, size, font)`
   d. Draw `wrapped_text` centered in box using `TEXT_COLOR` (use the returned `wrapped_text`, not `overlay.text_ja` directly, as it may contain a `\n` line break)
   e. Convert PIL Image → numpy array → `MoviePy.ImageClip`
   f. `.set_position((bbox_x, bbox_y))`, `.set_start(start_sec)`, `.set_end(end_sec)`
3. `CompositeVideoClip([original_clip, *overlay_clips])`
4. `write_videofile(output_path, codec=OUTPUT_CODEC, audio_codec=OUTPUT_AUDIO_CODEC, fps=OUTPUT_FPS or original_clip.fps)`

**Overlays with `text_ja == ""`** (translation failed) are silently skipped — no box is drawn.

**Output filename:** `<stem>_ja.mp4` in `videos-done/`
**Original move:** source `.mp4` is moved to `videos-done/<stem>.mp4` (overwritten silently if already exists)

---

## Technology Stack

| Purpose | Library | Notes |
|---|---|---|
| Scene detection | `scenedetect` (PySceneDetect) | ContentDetector |
| OCR | `paddlepaddle` + `paddleocr` | Korean language model |
| Frame extraction | `opencv-python` | Already in base conda env |
| Translation | `anthropic` | Model: `claude-opus-4-6` (confirmed valid; verify at docs.anthropic.com/en/docs/about-claude/models) |
| Video compositing | `moviepy` | Validated against MoviePy 1.x API (`set_position`, `set_start`, `set_end`) |
| Image drawing | `Pillow` | Already in base conda env |
| Fuzzy text match | `difflib` | stdlib, no extra dep |
| Font | `NotoSansJP-Bold.otf` | Download from https://fonts.google.com/noto/specimen/Noto+Sans+JP |

---

## Requirements (`requirements.txt`)

```
# Validated major versions noted — pin as needed for reproducibility
paddlepaddle          # >=2.5
paddleocr             # >=2.7
scenedetect[opencv]   # >=0.6
moviepy               # >=1.0,<2.0  (spec uses 1.x API)
anthropic             # >=0.25
Pillow                # >=10.0
numpy                 # >=1.24
opencv-python         # >=4.8
```

---

## Execution

```bash
# 1. Set API key
export ANTHROPIC_API_KEY=your_key_here

# 2. Download font
# Place NotoSansJP-Bold.otf into fonts/
# Download from: https://fonts.google.com/noto/specimen/Noto+Sans+JP

# 3. Drop input videos
# Place .mp4 files into videos-to-be-done/

# 4. Install dependencies (recommend using conda base env)
pip install -r requirements.txt

# 5. Run full pipeline
python translate_video.py

# Resume: skip OCR, redo translation + render
python translate_video.py --skip-detect

# Resume: skip OCR + translation, redo render only
python translate_video.py --skip-detect --skip-translate

# Force full re-run (ignore all checkpoints)
python translate_video.py --force
```

---

## Data Flow Summary

```
videos-to-be-done/<stem>.mp4
    │
    ▼ Stage 1: detect.py
checkpoints/<stem>/raw_detections.json   (raw TextOverlay list, text_ja="")
    │
    ▼ Stage 1.5: deduplicate.py
checkpoints/<stem>/detections.json       (deduplicated, merged time ranges)
    │
    ├─ [if empty: log "no Korean text", move original, skip]
    │
    ▼ Stage 2: translate.py
checkpoints/<stem>/translations.json     (TextOverlay list, text_ja populated)
    │
    ├─ [if all text_ja="": log "all translations failed", move original, skip]
    │
    ▼ Stage 3: render.py
videos-done/<stem>_ja.mp4               (final H.264 output, overlays applied)
videos-done/<stem>.mp4                  (original moved here)
```
