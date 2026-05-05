# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Paths ───────────────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).parent
VIDEOS_INPUT_DIR  = ROOT_DIR / "videos-to-be-done"
VIDEOS_OUTPUT_DIR = ROOT_DIR / "videos-done"
CHECKPOINTS_DIR   = ROOT_DIR / "checkpoints"

# ── API Keys ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.environ["ANTHROPIC_API_KEY"]   # raises KeyError immediately if unset
ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]  # raises KeyError immediately if unset

# ── OCR ─────────────────────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.6

# ── Graphic Zone Filter ─────────────────────────────────────────────────
# Overlays whose bbox center_y falls in this range are likely in-video UI
# graphics (product cards, pop-ups) — not speech subtitles — and are skipped.
GRAPHIC_ZONE_Y_MIN        = 0.25  # fraction of frame height
GRAPHIC_ZONE_Y_MAX        = 0.65
OVERLAY_MAX_AREA_FRACTION = 0.10

# ── Translation ─────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-opus-4-6"
MAX_TOKENS   = 4096

# ── TTS ─────────────────────────────────────────────────────────────────
# Gender is auto-detected from the video audio at runtime (pitch analysis).
# Override any single voice via ELEVENLABS_VOICE_ID to bypass auto-detection.
# Otherwise MALE / FEMALE voice IDs are used based on detected speaker pitch.
ELEVENLABS_VOICE_ID        = os.environ.get("ELEVENLABS_VOICE_ID", "")          # manual override
ELEVENLABS_VOICE_ID_FEMALE = os.environ.get("ELEVENLABS_VOICE_ID_FEMALE", "9BWtsMINqrJLrRacOk9x")  # Aria
ELEVENLABS_VOICE_ID_MALE   = os.environ.get("ELEVENLABS_VOICE_ID_MALE",   "pNInz6obpgDQGcFmaJgB")  # Adam
ELEVENLABS_MODEL           = "eleven_multilingual_v2"
