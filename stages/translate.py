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

_SYSTEM_PROMPT = (
    "You are a professional Japanese copywriter specialising in beauty and cosmetics "
    "advertising targeting Japanese girls aged 18–25. Translate each Korean phrase into "
    "natural, trendy Japanese suitable for a beauty ad campaign. Actively use contemporary "
    "beauty slang popular among late-teen to mid-twenties Japanese women — words like "
    "'うるつや', 'ぷるぷる', 'もちもち', 'つるつる', 'バズり', 'エモい', 'ガチ', "
    "'めちゃ盛れ', 'スキンケア沼' etc. where contextually appropriate.\n"
    "TRANSLATION FIDELITY RULES — follow strictly:\n"
    "1. Translate the FULL meaning faithfully. Do NOT shorten, omit, or summarise.\n"
    "2. Preserve tense and degree: hedge words like '~にくい', '~ようになった', "
    "'~やすい' must be kept — do not replace them with blunt negatives like 'なし'.\n"
    "3. Do NOT add exclamation marks (！) or intensifiers unless the original Korean "
    "explicitly contains them.\n"
    "4. Korean object marker 를/을 must map to Japanese を, not が.\n"
    "5. Keep brand names romanized — '브이쎄라' → 'V-THERA', '마미케어' → 'Mummy Care'. "
    "Do NOT leave Korean characters (Hangul) in the output; the font cannot render them.\n"
    "Respond ONLY with a JSON object mapping each Korean input to its Japanese translation. "
    "No explanations."
)

_RETRY_DELAYS = [2, 4]  # 3 total attempts: 1 initial + 2 retries (spec: "up to 3 attempts")


_NOT_CONFIRMED = "{Not confirmed}"


def _parse_json_response(text: str) -> dict:
    """Parse JSON from Claude response, stripping markdown code fences if present.

    Uses raw_decode to tolerate trailing prose the model sometimes appends
    after the JSON object.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    obj, _ = json.JSONDecoder().raw_decode(text.strip())
    return obj


def correct_korean(
    overlays: list[TextOverlay],
    client: anthropic.Anthropic,
) -> list[TextOverlay]:
    """Step 2a: send Korean text to Claude to fix OCR typos/broken syllables.
    Skips NOT_CONFIRMED entries. Returns overlays with corrected text_ko.
    """
    # Index unique, correctable texts
    unique: dict[str, str] = {}  # text_ko → numeric id
    for o in overlays:
        if o.text_ko != _NOT_CONFIRMED and o.text_ko not in unique:
            unique[o.text_ko] = str(len(unique))

    if not unique:
        return overlays

    id_to_original = {v: k for k, v in unique.items()}
    payload = {idx: text for text, idx in unique.items()}

    corrected_map: dict[str, str] = {}
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.MAX_TOKENS,
                system=_CORRECT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            )
            result = _parse_json_response(response.content[0].text)
            if isinstance(result, dict):
                corrected_map = result
                break
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Korean correction parse failed attempt %d: %s", attempt + 1, exc)
        except anthropic.APIError as exc:
            logger.warning("Korean correction API error attempt %d: %s", attempt + 1, exc)

    if not corrected_map:
        logger.warning("Korean correction failed — using original texts")
        return overlays

    # Build original_text → corrected_text lookup
    original_to_corrected = {
        id_to_original[idx]: corrected
        for idx, corrected in corrected_map.items()
        if idx in id_to_original
    }
    logger.info("Korean correction: %d texts corrected", len(original_to_corrected))

    return [
        TextOverlay(
            text_ko=original_to_corrected.get(o.text_ko, o.text_ko),
            text_ja=o.text_ja,
            bbox=o.bbox,
            start_sec=o.start_sec,
            end_sec=o.end_sec,
            confidence=o.confidence,
        )
        for o in overlays
    ]


def build_batch_payload(overlays: list[TextOverlay]) -> dict[str, str]:
    """Return {text_ko: ""} for all unique, translatable Korean strings."""
    seen: dict[str, str] = {}
    for o in overlays:
        if o.text_ko not in seen and o.text_ko != _NOT_CONFIRMED:
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
            result = _parse_json_response(response.content[0].text)
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


_SPEECH_CORRECT_PROMPT = (
    "You are a Korean speech-to-text post-processor for beauty and cosmetics video content. "
    "You receive a JSON object mapping numeric IDs to Korean ASR transcripts. "
    "Fix the following issues in each transcript:\n"
    "1. ASR homophone errors — Korean ASR often confuses similar-sounding syllables; "
    "correct to the most contextually likely word for a beauty ad.\n"
    "   COMMON ERRORS:\n"
    "   • ASR frequently mishears '브이쎄라'/'V세라' as '보이스', '이스라엘', '그리스', "
    "'비스라엘', '이비스', '브이스라', or other similar-sounding Korean words — "
    "replace ALL such mishearings with 'V세라'.\n"
    "   • '알고 나서도' appearing after a brand name is often a mishearing of '알고난 후' — "
    "check surrounding context; if the phrase is part of a before/after pattern "
    "('알기 전 / 알고난 후'), correct it to '알고난 후'.\n"
    "2. Brand names: normalize '브이쎄라'/'비세라'/'브이세라'/'보이스라'/'브이스' → 'V세라'; "
    "keep brand names intact.\n"
    "3. Beauty/cosmetics terms: '경락', '리프팅', '사각턱', '브이라인', '중주파' — "
    "correct any mis-transcribed variants.\n"
    "4. Keep natural Korean slang intact ('찐', '극복템', '갓성비', '개이득', '땡겨짐' etc.) — "
    "only fix clear transcription errors, not intentional informal speech.\n"
    "CRITICAL — PRESERVE REPETITIONS: Korean beauty ad speakers intentionally repeat key "
    "phrases 2-3 times for emphasis (e.g., 'V세라 알기 전, V세라 알고 난 후' repeated). "
    "These are REAL speech patterns, not transcription errors. "
    "PRESERVE ALL REPETITIONS exactly — do NOT consolidate or summarize repeated phrases.\n"
    "If a transcript needs no changes, return it as-is.\n"
    "Preserve the numeric ID keys exactly.\n"
    "Respond ONLY with a JSON object mapping the same numeric IDs to corrected Korean strings. "
    "No explanations."
)

_SPEECH_TRANSLATE_PROMPT = (
    "You are a professional Korean-to-Japanese translator for beauty and cosmetics "
    "video content. You receive a JSON object whose keys are numeric IDs and values are "
    "Korean speech transcripts. Translate each Korean value into natural, spoken Japanese "
    "suitable for voice-over narration. The translations will be read aloud by TTS, "
    "so use natural conversational Japanese — not written/formal style.\n"
    "RULES:\n"
    "1. Translate the FULL meaning faithfully. Do NOT shorten or omit.\n"
    "2. Use natural spoken Japanese (です/ます or casual depending on context).\n"
    "3. Keep brand names romanized — '브이쎄라' → 'V-THERA' etc.\n"
    "4. Do NOT leave Korean characters (Hangul) in the output.\n"
    "5. Preserve the numeric ID keys exactly — do NOT change them.\n"
    "Respond ONLY with a JSON object using the same numeric IDs as keys and Japanese "
    "translations as values. No explanations."
)


def _correct_speech_korean(
    texts: list[str],
    client: anthropic.Anthropic,
    ocr_hints: list[str] | None = None,
) -> dict[str, str]:
    """Fix ASR errors in Korean transcripts using Claude.

    Returns {original_text: corrected_text}. Falls back to identity map on failure.
    ocr_hints: phrases from high-confidence OCR that the speaker likely said.
    """
    if not texts:
        return {}

    system_prompt = _SPEECH_CORRECT_PROMPT
    if ocr_hints:
        hint_block = (
            "\n\nCONTEXT — RECURRING ON-SCREEN CAPTIONS (high-confidence OCR, appear 2+ times):\n"
            "These phrases were detected as subtitles/captions that repeat in the video. "
            "The speaker says each of them multiple times. "
            "Use these as ground-truth references when correcting garbled ASR words — "
            "if the ASR produces something that sounds like one of these, correct it to match exactly:\n"
            + "\n".join(f"  - {p}" for p in ocr_hints)
        )
        system_prompt = _SPEECH_CORRECT_PROMPT + hint_block

    id_to_text = {str(i): t for i, t in enumerate(texts)}
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=max(config.MAX_TOKENS, 8192),
                system=system_prompt,
                messages=[{"role": "user", "content": json.dumps(id_to_text, ensure_ascii=False)}],
            )
            result = _parse_json_response(response.content[0].text)
            if isinstance(result, dict):
                mapping: dict[str, str] = {}
                for idx, corrected in result.items():
                    original = id_to_text.get(str(idx))
                    if original:
                        if corrected != original:
                            logger.info("Speech correction: %s… → %s…", original[:50], corrected[:50])
                        mapping[original] = corrected
                return mapping
        except (json.JSONDecodeError, ValueError, anthropic.APIError) as exc:
            logger.warning("Speech Korean correction failed attempt %d: %s", attempt + 1, exc)

    logger.warning("All speech correction retries failed — using raw Whisper transcription")
    return {t: t for t in texts}


def translate_speech(
    segments: list[dict],
    checkpoint_dir: Path,
) -> list[dict]:
    """Translate Korean speech segments to Japanese for TTS.

    Accepts list of dicts with {text_ko, text_ja, start_sec, end_sec, confidence}.
    Returns same list with text_ja filled in (text_ko replaced with corrected Korean).
    Writes speech_translations.json checkpoint.
    """
    if not segments:
        return segments

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Load recurring OCR phrases as context hints for ASR correction.
    # Phrases detected more than once in OCR are likely on-screen captions
    # that the speaker is also saying — high-quality ground truth.
    ocr_hints: list[str] = []
    det_path = checkpoint_dir / "detections.json"
    if det_path.exists():
        dets = json.loads(det_path.read_text(encoding="utf-8"))
        phrase_counts: dict[str, int] = {}
        for d in dets:
            txt = (d.get("text_ko") or "").strip()
            if txt:
                phrase_counts[txt] = phrase_counts.get(txt, 0) + 1
        ocr_hints = [p for p, c in phrase_counts.items() if c >= 2]

    # Step 1: Correct ASR errors (slang, homophones, brand names) before translation.
    unique_raw: list[str] = []
    seen: set[str] = set()
    for s in segments:
        if s["text_ko"] and s["text_ko"] not in seen:
            unique_raw.append(s["text_ko"])
            seen.add(s["text_ko"])

    if not unique_raw:
        return segments

    correction_map = _correct_speech_korean(unique_raw, client, ocr_hints=ocr_hints)
    corrected_segments = [
        {**s, "text_ko": correction_map.get(s["text_ko"], s["text_ko"])}
        for s in segments
    ]

    # Step 2: Translate corrected Korean → Japanese using numeric IDs as keys
    # (numeric keys prevent Claude from accidentally translating Korean key text).
    unique_corrected: list[str] = []
    seen2: set[str] = set()
    for s in corrected_segments:
        if s["text_ko"] and s["text_ko"] not in seen2:
            unique_corrected.append(s["text_ko"])
            seen2.add(s["text_ko"])

    id_to_ko: dict[str, str] = {str(i): ko for i, ko in enumerate(unique_corrected)}
    ko_to_ja: dict[str, str] = {}

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=max(config.MAX_TOKENS, 8192),
                system=_SPEECH_TRANSLATE_PROMPT,
                messages=[{"role": "user", "content": json.dumps(
                    {i: ko for i, ko in id_to_ko.items()},
                    ensure_ascii=False,
                )}],
            )
            result = _parse_json_response(response.content[0].text)
            if isinstance(result, dict):
                for idx, ja in result.items():
                    ko = id_to_ko.get(str(idx))
                    if ko and ja:
                        ko_to_ja[ko] = ja
                break
        except (json.JSONDecodeError, ValueError, anthropic.APIError) as exc:
            logger.warning("Speech translation failed attempt %d: %s", attempt + 1, exc)
            if attempt == len(_RETRY_DELAYS):
                logger.error("All speech translation retries failed")
                return segments

    translated = []
    for corr_s in corrected_segments:
        translated.append({
            **corr_s,
            "text_ja": ko_to_ja.get(corr_s["text_ko"], ""),
        })

    out_path = checkpoint_dir / "speech_translations.json"
    out_path.write_text(
        json.dumps(translated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Stage 2: wrote %d speech translations to %s", len(translated), out_path)
    return translated


def run(detections: list[TextOverlay], checkpoint_dir: Path) -> list[TextOverlay]:
    """Stage 2: correct Korean text then translate to Japanese. Writes translations.json."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    logger.info("Stage 2a: correcting Korean text with Claude...")
    corrected = correct_korean(detections, client)
    translated = translate_batch(corrected, client)

    out_path = checkpoint_dir / "translations.json"
    out_path.write_text(
        json.dumps([o.to_dict() for o in translated], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Stage 2: wrote %d translations to %s", len(translated), out_path)
    return translated
