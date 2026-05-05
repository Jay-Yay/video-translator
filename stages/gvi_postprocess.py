# stages/gvi_postprocess.py
from __future__ import annotations
import logging
from pathlib import Path

import config
from models.text_overlay import TextOverlay

logger = logging.getLogger(__name__)


def remove_repeated_phrases(text: str) -> str:
    """Remove verbatim repeated phrases caused by GVI annotation artifacts.

    GVI sometimes concatenates duplicate OCR readings of the same word/phrase
    within one annotation segment. E.g.:
      "관리 전 관리 전 1달 후 1달 후" -> "관리 전 1달 후"
      "다만 효과가 좋아서 그런지 다만 효과가 좋아서 그런지 비싸게 파는데"
        -> "다만 효과가 좋아서 그런지 비싸게 파는데"

    Algorithm: find the shortest N-gram (N>=1) that appears at two or more
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
    """Remove overlays whose bbox area exceeds OVERLAY_MAX_AREA_FRACTION of the frame."""
    max_area = frame_w * frame_h * config.OVERLAY_MAX_AREA_FRACTION
    return [o for o in overlays if o.bbox[2] * o.bbox[3] < max_area]


def suppress_subfragments(overlays: list[TextOverlay]) -> list[TextOverlay]:
    """Remove overlays that are spatial+temporal sub-fragments of a larger overlay."""
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
            time_contained = (
                b.start_sec >= a.start_sec - 0.2
                and b.end_sec <= a.end_sec + 0.2
            )
            if time_contained and _iou(a.bbox, b.bbox) > 0.3:
                suppressed.add(j)

    return [o for i, o in enumerate(overlays) if i not in suppressed]


def suppress_text_subfragments(overlays: list[TextOverlay]) -> list[TextOverlay]:
    """Remove overlays whose Korean text is a sub-string of a concurrent longer overlay.

    Addresses cases where GVI emits both a full subtitle annotation and a
    shorter fragment of the same text with a slightly different bounding box,
    causing stacked duplicate overlays.
    """
    by_length = sorted(
        range(len(overlays)),
        key=lambda i: len(overlays[i].text_ko),
        reverse=True,
    )
    suppressed: set[int] = set()

    for pos, i in enumerate(by_length):
        if i in suppressed:
            continue
        a = overlays[i]
        for j in by_length[pos + 1:]:
            if j in suppressed:
                continue
            b = overlays[j]
            if b.text_ko and b.text_ko in a.text_ko:
                time_overlap = (
                    b.start_sec < a.end_sec + 0.5
                    and b.end_sec > a.start_sec - 0.5
                )
                if time_overlap:
                    suppressed.add(j)

    return [o for i, o in enumerate(overlays) if i not in suppressed]


def merge_recurring_overlays(
    overlays: list[TextOverlay],
    max_gap_sec: float = 1.0,
    min_iou: float = 0.5,
) -> list[TextOverlay]:
    """Collapse repeated detections of a persistent on-screen label.

    GVI splits a continuously-visible label (price tag, hook copy, etc.)
    into many short detection segments instead of one long span.  This
    pass groups overlays with identical text_ko whose bounding boxes
    overlap (IoU >= min_iou) and whose temporal gap is smaller than
    max_gap_sec, then merges each group into a single overlay spanning
    [first.start, last.end].
    """
    if not overlays:
        return overlays

    by_start = sorted(range(len(overlays)), key=lambda i: overlays[i].start_sec)
    merged_into: dict[int, int] = {}  # absorbed_idx -> kept_idx
    spans: dict[int, tuple[float, float]] = {}  # kept_idx -> (start, end)

    for i in by_start:
        if i in merged_into:
            continue
        a = overlays[i]
        spans[i] = (a.start_sec, a.end_sec)

        for j in by_start:
            if j == i or j in merged_into or j in spans:
                continue
            b = overlays[j]
            if b.text_ko != a.text_ko:
                continue
            if _iou(a.bbox, b.bbox) < min_iou:
                continue
            cur_start, cur_end = spans[i]
            gap = b.start_sec - cur_end
            if gap > max_gap_sec:
                continue
            spans[i] = (cur_start, max(cur_end, b.end_sec))
            merged_into[j] = i

    result: list[TextOverlay] = []
    for i in by_start:
        if i in merged_into:
            continue
        a = overlays[i]
        start, end = spans[i]
        if (start, end) == (a.start_sec, a.end_sec):
            result.append(a)
        else:
            result.append(TextOverlay(
                text_ko=a.text_ko,
                text_ja=a.text_ja,
                bbox=a.bbox,
                start_sec=start,
                end_sec=end,
                confidence=a.confidence,
            ))

    if len(result) < len(overlays):
        logger.info(
            "Recurring-overlay merge: %d -> %d (collapsed %d duplicates)",
            len(overlays), len(result), len(overlays) - len(result),
        )
    return result


def filter_middle_zone(
    overlays: list[TextOverlay],
    frame_h: int,
) -> list[TextOverlay]:
    """Remove overlays detected in the middle content zone.

    Text in the vertical band [GRAPHIC_ZONE_Y_MIN, GRAPHIC_ZONE_Y_MAX] of the
    frame is almost always part of in-video UI graphics (product cards, shopping
    pop-ups) rather than speech subtitles, so we skip it.
    """
    result = []
    for o in overlays:
        x, y, w, h = o.bbox
        center_y_frac = (y + h / 2) / frame_h
        if config.GRAPHIC_ZONE_Y_MIN <= center_y_frac <= config.GRAPHIC_ZONE_Y_MAX:
            logger.debug(
                "Skipping mid-zone graphic text (center_y=%.2f): %s",
                center_y_frac,
                o.text_ko[:40],
            )
            continue
        result.append(o)
    return result


def run(
    overlays: list[TextOverlay],
    frame_w: int,
    frame_h: int,
) -> list[TextOverlay]:
    """Apply all GVI post-processing passes in order."""
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
    cleaned = filter_large_overlays(cleaned, frame_w, frame_h)
    cleaned = suppress_subfragments(cleaned)
    cleaned = suppress_text_subfragments(cleaned)
    cleaned = merge_recurring_overlays(cleaned)
    logger.info(
        "GVI post-process: %d -> %d overlays", len(overlays), len(cleaned)
    )
    return cleaned
