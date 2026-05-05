# tests/test_gvi_postprocess.py
import pytest
from stages.gvi_postprocess import (
    remove_repeated_phrases,
    filter_large_overlays,
    suppress_subfragments,
    merge_recurring_overlays,
)
from models.text_overlay import TextOverlay
import config


def _ov(text_ko, bbox, start=0.0, end=5.0, conf=0.99):
    return TextOverlay(text_ko, "", bbox, start, end, conf)


# --- remove_repeated_phrases ---

def test_remove_no_repetition():
    assert remove_repeated_phrases("쓰기 전에는") == "쓰기 전에는"

def test_remove_leading_bigram_repeat():
    result = remove_repeated_phrases("관리 전 관리 전 1달 후 1달 후")
    assert result == "관리 전 1달 후"

def test_remove_leading_phrase_repeat():
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
    result = remove_repeated_phrases("전에는 전에는 쓰기?")
    assert result == "전에는 쓰기?"

def test_short_text_unchanged():
    assert remove_repeated_phrases("안녕") == "안녕"
    assert remove_repeated_phrases("안녕 하세요") == "안녕 하세요"


# --- filter_large_overlays ---

def test_filter_keeps_small_overlay():
    o = _ov("작은", (0, 0, 100, 50))
    result = filter_large_overlays([o], frame_w=1000, frame_h=1000)
    assert len(result) == 1

def test_filter_removes_oversized_overlay():
    # 876x350 = 306600px on 1080x1920 = 14.8% > 10% threshold
    o = _ov("광고", (99, 1108, 876, 350))
    result = filter_large_overlays([o], frame_w=1080, frame_h=1920)
    assert len(result) == 0

def test_filter_keeps_borderline_overlay():
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
    main = _ov("브이쎄라를 끊을 수 없는 이유", (262, 248, 487, 170), start=0.0, end=4.67)
    frag = _ov("끊을 수 없는 이유", (264, 326, 477, 90), start=1.33, end=2.0)
    result = suppress_subfragments([main, frag])
    texts = [o.text_ko for o in result]
    assert "브이쎄라를 끊을 수 없는 이유" in texts
    assert "끊을 수 없는 이유" not in texts

def test_suppress_keeps_non_overlapping_overlay():
    title = _ov("제목", (262, 248, 487, 170), start=0.0, end=4.67)
    subtitle = _ov("자막", (105, 1300, 870, 80), start=0.0, end=4.67)
    result = suppress_subfragments([title, subtitle])
    assert len(result) == 2

def test_suppress_keeps_non_overlapping_time():
    a = _ov("텍스트A", (262, 248, 487, 170), start=0.0, end=2.0)
    b = _ov("텍스트B", (262, 248, 487, 170), start=5.0, end=8.0)
    result = suppress_subfragments([a, b])
    assert len(result) == 2


# --- merge_recurring_overlays ---

def test_merge_collapses_repeated_persistent_label():
    # GVI splits a persistent on-screen price tag into 3 short detections
    # at the same bbox.  All three should collapse into one span.
    a = _ov("정상가 $229", (103, 1464, 276, 77), start=0.0, end=0.9)
    b = _ov("정상가 $229", (105, 1466, 274, 75), start=1.2, end=4.4)  # iou ~ 1
    c = _ov("정상가 $229", (104, 1463, 277, 78), start=4.6, end=6.5)
    result = merge_recurring_overlays([a, b, c])
    assert len(result) == 1
    assert result[0].text_ko == "정상가 $229"
    assert result[0].start_sec == 0.0
    assert result[0].end_sec == 6.5


def test_merge_keeps_distinct_text():
    a = _ov("정상가", (100, 1400, 200, 60), start=0.0, end=1.0)
    b = _ov("할인가", (100, 1400, 200, 60), start=1.5, end=2.5)
    result = merge_recurring_overlays([a, b])
    assert len(result) == 2


def test_merge_keeps_distant_bboxes_separate():
    # Same text at different on-screen positions stays separate.
    a = _ov("$229", (100, 100, 100, 50), start=0.0, end=1.0)
    b = _ov("$229", (800, 1500, 100, 50), start=1.5, end=2.5)
    result = merge_recurring_overlays([a, b])
    assert len(result) == 2


def test_merge_does_not_bridge_long_gap():
    a = _ov("정상가 $229", (100, 100, 200, 60), start=0.0, end=1.0)
    b = _ov("정상가 $229", (100, 100, 200, 60), start=10.0, end=11.0)  # 9s gap
    result = merge_recurring_overlays([a, b])
    assert len(result) == 2
