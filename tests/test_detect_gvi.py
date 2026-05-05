# tests/test_detect_gvi.py
from datetime import timedelta
from types import SimpleNamespace

from stages.detect_gvi import _split_words_into_utterances


def _w(word: str, start: float, end: float):
    return SimpleNamespace(
        word=word,
        start_time=timedelta(seconds=start),
        end_time=timedelta(seconds=end),
    )


def test_split_no_gap_keeps_single_utterance():
    words = [_w("안녕", 0.0, 0.5), _w("하세요", 0.5, 1.0)]
    out = _split_words_into_utterances(words, 0.9)
    assert len(out) == 1
    assert out[0]["text_ko"] == "안녕 하세요"
    assert out[0]["start_sec"] == 0.0
    assert out[0]["end_sec"] == 1.0


def test_split_breaks_at_silence_gap():
    # 1.0s gap between word 1 and word 2 should split into two utterances
    words = [
        _w("첫번째", 0.0, 0.5),
        _w("문장", 0.5, 1.0),
        _w("두번째", 2.0, 2.5),  # 1.0s gap > 0.3s threshold
        _w("문장", 2.5, 3.0),
    ]
    out = _split_words_into_utterances(words, 0.85)
    assert len(out) == 2
    assert out[0]["text_ko"] == "첫번째 문장"
    assert out[0]["end_sec"] == 1.0
    assert out[1]["text_ko"] == "두번째 문장"
    assert out[1]["start_sec"] == 2.0


def test_split_small_gap_does_not_break():
    # 0.2s gap is below the 0.3s threshold
    words = [
        _w("같은", 0.0, 0.5),
        _w("문장", 0.7, 1.2),  # 0.2s gap < 0.3s threshold
    ]
    out = _split_words_into_utterances(words, 0.9)
    assert len(out) == 1


def test_split_medium_gap_breaks():
    # 0.4s gap exceeds the 0.3s threshold → splits into two utterances
    words = [
        _w("V세라", 0.0, 0.5),
        _w("알기전", 0.9, 1.3),  # 0.4s gap > 0.3s threshold
    ]
    out = _split_words_into_utterances(words, 0.9)
    assert len(out) == 2


def test_split_empty_words_returns_empty():
    assert _split_words_into_utterances([], 0.9) == []


def test_split_propagates_confidence():
    words = [_w("테스트", 0.0, 0.5)]
    out = _split_words_into_utterances(words, 0.77)
    assert out[0]["confidence"] == 0.77
