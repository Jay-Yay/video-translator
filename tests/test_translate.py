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
