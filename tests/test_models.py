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
