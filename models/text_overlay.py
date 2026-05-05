# models/text_overlay.py
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class TextOverlay:
    text_ko: str
    text_ja: str
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    start_sec: float
    end_sec: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TextOverlay:
        return cls(
            text_ko=d["text_ko"],
            text_ja=d["text_ja"],
            bbox=tuple(int(v) for v in d["bbox"]),  # JSON array → tuple[int,...]
            start_sec=float(d["start_sec"]),
            end_sec=float(d["end_sec"]),
            confidence=float(d["confidence"]),
        )
