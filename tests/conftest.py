# tests/conftest.py
# Imports are inside fixtures (lazy) so conftest loads before models/ is implemented.
import pytest


@pytest.fixture
def sample_overlay():
    from models.text_overlay import TextOverlay
    return TextOverlay(
        text_ko="촉촉한 수분감",
        text_ja="",
        bbox=(100, 200, 300, 50),
        start_sec=1.0,
        end_sec=4.0,
        confidence=0.95,
    )


@pytest.fixture
def translated_overlay():
    from models.text_overlay import TextOverlay
    return TextOverlay(
        text_ko="촉촉한 수분감",
        text_ja="うるつやもちもち肌",
        bbox=(100, 200, 300, 50),
        start_sec=1.0,
        end_sec=4.0,
        confidence=0.95,
    )


@pytest.fixture
def sample_overlays():
    from models.text_overlay import TextOverlay
    return [
        TextOverlay("텍스트A", "", (10, 10, 200, 40), 0.0, 2.0, 0.9),
        TextOverlay("텍스트B", "", (10, 60, 200, 40), 3.0, 5.0, 0.85),
    ]
