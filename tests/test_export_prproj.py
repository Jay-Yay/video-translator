from pathlib import Path
from unittest.mock import patch
from models.text_overlay import TextOverlay
from stages import export_prproj


def _make_video(tmp_path: Path, name: str = "pre_2_1.mp4") -> Path:
    v = tmp_path / "input" / name
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_bytes(b"fake-video")
    return v


def _make_mp3(tmp_path: Path, name: str = "abc123.mp3") -> Path:
    mp3 = tmp_path / "tts" / name
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"fake-audio")
    return mp3


def test_run_creates_project_folder(tmp_path):
    video = _make_video(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()

    with patch("stages.export_prproj.prproj_builder.build_prproj"):
        export_prproj.run(video, [], {}, output_dir)

    assert (output_dir / "pre_2_1").is_dir()


def test_run_moves_original_video(tmp_path):
    video = _make_video(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()

    with patch("stages.export_prproj.prproj_builder.build_prproj"):
        export_prproj.run(video, [], {}, output_dir)

    assert (output_dir / "pre_2_1" / "pre_2_1.mp4").exists()
    assert not video.exists()


def test_run_copies_tts_mp3s(tmp_path):
    video = _make_video(tmp_path)
    mp3 = _make_mp3(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()
    tts_map = {"テスト": mp3}

    with patch("stages.export_prproj.prproj_builder.build_prproj"):
        export_prproj.run(video, [], tts_map, output_dir)

    assert (output_dir / "pre_2_1" / "tts" / "abc123.mp3").exists()


def test_run_returns_prproj_path(tmp_path):
    video = _make_video(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()

    with patch("stages.export_prproj.prproj_builder.build_prproj") as mock_build:
        result = export_prproj.run(video, [], {}, output_dir)

    assert result == output_dir / "pre_2_1" / "pre_2_1.prproj"
    mock_build.assert_called_once()


def test_run_passes_overlays_and_tts_to_builder(tmp_path):
    video = _make_video(tmp_path)
    mp3 = _make_mp3(tmp_path)
    output_dir = tmp_path / "videos-done"
    output_dir.mkdir()
    overlay = TextOverlay("테스트", "テスト", (0, 0, 100, 50), 0.0, 2.0, 0.9)
    tts_map = {"テスト": mp3}

    with patch("stages.export_prproj.prproj_builder.build_prproj") as mock_build:
        result = export_prproj.run(video, [overlay], tts_map, output_dir)

    stem = "pre_2_1"
    expected_video = output_dir / stem / "pre_2_1.mp4"
    expected_tts_map = {"テスト": output_dir / stem / "tts" / "abc123.mp3"}
    expected_prproj = output_dir / stem / f"{stem}.prproj"
    mock_build.assert_called_once_with(
        export_prproj._TEMPLATE_PATH,
        expected_video,
        [overlay],
        expected_tts_map,
        expected_prproj,
        speech_segments=None,
        bgm_path=None,
        relative_paths=False,
    )
    assert result == expected_prproj
