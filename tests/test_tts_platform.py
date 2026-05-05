from pathlib import Path
from stages import tts as tts_stage
from models.text_overlay import TextOverlay


def test_convert_to_wav_calls_ffmpeg(tmp_path, mocker):
    mock_run = mocker.patch("stages.tts.subprocess.run")
    mp3 = tmp_path / "clip.mp3"
    wav = tmp_path / "clip.wav"
    mp3.write_bytes(b"fake-mp3")

    tts_stage._convert_to_wav(mp3, wav)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert str(mp3) in cmd
    assert str(wav) in cmd
    assert "-ar" in cmd and "48000" in cmd
    assert "-ac" in cmd and "2" in cmd
    assert "pcm_s16le" in cmd


def test_convert_to_wav_returns_true_on_success(tmp_path, mocker):
    mocker.patch("stages.tts.subprocess.run")
    result = tts_stage._convert_to_wav(tmp_path / "a.mp3", tmp_path / "a.wav")
    assert result is True


def test_convert_to_wav_returns_false_on_error(tmp_path, mocker):
    import subprocess as _subprocess
    mocker.patch(
        "stages.tts.subprocess.run",
        side_effect=_subprocess.CalledProcessError(1, "ffmpeg"),
    )
    result = tts_stage._convert_to_wav(tmp_path / "a.mp3", tmp_path / "a.wav")
    assert result is False


def test_extract_bgm_uses_ffmpeg_for_conversion(tmp_path, mocker):
    """extract_bgm must use ffmpeg (not afconvert) for the final WAV conversion."""
    demucs_out = tmp_path / "_demucs" / "htdemucs" / "_raw_audio"
    demucs_out.mkdir(parents=True)
    fake_vocals = demucs_out / "no_vocals.wav"
    fake_vocals.write_bytes(b"fake")

    ffmpeg_conversion_cmd = []
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "ffmpeg" and str(fake_vocals) in cmd:
            ffmpeg_conversion_cmd.extend(cmd)
            (tmp_path / "bgm.wav").write_bytes(b"fake-wav")
        class R:
            returncode = 0
        return R()

    mocker.patch("stages.tts.subprocess.run", side_effect=fake_run)
    mocker.patch("stages.tts._wav_duration", return_value=60.0)

    tts_stage.extract_bgm(tmp_path / "video.mp4", tmp_path)

    assert "afconvert" not in calls, "afconvert must not be called on Linux/Modal"
    assert calls.count("ffmpeg") >= 1
    # Verify the bgm conversion uses correct ffmpeg audio flags
    assert "-ar" in ffmpeg_conversion_cmd
    assert "48000" in ffmpeg_conversion_cmd
    assert "-ac" in ffmpeg_conversion_cmd
    assert "2" in ffmpeg_conversion_cmd
    assert "pcm_s16le" in ffmpeg_conversion_cmd


def test_generate_tts_map_returns_map_for_overlays(tmp_path, mocker):
    overlays = [
        TextOverlay("한국어", "日本語テスト", (0, 0, 100, 50), 0.0, 2.0, 0.9),
        TextOverlay("한국어2", "", (0, 0, 100, 50), 2.0, 4.0, 0.9),  # empty — skip
    ]
    mock_synth = mocker.patch(
        "stages.tts._synthesize_texts",
        return_value={"日本語テスト": tmp_path / "abc.wav"},
    )
    mocker.patch("stages.tts._resolve_voice_id", return_value="voice-id-123")

    result = tts_stage.generate_tts_map(overlays, tmp_path / "checkpoints")

    mock_synth.assert_called_once_with(
        ["日本語テスト"], tmp_path / "checkpoints" / "tts", "voice-id-123"
    )
    assert result == {"日本語テスト": tmp_path / "abc.wav"}


def test_generate_tts_map_skips_empty_overlays(tmp_path, mocker):
    overlays = [TextOverlay("한국어", "", (0, 0, 100, 50), 0.0, 2.0, 0.9)]
    mock_synth = mocker.patch("stages.tts._synthesize_texts", return_value={})
    mocker.patch("stages.tts._resolve_voice_id", return_value="voice-id-123")

    result = tts_stage.generate_tts_map(overlays, tmp_path / "checkpoints")

    mock_synth.assert_called_once_with([], tmp_path / "checkpoints" / "tts", "voice-id-123")
    assert result == {}
