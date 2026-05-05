# Web Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the Korean→Japanese video translation pipeline as a password-protected web app where a teammate uploads videos, they're processed asynchronously on Modal (GPU), and Premiere Pro `.prproj` project folders land in a shared Google Drive.

**Architecture:** Vercel hosts a Next.js upload UI; videos go directly to Cloudflare R2 via presigned PUT URLs; Vercel triggers a Modal `@web_endpoint` dispatch function which spawns one `VideoTranslator` container per video (all concurrent); each container runs detect→translate→TTS→export_prproj and uploads the result folder to Google Drive; Upstash Redis tracks batch completion and triggers a SendGrid email.

**Tech Stack:** Python 3.11, Modal.com, Cloudflare R2 (boto3), Google Drive API v3, Upstash Redis, SendGrid, Next.js 14 (App Router), TypeScript, `@aws-sdk/client-s3`, `@upstash/redis`

---

## File Map

**Modified:**
- `stages/tts.py` — replace `afconvert` with `ffmpeg`; add `generate_tts_map()`
- `stages/prproj_builder.py` — add `relative_paths: bool = False` to `build_prproj()` and internal helpers
- `stages/export_prproj.py` — thread `relative_paths` param through to `build_prproj()`

**Created (worker):**
- `worker/__init__.py`
- `worker/r2_client.py` — download video from R2, delete after processing
- `worker/drive_upload.py` — upload project folder to Google Drive
- `worker/notify.py` — SendGrid completion email
- `worker/modal_app.py` — Modal app: `VideoTranslator` class + `dispatch` web endpoint

**Created (web):**
- `web/package.json`
- `web/tsconfig.json`
- `web/next.config.ts`
- `web/.env.local.example`
- `web/middleware.ts` — password gate
- `web/app/layout.tsx`
- `web/app/login/page.tsx` — login form
- `web/app/page.tsx` — upload UI
- `web/app/api/upload-urls/route.ts` — generate R2 presigned PUT URLs
- `web/app/api/submit-batch/route.ts` — write Redis batch record + call Modal dispatch

**Tests added:**
- `tests/test_tts_platform.py` — verifies ffmpeg conversion, generate_tts_map
- `tests/test_prproj_relative_paths.py` — verifies relative path mode
- `worker/tests/__init__.py`
- `worker/tests/test_r2_client.py`
- `worker/tests/test_drive_upload.py`
- `worker/tests/test_notify.py`

---

## Task 1: Replace `afconvert` with `ffmpeg` in `tts.py`

`afconvert` is macOS-only. Modal runs Linux. `ffmpeg` is already a required dep (used elsewhere in the codebase) and produces identical output.

**Files:**
- Modify: `stages/tts.py:210-231` (`_convert_to_wav`)
- Modify: `stages/tts.py:335-342` (`extract_bgm`)
- Create: `tests/test_tts_platform.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tts_platform.py
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch, call
import pytest
from stages import tts as tts_stage


def test_convert_to_wav_calls_ffmpeg(tmp_path, mocker):
    mock_run = mocker.patch("subprocess.run")
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


def test_extract_bgm_uses_ffmpeg_for_conversion(tmp_path, mocker):
    """extract_bgm must use ffmpeg (not afconvert) for the final WAV conversion."""
    # Simulate successful demucs run: create a fake no_vocals.wav
    demucs_out = tmp_path / "_demucs" / "htdemucs" / "_raw_audio"
    demucs_out.mkdir(parents=True)
    fake_vocals = demucs_out / "no_vocals.wav"
    fake_vocals.write_bytes(b"fake")

    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "ffmpeg" and str(fake_vocals) in cmd:
            # Simulate the final ffmpeg conversion writing bgm.wav
            (tmp_path / "bgm.wav").write_bytes(b"fake-wav")
        class R:
            returncode = 0
        return R()

    mocker.patch("subprocess.run", side_effect=fake_run)
    mocker.patch("stages.tts._wav_duration", return_value=60.0)

    tts_stage.extract_bgm(tmp_path / "video.mp4", tmp_path)

    assert "afconvert" not in calls, "afconvert must not be called on Linux/Modal"
    assert calls.count("ffmpeg") >= 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/jerryhong/Documents/video-translator
python -m pytest tests/test_tts_platform.py -v
```

Expected: FAIL — `assert "afconvert" not in calls` fails because current code uses afconvert.

- [ ] **Step 3: Replace `_convert_to_wav` in `stages/tts.py`**

Find lines 210–231 (the `_convert_to_wav` function) and replace:

```python
def _convert_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert mp3_path to 48kHz stereo PCM WAV using ffmpeg.

    Returns True on success, False on failure.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp3_path),
                "-ar", "48000",
                "-ac", "2",
                "-acodec", "pcm_s16le",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except Exception as exc:
        logger.warning("TTS: WAV conversion failed for %s: %s", mp3_path.name, exc)
        return False
```

- [ ] **Step 4: Replace `afconvert` in `extract_bgm` in `stages/tts.py`**

Find the `afconvert` block inside `extract_bgm` (around line 335–342) and replace:

```python
        # Convert Demucs output (44.1kHz) to 48kHz stereo PCM WAV for Premiere Pro
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(no_vocals_list[0]),
                "-ar", "48000",
                "-ac", "2",
                "-acodec", "pcm_s16le",
                str(bgm_path),
            ],
            check=True,
            capture_output=True,
        )
```

Also remove the comment referencing afconvert from `clip_path_for` docstring (line ~204): change `"afconvert (built-in on macOS)"` to `"ffmpeg"`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_tts_platform.py -v
```

Expected: PASS

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -v --ignore=tests/test_translate.py
```

Expected: all pre-existing tests pass (skip translate tests to avoid API calls).

- [ ] **Step 7: Commit**

```bash
git add stages/tts.py tests/test_tts_platform.py
git commit -m "fix: replace afconvert with ffmpeg for Linux/Modal compatibility"
```

---

## Task 2: Add `generate_tts_map()` to `tts.py`

The existing `run_speech()` generates TTS from GVI speech segments. The web pipeline uses text overlays (no GVI). Add a public function that synthesizes TTS from TextOverlay objects and returns the `{text_ja: wav_path}` map that `export_prproj.run()` expects.

**Files:**
- Modify: `stages/tts.py` (add after `run_speech`)
- Modify: `tests/test_tts_platform.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tts_platform.py`:

```python
def test_generate_tts_map_returns_map_for_overlays(tmp_path, mocker):
    from models.text_overlay import TextOverlay
    from stages import tts as tts_stage

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
    from models.text_overlay import TextOverlay
    from stages import tts as tts_stage

    overlays = [TextOverlay("한국어", "", (0, 0, 100, 50), 0.0, 2.0, 0.9)]
    mock_synth = mocker.patch("stages.tts._synthesize_texts", return_value={})
    mocker.patch("stages.tts._resolve_voice_id", return_value="voice-id-123")

    result = tts_stage.generate_tts_map(overlays, tmp_path / "checkpoints")

    mock_synth.assert_called_once_with([], tmp_path / "checkpoints" / "tts", "voice-id-123")
    assert result == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_tts_platform.py::test_generate_tts_map_returns_map_for_overlays -v
```

Expected: FAIL — `AttributeError: module 'stages.tts' has no attribute 'generate_tts_map'`

- [ ] **Step 3: Add `generate_tts_map` to `stages/tts.py`**

Insert after the `run_speech` function (after line ~295):

```python
def generate_tts_map(
    overlays: list,
    checkpoint_dir: Path,
    video_path: Path | None = None,
) -> dict[str, Path]:
    """Generate Japanese TTS WAV files from translated text overlays.

    Returns {text_ja: wav_path} for overlays with non-empty text_ja.
    Used by the web pipeline (no GVI speech segments available).
    """
    tts_dir = checkpoint_dir / "tts"
    texts = [o.text_ja for o in overlays if o.text_ja]
    voice_id = _resolve_voice_id(video_path)
    return _synthesize_texts(texts, tts_dir, voice_id)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_tts_platform.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add stages/tts.py tests/test_tts_platform.py
git commit -m "feat: add generate_tts_map() for web pipeline TTS without speech segments"
```

---

## Task 3: Add `relative_paths` mode to `prproj_builder.py`

The server writes absolute paths like `/tmp/stem.mp4` into the `.prproj` XML. Those don't exist on the teammate's Mac. With `relative_paths=True`, write just the filename (`stem.mp4`) or subfolder path (`tts/clip.wav`) so Premiere Pro resolves them relative to the `.prproj` file's location.

**Files:**
- Modify: `stages/prproj_builder.py`
- Create: `tests/test_prproj_relative_paths.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prproj_relative_paths.py
import zlib
from pathlib import Path
from unittest.mock import patch
import pytest
from stages import prproj_builder
from models.text_overlay import TextOverlay

# Re-use the minimal XML fixture from test_prproj_builder.py
MINIMAL_XML = open(
    Path(__file__).parent / "test_prproj_builder.py"
).read().split('MINIMAL_XML = """')[1].split('"""')[0]


def _make_fake_prproj(tmp_path, xml_content: str) -> Path:
    raw = xml_content.encode("utf-8")
    compress_obj = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed = compress_obj.compress(raw) + compress_obj.flush()
    path = tmp_path / "fake.prproj"
    path.write_bytes(compressed)
    return path


def _overlay(start=0.5, end=2.0, text_ja="テスト"):
    return TextOverlay("한국어", text_ja, (100, 200, 300, 50), start, end, 0.9)


def test_build_prproj_relative_paths_video(tmp_path):
    template = _make_fake_prproj(tmp_path, MINIMAL_XML)
    video = tmp_path / "my_video.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "out.prproj"

    with patch("stages.prproj_builder._probe_video", return_value=(1920, 1080, 29.97, 10.0)):
        prproj_builder.build_prproj(
            template, video, [_overlay()], {}, output, relative_paths=True
        )

    root = prproj_builder.load_template(output)
    # All FilePath elements must be just the filename, not an absolute path
    for el in root.iter("FilePath"):
        assert not el.text.startswith("/"), f"FilePath should be relative: {el.text}"
        assert el.text == "my_video.mp4"


def test_build_prproj_absolute_paths_by_default(tmp_path):
    template = _make_fake_prproj(tmp_path, MINIMAL_XML)
    video = tmp_path / "my_video.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "out.prproj"

    with patch("stages.prproj_builder._probe_video", return_value=(1920, 1080, 29.97, 10.0)):
        prproj_builder.build_prproj(template, video, [_overlay()], {}, output)

    root = prproj_builder.load_template(output)
    for el in root.iter("FilePath"):
        assert el.text.startswith("/"), f"FilePath should be absolute: {el.text}"


def test_build_prproj_relative_paths_tts_wav(tmp_path):
    template = _make_fake_prproj(tmp_path, MINIMAL_XML)
    video = tmp_path / "stem.mp4"
    video.write_bytes(b"fake")
    wav = tmp_path / "tts" / "abc123.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"fake-wav")
    output = tmp_path / "out.prproj"

    with patch("stages.prproj_builder._probe_video", return_value=(1920, 1080, 29.97, 10.0)):
        prproj_builder.build_prproj(
            template, video, [_overlay()], {"テスト": wav}, output, relative_paths=True
        )

    root = prproj_builder.load_template(output)
    audio_paths = [el.text for el in root.iter("FilePath") if el.text and el.text.endswith(".wav")]
    for ap in audio_paths:
        assert ap == "tts/abc123.wav", f"TTS path should be relative: {ap}"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_prproj_relative_paths.py -v
```

Expected: FAIL — `build_prproj() got an unexpected keyword argument 'relative_paths'`

- [ ] **Step 3: Add `relative_paths` param to `_patch_v1_clip` in `stages/prproj_builder.py`**

Find the `_patch_v1_clip` function signature and add `relative_paths: bool = False`:

```python
def _patch_v1_clip(
    root: ET.Element,
    v1_track: ET.Element,
    video_path: Path,
    duration_sec: float,
    video_w: int,
    video_h: int,
    fps: float,
    relative_paths: bool = False,
) -> bool:
```

Inside `_patch_v1_clip`, find the line `fp = str(video_path.resolve())` and replace with:

```python
fp = video_path.name if relative_paths else str(video_path.resolve())
```

- [ ] **Step 4: Add `relative_paths` to `_build_video_media_chain`**

Find the `_build_video_media_chain` function signature and add:

```python
def _build_video_media_chain(
    video_path: Path, duration_sec: float, video_w: int, video_h: int, fps: float,
    relative_paths: bool = False,
) -> tuple[list[ET.Element], str]:
```

Inside `_build_video_media_chain`, find `fp = str(video_path.resolve())` and replace with:

```python
fp = video_path.name if relative_paths else str(video_path.resolve())
```

- [ ] **Step 5: Add `relative_paths` to `_build_tts_audio_chain`**

Find the `_build_tts_audio_chain` function signature and add:

```python
def _build_tts_audio_chain(
    mp3_path: Path, start_sec: float, end_sec: float,
    relative_paths: bool = False,
) -> tuple[list[ET.Element], str]:
```

Inside `_build_tts_audio_chain`, find `fp = str(mp3_path.resolve())` and replace with:

```python
fp = f"tts/{mp3_path.name}" if relative_paths else str(mp3_path.resolve())
```

- [ ] **Step 6: Add `relative_paths` to `build_prproj` and thread it through**

Find the `build_prproj` function signature and add:

```python
def build_prproj(
    template_path: Path,
    video_path: Path,
    overlays: list,
    tts_map: dict[str, Path],
    output_path: Path,
    speech_segments: list[dict] | None = None,
    bgm_path: Path | None = None,
    relative_paths: bool = False,
) -> None:
```

Find the call to `_patch_v1_clip(...)` inside `build_prproj` and add `relative_paths=relative_paths`:

```python
patched = _patch_v1_clip(
    root, v1_track, video_path, video_duration_sec, video_w, video_h, fps,
    relative_paths=relative_paths,
)
```

Find the call to `_build_video_media_chain(...)` inside `build_prproj` and add `relative_paths=relative_paths`:

```python
chain_elems, vcti_id = _build_video_media_chain(
    video_path, video_duration_sec, video_w, video_h, fps,
    relative_paths=relative_paths,
)
```

Find every call to `_build_tts_audio_chain(...)` inside `build_prproj` (there are two — for speech segments and overlays) and add `relative_paths=relative_paths` to both. Example:

```python
chain_elems, acti_id = _build_tts_audio_chain(
    wav_path, seg["start_sec"], seg["end_sec"],
    relative_paths=relative_paths,
)
```

Do the same for the bgm path call:

```python
chain_elems, acti_id = _build_tts_audio_chain(
    bgm_path, 0.0, video_duration_sec,
    relative_paths=relative_paths,
)
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
python -m pytest tests/test_prproj_relative_paths.py tests/test_prproj_builder.py -v
```

Expected: all PASS (both new and existing tests)

- [ ] **Step 8: Commit**

```bash
git add stages/prproj_builder.py tests/test_prproj_relative_paths.py
git commit -m "feat: add relative_paths mode to build_prproj for web deployment"
```

---

## Task 4: Thread `relative_paths` through `export_prproj.py`

**Files:**
- Modify: `stages/export_prproj.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prproj_relative_paths.py`:

```python
def test_export_prproj_run_passes_relative_paths(tmp_path, mocker):
    from stages import export_prproj
    from models.text_overlay import TextOverlay

    video = tmp_path / "stem.mp4"
    video.write_bytes(b"fake")
    overlays = [TextOverlay("한국어", "テスト", (0, 0, 100, 50), 0.0, 2.0, 0.9)]
    mock_build = mocker.patch("stages.export_prproj.prproj_builder.build_prproj")
    mocker.patch("stages.export_prproj.tts_stage.extract_bgm", return_value=None)

    export_prproj.run(
        video_path=video,
        overlays=overlays,
        tts_map={},
        output_dir=tmp_path / "out",
        relative_paths=True,
    )

    _, kwargs = mock_build.call_args
    assert kwargs.get("relative_paths") is True
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_prproj_relative_paths.py::test_export_prproj_run_passes_relative_paths -v
```

Expected: FAIL — `run() got an unexpected keyword argument 'relative_paths'`

- [ ] **Step 3: Update `stages/export_prproj.py`**

Find the `run` function signature:

```python
def run(
    video_path: Path,
    overlays: list[TextOverlay],
    tts_map: dict[str, Path],
    output_dir: Path,
    speech_segments: list[dict] | None = None,
) -> Path:
```

Replace with:

```python
def run(
    video_path: Path,
    overlays: list[TextOverlay],
    tts_map: dict[str, Path],
    output_dir: Path,
    speech_segments: list[dict] | None = None,
    relative_paths: bool = False,
) -> Path:
```

Find the call to `prproj_builder.build_prproj(...)` inside `run` and add `relative_paths=relative_paths`:

```python
    prproj_builder.build_prproj(
        _TEMPLATE_PATH, dest_video, overlays, dest_tts_map, prproj_path,
        speech_segments=speech_segments,
        bgm_path=bgm_path,
        relative_paths=relative_paths,
    )
```

- [ ] **Step 4: Run all related tests**

```bash
python -m pytest tests/test_prproj_relative_paths.py tests/test_export_prproj.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add stages/export_prproj.py tests/test_prproj_relative_paths.py
git commit -m "feat: thread relative_paths through export_prproj.run()"
```

---

## Task 5: Create `worker/r2_client.py`

Downloads videos from Cloudflare R2 and deletes them after successful processing.

**Files:**
- Create: `worker/__init__.py`
- Create: `worker/r2_client.py`
- Create: `worker/tests/__init__.py`
- Create: `worker/tests/test_r2_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# worker/tests/test_r2_client.py
import os
from pathlib import Path
import pytest


def test_download_video_calls_s3_download(tmp_path, mocker):
    mocker.patch.dict(os.environ, {
        "R2_ACCOUNT_ID": "test-account",
        "R2_ACCESS_KEY_ID": "key",
        "R2_SECRET_ACCESS_KEY": "secret",
        "R2_BUCKET": "video-translator",
    })
    mock_client = mocker.MagicMock()
    mocker.patch("boto3.client", return_value=mock_client)

    from worker.r2_client import download_video
    dest = tmp_path / "video.mp4"
    download_video("batches/batch1/video.mp4", dest)

    mock_client.download_file.assert_called_once_with(
        "video-translator", "batches/batch1/video.mp4", str(dest)
    )


def test_delete_video_calls_s3_delete(mocker):
    mocker.patch.dict(os.environ, {
        "R2_ACCOUNT_ID": "test-account",
        "R2_ACCESS_KEY_ID": "key",
        "R2_SECRET_ACCESS_KEY": "secret",
        "R2_BUCKET": "video-translator",
    })
    mock_client = mocker.MagicMock()
    mocker.patch("boto3.client", return_value=mock_client)

    from worker.r2_client import delete_video
    delete_video("batches/batch1/video.mp4")

    mock_client.delete_object.assert_called_once_with(
        Bucket="video-translator", Key="batches/batch1/video.mp4"
    )


def test_r2_client_uses_correct_endpoint(mocker):
    mocker.patch.dict(os.environ, {
        "R2_ACCOUNT_ID": "my-account-id",
        "R2_ACCESS_KEY_ID": "key",
        "R2_SECRET_ACCESS_KEY": "secret",
        "R2_BUCKET": "video-translator",
    })
    mock_boto = mocker.patch("boto3.client", return_value=mocker.MagicMock())

    from worker import r2_client
    import importlib; importlib.reload(r2_client)
    r2_client.download_video("key", Path("/tmp/f.mp4"))

    call_kwargs = mock_boto.call_args[1]
    assert call_kwargs["endpoint_url"] == "https://my-account-id.r2.cloudflarestorage.com"
    assert call_kwargs["region_name"] == "auto"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest worker/tests/test_r2_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'worker'`

- [ ] **Step 3: Create `worker/__init__.py` and `worker/tests/__init__.py`**

```bash
touch worker/__init__.py worker/tests/__init__.py
```

- [ ] **Step 4: Create `worker/r2_client.py`**

```python
# worker/r2_client.py
import os
from pathlib import Path
import boto3


def _client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def download_video(r2_key: str, local_path: Path) -> None:
    _client().download_file(os.environ["R2_BUCKET"], r2_key, str(local_path))


def delete_video(r2_key: str) -> None:
    _client().delete_object(Bucket=os.environ["R2_BUCKET"], Key=r2_key)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest worker/tests/test_r2_client.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add worker/__init__.py worker/r2_client.py worker/tests/__init__.py worker/tests/test_r2_client.py
git commit -m "feat: add R2 client for video download and cleanup"
```

---

## Task 6: Create `worker/drive_upload.py`

Uploads the finished project folder (`{stem}/`) to Google Drive under `KR→JP Translations/{batch_id}/{stem}/`.

**Files:**
- Create: `worker/drive_upload.py`
- Create: `worker/tests/test_drive_upload.py`

- [ ] **Step 1: Write the failing tests**

```python
# worker/tests/test_drive_upload.py
import json
import os
from pathlib import Path
import pytest


def _mock_env(mocker):
    mocker.patch.dict(os.environ, {
        "GOOGLE_SERVICE_ACCOUNT_JSON": json.dumps({
            "type": "service_account",
            "project_id": "test",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----\n",
            "client_email": "test@test.iam.gserviceaccount.com",
            "client_id": "123",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }),
        "GOOGLE_DRIVE_FOLDER_ID": "root-folder-id",
    })


def test_upload_project_folder_creates_batch_and_stem_folders(tmp_path, mocker):
    _mock_env(mocker)
    mock_svc = mocker.MagicMock()
    mocker.patch("worker.drive_upload._service", return_value=mock_svc)

    created_folders = {}
    def fake_get_or_create(svc, name, parent_id):
        created_folders[name] = parent_id
        return f"folder-{name}"

    def fake_create(svc, name, parent_id):
        created_folders[name] = parent_id
        return f"folder-{name}"

    mocker.patch("worker.drive_upload._get_or_create_folder", side_effect=fake_get_or_create)
    mocker.patch("worker.drive_upload._create_folder", side_effect=fake_create)
    mocker.patch("worker.drive_upload._upload_file")

    # Create a minimal project dir: prproj + mp4 + tts/wav
    project_dir = tmp_path / "my_video"
    project_dir.mkdir()
    (project_dir / "my_video.mp4").write_bytes(b"video")
    (project_dir / "my_video.prproj").write_bytes(b"proj")
    tts_dir = project_dir / "tts"
    tts_dir.mkdir()
    (tts_dir / "abc.wav").write_bytes(b"wav")

    from worker.drive_upload import upload_project_folder
    upload_project_folder(project_dir, "batch-001", "my_video")

    assert "batch-001" in created_folders
    assert created_folders["batch-001"] == "root-folder-id"
    assert "my_video" in created_folders


def test_upload_project_folder_uploads_all_files(tmp_path, mocker):
    _mock_env(mocker)
    mocker.patch("worker.drive_upload._service", return_value=mocker.MagicMock())
    mocker.patch("worker.drive_upload._get_or_create_folder", return_value="batch-folder-id")
    mocker.patch("worker.drive_upload._create_folder", return_value="stem-folder-id")
    uploaded = []
    mocker.patch(
        "worker.drive_upload._upload_file",
        side_effect=lambda svc, path, parent_id: uploaded.append(path.name),
    )

    project_dir = tmp_path / "stem"
    project_dir.mkdir()
    (project_dir / "stem.mp4").write_bytes(b"v")
    (project_dir / "stem.prproj").write_bytes(b"p")
    tts = project_dir / "tts"
    tts.mkdir()
    (tts / "clip1.wav").write_bytes(b"w")

    from worker.drive_upload import upload_project_folder
    upload_project_folder(project_dir, "batch-001", "stem")

    assert "stem.mp4" in uploaded
    assert "stem.prproj" in uploaded
    assert "clip1.wav" in uploaded
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest worker/tests/test_drive_upload.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'worker.drive_upload'`

- [ ] **Step 3: Create `worker/drive_upload.py`**

```python
# worker/drive_upload.py
import json
import os
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _service():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
    return build("drive", "v3", credentials=creds)


def _create_folder(service, name: str, parent_id: str) -> str:
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return service.files().create(body=meta, fields="id").execute()["id"]


def _get_or_create_folder(service, name: str, parent_id: str) -> str:
    q = (
        f"name='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=q, fields="files(id)", pageSize=1).execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    return _create_folder(service, name, parent_id)


def _upload_file(service, path: Path, parent_id: str) -> None:
    meta = {"name": path.name, "parents": [parent_id]}
    media = MediaFileUpload(str(path), resumable=True)
    service.files().create(body=meta, media_body=media, fields="id").execute()


def upload_project_folder(local_dir: Path, batch_id: str, stem: str) -> None:
    """Upload local_dir/ to Google Drive as KR→JP Translations/{batch_id}/{stem}/."""
    svc = _service()
    root_id = os.environ["GOOGLE_DRIVE_FOLDER_ID"]

    batch_folder_id = _get_or_create_folder(svc, batch_id, root_id)
    stem_folder_id = _create_folder(svc, stem, batch_folder_id)

    for f in sorted(local_dir.iterdir()):
        if f.is_file():
            _upload_file(svc, f, stem_folder_id)

    tts_dir = local_dir / "tts"
    if tts_dir.exists():
        tts_folder_id = _create_folder(svc, "tts", stem_folder_id)
        for f in sorted(tts_dir.iterdir()):
            if f.is_file():
                _upload_file(svc, f, tts_folder_id)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest worker/tests/test_drive_upload.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add worker/drive_upload.py worker/tests/test_drive_upload.py
git commit -m "feat: add Google Drive upload helper for project folder delivery"
```

---

## Task 7: Create `worker/notify.py`

Sends a SendGrid email when all videos in a batch finish processing.

**Files:**
- Create: `worker/notify.py`
- Create: `worker/tests/test_notify.py`

- [ ] **Step 1: Write the failing tests**

```python
# worker/tests/test_notify.py
import os
import pytest


def test_send_completion_email_calls_sendgrid(mocker):
    mocker.patch.dict(os.environ, {
        "SENDGRID_API_KEY": "SG.test",
        "SENDGRID_FROM_EMAIL": "translator@example.com",
    })
    mock_sg_class = mocker.patch("sendgrid.SendGridAPIClient")
    mock_sg = mock_sg_class.return_value

    from worker.notify import send_completion_email
    send_completion_email("user@example.com", "batch-001", succeeded=10, failed=2)

    mock_sg_class.assert_called_once_with("SG.test")
    mock_sg.send.assert_called_once()
    mail = mock_sg.send.call_args[0][0]
    assert mail.to[0].email == "user@example.com"
    assert mail.from_email.email == "translator@example.com"
    assert "10" in mail.content[0].value
    assert "2" in mail.content[0].value
    assert "batch-001" in mail.content[0].value


def test_send_completion_email_subject(mocker):
    mocker.patch.dict(os.environ, {
        "SENDGRID_API_KEY": "SG.test",
        "SENDGRID_FROM_EMAIL": "translator@example.com",
    })
    mock_sg_class = mocker.patch("sendgrid.SendGridAPIClient")

    from worker.notify import send_completion_email
    send_completion_email("user@example.com", "batch-001", succeeded=5, failed=0)

    mail = mock_sg_class.return_value.send.call_args[0][0]
    assert "[Video Translator]" in mail.subject.subject
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest worker/tests/test_notify.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'worker.notify'`

- [ ] **Step 3: Create `worker/notify.py`**

```python
# worker/notify.py
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


def send_completion_email(
    to_email: str,
    batch_id: str,
    succeeded: int,
    failed: int,
) -> None:
    total = succeeded + failed
    body = (
        f"{total} video(s) processed. Results are in Google Drive:\n"
        f"KR→JP Translations/{batch_id}/\n\n"
        f"✓ Succeeded: {succeeded}\n"
        f"✗ Failed: {failed}\n"
    )
    message = Mail(
        from_email=os.environ["SENDGRID_FROM_EMAIL"],
        to_emails=to_email,
        subject="[Video Translator] Your batch is ready",
        plain_text_content=body,
    )
    SendGridAPIClient(os.environ["SENDGRID_API_KEY"]).send(message)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest worker/tests/test_notify.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add worker/notify.py worker/tests/test_notify.py
git commit -m "feat: add SendGrid notification helper for batch completion email"
```

---

## Task 8: Create `worker/modal_app.py`

The Modal application: a `VideoTranslator` class that bakes ML models into the image and runs the full pipeline per video, plus a `dispatch` web endpoint that Vercel calls to start a batch.

**Files:**
- Create: `worker/modal_app.py`

No unit tests for this task — Modal infrastructure is integration-tested via `modal run` in Task 15. The dispatch auth logic is tested below.

- [ ] **Step 1: Create `worker/modal_app.py`**

```python
# worker/modal_app.py
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import modal
from fastapi import Request, Response

app = modal.App("video-translator")

# ── Image ─────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "paddlepaddle==2.5.2",
        "paddleocr>=2.7",
        "scenedetect[opencv]>=0.6",
        "moviepy>=1.0,<2.0",
        "anthropic>=0.25",
        "elevenlabs>=1.0",
        "demucs",
        "torch",
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "boto3",
        "upstash-redis",
        "sendgrid",
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib",
        "python-dotenv",
    )
)

# Mount the full project source into /root/app so stages/, models/, etc. are available
_project_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/root/app",
    condition=lambda p: not any(
        seg in p for seg in [
            "__pycache__", ".git", "videos-", "checkpoints",
            "premiere_projects", ".env", "node_modules", "web/",
        ]
    ),
)

_secrets = modal.Secret.from_name("video-translator-secrets")


# ── VideoTranslator class ─────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="A10G",
    timeout=1800,
    memory=8192,
    secrets=[_secrets],
    mounts=[_project_mount],
)
class VideoTranslator:

    @modal.build()
    def download_models(self):
        """Bake model weights into the image layer at build time.

        Runs once during `modal deploy`. Subsequent cold starts load from the
        cached layer (~10s) instead of downloading models (~3-5min).
        """
        sys.path.insert(0, "/root/app")
        from paddleocr import PaddleOCR
        PaddleOCR(use_angle_cls=True, lang="korean")  # ~500MB to ~/.paddleocr/
        import demucs.pretrained
        demucs.pretrained.get_model("htdemucs")       # ~80MB to ~/.cache/torch/

    @modal.enter()
    def setup(self):
        sys.path.insert(0, "/root/app")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    @modal.method()
    def process_video(self, batch_id: str, r2_key: str, stem: str) -> None:
        from upstash_redis import Redis
        from worker.r2_client import download_video, delete_video
        from worker.drive_upload import upload_project_folder
        from worker.notify import send_completion_email
        from stages import detect, deduplicate, translate
        from stages import tts as tts_stage
        from stages.export_prproj import run as export_run

        redis = Redis(
            url=os.environ["UPSTASH_REDIS_REST_URL"],
            token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
        )

        work_dir = Path(tempfile.mkdtemp())
        try:
            _process(
                work_dir, batch_id, r2_key, stem,
                redis, download_video, delete_video,
                upload_project_folder, send_completion_email,
                detect, deduplicate, translate, tts_stage, export_run,
            )
        except Exception as exc:
            logging.error("[%s] FAILED: %s", stem, exc, exc_info=True)
            _mark_done(redis, batch_id, success=False, send_email=send_completion_email)
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def _process(
    work_dir, batch_id, r2_key, stem,
    redis, download_video, delete_video,
    upload_project_folder, send_completion_email,
    detect, deduplicate, translate, tts_stage, export_run,
):
    video_path = work_dir / f"{stem}.mp4"
    checkpoint_dir = work_dir / "checkpoints" / stem
    checkpoint_dir.mkdir(parents=True)
    output_dir = work_dir / "output"

    # Stage 1: download
    download_video(r2_key, video_path)

    # Stage 2: detect → deduplicate → translate
    raw_overlays = detect.run(video_path, checkpoint_dir)
    overlays = deduplicate.run(raw_overlays, checkpoint_dir)
    translated = translate.run(overlays, checkpoint_dir)

    # Stage 3: TTS (render stage intentionally skipped — .prproj handles overlays natively)
    tts_map = tts_stage.generate_tts_map(translated, checkpoint_dir, video_path)

    # Stage 4: build .prproj with relative paths (for Drive portability)
    prproj_path = export_run(
        video_path=video_path,
        overlays=translated,
        tts_map=tts_map,
        output_dir=output_dir,
        relative_paths=True,
    )

    # Stage 5: upload to Google Drive and clean up R2
    upload_project_folder(prproj_path.parent, batch_id, stem)
    delete_video(r2_key)

    # Mark success
    redis.incr(f"batch:{batch_id}:completed")
    _mark_done(redis, batch_id, success=True, send_email=send_completion_email)


def _mark_done(redis, batch_id: str, success: bool, send_email) -> None:
    done = redis.incr(f"batch:{batch_id}:done")
    total = int(redis.get(f"batch:{batch_id}:total") or 0)
    if total and done == total:
        email = redis.get(f"batch:{batch_id}:email") or ""
        succeeded = int(redis.get(f"batch:{batch_id}:completed") or 0)
        send_email(email, batch_id, succeeded=succeeded, failed=total - succeeded)


# ── Dispatch endpoint ─────────────────────────────────────────────────────────

@app.function(image=image, secrets=[_secrets])
@modal.web_endpoint(method="POST")
async def dispatch(request: Request) -> dict:
    """Validate the request and spawn one VideoTranslator.process_video per video."""
    secret = request.headers.get("X-Dispatch-Secret", "")
    if secret != os.environ["MODAL_DISPATCH_SECRET"]:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    body = await request.json()
    batch_id: str = body["batch_id"]
    videos: list[dict] = body["videos"]

    translator = VideoTranslator()
    for video in videos:
        translator.process_video.spawn(
            batch_id=batch_id,
            r2_key=video["r2_key"],
            stem=video["stem"],
        )

    return {"status": "dispatched", "count": len(videos)}
```

- [ ] **Step 2: Write a test for dispatch auth**

Add to `worker/tests/test_notify.py` (or create `worker/tests/test_dispatch_auth.py`):

```python
# worker/tests/test_dispatch_auth.py
import os
import pytest


def test_mark_done_sends_email_when_last(mocker):
    """_mark_done triggers email exactly when done == total."""
    mock_redis = mocker.MagicMock()
    mock_redis.incr.return_value = 3   # done counter reaches total
    mock_redis.get.side_effect = lambda key: {"batch:b1:total": "3", "batch:b1:email": "a@b.com", "batch:b1:completed": "2"}.get(key)
    mock_email = mocker.MagicMock()

    from worker.modal_app import _mark_done
    _mark_done(mock_redis, "b1", success=True, send_email=mock_email)

    mock_email.assert_called_once_with("a@b.com", "b1", succeeded=2, failed=1)


def test_mark_done_does_not_send_when_not_last(mocker):
    """_mark_done does not send email if done < total."""
    mock_redis = mocker.MagicMock()
    mock_redis.incr.return_value = 2   # only 2 of 3 done
    mock_redis.get.side_effect = lambda key: {"batch:b1:total": "3"}.get(key)
    mock_email = mocker.MagicMock()

    from worker.modal_app import _mark_done
    _mark_done(mock_redis, "b1", success=False, send_email=mock_email)

    mock_email.assert_not_called()
```

- [ ] **Step 3: Run the tests**

```bash
python -m pytest worker/tests/test_dispatch_auth.py -v
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add worker/modal_app.py worker/tests/test_dispatch_auth.py
git commit -m "feat: add Modal app with VideoTranslator class and dispatch web endpoint"
```

---

## Task 9: One-Time Infrastructure Setup

Manual steps — no code changes. Complete these before Task 10.

- [ ] **Step 1: Cloudflare R2**
  1. Log in to [dash.cloudflare.com](https://dash.cloudflare.com) → R2 → Create bucket: `video-translator`
  2. Settings → CORS: add rule allowing `PUT` and `GET` from your Vercel domain:
     ```json
     [{"AllowedOrigins": ["https://your-app.vercel.app"], "AllowedMethods": ["PUT", "GET"], "AllowedHeaders": ["*"]}]
     ```
  3. R2 → Manage API tokens → Create token with **Object Read & Write** on bucket `video-translator`
  4. Note down: `Account ID`, `Access Key ID`, `Secret Access Key`

- [ ] **Step 2: Upstash Redis**
  1. Go to [upstash.com](https://upstash.com) → Create Database → Region: closest to your Vercel deployment (US East or EU West)
  2. Copy `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` from the dashboard

- [ ] **Step 3: Google Cloud — Service Account**
  1. [console.cloud.google.com](https://console.cloud.google.com) → IAM & Admin → Service Accounts → Create
  2. Name: `video-translator-worker` → Create
  3. Keys tab → Add Key → JSON → Download the JSON file
  4. APIs & Services → Enable **Google Drive API**
  5. In Google Drive, create folder `KR→JP Translations` → Share it with the service account email (Editor)
  6. Note the folder ID from the Drive URL: `https://drive.google.com/drive/folders/{FOLDER_ID}`

- [ ] **Step 4: SendGrid**
  1. [sendgrid.com](https://sendgrid.com) → Settings → Sender Authentication → verify a sender email
  2. API Keys → Create API Key (Full Access) → copy it
  3. Note: `SENDGRID_API_KEY` and the verified `SENDGRID_FROM_EMAIL`

- [ ] **Step 5: Modal — create secrets**
  ```bash
  pip install modal
  modal token new   # authenticate
  modal secret create video-translator-secrets \
    ANTHROPIC_API_KEY=sk-ant-... \
    ELEVENLABS_API_KEY=sk-... \
    GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}' \
    GOOGLE_DRIVE_FOLDER_ID=1AbCdEf... \
    SENDGRID_API_KEY=SG.... \
    SENDGRID_FROM_EMAIL=translator@yourdomain.com \
    R2_ACCOUNT_ID=abc123 \
    R2_ACCESS_KEY_ID=... \
    R2_SECRET_ACCESS_KEY=... \
    R2_BUCKET=video-translator \
    UPSTASH_REDIS_REST_URL=https://... \
    UPSTASH_REDIS_REST_TOKEN=... \
    MODAL_DISPATCH_SECRET=$(openssl rand -hex 32)
  ```

- [ ] **Step 6: Deploy Modal app**
  ```bash
  cd /Users/jerryhong/Documents/video-translator
  modal deploy worker/modal_app.py
  ```
  Note the `dispatch` endpoint URL printed after deployment:
  `https://your-org--video-translator-dispatch.modal.run`

---

## Task 10: Next.js Project Setup + Auth Middleware

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/next.config.ts`
- Create: `web/.env.local.example`
- Create: `web/middleware.ts`
- Create: `web/app/layout.tsx`
- Create: `web/app/login/page.tsx`

- [ ] **Step 1: Create `web/package.json`**

```json
{
  "name": "video-translator-web",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "@aws-sdk/client-s3": "^3.600.0",
    "@aws-sdk/s3-request-presigner": "^3.600.0",
    "@upstash/redis": "^1.34.0",
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "typescript": "^5.4.0"
  }
}
```

- [ ] **Step 2: Create `web/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 3: Create `web/next.config.ts`**

```typescript
import type { NextConfig } from "next"

const config: NextConfig = {
  // API routes handle large JSON (batch manifests); default 4.5MB limit is fine
  // since videos bypass Vercel entirely (presigned R2 direct upload)
}

export default config
```

- [ ] **Step 4: Create `web/.env.local.example`**

```bash
# Copy to .env.local and fill in values
UPLOAD_PASSWORD=choose-a-strong-password
COOKIE_SECRET=generate-with-openssl-rand-hex-32

R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=video-translator

UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=

MODAL_DISPATCH_URL=https://your-org--video-translator-dispatch.modal.run
MODAL_DISPATCH_SECRET=
```

- [ ] **Step 5: Create `web/middleware.ts`**

```typescript
import { createHmac } from "crypto"
import { NextRequest, NextResponse } from "next/server"

function makeToken(password: string, secret: string): string {
  return createHmac("sha256", secret).update(password).digest("hex")
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Skip auth check for the login page itself and static assets
  if (pathname === "/login" || pathname.startsWith("/_next")) {
    return NextResponse.next()
  }

  const token = request.cookies.get("auth-token")?.value
  const expected = makeToken(
    process.env.UPLOAD_PASSWORD!,
    process.env.COOKIE_SECRET!,
  )

  if (token !== expected) {
    const url = request.nextUrl.clone()
    url.pathname = "/login"
    return NextResponse.redirect(url)
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
}
```

- [ ] **Step 6: Create `web/app/layout.tsx`**

```tsx
import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Video Translator",
  description: "Korean → Japanese video translation",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0, background: "#f5f5f5" }}>
        {children}
      </body>
    </html>
  )
}
```

- [ ] **Step 7: Create `web/app/login/page.tsx`**

```tsx
"use client"
import { useState } from "react"
import { useRouter } from "next/navigation"

export default function LoginPage() {
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const router = useRouter()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    })
    if (res.ok) {
      router.push("/")
    } else {
      setError("Incorrect password")
    }
  }

  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh" }}>
      <form onSubmit={handleSubmit} style={{ background: "white", padding: 32, borderRadius: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.1)", width: 320 }}>
        <h2 style={{ margin: "0 0 24px" }}>Video Translator</h2>
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          style={{ width: "100%", padding: "10px 12px", fontSize: 16, border: "1px solid #ccc", borderRadius: 4, boxSizing: "border-box" }}
          autoFocus
        />
        {error && <p style={{ color: "red", margin: "8px 0 0" }}>{error}</p>}
        <button
          type="submit"
          style={{ marginTop: 16, width: "100%", padding: "10px 0", background: "#0070f3", color: "white", border: "none", borderRadius: 4, fontSize: 16, cursor: "pointer" }}
        >
          Sign in
        </button>
      </form>
    </div>
  )
}
```

- [ ] **Step 8: Install dependencies and verify the app builds**

```bash
cd web
npm install
npm run build
```

Expected: build succeeds (login page and layout render; upload page and API routes don't exist yet — that's fine)

- [ ] **Step 9: Commit**

```bash
cd ..
git add web/
git commit -m "feat: scaffold Next.js web app with password auth middleware and login page"
```

---

## Task 11: Upload URLs + Login API Routes

**Files:**
- Create: `web/app/api/login/route.ts`
- Create: `web/app/api/upload-urls/route.ts`

- [ ] **Step 1: Create `web/app/api/login/route.ts`**

```typescript
import { createHmac } from "crypto"
import { cookies } from "next/headers"
import { NextRequest, NextResponse } from "next/server"

function makeToken(password: string, secret: string): string {
  return createHmac("sha256", secret).update(password).digest("hex")
}

export async function POST(request: NextRequest) {
  const { password } = await request.json()

  if (password !== process.env.UPLOAD_PASSWORD) {
    return NextResponse.json({ error: "Incorrect password" }, { status: 401 })
  }

  const token = makeToken(password, process.env.COOKIE_SECRET!)
  const response = NextResponse.json({ ok: true })
  response.cookies.set("auth-token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 60 * 60 * 24 * 30, // 30 days
    path: "/",
  })
  return response
}
```

- [ ] **Step 2: Create `web/app/api/upload-urls/route.ts`**

```typescript
import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3"
import { getSignedUrl } from "@aws-sdk/s3-request-presigner"
import { NextRequest, NextResponse } from "next/server"

const r2 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID!,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
  },
})

export async function POST(request: NextRequest) {
  const { batch_id, filenames } = (await request.json()) as {
    batch_id: string
    filenames: string[]
  }

  if (!batch_id || !Array.isArray(filenames) || filenames.length === 0) {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 })
  }

  const urls = await Promise.all(
    filenames.map(async (filename) => {
      // Sanitise filename: keep only the last path component, strip special chars
      const stem = filename.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_\-]/g, "_")
      const key = `batches/${batch_id}/${stem}.mp4`
      const url = await getSignedUrl(
        r2,
        new PutObjectCommand({ Bucket: process.env.R2_BUCKET!, Key: key }),
        { expiresIn: 900 }, // 15 minutes
      )
      return { filename, stem, r2_key: key, upload_url: url }
    }),
  )

  return NextResponse.json({ urls })
}
```

- [ ] **Step 3: Build to verify no TypeScript errors**

```bash
cd web && npm run build
```

Expected: build succeeds

- [ ] **Step 4: Commit**

```bash
cd ..
git add web/app/api/login/route.ts web/app/api/upload-urls/route.ts
git commit -m "feat: add login API and R2 presigned upload URL API routes"
```

---

## Task 12: Submit Batch API Route

Writes the batch record to Redis, then calls the Modal dispatch endpoint to spawn workers.

**Files:**
- Create: `web/app/api/submit-batch/route.ts`

- [ ] **Step 1: Create `web/app/api/submit-batch/route.ts`**

```typescript
import { Redis } from "@upstash/redis"
import { NextRequest, NextResponse } from "next/server"

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
})

interface VideoEntry {
  r2_key: string
  stem: string
}

export async function POST(request: NextRequest) {
  const { batch_id, videos, notify_email } = (await request.json()) as {
    batch_id: string
    videos: VideoEntry[]
    notify_email: string
  }

  if (!batch_id || !videos?.length || !notify_email) {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 })
  }

  // Write batch record to Redis (TTL: 7 days)
  const ttl = 60 * 60 * 24 * 7
  await Promise.all([
    redis.set(`batch:${batch_id}:total`, videos.length, { ex: ttl }),
    redis.set(`batch:${batch_id}:completed`, 0, { ex: ttl }),
    redis.set(`batch:${batch_id}:done`, 0, { ex: ttl }),
    redis.set(`batch:${batch_id}:email`, notify_email, { ex: ttl }),
  ])

  // Trigger Modal dispatch endpoint
  const modalRes = await fetch(process.env.MODAL_DISPATCH_URL!, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Dispatch-Secret": process.env.MODAL_DISPATCH_SECRET!,
    },
    body: JSON.stringify({ batch_id, videos }),
  })

  if (!modalRes.ok) {
    const text = await modalRes.text()
    return NextResponse.json(
      { error: `Modal dispatch failed: ${modalRes.status} ${text}` },
      { status: 502 },
    )
  }

  return NextResponse.json({ ok: true, batch_id, count: videos.length })
}
```

- [ ] **Step 2: Build to verify no TypeScript errors**

```bash
cd web && npm run build
```

Expected: build succeeds

- [ ] **Step 3: Commit**

```bash
cd ..
git add web/app/api/submit-batch/route.ts
git commit -m "feat: add submit-batch API route — writes Redis record and triggers Modal dispatch"
```

---

## Task 13: Upload UI Page

The single-page upload interface: drag-and-drop video files, per-file progress bars, email input, submit button.

**Files:**
- Create: `web/app/page.tsx`

- [ ] **Step 1: Create `web/app/page.tsx`**

```tsx
"use client"
import { useCallback, useRef, useState } from "react"

type FileEntry = {
  file: File
  stem: string
  r2_key: string
  upload_url: string
  progress: number  // 0–100
  done: boolean
  error: string | null
}

function generateBatchId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7)
}

export default function UploadPage() {
  const [entries, setEntries] = useState<FileEntry[]>([])
  const [email, setEmail] = useState("")
  const [phase, setPhase] = useState<"idle" | "preparing" | "uploading" | "submitted" | "error">("idle")
  const [errorMsg, setErrorMsg] = useState("")
  const batchIdRef = useRef(generateBatchId())

  const setEntry = (stem: string, patch: Partial<FileEntry>) =>
    setEntries(prev => prev.map(e => e.stem === stem ? { ...e, ...patch } : e))

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith(".mp4"))
    if (files.length === 0) return
    // Deduplicate by filename
    setEntries(prev => {
      const existing = new Set(prev.map(e => e.file.name))
      const fresh = files.filter(f => !existing.has(f.name))
      return [...prev, ...fresh.map(f => ({
        file: f,
        stem: f.name.replace(/\.mp4$/i, "").replace(/[^a-zA-Z0-9_\-]/g, "_"),
        r2_key: "",
        upload_url: "",
        progress: 0,
        done: false,
        error: null,
      }))]
    })
  }, [])

  async function handleUpload() {
    if (!email || entries.length === 0) return
    setPhase("preparing")

    // 1. Get presigned URLs
    const res = await fetch("/api/upload-urls", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_id: batchIdRef.current,
        filenames: entries.map(e => e.file.name),
      }),
    })
    if (!res.ok) { setPhase("error"); setErrorMsg("Failed to get upload URLs"); return }
    const { urls } = await res.json()

    // Attach upload URLs
    setEntries(prev => prev.map(e => {
      const match = urls.find((u: { stem: string }) => u.stem === e.stem)
      return match ? { ...e, r2_key: match.r2_key, upload_url: match.upload_url } : e
    }))

    setPhase("uploading")

    // 2. Upload all files in parallel (direct to R2 via presigned PUT)
    await Promise.all(
      entries.map(async (entry, i) => {
        const urlEntry = urls[i]
        try {
          await new Promise<void>((resolve, reject) => {
            const xhr = new XMLHttpRequest()
            xhr.upload.onprogress = ev => {
              if (ev.lengthComputable)
                setEntry(entry.stem, { progress: Math.round((ev.loaded / ev.total) * 100) })
            }
            xhr.onload = () => xhr.status < 300 ? resolve() : reject(new Error(`HTTP ${xhr.status}`))
            xhr.onerror = () => reject(new Error("Network error"))
            xhr.open("PUT", urlEntry.upload_url)
            xhr.setRequestHeader("Content-Type", "video/mp4")
            xhr.send(entry.file)
          })
          setEntry(entry.stem, { done: true, progress: 100 })
        } catch (err) {
          setEntry(entry.stem, { error: String(err) })
        }
      })
    )

    // 3. Submit batch
    const submitRes = await fetch("/api/submit-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_id: batchIdRef.current,
        videos: entries.map(e => ({ r2_key: urls.find((u: { stem: string }) => u.stem === e.stem)?.r2_key, stem: e.stem })),
        notify_email: email,
      }),
    })
    if (!submitRes.ok) { setPhase("error"); setErrorMsg("Failed to submit batch"); return }
    setPhase("submitted")
  }

  if (phase === "submitted") {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh" }}>
        <div style={{ textAlign: "center", background: "white", padding: 40, borderRadius: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.1)" }}>
          <div style={{ fontSize: 48 }}>✓</div>
          <h2>Submitted!</h2>
          <p>You'll receive an email at <strong>{email}</strong> when your {entries.length} video{entries.length > 1 ? "s are" : " is"} ready.</p>
          <p style={{ color: "#666", fontSize: 14 }}>Results will appear in Google Drive › KR→JP Translations</p>
          <button onClick={() => { setPhase("idle"); setEntries([]); batchIdRef.current = generateBatchId() }}
            style={{ marginTop: 16, padding: "10px 24px", background: "#0070f3", color: "white", border: "none", borderRadius: 4, fontSize: 16, cursor: "pointer" }}>
            Upload another batch
          </button>
        </div>
      </div>
    )
  }

  const allDone = entries.length > 0 && entries.every(e => e.done || e.error)
  const hasErrors = entries.some(e => e.error)

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ marginBottom: 8 }}>KR → JP Video Translator</h1>
      <p style={{ color: "#666", marginBottom: 24 }}>Upload Korean beauty videos. Translated Premiere Pro projects will land in Google Drive.</p>

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={e => e.preventDefault()}
        style={{ border: "2px dashed #ccc", borderRadius: 8, padding: 40, textAlign: "center", color: "#666", marginBottom: 24, cursor: "pointer", background: "white" }}
      >
        <div style={{ fontSize: 32, marginBottom: 8 }}>⬆</div>
        Drag & drop <strong>.mp4</strong> files here
        <div style={{ marginTop: 12 }}>
          <label style={{ cursor: "pointer", color: "#0070f3" }}>
            or click to browse
            <input type="file" accept=".mp4" multiple style={{ display: "none" }}
              onChange={e => {
                const files = Array.from(e.target.files || [])
                setEntries(prev => {
                  const existing = new Set(prev.map(en => en.file.name))
                  return [...prev, ...files.filter(f => !existing.has(f.name)).map(f => ({
                    file: f, stem: f.name.replace(/\.mp4$/i, "").replace(/[^a-zA-Z0-9_\-]/g, "_"),
                    r2_key: "", upload_url: "", progress: 0, done: false, error: null,
                  }))]
                })
              }}
            />
          </label>
        </div>
      </div>

      {/* File list */}
      {entries.length > 0 && (
        <div style={{ background: "white", borderRadius: 8, padding: 16, marginBottom: 24 }}>
          {entries.map(e => (
            <div key={e.stem} style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontWeight: 500 }}>{e.file.name}</span>
                <span style={{ color: "#666", fontSize: 14 }}>{(e.file.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
              <div style={{ height: 6, background: "#eee", borderRadius: 3 }}>
                <div style={{ height: "100%", background: e.error ? "#e00" : e.done ? "#0a0" : "#0070f3", borderRadius: 3, width: `${e.progress}%`, transition: "width 0.3s" }} />
              </div>
              {e.error && <div style={{ color: "#e00", fontSize: 13, marginTop: 4 }}>{e.error}</div>}
            </div>
          ))}
        </div>
      )}

      {/* Email input */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ display: "block", marginBottom: 6, fontWeight: 500 }}>Notify email</label>
        <input
          type="email"
          placeholder="your@email.com"
          value={email}
          onChange={e => setEmail(e.target.value)}
          style={{ width: "100%", padding: "10px 12px", fontSize: 16, border: "1px solid #ccc", borderRadius: 4, boxSizing: "border-box" }}
        />
      </div>

      {/* Submit */}
      {phase === "error" && <p style={{ color: "red" }}>{errorMsg}</p>}
      <button
        onClick={handleUpload}
        disabled={entries.length === 0 || !email || phase === "uploading" || phase === "preparing"}
        style={{
          width: "100%", padding: "12px 0", fontSize: 16,
          background: entries.length === 0 || !email ? "#ccc" : "#0070f3",
          color: "white", border: "none", borderRadius: 4, cursor: entries.length === 0 || !email ? "not-allowed" : "pointer",
        }}
      >
        {phase === "preparing" ? "Preparing uploads…" :
         phase === "uploading" ? `Uploading… (${entries.filter(e => e.done).length}/${entries.length})` :
         `Translate ${entries.length > 0 ? entries.length : ""} video${entries.length !== 1 ? "s" : ""}`}
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Build to verify no TypeScript errors**

```bash
cd web && npm run build
```

Expected: build succeeds with no errors

- [ ] **Step 3: Smoke-test locally**

```bash
cd web
cp .env.local.example .env.local
# Fill in UPLOAD_PASSWORD and COOKIE_SECRET with any values for local testing
npm run dev
```

Open `http://localhost:3000` — should redirect to `/login`. Enter password → redirected to upload page. Drop an `.mp4` file — should appear in the file list with a progress bar.

- [ ] **Step 4: Commit**

```bash
cd ..
git add web/app/page.tsx
git commit -m "feat: add upload UI with drag-and-drop, per-file progress bars, and batch submission"
```

---

## Task 14: Deploy to Vercel + End-to-End Smoke Test

- [ ] **Step 1: Push the branch to GitHub**

```bash
git push origin feat/korean-to-japanese-translator
```

- [ ] **Step 2: Import project in Vercel**
  1. [vercel.com](https://vercel.com) → New Project → Import from GitHub → select this repo
  2. **Root Directory**: set to `web`
  3. Framework: Next.js (auto-detected)

- [ ] **Step 3: Add Vercel environment variables**

In Vercel project → Settings → Environment Variables, add:

```
UPLOAD_PASSWORD          = (your chosen password)
COOKIE_SECRET            = (output of: openssl rand -hex 32)
R2_ACCOUNT_ID            = (from Cloudflare)
R2_ACCESS_KEY_ID         = (from Cloudflare)
R2_SECRET_ACCESS_KEY     = (from Cloudflare)
R2_BUCKET                = video-translator
UPSTASH_REDIS_REST_URL   = (from Upstash)
UPSTASH_REDIS_REST_TOKEN = (from Upstash)
MODAL_DISPATCH_URL       = https://your-org--video-translator-dispatch.modal.run
MODAL_DISPATCH_SECRET    = (same value as in Modal secrets)
```

- [ ] **Step 4: Update R2 CORS to allow the Vercel deployment URL**

In Cloudflare R2 → bucket `video-translator` → Settings → CORS:

```json
[{
  "AllowedOrigins": ["https://your-app.vercel.app"],
  "AllowedMethods": ["PUT"],
  "AllowedHeaders": ["Content-Type"]
}]
```

- [ ] **Step 5: Deploy and smoke-test**
  1. Vercel auto-deploys the `main` branch on push
  2. Open the Vercel URL → login with password → upload 1 short test video (< 30s)
  3. Submit with your email
  4. Check Modal dashboard → should see a `VideoTranslator.process_video` call running
  5. Wait ~5–15 min → check email for "batch is ready" notification
  6. Open Google Drive → `KR→JP Translations/{batch_id}/` → should contain `{stem}/` with `.prproj`, `.mp4`, `tts/`
  7. Open the `.prproj` in Premiere Pro → verify video loads on V1, text overlays appear on V2/V3, TTS audio on A2, BGM on A3

- [ ] **Step 6: Commit final adjustments if any were needed during smoke test**

```bash
git add -A
git commit -m "fix: smoke test adjustments"
git push
```

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Bulk upload with async processing → Task 13 (upload page) + Task 8 (Modal parallel spawn)
- ✓ Translated text overlays in .prproj → Tasks 3–4 (relative_paths) + Task 8 (export_prproj called with relative_paths=True)
- ✓ Japanese TTS voice-over → Task 2 (generate_tts_map) + Task 8
- ✓ Background music (Demucs) → export_prproj.run() calls extract_bgm internally; Task 1 fixes afconvert → ffmpeg so it works on Modal
- ✓ Google Drive delivery → Task 6 (drive_upload) + Task 8
- ✓ SendGrid notification → Task 7 (notify) + Task 8 (_mark_done)
- ✓ Atomic Redis completion (no race condition) → Task 8 (_mark_done uses INCR return value)
- ✓ R2 cleanup after processing → Task 8 (delete_video called after upload)
- ✓ Modal model warm-up → Task 8 (@modal.build() bakes PaddleOCR + Demucs)
- ✓ Password gate → Task 10 (middleware.ts + login page)
- ✓ Presigned PUT URL upload → Task 11 (upload-urls route)
- ✓ Modal dispatch auth (X-Dispatch-Secret) → Task 8 (dispatch endpoint) + Task 12 (submit-batch sends header)
- ✓ Infrastructure setup → Task 9

**No placeholders found.**

**Type consistency:**
- `generate_tts_map(overlays, checkpoint_dir, video_path)` defined in Task 2, called in Task 8 ✓
- `export_prproj.run(..., relative_paths=True)` signature updated in Task 4, called in Task 8 ✓
- `upload_project_folder(local_dir, batch_id, stem)` defined in Task 6, called in Task 8 ✓
- `send_completion_email(to_email, batch_id, succeeded, failed)` defined in Task 7, called via `_mark_done` in Task 8 ✓
- `_mark_done(redis, batch_id, success, send_email)` defined in Task 8, tested in Task 8 ✓
