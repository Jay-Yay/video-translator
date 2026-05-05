"""Tests for relative_paths mode in build_prproj."""
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest
from stages import prproj_builder
from models.text_overlay import TextOverlay

# Import MINIMAL_XML directly from the existing test module.
from tests.test_prproj_builder import MINIMAL_XML

# Extended XML that includes a V1 VideoClipTrack so the video-injection path is exercised.
# V1_TRACK_UID from prproj_builder = "e69ab9cb-ac32-4fd4-935b-b9af7c9b3eed"
_MINIMAL_XML_WITH_V1 = """<?xml version="1.0" encoding="UTF-8"?>
<PremiereData Version="3">
  <Project ObjectID="1">
    <Node Version="1"><Properties Version="1"/></Node>
  </Project>
  <Sequence ObjectUID="f265fac1-ff5a-4b9a-8e16-07bab379302e">
    <Node Version="1">
      <Properties Version="1">
        <MZ.Sequence.PreviewFrameSizeWidth>1920</MZ.Sequence.PreviewFrameSizeWidth>
        <MZ.Sequence.PreviewFrameSizeHeight>1080</MZ.Sequence.PreviewFrameSizeHeight>
      </Properties>
    </Node>
    <TrackGroups Version="1">
      <TrackGroup Version="1" Index="0">
        <Second ObjectRef="66"/>
      </TrackGroup>
      <TrackGroup Version="1" Index="1">
        <Second ObjectRef="53"/>
      </TrackGroup>
    </TrackGroups>
  </Sequence>
  <VideoTrackGroup ObjectID="66">
    <TrackGroup Version="1">
      <Tracks Version="1">
        <Track Index="0" ObjectURef="e69ab9cb-ac32-4fd4-935b-b9af7c9b3eed"/>
        <Track Index="1" ObjectURef="0412bfac-213d-427e-9419-4c61829333c4"/>
      </Tracks>
      <FrameRate>8475667200</FrameRate>
    </TrackGroup>
    <FrameRect>0,0,1920,1080</FrameRect>
  </VideoTrackGroup>
  <AudioTrackGroup ObjectID="53">
    <TrackGroup Version="1">
      <Tracks Version="1">
        <Track Index="0" ObjectURef="9dbc1352-d0c0-4447-a4bb-71b4f0036267"/>
        <Track Index="1" ObjectURef="421fb4d8-eeb4-45d2-9ffa-b91edd886274"/>
      </Tracks>
    </TrackGroup>
  </AudioTrackGroup>
  <VideoClipTrack ObjectUID="e69ab9cb-ac32-4fd4-935b-b9af7c9b3eed">
    <ClipTrack Version="2">
      <Track Version="4">
        <Node Version="1"><Properties Version="1"/></Node>
        <ID>1</ID>
        <Index>0</Index>
      </Track>
      <ClipItems Version="3">
        <TrackItems Version="1">
          <TrackItem Index="0" ObjectRef="200"/>
        </TrackItems>
      </ClipItems>
    </ClipTrack>
  </VideoClipTrack>
  <VideoClipTrack ObjectUID="0412bfac-213d-427e-9419-4c61829333c4">
    <ClipTrack Version="2">
      <Track Version="4">
        <Node Version="1"><Properties Version="1"/></Node>
        <ID>2</ID>
        <Index>1</Index>
      </Track>
      <ClipItems Version="3">
        <TrackItems Version="1">
          <TrackItem Index="0" ObjectRef="79"/>
        </TrackItems>
      </ClipItems>
    </ClipTrack>
  </VideoClipTrack>
  <AudioClipTrack ObjectUID="9dbc1352-d0c0-4447-a4bb-71b4f0036267">
    <ClipTrack Version="2">
      <Track Version="4">
        <Node Version="1"><Properties Version="1"/></Node>
        <ID>1</ID>
        <Index>0</Index>
      </Track>
      <ClipItems Version="3"/>
    </ClipTrack>
    <AudioTrack Version="12"/>
  </AudioClipTrack>
  <AudioClipTrack ObjectUID="421fb4d8-eeb4-45d2-9ffa-b91edd886274">
    <ClipTrack Version="2">
      <Track Version="4">
        <Node Version="1"><Properties Version="1"/></Node>
        <ID>2</ID>
        <Index>1</Index>
      </Track>
      <ClipItems Version="3"/>
    </ClipTrack>
    <AudioTrack Version="12"/>
  </AudioClipTrack>
  <VideoClipTrackItem ObjectID="200">
    <ClipTrackItem Version="8">
      <ComponentOwner Version="1">
        <Components ObjectRef="201"/>
      </ComponentOwner>
      <TrackItem Version="4">
        <End>2540160000000</End>
        <Start>0</Start>
      </TrackItem>
      <SubClip ObjectRef="202"/>
    </ClipTrackItem>
    <FrameRect>0,0,1920,1080</FrameRect>
  </VideoClipTrackItem>
  <VideoComponentChain ObjectID="201">
    <ComponentChain Version="3">
      <Components Version="1"/>
    </ComponentChain>
  </VideoComponentChain>
  <SubClip ObjectID="202">
    <Clip ObjectRef="203"/>
    <MasterClip ObjectURef="master-clip-v1-uid"/>
    <Name>video</Name>
  </SubClip>
  <VideoClip ObjectID="203">
    <Clip Version="18">
      <Node Version="1"><Properties Version="1"/></Node>
      <InPoint>0</InPoint>
      <OutPoint>2540160000000</OutPoint>
      <ClipID>clip-v1-1</ClipID>
    </Clip>
  </VideoClip>
  <MasterClip ObjectUID="master-clip-v1-uid">
    <Clips Version="1">
      <Clip ObjectRef="203"/>
    </Clips>
    <Name>video</Name>
  </MasterClip>
  <VideoMediaSource ObjectID="210"
      ClassID="e64ddf74-8fac-4682-8aa8-0e0ca2248949" Version="2">
    <MediaSource Version="1">
      <Media ObjectURef="media-v1-uid"/>
    </MediaSource>
  </VideoMediaSource>
  <VideoStream ObjectID="211"
      ClassID="a36e4719-3ec6-4a0c-ab11-8b4aab377aa5" Version="22">
    <Duration>2540160000000</Duration>
    <FrameRect>0,0,1920,1080</FrameRect>
    <FrameRate>8475667200</FrameRate>
  </VideoStream>
  <Media ObjectUID="media-v1-uid"
      ClassID="7a5c103e-f3ac-4391-b6b4-7cc3d2f9a7ff" Version="30">
    <VideoStream ObjectRef="211"/>
    <FilePath>/tmp/placeholder.mp4</FilePath>
    <ActualMediaFilePath>/tmp/placeholder.mp4</ActualMediaFilePath>
    <Title>placeholder</Title>
    <Infinite>false</Infinite>
  </Media>
  <VideoClipTrackItem ObjectID="79">
    <ClipTrackItem Version="8">
      <ComponentOwner Version="1">
        <Components ObjectRef="93"/>
      </ComponentOwner>
      <TrackItem Version="4">
        <End>1262874412800</End>
      </TrackItem>
      <SubClip ObjectRef="94"/>
    </ClipTrackItem>
    <FrameRect>0,0,1920,1080</FrameRect>
  </VideoClipTrackItem>
  <VideoComponentChain ObjectID="93">
    <ComponentChain Version="3">
      <Components Version="1">
        <Component Index="0" ObjectRef="111"/>
      </Components>
    </ComponentChain>
  </VideoComponentChain>
  <SubClip ObjectID="94">
    <Clip ObjectRef="112"/>
    <MasterClip ObjectURef="master-clip-uid-1"/>
    <Name>Graphic</Name>
  </SubClip>
  <VideoFilterComponent ObjectID="111">
    <Component Version="7">
      <Params Version="1">
        <Param Index="0" ObjectRef="123"/>
        <Param Index="1" ObjectRef="125"/>
      </Params>
      <DisplayName>Text</DisplayName>
      <InstanceName>PLACEHOLDER</InstanceName>
    </Component>
    <MatchName>AE.ADBE Text</MatchName>
  </VideoFilterComponent>
  <VideoClip ObjectID="112">
    <Clip Version="18">
      <Node Version="1"><Properties Version="1"/></Node>
      <InPoint>914456685542400</InPoint>
      <OutPoint>915719559955200</OutPoint>
      <ClipID>clip-id-1</ClipID>
    </Clip>
  </VideoClip>
  <ArbVideoComponentParam ObjectID="123">
    <Name>Source Text</Name>
    <ParameterID>1</ParameterID>
    <StartKeyframePosition>-91445760000000000</StartKeyframePosition>
    <StartKeyframeValue Encoding="base64">RAEAAAAAAABEMyIRDAAAAAAABgAKAAQABgAAAGQAAAAAAF4AGAAQAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAABcABwBeAAAAAAAAARAAAAAcAAAANAAAAAAAAQBg////ZP///2j///9s////AQAAAAQAAAAMAAAATHVjaWRhR3JhbmRlAAAAAAEAAAAMAAAACAAOAAQACAAIAAAAaAAAADwAAAAAADYAFAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAACAAEADYAAAACAAAADAAAAAwAAAAAAIBA9P////j////8////BAAEAAQAAAALAAAAUExBQ0VIT0xERVIA</StartKeyframeValue>
  </ArbVideoComponentParam>
  <PointComponentParam ObjectID="125">
    <Name>Position</Name>
    <ParameterID>3</ParameterID>
    <StartKeyframe>-91445760000000000,0.28674697875976562:0.49356222152709961,0,0,0,0,0,0,5,4,0,0,0,0</StartKeyframe>
  </PointComponentParam>
</PremiereData>"""


def _make_fake_prproj(tmp_path, xml_content: str) -> Path:
    raw = xml_content.encode("utf-8")
    compress_obj = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed = compress_obj.compress(raw) + compress_obj.flush()
    path = tmp_path / "fake.prproj"
    path.write_bytes(compressed)
    return path


def _overlay(start=0.5, end=2.0, text_ja="テスト"):
    return TextOverlay("한국어", text_ja, (100, 200, 300, 50), start, end, 0.9)


@patch("stages.prproj_builder._probe_video", return_value=(1920, 1080, 29.97, 10.0))
def test_build_prproj_relative_paths_video(mock_probe, tmp_path):
    """With relative_paths=True, FilePath elements contain just the filename, not an abs path."""
    template = _make_fake_prproj(tmp_path, _MINIMAL_XML_WITH_V1)
    video = tmp_path / "my_video.mp4"
    video.write_bytes(b"fake")
    overlay = _overlay()
    output = tmp_path / "out.prproj"

    prproj_builder.build_prproj(
        template, video, [overlay], {}, output, relative_paths=True
    )

    root = prproj_builder.load_template(output)
    file_paths = [el.text for el in root.iter("FilePath") if el.text]
    # There should be at least one FilePath (for the video)
    assert file_paths, "No FilePath elements found in output"
    video_paths = [p for p in file_paths if p.endswith(".mp4")]
    assert video_paths, "No .mp4 FilePath found"
    for fp in video_paths:
        assert not fp.startswith("/"), f"Expected relative path, got: {fp!r}"
        assert fp == "my_video.mp4", f"Expected 'my_video.mp4', got: {fp!r}"


@patch("stages.prproj_builder._probe_video", return_value=(1920, 1080, 29.97, 10.0))
def test_build_prproj_absolute_paths_by_default(mock_probe, tmp_path):
    """Without relative_paths, FilePath elements contain absolute paths (starting with /)."""
    template = _make_fake_prproj(tmp_path, _MINIMAL_XML_WITH_V1)
    video = tmp_path / "my_video.mp4"
    video.write_bytes(b"fake")
    overlay = _overlay()
    output = tmp_path / "out.prproj"

    prproj_builder.build_prproj(
        template, video, [overlay], {}, output
    )

    root = prproj_builder.load_template(output)
    file_paths = [el.text for el in root.iter("FilePath") if el.text]
    assert file_paths, "No FilePath elements found in output"
    video_paths = [p for p in file_paths if p.endswith(".mp4")]
    assert video_paths, "No .mp4 FilePath found"
    for fp in video_paths:
        assert fp.startswith("/"), f"Expected absolute path, got: {fp!r}"


@patch("stages.prproj_builder._probe_video", return_value=(1920, 1080, 29.97, 10.0))
def test_build_prproj_relative_paths_tts_wav(mock_probe, tmp_path):
    """With relative_paths=True, WAV FilePath elements use 'tts/<filename>' format."""
    template = _make_fake_prproj(tmp_path, MINIMAL_XML)
    video = tmp_path / "my_video.mp4"
    video.write_bytes(b"fake")

    tts_dir = tmp_path / "tts"
    tts_dir.mkdir()
    wav = tts_dir / "abc123.wav"
    wav.write_bytes(b"fake wav")

    overlay = _overlay(start=0.5, end=2.0, text_ja="テスト")
    tts_map = {"テスト": wav}
    output = tmp_path / "out.prproj"

    prproj_builder.build_prproj(
        template, video, [overlay], tts_map, output, relative_paths=True
    )

    root = prproj_builder.load_template(output)
    file_paths = [el.text for el in root.iter("FilePath") if el.text]
    wav_paths = [p for p in file_paths if p.endswith(".wav")]
    assert wav_paths, "No .wav FilePath found in output"
    for fp in wav_paths:
        assert fp == "tts/abc123.wav", f"Expected 'tts/abc123.wav', got: {fp!r}"
        assert not fp.startswith("/"), f"Expected relative path, got: {fp!r}"


def test_export_prproj_run_passes_relative_paths(tmp_path, mocker):
    """export_prproj.run passes relative_paths parameter to build_prproj."""
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
