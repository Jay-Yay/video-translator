import base64
import struct
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from stages import prproj_builder
from models.text_overlay import TextOverlay


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_fake_prproj(tmp_path, xml_content: str) -> Path:
    """Create a gzip-compressed .prproj file from XML string."""
    raw = xml_content.encode("utf-8")
    compress_obj = zlib.compressobj(
        zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, zlib.MAX_WBITS | 16
    )
    compressed = compress_obj.compress(raw) + compress_obj.flush()
    path = tmp_path / "fake.prproj"
    path.write_bytes(compressed)
    return path


def _overlay_at(start: float, end: float, text_ja: str, bbox=(100, 200, 300, 50)):
    return TextOverlay(
        text_ko="한국어",
        text_ja=text_ja,
        bbox=bbox,
        start_sec=start,
        end_sec=end,
        confidence=0.95,
    )


# Minimal XML matching the real base.prproj structure
MINIMAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
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
        <Track Index="0" ObjectURef="v1-track-uid"/>
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


def _make_prproj_with_text_clip(tmp_path) -> Path:
    return _make_fake_prproj(tmp_path, MINIMAL_XML)


# ── sec_to_ticks ───────────────────────────────────────────────────────


def test_one_second_is_254016000000_ticks():
    # TrackItem Start/End use the sequence scale: 254016000000 ticks/sec
    assert prproj_builder.sec_to_ticks(1.0) == 254016000000


def test_zero_seconds():
    assert prproj_builder.sec_to_ticks(0.0) == 0


def test_fractional_seconds():
    assert prproj_builder.sec_to_ticks(2.5) == 635040000000


# ── Task 4: load_template + save_prproj ────────────────────────────────


def test_load_template_returns_element_tree_root(tmp_path):
    prproj_path = _make_fake_prproj(tmp_path, MINIMAL_XML)
    root = prproj_builder.load_template(prproj_path)
    assert root.tag == "PremiereData"


def test_save_prproj_round_trips(tmp_path):
    prproj_path = _make_fake_prproj(tmp_path, MINIMAL_XML)
    root = prproj_builder.load_template(prproj_path)
    out = tmp_path / "out.prproj"
    prproj_builder.save_prproj(root, out)
    root2 = prproj_builder.load_template(out)
    assert root2.find(".//FrameRect").text == "0,0,1920,1080"


def test_decompress_standard_zlib(tmp_path):
    """Ensure standard zlib compressed data also works."""
    raw = b"<Root><Child>data</Child></Root>"
    compressed = zlib.compress(raw)
    path = tmp_path / "std.prproj"
    path.write_bytes(compressed)
    root = prproj_builder.load_template(path)
    assert root.tag == "Root"


def test_decompress_invalid_raises(tmp_path):
    path = tmp_path / "bad.prproj"
    path.write_bytes(b"not compressed at all")
    with pytest.raises(ValueError, match="Cannot decompress"):
        prproj_builder.load_template(path)


# ── Task 5: update_sequence_settings + new_object_id ───────────────────


def test_update_sequence_settings_sets_dimensions(tmp_path):
    prproj_path = _make_fake_prproj(tmp_path, MINIMAL_XML)
    root = prproj_builder.load_template(prproj_path)
    prproj_builder.update_sequence_settings(root, video_w=3840, video_h=2160, fps=25.0)
    vtg = root.find(f".//*[@ObjectID='66']")
    assert vtg.find("FrameRect").text == "0,0,3840,2160"


def test_update_sequence_settings_sets_preview_size(tmp_path):
    prproj_path = _make_fake_prproj(tmp_path, MINIMAL_XML)
    root = prproj_builder.load_template(prproj_path)
    prproj_builder.update_sequence_settings(root, video_w=1280, video_h=720, fps=30.0)
    width_el = root.find(".//MZ.Sequence.PreviewFrameSizeWidth")
    height_el = root.find(".//MZ.Sequence.PreviewFrameSizeHeight")
    assert width_el.text == "1280"
    assert height_el.text == "720"


def test_new_object_id_is_unique():
    ids = {prproj_builder.new_object_id() for _ in range(100)}
    assert len(ids) == 100


def test_new_object_id_is_integer_string():
    oid = prproj_builder.new_object_id()
    assert oid.isdigit()


# ── Task 6: clone_text_clip ────────────────────────────────────────────


def test_patch_source_text_blob_replaces_text():
    original_b64 = (
        "RAEAAAAAAABEMyIRDAAAAAAABgAKAAQABgAAAGQAAAAAAF4AGAAQAAwAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFQAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAABcABwBeAAAAAAAAARAAAAAc"
        "AAAANAAAAAAAAQBg////ZP///2j///9s////AQAAAAQAAAAMAAAATHVjaWRhR3Jh"
        "bmRlAAAAAAEAAAAMAAAACAAOAAQACAAIAAAAaAAAADwAAAAAADYAFAAAAAAAAAAA"
        "AAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAACAAEADYAAAAC"
        "AAAADAAAAAwAAAAAAIBA9P////j////8////BAAEAAQAAAALAAAAUExBQ0VIT0xE"
        "RVIA"
    )
    # Use the real blob from the template
    real_b64 = "RAEAAAAAAABEMyIRDAAAAAAABgAKAAQABgAAAGQAAAAAAF4AGAAQAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAABcABwBeAAAAAAAAARAAAAAcAAAANAAAAAAAAQBg////ZP///2j///9s////AQAAAAQAAAAMAAAATHVjaWRhR3JhbmRlAAAAAAEAAAAMAAAACAAOAAQACAAIAAAAaAAAADwAAAAAADYAFAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAACAAEADYAAAACAAAADAAAAAwAAAAAAIBA9P////j////8////BAAEAAQAAAALAAAAUExBQ0VIT0xERVIA"
    new_b64 = prproj_builder._patch_source_text_blob(real_b64, "こんにちは")
    raw = base64.b64decode(new_b64)
    # Check that the new text is in the blob
    text_bytes = "こんにちは".encode("utf-8")
    assert text_bytes in raw
    # Check that PLACEHOLDER is gone
    assert b"PLACEHOLDER" not in raw
    # Check text length field
    header_size = struct.unpack_from("<I", raw, 0)[0]
    text_len = struct.unpack_from("<I", raw, header_size - 4)[0]
    assert text_len == len(text_bytes)


def test_clone_text_clip_creates_new_ids(tmp_path):
    prproj_path = _make_prproj_with_text_clip(tmp_path)
    root = prproj_builder.load_template(prproj_path)
    template_clip = root.find(f".//*[@ObjectID='79']")
    overlay = _overlay_at(1.0, 3.0, "テスト")
    cloned_clip, cloned_refs = prproj_builder.clone_text_clip(
        root, template_clip, overlay, 1920, 1080
    )
    # The cloned clip should have a different ObjectID
    assert cloned_clip.get("ObjectID") != "79"
    # All cloned refs should have remapped IDs
    original_ids = {"93", "94", "111", "112", "123", "125"}
    cloned_ids = {e.get("ObjectID") for e in cloned_refs if e.get("ObjectID")}
    assert cloned_ids.isdisjoint(original_ids)


def test_clone_text_clip_sets_timing(tmp_path):
    prproj_path = _make_prproj_with_text_clip(tmp_path)
    root = prproj_builder.load_template(prproj_path)
    template_clip = root.find(f".//*[@ObjectID='79']")
    overlay = _overlay_at(1.5, 4.0, "タイミング")
    cloned_clip, _ = prproj_builder.clone_text_clip(
        root, template_clip, overlay, 1920, 1080
    )
    ti = cloned_clip.find(".//TrackItem")
    assert ti.find("End").text == str(prproj_builder.sec_to_ticks(4.0))
    assert ti.find("Start").text == str(prproj_builder.sec_to_ticks(1.5))


def test_clone_text_clip_sets_position(tmp_path):
    prproj_path = _make_prproj_with_text_clip(tmp_path)
    root = prproj_builder.load_template(prproj_path)
    template_clip = root.find(f".//*[@ObjectID='79']")
    # bbox at center of frame
    overlay = _overlay_at(0.0, 1.0, "位置テスト", bbox=(860, 490, 200, 100))
    _, cloned_refs = prproj_builder.clone_text_clip(
        root, template_clip, overlay, 1920, 1080
    )
    # Find the position param in cloned refs
    pos_param = None
    for elem in cloned_refs:
        name = elem.find("Name")
        if name is not None and name.text == "Position":
            pos_param = elem
            break
    assert pos_param is not None
    kf = pos_param.find("StartKeyframe")
    # Center of 1920x1080: (960/1920):(540/1080) = 0.5:0.5
    assert "0.5:0.5" in kf.text


def test_set_position_keyframe():
    elem = ET.fromstring(
        '<PointComponentParam>'
        '<StartKeyframe>-91445760000000000,0.3:0.5,0,0,0,0,0,0,5,4,0,0,0,0</StartKeyframe>'
        '</PointComponentParam>'
    )
    prproj_builder._set_position_keyframe(elem, 0.75, 0.25)
    kf = elem.find("StartKeyframe").text
    parts = kf.split(",")
    assert parts[1] == "0.75:0.25"
    # Other parts should be unchanged
    assert parts[0] == "-91445760000000000"
    assert parts[2] == "0"


# ── Task 7: build_prproj ──────────────────────────────────────────────


def test_build_prproj_creates_output_file(tmp_path):
    template_path = _make_prproj_with_text_clip(tmp_path)
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    overlays = [_overlay_at(0.5, 2.0, "こんにちは")]
    tts_map = {}
    output = tmp_path / "out.prproj"

    with patch("stages.prproj_builder._probe_video") as mock_probe:
        mock_probe.return_value = (1920, 1080, 29.97, 300.0)
        prproj_builder.build_prproj(template_path, video, overlays, tts_map, output)

    assert output.exists()
    root = prproj_builder.load_template(output)
    assert root.tag == "PremiereData"


def test_build_prproj_injects_text_clips(tmp_path):
    template_path = _make_prproj_with_text_clip(tmp_path)
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    overlays = [
        _overlay_at(0.5, 2.0, "クリップ1"),
        _overlay_at(3.0, 5.0, "クリップ2"),
    ]
    output = tmp_path / "out.prproj"

    with patch("stages.prproj_builder._probe_video") as mock_probe:
        mock_probe.return_value = (1920, 1080, 29.97, 300.0)
        prproj_builder.build_prproj(template_path, video, overlays, {}, output)

    root = prproj_builder.load_template(output)
    # Should have new VideoClipTrackItem elements (the originals plus clones)
    vctis = root.findall(".//VideoClipTrackItem")
    # Template clip is removed; 2 clones should be present
    assert len(vctis) == 2


def test_build_prproj_skips_empty_text(tmp_path):
    template_path = _make_prproj_with_text_clip(tmp_path)
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    overlays = [
        _overlay_at(0.5, 2.0, "有効"),
        _overlay_at(3.0, 5.0, ""),  # empty — should be skipped
    ]
    output = tmp_path / "out.prproj"

    with patch("stages.prproj_builder._probe_video") as mock_probe:
        mock_probe.return_value = (1920, 1080, 29.97, 300.0)
        prproj_builder.build_prproj(template_path, video, overlays, {}, output)

    root = prproj_builder.load_template(output)
    # Only 1 overlay should be injected (not 2)
    # Template clip removed; only the 1 valid overlay was cloned
    vctis = root.findall(".//VideoClipTrackItem")
    assert len(vctis) == 1


def test_build_prproj_removes_template_clip_from_root(tmp_path):
    template_path = _make_prproj_with_text_clip(tmp_path)
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    overlays = [_overlay_at(0.5, 2.0, "クリップ")]
    output = tmp_path / "out.prproj"

    with patch("stages.prproj_builder._probe_video") as mock_probe:
        mock_probe.return_value = (1920, 1080, 29.97, 300.0)
        prproj_builder.build_prproj(template_path, video, overlays, {}, output)

    root = prproj_builder.load_template(output)
    # Template clip (ObjectID="79") and its refs should be gone
    assert root.find(".//*[@ObjectID='79']") is None


def test_build_prproj_mutes_a1(tmp_path):
    template_path = _make_prproj_with_text_clip(tmp_path)
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "out.prproj"

    # A non-empty tts_map triggers A1 muting (no TTS → no mute)
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"fake")
    tts_map = {"dummy": dummy_wav}

    with patch("stages.prproj_builder._probe_video") as mock_probe:
        mock_probe.return_value = (1920, 1080, 29.97, 300.0)
        prproj_builder.build_prproj(template_path, video, [], tts_map, output)

    root = prproj_builder.load_template(output)
    a1 = root.find(f".//*[@ObjectUID='{prproj_builder.A1_TRACK_UID}']")
    mute = a1.find(".//TL.SQTrackMuted")
    assert mute is not None
    assert mute.text == "1"


def test_build_prproj_template_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Template not found"):
        prproj_builder.build_prproj(
            tmp_path / "nonexistent.prproj",
            tmp_path / "v.mp4",
            [],
            {},
            tmp_path / "out.prproj",
        )


def test_probe_video_fallback(tmp_path):
    """When cv2 can't open video, fallback dimensions are used."""
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"not a video")
    w, h, fps, dur = prproj_builder._probe_video(video)
    assert w == 1920
    assert h == 1080
    assert fps == pytest.approx(29.97)
    assert dur == 0.0


def test_build_tts_audio_chain_returns_elements_and_acti_id():
    """_build_tts_audio_chain returns a list of elements and an AudioClipTrackItem ObjectID."""
    from pathlib import Path
    import xml.etree.ElementTree as ET
    elems, acti_id = prproj_builder._build_tts_audio_chain(Path("/tmp/x.mp3"), 1.0, 3.0)
    # Must return non-empty list and a string ID
    assert elems
    assert isinstance(acti_id, str)
    # The last element must be the AudioClipTrackItem
    acti = elems[-1]
    assert acti.tag == "AudioClipTrackItem"
    assert acti.get("ObjectID") == acti_id
    # Check TrackItem Start/End are set correctly
    ti = acti.find(".//TrackItem")
    assert ti is not None
    start_ticks = prproj_builder.sec_to_ticks(1.0)
    end_ticks = prproj_builder.sec_to_ticks(3.0)
    assert ti.findtext("Start") == str(start_ticks)
    assert ti.findtext("End") == str(end_ticks)
    # All elements must have a ClassID (PP rejects elements without one)
    for elem in elems:
        if elem.tag != "Media":  # Media uses ObjectUID, not ObjectID
            assert elem.get("ClassID"), f"{elem.tag} missing ClassID"
