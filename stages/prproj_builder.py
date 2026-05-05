"""Premiere Pro project file builder with coordinate and timecode math utilities."""

from __future__ import annotations

import base64
import copy
import itertools
import logging
import struct
import uuid
import wave
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

# PP uses two different tick rates:
#   - _PP_TICKS_PER_SEC (254 billion): TrackItem Start/End on the sequence timeline,
#     VideoStream Duration, and all sequence-position values.
#   - _PP_CLIP_TICKS_PER_SEC (254 million): VideoClip InPoint/OutPoint (clip-internal
#     timescale). These are 1/1000 of the sequence scale.
_PP_TICKS_PER_SEC = 254016000000
_PP_CLIP_TICKS_PER_SEC = 254016000

# ── Real ObjectUIDs / ObjectIDs from templates/base.prproj ──────────────
# Video track V1 (Index 0) — original video goes here
V1_TRACK_UID = "e69ab9cb-ac32-4fd4-935b-b9af7c9b3eed"
# Video track V2 (Index 1) — text overlays go here
V2_TRACK_UID = "0412bfac-213d-427e-9419-4c61829333c4"
# Template text clip on V2 (VideoClipTrackItem)
TEMPLATE_TEXT_CLIP_OBJ_ID = "79"
# Audio track A1 (Index 0) — original Korean audio, to be muted
A1_TRACK_UID = "9dbc1352-d0c0-4447-a4bb-71b4f0036267"
# Audio track A2 (Index 1) — Japanese TTS audio goes here
A2_TRACK_UID = "421fb4d8-eeb4-45d2-9ffa-b91edd886274"
# Audio track A3 (Index 2) — BGM (background music, vocals removed) goes here
A3_TRACK_UID = "c565d0f6-b023-451a-a07a-f7eba8009335"
# Video track V3 (Index 2) — overflow text overlays when multiple texts are simultaneous
V3_TRACK_UID = "0e169328-16c3-49e9-bfbb-a3b6261f4dab"
# Number of additional overflow tracks (V4, V5, ...) added programmatically
# beyond the V2/V3 in the template.  Three extra tracks lets us absorb up to
# six simultaneous overlays before any get dropped.
EXTRA_OVERFLOW_TRACK_COUNT = 3
# VideoTrackGroup ObjectID — holds FrameRect for sequence resolution
VIDEO_TRACK_GROUP_OBJ_ID = "66"
# ArbVideoComponentParam ObjectID for "Source Text" (the text binary blob)
SOURCE_TEXT_PARAM_OBJ_ID = "123"
# PointComponentParam ObjectID for "Position" (normalized x:y)
POSITION_PARAM_OBJ_ID = "125"
# VideoClip ObjectID holding InPoint/OutPoint
VIDEO_CLIP_OBJ_ID = "112"
# VideoFilterComponent ObjectID for the text effect (has InstanceName)
TEXT_FILTER_OBJ_ID = "111"
# VideoComponentChain for the text clip
TEXT_COMP_CHAIN_OBJ_ID = "93"
# SubClip ObjectID
SUBCLIP_OBJ_ID = "94"


# ── Math utilities ──────────────────────────────────────────────────────


def sec_to_ticks(sec: float) -> int:
    """Convert seconds to Premiere Pro internal ticks (1 tick = 1/254016000 s).

    Args:
        sec: time in seconds

    Returns:
        time in Premiere Pro ticks
    """
    return int(sec * _PP_TICKS_PER_SEC)


# ── Task 4: load / save ────────────────────────────────────────────────


def _decompress(data: bytes) -> bytes:
    """Try standard zlib, raw deflate, gzip — return first that succeeds."""
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 16):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    raise ValueError("Cannot decompress .prproj — unknown compression format")


def load_template(template_path: Path) -> ET.Element:
    """Decompress .prproj and return parsed XML root element."""
    data = template_path.read_bytes()
    xml_bytes = _decompress(data)
    return ET.fromstring(xml_bytes)


def save_prproj(root: ET.Element, output_path: Path) -> None:
    """Serialize root to XML, gzip-compress, and write to output_path.

    PP2026 .prproj files use gzip compression (wbits = MAX_WBITS | 16).
    The XML declaration must use double quotes and uppercase UTF-8 to match
    what Premiere Pro produces — single-quote/lowercase form causes load failure.
    """
    body = ET.tostring(root, encoding="unicode")
    xml_bytes = ('<?xml version="1.0" encoding="UTF-8" ?>\n' + body).encode("utf-8")
    # Use gzip format to match original compression
    compress_obj = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed = compress_obj.compress(xml_bytes) + compress_obj.flush()
    output_path.write_bytes(compressed)


# ── Task 5: sequence settings + new_object_id ──────────────────────────

_id_counter = itertools.count(100000)


def new_object_id() -> str:
    """Generate a unique integer ObjectID string."""
    return str(next(_id_counter))


def update_sequence_settings(
    root: ET.Element, video_w: int, video_h: int, fps: float
) -> None:
    """Update sequence resolution in-place.

    Updates every FrameRect in the document (VideoTrackGroup and all VCTIs),
    the preview size metadata, and the sequence zoom/work-area out point so
    the timeline view matches the actual video duration.
    """
    frame_rect = f"0,0,{video_w},{video_h}"

    # VideoTrackGroup.FrameRect defines the sequence canvas
    vtg = root.find(f".//VideoTrackGroup[@ObjectID='{VIDEO_TRACK_GROUP_OBJ_ID}']")
    if vtg is None:
        # Fallback: first element with that ObjectID (scoped IDs may collide)
        vtg = root.find(f".//*[@ObjectID='{VIDEO_TRACK_GROUP_OBJ_ID}']")
    if vtg is not None:
        fr_elem = vtg.find("FrameRect")
        if fr_elem is not None:
            fr_elem.text = frame_rect

    # Update ALL VideoClipTrackItem FrameRect values (text clips cloned from
    # the template retain the template's 1920×1080 value otherwise).
    for vcti in root.findall(".//VideoClipTrackItem"):
        fr = vcti.find("FrameRect")
        if fr is not None:
            fr.text = frame_rect

    # Preview/export size metadata
    for el in root.iter("MZ.Sequence.PreviewFrameSizeWidth"):
        el.text = str(video_w)
    for el in root.iter("MZ.Sequence.PreviewFrameSizeHeight"):
        el.text = str(video_h)


# ── Task 6: clone_text_clip ────────────────────────────────────────────
#
# PP2026 Essential Graphics text structure (from base.prproj inspection):
#
# VideoClipTrackItem (ObjectID="57")
#   └─ ClipTrackItem
#       ├─ ComponentOwner → Components (ObjectRef="69")
#       │   └─ VideoComponentChain (ObjectID="69")
#       │       └─ ComponentChain → Components
#       │           └─ VideoFilterComponent (ObjectID="85")
#       │               ├─ Component
#       │               │   ├─ Params → Param refs (97-118)
#       │               │   ├─ DisplayName = "Text"
#       │               │   └─ InstanceName = "PLACEHOLDER"
#       │               └─ MatchName = "AE.ADBE Text"
#       ├─ TrackItem → End (ticks)
#       └─ SubClip (ObjectRef="70")
#           └─ Clip (ObjectRef="86") → InPoint / OutPoint
#
# Key params inside VideoFilterComponent:
#   ObjectID 97: ArbVideoComponentParam "Source Text" — binary blob with text
#   ObjectID 99: PointComponentParam "Position" — normalized coords "x:y"


def _patch_source_text_blob(blob_b64: str, new_text: str) -> str:
    """Replace the text string inside the Source Text binary blob.

    The blob has a header (size recorded in first 4 bytes as LE uint32),
    followed by the text portion: [text_len:uint32_LE][text_bytes][null].

    We replace the text portion and update the header size field.
    """
    raw = base64.b64decode(blob_b64)
    # Original header size is the first uint32 LE — equals offset of text_len field
    old_header_size = struct.unpack_from("<I", raw, 0)[0]

    # Encode new text as UTF-8, null-terminated
    text_bytes = new_text.encode("utf-8")
    text_len = len(text_bytes)

    # The text portion starts at old_header_size:
    #   [text_len: 4 bytes LE][text_bytes][null terminator]
    header_data = bytes(raw[:old_header_size - 4])

    # Build new text portion
    text_portion = struct.pack("<I", text_len) + text_bytes + b"\x00"

    new_raw = header_data + text_portion

    return base64.b64encode(new_raw).decode("ascii")


def _set_position_keyframe(pos_elem: ET.Element, norm_x: float, norm_y: float) -> None:
    """Update a PointComponentParam StartKeyframe with new normalized position.

    The keyframe format is:
    '-91445760000000000,X:Y,0,0,0,0,0,0,5,4,0,0,0,0'
    """
    kf = pos_elem.find("StartKeyframe")
    if kf is not None and kf.text:
        parts = kf.text.split(",")
        # Second element is "X:Y"
        parts[1] = f"{norm_x}:{norm_y}"
        kf.text = ",".join(parts)


def _collect_related_objects(root: ET.Element, clip_elem: ET.Element) -> list[ET.Element]:
    """Collect all top-level elements that clip_elem references.

    Returns a list of elements that must be deep-copied along with the clip:
    - VideoComponentChain (69) → VideoFilterComponent (85) → all Params (97-118)
    - SubClip (70) → VideoClip (86) → VideoMediaSource (119) → Media, etc.

    We collect them by walking ObjectRef / ObjectURef attributes.
    """
    collected = []
    # Traverse the tree collecting all referenced objects
    visited: set[str] = set()
    _collect_refs_recursive(root, clip_elem, visited, collected)
    return collected


def _collect_refs_recursive(
    root: ET.Element,
    elem: ET.Element,
    visited: set[str],
    collected: list[ET.Element],
) -> None:
    """Recursively collect elements referenced by ObjectRef attributes."""
    for child in elem.iter():
        ref = child.get("ObjectRef")
        if ref and ref not in visited:
            visited.add(ref)
            target = root.find(f".//*[@ObjectID='{ref}']")
            if target is not None:
                collected.append(target)
                _collect_refs_recursive(root, target, visited, collected)


def clone_text_clip(
    root: ET.Element,
    template_clip: ET.Element,
    overlay: "TextOverlay",
    video_w: int,
    video_h: int,
) -> tuple[ET.Element, list[ET.Element]]:
    """Deep-copy the template text clip and its referenced objects, updating text/position/timing.

    Returns (cloned_clip, list_of_cloned_referenced_elements) so caller can
    append all of them to the XML tree.
    """
    # Collect all referenced elements from the template
    ref_elems = _collect_related_objects(root, template_clip)

    # Build a mapping of old ObjectID → new ObjectID for all elements we clone
    id_map: dict[str, str] = {}
    old_clip_id = template_clip.get("ObjectID", "")
    new_clip_id = new_object_id()
    id_map[old_clip_id] = new_clip_id

    for elem in ref_elems:
        old_id = elem.get("ObjectID", "")
        if old_id:
            id_map[old_id] = new_object_id()

    # Also handle ObjectUID references
    uid_map: dict[str, str] = {}
    for elem in ref_elems:
        old_uid = elem.get("ObjectUID", "")
        if old_uid:
            uid_map[old_uid] = str(uuid.uuid4())

    # Deep-copy everything
    cloned_clip = copy.deepcopy(template_clip)
    cloned_refs = [copy.deepcopy(e) for e in ref_elems]

    # Remap ObjectIDs in all cloned elements
    all_cloned = [cloned_clip] + cloned_refs
    for elem in all_cloned:
        _remap_ids(elem, id_map, uid_map)

    # --- Set text content ---
    # Find the Source Text param (ArbVideoComponentParam with Name="Source Text")
    source_text_new_id = id_map.get(SOURCE_TEXT_PARAM_OBJ_ID, "")
    for elem in cloned_refs:
        oid = elem.get("ObjectID", "")
        if oid == source_text_new_id:
            kf_val = elem.find("StartKeyframeValue")
            if kf_val is not None and kf_val.text:
                kf_val.text = _patch_source_text_blob(
                    kf_val.text.strip(), overlay.text_ja
                )
            break

    # --- Clear InstanceName ---
    # PP's Graphics panel shows the InstanceName of each EG clip.  Setting it
    # to empty prevents the translated text from cluttering that panel.
    text_filter_new_id = id_map.get(TEXT_FILTER_OBJ_ID, "")
    for elem in cloned_refs:
        oid = elem.get("ObjectID", "")
        if oid == text_filter_new_id:
            inst = elem.find(".//InstanceName")
            if inst is not None:
                inst.text = ""
            break

    # --- Set position ---
    pos_new_id = id_map.get(POSITION_PARAM_OBJ_ID, "")
    for elem in cloned_refs:
        oid = elem.get("ObjectID", "")
        if oid == pos_new_id:
            # Convert pixel bbox to normalized 0-1 position.
            # Clamp horizontally to keep the box anchor at least 15% from each
            # edge — Japanese translations are often longer than the source
            # Korean, and edge-pinned overlays would otherwise overflow off-
            # screen and get clipped.
            x, y, w, h = overlay.bbox
            norm_x = max(0.15, min(0.85, (x + w / 2) / video_w))
            norm_y = (y + h / 2) / video_h
            _set_position_keyframe(elem, norm_x, norm_y)
            break

    # --- Set timing (End tick on ClipTrackItem/TrackItem) ---
    ti = cloned_clip.find(".//TrackItem")
    if ti is not None:
        end_elem = ti.find("End")
        if end_elem is not None:
            end_elem.text = str(sec_to_ticks(overlay.end_sec))
        # Add Start if not present
        start_elem = ti.find("Start")
        if start_elem is None:
            start_elem = ET.SubElement(ti, "Start")
        start_elem.text = str(sec_to_ticks(overlay.start_sec))

    # Note: VideoClip InPoint/OutPoint for graphics use the synthetic-media
    # timescale (1-hour offset, different rate) — leave template values intact.

    return cloned_clip, cloned_refs


def _remap_ids(
    elem: ET.Element,
    id_map: dict[str, str],
    uid_map: dict[str, str],
) -> None:
    """Recursively remap ObjectID, ObjectRef, ObjectUID, ObjectURef in an element tree."""
    for node in elem.iter():
        # Remap ObjectID
        oid = node.get("ObjectID")
        if oid and oid in id_map:
            node.set("ObjectID", id_map[oid])
        # Remap ObjectRef
        oref = node.get("ObjectRef")
        if oref and oref in id_map:
            node.set("ObjectRef", id_map[oref])
        # Remap ObjectUID
        ouid = node.get("ObjectUID")
        if ouid and ouid in uid_map:
            node.set("ObjectUID", uid_map[ouid])
        # Remap ObjectURef
        ouref = node.get("ObjectURef")
        if ouref and ouref in uid_map:
            node.set("ObjectURef", uid_map[ouref])


# ── Task 7: build_prproj ───────────────────────────────────────────────

_FALLBACK_W, _FALLBACK_H, _FALLBACK_FPS = 1920, 1080, 29.97


def _probe_video(video_path: Path) -> tuple[int, int, float, float]:
    """Probe video dimensions, fps, and duration using OpenCV.

    Returns (width, height, fps, duration_sec).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s — using fallback dimensions", video_path.name)
        return _FALLBACK_W, _FALLBACK_H, _FALLBACK_FPS, 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or _FALLBACK_FPS
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return (w or _FALLBACK_W), (h or _FALLBACK_H), fps, duration_sec


def _build_video_media_chain(
    video_path: Path, duration_sec: float, video_w: int, video_h: int, fps: float,
    relative_paths: bool = False,
) -> tuple[list[ET.Element], str]:
    """Build the complete PP2026 video clip object graph for placement on V1.

    Chain (ClassIDs from base.prproj inspection):
      VideoStream → Media → VideoMediaSource → VideoClip(master)
      ClipLoggingInfo + ClipChannelGroupVectorSerializer → MasterClip
      VideoClip(timeline) → SubClip(→MasterClip) → VideoClipTrackItem

    Returns (list_of_root_elements, VideoClipTrackItem_ObjectID).
    """
    duration_ticks = sec_to_ticks(duration_sec)            # sequence scale (254B/s)
    duration_clip_ticks = int(duration_sec * _PP_CLIP_TICKS_PER_SEC)  # clip scale (254M/s)
    # FrameRate encoding: template stores 8475667200 for ~29.97fps.
    # Scale proportionally for other frame rates.
    frame_rate_enc = round(8475667200 * 29.97 / fps) if fps > 0 else 8475667200
    media_uid = str(uuid.uuid4())
    master_clip_uid = str(uuid.uuid4())

    # ── VideoStream ───────────────────────────────────────────────────────
    vs_id = new_object_id()
    vs = ET.Element(
        "VideoStream",
        ObjectID=vs_id,
        ClassID="a36e4719-3ec6-4a0c-ab11-8b4aab377aa5",
        Version="22",
    )
    ET.SubElement(vs, "Duration").text = str(duration_ticks)
    ET.SubElement(vs, "FrameRect").text = f"0,0,{video_w},{video_h}"
    ET.SubElement(vs, "FrameRate").text = str(frame_rate_enc)

    # ── Media ─────────────────────────────────────────────────────────────
    media = ET.Element(
        "Media",
        ObjectUID=media_uid,
        ClassID="7a5c103e-f3ac-4391-b6b4-7cc3d2f9a7ff",
        Version="30",
    )
    ET.SubElement(media, "VideoStream", ObjectRef=vs_id)
    fp = video_path.name if relative_paths else str(video_path.resolve())
    ET.SubElement(media, "FilePath").text = fp
    ET.SubElement(media, "ActualMediaFilePath").text = fp
    ET.SubElement(media, "Title").text = video_path.stem
    ET.SubElement(media, "Infinite").text = "false"

    # ── VideoMediaSource ──────────────────────────────────────────────────
    vms_id = new_object_id()
    vms = ET.Element(
        "VideoMediaSource",
        ObjectID=vms_id,
        ClassID="e64ddf74-8fac-4682-8aa8-0e0ca2248949",
        Version="2",
    )
    ms = ET.SubElement(vms, "MediaSource", Version="4")
    ET.SubElement(ms, "Content", Version="10")
    ET.SubElement(ms, "Media", ObjectURef=media_uid)
    ET.SubElement(vms, "OriginalDuration").text = str(duration_ticks)

    # ── VideoClip (master — registered in MasterClip) ─────────────────────
    vc_master_id = new_object_id()
    vc_master = ET.Element(
        "VideoClip",
        ObjectID=vc_master_id,
        ClassID="9308dbef-2440-4acb-9ab2-953b9a4e82ec",
        Version="11",
    )
    ce = ET.SubElement(vc_master, "Clip", Version="18")
    _node = ET.SubElement(ce, "Node", Version="1")
    _props = ET.SubElement(_node, "Properties", Version="1")
    ET.SubElement(_props, "BE.Prefs.SyntheticMedia.DefaultIsDropFrame").text = "true"
    ET.SubElement(ce, "Source", ObjectRef=vms_id)
    ET.SubElement(ce, "InPoint").text = "0"
    ET.SubElement(ce, "OutPoint").text = str(duration_clip_ticks)
    ET.SubElement(ce, "ClipID").text = str(uuid.uuid4())
    ET.SubElement(ce, "InUse").text = "false"

    # ── ClipLoggingInfo ───────────────────────────────────────────────────
    cli_id = new_object_id()
    cli = ET.Element(
        "ClipLoggingInfo",
        ObjectID=cli_id,
        ClassID="77ab7fdd-dcdf-465d-9906-7a330ca1e738",
        Version="10",
    )
    ET.SubElement(cli, "CaptureMode").text = "2"
    ET.SubElement(cli, "ClipName").text = video_path.stem
    ET.SubElement(cli, "TimecodeFormat").text = "102"
    ET.SubElement(cli, "MediaFrameRate").text = str(frame_rate_enc)

    # ── ClipChannelGroupVectorSerializer ──────────────────────────────────
    ccg_id = new_object_id()
    ccg = ET.Element(
        "ClipChannelGroupVectorSerializer",
        ObjectID=ccg_id,
        ClassID="a3127a8c-95d4-456e-a7f5-171b3f922426",
        Version="1",
    )

    # ── MasterClip ────────────────────────────────────────────────────────
    mc = ET.Element(
        "MasterClip",
        ObjectUID=master_clip_uid,
        ClassID="fb11c33a-b0a9-4465-aa94-b6d5db2628cf",
        Version="12",
    )
    ET.SubElement(mc, "LoggingInfo", ObjectRef=cli_id)
    clips_elem = ET.SubElement(mc, "Clips", Version="1")
    ET.SubElement(clips_elem, "Clip", Index="0", ObjectRef=vc_master_id)
    ET.SubElement(mc, "AudioClipChannelGroups", ObjectRef=ccg_id)
    ET.SubElement(mc, "Name").text = video_path.stem
    ET.SubElement(mc, "MasterClipChangeVersion").text = "0"

    # ── VideoClip (timeline — placed on track via SubClip) ────────────────
    vc_tl_id = new_object_id()
    vc_tl = ET.Element(
        "VideoClip",
        ObjectID=vc_tl_id,
        ClassID="9308dbef-2440-4acb-9ab2-953b9a4e82ec",
        Version="11",
    )
    ce2 = ET.SubElement(vc_tl, "Clip", Version="18")
    _node2 = ET.SubElement(ce2, "Node", Version="1")
    _props2 = ET.SubElement(_node2, "Properties", Version="1")
    ET.SubElement(_props2, "BE.Prefs.SyntheticMedia.DefaultIsDropFrame").text = "true"
    ET.SubElement(ce2, "Source", ObjectRef=vms_id)
    ET.SubElement(ce2, "InPoint").text = "0"
    ET.SubElement(ce2, "OutPoint").text = str(duration_clip_ticks)
    ET.SubElement(ce2, "ClipID").text = str(uuid.uuid4())

    # ── SubClip ───────────────────────────────────────────────────────────
    sc_id = new_object_id()
    sc = ET.Element(
        "SubClip",
        ObjectID=sc_id,
        ClassID="e0c58dc9-dbdd-4166-aef7-5db7e3f22e84",
        Version="6",
    )
    ET.SubElement(sc, "Clip", ObjectRef=vc_tl_id)
    ET.SubElement(sc, "MasterClip", ObjectURef=master_clip_uid)
    ET.SubElement(sc, "Name").text = video_path.stem
    ET.SubElement(sc, "OrigChGrp").text = "0"

    # ── VideoComponentChain (motion/opacity — required by every VCTI) ────────
    vcc_id = new_object_id()
    vcc = ET.Element(
        "VideoComponentChain",
        ObjectID=vcc_id,
        ClassID="0970e08a-f58f-4108-b29a-1a717b8e12e2",
        Version="3",
    )
    ET.SubElement(vcc, "DefaultMotion").text = "true"
    ET.SubElement(vcc, "DefaultOpacity").text = "true"
    ET.SubElement(vcc, "DefaultMotionComponentID").text = "1"
    ET.SubElement(vcc, "DefaultOpacityComponentID").text = "2"
    cc = ET.SubElement(vcc, "ComponentChain", Version="3")
    cc_node = ET.SubElement(cc, "Node", Version="1")
    cc_props = ET.SubElement(cc_node, "Properties", Version="1")
    ET.SubElement(cc_props, "MZ.ComponentChain.ActiveComponentID").text = "2"
    ET.SubElement(cc_props, "MZ.ComponentChain.ActiveComponentParamIndex").text = "4294967295"
    ET.SubElement(cc, "Components", Version="1")  # empty — no video filters

    # ── VideoClipTrackItem ────────────────────────────────────────────────
    vcti_id = new_object_id()
    vcti = ET.Element(
        "VideoClipTrackItem",
        ObjectID=vcti_id,
        ClassID="368b0406-29e3-4923-9fcd-094fbf9a1089",
        Version="8",
    )
    cti = ET.SubElement(vcti, "ClipTrackItem", Version="8")
    # ComponentOwner must come before TrackItem — required for PP to parse VCTI
    co = ET.SubElement(cti, "ComponentOwner", Version="1")
    ET.SubElement(co, "Components", ObjectRef=vcc_id)
    ti = ET.SubElement(cti, "TrackItem", Version="4")
    ET.SubElement(ti, "Start").text = "0"
    ET.SubElement(ti, "End").text = str(duration_ticks)
    ET.SubElement(cti, "SubClip", ObjectRef=sc_id)
    # Required fields on every VideoClipTrackItem
    ET.SubElement(vcti, "PixelAspectRatio").text = "1,1"
    ET.SubElement(vcti, "ToneMapSettings").text = '{"peak":-1,"version":3}'
    ET.SubElement(vcti, "FrameRect").text = f"0,0,{video_w},{video_h}"

    return [vs, media, vms, vc_master, cli, ccg, mc, vc_tl, sc, vcc, vcti], vcti_id


def _find_track_by_uid(root: ET.Element, uid: str) -> ET.Element | None:
    """Find a track element by its ObjectUID attribute."""
    return root.find(f".//*[@ObjectUID='{uid}']")


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
    """Patch an existing V1 VideoClipTrackItem in the template to point at video_path.

    When the template already has a real video on V1 (i.e. the user placed any
    MP4 in PP before saving the template), we update only the file path and
    timing instead of fabricating a whole new object graph.  This is far more
    reliable because the Media/VideoStream objects come from PP itself.

    Returns True if a clip was found and patched, False otherwise.
    """
    track_items = v1_track.find(".//TrackItems")
    if track_items is None:
        return False
    ti_ref = track_items.find("TrackItem")
    if ti_ref is None:
        return False

    vcti_id = ti_ref.get("ObjectRef")
    vcti = root.find(f".//VideoClipTrackItem[@ObjectID='{vcti_id}']")
    if vcti is None:
        return False

    duration_ticks = sec_to_ticks(duration_sec)                        # sequence scale (254B/s)
    duration_clip_ticks = int(duration_sec * _PP_CLIP_TICKS_PER_SEC)  # clip scale (254M/s)
    frame_rate_enc = round(8475667200 * 29.97 / fps) if fps > 0 else 8475667200
    fp = video_path.name if relative_paths else str(video_path.resolve())

    # Patch TrackItem End
    ti = vcti.find(".//TrackItem")
    if ti is not None:
        end = ti.find("End")
        if end is not None:
            end.text = str(duration_ticks)
        start = ti.find("Start")
        if start is None:
            start = ET.SubElement(ti, "Start")
        start.text = "0"

    # Patch VideoClip InPoint/OutPoint (both master and timeline copies)
    sc_ref_el = vcti.find(".//SubClip")
    if sc_ref_el is not None:
        sc_id_val = sc_ref_el.get("ObjectRef", "")
        sc = root.find(f".//SubClip[@ObjectID='{sc_id_val}']")
        if sc is not None:
            vc_tl_ref = sc.find("Clip")
            mc_uref_el = sc.find("MasterClip")
            if vc_tl_ref is not None and mc_uref_el is not None:
                mc_uid_val = mc_uref_el.get("ObjectURef", "")
                mc = root.find(f".//*[@ObjectUID='{mc_uid_val}']")
                vc_tl_id = vc_tl_ref.get("ObjectRef", "")
                vc_master_id = ""
                if mc is not None:
                    vc_master_ref_el = mc.find(".//Clips/Clip")
                    if vc_master_ref_el is not None:
                        vc_master_id = vc_master_ref_el.get("ObjectRef", "")
                for vc in root.findall(".//VideoClip"):
                    clip = vc.find("Clip")
                    if clip is None:
                        continue
                    if vc.get("ObjectID", "") in (vc_tl_id, vc_master_id):
                        inp = clip.find("InPoint")
                        outp = clip.find("OutPoint")
                        if inp is not None:
                            inp.text = "0"
                        if outp is not None:
                            outp.text = str(duration_clip_ticks)

    # Patch Media FilePath / ActualMediaFilePath / Title
    sc_ref_elem = vcti.find(".//SubClip")
    if sc_ref_elem is not None:
        sc_id_for_media = sc_ref_elem.get("ObjectRef", "")
        sc = root.find(f".//SubClip[@ObjectID='{sc_id_for_media}']")
        if sc is not None:
            mc_uref_val = sc.find("MasterClip")
            if mc_uref_val is not None:
                mc_uid = mc_uref_val.get("ObjectURef", "")
                mc = root.find(f".//*[@ObjectUID='{mc_uid}']")
                if mc is not None:
                    name_el = mc.find("Name")
                    if name_el is not None:
                        name_el.text = video_path.stem

    # Find Media via VideoMediaSource chain
    for vms in root.findall(".//VideoMediaSource"):
        ms = vms.find("MediaSource")
        if ms is None:
            continue
        media_uref = ms.find("Media")
        if media_uref is None:
            continue
        media_uid_val = media_uref.get("ObjectURef", "")
        media = root.find(f".//*[@ObjectUID='{media_uid_val}']")
        if media is None:
            continue
        for tag in ("FilePath", "ActualMediaFilePath"):
            el = media.find(tag)
            if el is not None:
                el.text = fp
        title = media.find("Title")
        if title is not None:
            title.text = video_path.stem
        # Patch VideoStream Duration / FrameRect / FrameRate
        vs_ref = media.find("VideoStream")
        if vs_ref is not None:
            vs_oid = vs_ref.get("ObjectRef", "")
            vs = root.find(f".//VideoStream[@ObjectID='{vs_oid}']")
            if vs is not None:
                dur_el = vs.find("Duration")
                if dur_el is not None:
                    dur_el.text = str(duration_ticks)
                fr_el = vs.find("FrameRect")
                if fr_el is not None:
                    fr_el.text = f"0,0,{video_w},{video_h}"
                frate_el = vs.find("FrameRate")
                if frate_el is not None:
                    frate_el.text = str(frame_rate_enc)
        break  # only one video MediaSource on V1

    logger.info("Patched existing V1 clip → %s (%.1f s)", video_path.name, duration_sec)
    return True


def _get_or_create_track_items(track: ET.Element) -> ET.Element:
    """Get or create the TrackItems element within a ClipTrack/ClipItems."""
    clip_items = track.find(".//ClipItems")
    if clip_items is None:
        clip_track = track.find("ClipTrack")
        if clip_track is None:
            clip_track = ET.SubElement(track, "ClipTrack")
        clip_items = ET.SubElement(clip_track, "ClipItems")
    track_items = clip_items.find("TrackItems")
    if track_items is None:
        track_items = ET.Element("TrackItems", Version="1")
        clip_items.insert(0, track_items)
    return track_items


def _mute_track(track: ET.Element) -> None:
    """Mute an audio track by setting its fader Volume to 0.

    In PP2026, audio track muting is done via the AudioFader component's
    Volume param or by directly manipulating the Mute param.
    We find the AudioFader's Mute param (referenced through the track's
    ComponentOwner) and set its StartKeyframe to indicate muted state.

    As a simpler approach, we add a property to the track's Node/Properties.
    """
    # Find the Properties element in the track
    props = track.find(".//Properties")
    if props is not None:
        mute_elem = props.find("TL.SQTrackMuted")
        if mute_elem is None:
            mute_elem = ET.SubElement(props, "TL.SQTrackMuted")
        mute_elem.text = "1"


def _build_tts_audio_chain(
    wav_path: Path, start_sec: float, end_sec: float,
    relative_paths: bool = False,
) -> tuple[list[ET.Element], str]:
    """Build the PP2026 audio clip object graph for a 48kHz stereo WAV placed on an audio track.

    Chain mirrors the A1 AudioClipTrackItem in base.prproj:
      AudioStream → Media → AudioMediaSource → AudioClip(master)
      MasterClip(→AudioClip master) + AudioClip(timeline) → SubClip
      AudioComponentChain + SubClip → AudioClipTrackItem

    Returns (list_of_elements, AudioClipTrackItem_ObjectID).

    Tick rate for audio: all values (TrackItem Start/End, AudioClip InPoint/OutPoint,
    AudioStream Duration) use the 254B scale (_PP_TICKS_PER_SEC).
    """
    clip_dur_sec = end_sec - start_sec
    start_ticks = sec_to_ticks(start_sec)
    end_ticks = sec_to_ticks(end_sec)
    clip_ticks = sec_to_ticks(clip_dur_sec)
    # Use a generous OutPoint for master clip (whole wav may be longer than overlay)
    # 60-second cap is safe for all TTS clips; TrackItem End controls actual playback end
    master_out = sec_to_ticks(max(clip_dur_sec, 60.0))

    media_uid = str(uuid.uuid4())
    master_clip_uid = str(uuid.uuid4())
    fp = f"tts/{wav_path.name}" if relative_paths else str(wav_path.resolve())
    name = wav_path.name

    # 48kHz stereo PCM WAV (ElevenLabs MP3 converted by ffmpeg in stages/tts.py).
    # FrameRate = _PP_TICKS_PER_SEC / sample_rate_hz = 254016000000 / 48000
    audio_frame_rate = _PP_TICKS_PER_SEC // 48000  # 5292000
    stereo_layout = '[{"channellabel":100},{"channellabel":101}]'

    # ── AudioStream ───────────────────────────────────────────────────────
    as_id = new_object_id()
    as_elem = ET.Element(
        "AudioStream",
        ObjectID=as_id,
        ClassID="0b5cf52f-2b85-4863-890b-8844b64ecfe9",
        Version="8",
    )
    ET.SubElement(as_elem, "AudioChannelLayout").text = stereo_layout
    ET.SubElement(as_elem, "Duration").text = str(master_out)
    ET.SubElement(as_elem, "SampleType").text = "7"
    ET.SubElement(as_elem, "FrameRate").text = str(audio_frame_rate)

    # ── Media (mp3 file container) ────────────────────────────────────────
    media = ET.Element(
        "Media",
        ObjectUID=media_uid,
        ClassID="7a5c103e-f3ac-4391-b6b4-7cc3d2f9a7ff",
        Version="30",
    )
    ET.SubElement(media, "AudioStream", ObjectRef=as_id)
    ET.SubElement(media, "FilePath").text = fp
    ET.SubElement(media, "ActualMediaFilePath").text = fp
    ET.SubElement(media, "Title").text = name
    ET.SubElement(media, "Infinite").text = "false"

    # ── AudioMediaSource ──────────────────────────────────────────────────
    ams_id = new_object_id()
    ams = ET.Element(
        "AudioMediaSource",
        ObjectID=ams_id,
        ClassID="f588da05-fc2a-4fbc-9383-74d653b379e3",
        Version="2",
    )
    ms = ET.SubElement(ams, "MediaSource", Version="4")
    ET.SubElement(ms, "Content", Version="10")
    ET.SubElement(ms, "Media", ObjectURef=media_uid)
    ET.SubElement(ams, "OriginalDuration").text = str(master_out)

    # ── AudioClip (master — registered in MasterClip.Clips) ───────────────
    ac_master_id = new_object_id()
    ac_master = ET.Element(
        "AudioClip",
        ObjectID=ac_master_id,
        ClassID="b8830d03-de02-41ee-84ec-fe566dc70cd9",
        Version="8",
    )
    ce_m = ET.SubElement(ac_master, "Clip", Version="18")
    ET.SubElement(ce_m, "Source", ObjectRef=ams_id)
    ET.SubElement(ce_m, "InPoint").text = "0"
    ET.SubElement(ce_m, "OutPoint").text = str(master_out)
    ET.SubElement(ce_m, "ClipID").text = str(uuid.uuid4())
    ET.SubElement(ac_master, "AudioChannelLayout").text = stereo_layout

    # ── MasterClip ────────────────────────────────────────────────────────
    mc = ET.Element(
        "MasterClip",
        ObjectUID=master_clip_uid,
        ClassID="fb11c33a-b0a9-4465-aa94-b6d5db2628cf",
        Version="12",
    )
    clips_elem = ET.SubElement(mc, "Clips", Version="1")
    ET.SubElement(clips_elem, "Clip", Index="0", ObjectRef=ac_master_id)
    ET.SubElement(mc, "Name").text = name
    ET.SubElement(mc, "MasterClipChangeVersion").text = "0"

    # ── AudioClip (timeline — placed on track via SubClip) ────────────────
    ac_tl_id = new_object_id()
    ac_tl = ET.Element(
        "AudioClip",
        ObjectID=ac_tl_id,
        ClassID="b8830d03-de02-41ee-84ec-fe566dc70cd9",
        Version="8",
    )
    ce_t = ET.SubElement(ac_tl, "Clip", Version="18")
    ET.SubElement(ce_t, "Source", ObjectRef=ams_id)
    ET.SubElement(ce_t, "InPoint").text = "0"
    ET.SubElement(ce_t, "OutPoint").text = str(clip_ticks)
    ET.SubElement(ce_t, "ClipID").text = str(uuid.uuid4())
    ET.SubElement(ac_tl, "AudioChannelLayout").text = stereo_layout

    # ── SecondaryContents (channel routing — required for audio playback) ──
    # Template AudioClip 92 (A1, which works) has two SecondaryContent items,
    # one per stereo channel, each pointing back to the AudioMediaSource.
    # Without these PP cannot route the clip audio to the stereo output bus
    # and the track plays silently even though the waveform is visible.
    sc_ch0_id = new_object_id()
    sc_ch0 = ET.Element(
        "SecondaryContent",
        ObjectID=sc_ch0_id,
        ClassID="f9d004b5-cb04-4e2f-af6f-64fadc2c4be9",
        Version="1",
    )
    ET.SubElement(sc_ch0, "Content", ObjectRef=ams_id)
    ET.SubElement(sc_ch0, "ChannelIndex").text = "0"

    sc_ch1_id = new_object_id()
    sc_ch1 = ET.Element(
        "SecondaryContent",
        ObjectID=sc_ch1_id,
        ClassID="f9d004b5-cb04-4e2f-af6f-64fadc2c4be9",
        Version="1",
    )
    ET.SubElement(sc_ch1, "Content", ObjectRef=ams_id)
    ET.SubElement(sc_ch1, "ChannelIndex").text = "1"

    sec_contents = ET.SubElement(ac_tl, "SecondaryContents", Version="1")
    ET.SubElement(sec_contents, "SecondaryContentItem", Index="0", ObjectRef=sc_ch0_id)
    ET.SubElement(sec_contents, "SecondaryContentItem", Index="1", ObjectRef=sc_ch1_id)

    # ── SubClip ───────────────────────────────────────────────────────────
    sc_id = new_object_id()
    sc = ET.Element(
        "SubClip",
        ObjectID=sc_id,
        ClassID="e0c58dc9-dbdd-4166-aef7-5db7e3f22e84",
        Version="6",
    )
    ET.SubElement(sc, "Clip", ObjectRef=ac_tl_id)
    ET.SubElement(sc, "MasterClip", ObjectURef=master_clip_uid)
    ET.SubElement(sc, "Name").text = name
    ET.SubElement(sc, "OrigChGrp").text = "0"

    # ── AudioComponentChain ───────────────────────────────────────────────
    acc_id = new_object_id()
    acc = ET.Element(
        "AudioComponentChain",
        ObjectID=acc_id,
        ClassID="3cb131d1-d3c0-47ae-a19a-bdf75ea11674",
        Version="4",
    )
    ET.SubElement(acc, "DefaultVol").text = "true"
    ET.SubElement(acc, "DefaultVolumeComponentID").text = "1"
    ET.SubElement(acc, "DefaultChannelVolumeComponentID").text = "2"
    cc = ET.SubElement(acc, "ComponentChain", Version="3")
    cc_node = ET.SubElement(cc, "Node", Version="1")
    cc_props = ET.SubElement(cc_node, "Properties", Version="1")
    ET.SubElement(cc_props, "MZ.ComponentChain.ActiveComponentID").text = "1"
    ET.SubElement(cc_props, "MZ.ComponentChain.ActiveComponentParamIndex").text = "4294967295"
    ET.SubElement(acc, "AudioChannelLayout").text = stereo_layout
    ET.SubElement(acc, "ChannelType").text = "1"  # 1 = stereo

    # ── AudioClipTrackItem ────────────────────────────────────────────────
    acti_id = new_object_id()
    acti = ET.Element(
        "AudioClipTrackItem",
        ObjectID=acti_id,
        ClassID="064ec682-9ba6-11d5-af2d-9ca32c7d6164",
        Version="11",
    )
    cti = ET.SubElement(acti, "ClipTrackItem", Version="8")
    co = ET.SubElement(cti, "ComponentOwner", Version="1")
    ET.SubElement(co, "Components", ObjectRef=acc_id)
    ti = ET.SubElement(cti, "TrackItem", Version="4")
    ET.SubElement(ti, "Start").text = str(start_ticks)
    ET.SubElement(ti, "End").text = str(end_ticks)
    ET.SubElement(cti, "SubClip", ObjectRef=sc_id)
    ET.SubElement(acti, "ID").text = str(uuid.uuid4())
    ET.SubElement(acti, "PreRenderComponentChainHashVersion").text = "1"

    return [as_elem, media, ams, ac_master, mc, sc_ch0, sc_ch1, ac_tl, sc, acc, acti], acti_id


def _add_overflow_video_tracks(root: ET.Element, count: int) -> list[str]:
    """Append `count` empty video tracks by cloning the V3 track structure.

    PP rejects sequences whose VideoTrackGroup.Tracks list references a UID
    that has no corresponding VideoClipTrack object — so each new entry in
    the Tracks list needs a matching cloned VideoClipTrack with a fresh UID,
    incremented Track ID, and incremented track Index.

    Returns the list of new ObjectUIDs in placement order (V4, V5, …).
    """
    if count <= 0:
        return []

    vtg = root.find(f".//VideoTrackGroup[@ObjectID='{VIDEO_TRACK_GROUP_OBJ_ID}']")
    if vtg is None:
        logger.warning("VideoTrackGroup not found — cannot add overflow tracks")
        return []
    tracks_elem = vtg.find("TrackGroup/Tracks")
    next_track_id_elem = vtg.find("TrackGroup/NextTrackID")
    if tracks_elem is None or next_track_id_elem is None:
        logger.warning("VideoTrackGroup malformed — cannot add overflow tracks")
        return []

    template_track = root.find(f".//*[@ObjectUID='{V3_TRACK_UID}']")
    if template_track is None:
        logger.warning("V3 track template not found — cannot add overflow tracks")
        return []

    starting_index = len(list(tracks_elem.findall("Track")))  # V3 → 3
    starting_track_id = int(next_track_id_elem.text or starting_index)

    new_uids: list[str] = []
    for offset in range(count):
        new_uid = str(uuid.uuid4())
        new_index = starting_index + offset
        new_track_id = starting_track_id + offset

        cloned = copy.deepcopy(template_track)
        cloned.set("ObjectUID", new_uid)
        # Rewrite all Track ID + Index fields so PP sees a distinct track
        id_elem = cloned.find("ClipTrack/Track/ID")
        if id_elem is not None:
            id_elem.text = str(new_track_id)
        for idx_elem in cloned.findall(".//Index"):
            idx_elem.text = str(new_index)

        root.append(cloned)
        ET.SubElement(
            tracks_elem, "Track", Index=str(new_index), ObjectURef=new_uid,
        )
        new_uids.append(new_uid)

    next_track_id_elem.text = str(starting_track_id + count)
    logger.info("Added %d overflow video tracks (V%d-V%d)",
                count, starting_index + 1, starting_index + count)
    return new_uids


def _dedup_overlays(overlays: list, video_duration_sec: float) -> list:
    """Return a non-overlapping subset of overlays suitable for a single timeline track.

    Two filtering passes:
    1. Drop "full-span" GVI artifacts: clips whose duration exceeds 50 % of the
       video duration (GVI sometimes extends a brief text annotation to cover the
       whole clip, producing a clip that blocks everything else).
    2. Greedy interval scheduling: sort by (start_sec, end_sec) so shorter clips
       at the same start time are preferred, then keep each clip only if it begins
       at or after the previous kept clip's end.
    """
    span_limit = video_duration_sec * 0.5
    filtered = [o for o in overlays if (o.end_sec - o.start_sec) < span_limit]

    # Sort: primary start_sec asc, secondary end_sec asc (prefer shorter clips)
    filtered.sort(key=lambda o: (o.start_sec, o.end_sec))

    result: list = []
    prev_end = 0.0
    for overlay in filtered:
        if overlay.start_sec >= prev_end:
            result.append(overlay)
            prev_end = overlay.end_sec

    dropped = len(overlays) - len(result)
    if dropped:
        logger.info(
            "Overlap dedup: kept %d / %d overlays (%d dropped: %d full-span, %d overlapping)",
            len(result), len(overlays), dropped,
            len(overlays) - len(filtered),
            len(filtered) - len(result),
        )
    return result


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
    """Build a Premiere Pro project from template with text overlays and TTS audio.

    Args:
        template_path: Path to base.prproj template
        video_path: Path to the source video file
        overlays: List of TextOverlay objects
        tts_map: Mapping of Japanese text → TTS WAV file path
        output_path: Where to write the output .prproj
        speech_segments: Translated speech segments with timing for TTS placement
        bgm_path: Optional path to background-music-only WAV (vocals removed by Demucs).
            If provided, placed on A3 so editors can blend BGM under the Japanese TTS.
    """
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path}\n"
            "Create it in Premiere Pro 2026 (see Task 3 in the implementation plan)."
        )

    root = load_template(template_path)
    video_w, video_h, fps, video_duration_sec = _probe_video(video_path)
    update_sequence_settings(root, video_w, video_h, fps)

    # Resolve overlapping overlays.
    # PP cannot handle overlapping clips on the same track — it only shows one
    # of them.  We peel off non-overlapping subsets onto V2, V3, V4, … so that
    # simultaneous overlays each land on their own track instead of being
    # dropped.
    has_translation = [o for o in overlays if o.text_ja]
    span_limit = video_duration_sec * 0.5
    eligible = [o for o in has_translation if (o.end_sec - o.start_sec) < span_limit]

    overflow_uids = _add_overflow_video_tracks(root, EXTRA_OVERFLOW_TRACK_COUNT)
    track_uids = [V2_TRACK_UID, V3_TRACK_UID, *overflow_uids]

    # Build per-track allocations by peeling off the lowest-start non-overlapping
    # subset each pass.  Stops when overlays run out or tracks run out.
    template_clip = root.find(f".//VideoClipTrackItem[@ObjectID='{TEMPLATE_TEXT_CLIP_OBJ_ID}']")

    if template_clip is None:
        logger.warning(
            "Template text clip not found — text overlays skipped. "
            "Check TEMPLATE_TEXT_CLIP_OBJ_ID constant."
        )
    else:
        remaining = list(eligible)
        track_allocations: list[tuple[str, list]] = []
        for uid in track_uids:
            if not remaining:
                break
            picked = _dedup_overlays(remaining, video_duration_sec)
            picked_ids = {id(o) for o in picked}
            track_allocations.append((uid, picked))
            remaining = [o for o in remaining if id(o) not in picked_ids]

        if remaining:
            logger.warning(
                "%d overlays dropped — exhausted %d video tracks. "
                "Increase EXTRA_OVERFLOW_TRACK_COUNT to absorb more simultaneous overlays.",
                len(remaining), len(track_uids),
            )

        # Inject onto each track in order
        for track_idx, (uid, allocated) in enumerate(track_allocations):
            track = _find_track_by_uid(root, uid)
            if track is None:
                logger.warning("Track UID %s not found — skipping %d overlays",
                               uid, len(allocated))
                continue
            track_items = _get_or_create_track_items(track)

            # On V2 only: strip the template clip reference left over from base.prproj
            if track_idx == 0:
                for ti_child in list(track_items):
                    if ti_child.get("ObjectRef") == TEMPLATE_TEXT_CLIP_OBJ_ID:
                        track_items.remove(ti_child)

            for injected, overlay in enumerate(allocated):
                cloned_clip, cloned_refs = clone_text_clip(
                    root, template_clip, overlay, video_w, video_h
                )
                for ref_elem in cloned_refs:
                    root.append(ref_elem)
                ET.SubElement(
                    track_items,
                    "TrackItem",
                    Index=str(injected),
                    ObjectRef=cloned_clip.get("ObjectID", ""),
                )
                root.append(cloned_clip)
            logger.info("Injected %d text clips on V%d", len(allocated), track_idx + 2)

        # Remove the original template clip itself from root.  Its referenced
        # objects stay in place as orphans because some are shared with other
        # template elements; removing them would create dangling refs.
        try:
            root.remove(template_clip)
        except ValueError:
            pass

    # --- Inject video on V1 ---
    # Strategy: if the template already has a real video on V1 (from PP import),
    # patch that clip's file path + duration.  This is far more reliable than
    # fabricating the full media object graph from scratch.  Fall back to
    # fabrication only when the template has no V1 clip (legacy / empty template).
    if video_duration_sec > 0 and video_path.exists():
        v1_track = _find_track_by_uid(root, V1_TRACK_UID)
        if v1_track is not None:
            patched = _patch_v1_clip(
                root, v1_track, video_path, video_duration_sec, video_w, video_h, fps,
                relative_paths=relative_paths,
            )
            if not patched:
                # Template has no V1 clip — fabricate the full chain.
                # NOTE: fabricated chains may not play correctly in PP2026 because
                # the VideoStream lacks codec metadata.  Update the template to
                # include a real video on V1 for reliable playback.
                logger.warning(
                    "Template has no V1 clip — fabricating media chain "
                    "(update template with a real video on V1 for reliable playback)"
                )
                chain_elems, vcti_id = _build_video_media_chain(
                    video_path, video_duration_sec, video_w, video_h, fps,
                    relative_paths=relative_paths,
                )
                for elem in chain_elems:
                    root.append(elem)
                v1_items = _get_or_create_track_items(v1_track)
                ET.SubElement(v1_items, "TrackItem", Index="0", ObjectRef=vcti_id)
                logger.info("Fabricated video clip on V1 (%.1f s)", video_duration_sec)
        else:
            logger.warning("V1 track not found — video not injected. Check V1_TRACK_UID.")
    else:
        logger.warning("Video not found or duration 0 — V1 injection skipped")

    # --- Mute A1 (original Korean audio) — only when TTS replaces it ---
    if tts_map:
        a1_track = _find_track_by_uid(root, A1_TRACK_UID)
        if a1_track is not None:
            _mute_track(a1_track)
            logger.info("Muted A1 track (Korean audio)")
        else:
            logger.warning(
                "A1 track not found — Korean audio not muted. Check A1_TRACK_UID."
            )
    else:
        logger.info("No TTS audio — keeping original A1 audio")

    # --- Inject TTS audio on A2 ---
    # TTS clips are placed using speech segment timing (from GVI speech transcription),
    # NOT overlay text timing.  Speech segments represent what was actually spoken;
    # overlay text is what appeared on screen (subtitles, labels).
    a2_track = _find_track_by_uid(root, A2_TRACK_UID)
    tts_end_sec = video_duration_sec  # tracks furthest TTS end; BGM will match this
    if a2_track is not None and tts_map:
        a2_items = _get_or_create_track_items(a2_track)
        injected_audio = 0

        # Use speech segments for timing if available; fall back to overlays
        if speech_segments:
            for seg in speech_segments:
                wav_path = tts_map.get(seg.get("text_ja", ""))
                if wav_path is None or not wav_path.exists():
                    continue
                # Use actual WAV duration so TTS plays to its natural end even
                # if it extends past the original video length.
                try:
                    with wave.open(str(wav_path)) as _w:
                        wav_dur = _w.getnframes() / _w.getframerate()
                except Exception:
                    wav_dur = seg["end_sec"] - seg["start_sec"]
                clip_end = seg["start_sec"] + wav_dur
                tts_end_sec = max(tts_end_sec, clip_end)
                chain_elems, acti_id = _build_tts_audio_chain(
                    wav_path, seg["start_sec"], clip_end,
                    relative_paths=relative_paths,
                )
                for elem in chain_elems:
                    root.append(elem)
                ET.SubElement(
                    a2_items,
                    "TrackItem",
                    Index=str(injected_audio),
                    ObjectRef=acti_id,
                )
                injected_audio += 1
        else:
            # Web-pipeline fallback: no speech segments available, so derive
            # TTS timing from the V2-allocated overlay set (matches pre-refactor
            # behavior — only the first non-overlapping subset gets audio).
            fallback_overlays = (
                track_allocations[0][1]
                if 'track_allocations' in locals() and track_allocations
                else _dedup_overlays(eligible, video_duration_sec)
            )
            for overlay in fallback_overlays:
                wav_path = tts_map.get(overlay.text_ja)
                if wav_path is None or not wav_path.exists():
                    continue
                tts_end_sec = max(tts_end_sec, overlay.end_sec)
                chain_elems, acti_id = _build_tts_audio_chain(
                    wav_path, overlay.start_sec, overlay.end_sec,
                    relative_paths=relative_paths,
                )
                for elem in chain_elems:
                    root.append(elem)
                ET.SubElement(
                    a2_items,
                    "TrackItem",
                    Index=str(injected_audio),
                    ObjectRef=acti_id,
                )
                injected_audio += 1
        logger.info("Injected %d TTS audio clips on A2", injected_audio)
    elif tts_map:
        logger.warning("A2 track not found — TTS audio not injected. Check A2_TRACK_UID.")

    # --- Inject BGM on A3 (background music with vocals removed) ---
    # BGM duration matches TTS end so music and voice-over finish together.
    # Capped to actual BGM WAV length to avoid requesting silence past file end.
    if bgm_path is not None and bgm_path.exists():
        a3_track = _find_track_by_uid(root, A3_TRACK_UID)
        if a3_track is not None:
            a3_items = _get_or_create_track_items(a3_track)
            try:
                with wave.open(str(bgm_path)) as _bw:
                    bgm_dur = _bw.getnframes() / _bw.getframerate()
            except Exception:
                bgm_dur = tts_end_sec
            bgm_end = min(tts_end_sec, bgm_dur)
            chain_elems, acti_id = _build_tts_audio_chain(
                bgm_path, 0.0, bgm_end,
                relative_paths=relative_paths,
            )
            for elem in chain_elems:
                root.append(elem)
            ET.SubElement(a3_items, "TrackItem", Index="0", ObjectRef=acti_id)
            logger.info("Injected BGM on A3 (%.1f s)", bgm_end)
        else:
            logger.warning("A3 track not found — BGM not injected. Check A3_TRACK_UID.")

    save_prproj(root, output_path)
    logger.info("Wrote %s", output_path)
