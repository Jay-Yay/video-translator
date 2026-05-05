# stages/detect_gvi.py
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path

import cv2
from google.cloud import videointelligence, storage

import config
from models.text_overlay import TextOverlay
from stages import gvi_postprocess

logger = logging.getLogger(__name__)

# GVI inline input limit is 20 MB; larger files must be uploaded to GCS first.
_GVI_INLINE_LIMIT_BYTES = 20 * 1024 * 1024


def _upload_to_gcs(video_path: Path) -> tuple[str, str, str]:
    """Upload video to a temporary GCS object. Returns (bucket_name, blob_name, gs_uri)."""
    gcs = storage.Client()
    project = gcs.project or "video-translator"
    bucket_name = f"{project}-gvi-tmp"

    # Create bucket if it doesn't exist
    try:
        bucket = gcs.get_bucket(bucket_name)
    except Exception:
        bucket = gcs.create_bucket(bucket_name, location="US")
        logger.info("Created GCS bucket: %s", bucket_name)

    blob_name = f"{uuid.uuid4().hex}/{video_path.name}"
    blob = bucket.blob(blob_name)
    logger.info("Uploading %s to gs://%s/%s …", video_path.name, bucket_name, blob_name)
    blob.upload_from_filename(str(video_path))
    return bucket_name, blob_name, f"gs://{bucket_name}/{blob_name}"


def _delete_gcs_object(bucket_name: str, blob_name: str) -> None:
    try:
        gcs = storage.Client()
        gcs.bucket(bucket_name).blob(blob_name).delete()
        logger.info("Deleted temporary GCS object gs://%s/%s", bucket_name, blob_name)
    except Exception as exc:
        logger.warning("Could not delete GCS object: %s", exc)


def contains_hangul(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (
            0xAC00 <= cp <= 0xD7A3
            or 0x1100 <= cp <= 0x11FF
            or 0x3130 <= cp <= 0x318F
        ):
            return True
    return False


def _video_dims(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _split_words_into_utterances(
    words, confidence: float, gap_threshold: float = 0.3,
) -> list[dict]:
    """Group consecutive words into utterances split at silence gaps.

    Whenever the gap between the end of one word and the start of the next
    exceeds gap_threshold (seconds), a new utterance begins.  Each returned
    dict has the same shape as a GVI speech segment: {text_ko, text_ja,
    start_sec, end_sec, confidence}.
    """
    if not words:
        return []

    utterances: list[dict] = []
    cur_words: list = [words[0]]
    cur_start = words[0].start_time.total_seconds()
    prev_end = words[0].end_time.total_seconds()

    for w in words[1:]:
        w_start = w.start_time.total_seconds()
        w_end = w.end_time.total_seconds()
        if w_start - prev_end > gap_threshold:
            text = " ".join(getattr(x, "word", "") for x in cur_words).strip()
            if text:
                utterances.append({
                    "text_ko": text,
                    "text_ja": "",
                    "start_sec": cur_start,
                    "end_sec": prev_end,
                    "confidence": confidence,
                })
            cur_words = [w]
            cur_start = w_start
        else:
            cur_words.append(w)
        prev_end = w_end

    text = " ".join(getattr(x, "word", "") for x in cur_words).strip()
    if text:
        utterances.append({
            "text_ko": text,
            "text_ja": "",
            "start_sec": cur_start,
            "end_sec": prev_end,
            "confidence": confidence,
        })
    return utterances


def _normalized_bbox(
    vertices, width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert GVI normalized vertices (0–1) to pixel (x, y, w, h)."""
    xs = [int(v.x * width) for v in vertices]
    ys = [int(v.y * height) for v in vertices]
    x, y = min(xs), min(ys)
    return x, y, max(xs) - x, max(ys) - y


def run(video_path: Path, checkpoint_dir: Path) -> tuple[list[TextOverlay], bool]:
    """Stage 1 (GVI): detect Korean text via Google Video Intelligence API.

    GVI returns pre-deduplicated text annotations with start/end timestamps
    and bounding boxes — no separate deduplication pass needed.

    Also performs speech transcription to detect whether the video contains
    Korean speech audio.  Videos without Korean speech should keep their
    original audio track instead of being replaced by TTS.

    Writes detections.json directly (skips raw_detections.json).

    Returns (overlays, has_korean_speech).
    """
    file_size = video_path.stat().st_size
    client = videointelligence.VideoIntelligenceServiceClient()

    speech_config = videointelligence.SpeechTranscriptionConfig(
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    video_context = videointelligence.VideoContext(
        speech_transcription_config=speech_config,
    )

    gcs_bucket = gcs_blob = None
    if file_size > _GVI_INLINE_LIMIT_BYTES:
        logger.info(
            "%s is %.1f MB — uploading to GCS for detection…",
            video_path.name, file_size / 1_048_576,
        )
        gcs_bucket, gcs_blob, gs_uri = _upload_to_gcs(video_path)
        logger.info("Sending %s to Google Video Intelligence API via GCS…", video_path.name)
        # Run text detection and speech transcription as separate calls to avoid
        # GVI interference between features (combined calls can drop text results).
        text_op = client.annotate_video(
            request={
                "features": [videointelligence.Feature.TEXT_DETECTION],
                "input_uri": gs_uri,
            }
        )
        speech_op = client.annotate_video(
            request={
                "features": [videointelligence.Feature.SPEECH_TRANSCRIPTION],
                "input_uri": gs_uri,
                "video_context": video_context,
            }
        )
    else:
        input_content = video_path.read_bytes()
        logger.info("Sending %s to Google Video Intelligence API…", video_path.name)
        text_op = client.annotate_video(
            request={
                "features": [videointelligence.Feature.TEXT_DETECTION],
                "input_content": input_content,
            }
        )
        speech_op = client.annotate_video(
            request={
                "features": [videointelligence.Feature.SPEECH_TRANSCRIPTION],
                "input_content": input_content,
                "video_context": video_context,
            }
        )

    logger.info("Waiting for GVI results (this usually takes 30–60 s)…")
    text_result = text_op.result(timeout=600)
    speech_result = speech_op.result(timeout=600)

    if gcs_bucket and gcs_blob:
        _delete_gcs_object(gcs_bucket, gcs_blob)

    width, height = _video_dims(video_path)
    overlays: list[TextOverlay] = []

    text_annotations = text_result.annotation_results[0]
    for annotation in text_annotations.text_annotations:
        text = annotation.text
        if not contains_hangul(text):
            continue

        for segment in annotation.segments:
            confidence = float(segment.confidence)
            if confidence < config.OCR_CONFIDENCE_THRESHOLD:
                continue

            start_sec = segment.segment.start_time_offset.total_seconds()
            end_sec = segment.segment.end_time_offset.total_seconds()

            # Use the bounding box from the first detected frame in the segment
            if not segment.frames:
                continue
            vertices = segment.frames[0].rotated_bounding_box.vertices
            bbox = _normalized_bbox(vertices, width, height)

            overlays.append(TextOverlay(
                text_ko=text,
                text_ja="",
                bbox=bbox,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=confidence,
            ))

    overlays.sort(key=lambda o: o.start_sec)
    overlays = gvi_postprocess.run(overlays, width, height)

    # ── Extract Korean speech transcription segments ─────────────────────
    # GVI returns one transcript per speech_transcription annotation, but
    # often returns the full video as a single annotation.  Split each
    # transcript at silence gaps (>= SPEECH_GAP_THRESHOLD between consecutive
    # words) so each utterance becomes its own segment with accurate start/end
    # times — TTS clips can then be placed per-utterance rather than as one
    # giant clip stretched over the whole video.
    speech_segments: list[dict] = []
    speech_annotations = speech_result.annotation_results[0]
    for speech in getattr(speech_annotations, "speech_transcriptions", []):
        if not speech.alternatives:
            continue
        best = speech.alternatives[0]
        if best.confidence < 0.5:
            continue
        words = list(best.words)
        if not words:
            continue
        speech_segments.extend(
            _split_words_into_utterances(words, float(best.confidence))
        )

    has_korean_speech = len(speech_segments) > 0
    logger.info(
        "GVI Stage 1: Korean speech detected = %s (%d segments)",
        has_korean_speech, len(speech_segments),
    )

    out_path = checkpoint_dir / "detections.json"
    out_path.write_text(
        json.dumps([o.to_dict() for o in overlays], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Save speech flag and transcription segments
    speech_flag_path = checkpoint_dir / "has_korean_speech.json"
    speech_flag_path.write_text(json.dumps(has_korean_speech), encoding="utf-8")
    speech_path = checkpoint_dir / "speech_segments.json"
    speech_path.write_text(
        json.dumps(speech_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "GVI Stage 1: detected %d Korean text segments → %s", len(overlays), out_path
    )
    return overlays, has_korean_speech
