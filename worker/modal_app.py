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

_IGNORE_PATTERNS = [
    "__pycache__", ".git", "videos-", "checkpoints",
    "premiere_projects", ".env", "node_modules", "web/",
]

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
    # Mount the full project source into /root/app so stages/, models/, etc. are available
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=lambda p: any(seg in str(p) for seg in _IGNORE_PATTERNS),
    )
)

_secrets = modal.Secret.from_name("video-translator-secrets")



# ── VideoTranslator class ─────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="A10G",
    timeout=1800,
    memory=8192,
    secrets=[_secrets],
)
class VideoTranslator:

    @modal.enter()
    def download_models(self):
        """Download and cache model weights on container startup."""
        sys.path.insert(0, "/root/app")
        from paddleocr import PaddleOCR
        PaddleOCR(use_angle_cls=True, lang="korean")
        import demucs.pretrained
        demucs.pretrained.get_model("htdemucs")

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
@modal.fastapi_endpoint(method="POST")
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
