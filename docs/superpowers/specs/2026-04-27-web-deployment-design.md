# Web Deployment Design — Korean→Japanese Video Translator

**Date:** 2026-04-27  
**Status:** Approved

---

## Goal

Deploy the existing local CLI pipeline as a web app so a teammate can:
1. Upload Korean beauty marketing videos in bulk via a browser
2. Have each video translated asynchronously (text overlays + Japanese TTS voice-over + BGM)
3. Receive a Premiere Pro `.prproj` per video in a shared Google Drive folder, ready to open

---

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Frontend | Vercel (Next.js) | Upload UI, password gate, presigned URL generation |
| File storage | Cloudflare R2 | Receives raw video uploads; holds intermediate files |
| ML worker | Modal.com | Runs the full pipeline per video, in parallel, with GPU |
| Job tracking | Upstash Redis | Tracks batch progress (queued / completed / failed per video) |
| Result delivery | Google Drive API | Modal uploads finished project folder to shared Drive folder |
| Notification | SendGrid | "Your batch is ready" email when all videos in a batch are done |

---

## Architecture Overview

```
Browser (teammate)
  │  1. password gate (env-var secret, checked in Next.js middleware)
  │  2. drag-and-drop video files
  │  3. browser fetches presigned R2 PUT URLs from /api/upload-urls
  │  4. browser PUTs each video directly to R2 (bypasses Vercel 4.5 MB limit)
  │  5. browser POSTs batch manifest {batch_id, r2_keys[]} to /api/submit-batch
  │  6. UI shows "Submitted — you'll get an email when done"
  ▼
Vercel API routes
  │  /api/upload-urls  → generates R2 presigned PUT URLs (one per file)
  │  /api/submit-batch → creates batch record in Redis, calls Modal batch endpoint
  ▼
Cloudflare R2  ←── raw .mp4 files land here (key: batches/{batch_id}/{stem}.mp4)
  ▼
Modal.com  (one container per video, all run concurrently via Modal .map())
  │  per-video container:
  │    1. Download .mp4 from R2 to /tmp/
  │    2. Run detect stage  (PaddleOCR + scene detection, CPU)
  │    3. Run deduplicate stage (CPU)
  │    4. Run translate stage (Claude API)
  │    5. Run TTS stage (ElevenLabs API)
  │    6. Run Demucs BGM extraction (GPU — A10G)
  │    7. Run export_prproj stage (builds .prproj with relative paths)
  │    8. Upload project folder to Google Drive
  │    9. Update Redis: batch_id completed_count++
  │   10. If completed_count == total: send SendGrid email
  ▼
Google Drive (shared folder)
  KR→JP Translations/
    {batch_id}/
      {stem}/
        {stem}.mp4          ← original Korean video
        {stem}.prproj       ← Premiere Pro 2026 project (relative paths)
        tts/
          *.wav             ← Japanese TTS audio clips
```

---

## Section 1: Upload Flow

### Password Gate
- Next.js middleware (`middleware.ts`) checks a cookie set by a `/login` page
- `/login` accepts a single shared password stored in `UPLOAD_PASSWORD` env var on Vercel
- No database, no user accounts — one password for the whole app
- Cookie is `HttpOnly`, `Secure`, expires in 30 days

### File Upload
- `/api/upload-urls` (POST): accepts `{filenames: string[]}`, returns presigned R2 PUT URLs (15-minute expiry)
- **Small files (<100 MB)**: browser uploads directly with `fetch(presignedUrl, {method: 'PUT', body: file})`
- **Large files (≥100 MB)**: browser uses S3 multipart upload via R2's S3-compatible API — `/api/upload-urls` returns a multipart upload ID + per-part presigned URLs; browser uploads 50 MB parts in parallel; `/api/complete-multipart` finalises the upload. 1080p beauty videos at 5 min can reach 500 MB–2 GB, so multipart is the default path in practice
- Progress bar per file shown in UI (using `XMLHttpRequest` upload progress events)
- `/api/submit-batch` (POST): accepts `{batch_id, r2_keys[], notify_email}`, writes batch record to Redis, triggers Modal

### Batch Record in Redis
```
batch:{batch_id}:total     → N (integer)
batch:{batch_id}:completed → 0 (incremented by each worker on success)
batch:{batch_id}:failed    → 0 (incremented on failure)
batch:{batch_id}:email     → teammate@company.com
batch:{batch_id}:videos    → JSON array of {r2_key, stem}
```
TTL: 7 days.

---

## Section 2: Modal Worker

### Image
A custom Modal image based on `python:3.11-slim` with:
- `paddlepaddle`, `paddleocr`
- `anthropic`, `elevenlabs`
- `demucs`, `torch` (CUDA)
- `ffmpeg` (system package)
- `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`
- `boto3` (R2 access via S3-compatible API)
- `upstash-redis`, `sendgrid`
- The existing `stages/`, `models/`, `utils/`, `templates/` from this repo (mounted via Modal)

### Model Warm-up (Cold Start)
PaddleOCR (~500MB) and Demucs (~80MB) both download model weights on first use. Without pre-loading, the first container in a batch would stall 3–5 minutes before processing starts.

Use Modal's `@app.cls` + `@modal.build()` to bake model weights into the image at build time:
```python
@app.cls(gpu="A10G", ...)
class VideoTranslator:
    @modal.build()
    def download_models(self):
        from paddleocr import PaddleOCR
        PaddleOCR(use_angle_cls=True, lang="korean")  # triggers weight download
        import demucs.pretrained
        demucs.pretrained.get_model("htdemucs")       # triggers weight download

    @modal.enter()
    def load_models(self):
        # Initialize models into self.* for reuse across calls in same container
        ...

    @modal.method()
    def process_video(self, batch_id: str, r2_key: str, stem: str) -> None:
        ...
```
This runs once at image build; subsequent container starts load from the baked layer (~10s, not 3–5min).

### Function Signature
```python
@app.function(
    gpu="A10G",
    timeout=1800,          # 30 min per video max
    memory=8192,
    retries=1,
)
def process_video(batch_id: str, r2_key: str, stem: str) -> None:
    ...
```

### Processing Steps (inside the function)
1. Download `r2_key` from R2 → `/tmp/{stem}.mp4`
2. Create `/tmp/checkpoints/{stem}/` for stage checkpoints
3. Run `detect.run()`, `deduplicate.run()`, `translate.run()` — **Stage 3 (moviepy render/burn) is intentionally skipped**: the `.prproj` handles text overlays natively as Essential Graphics clips; no rendered video is produced
4. Run `tts.run()` (ElevenLabs TTS audio generation)
5. Run `demucs` for BGM extraction (GPU)
6. Run `export_prproj.run()` with `relative_paths=True`
   - Output: `/tmp/output/{stem}/` containing `.prproj`, `.mp4`, `tts/*.wav`
7. Upload `/tmp/output/{stem}/` to Google Drive at `KR→JP Translations/{batch_id}/{stem}/`
8. Delete source `r2_key` from R2 (cleanup — prevents R2 storage from accumulating at ~6 GB/day)
9. Atomically increment `batch:{batch_id}:completed` in Redis; if the returned value equals `total`, call `send_completion_email()`

### Batch Dispatch from Vercel
Vercel's `/api/submit-batch` (Node.js) cannot call Modal Python functions directly. Instead:
- A Modal `@app.web_endpoint()` is deployed (e.g. `https://your-org--video-translator-dispatch.modal.run`)
- Vercel POSTs the batch manifest `{batch_id, videos[]}` to this endpoint via `fetch()`
- The Modal web endpoint function calls `process_video.spawn()` for each video, launching all concurrently
- The endpoint returns immediately (200 OK); processing continues asynchronously in Modal containers

The `MODAL_DISPATCH_URL` and a shared `MODAL_DISPATCH_SECRET` header token are stored as Vercel env vars.

### Error Handling
- Transient errors (network blips, API rate limits) are retried with exponential back-off inside the function before being considered failures — do NOT increment `failed` on caught transient errors
- Unrecoverable failures (bad video file, pipeline exception after retries) increment `batch:{batch_id}:failed` atomically; the same atomic INCR + compare pattern triggers the completion email: `if INCR(completed) + GET(failed) == total` OR `if GET(completed) + INCR(failed) == total`
- After all complete (success + failure), email reports: "12 succeeded, 2 failed"
- Failed videos leave their R2 source file in place for manual retry; a `batch:{batch_id}:errors` list stores `{stem, error_message}` for each failure

### Modal Dispatch Auth
The Modal `@app.web_endpoint()` validates every inbound request:
```python
if request.headers.get("X-Dispatch-Secret") != os.environ["MODAL_DISPATCH_SECRET"]:
    return Response(status_code=403)
```
`MODAL_DISPATCH_SECRET` is stored both as a Modal Secret (for the endpoint) and as a Vercel env var (for the caller).

---

## Section 3: Relative Paths in `.prproj`

**Problem:** `prproj_builder.py` currently writes `str(video_path.resolve())` — an absolute server path — into the `.prproj` XML. Premiere Pro will show "Media Offline" when she opens it on her Mac.

**Solution:** Add `relative_paths: bool = False` parameter to `build_prproj()` and `export_prproj.run()`. When `True`:
- The video path written to `FilePath` / `ActualMediaFilePath` is just `{stem}.mp4` (relative to the `.prproj` file)
- TTS WAV paths are written as `tts/{filename}.wav`
- BGM path is written as `tts/bgm.wav`

Premiere Pro resolves relative paths from the `.prproj` file's containing folder, so as long as the folder structure is:
```
{stem}/
  {stem}.mp4
  {stem}.prproj
  tts/
    *.wav
```
…all media links correctly regardless of Drive mount point.

---

## Section 4: Google Drive Upload

- Service account credentials stored as a Modal Secret (`GOOGLE_SERVICE_ACCOUNT_JSON`)
- Shared Drive folder ID stored as `GOOGLE_DRIVE_FOLDER_ID` env var
- The service account is added as an Editor on the shared Drive folder once at setup
- Upload uses `google-api-python-client` with `MediaFileUpload`
- Folder structure created programmatically: `KR→JP Translations/{batch_id}/{stem}/`

---

## Section 5: Notification

- SendGrid API key stored as a Modal Secret
- Email sent from a verified sender (e.g. `translator@yourdomain.com`)
- Email body:
  ```
  Subject: [Video Translator] Your batch is ready
  
  {N} videos processed. Results are in Google Drive:
  KR→JP Translations/{batch_id}/
  
  ✓ Succeeded: 12
  ✗ Failed: 0
  ```

---

## Section 6: Frontend UI

Simple, functional — not a full design system. Single page:

1. **Password screen**: centered form, password input, submit button
2. **Upload screen**:
   - Drag-and-drop zone (accepts `.mp4` only)
   - File list with per-file size and upload progress bar
   - Email input (pre-filled from cookie if previously entered)
   - "Translate All" button (disabled until all uploads complete)
   - After submit: "Submitted! You'll receive an email when your X videos are ready."

No job status page — notification-only per requirements.

---

## Section 7: Required Code Changes to Existing Pipeline

| File | Change |
|---|---|
| `stages/prproj_builder.py` | Add `relative_paths: bool = False` param to `build_prproj()`; when True, strip absolute prefix from FilePath/ActualMediaFilePath |
| `stages/export_prproj.py` | Pass `relative_paths` through to `build_prproj()` |
| `stages/tts.py` | Ensure `extract_bgm()` works with a `/tmp/` working directory (no config.ROOT_DIR dependency) |
| `config.py` | Guard `ANTHROPIC_API_KEY` and `ELEVENLABS_API_KEY` to not raise on import if set via Modal Secrets (already uses `os.environ` so this may work as-is) |

New files:
| File | Purpose |
|---|---|
| `web/` | Next.js app (upload UI, API routes, middleware) |
| `worker/modal_app.py` | Modal app definition with `process_video` function |
| `worker/drive_upload.py` | Google Drive upload helper |
| `worker/r2_client.py` | R2 download helper (boto3 S3-compatible) |
| `worker/notify.py` | SendGrid email helper |

---

## Infrastructure Setup (one-time)

1. **Cloudflare R2**: create bucket `video-translator`, generate API token with R2 read/write, configure CORS to allow `PUT` from the Vercel app domain (required for browser direct uploads)
2. **Modal**: create account, set secrets (`ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`, `GOOGLE_SERVICE_ACCOUNT_JSON`, `SENDGRID_API_KEY`, `R2_*`)
3. **Upstash**: create Redis database, copy `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN`
4. **Google Cloud**: create service account, enable Drive API, share target folder with service account email
5. **SendGrid**: verify sender domain, generate API key
6. **Vercel**: add env vars (`UPLOAD_PASSWORD`, `R2_*`, `UPSTASH_*`, `MODAL_TOKEN_*`), connect GitHub repo

---

## Estimated Cost (30 videos/day, 3 min avg)

| Service | Estimate |
|---|---|
| Modal (A10G GPU, ~10 min/video) | ~$1.50/day |
| Cloudflare R2 | ~$0 (free tier covers this volume) |
| Upstash Redis | $0 (free tier) |
| SendGrid | $0 (free tier: 100 emails/day) |
| Vercel | $0 (free tier) |
| **Total** | **~$45/month** |
