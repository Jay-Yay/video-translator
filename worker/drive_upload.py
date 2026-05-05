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
