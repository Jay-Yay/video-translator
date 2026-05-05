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
