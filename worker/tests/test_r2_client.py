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
