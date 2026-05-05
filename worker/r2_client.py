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
