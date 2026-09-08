from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.logger import Logger

logger = Logger.get_logger("s3_storage", "s3_storage.log")

class S3Storage:
    """Generic s3 storage service."""
    
    def __init__(
        self,
        bucket_name: str,
        region_name: str
    ) -> None:
        self.bucket_name = bucket_name
        self.client = boto3.client(
            "s3",
            region_name=region_name
        )

    def upload_file(
        self,
        local_path: Path,
        s3_key: str,
    ) -> None:
        """Upload a local file to S3."""
        
        logger.info(
            "Uploading file to s3://%s/%s",
            self.bucket_name,
            s3_key
        )

        try:
            self.client.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket_name,
                Key=s3_key
            )

        except (BotoCoreError, ClientError) as exc:
            logger.exception(
                "S3 upload failed: s3://%s/%s",
                self.bucket_name,
                s3_key
            )

            raise RuntimeError(
                f"S3 upload failed: "
                f"s3://{self.bucket_name}/{s3_key}"
            ) from exc

        logger.info(
            "File uploaded successfully: "
            "s3://%s/%s",
            self.bucket_name,
            s3_key
        )

    def download_file(
        self,
        s3_key: str,
        local_path: Path
    ) -> None:
        """Download an S3 object to a local file."""

        logger.info(
            "Downloading file from s3://%s/%s",
            self.bucket_name,
            s3_key
        )

        try:
            local_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            self.client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=str(local_path)
            )

        except (BotoCoreError, ClientError) as exc:
            logger.exception(
                "S3 download failed: s3://%s/%s",
                self.bucket_name,
                s3_key
            )

            raise RuntimeError(
                f"S3 download failed: "
                f"s3://{self.bucket_name}/{s3_key}"
            ) from exc

        logger.info(
            "File downloaded successfully: %s",
            local_path
        )

    def object_exists(
        self,
        s3_key: str
    ) -> bool:
        """Check whether an S3 object exists."""

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            logger.debug(
                "S3 object exists: s3://%s/%s",
                self.bucket_name,
                s3_key
            )

            return True

        except ClientError as exc:
            error_code = exc.response.get(
                "Error",
                {}
            ).get("Code")

            if error_code in {"404", "NoSuchKey"}:
                logger.debug(
                    "S3 object does not exist: "
                    "s3://%s/%s",
                    self.bucket_name,
                    s3_key
                )

                return False

            logger.exception(
                "S3 object check failed: "
                "s3://%s/%s",
                self.bucket_name,
                s3_key
            )

            raise RuntimeError(
                f"S3 object check failed: "
                f"s3://{self.bucket_name}/{s3_key}"
            ) from exc

        except BotoCoreError as exc:
            logger.exception(
                "S3 object check failed: "
                "s3://%s/%s",
                self.bucket_name,
                s3_key
            )

            raise RuntimeError(
                f"S3 object check failed: "
                f"s3://{self.bucket_name}/{s3_key}"
            ) from exc