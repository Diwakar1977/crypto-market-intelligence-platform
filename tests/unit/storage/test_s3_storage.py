from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from src.storage.s3_storage import S3Storage


@pytest.fixture
def s3_storage() -> Generator[S3Storage, None, None]:
    """Create S3Storage with a mocked boto3 client."""

    mock_s3_client = MagicMock()

    with patch(
        "src.storage.s3_storage.boto3.client",
        return_value=mock_s3_client,
    ):
        storage = S3Storage(
            bucket_name="test-bucket",
            region_name="ap-south-1",
        )

        storage.client = cast(Any, mock_s3_client)

        yield storage


# ============================================================
# UPLOAD FILE
# ============================================================


def test_upload_file_success(
    s3_storage: S3Storage,
    tmp_path: Path,
) -> None:
    """Test successful S3 file upload."""

    local_path = tmp_path / "data.json"

    local_path.write_text(
        '{"bitcoin": 100}',
        encoding="utf-8",
    )

    s3_storage.upload_file(
        local_path=local_path,
        s3_key="raw/crypto_market/data.json",
    )

    client = cast(MagicMock, s3_storage.client)

    client.upload_file.assert_called_once_with(
        Filename=str(local_path),
        Bucket="test-bucket",
        Key="raw/crypto_market/data.json",
    )


def test_upload_file_failure(
    s3_storage: S3Storage,
    tmp_path: Path,
) -> None:
    """Test S3 upload failure."""

    local_file = tmp_path / "data.json"

    local_file.write_text(
        '{"bitcoin": 100}',
        encoding="utf-8",
    )

    client = cast(MagicMock, s3_storage.client)

    client.upload_file.side_effect = ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Access denied",
            }
        },
        "UploadFile",
    )

    with pytest.raises(
        RuntimeError,
        match="S3 upload failed",
    ):
        s3_storage.upload_file(
            local_path=local_file,
            s3_key="raw/crypto_market/data.json",
        )


# ============================================================
# DOWNLOAD FILE
# ============================================================


def test_download_file_success(
    s3_storage: S3Storage,
    tmp_path: Path,
) -> None:
    """Test successful S3 file download."""

    local_file = tmp_path / "output" / "data.json"

    s3_storage.download_file(
        s3_key="raw/crypto_market/data.json",
        local_path=local_file,
    )

    client = cast(MagicMock, s3_storage.client)

    client.download_file.assert_called_once_with(
        Bucket="test-bucket",
        Key="raw/crypto_market/data.json",
        Filename=str(local_file),
    )

    assert local_file.parent.exists()


def test_download_file_failure(
    s3_storage: S3Storage,
    tmp_path: Path,
) -> None:
    """Test S3 download failure."""

    local_file = tmp_path / "data.json"

    client = cast(MagicMock, s3_storage.client)

    client.download_file.side_effect = ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Access denied",
            }
        },
        "DownloadFile",
    )

    with pytest.raises(
        RuntimeError,
        match="S3 download failed",
    ):
        s3_storage.download_file(
            s3_key="raw/crypto_market/data.json",
            local_path=local_file,
        )


# ============================================================
# OBJECT EXISTS
# ============================================================


def test_object_exists_returns_true(
    s3_storage: S3Storage,
) -> None:
    """Test existing S3 object."""

    result = s3_storage.object_exists(
        "raw/crypto_market/data.json",
    )

    assert result is True

    client = cast(MagicMock, s3_storage.client)

    client.head_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="raw/crypto_market/data.json",
    )


def test_object_exists_returns_false_404(
    s3_storage: S3Storage,
) -> None:
    """Test missing S3 object."""

    client = cast(MagicMock, s3_storage.client)

    client.head_object.side_effect = ClientError(
        {
            "Error": {
                "Code": "404",
                "Message": "Not Found",
            }
        },
        "HeadObject",
    )

    result = s3_storage.object_exists(
        "raw/crypto_market/data.json",
    )

    assert result is False


def test_object_exists_for_aws_error(
    s3_storage: S3Storage,
) -> None:
    """Test unexpected S3 object check failure."""

    client = cast(MagicMock, s3_storage.client)

    client.head_object.side_effect = ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Access denied",
            }
        },
        "HeadObject",
    )

    with pytest.raises(
        RuntimeError,
        match="S3 object check failed",
    ):
        s3_storage.object_exists(
            "raw/crypto_market/data.json",
        )


def test_object_exists_for_boto_error(
    s3_storage: S3Storage,
) -> None:
    """Test BotoCoreError during object check."""

    client = cast(MagicMock, s3_storage.client)

    client.head_object.side_effect = BotoCoreError()

    with pytest.raises(
        RuntimeError,
        match="S3 object check failed",
    ):
        s3_storage.object_exists(
            "raw/crypto_market/data.json",
        )
