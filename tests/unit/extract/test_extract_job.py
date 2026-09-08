from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extract.extract_job import (
    ExtractJob,
    ExtractResult,
    create_extract_job,
    run_extract_job,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mocked CoinGecko client."""
    return MagicMock()


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mocked S3 storage."""
    return MagicMock()


@pytest.fixture
def extract_job(
    mock_client: MagicMock,
    mock_storage: MagicMock,
) -> ExtractJob:
    """Create an ExtractJob with mocked dependencies."""
    return ExtractJob(
        client=mock_client,
        storage=mock_storage,
    )


# ---------------------------------------------------------------------------
# TEST: SUCCESSFUL EXTRACTION
# ---------------------------------------------------------------------------


def test_run_success(
    extract_job: ExtractJob,
    mock_client: MagicMock,
    mock_storage: MagicMock,
) -> None:
    """Test successful extraction and S3 upload."""

    records = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "current_price": 100000.0,
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "current_price": 4000.0,
        },
    ]

    mock_client.fetch_market_data.return_value = records

    with patch(
        "src.extract.extract_job.CONFIG",
        {
            "application": {
                "raw_dataset": "crypto_market",
            },
            "s3": {
                "bucket": "test-bucket",
            },
        },
    ), patch(
        "src.extract.extract_job.PathBuilder.build_raw_path",
        return_value="raw/crypto_market/2026-09-01/data.ndjson",
    ):

        result = extract_job.run()

    # Verify result type
    assert isinstance(result, ExtractResult)

    # Verify result values
    assert result.s3_key == (
        "raw/crypto_market/2026-09-01/data.ndjson"
    )

    assert result.record_count == 2

    # Verify CoinGecko was called once
    mock_client.fetch_market_data.assert_called_once()

    # Verify S3 upload was called once
    mock_storage.upload_file.assert_called_once()

    # Verify upload arguments
    upload_call = mock_storage.upload_file.call_args

    assert upload_call.kwargs["s3_key"] == (
        "raw/crypto_market/2026-09-01/data.ndjson"
    )

    # Temporary file should have been passed
    local_path = upload_call.kwargs["local_path"]

    assert isinstance(local_path, Path)

    # TemporaryDirectory should clean the file
    assert not local_path.exists()


# ---------------------------------------------------------------------------
# TEST: EMPTY RECORDS
# ---------------------------------------------------------------------------


def test_run_with_empty_records(
    extract_job: ExtractJob,
    mock_client: MagicMock,
    mock_storage: MagicMock,
) -> None:
    """Test extraction when CoinGecko returns no records."""

    mock_client.fetch_market_data.return_value = []

    with patch(
        "src.extract.extract_job.CONFIG",
        {
            "application": {
                "raw_dataset": "crypto_market",
            },
            "s3": {
                "bucket": "test-bucket",
            },
        },
    ):

        with pytest.raises(
            ValueError,
            match="CoinGecko returned zero records",
        ):
            extract_job.run()

    # CoinGecko should still be called
    mock_client.fetch_market_data.assert_called_once()

    # S3 upload must NOT happen
    mock_storage.upload_file.assert_not_called()


# ---------------------------------------------------------------------------
# TEST: CLIENT FAILURE
# ---------------------------------------------------------------------------


def test_run_when_client_fails(
    extract_job: ExtractJob,
    mock_client: MagicMock,
    mock_storage: MagicMock,
) -> None:
    """Test that client exceptions are propagated."""

    mock_client.fetch_market_data.side_effect = RuntimeError(
        "CoinGecko API failed"
    )

    with pytest.raises(
        RuntimeError,
        match="CoinGecko API failed",
    ):
        extract_job.run()

    # Client should be called once
    mock_client.fetch_market_data.assert_called_once()

    # S3 should not be called
    mock_storage.upload_file.assert_not_called()


# ---------------------------------------------------------------------------
# TEST: S3 UPLOAD FAILURE
# ---------------------------------------------------------------------------


def test_run_when_s3_upload_fails(
    extract_job: ExtractJob,
    mock_client: MagicMock,
    mock_storage: MagicMock,
) -> None:
    """Test that S3 upload exceptions are propagated."""

    records = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "current_price": 100000.0,
        }
    ]

    mock_client.fetch_market_data.return_value = records

    mock_storage.upload_file.side_effect = RuntimeError(
        "S3 upload failed"
    )

    with patch(
        "src.extract.extract_job.CONFIG",
        {
            "application": {
                "raw_dataset": "crypto_market",
            },
            "s3": {
                "bucket": "test-bucket",
            },
        },
    ), patch(
        "src.extract.extract_job.PathBuilder.build_raw_path",
        return_value="raw/crypto_market/2026-09-01/data.ndjson",
    ):

        with pytest.raises(
            RuntimeError,
            match="S3 upload failed",
        ):
            extract_job.run()

    # Client should be called once
    mock_client.fetch_market_data.assert_called_once()

    # Upload should be attempted once
    mock_storage.upload_file.assert_called_once()


# ---------------------------------------------------------------------------
# TEST: NDJSON WRITING
# ---------------------------------------------------------------------------


def test_write_ndjson(tmp_path: Path) -> None:
    """Test writing records to an NDJSON file."""

    records = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "price": 100000.0,
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "price": 4000.0,
        },
    ]

    local_path = tmp_path / "market.ndjson"

    ExtractJob._write_ndjson(
        records=records,
        local_path=local_path,
    )

    # File should exist
    assert local_path.exists()

    # Read file
    lines = local_path.read_text(
        encoding="utf-8"
    ).splitlines()

    # Two records = two lines
    assert len(lines) == 2

    # Validate JSON content
    assert json.loads(lines[0]) == records[0]
    assert json.loads(lines[1]) == records[1]


# ---------------------------------------------------------------------------
# TEST: NDJSON UNICODE
# ---------------------------------------------------------------------------


def test_write_ndjson_with_unicode(
    tmp_path: Path,
) -> None:
    """Test writing NDJSON with Unicode characters."""

    records = [
        {
            "id": "bitcoin",
            "name": "Bitcoin ₹",
        }
    ]

    local_path = tmp_path / "unicode.ndjson"

    ExtractJob._write_ndjson(
        records=records,
        local_path=local_path,
    )

    # Read file
    content = local_path.read_text(
        encoding="utf-8"
    )

    # Unicode should remain unchanged
    assert "Bitcoin ₹" in content

    # Validate JSON
    assert json.loads(content) == records[0]


# ---------------------------------------------------------------------------
# TEST: CREATE EXTRACT JOB
# ---------------------------------------------------------------------------


def test_create_extract_job() -> None:
    """Test creation of a configured ExtractJob."""

    with patch(
        "src.extract.extract_job.CONFIG",
        {
            "aws": {
                "region": "ap-south-1",
            },
            "s3": {
                "bucket": "test-bucket",
            },
        },
    ), patch(
        "src.extract.extract_job.CoinGeckoClient"
    ) as mock_client_class, patch(
        "src.extract.extract_job.S3Storage"
    ) as mock_storage_class:

        job = create_extract_job()

    # Verify object type
    assert isinstance(job, ExtractJob)

    # Verify CoinGecko client creation
    mock_client_class.assert_called_once()

    # Verify S3 storage creation
    mock_storage_class.assert_called_once_with(
        bucket_name="test-bucket",
        region_name="ap-south-1",
    )

    # Verify dependencies were injected
    assert job.client == mock_client_class.return_value
    assert job.storage == mock_storage_class.return_value


# ---------------------------------------------------------------------------
# TEST: RUN EXTRACT JOB
# ---------------------------------------------------------------------------


def test_run_extract_job() -> None:
    """Test the run_extract_job convenience function."""

    mock_job = MagicMock()

    expected_result = ExtractResult(
        s3_key="raw/crypto_market/2026-09-01/data.ndjson",
        record_count=100,
    )

    mock_job.run.return_value = expected_result

    with patch(
        "src.extract.extract_job.create_extract_job",
        return_value=mock_job,
    ) as mock_create:

        result = run_extract_job()

    # Verify returned result
    assert isinstance(result, ExtractResult)

    assert result.s3_key == (
        "raw/crypto_market/2026-09-01/data.ndjson"
    )

    assert result.record_count == 100

    # Verify job creation
    mock_create.assert_called_once()

    # Verify job execution
    mock_job.run.assert_called_once()