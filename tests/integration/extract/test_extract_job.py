from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extract.extract_job import ExtractJob, ExtractResult

# =====================================================================
# TEST DATA
# =====================================================================


@pytest.fixture
def sample_records() -> list[dict[str, object]]:
    """Return realistic CoinGecko-style market records."""

    return [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 105000.50,
            "market_cap": 2100000000000,
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "name": "Ethereum",
            "current_price": 4200.75,
            "market_cap": 500000000000,
        },
    ]


# =====================================================================
# END-TO-END EXTRACT JOB
# =====================================================================


def test_extract_job_end_to_end_with_local_storage(
    sample_records: list[dict[str, object]],
) -> None:
    """
    Test the complete ExtractJob flow.

    Real S3 is NOT used.
    """

    client = MagicMock()
    storage = MagicMock()

    client.fetch_market_data.return_value = sample_records

    captured_content: dict[str, str] = {}

    def fake_upload_file(
        local_path: Path,
        s3_key: str,
    ) -> None:
        """Capture file content before TemporaryDirectory deletes it."""

        assert local_path.exists()

        captured_content["s3_key"] = s3_key

        captured_content["content"] = local_path.read_text(
            encoding="utf-8",
        )

    storage.upload_file.side_effect = fake_upload_file

    job = ExtractJob(
        client=client,
        storage=storage,
    )

    expected_s3_key = "raw_data/crypto_market/" "crypto_market.ndjson"

    with patch(
        "src.extract.extract_job.PathBuilder.build_raw_path",
        return_value=expected_s3_key,
    ):
        result = job.run()

    # ================================================================
    # VERIFY RESULT
    # ================================================================

    assert isinstance(result, ExtractResult)

    assert result.s3_key == expected_s3_key

    assert result.record_count == 2

    # ================================================================
    # VERIFY CLIENT
    # ================================================================

    client.fetch_market_data.assert_called_once_with()

    # ================================================================
    # VERIFY STORAGE
    # ================================================================

    storage.upload_file.assert_called_once()

    assert captured_content["s3_key"] == expected_s3_key

    # ================================================================
    # VERIFY NDJSON CONTENT
    # ================================================================

    content = captured_content["content"]

    lines = content.splitlines()

    assert len(lines) == 2

    first_record = json.loads(lines[0])

    second_record = json.loads(lines[1])

    assert first_record == sample_records[0]

    assert second_record == sample_records[1]


# =====================================================================
# ZERO RECORDS
# =====================================================================


def test_extract_job_zero_records() -> None:
    """Fail when the source returns zero records."""

    client = MagicMock()

    storage = MagicMock()

    client.fetch_market_data.return_value = []

    job = ExtractJob(
        client=client,
        storage=storage,
    )

    with pytest.raises(
        ValueError,
        match="CoinGecko returned zero records.",
    ):
        job.run()

    client.fetch_market_data.assert_called_once_with()

    storage.upload_file.assert_not_called()


# =====================================================================
# STORAGE FAILURE
# =====================================================================


def test_extract_job_storage_failure(
    sample_records: list[dict[str, object]],
) -> None:
    """Propagate storage upload failures."""

    client = MagicMock()

    storage = MagicMock()

    client.fetch_market_data.return_value = sample_records

    storage.upload_file.side_effect = RuntimeError("Storage upload failed")

    job = ExtractJob(
        client=client,
        storage=storage,
    )

    with (
        patch(
            "src.extract.extract_job.PathBuilder.build_raw_path",
            return_value=("raw_data/crypto_market/" "crypto_market.ndjson"),
        ),
        pytest.raises(
            RuntimeError,
            match="Storage upload failed",
        ),
    ):
        job.run()

    client.fetch_market_data.assert_called_once_with()

    storage.upload_file.assert_called_once()


# =====================================================================
# CLIENT FAILURE
# =====================================================================


def test_extract_job_client_failure() -> None:
    """Propagate CoinGecko client failures."""

    client = MagicMock()

    storage = MagicMock()

    client.fetch_market_data.side_effect = RuntimeError("CoinGecko API failed")

    job = ExtractJob(
        client=client,
        storage=storage,
    )

    with pytest.raises(
        RuntimeError,
        match="CoinGecko API failed",
    ):
        job.run()

    client.fetch_market_data.assert_called_once_with()

    storage.upload_file.assert_not_called()


# =====================================================================
# NDJSON WRITING
# =====================================================================


def test_extract_job_writes_valid_ndjson(
    sample_records: list[dict[str, object]],
    tmp_path: Path,
) -> None:
    """Verify ExtractJob creates valid NDJSON data."""

    local_path = tmp_path / "crypto_market.ndjson"

    ExtractJob._write_ndjson(
        records=sample_records,
        local_path=local_path,
    )

    assert local_path.exists()

    lines = local_path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == len(sample_records)

    parsed_records = [json.loads(line) for line in lines]

    assert parsed_records == sample_records


# =====================================================================
# EMPTY NDJSON
# =====================================================================


def test_extract_job_writes_empty_ndjson(
    tmp_path: Path,
) -> None:
    """Verify an empty NDJSON file is created."""

    local_path = tmp_path / "empty.ndjson"

    ExtractJob._write_ndjson(
        records=[],
        local_path=local_path,
    )

    assert local_path.exists()

    content = local_path.read_text(
        encoding="utf-8",
    )

    assert content == ""


# =====================================================================
# UNICODE DATA
# =====================================================================


def test_extract_job_preserves_unicode(
    tmp_path: Path,
) -> None:
    """Verify Unicode characters are preserved."""

    records = [
        {
            "id": "bitcoin",
            "name": "Bitcoin ₹",
            "description": "Crypto",
        },
    ]

    local_path = tmp_path / "unicode.ndjson"

    ExtractJob._write_ndjson(
        records=records,
        local_path=local_path,
    )

    content = local_path.read_text(
        encoding="utf-8",
    )

    assert "₹" in content

    assert "Crypto" in content

    parsed = json.loads(content)

    assert parsed == records[0]
