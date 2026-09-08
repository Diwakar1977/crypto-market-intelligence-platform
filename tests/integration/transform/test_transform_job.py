from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import SparkSession

from src.schema.schema_inferer import SchemaInferer
from src.schema.schema_manager import SchemaManager
from src.storage.parquet_writer import ParquetWriter
from src.storage.path_builder import PathBuilder
from src.transform.data_normalizer import DataNormalizer
from src.transform.data_validator import DataValidator
from src.transform.feature_engineer import FeatureEngineer
from src.transform.transform_job import TransformJob


# =====================================================================
# SPARK FIXTURE
# =====================================================================


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Create one local Spark session for integration tests."""

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("TransformJobIntegrationTest")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )

    yield session

    session.stop()


# =====================================================================
# SAMPLE RAW RECORDS
# =====================================================================


@pytest.fixture
def sample_records() -> list[dict[str, object]]:
    """
    Return complete realistic CoinGecko-style raw records.

    IMPORTANT:
    These records contain every column required by:
        - DataValidator
        - FeatureEngineer

    Dictionary order intentionally represents the original
    CoinGecko API / NDJSON column order.
    """

    return [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 105000.50,
            "market_cap": 2100000000000,
            "market_cap_rank": 1,
            "fully_diluted_valuation": 2200000000000,
            "total_volume": 45000000000.00,
            "high_24h": 106000.00,
            "low_24h": 103000.00,
            "price_change_24h": 2500.50,
            "price_change_percentage_24h": 2.50,
            "market_cap_change_24h": 50000000000.00,
            "market_cap_change_percentage_24h": 2.44,
            "circulating_supply": 19750000.0,
            "total_supply": 21000000.0,
            "max_supply": 21000000.0,
            "ath": 126000.00,
            "ath_change_percentage": -16.67,
            "ath_date": "2025-10-06T00:00:00Z",
            "atl": 67.81,
            "atl_change_percentage": 154800.00,
            "atl_date": "2013-07-06T00:00:00Z",
            "last_updated": "2026-09-08T10:00:00Z",
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "name": "Ethereum",
            "current_price": 4200.75,
            "market_cap": 500000000000,
            "market_cap_rank": 2,
            "fully_diluted_valuation": 505000000000,
            "total_volume": 18000000000.00,
            "high_24h": 4300.00,
            "low_24h": 4100.00,
            "price_change_24h": -52.25,
            "price_change_percentage_24h": -1.25,
            "market_cap_change_24h": -6000000000.00,
            "market_cap_change_percentage_24h": -1.18,
            "circulating_supply": 120500000.0,
            "total_supply": 120500000.0,
            "max_supply": None,
            "ath": 4878.26,
            "ath_change_percentage": -13.89,
            "ath_date": "2021-11-10T14:24:19Z",
            "atl": 0.432979,
            "atl_change_percentage": 970000.00,
            "atl_date": "2015-10-20T00:00:00Z",
            "last_updated": "2026-09-08T10:00:00Z",
        },
    ]


# =====================================================================
# WRITE LOCAL NDJSON
# =====================================================================


def write_ndjson(
    file_path: Path,
    records: list[dict[str, object]],
) -> None:
    """Write test records as NDJSON."""

    content = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        for record in records
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )


# =====================================================================
# TRANSFORM JOB FIXTURE
# =====================================================================


@pytest.fixture
def transform_job(
    spark: SparkSession,
) -> TransformJob:
    """Create TransformJob with real transformation components."""

    return TransformJob(
        spark=spark,
        schema_inferer=SchemaInferer(),
        schema_manager=SchemaManager(),
        data_validator=DataValidator(),
        data_normalizer=DataNormalizer(),
        feature_engineer=FeatureEngineer(),
        path_builder=PathBuilder(),
        parquet_writer=ParquetWriter(),
    )


# =====================================================================
# COMPLETE TRANSFORM JOB
# =====================================================================


def test_transform_job_end_to_end(
    transform_job: TransformJob,
    sample_records: list[dict[str, object]],
    tmp_path: Path,
) -> None:
    """
    Integration test for the complete TransformJob pipeline.

    Real:
        - Spark
        - Spark JSON reader
        - SchemaInferer
        - SchemaManager
        - DataValidator
        - DataNormalizer
        - FeatureEngineer
        - column ordering

    Mocked:
        - original S3 column-order lookup
        - ParquetWriter.write
        - CONFIG

    No real AWS/S3 resources are used.
    """

    # =================================================================
    # CREATE REAL LOCAL RAW NDJSON
    # =================================================================

    raw_file = tmp_path / "crypto_market.ndjson"

    write_ndjson(
        file_path=raw_file,
        records=sample_records,
    )

    # Spark reads the real local NDJSON file.
    raw_input_path = raw_file.as_uri()

    # =================================================================
    # EXPECTED OUTPUT
    # =================================================================

    expected_processed_path = (
        "s3a://test-bucket/"
        "processed_data/crypto_market/"
        "crypto_market.parquet"
    )

    # =================================================================
    # ORIGINAL RAW COLUMN ORDER
    # =================================================================

    # IMPORTANT:
    # This must match the order in sample_records exactly.
    #
    # Feature-engineered columns are NOT included here.
    # They must appear after these raw columns.

    expected_raw_columns = [
        "id",
        "symbol",
        "name",
        "current_price",
        "market_cap",
        "market_cap_rank",
        "fully_diluted_valuation",
        "total_volume",
        "high_24h",
        "low_24h",
        "price_change_24h",
        "price_change_percentage_24h",
        "market_cap_change_24h",
        "market_cap_change_percentage_24h",
        "circulating_supply",
        "total_supply",
        "max_supply",
        "ath",
        "ath_change_percentage",
        "ath_date",
        "atl",
        "atl_change_percentage",
        "atl_date",
        "last_updated",
    ]

    # =================================================================
    # CAPTURE PARQUET OUTPUT
    # =================================================================

    captured: dict[str, object] = {}

    def fake_write(
        df: object,
        output_path: str,
        mode: str,
    ) -> None:
        """Capture DataFrame instead of writing real Parquet."""

        captured["df"] = df
        captured["output_path"] = output_path
        captured["mode"] = mode

    # =================================================================
    # PATCH ONLY EXTERNAL BOUNDARIES
    # =================================================================

    with patch(
        "src.transform.transform_job.CONFIG",
        {
            "application": {
                "raw_dataset": "crypto_market",
            },
            "s3": {
                "bucket": "test-bucket",
            },
        },
    ), patch.object(
        transform_job,
        "_get_original_json_column_order",
        return_value=expected_raw_columns,
    ), patch.object(
        transform_job.parquet_writer,
        "write",
        side_effect=fake_write,
    ), patch.object(
        transform_job.path_builder,
        "build_processed_path",
        return_value=(
            "processed_data/crypto_market/"
            "crypto_market.parquet"
        ),
    ):

        # =============================================================
        # RUN COMPLETE TRANSFORM JOB
        # =============================================================

        result = transform_job.run(
            input_path=raw_input_path,
        )

    # =================================================================
    # VERIFY RESULT PATH
    # =================================================================

    assert result == expected_processed_path

    # =================================================================
    # VERIFY PARQUET WRITE
    # =================================================================

    assert captured["output_path"] == (
        expected_processed_path
    )

    assert captured["mode"] == "append"

    assert captured["df"] is not None

    processed_df = captured["df"]

    # =================================================================
    # VERIFY SPARK DATAFRAME
    # =================================================================

    assert hasattr(
        processed_df,
        "columns",
    )

    # =================================================================
    # VERIFY RECORD COUNT
    # =================================================================

    assert processed_df.count() == 2

    # =================================================================
    # GET ACTUAL COLUMNS
    # =================================================================

    actual_columns = processed_df.columns

    # =================================================================
    # VERIFY ALL RAW COLUMNS EXIST
    # =================================================================

    for column in expected_raw_columns:
        assert column in actual_columns

    # =================================================================
    # VERIFY RAW COLUMN ORDER
    # =================================================================

    assert actual_columns[
        : len(expected_raw_columns)
    ] == expected_raw_columns

    # =================================================================
    # VERIFY NO DUPLICATE COLUMNS
    # =================================================================

    assert len(actual_columns) == len(
        set(actual_columns)
    )

    # =================================================================
    # VERIFY DERIVED COLUMNS COME LAST
    # =================================================================

    derived_columns = [
        column
        for column in actual_columns
        if column not in expected_raw_columns
    ]

    assert actual_columns[
        len(expected_raw_columns) :
    ] == derived_columns

    # =================================================================
    # VERIFY DERIVED COLUMNS ACTUALLY EXIST
    # =================================================================

    assert len(derived_columns) > 0

    # =================================================================
    # VERIFY ROW VALUES
    # =================================================================

    rows = processed_df.collect()

    assert len(rows) == 2

    bitcoin = next(
        row
        for row in rows
        if row["id"] == "bitcoin"
    )

    ethereum = next(
        row
        for row in rows
        if row["id"] == "ethereum"
    )

    assert bitcoin["symbol"] == "btc"
    assert bitcoin["name"] == "Bitcoin"

    assert ethereum["symbol"] == "eth"
    assert ethereum["name"] == "Ethereum"


# =====================================================================
# EMPTY INPUT PATH
# =====================================================================


def test_transform_job_empty_input_path(
    transform_job: TransformJob,
) -> None:
    """Reject an empty input path."""

    with pytest.raises(
        ValueError,
        match="Input path cannot be empty.",
    ):
        transform_job.run(
            input_path="",
        )


# =====================================================================
# INVALID INPUT PATH
# =====================================================================


def test_transform_job_invalid_input_path() -> None:
    """Reject an invalid input path."""

    with pytest.raises(
        ValueError,
        match="Expected s3a:// input path",
    ):
        TransformJob._get_original_json_column_order(
            "invalid-path",
        )


# =====================================================================
# EMPTY S3 OBJECT
# =====================================================================


def test_transform_job_empty_s3_object() -> None:
    """Fail when the S3 NDJSON object is empty."""

    raw_input_path = (
        "s3a://test-bucket/"
        "raw_data/crypto_market/"
        "empty.ndjson"
    )

    fake_body = MagicMock()

    fake_body.readline.return_value = b""

    fake_s3 = MagicMock()

    fake_s3.get_object.return_value = {
        "Body": fake_body,
    }

    with patch(
        "src.transform.transform_job.boto3.client",
        return_value=fake_s3,
    ):

        with pytest.raises(
            ValueError,
            match="Raw NDJSON file is empty.",
        ):
            TransformJob._get_original_json_column_order(
                raw_input_path,
            )

    fake_s3.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key=(
            "raw_data/crypto_market/"
            "empty.ndjson"
        ),
    )

    fake_body.close.assert_called_once_with()


# =====================================================================
# INVALID JSON
# =====================================================================


def test_transform_job_invalid_json() -> None:
    """Fail when the first NDJSON line is invalid JSON."""

    raw_input_path = (
        "s3a://test-bucket/"
        "raw_data/crypto_market/"
        "invalid.ndjson"
    )

    fake_body = MagicMock()

    fake_body.readline.return_value = (
        b"{invalid-json}\n"
    )

    fake_s3 = MagicMock()

    fake_s3.get_object.return_value = {
        "Body": fake_body,
    }

    with patch(
        "src.transform.transform_job.boto3.client",
        return_value=fake_s3,
    ):

        with pytest.raises(
            ValueError,
            match="First NDJSON line is not valid JSON.",
        ):
            TransformJob._get_original_json_column_order(
                raw_input_path,
            )

    fake_body.close.assert_called_once_with()


# =====================================================================
# FIRST RECORD NOT JSON OBJECT
# =====================================================================


def test_transform_job_first_record_not_object() -> None:
    """Fail when the first NDJSON record is not an object."""

    raw_input_path = (
        "s3a://test-bucket/"
        "raw_data/crypto_market/"
        "invalid.ndjson"
    )

    fake_body = MagicMock()

    fake_body.readline.return_value = (
        b"[1, 2, 3]\n"
    )

    fake_s3 = MagicMock()

    fake_s3.get_object.return_value = {
        "Body": fake_body,
    }

    with patch(
        "src.transform.transform_job.boto3.client",
        return_value=fake_s3,
    ):

        with pytest.raises(
            ValueError,
            match=(
                "First NDJSON record must be a JSON object."
            ),
        ):
            TransformJob._get_original_json_column_order(
                raw_input_path,
            )

    fake_body.close.assert_called_once_with()
