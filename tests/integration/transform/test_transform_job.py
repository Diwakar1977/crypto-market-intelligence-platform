from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import DataFrame, SparkSession

from src.schema.schema_inferer import SchemaInferer
from src.schema.schema_manager import SchemaManager
from spark.spark_session import SparkSessionFactory
from src.storage.parquet_writer import ParquetWriter
from src.storage.path_builder import PathBuilder
from src.transform.data_normalizer import DataNormalizer
from src.transform.data_validator import DataValidator
from src.transform.feature_engineer import FeatureEngineer
from src.transform.transform_job import TransformJob

# =====================================================================
# TYPES
# =====================================================================


class CapturedWrite(TypedDict):
    """Captured ParquetWriter.write arguments."""

    df: DataFrame
    output_path: str
    mode: str


# =====================================================================
# SPARK FIXTURE
# =====================================================================


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Create one application-configured Spark session."""

    session = SparkSessionFactory.create()

    yield session

    session.stop()


# =====================================================================
# SAMPLE RAW RECORDS
# =====================================================================


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    """Return realistic CoinGecko-style raw records."""

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
# WRITE NDJSON
# =====================================================================


def write_ndjson(
    file_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records as NDJSON."""

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
# END-TO-END TRANSFORM TEST
# =====================================================================


def test_transform_job_end_to_end(
    transform_job: TransformJob,
    sample_records: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Verify the complete local TransformJob flow."""

    raw_file = tmp_path / "crypto_market.ndjson"

    write_ndjson(
        file_path=raw_file,
        records=sample_records,
    )

    raw_input_path = raw_file.as_uri()

    expected_processed_path = (
        "s3a://test-bucket/" "processed_data/crypto_market/" "crypto_market.parquet"
    )

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

    captured: CapturedWrite = {
        "df": transform_job.spark.createDataFrame(
            [],
            "id string",
        ),
        "output_path": "",
        "mode": "",
    }

    def fake_write(
        df: DataFrame,
        output_path: str,
        mode: str,
    ) -> None:
        """Capture Parquet write arguments."""

        captured["df"] = df
        captured["output_path"] = output_path
        captured["mode"] = mode

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            {
                "application": {
                    "raw_dataset": "crypto_market",
                },
                "s3": {
                    "bucket": "test-bucket",
                },
            },
        ),
        patch.object(
            transform_job,
            "_get_original_json_column_order",
            return_value=expected_raw_columns,
        ),
        patch.object(
            transform_job.parquet_writer,
            "write",
            side_effect=fake_write,
        ),
        patch.object(
            transform_job.path_builder,
            "build_processed_path",
            return_value=("processed_data/crypto_market/" "crypto_market.parquet"),
        ),
    ):
        result = transform_job.run(
            input_path=raw_input_path,
        )

    assert result == expected_processed_path

    assert captured["output_path"] == expected_processed_path
    assert captured["mode"] == "append"

    processed_df = captured["df"]

    assert processed_df.count() == 2

    actual_columns = processed_df.columns

    assert len(actual_columns) == len(set(actual_columns))

    for column in expected_raw_columns:
        assert column in actual_columns

    assert actual_columns[: len(expected_raw_columns)] == expected_raw_columns

    derived_columns = [
        column for column in actual_columns if column not in expected_raw_columns
    ]

    assert len(derived_columns) > 0

    assert actual_columns[len(expected_raw_columns) :] == derived_columns

    rows = processed_df.collect()

    assert len(rows) == 2

    bitcoin = next(row for row in rows if row["id"] == "bitcoin")

    ethereum = next(row for row in rows if row["id"] == "ethereum")

    assert bitcoin["symbol"] == "btc"
    assert bitcoin["name"] == "Bitcoin"

    assert ethereum["symbol"] == "eth"
    assert ethereum["name"] == "Ethereum"


# =====================================================================
# EMPTY INPUT
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
    """Reject a non-S3A input path."""

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
    """Reject an empty S3 NDJSON object."""

    raw_input_path = "s3a://test-bucket/" "raw_data/crypto_market/" "empty.ndjson"

    fake_body = MagicMock()
    fake_body.readline.return_value = b""

    fake_s3 = MagicMock()
    fake_s3.get_object.return_value = {
        "Body": fake_body,
    }

    with (
        patch(
            "src.transform.transform_job.boto3.client",
            return_value=fake_s3,
        ),
        pytest.raises(
            ValueError,
            match="Raw NDJSON file is empty.",
        ),
    ):
        TransformJob._get_original_json_column_order(
            raw_input_path,
        )

    fake_s3.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="raw_data/crypto_market/empty.ndjson",
    )

    fake_body.close.assert_called_once_with()


# =====================================================================
# INVALID JSON
# =====================================================================


def test_transform_job_invalid_json() -> None:
    """Reject invalid JSON in the first NDJSON record."""

    raw_input_path = "s3a://test-bucket/" "raw_data/crypto_market/" "invalid.ndjson"

    fake_body = MagicMock()
    fake_body.readline.return_value = b"{invalid-json}\n"

    fake_s3 = MagicMock()
    fake_s3.get_object.return_value = {
        "Body": fake_body,
    }

    with (
        patch(
            "src.transform.transform_job.boto3.client",
            return_value=fake_s3,
        ),
        pytest.raises(
            ValueError,
            match="First NDJSON line is not valid JSON.",
        ),
    ):
        TransformJob._get_original_json_column_order(
            raw_input_path,
        )

    fake_body.close.assert_called_once_with()


# =====================================================================
# FIRST RECORD NOT OBJECT
# =====================================================================


def test_transform_job_first_record_not_object() -> None:
    """Reject a JSON array as the first NDJSON record."""

    raw_input_path = "s3a://test-bucket/" "raw_data/crypto_market/" "invalid.ndjson"

    fake_body = MagicMock()
    fake_body.readline.return_value = b"[1, 2, 3]\n"

    fake_s3 = MagicMock()
    fake_s3.get_object.return_value = {
        "Body": fake_body,
    }

    with (
        patch(
            "src.transform.transform_job.boto3.client",
            return_value=fake_s3,
        ),
        pytest.raises(
            TypeError,
            match="First NDJSON record must be a JSON object.",
        ),
    ):
        TransformJob._get_original_json_column_order(
            raw_input_path,
        )

    fake_body.close.assert_called_once_with()


# =====================================================================
# EMPTY JSON OBJECT
# =====================================================================


def test_transform_job_empty_json_object() -> None:
    """Reject an empty JSON object."""

    raw_input_path = (
        "s3a://test-bucket/" "raw_data/crypto_market/" "empty-object.ndjson"
    )

    fake_body = MagicMock()
    fake_body.readline.return_value = b"{}\n"

    fake_s3 = MagicMock()
    fake_s3.get_object.return_value = {
        "Body": fake_body,
    }

    with (
        patch(
            "src.transform.transform_job.boto3.client",
            return_value=fake_s3,
        ),
        pytest.raises(
            ValueError,
            match="First NDJSON record contains no columns.",
        ),
    ):
        TransformJob._get_original_json_column_order(
            raw_input_path,
        )

    fake_body.close.assert_called_once_with()


# =====================================================================
# ORDER PROCESSED COLUMNS
# =====================================================================


def test_order_processed_columns() -> None:
    """Keep raw columns first and derived columns last."""

    raw_columns = [
        "id",
        "symbol",
        "name",
        "current_price",
    ]

    spark = SparkSessionFactory.create()

    try:
        df = spark.createDataFrame(
            [
                (
                    "btc",
                    "Bitcoin",
                    100.0,
                    "bitcoin",
                    10.0,
                ),
            ],
            [
                "symbol",
                "name",
                "current_price",
                "id",
                "price_direction",
            ],
        )

        result = TransformJob._order_processed_columns(
            raw_column_order=raw_columns,
            processed_df=df,
        )

        assert result.columns == [
            "id",
            "symbol",
            "name",
            "current_price",
            "price_direction",
        ]
    finally:
        spark.stop()


# =====================================================================
# FINAL COLUMN VALIDATION
# =====================================================================


def test_validate_final_columns_success() -> None:
    """Accept correctly ordered processed columns."""

    raw_columns = [
        "id",
        "symbol",
        "name",
    ]

    spark = SparkSessionFactory.create()

    try:
        df = spark.createDataFrame(
            [
                (
                    "bitcoin",
                    "btc",
                    "Bitcoin",
                    1.0,
                ),
            ],
            [
                "id",
                "symbol",
                "name",
                "price_direction",
            ],
        )

        TransformJob._validate_final_columns(
            raw_column_order=raw_columns,
            processed_df=df,
        )
    finally:
        spark.stop()


def test_validate_final_columns_rejects_wrong_order() -> None:
    """Reject incorrectly ordered processed columns."""

    raw_columns = [
        "id",
        "symbol",
        "name",
    ]

    spark = SparkSessionFactory.create()

    try:
        df = spark.createDataFrame(
            [
                (
                    "btc",
                    "bitcoin",
                    "Bitcoin",
                ),
            ],
            [
                "symbol",
                "id",
                "name",
            ],
        )

        with pytest.raises(
            ValueError,
            match="RAW column order validation failed",
        ):
            TransformJob._validate_final_columns(
                raw_column_order=raw_columns,
                processed_df=df,
            )
    finally:
        spark.stop()
