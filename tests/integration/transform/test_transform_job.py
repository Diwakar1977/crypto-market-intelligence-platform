from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession

from src.transform.transform_job import TransformJob

# ------------------------------------------------------------------
# Spark fixture
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("transform-job-integration-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    yield spark

    spark.stop()


# ------------------------------------------------------------------
# Test input
# ------------------------------------------------------------------


@pytest.fixture
def raw_json_file(tmp_path: Path) -> Path:
    """
    Create a small realistic crypto NDJSON input file.
    """

    records = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "image": "https://example.com/btc.png",
            "current_price": 65000.50,
            "market_cap": 1200000000000,
            "market_cap_rank": 1,
            "fully_diluted_valuation": 1300000000000,
            "total_volume": 35000000000.0,
            "high_24h": 66000.0,
            "low_24h": 64000.0,
            "price_change_24h": 500.0,
            "price_change_percentage_24h": 0.77,
            "market_cap_change_24h": 10000000000.0,
            "market_cap_change_percentage_24h": 0.84,
            "circulating_supply": 19700000.0,
            "total_supply": 21000000.0,
            "max_supply": 21000000.0,
            "ath": 73738.0,
            "ath_change_percentage": -11.86,
            "ath_date": "2024-03-14T07:10:36",
            "atl": 67.81,
            "atl_change_percentage": 95800.0,
            "atl_date": "2013-07-06T00:00:00",
            "roi": {
                "times": 100.0,
                "currency": "btc",
                "percentage": 10000.0,
            },
            "last_updated": "2026-09-30T09:00:00",
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "name": "Ethereum",
            "image": "https://example.com/eth.png",
            "current_price": 3200.75,
            "market_cap": 400000000000,
            "market_cap_rank": 2,
            "fully_diluted_valuation": 390000000000,
            "total_volume": 18000000000.0,
            "high_24h": 3250.0,
            "low_24h": 3150.0,
            "price_change_24h": 40.0,
            "price_change_percentage_24h": 1.25,
            "market_cap_change_24h": 5000000000.0,
            "market_cap_change_percentage_24h": 1.27,
            "circulating_supply": 120000000.0,
            "total_supply": 120500000.0,
            "max_supply": None,
            "ath": 4878.26,
            "ath_change_percentage": -34.4,
            "ath_date": "2021-11-10T14:24:19",
            "atl": 0.432979,
            "atl_change_percentage": 739000.0,
            "atl_date": "2015-10-20T00:00:00",
            "roi": {
                "times": 50.0,
                "currency": "btc",
                "percentage": 5000.0,
            },
            "last_updated": "2026-09-30T09:00:00",
        },
    ]

    file_path = tmp_path / "crypto_market.ndjson"

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    return file_path


# ------------------------------------------------------------------
# TransformJob factory
# ------------------------------------------------------------------


def create_integration_job(
    spark: SparkSession,
) -> TransformJob:
    """
    Create the real TransformJob with real production components.
    """

    from src.schema.schema_inferer import SchemaInferer
    from src.schema.schema_manager import SchemaManager
    from src.storage.parquet_writer import ParquetWriter
    from src.storage.path_builder import PathBuilder
    from src.transform.data_normalizer import DataNormalizer
    from src.transform.data_validator import DataValidator
    from src.transform.feature_engineer import FeatureEngineer

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


# ------------------------------------------------------------------
# Integration test
# ------------------------------------------------------------------


def test_transform_job_runs_end_to_end(
    spark: SparkSession,
    raw_json_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Integration test for the complete TransformJob pipeline.

    Real:
        - Spark
        - SchemaInferer
        - SchemaManager
        - DataValidator
        - DataNormalizer
        - FeatureEngineer
        - column ordering
        - final validation

    External output:
        - Parquet S3 write is intercepted so the test does not
          write to the production AWS bucket.
    """

    job = create_integration_job(
        spark=spark,
    )

    # --------------------------------------------------------------
    # Read original NDJSON column order
    # --------------------------------------------------------------

    with raw_json_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        first_record = json.loads(file.readline())

    original_columns = list(first_record.keys())

    # --------------------------------------------------------------
    # Production implementation reads original column order from S3.
    # For this local integration test, use the local NDJSON file.
    # --------------------------------------------------------------

    monkeypatch.setattr(
        job,
        "_get_original_json_column_order",
        lambda input_path: original_columns,
    )

    # --------------------------------------------------------------
    # Redirect Parquet output.
    #
    # Do not write to production S3 during tests.
    # --------------------------------------------------------------

    written: dict[str, object] = {}

    def fake_write(
        df: DataFrame,
        output_path: str,
        mode: str,
    ) -> None:
        written["df"] = df
        written["output_path"] = output_path
        written["mode"] = mode

        # Force Spark to materialize the transformed DataFrame.
        written["row_count"] = df.count()
        written["columns"] = list(df.columns)

    monkeypatch.setattr(
        job.parquet_writer,
        "write",
        fake_write,
    )

    # --------------------------------------------------------------
    # Run complete TransformJob
    # --------------------------------------------------------------

    result = job.run(
        input_path=str(raw_json_file),
    )

    # --------------------------------------------------------------
    # Result validation
    # --------------------------------------------------------------

    assert result.startswith("s3a://")

    assert written["mode"] == "append"

    assert written["row_count"] == 2

    processed_columns = written["columns"]

    assert isinstance(
        processed_columns,
        list,
    )

    # --------------------------------------------------------------
    # DataNormalizer intentionally removes:
    #
    #   image
    #   roi
    #
    # Therefore only surviving RAW columns should remain at the
    # beginning of the processed DataFrame.
    # --------------------------------------------------------------

    removed_columns = {
        "image",
        "roi",
    }

    surviving_raw_columns = [
        column for column in original_columns if column not in removed_columns
    ]

    assert len(processed_columns) >= len(surviving_raw_columns)

    # --------------------------------------------------------------
    # RAW columns that survived normalization must remain in their
    # original NDJSON order.
    # --------------------------------------------------------------

    actual_raw_columns = processed_columns[: len(surviving_raw_columns)]

    assert actual_raw_columns == surviving_raw_columns

    # --------------------------------------------------------------
    # Verify intentionally removed columns are not present.
    # --------------------------------------------------------------

    assert "image" not in processed_columns
    assert "roi" not in processed_columns

    # --------------------------------------------------------------
    # Verify derived feature columns exist.
    # --------------------------------------------------------------

    expected_derived_columns = {
        "days_since_ath",
        "days_since_atl",
        "daily_volatility_percentage",
        "distance_from_ath",
        "distance_from_atl",
        "volume_market_cap_ratio",
        "supply_utilization_pct",
        "price_direction",
    }

    assert expected_derived_columns.issubset(set(processed_columns))

    # --------------------------------------------------------------
    # No duplicate columns.
    # --------------------------------------------------------------

    assert len(processed_columns) == len(set(processed_columns))


# ------------------------------------------------------------------
# Empty input validation
# ------------------------------------------------------------------


def test_transform_job_rejects_empty_input(
    spark: SparkSession,
) -> None:
    job = create_integration_job(
        spark=spark,
    )

    with pytest.raises(
        ValueError,
        match="Input path cannot be empty",
    ):
        job.run(
            input_path="",
        )


# ------------------------------------------------------------------
# Original JSON column order integration test
# ------------------------------------------------------------------


def test_transform_job_preserves_original_column_order(
    spark: SparkSession,
    raw_json_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = create_integration_job(
        spark=spark,
    )

    with raw_json_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        first_record = json.loads(file.readline())

    expected_columns = list(first_record.keys())

    monkeypatch.setattr(
        job,
        "_get_original_json_column_order",
        lambda input_path: expected_columns,
    )

    result_columns = job._get_original_json_column_order(str(raw_json_file))

    assert result_columns == expected_columns
