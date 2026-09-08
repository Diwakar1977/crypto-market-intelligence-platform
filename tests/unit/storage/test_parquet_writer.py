from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession

from src.spark.spark_session import SparkSessionFactory
from src.storage.parquet_writer import ParquetWriter


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create SparkSession for Parquet writer tests."""

    return SparkSessionFactory.create()


@pytest.fixture
def parquet_writer() -> ParquetWriter:
    """Create ParquetWriter instance."""

    return ParquetWriter()


@pytest.fixture
def sample_dataframe(
    spark: SparkSession,
) -> DataFrame:
    """Create sample DataFrame for Parquet tests."""

    data = [
        ("bitcoin", "btc", 100000.0),
        ("ethereum", "eth", 4000.0),
        ("solana", "sol", 200.0),
    ]

    columns = [
        "id",
        "symbol",
        "current_price",
    ]

    return spark.createDataFrame(
        data,
        columns,
    )


def test_write_parquet_success(
    parquet_writer: ParquetWriter,
    sample_dataframe: DataFrame,
    tmp_path: Path,
) -> None:
    """Write DataFrame successfully to Parquet."""

    output_path = str(
        tmp_path / "processed"
    )

    parquet_writer.write(
        sample_dataframe,
        output_path,
    )

    result = sample_dataframe.sparkSession.read.parquet(
        output_path
    )

    assert result.count() == 3
    assert result.columns == [
        "id",
        "symbol",
        "current_price",
    ]


def test_write_parquet_preserves_data(
    parquet_writer: ParquetWriter,
    sample_dataframe: DataFrame,
    tmp_path: Path,
) -> None:
    """Verify written Parquet data matches source data."""

    output_path = str(
        tmp_path / "processed"
    )

    parquet_writer.write(
        sample_dataframe,
        output_path,
    )

    result = sample_dataframe.sparkSession.read.parquet(
        output_path
    )

    actual = {
        row["id"]: row["current_price"]
        for row in result.collect()
    }

    expected = {
        "bitcoin": 100000.0,
        "ethereum": 4000.0,
        "solana": 200.0,
    }

    assert actual == expected


def test_write_parquet_append_mode(
    parquet_writer: ParquetWriter,
    sample_dataframe: DataFrame,
    tmp_path: Path,
) -> None:
    """Verify append mode adds records to existing dataset."""

    output_path = str(
        tmp_path / "processed"
    )

    parquet_writer.write(
        sample_dataframe,
        output_path,
    )

    parquet_writer.write(
        sample_dataframe,
        output_path,
        mode="append",
    )

    result = sample_dataframe.sparkSession.read.parquet(
        output_path
    )

    assert result.count() == 6


def test_write_parquet_overwrite_mode(
    parquet_writer: ParquetWriter,
    sample_dataframe: DataFrame,
    tmp_path: Path,
) -> None:
    """Verify overwrite mode replaces existing dataset."""

    output_path = str(
        tmp_path / "processed"
    )

    parquet_writer.write(
        sample_dataframe,
        output_path,
    )

    replacement_data = [
        ("cardano", "ada", 1.0),
    ]

    replacement_df = sample_dataframe.sparkSession.createDataFrame(
        replacement_data,
        sample_dataframe.columns,
    )

    parquet_writer.write(
        replacement_df,
        output_path,
        mode="overwrite",
    )

    result = sample_dataframe.sparkSession.read.parquet(
        output_path
    )

    assert result.count() == 1

    row = result.first()

    assert row is not None
    assert row["id"] == "cardano"
    assert row["symbol"] == "ada"
    assert row["current_price"] == 1.0


def test_write_parquet_empty_path(
    parquet_writer: ParquetWriter,
    sample_dataframe: DataFrame,
) -> None:
    """Raise ValueError when output path is empty."""

    with pytest.raises(
        ValueError,
        match="Output path cannot be empty",
    ):
        parquet_writer.write(
            sample_dataframe,
            "",
        )


def test_write_parquet_invalid_mode(
    parquet_writer: ParquetWriter,
    sample_dataframe: DataFrame,
    tmp_path: Path,
) -> None:
    """Raise an error when an invalid Spark write mode is supplied."""

    output_path = str(
        tmp_path / "processed"
    )

    with pytest.raises(Exception):
        parquet_writer.write(
            sample_dataframe,
            output_path,
            mode="invalid_mode",
        )
