from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from spark.spark_session import SparkSessionFactory
from src.transform.data_normalizer import DataNormalizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_FILE = PROJECT_ROOT / "data" / "sample" / "crypto_market_sample.ndjson"


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create SparkSession using the project SparkSessionFactory."""

    return SparkSessionFactory.create()


@pytest.fixture(scope="module")
def normalizer() -> DataNormalizer:
    """Create DataNormalizer instance."""

    return DataNormalizer()


@pytest.fixture(scope="module")
def sample_schema() -> StructType:
    """Create schema for DataNormalizer unit tests."""

    return StructType(
        [
            StructField("id", StringType(), True),
            StructField("symbol", StringType(), True),
            StructField("name", StringType(), True),
            StructField("image", StringType(), True),
            StructField("current_price", DoubleType(), True),
            StructField("market_cap", DoubleType(), True),
            StructField("market_cap_rank", IntegerType(), True),
            StructField(
                "fully_diluted_valuation",
                DoubleType(),
                True,
            ),
            StructField("total_volume", DoubleType(), True),
            StructField("high_24h", DoubleType(), True),
            StructField("low_24h", DoubleType(), True),
            StructField("price_change_24h", DoubleType(), True),
            StructField(
                "price_change_percentage_24h",
                DoubleType(),
                True,
            ),
            StructField(
                "market_cap_change_24h",
                DoubleType(),
                True,
            ),
            StructField(
                "market_cap_change_percentage_24h",
                DoubleType(),
                True,
            ),
            StructField(
                "circulating_supply",
                DoubleType(),
                True,
            ),
            StructField("total_supply", DoubleType(), True),
            StructField("max_supply", DoubleType(), True),
            StructField("ath", DoubleType(), True),
            StructField(
                "ath_change_percentage",
                DoubleType(),
                True,
            ),
            StructField(
                "ath_date",
                TimestampType(),
                True,
            ),
            StructField("atl", DoubleType(), True),
            StructField(
                "atl_change_percentage",
                DoubleType(),
                True,
            ),
            StructField(
                "atl_date",
                TimestampType(),
                True,
            ),
            StructField(
                "last_updated",
                TimestampType(),
                True,
            ),
            StructField("roi", StringType(), True),
        ]
    )


def create_test_dataframe(
    spark: SparkSession,
    schema: StructType,
    data: list[tuple],
) -> DataFrame:
    """Create a DataFrame for unit tests."""

    return spark.createDataFrame(
        data,
        schema=schema,
    )


def create_valid_row() -> tuple:
    """Create one valid cryptocurrency market record."""

    timestamp = datetime(
        2026,
        8,
        26,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    return (
        "bitcoin",
        "btc",
        "Bitcoin",
        "image",
        78910.123456,
        1500000000000.123,
        1,
        1600000000000.123,
        50000000000.123,
        80000.123456,
        70000.654321,
        100.123456,
        2.345678,
        5000000.123456,
        1.234567,
        19000000.123456,
        21000000.123456,
        21000000.123456,
        108000.123456,
        45.123456,
        timestamp,
        100.123456,
        -20.123456,
        timestamp,
        timestamp,
        "roi",
    )


def test_drop_unwanted_columns(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Drop image and roi columns."""

    df = create_test_dataframe(
        spark,
        sample_schema,
        [create_valid_row()],
    )

    result = normalizer._drop_unwanted_columns(df)

    assert "image" not in result.columns
    assert "roi" not in result.columns


def test_trim_string_columns(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Trim whitespace from configured string columns."""

    row = list(create_valid_row())

    row[0] = "  bitcoin  "
    row[1] = "  btc "
    row[2] = " Bitcoin  "

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._trim_string_columns(df)

    output = result.first()

    assert output is not None
    assert output["id"] == "bitcoin"
    assert output["symbol"] == "btc"
    assert output["name"] == "Bitcoin"


def test_replace_invalid_strings(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Replace empty string values with NULL."""

    row = list(create_valid_row())

    row[0] = ""
    row[1] = ""
    row[2] = ""

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._replace_invalid_strings(df)

    output = result.first()

    assert output is not None
    assert output["id"] is None
    assert output["symbol"] is None
    assert output["name"] is None


def test_remove_duplicates(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Remove duplicate records using id and last_updated."""

    row = create_valid_row()

    df = create_test_dataframe(
        spark,
        sample_schema,
        [row, row],
    )

    assert df.count() == 2

    result = normalizer._remove_duplicates(df)

    assert result.count() == 1


def test_normalize_negative_values(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Replace negative numeric values with NULL."""

    row = list(create_valid_row())

    row[4] = -100.0
    row[5] = -200.0
    row[9] = -300.0
    row[10] = -400.0
    row[18] = -500.0
    row[21] = -600.0

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._normalize_negative_values(df)

    output = result.first()

    assert output is not None
    assert output["current_price"] is None
    assert output["market_cap"] is None
    assert output["high_24h"] is None
    assert output["low_24h"] is None
    assert output["ath"] is None
    assert output["atl"] is None


def test_normalize_positive_integer_values(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Replace zero and negative rank values with NULL."""

    row = list(create_valid_row())

    row[6] = 0

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._normalize_positive_integer_values(df)

    output = result.first()

    assert output is not None
    assert output["market_cap_rank"] is None


def test_normalize_positive_integer_negative_value(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Replace negative rank values with NULL."""

    row = list(create_valid_row())

    row[6] = -1

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._normalize_positive_integer_values(df)

    output = result.first()

    assert output is not None
    assert output["market_cap_rank"] is None


def test_relationship_validation(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Replace high/low relationship violations with NULL."""

    row = list(create_valid_row())

    row[9] = 70000.0
    row[10] = 80000.0

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._normalize_relationship_values(df)

    output = result.select(
        "high_24h",
        "low_24h",
    ).first()

    assert output is not None
    assert output["high_24h"] is None
    assert output["low_24h"] is None


def test_relationship_validation_valid_values(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Keep high and low values when relationship is valid."""

    row = list(create_valid_row())

    row[9] = 80000.0
    row[10] = 70000.0

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._normalize_relationship_values(df)

    output = result.select(
        "high_24h",
        "low_24h",
    ).first()

    assert output is not None
    assert output["high_24h"] == 80000.0
    assert output["low_24h"] == 70000.0


def test_round_numeric_columns(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Round configured numeric columns to two decimals."""

    row = list(create_valid_row())

    row[4] = 78910.123456
    row[9] = 80000.987654
    row[10] = 70000.456789
    row[11] = 123.456789
    row[12] = 5.678912
    row[18] = 108000.123456
    row[19] = 45.678912
    row[21] = 100.987654
    row[22] = -20.123456

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row)],
    )

    result = normalizer._round_numeric_columns(df)

    output = result.first()

    assert output is not None
    assert output["current_price"] == 78910.12
    assert output["high_24h"] == 80000.99
    assert output["low_24h"] == 70000.46
    assert output["price_change_24h"] == 123.46
    assert output["price_change_percentage_24h"] == 5.68
    assert output["ath"] == 108000.12
    assert output["ath_change_percentage"] == 45.68
    assert output["atl"] == 100.99
    assert output["atl_change_percentage"] == -20.12


def test_normalize_full_pipeline(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Run the complete normalization pipeline."""

    row = list(create_valid_row())

    row[0] = "  bitcoin  "
    row[1] = " btc "
    row[2] = " Bitcoin "

    row[4] = 78910.123456
    row[9] = 80000.123456
    row[10] = 70000.987654

    df = create_test_dataframe(
        spark,
        sample_schema,
        [tuple(row), tuple(row)],
    )

    result = normalizer.normalize(df)

    assert result.count() == 1

    assert "image" not in result.columns
    assert "roi" not in result.columns

    output = result.first()

    assert output is not None
    assert output["id"] == "bitcoin"
    assert output["symbol"] == "btc"
    assert output["name"] == "Bitcoin"

    assert output["current_price"] == 78910.12
    assert output["high_24h"] == 80000.12
    assert output["low_24h"] == 70000.99


def test_normalize_empty_dataframe(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Normalize an empty DataFrame without failure."""

    df = spark.createDataFrame(
        [],
        schema=sample_schema,
    )

    result = normalizer.normalize(df)

    assert result.count() == 0
    assert "image" not in result.columns
    assert "roi" not in result.columns


def test_normalizer_preserves_valid_values(
    spark: SparkSession,
    normalizer: DataNormalizer,
    sample_schema: StructType,
) -> None:
    """Verify valid values remain valid after normalization."""

    row = create_valid_row()

    df = create_test_dataframe(
        spark,
        sample_schema,
        [row],
    )

    result = normalizer.normalize(df)

    output = result.first()

    assert output is not None

    assert output["id"] == "bitcoin"
    assert output["symbol"] == "btc"
    assert output["name"] == "Bitcoin"

    assert output["market_cap_rank"] == 1

    assert output["high_24h"] == 80000.12
    assert output["low_24h"] == 70000.65
