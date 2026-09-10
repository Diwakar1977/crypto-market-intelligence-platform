from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.spark.spark_session import SparkSessionFactory
from src.transform.data_validator import (
    DataValidator,
    ValidationResult,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create SparkSession using the project SparkSessionFactory."""

    return SparkSessionFactory.create()


@pytest.fixture
def validator() -> DataValidator:
    """Create DataValidator instance."""

    return DataValidator()


@pytest.fixture
def valid_timestamp() -> datetime:
    """Return a timezone-aware test timestamp."""

    return datetime(
        2026,
        8,
        26,
        4,
        34,
        20,
        tzinfo=timezone.utc,
    )


def create_test_dataframe(
    spark: SparkSession,
    data: list[tuple],
) -> DataFrame:
    """Create a test cryptocurrency DataFrame."""

    schema = StructType(
        [
            StructField(
                "id",
                StringType(),
                True,
            ),
            StructField(
                "symbol",
                StringType(),
                True,
            ),
            StructField(
                "name",
                StringType(),
                True,
            ),
            StructField(
                "current_price",
                DoubleType(),
                True,
            ),
            StructField(
                "market_cap",
                LongType(),
                True,
            ),
            StructField(
                "market_cap_rank",
                LongType(),
                True,
            ),
            StructField(
                "total_volume",
                DoubleType(),
                True,
            ),
            StructField(
                "high_24h",
                DoubleType(),
                True,
            ),
            StructField(
                "low_24h",
                DoubleType(),
                True,
            ),
            StructField(
                "last_updated",
                TimestampType(),
                True,
            ),
        ]
    )

    return spark.createDataFrame(
        data,
        schema,
    )


def test_valid_data(
    spark: SparkSession,
    validator: DataValidator,
    valid_timestamp: datetime,
) -> None:
    """Validate a completely valid cryptocurrency record."""

    data = [
        (
            "bitcoin",
            "btc",
            "Bitcoin",
            78910.0,
            1584176549320,
            1,
            35941830125.0,
            80839.0,
            77955.0,
            valid_timestamp,
        ),
    ]

    df = create_test_dataframe(
        spark,
        data,
    )

    result = validator.validate(df)

    assert isinstance(
        result,
        ValidationResult,
    )
    assert result.is_valid is True
    assert result.invalid_count == 0


def test_missing_required_column(
    spark: SparkSession,
    validator: DataValidator,
) -> None:
    """Raise ValueError when a required column is missing."""

    data = [
        (
            "bitcoin",
            "btc",
            "Bitcoin",
            78910.0,
            1584176549320,
            1,
            35941830125.0,
        ),
    ]

    schema = StructType(
        [
            StructField(
                "id",
                StringType(),
                True,
            ),
            StructField(
                "symbol",
                StringType(),
                True,
            ),
            StructField(
                "name",
                StringType(),
                True,
            ),
            StructField(
                "current_price",
                DoubleType(),
                True,
            ),
            StructField(
                "market_cap",
                LongType(),
                True,
            ),
            StructField(
                "market_cap_rank",
                LongType(),
                True,
            ),
            StructField(
                "total_volume",
                DoubleType(),
                True,
            ),
        ]
    )

    df = spark.createDataFrame(
        data,
        schema,
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns: last_updated",
    ):
        validator.validate(df)


def test_null_required_value(
    spark: SparkSession,
    validator: DataValidator,
    valid_timestamp: datetime,
) -> None:
    """Detect NULL values in required columns."""

    data = [
        (
            None,
            "btc",
            "Bitcoin",
            78910.0,
            1584176549320,
            1,
            35941830125.0,
            80839.0,
            77955.0,
            valid_timestamp,
        ),
    ]

    df = create_test_dataframe(
        spark,
        data,
    )

    result = validator.validate(df)

    assert result.is_valid is False
    assert result.invalid_count == 1


def test_negative_value(
    spark: SparkSession,
    validator: DataValidator,
    valid_timestamp: datetime,
) -> None:
    """Detect negative values in non-negative columns."""

    data = [
        (
            "bitcoin",
            "btc",
            "Bitcoin",
            -100.0,
            1584176549320,
            1,
            35941830125.0,
            80839.0,
            77955.0,
            valid_timestamp,
        ),
    ]

    df = create_test_dataframe(
        spark,
        data,
    )

    result = validator.validate(df)

    assert result.is_valid is False
    assert result.invalid_count == 1


def test_invalid_market_cap_rank(
    spark: SparkSession,
    validator: DataValidator,
    valid_timestamp: datetime,
) -> None:
    """Detect zero or negative market-cap rank."""

    data = [
        (
            "bitcoin",
            "btc",
            "Bitcoin",
            78910.0,
            1584176549320,
            0,
            35941830125.0,
            80839.0,
            77955.0,
            valid_timestamp,
        ),
    ]

    df = create_test_dataframe(
        spark,
        data,
    )

    result = validator.validate(df)

    assert result.is_valid is False
    assert result.invalid_count == 1


def test_high_price_less_than_low_price(
    spark: SparkSession,
    validator: DataValidator,
    valid_timestamp: datetime,
) -> None:
    """Detect invalid high/low price relationship."""

    data = [
        (
            "bitcoin",
            "btc",
            "Bitcoin",
            78910.0,
            1584176549320,
            1,
            35941830125.0,
            70000.0,
            80000.0,
            valid_timestamp,
        ),
    ]

    df = create_test_dataframe(
        spark,
        data,
    )

    result = validator.validate(df)

    assert result.is_valid is False
    assert result.invalid_count == 1
