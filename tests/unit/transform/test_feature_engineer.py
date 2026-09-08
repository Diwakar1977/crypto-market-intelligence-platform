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
from src.transform.feature_engineer import FeatureEngineer


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Create SparkSession for feature engineering tests."""

    spark = SparkSessionFactory.create()

    yield spark

    spark.stop()


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Create FeatureEngineer instance."""

    return FeatureEngineer()


@pytest.fixture
def sample_schema() -> StructType:
    """Create schema required for feature engineering."""

    return StructType(
        [
            StructField("id", StringType(), True),
            StructField("symbol", StringType(), True),
            StructField("name", StringType(), True),
            StructField("current_price", DoubleType(), True),
            StructField("high_24h", DoubleType(), True),
            StructField("low_24h", DoubleType(), True),
            StructField("ath", DoubleType(), True),
            StructField("ath_date", TimestampType(), True),
            StructField("atl", DoubleType(), True),
            StructField("atl_date", TimestampType(), True),
            StructField("total_volume", DoubleType(), True),
            StructField("market_cap", DoubleType(), True),
            StructField("circulating_supply", DoubleType(), True),
            StructField("max_supply", DoubleType(), True),
            StructField("price_change_24h", DoubleType(), True),
            StructField("market_cap_rank", LongType(), True),
        ]
    )


@pytest.fixture
def sample_dataframe(
    spark: SparkSession,
    sample_schema: StructType,
) -> DataFrame:
    """Create sample cryptocurrency market DataFrame."""

    data = [
        (
            "bitcoin",
            "btc",
            "Bitcoin",
            90000.0,
            95000.0,
            85000.0,
            120000.0,
            datetime(
                2024,
                3,
                14,
                tzinfo=timezone.utc,
            ),
            1000.0,
            datetime(
                2013,
                7,
                6,
                tzinfo=timezone.utc,
            ),
            50000000000.0,
            1700000000000.0,
            19000000.0,
            21000000.0,
            1000.0,
            1,
        ),
    ]

    return spark.createDataFrame(
        data,
        schema=sample_schema,
    )


def test_validate_required_columns_success(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Validate that all required columns are present."""

    feature_engineer._validate_required_columns(
        sample_dataframe
    )


def test_validate_required_columns_failure(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Raise ValueError when required columns are missing."""

    df = sample_dataframe.drop("ath_date")

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        feature_engineer._validate_required_columns(df)


def test_add_days_since_records(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Create days-since-ATH and days-since-ATL features."""

    result = feature_engineer._add_days_since_records(
        sample_dataframe
    )

    assert "days_since_ath" in result.columns
    assert "days_since_atl" in result.columns

    row = result.select(
        "days_since_ath",
        "days_since_atl",
    ).first()

    assert row is not None
    assert row["days_since_ath"] is not None
    assert row["days_since_atl"] is not None


def test_add_daily_volatility(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Calculate daily price volatility."""

    result = feature_engineer._add_daily_volatility(
        sample_dataframe
    )

    row = result.select(
        "daily_volatility_percentage"
    ).first()

    assert row is not None
    assert row["daily_volatility_percentage"] == 0.11


def test_add_distance_from_ath(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Calculate distance from all-time high."""

    result = feature_engineer._add_distance_from_ath(
        sample_dataframe
    )

    row = result.select(
        "distance_from_ath"
    ).first()

    assert row is not None
    assert row["distance_from_ath"] == 0.25


def test_add_distance_from_atl(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Calculate distance from all-time low."""

    result = feature_engineer._add_distance_from_atl(
        sample_dataframe
    )

    row = result.select(
        "distance_from_atl"
    ).first()

    assert row is not None
    assert row["distance_from_atl"] == -89.0


def test_add_volume_market_cap_ratio(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Calculate volume-to-market-cap ratio."""

    result = feature_engineer._add_volume_market_cap_ratio(
        sample_dataframe
    )

    row = result.select(
        "volume_market_cap_ratio"
    ).first()

    assert row is not None
    assert row["volume_market_cap_ratio"] == 0.0294


def test_add_supply_utilization(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Calculate circulating supply utilization."""

    result = feature_engineer._add_supply_utilization(
        sample_dataframe
    )

    row = result.select(
        "supply_utilization_pct"
    ).first()

    assert row is not None
    assert row["supply_utilization_pct"] == 0.9


def test_add_price_direction(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Classify positive price movement as UP."""

    result = feature_engineer._add_price_direction(
        sample_dataframe
    )

    row = result.select(
        "price_direction"
    ).first()

    assert row is not None
    assert row["price_direction"] == "UP"


def test_price_direction_down(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Classify negative price movement as DOWN."""

    df = sample_dataframe.withColumn(
        "price_change_24h",
        sample_dataframe["price_change_24h"] * -1,
    )

    result = feature_engineer._add_price_direction(df)

    row = result.select(
        "price_direction"
    ).first()

    assert row is not None
    assert row["price_direction"] == "DOWN"


def test_price_direction_flat(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Classify zero price movement as FLAT."""

    df = sample_dataframe.withColumn(
        "price_change_24h",
        sample_dataframe["price_change_24h"] * 0,
    )

    result = feature_engineer._add_price_direction(df)

    row = result.select(
        "price_direction"
    ).first()

    assert row is not None
    assert row["price_direction"] == "FLAT"


def test_transform(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Create all configured feature columns."""

    result = feature_engineer.transform(
        sample_dataframe
    )

    for column_name in FeatureEngineer.FEATURE_COLUMNS:
        assert column_name in result.columns


def test_transform_preserves_original_columns(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Ensure original columns are preserved."""

    original_columns = set(
        sample_dataframe.columns
    )

    result = feature_engineer.transform(
        sample_dataframe
    )

    assert original_columns.issubset(
        set(result.columns)
    )


def test_transform_creates_expected_feature_values(
    feature_engineer: FeatureEngineer,
    sample_dataframe: DataFrame,
) -> None:
    """Validate complete feature-engineering output."""

    result = feature_engineer.transform(
        sample_dataframe
    )

    row = result.select(
        "daily_volatility_percentage",
        "distance_from_ath",
        "distance_from_atl",
        "volume_market_cap_ratio",
        "supply_utilization_pct",
        "price_direction",
    ).first()

    assert row is not None

    assert row["daily_volatility_percentage"] == 0.11
    assert row["distance_from_ath"] == 0.25
    assert row["distance_from_atl"] == -89.0
    assert row["volume_market_cap_ratio"] == 0.0294
    assert row["supply_utilization_pct"] == 0.9
    assert row["price_direction"] == "UP"