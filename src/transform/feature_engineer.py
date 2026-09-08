from __future__ import annotations

from typing import ClassVar

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, current_date, datediff, expr, lit, round, when

from src.utils.logger import Logger

logger = Logger.get_logger(
    "feature_engineer",
    "feature_engineer.log"
)

class FeatureEngineer:
    """Create derived business features for cryptocurrency market data."""

    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "days_since_ath",
        "days_since_atl",
        "daily_volatility_percentage",
        "distance_from_ath",
        "distance_from_atl",
        "volume_market_cap_ratio",
        "supply_utilization_pct",
        "price_direction"
    )

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "ath_date",
        "atl_date",
        "high_24h",
        "low_24h",
        "current_price",
        "ath",
        "atl",
        "total_volume",
        "market_cap",
        "circulating_supply",
        "max_supply",
        "price_change_24h"
    )

    def transform(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Create configured cryptocurrency market features."""

        logger.info("Starating feature engineering.")

        try:
            self._validate_required_columns(df)

            result = df

            result = self._add_days_since_records(result)

            result = self._add_daily_volatility(result)

            result = self._add_distance_from_ath(result)

            result = self._add_distance_from_atl(result)

            result = self._add_volume_market_cap_ratio(result)

            result = self._add_supply_utilization(result)

            result = self._add_price_direction(result)

            logger.info("Feature engineering completed successfully.")

            return result

        except Exception:
            logger.exception("Feature engineering failed.")
            raise

    def _validate_required_columns(
        self,
        df: DataFrame
    ) -> None:
        """Validate columns required for feature engineering."""

        missing_columns = [
            column 
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns for feature engineering:",
                f"{missing_columns}"
            )

    def _add_days_since_records(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Calculate days since ATH and ATL."""

        return (
            df.withColumn(
                "days_since_ath",
                datediff(
                    current_date(),
                    col("ath_date")
                )
            )
            .withColumn(
                "days_since_atl",
                datediff(
                    current_date(),
                    col("atl_date")
                )
            )
        )

    def _add_daily_volatility(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Calculate daily price volatility."""

        return df.withColumn(
            "daily_volatility_percentage",
            round(
                expr(
                    "try_divide(high_24h - low_24h, current_price)"
                ),
                2
            )
        )

    def _add_distance_from_ath(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Calculate normalized distance from all-time high."""

        return df.withColumn(
            "distance_from_ath",
            round(
                expr(
                    "try_divide(ath - current_price, ath)"
                ),
                2
            )
        )

    def _add_distance_from_atl(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Calculate normalize distance from all-time low."""

        return df.withColumn(
            "distance_from_atl",
            round(
                expr(
                    "try_divide(atl - current_price, atl)"
                ),
                2
            )
        )

    def _add_volume_market_cap_ratio(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Calculate trading volume to market-cap ration."""

        return df.withColumn(
            "volume_market_cap_ratio",
            round(
                expr(
                    "try_divide(total_volume, market_cap)"
                ),
                4
            )
        )

    def _add_supply_utilization(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Calculate circulating supply utilization."""

        return df.withColumn(
            "supply_utilization_pct",
            round(
                expr(
                    "try_divide(circulating_supply, max_supply)"
                ),
                2
            )
        )

    def _add_price_direction(
        self,
        df: DataFrame
    ) -> DataFrame:
        """Classify 24-hour price movement."""

        return df.withColumn(
            "price_direction",
            when(
                col("price_change_24h") > 0,
                lit("UP")
            )
            .when(
                col("price_change_24h") < 0,
                lit("DOWN")
            )
            .otherwise(
                lit("FLAT")
            )
        )