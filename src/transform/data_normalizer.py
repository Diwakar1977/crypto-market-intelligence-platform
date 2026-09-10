from __future__ import annotations

from typing import ClassVar

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, round, trim, when

from src.utils.logger import Logger

logger = Logger.get_logger(
    "data_normalizer",
    "data_normalizer.log",
)


class DataNormalizer:
    """Normalize and clean cryptocurrency market data."""

    DROP_COLUMNS: ClassVar[tuple[str, ...]] = (
        "image",
        "roi",
    )

    STRING_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "symbol",
        "name",
    )

    NON_NEGATIVE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "current_price",
        "market_cap",
        "fully_diluted_valuation",
        "total_volume",
        "high_24h",
        "low_24h",
        "circulating_supply",
        "total_supply",
        "max_supply",
        "ath",
        "atl",
    )

    POSITIVE_INTEGER_COLUMNS: ClassVar[tuple[str, ...]] = ("market_cap_rank",)

    RELATIONSHIP_VALIDATION_RULES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("high_24h", "low_24h"),
    )

    ROUNDING_RULES: ClassVar[dict[str, int]] = {
        "current_price": 2,
        "high_24h": 2,
        "low_24h": 2,
        "price_change_24h": 2,
        "price_change_percentage_24h": 2,
        "market_cap_change_24h": 2,
        "market_cap_change_percentage_24h": 2,
        "ath": 2,
        "ath_change_percentage": 2,
        "atl": 2,
        "atl_change_percentage": 2,
    }

    DEDUPLICATE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "last_updated",
    )

    def normalize(
        self,
        df: DataFrame,
    ) -> DataFrame:
        """Clean and normalize cryptocurrency market data."""

        logger.info("Starting data normalization.")

        try:
            result = df

            result = self._drop_unwanted_columns(
                result,
            )

            result = self._trim_string_columns(
                result,
            )

            result = self._replace_invalid_strings(
                result,
            )

            result = self._remove_duplicates(
                result,
            )

            result = self._normalize_negative_values(
                result,
            )

            result = self._normalize_positive_integer_values(
                result,
            )

            result = self._normalize_relationship_values(
                result,
            )

            result = self._round_numeric_columns(
                result,
            )

            logger.info("Data normalization completed successfully.")

            return result

        except Exception:
            logger.exception("Unexpected error during data normalization.")
            raise

    # =============================================================
    # DROP COLUMNS
    # =============================================================

    def _drop_unwanted_columns(
        self,
        df: DataFrame,
    ) -> DataFrame:

        columns_to_drop = [
            column for column in self.DROP_COLUMNS if column in df.columns
        ]

        if not columns_to_drop:
            return df

        logger.info(
            "Dropping unwanted columns: %s",
            columns_to_drop,
        )

        return df.drop(
            *columns_to_drop,
        )

    # =============================================================
    # TRIM STRINGS
    # =============================================================

    def _trim_string_columns(
        self,
        df: DataFrame,
    ) -> DataFrame:

        result = df

        for column_name in self.STRING_COLUMNS:

            if column_name not in result.columns:
                continue

            result = result.withColumn(
                column_name,
                trim(
                    col(column_name),
                ),
            )

        return result

    # =============================================================
    # EMPTY STRINGS
    # =============================================================

    def _replace_invalid_strings(
        self,
        df: DataFrame,
    ) -> DataFrame:

        result = df

        for column_name in self.STRING_COLUMNS:

            if column_name not in result.columns:
                continue

            result = result.withColumn(
                column_name,
                when(
                    col(column_name) == "",
                    lit(None),
                ).otherwise(col(column_name)),
            )

        return result

    # =============================================================
    # DUPLICATES
    # =============================================================

    def _remove_duplicates(
        self,
        df: DataFrame,
    ) -> DataFrame:

        columns = [
            column for column in self.DEDUPLICATE_COLUMNS if column in df.columns
        ]

        if not columns:
            logger.warning(
                "No deduplication columns found. " "Duplicate removal skipped."
            )
            return df

        before_count = df.count()

        result = df.dropDuplicates(
            columns,
        )

        after_count = result.count()

        duplicate_count = before_count - after_count

        if duplicate_count > 0:
            logger.warning(
                "Removed %d duplicate records.",
                duplicate_count,
            )
        else:
            logger.info("No duplicate records found.")

        return result

    # =============================================================
    # NEGATIVE VALUES
    # =============================================================

    def _normalize_negative_values(
        self,
        df: DataFrame,
    ) -> DataFrame:

        result = df

        affected_columns: list[str] = []

        for column_name in self.NON_NEGATIVE_COLUMNS:

            if column_name not in result.columns:
                continue

            invalid_condition = col(column_name) < 0

            invalid_count = result.filter(
                invalid_condition,
            ).count()

            if invalid_count == 0:
                continue

            affected_columns.append(
                column_name,
            )

            result = result.withColumn(
                column_name,
                when(
                    invalid_condition,
                    lit(None),
                ).otherwise(col(column_name)),
            )

        if affected_columns:
            logger.warning(
                "Negative values replaced with NULL " "in columns: %s",
                affected_columns,
            )

        return result

    # =============================================================
    # POSITIVE INTEGER
    # =============================================================

    def _normalize_positive_integer_values(
        self,
        df: DataFrame,
    ) -> DataFrame:

        result = df

        affected_columns: list[str] = []

        for column_name in self.POSITIVE_INTEGER_COLUMNS:

            if column_name not in result.columns:
                continue

            invalid_condition = col(column_name) <= 0

            invalid_count = result.filter(
                invalid_condition,
            ).count()

            if invalid_count == 0:
                continue

            affected_columns.append(
                column_name,
            )

            result = result.withColumn(
                column_name,
                when(
                    invalid_condition,
                    lit(None),
                ).otherwise(col(column_name)),
            )

        if affected_columns:
            logger.warning(
                "Invalid positive integer values replaced " "with NULL in columns: %s",
                affected_columns,
            )

        return result

    # =============================================================
    # RELATIONSHIP
    # =============================================================

    def _normalize_relationship_values(
        self,
        df: DataFrame,
    ) -> DataFrame:

        result = df

        for higher_column, lower_column in self.RELATIONSHIP_VALIDATION_RULES:

            if (
                higher_column not in result.columns
                or lower_column not in result.columns
            ):
                continue

            invalid_condition = col(higher_column) < col(lower_column)

            invalid_count = result.filter(
                invalid_condition,
            ).count()

            if invalid_count == 0:
                continue

            logger.warning(
                "Relationship violation found between " "'%s' and '%s' in %d records.",
                higher_column,
                lower_column,
                invalid_count,
            )

            result = result.select(
                *[
                    (
                        when(
                            invalid_condition,
                            lit(None),
                        )
                        .otherwise(col(column_name))
                        .alias(column_name)
                        if column_name
                        in {
                            higher_column,
                            lower_column,
                        }
                        else col(column_name)
                    )
                    for column_name in result.columns
                ]
            )

        return result

    # =============================================================
    # ROUND NUMBERS
    # =============================================================

    def _round_numeric_columns(
        self,
        df: DataFrame,
    ) -> DataFrame:

        result = df

        for column_name, decimal_places in self.ROUNDING_RULES.items():

            if column_name not in result.columns:
                continue

            result = result.withColumn(
                column_name,
                round(
                    col(column_name),
                    decimal_places,
                ),
            )

        logger.info("Configured numeric columns rounded successfully.")

        return result
