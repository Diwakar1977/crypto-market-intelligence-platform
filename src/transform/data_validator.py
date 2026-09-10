from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from src.utils.logger import Logger

logger = Logger.get_logger(
    "data_validator",
    "data_validator.log",
)


@dataclass(frozen=True)
class ValidationResult:
    """Represent the result of data validation."""

    is_valid: bool
    invalid_count: int


class DataValidator:
    """Validate cryptocurrency market data."""

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "symbol",
        "name",
        "current_price",
        "market_cap",
        "market_cap_rank",
        "total_volume",
        "last_updated",
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

    def validate(
        self,
        df: DataFrame,
    ) -> ValidationResult:
        """Validate cryptocurrency market data."""

        logger.info("Starting data validation.")

        try:
            self._validate_required_columns(df)

            invalid_count = 0

            invalid_count += self._count_null_values(df)
            invalid_count += self._count_negative_values(df)
            invalid_count += self._count_invalid_positive_values(df)
            invalid_count += self._count_relationship_violations(df)

            is_valid = invalid_count == 0

            if is_valid:
                logger.info("Data validation completed successfully.")
            else:
                logger.error(
                    "Data validation failed. " "Invalid records detected: %d",
                    invalid_count,
                )

            return ValidationResult(
                is_valid=is_valid,
                invalid_count=invalid_count,
            )

        except ValueError:
            raise

        except Exception:
            logger.exception("Unexpected error during data validation.")
            raise

    def _validate_required_columns(
        self,
        df: DataFrame,
    ) -> None:
        """Validate that required columns exist."""

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in df.columns
        ]

        if missing_columns:
            logger.error(
                "Missing required columns: %s",
                missing_columns,
            )

            raise ValueError(
                "Missing required columns: " f"{', '.join(missing_columns)}"
            )

    def _count_null_values(
        self,
        df: DataFrame,
    ) -> int:
        """Count records with NULL required values."""

        conditions = [col(column).isNull() for column in self.REQUIRED_COLUMNS]

        if not conditions:
            return 0

        invalid_condition = conditions[0]

        for condition in conditions[1:]:
            invalid_condition |= condition

        count = df.filter(invalid_condition).count()

        if count > 0:
            logger.warning(
                "Required NULL values found in %d records.",
                count,
            )

        return count

    def _count_negative_values(
        self,
        df: DataFrame,
    ) -> int:
        """Count records containing negative values."""

        conditions = [
            col(column) < 0
            for column in self.NON_NEGATIVE_COLUMNS
            if column in df.columns
        ]

        if not conditions:
            return 0

        invalid_condition = conditions[0]

        for condition in conditions[1:]:
            invalid_condition |= condition

        count = df.filter(invalid_condition).count()

        if count > 0:
            logger.warning(
                "Negative values found in %d records.",
                count,
            )

        return count

    def _count_invalid_positive_values(
        self,
        df: DataFrame,
    ) -> int:
        """Count records with invalid positive values."""

        conditions = [
            col(column) <= 0
            for column in self.POSITIVE_INTEGER_COLUMNS
            if column in df.columns
        ]

        if not conditions:
            return 0

        invalid_condition = conditions[0]

        for condition in conditions[1:]:
            invalid_condition |= condition

        count = df.filter(invalid_condition).count()

        if count > 0:
            logger.warning(
                "Invalid positive values found in %d records.",
                count,
            )

        return count

    def _count_relationship_violations(
        self,
        df: DataFrame,
    ) -> int:
        """Count records violating column relationships."""

        conditions = []

        for higher_column, lower_column in self.RELATIONSHIP_VALIDATION_RULES:
            if higher_column not in df.columns or lower_column not in df.columns:
                continue

            conditions.append(col(higher_column) < col(lower_column))

        if not conditions:
            return 0

        invalid_condition = conditions[0]

        for condition in conditions[1:]:
            invalid_condition |= condition

        count = df.filter(invalid_condition).count()

        if count > 0:
            logger.warning(
                "Relationship violations found in %d records.",
                count,
            )

        return count
