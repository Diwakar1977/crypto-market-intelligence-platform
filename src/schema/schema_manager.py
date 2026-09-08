from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    LongType,
    StringType,
    StructType,
    TimestampType,
)

from src.utils.logger import Logger


logger = Logger.get_logger(
    "schema_manager",
    "schema_manager.log",
)


@dataclass(frozen=True)
class SchemaChange:
    """Represent a schema type change."""

    column_name: str
    old_type: str
    new_type: str


class SchemaManager:
    """Manage, validate and apply inferred Spark schemas."""

    TYPE_MAPPING: ClassVar[dict[str, str]] = {
        "string": "StringType",
        "integer": "LongType",
        "double": "DoubleType",
        "boolean": "BooleanType",
        "timestamp": "TimestampType",
        "object": "StructType",
        "array": "ArrayType",
    }

    # =============================================================
    # NORMALIZE
    # =============================================================

    def normalize(
        self,
        schema: dict[str, str],
    ) -> dict[str, str]:
        """
        Convert normalized logical types into Spark type names.

        Column order is preserved exactly as received.
        """

        logger.info(
            "Starting schema normalization for %d columns",
            len(schema),
        )

        if not schema:
            raise ValueError(
                "Schema cannot be empty."
            )

        normalized_schema: dict[str, str] = {}

        for column_name, data_type in schema.items():

            spark_type = self.TYPE_MAPPING.get(
                data_type,
            )

            if spark_type is None:
                logger.warning(
                    "Unknown logical type '%s' for column '%s'. "
                    "Using StringType.",
                    data_type,
                    column_name,
                )

                spark_type = "StringType"

            normalized_schema[column_name] = spark_type

        logger.info(
            "Schema normalization completed successfully. "
            "Columns normalized=%d",
            len(normalized_schema),
        )

        return normalized_schema

    # =============================================================
    # VALIDATE
    # =============================================================

    def validate(
        self,
        schema: dict[str, str],
    ) -> None:
        """Validate normalized Spark type names."""

        logger.info(
            "Starting schema validation for %d columns",
            len(schema),
        )

        if not schema:
            raise ValueError(
                "Schema cannot be empty."
            )

        valid_types = set(
            self.TYPE_MAPPING.values()
        )

        for column_name, data_type in schema.items():

            if data_type not in valid_types:
                raise ValueError(
                    f"Unsupported type '{data_type}' "
                    f"for column '{column_name}'."
                )

        logger.info(
            "Schema validation completed successfully."
        )

    # =============================================================
    # ACTUAL SPARK TYPE
    # =============================================================

    @staticmethod
    def _spark_data_type(
        spark_type_name: str,
    ) -> DataType:
        """Convert Spark type name into actual Spark DataType."""

        mapping: dict[str, DataType] = {
            "StringType": StringType(),
            "LongType": LongType(),
            "DoubleType": DoubleType(),
            "BooleanType": BooleanType(),
            "TimestampType": TimestampType(),
            "StructType": StructType([]),
            "ArrayType": ArrayType(StringType()),
        }

        if spark_type_name not in mapping:
            raise ValueError(
                f"Unsupported Spark type: {spark_type_name}"
            )

        return mapping[spark_type_name]

    # =============================================================
    # APPLY SCHEMA
    # =============================================================

    def apply_schema(
        self,
        df: DataFrame,
        normalized_schema: dict[str, str],
    ) -> DataFrame:
        """
        Apply the managed schema to the Spark DataFrame.

        IMPORTANT:
        - Column order is preserved.
        - Only datatype conversion is performed.
        - No columns are reordered.
        - No columns are added or removed.
        - Nested StructType/ArrayType columns keep Spark's
          inferred nested datatype.
        """

        logger.info(
            "Starting managed schema application."
        )

        if not normalized_schema:
            raise ValueError(
                "Cannot apply an empty schema."
            )

        result = df

        # ---------------------------------------------------------
        # Preserve exact original column order.
        # ---------------------------------------------------------

        original_columns = list(df.columns)

        for column_name in original_columns:

            if column_name not in normalized_schema:
                logger.warning(
                    "Column '%s' is not present in managed schema. "
                    "Leaving datatype unchanged.",
                    column_name,
                )
                continue

            target_type = normalized_schema[column_name]

            # -----------------------------------------------------
            # TIMESTAMP
            # -----------------------------------------------------

            if target_type == "TimestampType":

                result = result.withColumn(
                    column_name,
                    to_timestamp(
                        col(column_name),
                    ),
                )

                continue

            # -----------------------------------------------------
            # LONG
            # -----------------------------------------------------

            if target_type == "LongType":

                result = result.withColumn(
                    column_name,
                    col(column_name).cast(
                        LongType()
                    ),
                )

                continue

            # -----------------------------------------------------
            # DOUBLE
            # -----------------------------------------------------

            if target_type == "DoubleType":

                result = result.withColumn(
                    column_name,
                    col(column_name).cast(
                        DoubleType()
                    ),
                )

                continue

            # -----------------------------------------------------
            # BOOLEAN
            # -----------------------------------------------------

            if target_type == "BooleanType":

                result = result.withColumn(
                    column_name,
                    col(column_name).cast(
                        BooleanType()
                    ),
                )

                continue

            # -----------------------------------------------------
            # STRING
            # -----------------------------------------------------

            if target_type == "StringType":

                result = result.withColumn(
                    column_name,
                    col(column_name).cast(
                        StringType()
                    ),
                )

                continue

            # -----------------------------------------------------
            # STRUCT / ARRAY
            # -----------------------------------------------------

            if target_type in {
                "StructType",
                "ArrayType",
            }:
                # Spark already inferred the nested datatype.
                # Keep it unchanged.
                continue

            raise ValueError(
                f"Unsupported managed type '{target_type}' "
                f"for column '{column_name}'."
            )

        # ---------------------------------------------------------
        # Restore EXACT original column order.
        # ---------------------------------------------------------

        result = result.select(
            *original_columns,
        )

        logger.info(
            "Managed schema applied successfully."
        )

        return result
