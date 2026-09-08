from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.utils.logger import Logger

logger = Logger.get_logger(
    "schema_loader",
    "schema_loader.log",
)


class SchemaLoader:
    """Load a JSON schema and convert it to a Spark StructType."""

    TYPE_MAPPING: ClassVar[dict[str, DataType]] = {
        "StringType": StringType(),
        "LongType": LongType(),
        "DoubleType": DoubleType(),
        "BooleanType": BooleanType(),
        "TimestampType": TimestampType(),
        "StructType": StructType(),
        "ArrayType": ArrayType(StringType()),
    }

    def load(
        self,
        schema_path: Path,
    ) -> StructType:
        """Load a JSON schema from a file."""

        logger.info(
            "Starting schema loading from: %s",
            schema_path,
        )

        if not schema_path.exists():
            logger.error(
                "Schema file not found: %s",
                schema_path,
            )

            raise FileNotFoundError(
                f"Schema file not found: {schema_path}"
            )

        with schema_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            schema_data: Any = json.load(file)

        logger.info(
            "Schema JSON loaded successfully from: %s",
            schema_path,
        )

        if not isinstance(schema_data, dict):
            logger.error(
                "Schema validation failed: "
                "schema must contain a JSON object"
            )

            raise TypeError(
                "Schema must contain a JSON object."
            )

        fields: list[StructField] = []

        for column_name, data_type in schema_data.items():

            if not isinstance(column_name, str):
                logger.error(
                    "Schema validation failed: "
                    "column name must be a string"
                )

                raise TypeError(
                    "Schema column name must be a string."
                )

            if not isinstance(data_type, str):
                logger.error(
                    "Schema validation failed: "
                    "type for column '%s' must be a string",
                    column_name,
                )

                raise TypeError(
                    f"Schema type for column "
                    f"'{column_name}' must be a string."
                )

            spark_type = self.TYPE_MAPPING.get(data_type)

            if spark_type is None:
                logger.error(
                    "Unsupported Spark type '%s' "
                    "for column '%s'",
                    data_type,
                    column_name,
                )

                raise ValueError(
                    f"Unsupported Spark type "
                    f"'{data_type}' for column "
                    f"'{column_name}'."
                )

            fields.append(
                StructField(
                    name=column_name,
                    dataType=spark_type,
                    nullable=True,
                )
            )

        logger.info(
            "Schema loaded successfully. "
            "Columns loaded: %d",
            len(fields),
        )

        return StructType(fields)