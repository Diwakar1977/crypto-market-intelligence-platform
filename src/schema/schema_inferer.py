from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar

from src.utils.logger import Logger

logger = Logger.get_logger(
    "schema_inferer",
    "schema_inferer.log",
)


class SchemaInferer:
    """Infer normalized data types automatically from source records."""

    TYPE_MAP: ClassVar[dict[type, str]] = {
        bool: "boolean",
        int: "integer",
        float: "double",
        str: "string",
        datetime: "timestamp",
        date: "timestamp",
    }

    NUMERIC_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "integer",
            "double",
        }
    )

    def infer(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, str]:
        """
        Infer column names and data types automatically.

        Rules
        -----
        integer + integer
            -> integer

        integer + double
            -> double

        double + double
            -> double

        ISO-8601 date/time string
            -> timestamp

        None values
            -> ignored during type inference

        Columns containing only None
            -> string

        Column order
            -> first-seen source order
        """

        logger.info(
            "Starting schema inference for %d records",
            len(records),
        )

        if not records:
            raise ValueError("Cannot infer schema from empty records.")

        # Preserve the original source column order.
        column_order: list[str] = []

        # Store inferred datatype for each column.
        schema: dict[str, str] = {}

        # ---------------------------------------------------------
        # Infer every column from every record
        # ---------------------------------------------------------

        for record_number, record in enumerate(records, start=1):

            if not isinstance(record, dict):
                raise TypeError(
                    "Every record must be a dictionary. "
                    f"Invalid record at position {record_number}."
                )

            for column_name, value in record.items():

                # Preserve first-seen source order.
                if column_name not in column_order:
                    column_order.append(column_name)

                # None never determines datatype.
                if value is None:
                    continue

                inferred_type = self._infer_type(value)

                if column_name not in schema:
                    schema[column_name] = inferred_type
                    continue

                schema[column_name] = self._resolve_type(
                    current_type=schema[column_name],
                    new_type=inferred_type,
                )

        # ---------------------------------------------------------
        # Columns containing only NULL
        # ---------------------------------------------------------

        for column_name in column_order:
            if column_name not in schema:
                schema[column_name] = "string"

        # ---------------------------------------------------------
        # Preserve original column order
        # ---------------------------------------------------------

        ordered_schema = {
            column_name: schema[column_name] for column_name in column_order
        }

        logger.info(
            "Schema inference completed successfully. " "Columns inferred=%d",
            len(ordered_schema),
        )

        return ordered_schema

    # =============================================================
    # TYPE INFERENCE
    # =============================================================

    def _infer_type(
        self,
        value: Any,
    ) -> str:
        """Infer the normalized type of a single value."""

        # ---------------------------------------------------------
        # IMPORTANT:
        # bool must be checked before int because Python bool
        # is a subclass of int.
        # ---------------------------------------------------------

        if isinstance(value, bool):
            return "boolean"

        # ---------------------------------------------------------
        # Python integer
        # ---------------------------------------------------------

        if isinstance(value, int):
            return "integer"

        # ---------------------------------------------------------
        # Python float
        # ---------------------------------------------------------

        if isinstance(value, float):
            return "double"

        # ---------------------------------------------------------
        # Python datetime
        # ---------------------------------------------------------

        if isinstance(value, datetime):
            return "timestamp"

        # ---------------------------------------------------------
        # Python date
        # ---------------------------------------------------------

        if isinstance(value, date):
            return "timestamp"

        # ---------------------------------------------------------
        # String
        # ---------------------------------------------------------

        if isinstance(value, str):

            if self._is_timestamp(value):
                return "timestamp"

            return "string"

        # ---------------------------------------------------------
        # Array
        # ---------------------------------------------------------

        if isinstance(value, list):
            return "array"

        # ---------------------------------------------------------
        # Object
        # ---------------------------------------------------------

        if isinstance(value, dict):
            return "object"

        # Unknown types safely become string.
        return "string"

    # =============================================================
    # TIMESTAMP DETECTION
    # =============================================================

    @staticmethod
    def _is_timestamp(
        value: str,
    ) -> bool:
        """
        Detect ISO-8601 date/time strings.

        Examples accepted:

            2026-09-02T10:30:00Z
            2026-09-02T10:30:00+00:00
            2026-09-02 10:30:00
            2026-09-02
        """

        cleaned_value = value.strip()

        if not cleaned_value:
            return False

        # Avoid treating arbitrary short strings as timestamps.
        if len(cleaned_value) < 8:
            return False

        try:
            datetime.fromisoformat(
                cleaned_value.replace(
                    "Z",
                    "+00:00",
                )
            )
            return True

        except ValueError:
            pass

        # ---------------------------------------------------------
        # Date-only value
        # ---------------------------------------------------------

        try:
            date.fromisoformat(
                cleaned_value,
            )
            return True

        except ValueError:
            return False

    # =============================================================
    # TYPE RESOLUTION
    # =============================================================

    def _resolve_type(
        self,
        current_type: str,
        new_type: str,
    ) -> str:
        """
        Resolve datatype conflicts across records.

        Numeric rule:

            integer + integer = integer
            integer + double   = double
            double + double    = double

        Timestamp rule:

            timestamp + string = timestamp
            string + timestamp = timestamp

        Other incompatible types:

            -> string
        """

        # Same datatype.
        if current_type == new_type:
            return current_type

        # ---------------------------------------------------------
        # Numeric widening
        # ---------------------------------------------------------

        if current_type in self.NUMERIC_TYPES and new_type in self.NUMERIC_TYPES:
            return "double"

        # ---------------------------------------------------------
        # Timestamp conflicts
        # ---------------------------------------------------------

        if current_type == "timestamp" and new_type == "string":
            return "timestamp"

        if current_type == "string" and new_type == "timestamp":
            return "timestamp"

        # ---------------------------------------------------------
        # Incompatible types
        # ---------------------------------------------------------

        return "string"
