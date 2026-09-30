from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pyarrow as pa


@dataclass(frozen=True)
class RedshiftColumn:
    """Represent one Redshift table column."""

    name: str
    data_type: str
    nullable: bool = True

    def to_sql(self) -> str:
        """Generate the SQL definition for this column."""

        nullability = "" if self.nullable else " NOT NULL"

        return (
            f"{PyArrowSchemaMapper.quote_identifier(self.name)} "
            f"{self.data_type}"
            f"{nullability}"
        )


class PyArrowSchemaMapper:
    """
    Convert a PyArrow schema into a Redshift CREATE TABLE definition.

    This class contains no:
        - boto3 logic
        - S3 logic
        - Redshift connection logic
        - Spark logic

    It is responsible only for schema/type mapping and SQL generation.
    """

    DEFAULT_VARCHAR_LENGTH: ClassVar[int] = 65535

    # ------------------------------------------------------------------
    # Basic Arrow → Redshift type mapping
    # ------------------------------------------------------------------

    TYPE_MAPPING: ClassVar[dict[str, str]] = {
        "bool": "BOOLEAN",
        "int8": "SMALLINT",
        "int16": "SMALLINT",
        "int32": "INTEGER",
        "int64": "BIGINT",
        "uint8": "SMALLINT",
        "uint16": "INTEGER",
        "uint32": "BIGINT",
        "uint64": "DECIMAL(20,0)",
        "float": "REAL",
        "double": "DOUBLE PRECISION",
        "string": f"VARCHAR({DEFAULT_VARCHAR_LENGTH})",
        "large_string": f"VARCHAR({DEFAULT_VARCHAR_LENGTH})",
        "binary": "VARBYTE",
        "large_binary": "VARBYTE",
        "date32[day]": "DATE",
        "date64[ms]": "DATE",
        "time32[s]": "TIME",
        "time32[ms]": "TIME",
        "time64[us]": "TIME",
        "time64[ns]": "TIME",
    }

    # ------------------------------------------------------------------
    # Identifier handling
    # ------------------------------------------------------------------

    @staticmethod
    def quote_identifier(identifier: str) -> str:
        """
        Safely quote a Redshift identifier.

        Example:
            price → "price"
            user"name → "user""name"
        """

        if not identifier or not identifier.strip():
            raise ValueError("SQL identifier cannot be empty.")

        escaped_identifier = identifier.replace('"', '""')

        return f'"{escaped_identifier}"'

    # ------------------------------------------------------------------
    # Data type mapping
    # ------------------------------------------------------------------

    @classmethod
    def map_data_type(cls, data_type: pa.DataType) -> str:
        """
        Convert one PyArrow data type into a Redshift SQL type.
        """

        type_string = str(data_type)

        # --------------------------------------------------------------
        # Direct mappings
        # --------------------------------------------------------------

        if type_string in cls.TYPE_MAPPING:
            return cls.TYPE_MAPPING[type_string]

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        if pa.types.is_timestamp(data_type):
            if data_type.tz:
                return "TIMESTAMPTZ"

            return "TIMESTAMP"

        # --------------------------------------------------------------
        # Decimal
        # --------------------------------------------------------------

        if pa.types.is_decimal(data_type):
            precision = data_type.precision
            scale = data_type.scale

            if precision > 38:
                raise TypeError(
                    "Decimal precision greater than 38 is not supported "
                    f"for Redshift: {data_type}"
                )

            return f"DECIMAL({precision},{scale})"

        # --------------------------------------------------------------
        # Duration
        # --------------------------------------------------------------

        if pa.types.is_duration(data_type):
            return "BIGINT"

        # --------------------------------------------------------------
        # Dictionary
        # --------------------------------------------------------------

        if pa.types.is_dictionary(data_type):
            return cls.map_data_type(data_type.value_type)

        # --------------------------------------------------------------
        # List / Array
        # --------------------------------------------------------------

        if pa.types.is_list(data_type):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        if pa.types.is_large_list(data_type):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        if pa.types.is_fixed_size_list(data_type):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        # --------------------------------------------------------------
        # Struct
        # --------------------------------------------------------------

        if pa.types.is_struct(data_type):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        # --------------------------------------------------------------
        # Map
        # --------------------------------------------------------------

        if pa.types.is_map(data_type):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        # --------------------------------------------------------------
        # Null
        # --------------------------------------------------------------

        if pa.types.is_null(data_type):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        # --------------------------------------------------------------
        # Unsupported type
        # --------------------------------------------------------------

        raise TypeError("Unsupported PyArrow data type: " f"{data_type}")

    # ------------------------------------------------------------------
    # Schema mapping
    # ------------------------------------------------------------------

    @classmethod
    def map_schema(
        cls,
        schema: pa.Schema,
    ) -> list[RedshiftColumn]:
        """
        Convert a complete PyArrow schema into Redshift columns.
        """

        cls.validate_schema(schema)

        columns: list[RedshiftColumn] = []

        for field in schema:
            columns.append(
                RedshiftColumn(
                    name=field.name,
                    data_type=cls.map_data_type(field.type),
                    nullable=field.nullable,
                )
            )

        return columns

    # ------------------------------------------------------------------
    # Generate column SQL
    # ------------------------------------------------------------------

    @classmethod
    def generate_columns_sql(
        cls,
        schema: pa.Schema,
    ) -> str:
        """
        Generate the column section of a CREATE TABLE statement.
        """

        columns = cls.map_schema(schema)

        return ",\n".join(f"    {column.to_sql()}" for column in columns)

    # ------------------------------------------------------------------
    # Generate CREATE TABLE SQL
    # ------------------------------------------------------------------

    @classmethod
    def generate_create_table_sql(
        cls,
        schema: pa.Schema,
        schema_name: str,
        table_name: str,
        if_not_exists: bool = True,
    ) -> str:
        """
        Generate a complete Redshift CREATE TABLE statement.
        """

        if not schema_name or not schema_name.strip():
            raise ValueError("schema_name cannot be empty.")

        if not table_name or not table_name.strip():
            raise ValueError("table_name cannot be empty.")

        cls.validate_schema(schema)

        quoted_schema = cls.quote_identifier(schema_name)
        quoted_table = cls.quote_identifier(table_name)

        if if_not_exists:
            create_clause = "CREATE TABLE IF NOT EXISTS"
        else:
            create_clause = "CREATE TABLE"

        columns_sql = cls.generate_columns_sql(schema)

        return (
            f"{create_clause} "
            f"{quoted_schema}.{quoted_table} (\n"
            f"{columns_sql}\n"
            ");"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_schema(schema: pa.Schema) -> None:
        """
        Validate the incoming PyArrow schema.
        """

        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be an instance of pyarrow.Schema.")

        if len(schema) == 0:
            raise ValueError("PyArrow schema cannot be empty.")

        seen_names: set[str] = set()

        for field in schema:
            if not field.name or not field.name.strip():
                raise ValueError("PyArrow schema contains an empty column name.")

            if field.name in seen_names:
                raise ValueError(
                    "Duplicate column name found in PyArrow schema: " f"{field.name}"
                )

            seen_names.add(field.name)

            # Make sure every field type can be mapped before
            # generating the CREATE TABLE statement.
            PyArrowSchemaMapper.map_data_type(field.type)
