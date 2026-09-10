from dataclasses import dataclass
from typing import ClassVar

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructType,
    TimestampType,
)


@dataclass(frozen=True)
class RedshiftColumn:
    """Represent a Redshift table column definition."""

    name: str
    data_type: str
    nullable: bool = True

    def to_sql(self) -> str:
        """Convert the column definition into Redshift SQL."""

        nullability = "" if self.nullable else " NOT NULL"

        return (
            f"{RedshiftSchemaMapper.quote_identifier(self.name)} "
            f"{self.data_type}"
            f"{nullability}"
        )


class RedshiftSchemaMapper:
    """
    Pure Spark-to-Redshift schema mapper.

    Responsibilities:
    - Map Spark data types to Redshift data types.
    - Map Spark
    - StructType to Redshift columns.
    - Generate CREATE TABLE SQL.
    - Validate schema definitions.

    This class does not:
    - connect to Redshift.
    - access AWS.
    - accessS3.
    - execute SQL.
    - perform logging.
    """

    DEFAULT_VARCHAR_LENGTH: ClassVar[int] = 65535

    TYPE_MAPPING: ClassVar[dict[type[DataType], str]] = {
        StringType: f"VARCHAR({DEFAULT_VARCHAR_LENGTH})",
        IntegerType: "INTEGER",
        LongType: "BIGINT",
        FloatType: "REAL",
        DoubleType: "DOUBLE PRECISION",
        BooleanType: "BOOLEAN",
        DateType: "DATE",
        TimestampType: "TIMESTAMP",
    }

    @staticmethod
    def quote_identifier(identifier: str) -> str:
        """Safely quote a Redshift SQL identifier."""

        if not identifier:
            raise ValueError("SQL identifier cannot be empty.")

        escaped_identifier = identifier.replace('"', '""')

        return f'"{escaped_identifier}"'

    @classmethod
    def map_data_type(cls, data_type: DataType) -> str:
        """Map s Spark DataType to a Redshift SQL data type."""

        for spark_type, redshift_type in cls.TYPE_MAPPING.items():
            if isinstance(data_type, spark_type):
                return redshift_type

        if isinstance(data_type, ArrayType):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        if isinstance(data_type, MapType):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        if isinstance(data_type, StructType):
            return f"VARCHAR({cls.DEFAULT_VARCHAR_LENGTH})"

        raise TypeError("Unsupported Spark data type: " f"{data_type.simpleString()}")

    @classmethod
    def map_schema(cls, schema: StructType) -> list[RedshiftColumn]:
        """Map a complete Spark schema to Redshift columns."""

        cls.validate_schema(schema)

        return [
            RedshiftColumn(
                name=field.name,
                data_type=cls.map_data_type(field.dataType),
                nullable=field.nullable,
            )
            for field in schema.fields
        ]

    @classmethod
    def generate_create_table_sql(
        cls,
        schema: StructType,
        schema_name: str,
        table_name: str,
        if_not_exists: bool = True,
    ) -> str:
        """Generate a Redshift CREATE TABLE statement."""

        if not schema_name:
            raise ValueError("Redshift schema name cannot be empty.")

        if not table_name:
            raise ValueError("Redshift table name cannot be empty.")

        columns = cls.map_schema(schema)

        column_sql = ",\n".join(f"  {column.to_sql()}" for column in columns)

        existence_clause = "IF NOT EXISTS " if if_not_exists else ""

        return (
            f"CREATE TABLE {existence_clause}"
            f"{cls.quote_identifier(schema_name)}."
            f"{cls.quote_identifier(table_name)} (\n{column_sql}\n);"
        )

    @classmethod
    def generate_columns_sql(cls, schema: StructType) -> str:
        """Generate only Redshift column definitions."""

        columns = cls.map_schema(schema)

        return ",\n".join(f"  {column.to_sql()}" for column in columns)

    @classmethod
    def validate_schema(cls, schema: StructType) -> None:
        """Validate that a Spark schema can be mapped safely."""

        if not isinstance(schema, StructType):
            raise TypeError("schema must be a Spark StructType.")

        if not schema.fields:
            raise ValueError("Spark schema contains no fields.")

        field_names: set[str] = set()

        for field in schema.fields:
            if not field.name:
                raise ValueError("Schema contains a column with an empty name.")

            normalized_name = field.name.lower()

            if normalized_name in field_names:
                raise ValueError(f"Duplicate column name detected: {field.name}")

            field_names.add(normalized_name)

            cls.map_data_type(field.dataType)
