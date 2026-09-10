from __future__ import annotations

import pytest
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.load.redshift_schema_mapper import (
    RedshiftColumn,
    RedshiftSchemaMapper,
)


def test_redshift_column_to_sql() -> None:
    """Test nullable Redshift column SQL generation."""

    column = RedshiftColumn(
        name="price",
        data_type="DOUBLE PRECISION",
    )

    assert column.to_sql() == '"price" DOUBLE PRECISION'


def test_redshift_column_not_null() -> None:
    """Test NOT NULL Redshift column SQL generation."""

    column = RedshiftColumn(
        name="id",
        data_type="VARCHAR(65535)",
        nullable=False,
    )

    assert column.to_sql() == '"id" VARCHAR(65535) NOT NULL'


def test_map_data_type() -> None:
    """Test supported Spark-to-Redshift type mappings."""

    assert RedshiftSchemaMapper.map_data_type(StringType()) == "VARCHAR(65535)"

    assert RedshiftSchemaMapper.map_data_type(IntegerType()) == "INTEGER"

    assert RedshiftSchemaMapper.map_data_type(LongType()) == "BIGINT"

    assert RedshiftSchemaMapper.map_data_type(FloatType()) == "REAL"

    assert RedshiftSchemaMapper.map_data_type(DoubleType()) == "DOUBLE PRECISION"

    assert RedshiftSchemaMapper.map_data_type(BooleanType()) == "BOOLEAN"

    assert RedshiftSchemaMapper.map_data_type(DateType()) == "DATE"

    assert RedshiftSchemaMapper.map_data_type(TimestampType()) == "TIMESTAMP"


def test_map_complex_data_types() -> None:
    """Test Spark complex type mappings."""

    assert (
        RedshiftSchemaMapper.map_data_type(ArrayType(StringType())) == "VARCHAR(65535)"
    )

    assert (
        RedshiftSchemaMapper.map_data_type(MapType(StringType(), StringType()))
        == "VARCHAR(65535)"
    )

    assert (
        RedshiftSchemaMapper.map_data_type(
            StructType(
                [
                    StructField(
                        "nested",
                        StringType(),
                    )
                ]
            )
        )
        == "VARCHAR(65535)"
    )


def test_map_schema() -> None:
    """Test complete Spark schema mapping."""

    schema = StructType(
        [
            StructField(
                "id",
                StringType(),
                nullable=False,
            ),
            StructField(
                "price",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "market_cap",
                LongType(),
                nullable=True,
            ),
        ]
    )

    columns = RedshiftSchemaMapper.map_schema(schema)

    assert len(columns) == 3

    assert columns[0].name == "id"
    assert columns[0].data_type == "VARCHAR(65535)"
    assert columns[0].nullable is False

    assert columns[1].name == "price"
    assert columns[1].data_type == "DOUBLE PRECISION"
    assert columns[1].nullable is True

    assert columns[2].name == "market_cap"
    assert columns[2].data_type == "BIGINT"
    assert columns[2].nullable is True


def test_quote_identifier() -> None:
    """Test SQL identifier quoting."""

    assert RedshiftSchemaMapper.quote_identifier("crypto_market") == '"crypto_market"'


def test_quote_identifier_escapes_quotes() -> None:
    """Test SQL identifier quote escaping."""

    assert RedshiftSchemaMapper.quote_identifier('coin"name') == '"coin""name"'


def test_quote_identifier_rejects_empty_value() -> None:
    """Test empty SQL identifier validation."""

    with pytest.raises(
        ValueError,
        match="SQL identifier cannot be empty",
    ):
        RedshiftSchemaMapper.quote_identifier("")


def test_generate_columns_sql() -> None:
    """Test Redshift column SQL generation."""

    schema = StructType(
        [
            StructField(
                "id",
                StringType(),
                nullable=False,
            ),
            StructField(
                "price",
                DoubleType(),
            ),
        ]
    )

    result = RedshiftSchemaMapper.generate_columns_sql(schema)

    expected = '  "id" VARCHAR(65535) NOT NULL,\n' '  "price" DOUBLE PRECISION'

    assert result == expected


def test_generate_create_table_sql() -> None:
    """Test complete CREATE TABLE SQL generation."""

    schema = StructType(
        [
            StructField(
                "id",
                StringType(),
                nullable=False,
            ),
            StructField(
                "price",
                DoubleType(),
            ),
            StructField(
                "market_cap",
                LongType(),
            ),
        ]
    )

    result = RedshiftSchemaMapper.generate_create_table_sql(
        schema=schema,
        schema_name="public",
        table_name="crypto_market",
    )

    expected = (
        'CREATE TABLE IF NOT EXISTS "public"."crypto_market" (\n'
        '  "id" VARCHAR(65535) NOT NULL,\n'
        '  "price" DOUBLE PRECISION,\n'
        '  "market_cap" BIGINT\n'
        ");"
    )

    assert result == expected


def test_generate_create_table_without_if_not_exists() -> None:
    """Test CREATE TABLE SQL without IF NOT EXISTS."""

    schema = StructType(
        [
            StructField(
                "id",
                StringType(),
            )
        ]
    )

    result = RedshiftSchemaMapper.generate_create_table_sql(
        schema=schema,
        schema_name="public",
        table_name="crypto_market",
        if_not_exists=False,
    )

    expected = (
        'CREATE TABLE "public"."crypto_market" (\n' '  "id" VARCHAR(65535)\n' ");"
    )

    assert result == expected


def test_validate_schema_success() -> None:
    """Test valid Spark schema validation."""

    schema = StructType(
        [
            StructField("id", StringType()),
            StructField("price", DoubleType()),
            StructField("volume", LongType()),
        ]
    )

    RedshiftSchemaMapper.validate_schema(schema)


def test_validate_schema_rejects_empty_schema() -> None:
    """Test empty schema validation."""

    schema = StructType([])

    with pytest.raises(
        ValueError,
        match="Spark schema contains no fields",
    ):
        RedshiftSchemaMapper.validate_schema(schema)


def test_validate_schema_rejects_duplicate_columns() -> None:
    """Test duplicate column validation."""

    schema = StructType(
        [
            StructField("id", StringType()),
            StructField("ID", LongType()),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate column name detected",
    ):
        RedshiftSchemaMapper.validate_schema(schema)


def test_validate_schema_rejects_invalid_schema() -> None:
    """Test invalid schema type validation."""

    with pytest.raises(
        TypeError,
        match="schema must be a Spark StructType",
    ):
        RedshiftSchemaMapper.validate_schema("invalid")  # type: ignore[arg-type]


def test_generate_create_table_requires_schema_name() -> None:
    """Test Redshift schema name validation."""

    schema = StructType(
        [
            StructField("id", StringType()),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Redshift schema name cannot be empty",
    ):
        RedshiftSchemaMapper.generate_create_table_sql(
            schema=schema,
            schema_name="",
            table_name="crypto_market",
        )


def test_generate_create_table_requires_table_name() -> None:
    """Test Redshift table name validation."""

    schema = StructType(
        [
            StructField("id", StringType()),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Redshift table name cannot be empty",
    ):
        RedshiftSchemaMapper.generate_create_table_sql(
            schema=schema,
            schema_name="public",
            table_name="",
        )


def test_map_data_type_rejects_unsupported_type() -> None:
    """Test unsupported Spark data type validation."""

    from pyspark.sql.types import BinaryType

    with pytest.raises(
        TypeError,
        match="Unsupported Spark data type",
    ):
        RedshiftSchemaMapper.map_data_type(BinaryType())
