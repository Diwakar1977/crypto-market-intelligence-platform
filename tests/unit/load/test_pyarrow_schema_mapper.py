from __future__ import annotations

import pyarrow as pa
import pytest

from src.load.pyarrow_schema_mapper import (
    PyArrowSchemaMapper,
    RedshiftColumn,
)

# ------------------------------------------------------------------
# RedshiftColumn
# ------------------------------------------------------------------


def test_redshift_column_to_sql() -> None:
    column = RedshiftColumn(
        name="current_price",
        data_type="DOUBLE PRECISION",
    )

    assert column.to_sql() == '"current_price" DOUBLE PRECISION'


def test_redshift_column_to_sql_not_null() -> None:
    column = RedshiftColumn(
        name="id",
        data_type="BIGINT",
        nullable=False,
    )

    assert column.to_sql() == '"id" BIGINT NOT NULL'


def test_redshift_column_identifier_is_escaped() -> None:
    column = RedshiftColumn(
        name='user"name',
        data_type="VARCHAR(100)",
    )

    assert column.to_sql() == '"user""name" VARCHAR(100)'


# ------------------------------------------------------------------
# Identifier
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        " ",
        "   ",
    ],
)
def test_quote_identifier_rejects_empty_identifier(
    identifier: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="SQL identifier cannot be empty",
    ):
        PyArrowSchemaMapper.quote_identifier(identifier)


def test_quote_identifier() -> None:
    assert PyArrowSchemaMapper.quote_identifier("price") == '"price"'


def test_quote_identifier_escapes_double_quotes() -> None:
    assert PyArrowSchemaMapper.quote_identifier('user"name') == '"user""name"'


# ------------------------------------------------------------------
# Basic Arrow → Redshift mappings
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arrow_type", "expected"),
    [
        (pa.bool_(), "BOOLEAN"),
        (pa.int8(), "SMALLINT"),
        (pa.int16(), "SMALLINT"),
        (pa.int32(), "INTEGER"),
        (pa.int64(), "BIGINT"),
        (pa.uint8(), "SMALLINT"),
        (pa.uint16(), "INTEGER"),
        (pa.uint32(), "BIGINT"),
        (pa.uint64(), "DECIMAL(20,0)"),
        (pa.float32(), "REAL"),
        (pa.float64(), "DOUBLE PRECISION"),
        (pa.string(), "VARCHAR(65535)"),
        (pa.large_string(), "VARCHAR(65535)"),
        (pa.binary(), "VARBYTE"),
        (pa.large_binary(), "VARBYTE"),
        (pa.date32(), "DATE"),
        (pa.date64(), "DATE"),
        (pa.time32("s"), "TIME"),
        (pa.time32("ms"), "TIME"),
        (pa.time64("us"), "TIME"),
        (pa.time64("ns"), "TIME"),
    ],
)
def test_map_data_type(
    arrow_type: pa.DataType,
    expected: str,
) -> None:
    assert PyArrowSchemaMapper.map_data_type(arrow_type) == expected


# ------------------------------------------------------------------
# Timestamp
# ------------------------------------------------------------------


def test_timestamp_without_timezone() -> None:
    data_type = pa.timestamp("us")

    assert PyArrowSchemaMapper.map_data_type(data_type) == "TIMESTAMP"


def test_timestamp_with_timezone() -> None:
    data_type = pa.timestamp(
        "us",
        tz="UTC",
    )

    assert PyArrowSchemaMapper.map_data_type(data_type) == "TIMESTAMPTZ"


# ------------------------------------------------------------------
# Decimal
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("precision", "scale", "expected"),
    [
        (10, 2, "DECIMAL(10,2)"),
        (18, 4, "DECIMAL(18,4)"),
        (38, 10, "DECIMAL(38,10)"),
    ],
)
def test_decimal_mapping(
    precision: int,
    scale: int,
    expected: str,
) -> None:
    data_type = pa.decimal128(
        precision,
        scale,
    )

    assert PyArrowSchemaMapper.map_data_type(data_type) == expected


def test_decimal_precision_above_redshift_limit() -> None:
    data_type = pa.decimal256(39, 10)

    with pytest.raises(
        TypeError,
        match="Decimal precision greater than 38",
    ):
        PyArrowSchemaMapper.map_data_type(data_type)


# ------------------------------------------------------------------
# Duration
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit",
    [
        "s",
        "ms",
        "us",
        "ns",
    ],
)
def test_duration_mapping(unit: str) -> None:
    data_type = pa.duration(unit)

    assert PyArrowSchemaMapper.map_data_type(data_type) == "BIGINT"


# ------------------------------------------------------------------
# Dictionary
# ------------------------------------------------------------------


def test_dictionary_mapping() -> None:
    data_type = pa.dictionary(
        pa.int32(),
        pa.string(),
    )

    assert PyArrowSchemaMapper.map_data_type(data_type) == "VARCHAR(65535)"


# ------------------------------------------------------------------
# List / Array
# ------------------------------------------------------------------


def test_list_mapping() -> None:
    data_type = pa.list_(pa.string())

    assert PyArrowSchemaMapper.map_data_type(data_type) == "VARCHAR(65535)"


def test_large_list_mapping() -> None:
    data_type = pa.large_list(pa.string())

    assert PyArrowSchemaMapper.map_data_type(data_type) == "VARCHAR(65535)"


def test_fixed_size_list_mapping() -> None:
    data_type = pa.list_(
        pa.float64(),
        list_size=3,
    )

    assert PyArrowSchemaMapper.map_data_type(data_type) == "VARCHAR(65535)"


# ------------------------------------------------------------------
# Struct
# ------------------------------------------------------------------


def test_struct_mapping() -> None:
    data_type = pa.struct(
        [
            pa.field(
                "amount",
                pa.float64(),
            ),
            pa.field(
                "currency",
                pa.string(),
            ),
        ]
    )

    assert PyArrowSchemaMapper.map_data_type(data_type) == "VARCHAR(65535)"


# ------------------------------------------------------------------
# Map
# ------------------------------------------------------------------


def test_map_mapping() -> None:
    data_type = pa.map_(
        pa.string(),
        pa.string(),
    )

    assert PyArrowSchemaMapper.map_data_type(data_type) == "VARCHAR(65535)"


# ------------------------------------------------------------------
# Null
# ------------------------------------------------------------------


def test_null_mapping() -> None:
    data_type = pa.null()

    assert PyArrowSchemaMapper.map_data_type(data_type) == "VARCHAR(65535)"


# ------------------------------------------------------------------
# Unsupported type
# ------------------------------------------------------------------


def test_unsupported_type() -> None:
    data_type = pa.decimal256(76, 20)

    with pytest.raises(
        TypeError,
        match="Decimal precision greater than 38",
    ):
        PyArrowSchemaMapper.map_data_type(data_type)


# ------------------------------------------------------------------
# Schema validation
# ------------------------------------------------------------------


def test_validate_schema_success() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("price", pa.float64()),
        ]
    )

    PyArrowSchemaMapper.validate_schema(schema)


def test_validate_schema_rejects_non_schema() -> None:
    with pytest.raises(
        TypeError,
        match="schema must be an instance of pyarrow.Schema",
    ):
        PyArrowSchemaMapper.validate_schema("invalid")


def test_validate_schema_rejects_empty_schema() -> None:
    schema = pa.schema([])

    with pytest.raises(
        ValueError,
        match="PyArrow schema cannot be empty",
    ):
        PyArrowSchemaMapper.validate_schema(schema)


def test_validate_schema_rejects_empty_column_name() -> None:
    schema = pa.schema(
        [
            pa.field(
                "",
                pa.string(),
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty column name",
    ):
        PyArrowSchemaMapper.validate_schema(schema)


def test_validate_schema_rejects_duplicate_column_names() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
            ),
            pa.field(
                "id",
                pa.string(),
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate column name",
    ):
        PyArrowSchemaMapper.validate_schema(schema)


# ------------------------------------------------------------------
# map_schema
# ------------------------------------------------------------------


def test_map_schema() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
            ),
            pa.field(
                "symbol",
                pa.string(),
            ),
            pa.field(
                "current_price",
                pa.float64(),
            ),
            pa.field(
                "last_updated",
                pa.timestamp("us"),
            ),
        ]
    )

    columns = PyArrowSchemaMapper.map_schema(schema)

    assert len(columns) == 4

    assert columns[0] == RedshiftColumn(
        name="id",
        data_type="BIGINT",
        nullable=True,
    )

    assert columns[1] == RedshiftColumn(
        name="symbol",
        data_type="VARCHAR(65535)",
        nullable=True,
    )

    assert columns[2] == RedshiftColumn(
        name="current_price",
        data_type="DOUBLE PRECISION",
        nullable=True,
    )

    assert columns[3] == RedshiftColumn(
        name="last_updated",
        data_type="TIMESTAMP",
        nullable=True,
    )


def test_map_schema_preserves_nullability() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
                nullable=False,
            ),
            pa.field(
                "symbol",
                pa.string(),
                nullable=True,
            ),
        ]
    )

    columns = PyArrowSchemaMapper.map_schema(schema)

    assert columns[0].nullable is False
    assert columns[1].nullable is True


# ------------------------------------------------------------------
# generate_columns_sql
# ------------------------------------------------------------------


def test_generate_columns_sql() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
            ),
            pa.field(
                "symbol",
                pa.string(),
            ),
            pa.field(
                "current_price",
                pa.float64(),
            ),
        ]
    )

    result = PyArrowSchemaMapper.generate_columns_sql(schema)

    expected = (
        '    "id" BIGINT,\n'
        '    "symbol" VARCHAR(65535),\n'
        '    "current_price" DOUBLE PRECISION'
    )

    assert result == expected


def test_generate_columns_sql_with_not_null() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
                nullable=False,
            ),
            pa.field(
                "symbol",
                pa.string(),
                nullable=True,
            ),
        ]
    )

    result = PyArrowSchemaMapper.generate_columns_sql(schema)

    expected = '    "id" BIGINT NOT NULL,\n' '    "symbol" VARCHAR(65535)'

    assert result == expected


# ------------------------------------------------------------------
# CREATE TABLE SQL
# ------------------------------------------------------------------


def test_generate_create_table_sql() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
            ),
            pa.field(
                "symbol",
                pa.string(),
            ),
            pa.field(
                "current_price",
                pa.float64(),
            ),
        ]
    )

    result = PyArrowSchemaMapper.generate_create_table_sql(
        schema=schema,
        schema_name="public",
        table_name="crypto_market",
    )

    expected = (
        'CREATE TABLE IF NOT EXISTS "public"."crypto_market" (\n'
        '    "id" BIGINT,\n'
        '    "symbol" VARCHAR(65535),\n'
        '    "current_price" DOUBLE PRECISION\n'
        ");"
    )

    assert result == expected


def test_generate_create_table_sql_without_if_not_exists() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
            ),
        ]
    )

    result = PyArrowSchemaMapper.generate_create_table_sql(
        schema=schema,
        schema_name="public",
        table_name="crypto_market",
        if_not_exists=False,
    )

    assert result.startswith('CREATE TABLE "public"."crypto_market"')


@pytest.mark.parametrize(
    ("schema_name", "table_name", "message"),
    [
        (
            "",
            "crypto_market",
            "schema_name cannot be empty",
        ),
        (
            "public",
            "",
            "table_name cannot be empty",
        ),
    ],
)
def test_generate_create_table_sql_validation(
    schema_name: str,
    table_name: str,
    message: str,
) -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        PyArrowSchemaMapper.generate_create_table_sql(
            schema=schema,
            schema_name=schema_name,
            table_name=table_name,
        )


# ------------------------------------------------------------------
# Realistic crypto schema
# ------------------------------------------------------------------


def test_crypto_market_schema_mapping() -> None:
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.string(),
            ),
            pa.field(
                "symbol",
                pa.string(),
            ),
            pa.field(
                "name",
                pa.string(),
            ),
            pa.field(
                "image",
                pa.string(),
            ),
            pa.field(
                "current_price",
                pa.float64(),
            ),
            pa.field(
                "market_cap",
                pa.int64(),
            ),
            pa.field(
                "market_cap_rank",
                pa.int64(),
            ),
            pa.field(
                "fully_diluted_valuation",
                pa.int64(),
            ),
            pa.field(
                "total_volume",
                pa.float64(),
            ),
            pa.field(
                "high_24h",
                pa.float64(),
            ),
            pa.field(
                "low_24h",
                pa.float64(),
            ),
            pa.field(
                "price_change_24h",
                pa.float64(),
            ),
            pa.field(
                "price_change_percentage_24h",
                pa.float64(),
            ),
            pa.field(
                "market_cap_change_24h",
                pa.float64(),
            ),
            pa.field(
                "market_cap_change_percentage_24h",
                pa.float64(),
            ),
            pa.field(
                "circulating_supply",
                pa.float64(),
            ),
            pa.field(
                "total_supply",
                pa.float64(),
            ),
            pa.field(
                "max_supply",
                pa.float64(),
            ),
            pa.field(
                "ath",
                pa.float64(),
            ),
            pa.field(
                "ath_change_percentage",
                pa.float64(),
            ),
            pa.field(
                "ath_date",
                pa.timestamp("us"),
            ),
            pa.field(
                "atl",
                pa.float64(),
            ),
            pa.field(
                "atl_change_percentage",
                pa.float64(),
            ),
            pa.field(
                "atl_date",
                pa.timestamp("us"),
            ),
            pa.field(
                "roi",
                pa.struct(
                    [
                        pa.field(
                            "times",
                            pa.float64(),
                        ),
                        pa.field(
                            "currency",
                            pa.string(),
                        ),
                        pa.field(
                            "percentage",
                            pa.float64(),
                        ),
                    ]
                ),
            ),
            pa.field(
                "last_updated",
                pa.timestamp("us"),
            ),
        ]
    )

    columns = PyArrowSchemaMapper.map_schema(schema)

    # There are 26 columns in the crypto_market schema.
    assert len(columns) == 26

    # First column: id
    assert columns[0].data_type == "VARCHAR(65535)"

    # Fifth column: current_price
    assert columns[4].data_type == "DOUBLE PRECISION"

    # Sixth column: market_cap
    assert columns[5].data_type == "BIGINT"

    # Twenty-first column: ath_date
    assert columns[20].data_type == "TIMESTAMP"

    # Twenty-fifth column: roi
    assert columns[24].data_type == "VARCHAR(65535)"

    # Twenty-sixth column: last_updated
    assert columns[25].data_type == "TIMESTAMP"


# ------------------------------------------------------------------
# SQL identifier safety in CREATE TABLE
# ------------------------------------------------------------------


def test_create_table_sql_escapes_identifiers() -> None:
    schema = pa.schema(
        [
            pa.field(
                'user"name',
                pa.string(),
            ),
        ]
    )

    result = PyArrowSchemaMapper.generate_create_table_sql(
        schema=schema,
        schema_name="public",
        table_name='crypto"market',
    )

    assert '"public"."crypto""market"' in result
    assert '"user""name" VARCHAR(65535)' in result
