from __future__ import annotations

import json
from pathlib import Path

import pytest
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructType,
    TimestampType,
)

from src.schema.schema_loader import SchemaLoader


@pytest.fixture
def schema_loader() -> SchemaLoader:
    """Create a SchemaLoader instance."""

    return SchemaLoader()


def create_schema_file(
    tmp_path: Path,
    schema: object,
) -> Path:
    """Create a temporary JSON schema file."""

    schema_path = tmp_path / "schema.json"

    with schema_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(schema, file)

    return schema_path


def test_load_schema(
    schema_loader: SchemaLoader,
    tmp_path: Path,
) -> None:
    """Load a valid schema into Spark StructType."""

    schema = {
        "id": "StringType",
        "market_cap": "LongType",
        "current_price": "DoubleType",
        "is_active": "BooleanType",
        "last_updated": "TimestampType",
        "roi": "StructType",
        "tags": "ArrayType",
    }

    schema_path = create_schema_file(
        tmp_path,
        schema,
    )

    result = schema_loader.load(schema_path)

    assert isinstance(result, StructType)

    assert result["id"].dataType == StringType()
    assert result["market_cap"].dataType == LongType()
    assert result["current_price"].dataType == DoubleType()
    assert result["is_active"].dataType == BooleanType()
    assert result["last_updated"].dataType == TimestampType()
    assert result["roi"].dataType == StructType()
    assert result["tags"].dataType == ArrayType(
        StringType()
    )


def test_load_schema_preserves_column_order(
    schema_loader: SchemaLoader,
    tmp_path: Path,
) -> None:
    """Preserve column order from the JSON schema."""

    schema = {
        "id": "StringType",
        "symbol": "StringType",
        "current_price": "DoubleType",
        "market_cap": "LongType",
    }

    schema_path = create_schema_file(
        tmp_path,
        schema,
    )

    result = schema_loader.load(schema_path)

    assert result.fieldNames() == [
        "id",
        "symbol",
        "current_price",
        "market_cap",
    ]


def test_load_empty_schema(
    schema_loader: SchemaLoader,
    tmp_path: Path,
) -> None:
    """Load an empty JSON object as an empty StructType."""

    schema_path = create_schema_file(
        tmp_path,
        {},
    )

    result = schema_loader.load(schema_path)

    assert isinstance(result, StructType)
    assert result.fields == []


def test_load_schema_file_not_found(
    schema_loader: SchemaLoader,
    tmp_path: Path,
) -> None:
    """Raise FileNotFoundError when schema file does not exist."""

    schema_path = tmp_path / "missing_schema.json"

    with pytest.raises(
        FileNotFoundError,
        match="Schema file not found",
    ):
        schema_loader.load(schema_path)


@pytest.mark.parametrize(
    "invalid_schema",
    [
        [],
        ["StringType"],
        "invalid",
        123,
        None,
    ],
)
def test_load_invalid_schema_structure(
    schema_loader: SchemaLoader,
    tmp_path: Path,
    invalid_schema: object,
) -> None:
    """Raise TypeError when schema is not a JSON object."""

    schema_path = create_schema_file(
        tmp_path,
        invalid_schema,
    )

    with pytest.raises(
        TypeError,
        match="Schema must contain a JSON object",
    ):
        schema_loader.load(schema_path)


def test_load_non_string_data_type(
    schema_loader: SchemaLoader,
    tmp_path: Path,
) -> None:
    """Raise TypeError when schema type is not a string."""

    schema = {
        "price": 123,
    }

    schema_path = create_schema_file(
        tmp_path,
        schema,
    )

    with pytest.raises(
        TypeError,
        match=(
            "Schema type for column 'price' "
            "must be a string"
        ),
    ):
        schema_loader.load(schema_path)


def test_load_unsupported_spark_type(
    schema_loader: SchemaLoader,
    tmp_path: Path,
) -> None:
    """Raise ValueError for an unsupported Spark type."""

    schema = {
        "price": "DecimalType",
    }

    schema_path = create_schema_file(
        tmp_path,
        schema,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported Spark type "
            "'DecimalType' for column 'price'"
        ),
    ):
        schema_loader.load(schema_path)


def test_all_supported_types_are_mapped(
    schema_loader: SchemaLoader,
) -> None:
    """Ensure all supported Spark types are configured."""

    expected_types = {
        "StringType",
        "LongType",
        "DoubleType",
        "BooleanType",
        "TimestampType",
        "StructType",
        "ArrayType",
    }

    assert set(schema_loader.TYPE_MAPPING) == expected_types