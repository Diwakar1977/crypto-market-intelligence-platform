from __future__ import annotations

from collections.abc import Generator

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType

from src.schema.schema_manager import SchemaManager

# =====================================================================
# FIXTURE
# =====================================================================


@pytest.fixture
def schema_manager() -> SchemaManager:
    """Create a SchemaManager instance."""
    return SchemaManager()


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Create a SparkSession for schema tests."""

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("schema-manager-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield spark

    spark.stop()


# =====================================================================
# NORMALIZE
# =====================================================================


def test_normalize_schema(
    schema_manager: SchemaManager,
) -> None:
    """Normalize inferred schema to Spark data types."""

    schema = {
        "id": "string",
        "market_cap": "integer",
        "current_price": "double",
        "is_active": "boolean",
        "last_updated": "timestamp",
        "roi": "object",
        "tags": "array",
    }

    result = schema_manager.normalize(schema)

    assert result == {
        "id": "StringType",
        "market_cap": "LongType",
        "current_price": "DoubleType",
        "is_active": "BooleanType",
        "last_updated": "TimestampType",
        "roi": "StructType",
        "tags": "ArrayType",
    }


def test_normalize_unknown_type_to_string_type(
    schema_manager: SchemaManager,
) -> None:
    """Normalize unknown types to StringType."""

    schema = {
        "unknown_column": "unknown",
    }

    result = schema_manager.normalize(schema)

    assert result == {
        "unknown_column": "StringType",
    }


def test_normalize_empty_schema(
    schema_manager: SchemaManager,
) -> None:
    """Raise ValueError when normalizing an empty schema."""

    with pytest.raises(
        ValueError,
        match="Schema cannot be empty.",
    ):
        schema_manager.normalize({})


def test_normalize_preserves_column_names(
    schema_manager: SchemaManager,
) -> None:
    """Ensure normalization does not change column names."""

    schema = {
        "current_price": "double",
        "market_cap_rank": "integer",
        "last_updated": "timestamp",
    }

    result = schema_manager.normalize(schema)

    assert set(result.keys()) == {
        "current_price",
        "market_cap_rank",
        "last_updated",
    }


def test_normalize_preserves_column_order(
    schema_manager: SchemaManager,
) -> None:
    """Ensure normalization preserves input column order."""

    schema = {
        "id": "string",
        "symbol": "string",
        "current_price": "double",
        "market_cap": "integer",
    }

    result = schema_manager.normalize(schema)

    assert list(result.keys()) == [
        "id",
        "symbol",
        "current_price",
        "market_cap",
    ]


def test_normalize_all_supported_types(
    schema_manager: SchemaManager,
) -> None:
    """Normalize every supported logical type."""

    schema = {
        "string_col": "string",
        "integer_col": "integer",
        "double_col": "double",
        "boolean_col": "boolean",
        "timestamp_col": "timestamp",
        "object_col": "object",
        "array_col": "array",
    }

    result = schema_manager.normalize(schema)

    assert result == {
        "string_col": "StringType",
        "integer_col": "LongType",
        "double_col": "DoubleType",
        "boolean_col": "BooleanType",
        "timestamp_col": "TimestampType",
        "object_col": "StructType",
        "array_col": "ArrayType",
    }


# =====================================================================
# VALIDATE
# =====================================================================


def test_validate_valid_schema(
    schema_manager: SchemaManager,
) -> None:
    """Validate a schema containing supported Spark data types."""

    schema = {
        "id": "StringType",
        "market_cap": "LongType",
        "current_price": "DoubleType",
        "is_active": "BooleanType",
        "last_updated": "TimestampType",
        "roi": "StructType",
        "tags": "ArrayType",
    }

    schema_manager.validate(schema)


def test_validate_empty_schema(
    schema_manager: SchemaManager,
) -> None:
    """Raise ValueError when validating an empty schema."""

    with pytest.raises(
        ValueError,
        match="Schema cannot be empty.",
    ):
        schema_manager.validate({})


def test_validate_unsupported_type(
    schema_manager: SchemaManager,
) -> None:
    """Raise ValueError for unsupported Spark type."""

    schema = {
        "price": "DecimalType",
    }

    with pytest.raises(
        ValueError,
        match="Unsupported type 'DecimalType' for column 'price'",
    ):
        schema_manager.validate(schema)


def test_validate_multiple_columns(
    schema_manager: SchemaManager,
) -> None:
    """Validate multiple columns with supported data types."""

    schema = {
        "id": "StringType",
        "symbol": "StringType",
        "market_cap": "LongType",
        "price": "DoubleType",
        "last_updated": "TimestampType",
    }

    schema_manager.validate(schema)


def test_validate_each_supported_type(
    schema_manager: SchemaManager,
) -> None:
    """Validate every supported Spark type individually."""

    supported_types = [
        "StringType",
        "LongType",
        "DoubleType",
        "BooleanType",
        "TimestampType",
        "StructType",
        "ArrayType",
    ]

    for spark_type in supported_types:
        schema_manager.validate(
            {
                "test_column": spark_type,
            }
        )


# =====================================================================
# SPARK DATA TYPE
# =====================================================================


def test_spark_data_type_string(
    schema_manager: SchemaManager,
) -> None:
    """Convert StringType name to actual Spark DataType."""

    result = schema_manager._spark_data_type("StringType")

    assert result.typeName() == "string"


def test_spark_data_type_long(
    schema_manager: SchemaManager,
) -> None:
    """Convert LongType name to actual Spark DataType."""

    result = schema_manager._spark_data_type("LongType")

    assert result.typeName() == "long"


def test_spark_data_type_double(
    schema_manager: SchemaManager,
) -> None:
    """Convert DoubleType name to actual Spark DataType."""

    result = schema_manager._spark_data_type("DoubleType")

    assert result.typeName() == "double"


def test_spark_data_type_boolean(
    schema_manager: SchemaManager,
) -> None:
    """Convert BooleanType name to actual Spark DataType."""

    result = schema_manager._spark_data_type("BooleanType")

    assert result.typeName() == "boolean"


def test_spark_data_type_timestamp(
    schema_manager: SchemaManager,
) -> None:
    """Convert TimestampType name to actual Spark DataType."""

    result = schema_manager._spark_data_type("TimestampType")

    assert result.typeName() == "timestamp"


def test_spark_data_type_struct(
    schema_manager: SchemaManager,
) -> None:
    """Convert StructType name to actual Spark DataType."""

    result = schema_manager._spark_data_type("StructType")

    assert result.typeName() == "struct"


def test_spark_data_type_array(
    schema_manager: SchemaManager,
) -> None:
    """Convert ArrayType name to actual Spark DataType."""

    result = schema_manager._spark_data_type("ArrayType")

    assert result.typeName() == "array"


def test_spark_data_type_unsupported(
    schema_manager: SchemaManager,
) -> None:
    """Raise ValueError for unsupported Spark type."""

    with pytest.raises(
        ValueError,
        match="Unsupported Spark type: DecimalType",
    ):
        schema_manager._spark_data_type("DecimalType")


# =====================================================================
# APPLY SCHEMA
# =====================================================================


def test_apply_schema_string_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Apply StringType to a column."""

    data = [
        (123, "bitcoin"),
    ]

    df = spark.createDataFrame(
        data,
        [
            "rank",
            "id",
        ],
    )

    normalized_schema = {
        "rank": "LongType",
        "id": "StringType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.schema["id"].dataType.typeName() == "string"

    assert result.columns == [
        "rank",
        "id",
    ]

    assert result.collect()[0]["id"] == "bitcoin"


def test_apply_schema_long_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Apply LongType to a column."""

    data = [
        ("100",),
    ]

    df = spark.createDataFrame(
        data,
        ["market_cap_rank"],
    )

    normalized_schema = {
        "market_cap_rank": "LongType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.schema["market_cap_rank"].dataType.typeName() == "long"

    assert result.collect()[0]["market_cap_rank"] == 100


def test_apply_schema_double_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Apply DoubleType to a column."""

    data = [
        ("123.45",),
    ]

    df = spark.createDataFrame(
        data,
        ["current_price"],
    )

    normalized_schema = {
        "current_price": "DoubleType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.schema["current_price"].dataType.typeName() == "double"

    assert result.collect()[0]["current_price"] == pytest.approx(123.45)


def test_apply_schema_boolean_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Apply BooleanType to a column."""

    data = [
        ("true",),
    ]

    df = spark.createDataFrame(
        data,
        ["is_active"],
    )

    normalized_schema = {
        "is_active": "BooleanType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.schema["is_active"].dataType.typeName() == "boolean"

    assert result.collect()[0]["is_active"] is True


def test_apply_schema_timestamp_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Apply TimestampType to a timestamp column."""

    data = [
        ("2026-09-04 09:00:00",),
    ]

    df = spark.createDataFrame(
        data,
        ["last_updated"],
    )

    normalized_schema = {
        "last_updated": "TimestampType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.schema["last_updated"].dataType.typeName() == "timestamp"

    assert str(result.collect()[0]["last_updated"]) == "2026-09-04 09:00:00"


def test_apply_schema_keeps_struct_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Keep Spark-inferred StructType unchanged."""

    data = [
        (
            "bitcoin",
            Row(source="coingecko"),
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "id",
            "metadata",
        ],
    )

    original_struct_type = df.schema["metadata"].dataType

    assert isinstance(
        original_struct_type,
        StructType,
    )

    normalized_schema = {
        "id": "StringType",
        "metadata": "StructType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert isinstance(
        result.schema["metadata"].dataType,
        StructType,
    )

    assert result.columns == [
        "id",
        "metadata",
    ]

    row = result.collect()[0]

    assert row["id"] == "bitcoin"
    assert row["metadata"]["source"] == "coingecko"


def test_apply_schema_keeps_array_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Keep Spark-inferred ArrayType unchanged."""

    data = [
        (
            "bitcoin",
            ["crypto", "coin"],
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "id",
            "tags",
        ],
    )

    assert df.schema["tags"].dataType.typeName() == "array"

    normalized_schema = {
        "id": "StringType",
        "tags": "ArrayType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.schema["tags"].dataType.typeName() == "array"

    assert result.columns == [
        "id",
        "tags",
    ]

    row = result.collect()[0]

    assert row["id"] == "bitcoin"
    assert row["tags"] == ["crypto", "coin"]


def test_apply_schema_preserves_original_column_order(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Preserve the exact original DataFrame column order."""

    data = [
        (
            "bitcoin",
            "123.45",
            "1000000",
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "id",
            "current_price",
            "market_cap",
        ],
    )

    normalized_schema = {
        "market_cap": "LongType",
        "current_price": "DoubleType",
        "id": "StringType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.columns == [
        "id",
        "current_price",
        "market_cap",
    ]


def test_apply_schema_ignores_column_not_in_schema(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Leave columns unchanged when they are absent from managed schema."""

    data = [
        (
            "bitcoin",
            "extra-value",
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "id",
            "extra_column",
        ],
    )

    original_type = df.schema["extra_column"].dataType.typeName()

    normalized_schema = {
        "id": "StringType",
    }

    result = schema_manager.apply_schema(
        df,
        normalized_schema,
    )

    assert result.columns == [
        "id",
        "extra_column",
    ]

    assert result.schema["extra_column"].dataType.typeName() == original_type


def test_apply_schema_empty_schema(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Raise ValueError when applying an empty schema."""

    df = spark.createDataFrame(
        [
            ("bitcoin",),
        ],
        ["id"],
    )

    with pytest.raises(
        ValueError,
        match="Cannot apply an empty schema.",
    ):
        schema_manager.apply_schema(
            df,
            {},
        )


def test_apply_schema_unsupported_type(
    schema_manager: SchemaManager,
    spark: SparkSession,
) -> None:
    """Raise ValueError for an unsupported managed type."""

    df = spark.createDataFrame(
        [
            ("bitcoin",),
        ],
        ["id"],
    )

    normalized_schema = {
        "id": "DecimalType",
    }

    with pytest.raises(
        ValueError,
        match="Unsupported managed type 'DecimalType' for column 'id'",
    ):
        schema_manager.apply_schema(
            df,
            normalized_schema,
        )
