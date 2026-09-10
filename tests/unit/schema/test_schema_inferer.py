from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest

from src.schema.schema_inferer import SchemaInferer

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def schema_inferer() -> SchemaInferer:
    """Create a SchemaInferer instance."""
    return SchemaInferer()


# =====================================================================
# BASIC SCHEMA INFERENCE
# =====================================================================


def test_infer_basic_schema(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer basic primitive Python types correctly."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "rank": 1,
            "price": 65000.50,
            "active": True,
        }
    ]

    result = schema_inferer.infer(records)

    assert result == {
        "id": "string",
        "rank": "integer",
        "price": "double",
        "active": "boolean",
    }


def test_infer_regular_strings(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer normal strings as string."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
        }
    ]

    result = schema_inferer.infer(records)

    assert result == {
        "id": "string",
        "symbol": "string",
        "name": "string",
    }


# =====================================================================
# TIMESTAMP
# =====================================================================


def test_infer_timestamp(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer ISO-8601 timestamps correctly."""

    records: list[dict[str, Any]] = [
        {
            "ath_date": "2025-01-15T12:30:45.000Z",
            "atl_date": "2013-07-06T00:00:00Z",
            "last_updated": "2026-08-26T05:30:45.123Z",
        }
    ]

    result = schema_inferer.infer(records)

    assert result == {
        "ath_date": "timestamp",
        "atl_date": "timestamp",
        "last_updated": "timestamp",
    }


def test_infer_timestamp_with_timezone_offset(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer timestamps containing timezone offsets."""

    records: list[dict[str, Any]] = [
        {
            "created_at": "2026-08-26T10:30:00+05:30",
        }
    ]

    result = schema_inferer.infer(records)

    assert result["created_at"] == "timestamp"


def test_infer_timestamp_with_space_separator(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer timestamps using a space between date and time."""

    records: list[dict[str, Any]] = [
        {
            "created_at": "2026-08-26 10:30:00",
        }
    ]

    result = schema_inferer.infer(records)

    assert result["created_at"] == "timestamp"


def test_infer_date_only_string_as_timestamp(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer date-only ISO strings as timestamp."""

    records: list[dict[str, Any]] = [
        {
            "created_date": "2026-08-26",
        }
    ]

    result = schema_inferer.infer(records)

    assert result["created_date"] == "timestamp"


def test_is_timestamp_valid_zulu(
    schema_inferer: SchemaInferer,
) -> None:
    """Return True for a valid UTC timestamp."""

    assert (
        schema_inferer._is_timestamp(
            "2026-09-02T20:00:00Z",
        )
        is True
    )


def test_is_timestamp_valid_offset(
    schema_inferer: SchemaInferer,
) -> None:
    """Return True for a timestamp with timezone offset."""

    assert (
        schema_inferer._is_timestamp(
            "2026-09-02T20:00:00+05:30",
        )
        is True
    )


def test_is_timestamp_valid_space_format(
    schema_inferer: SchemaInferer,
) -> None:
    """Return True for a timestamp with space separator."""

    assert (
        schema_inferer._is_timestamp(
            "2026-09-02 20:00:00",
        )
        is True
    )


def test_is_timestamp_valid_date_only(
    schema_inferer: SchemaInferer,
) -> None:
    """Return True for a valid ISO date."""

    assert (
        schema_inferer._is_timestamp(
            "2026-09-02",
        )
        is True
    )


def test_is_timestamp_invalid(
    schema_inferer: SchemaInferer,
) -> None:
    """Return False for a non-timestamp string."""

    assert (
        schema_inferer._is_timestamp(
            "not-a-timestamp",
        )
        is False
    )


def test_is_timestamp_normal_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Return False for a normal string."""

    assert (
        schema_inferer._is_timestamp(
            "bitcoin",
        )
        is False
    )


def test_is_timestamp_empty_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Return False for an empty string."""

    assert schema_inferer._is_timestamp("") is False


def test_is_timestamp_whitespace_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Return False for whitespace-only strings."""

    assert schema_inferer._is_timestamp("   ") is False


def test_is_timestamp_short_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Return False for strings shorter than eight characters."""

    assert schema_inferer._is_timestamp("2026") is False


# =====================================================================
# PYTHON DATETIME / DATE
# =====================================================================


def test_infer_python_datetime(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer Python datetime values as timestamp."""

    value = datetime(
        2026,
        9,
        2,
        20,
        0,
        0,
        tzinfo=timezone.utc,
    )

    records: list[dict[str, Any]] = [
        {
            "created_at": value,
        }
    ]

    result = schema_inferer.infer(records)

    assert result["created_at"] == "timestamp"


def test_infer_python_date(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer Python date values as timestamp."""

    value = date(
        2026,
        9,
        2,
    )

    records: list[dict[str, Any]] = [
        {
            "created_date": value,
        }
    ]

    result = schema_inferer.infer(records)

    assert result["created_date"] == "timestamp"


# =====================================================================
# NUMERIC TYPES
# =====================================================================


def test_infer_all_integers(
    schema_inferer: SchemaInferer,
) -> None:
    """Keep integer when all values are integers."""

    records: list[dict[str, Any]] = [
        {"rank": 1},
        {"rank": 2},
        {"rank": 3},
    ]

    result = schema_inferer.infer(records)

    assert result["rank"] == "integer"


def test_infer_all_doubles(
    schema_inferer: SchemaInferer,
) -> None:
    """Keep double when all values are floating-point values."""

    records: list[dict[str, Any]] = [
        {"price": 100.10},
        {"price": 200.20},
        {"price": 300.30},
    ]

    result = schema_inferer.infer(records)

    assert result["price"] == "double"


def test_infer_integer_then_double(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve integer followed by double to double."""

    records: list[dict[str, Any]] = [
        {"price": 100},
        {"price": 100.50},
    ]

    result = schema_inferer.infer(records)

    assert result["price"] == "double"


def test_infer_double_then_integer(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve double followed by integer to double."""

    records: list[dict[str, Any]] = [
        {"price": 100.50},
        {"price": 100},
    ]

    result = schema_inferer.infer(records)

    assert result["price"] == "double"


# =====================================================================
# NONE HANDLING
# =====================================================================


def test_none_then_double(
    schema_inferer: SchemaInferer,
) -> None:
    """None must not determine datatype."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "price": None,
            "rank": 1,
        },
        {
            "id": "ethereum",
            "price": 3000.50,
            "rank": 2,
        },
    ]

    result = schema_inferer.infer(records)

    assert result == {
        "id": "string",
        "price": "double",
        "rank": "integer",
    }


def test_none_then_integer(
    schema_inferer: SchemaInferer,
) -> None:
    """A later integer determines the type after None."""

    records: list[dict[str, Any]] = [
        {"rank": None},
        {"rank": 10},
    ]

    result = schema_inferer.infer(records)

    assert result["rank"] == "integer"


def test_none_then_string(
    schema_inferer: SchemaInferer,
) -> None:
    """A later string determines the type after None."""

    records: list[dict[str, Any]] = [
        {"name": None},
        {"name": "Bitcoin"},
    ]

    result = schema_inferer.infer(records)

    assert result["name"] == "string"


def test_none_then_timestamp(
    schema_inferer: SchemaInferer,
) -> None:
    """A later timestamp determines the type after None."""

    records: list[dict[str, Any]] = [
        {"last_updated": None},
        {"last_updated": "2026-09-02T20:00:00Z"},
    ]

    result = schema_inferer.infer(records)

    assert result["last_updated"] == "timestamp"


def test_none_then_array(
    schema_inferer: SchemaInferer,
) -> None:
    """A later array determines the type after None."""

    records: list[dict[str, Any]] = [
        {"tags": None},
        {"tags": ["btc", "eth"]},
    ]

    result = schema_inferer.infer(records)

    assert result["tags"] == "array"


def test_none_then_object(
    schema_inferer: SchemaInferer,
) -> None:
    """A later object determines the type after None."""

    records: list[dict[str, Any]] = [
        {"roi": None},
        {
            "roi": {
                "times": 2.5,
            }
        },
    ]

    result = schema_inferer.infer(records)

    assert result["roi"] == "object"


def test_all_none_column_defaults_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """A column containing only None values defaults to string."""

    records: list[dict[str, Any]] = [
        {"image": None},
        {"image": None},
        {"image": None},
    ]

    result = schema_inferer.infer(records)

    assert result["image"] == "string"


def test_none_after_real_value_is_ignored(
    schema_inferer: SchemaInferer,
) -> None:
    """None after a real value must not change datatype."""

    records: list[dict[str, Any]] = [
        {"price": 65000.50},
        {"price": None},
        {"price": 70000.25},
    ]

    result = schema_inferer.infer(records)

    assert result["price"] == "double"


def test_multiple_none_values_before_real_value(
    schema_inferer: SchemaInferer,
) -> None:
    """Multiple None values before a real value are ignored."""

    records: list[dict[str, Any]] = [
        {"price": None},
        {"price": None},
        {"price": 65000.50},
        {"price": None},
    ]

    result = schema_inferer.infer(records)

    assert result["price"] == "double"


# =====================================================================
# ARRAY AND OBJECT
# =====================================================================


def test_infer_array(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer list values as array."""

    records: list[dict[str, Any]] = [
        {
            "tags": ["crypto", "bitcoin"],
        }
    ]

    result = schema_inferer.infer(records)

    assert result["tags"] == "array"


def test_infer_object(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer dictionary values as object."""

    records: list[dict[str, Any]] = [
        {
            "roi": {
                "times": 2.5,
                "currency": "btc",
            },
        }
    ]

    result = schema_inferer.infer(records)

    assert result["roi"] == "object"


# =====================================================================
# EMPTY INPUT / INVALID RECORDS
# =====================================================================


def test_infer_empty_records(
    schema_inferer: SchemaInferer,
) -> None:
    """Raise ValueError when records are empty."""

    with pytest.raises(
        ValueError,
        match="Cannot infer schema from empty records.",
    ):
        schema_inferer.infer([])


def test_infer_invalid_record_type(
    schema_inferer: SchemaInferer,
) -> None:
    """Raise TypeError when a record is not a dictionary."""

    records: list[Any] = [
        {
            "id": "bitcoin",
        },
        ["invalid"],
    ]

    with pytest.raises(
        TypeError,
        match="Every record must be a dictionary",
    ):
        schema_inferer.infer(records)


def test_infer_invalid_first_record(
    schema_inferer: SchemaInferer,
) -> None:
    """Raise TypeError when the first record is invalid."""

    records: list[Any] = [
        ["invalid"],
    ]

    with pytest.raises(
        TypeError,
        match="Invalid record at position 1",
    ):
        schema_inferer.infer(records)


# =====================================================================
# CONFLICT RESOLUTION
# =====================================================================


def test_string_and_integer_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve incompatible string and integer values to string."""

    records: list[dict[str, Any]] = [
        {"value": "bitcoin"},
        {"value": 100},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_integer_and_string_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve integer followed by string to string."""

    records: list[dict[str, Any]] = [
        {"value": 100},
        {"value": "bitcoin"},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_timestamp_and_string_resolve_to_timestamp(
    schema_inferer: SchemaInferer,
) -> None:
    """Timestamp followed by normal string remains timestamp."""

    records: list[dict[str, Any]] = [
        {"date": "2025-01-15T12:30:45Z"},
        {"date": "unknown"},
    ]

    result = schema_inferer.infer(records)

    assert result["date"] == "timestamp"


def test_string_and_timestamp_resolve_to_timestamp(
    schema_inferer: SchemaInferer,
) -> None:
    """Normal string followed by timestamp resolves to timestamp."""

    records: list[dict[str, Any]] = [
        {"date": "unknown"},
        {"date": "2025-01-15T12:30:45Z"},
    ]

    result = schema_inferer.infer(records)

    assert result["date"] == "timestamp"


def test_boolean_and_integer_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve boolean and integer conflict to string."""

    records: list[dict[str, Any]] = [
        {"value": True},
        {"value": 1},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_integer_and_boolean_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve integer and boolean conflict to string."""

    records: list[dict[str, Any]] = [
        {"value": 1},
        {"value": True},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_array_and_object_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve array and object conflict to string."""

    records: list[dict[str, Any]] = [
        {"value": ["bitcoin"]},
        {"value": {"name": "bitcoin"}},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_string_and_boolean_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve string and boolean conflict to string."""

    records: list[dict[str, Any]] = [
        {"value": "true"},
        {"value": True},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_double_and_string_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve double and string conflict to string."""

    records: list[dict[str, Any]] = [
        {"value": 100.5},
        {"value": "bitcoin"},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_array_and_string_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve array and string conflict to string."""

    records: list[dict[str, Any]] = [
        {"value": ["bitcoin"]},
        {"value": "bitcoin"},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


def test_object_and_string_resolve_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Resolve object and string conflict to string."""

    records: list[dict[str, Any]] = [
        {"value": {"name": "bitcoin"}},
        {"value": "bitcoin"},
    ]

    result = schema_inferer.infer(records)

    assert result["value"] == "string"


# =====================================================================
# COLUMN ORDER
# =====================================================================


def test_preserve_source_column_order(
    schema_inferer: SchemaInferer,
) -> None:
    """Preserve first-seen source column order."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 65000.0,
            "market_cap": 1000000.0,
        }
    ]

    result = schema_inferer.infer(records)

    assert list(result.keys()) == [
        "id",
        "symbol",
        "name",
        "current_price",
        "market_cap",
    ]


def test_preserve_column_order_with_initial_none(
    schema_inferer: SchemaInferer,
) -> None:
    """Preserve position of initially None column."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "price": None,
            "rank": 1,
        },
        {
            "price": 65000.0,
        },
    ]

    result = schema_inferer.infer(records)

    assert list(result.keys()) == [
        "id",
        "price",
        "rank",
    ]

    assert result["price"] == "double"


def test_preserve_column_order_across_multiple_records(
    schema_inferer: SchemaInferer,
) -> None:
    """Preserve first-seen order across multiple records."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "symbol": "btc",
        },
        {
            "name": "Bitcoin",
            "price": 65000.0,
        },
        {
            "market_cap": 1000000.0,
            "rank": 1,
        },
    ]

    result = schema_inferer.infer(records)

    assert list(result.keys()) == [
        "id",
        "symbol",
        "name",
        "price",
        "market_cap",
        "rank",
    ]


# =====================================================================
# ALL COLUMNS
# =====================================================================


def test_schema_contains_all_columns(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer all columns across multiple records."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "price": 65000.0,
        },
        {
            "id": "ethereum",
            "price": 3500.0,
            "rank": 2,
        },
    ]

    result = schema_inferer.infer(records)

    assert result == {
        "id": "string",
        "price": "double",
        "rank": "integer",
    }


def test_infer_crypto_market_columns(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer representative CoinGecko crypto market columns."""

    records: list[dict[str, Any]] = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 100000.0,
            "market_cap": 2000000000000.0,
            "market_cap_rank": 1,
            "total_volume": 50000000000.0,
            "high_24h": 101000.0,
            "low_24h": 99000.0,
            "ath": 126000.0,
            "ath_change_percentage": -20.0,
            "ath_date": "2025-01-15T12:30:45Z",
            "atl": 67.81,
            "atl_change_percentage": 147000.0,
            "atl_date": "2013-07-06T00:00:00Z",
            "circulating_supply": 19000000.0,
            "max_supply": 21000000.0,
            "last_updated": "2026-09-02T20:00:00Z",
        }
    ]

    result = schema_inferer.infer(records)

    assert result == {
        "id": "string",
        "symbol": "string",
        "name": "string",
        "current_price": "double",
        "market_cap": "double",
        "market_cap_rank": "integer",
        "total_volume": "double",
        "high_24h": "double",
        "low_24h": "double",
        "ath": "double",
        "ath_change_percentage": "double",
        "ath_date": "timestamp",
        "atl": "double",
        "atl_change_percentage": "double",
        "atl_date": "timestamp",
        "circulating_supply": "double",
        "max_supply": "double",
        "last_updated": "timestamp",
    }


# =====================================================================
# DIRECT _INFER_TYPE TESTS
# =====================================================================


def test_infer_type_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer string value."""

    assert schema_inferer._infer_type("bitcoin") == "string"


def test_infer_type_integer(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer integer value."""

    assert schema_inferer._infer_type(100) == "integer"


def test_infer_type_float(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer float value."""

    assert schema_inferer._infer_type(100.50) == "double"


def test_infer_type_boolean(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer boolean value."""

    assert schema_inferer._infer_type(True) == "boolean"


def test_infer_type_array(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer list value."""

    assert schema_inferer._infer_type(["btc", "eth"]) == "array"


def test_infer_type_object(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer dictionary value."""

    assert schema_inferer._infer_type({"symbol": "btc"}) == "object"


def test_infer_type_datetime(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer Python datetime."""

    value = datetime(
        2026,
        9,
        2,
        20,
        0,
        tzinfo=timezone.utc,
    )

    assert schema_inferer._infer_type(value) == "timestamp"


def test_infer_type_date(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer Python date."""

    value = date(
        2026,
        9,
        2,
    )

    assert schema_inferer._infer_type(value) == "timestamp"


def test_infer_type_timestamp_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Infer ISO timestamp string."""

    value = "2026-09-02T20:00:00Z"

    assert schema_inferer._infer_type(value) == "timestamp"


def test_infer_type_unknown_value_defaults_to_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Unknown Python types should default to string."""

    class CustomObject:
        pass

    value = CustomObject()

    assert schema_inferer._infer_type(value) == "string"


# =====================================================================
# DIRECT _RESOLVE_TYPE TESTS
# =====================================================================


def test_resolve_same_type(
    schema_inferer: SchemaInferer,
) -> None:
    """Same types remain unchanged."""

    assert (
        schema_inferer._resolve_type(
            "string",
            "string",
        )
        == "string"
    )


def test_resolve_integer_integer(
    schema_inferer: SchemaInferer,
) -> None:
    """Integer and integer resolve to integer."""

    assert (
        schema_inferer._resolve_type(
            "integer",
            "integer",
        )
        == "integer"
    )


def test_resolve_double_double(
    schema_inferer: SchemaInferer,
) -> None:
    """Double and double resolve to double."""

    assert (
        schema_inferer._resolve_type(
            "double",
            "double",
        )
        == "double"
    )


def test_resolve_boolean_boolean(
    schema_inferer: SchemaInferer,
) -> None:
    """Boolean and boolean resolve to boolean."""

    assert (
        schema_inferer._resolve_type(
            "boolean",
            "boolean",
        )
        == "boolean"
    )


def test_resolve_integer_double(
    schema_inferer: SchemaInferer,
) -> None:
    """Integer and double resolve to double."""

    assert (
        schema_inferer._resolve_type(
            "integer",
            "double",
        )
        == "double"
    )


def test_resolve_double_integer(
    schema_inferer: SchemaInferer,
) -> None:
    """Double and integer resolve to double."""

    assert (
        schema_inferer._resolve_type(
            "double",
            "integer",
        )
        == "double"
    )


def test_resolve_string_integer(
    schema_inferer: SchemaInferer,
) -> None:
    """String and integer resolve to string."""

    assert (
        schema_inferer._resolve_type(
            "string",
            "integer",
        )
        == "string"
    )


def test_resolve_integer_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Integer and string resolve to string."""

    assert (
        schema_inferer._resolve_type(
            "integer",
            "string",
        )
        == "string"
    )


def test_resolve_boolean_integer(
    schema_inferer: SchemaInferer,
) -> None:
    """Boolean and integer resolve to string."""

    assert (
        schema_inferer._resolve_type(
            "boolean",
            "integer",
        )
        == "string"
    )


def test_resolve_integer_boolean(
    schema_inferer: SchemaInferer,
) -> None:
    """Integer and boolean resolve to string."""

    assert (
        schema_inferer._resolve_type(
            "integer",
            "boolean",
        )
        == "string"
    )


def test_resolve_array_object(
    schema_inferer: SchemaInferer,
) -> None:
    """Array and object resolve to string."""

    assert (
        schema_inferer._resolve_type(
            "array",
            "object",
        )
        == "string"
    )


def test_resolve_object_array(
    schema_inferer: SchemaInferer,
) -> None:
    """Object and array resolve to string."""

    assert (
        schema_inferer._resolve_type(
            "object",
            "array",
        )
        == "string"
    )


def test_resolve_timestamp_string(
    schema_inferer: SchemaInferer,
) -> None:
    """Timestamp and string resolve to timestamp."""

    assert (
        schema_inferer._resolve_type(
            "timestamp",
            "string",
        )
        == "timestamp"
    )


def test_resolve_string_timestamp(
    schema_inferer: SchemaInferer,
) -> None:
    """String and timestamp resolve to timestamp."""

    assert (
        schema_inferer._resolve_type(
            "string",
            "timestamp",
        )
        == "timestamp"
    )


# =====================================================================
# TYPE MAP
# =====================================================================


def test_type_map_contains_expected_types() -> None:
    """Verify all primitive Python type mappings."""

    assert SchemaInferer.TYPE_MAP == {
        bool: "boolean",
        int: "integer",
        float: "double",
        str: "string",
        datetime: "timestamp",
        date: "timestamp",
    }


def test_numeric_types_contains_integer() -> None:
    """Verify integer is treated as numeric."""

    assert "integer" in SchemaInferer.NUMERIC_TYPES


def test_numeric_types_contains_double() -> None:
    """Verify double is treated as numeric."""

    assert "double" in SchemaInferer.NUMERIC_TYPES
