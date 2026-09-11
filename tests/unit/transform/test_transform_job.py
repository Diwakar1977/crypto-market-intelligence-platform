from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.transform.transform_job import (
    TransformJob,
    create_transform_job,
    run_transform_job,
)

# ============================================================
# TEST DATA
# ============================================================

INPUT_PATH = (
    "s3a://crypto-etl-dev/"
    "raw_data/crypto_market/"
    "year=2026/month=09/day=02/"
    "run_time=210000.ndjson"
)

PROCESSED_KEY = (
    "processed_data/crypto_market/" "year=2026/month=09/day=02/" "time=210000/"
)

PROCESSED_PATH = (
    "s3a://crypto-etl-dev/"
    "processed_data/crypto_market/"
    "year=2026/month=09/day=02/"
    "time=210000/"
)

RAW_COLUMNS = [
    "id",
    "symbol",
    "name",
    "current_price",
    "market_cap",
    "market_cap_rank",
    "total_volume",
    "last_updated",
]

FEATURE_COLUMNS = [
    "days_since_ath",
    "days_since_atl",
]

FINAL_COLUMNS = RAW_COLUMNS + FEATURE_COLUMNS


# ============================================================
# CONFIG
# ============================================================


@pytest.fixture
def mock_config() -> dict[str, dict[str, str]]:
    return {
        "application": {
            "raw_dataset": "crypto_market",
        },
        "s3": {
            "bucket": "crypto-etl-dev",
        },
    }


# ============================================================
# TRANSFORM JOB FIXTURE
# ============================================================


@pytest.fixture
def transform_job() -> Any:
    """
    Create a TransformJob with mocked dependencies.

    Any is intentional here.

    TransformJob contains strongly typed production methods such as
    Spark DataFrameReader.json(), while unit tests replace those
    collaborators with MagicMock objects. Mypy otherwise sees the
    production method signatures and rejects MagicMock attributes such
    as return_value, side_effect, assert_called_once_with, etc.
    """

    return TransformJob(
        spark=MagicMock(),
        schema_inferer=MagicMock(),
        schema_manager=MagicMock(),
        data_validator=MagicMock(),
        data_normalizer=MagicMock(),
        feature_engineer=MagicMock(),
        path_builder=MagicMock(),
        parquet_writer=MagicMock(),
    )


# ============================================================
# COMMON PIPELINE
# ============================================================


def configure_pipeline(
    transform_job: Any,
) -> tuple[
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """
    Configure a completely successful TransformJob pipeline.

    Returns:
        raw_df
        typed_df
        normalized_df
        processed_df
    """

    # --------------------------------------------------------
    # RAW DATAFRAME
    # --------------------------------------------------------

    raw_df = MagicMock()
    raw_df.columns = RAW_COLUMNS.copy()
    raw_df.count.return_value = 100

    row = MagicMock()
    row.asDict.return_value = {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "current_price": 100000.0,
        "market_cap": 2000000000000.0,
        "market_cap_rank": 1,
        "total_volume": 50000000000.0,
        "last_updated": "2026-09-02T21:00:00Z",
    }

    raw_df.collect.return_value = [row]

    transform_job.spark.read.json.return_value = raw_df

    # --------------------------------------------------------
    # SCHEMA INFERENCE
    # --------------------------------------------------------

    inferred_schema = {
        "id": "string",
        "symbol": "string",
        "name": "string",
        "current_price": "double",
        "market_cap": "double",
        "market_cap_rank": "integer",
        "total_volume": "double",
        "last_updated": "timestamp",
    }

    transform_job.schema_inferer.infer.return_value = inferred_schema

    # --------------------------------------------------------
    # SCHEMA MANAGEMENT
    # --------------------------------------------------------

    normalized_schema = {
        "id": "string",
        "symbol": "string",
        "name": "string",
        "current_price": "double",
        "market_cap": "double",
        "market_cap_rank": "integer",
        "total_volume": "double",
        "last_updated": "timestamp",
    }

    transform_job.schema_manager.normalize.return_value = normalized_schema

    transform_job.schema_manager.validate.return_value = None

    # --------------------------------------------------------
    # APPLY SCHEMA
    # --------------------------------------------------------

    typed_df = MagicMock()
    typed_df.columns = RAW_COLUMNS.copy()

    transform_job.schema_manager.apply_schema.return_value = typed_df

    # --------------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------------

    validation_result = SimpleNamespace(
        is_valid=True,
        invalid_count=0,
    )

    transform_job.data_validator.validate.return_value = validation_result

    # --------------------------------------------------------
    # DATA NORMALIZATION
    # --------------------------------------------------------

    normalized_df = MagicMock()
    normalized_df.columns = RAW_COLUMNS.copy()

    transform_job.data_normalizer.normalize.return_value = normalized_df

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    processed_df = MagicMock()
    processed_df.columns = FINAL_COLUMNS.copy()

    transform_job.feature_engineer.transform.return_value = processed_df

    # --------------------------------------------------------
    # ORDERED DATAFRAME
    # --------------------------------------------------------

    ordered_df = MagicMock()
    ordered_df.columns = FINAL_COLUMNS.copy()

    processed_df.select.return_value = ordered_df

    # --------------------------------------------------------
    # PATH
    # --------------------------------------------------------

    transform_job.path_builder.build_processed_path.return_value = PROCESSED_KEY

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    transform_job.parquet_writer.write.return_value = None

    return (
        raw_df,
        typed_df,
        normalized_df,
        processed_df,
    )


# ============================================================
# _read_raw_data
# ============================================================


def test_read_raw_data_success(
    transform_job: Any,
) -> None:

    expected_df = MagicMock()

    transform_job.spark.read.json.return_value = expected_df

    result = transform_job._read_raw_data(INPUT_PATH)

    assert result is expected_df

    transform_job.spark.read.json.assert_called_once_with(
        INPUT_PATH,
    )


def test_read_raw_data_empty_path(
    transform_job: Any,
) -> None:

    with pytest.raises(
        ValueError,
        match="Input path cannot be empty",
    ):
        transform_job._read_raw_data("")

    transform_job.spark.read.json.assert_not_called()


# ============================================================
# ORIGINAL JSON COLUMN ORDER
# ============================================================


@patch("src.transform.transform_job.boto3.client")
def test_get_original_json_column_order_success(
    mock_boto_client: MagicMock,
) -> None:

    body = MagicMock()

    body.readline.return_value = b'{"name":"Bitcoin","id":"bitcoin","symbol":"btc"}\n'

    s3 = MagicMock()

    s3.get_object.return_value = {
        "Body": body,
    }

    mock_boto_client.return_value = s3

    result = TransformJob._get_original_json_column_order(
        "s3a://crypto-etl-dev/raw_data/file.ndjson",
    )

    assert result == [
        "name",
        "id",
        "symbol",
    ]

    mock_boto_client.assert_called_once_with("s3")

    s3.get_object.assert_called_once_with(
        Bucket="crypto-etl-dev",
        Key="raw_data/file.ndjson",
    )

    body.readline.assert_called_once()
    body.close.assert_called_once()


def test_get_original_json_column_order_empty_path() -> None:

    with pytest.raises(
        ValueError,
        match="Input path cannot be empty",
    ):
        TransformJob._get_original_json_column_order("")


def test_get_original_json_column_order_invalid_scheme() -> None:

    with pytest.raises(
        ValueError,
        match="Expected s3a:// input path",
    ):
        TransformJob._get_original_json_column_order(
            "s3://bucket/file.ndjson",
        )


def test_get_original_json_column_order_invalid_path() -> None:

    with pytest.raises(
        ValueError,
        match="Invalid S3A path",
    ):
        TransformJob._get_original_json_column_order(
            "s3a://bucket",
        )


@patch("src.transform.transform_job.boto3.client")
def test_get_original_json_column_order_empty_file(
    mock_boto_client: MagicMock,
) -> None:

    body = MagicMock()
    body.readline.return_value = b""

    s3 = MagicMock()

    s3.get_object.return_value = {
        "Body": body,
    }

    mock_boto_client.return_value = s3

    with pytest.raises(
        ValueError,
        match="Raw NDJSON file is empty",
    ):
        TransformJob._get_original_json_column_order(
            "s3a://crypto-etl-dev/raw_data/file.ndjson",
        )

    body.close.assert_called_once()


@patch("src.transform.transform_job.boto3.client")
def test_get_original_json_column_order_invalid_json(
    mock_boto_client: MagicMock,
) -> None:

    body = MagicMock()
    body.readline.return_value = b"invalid-json\n"

    s3 = MagicMock()

    s3.get_object.return_value = {
        "Body": body,
    }

    mock_boto_client.return_value = s3

    with pytest.raises(
        ValueError,
        match="First NDJSON line is not valid JSON",
    ):
        TransformJob._get_original_json_column_order(
            "s3a://crypto-etl-dev/raw_data/file.ndjson",
        )

    body.close.assert_called_once()


@patch("src.transform.transform_job.boto3.client")
def test_get_original_json_column_order_non_object(
    mock_boto_client: MagicMock,
) -> None:

    body = MagicMock()
    body.readline.return_value = b'["bitcoin"]\n'

    s3 = MagicMock()

    s3.get_object.return_value = {
        "Body": body,
    }

    mock_boto_client.return_value = s3

    with pytest.raises(
        TypeError,
        match="First NDJSON record must be a JSON object",
    ):
        TransformJob._get_original_json_column_order(
            "s3a://crypto-etl-dev/raw_data/file.ndjson",
        )

    body.close.assert_called_once()


# ============================================================
# SOURCE COLUMN VALIDATION
# ============================================================


def test_validate_source_columns_success() -> None:

    df = MagicMock()
    df.columns = RAW_COLUMNS.copy()

    TransformJob._validate_source_columns(
        original_json_column_order=RAW_COLUMNS.copy(),
        raw_df=df,
    )


def test_validate_source_columns_failure() -> None:

    df = MagicMock()
    df.columns = RAW_COLUMNS[:-1]

    with pytest.raises(
        ValueError,
        match="Columns found in original NDJSON but missing",
    ):
        TransformJob._validate_source_columns(
            original_json_column_order=RAW_COLUMNS.copy(),
            raw_df=df,
        )


# ============================================================
# NORMALIZED COLUMN VALIDATION
# ============================================================


def test_validate_normalized_columns_success() -> None:

    df = MagicMock()
    df.columns = RAW_COLUMNS.copy()

    TransformJob._validate_normalized_columns(
        raw_column_order=RAW_COLUMNS.copy(),
        normalized_df=df,
    )


def test_validate_normalized_columns_failure() -> None:

    df = MagicMock()
    df.columns = RAW_COLUMNS[:-1]

    with pytest.raises(
        ValueError,
        match="Required RAW columns are missing after normalization",
    ):
        TransformJob._validate_normalized_columns(
            raw_column_order=RAW_COLUMNS.copy(),
            normalized_df=df,
        )


# ============================================================
# ORDER PROCESSED COLUMNS
# ============================================================


def test_order_processed_columns_success(
    transform_job: Any,
) -> None:

    processed_df = MagicMock()

    processed_df.columns = [
        "days_since_ath",
        "id",
        "symbol",
        "name",
        "days_since_atl",
    ]

    ordered_df = MagicMock()

    processed_df.select.return_value = ordered_df

    result = transform_job._order_processed_columns(
        raw_column_order=[
            "id",
            "symbol",
            "name",
        ],
        processed_df=processed_df,
    )

    assert result is ordered_df

    processed_df.select.assert_called_once_with(
        "id",
        "symbol",
        "name",
        "days_since_ath",
        "days_since_atl",
    )


def test_order_processed_columns_empty(
    transform_job: Any,
) -> None:

    processed_df = MagicMock()
    processed_df.columns = []

    with pytest.raises(
        ValueError,
        match="No columns available after transformation",
    ):
        transform_job._order_processed_columns(
            raw_column_order=[],
            processed_df=processed_df,
        )


# ============================================================
# FINAL COLUMN VALIDATION
# ============================================================


def test_validate_final_columns_success() -> None:

    df = MagicMock()

    df.columns = FINAL_COLUMNS.copy()

    TransformJob._validate_final_columns(
        raw_column_order=RAW_COLUMNS.copy(),
        processed_df=df,
    )


def test_validate_final_columns_too_few_columns() -> None:

    df = MagicMock()

    df.columns = RAW_COLUMNS[:-1]

    with pytest.raises(
        ValueError,
        match="Processed DataFrame contains fewer columns",
    ):
        TransformJob._validate_final_columns(
            raw_column_order=RAW_COLUMNS.copy(),
            processed_df=df,
        )


def test_validate_final_columns_wrong_order() -> None:

    df = MagicMock()

    df.columns = [
        "symbol",
        "id",
        "name",
        "current_price",
        "market_cap",
        "market_cap_rank",
        "total_volume",
        "last_updated",
    ] + FEATURE_COLUMNS

    with pytest.raises(
        ValueError,
        match="RAW column order validation failed",
    ):
        TransformJob._validate_final_columns(
            raw_column_order=RAW_COLUMNS.copy(),
            processed_df=df,
        )


def test_validate_final_columns_duplicate_columns() -> None:

    df = MagicMock()

    df.columns = [
        "id",
        "symbol",
        "name",
        "current_price",
        "market_cap",
        "market_cap_rank",
        "total_volume",
        "last_updated",
        "days_since_ath",
        "days_since_ath",
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate columns detected",
    ):
        TransformJob._validate_final_columns(
            raw_column_order=RAW_COLUMNS.copy(),
            processed_df=df,
        )


# ============================================================
# VALIDATION RESULT
# ============================================================


def test_handle_validation_result_success() -> None:

    result = SimpleNamespace(
        is_valid=True,
        invalid_count=0,
    )

    # Positional argument intentionally used here.
    # The production method does not expose "result" as a
    # keyword parameter.
    TransformJob._handle_validation_result(
        cast(Any, result),
    )


def test_handle_validation_result_failure() -> None:

    result = SimpleNamespace(
        is_valid=False,
        invalid_count=10,
    )

    with pytest.raises(
        ValueError,
        match=r"Data validation failed.*10",
    ):
        TransformJob._handle_validation_result(
            cast(Any, result),
        )


# ============================================================
# RUN - EMPTY INPUT
# ============================================================


def test_run_empty_input_path(
    transform_job: Any,
) -> None:

    with pytest.raises(
        ValueError,
        match="Input path cannot be empty",
    ):
        transform_job.run("")

    transform_job.spark.read.json.assert_not_called()


# ============================================================
# RUN - SUCCESS
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_success(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    (
        raw_df,
        typed_df,
        normalized_df,
        processed_df,
    ) = configure_pipeline(transform_job)

    with patch(
        "src.transform.transform_job.CONFIG",
        mock_config,
    ):
        result = transform_job.run(INPUT_PATH)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    assert result == PROCESSED_PATH

    # --------------------------------------------------------
    # ORIGINAL ORDER
    # --------------------------------------------------------

    mock_get_column_order.assert_called_once_with(
        input_path=INPUT_PATH,
    )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    transform_job.spark.read.json.assert_called_once_with(
        INPUT_PATH,
    )

    raw_df.count.assert_called_once()
    raw_df.collect.assert_called_once()

    # --------------------------------------------------------
    # SCHEMA INFERENCE
    # --------------------------------------------------------

    transform_job.schema_inferer.infer.assert_called_once()

    records = transform_job.schema_inferer.infer.call_args.args[0]

    assert len(records) == 1
    assert records[0]["id"] == "bitcoin"

    # --------------------------------------------------------
    # SCHEMA MANAGEMENT
    # --------------------------------------------------------

    transform_job.schema_manager.normalize.assert_called_once_with(
        transform_job.schema_inferer.infer.return_value,
    )

    transform_job.schema_manager.validate.assert_called_once_with(
        transform_job.schema_manager.normalize.return_value,
    )

    # --------------------------------------------------------
    # APPLY SCHEMA
    # --------------------------------------------------------

    transform_job.schema_manager.apply_schema.assert_called_once_with(
        df=raw_df,
        normalized_schema=(transform_job.schema_manager.normalize.return_value),
    )

    # --------------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------------

    transform_job.data_validator.validate.assert_called_once_with(
        typed_df,
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    transform_job.data_normalizer.normalize.assert_called_once_with(
        typed_df,
    )

    assert normalized_df.columns == RAW_COLUMNS

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    transform_job.feature_engineer.transform.assert_called_once_with(
        normalized_df,
    )

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    processed_df.select.assert_called_once_with(
        *FINAL_COLUMNS,
    )

    # --------------------------------------------------------
    # PATH
    # --------------------------------------------------------

    transform_job.path_builder.build_processed_path.assert_called_once_with(
        dataset_name="crypto_market",
    )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    transform_job.parquet_writer.write.assert_called_once_with(
        df=processed_df.select.return_value,
        output_path=PROCESSED_PATH,
        mode="append",
    )

    # --------------------------------------------------------
    # FINAL COLUMNS
    # --------------------------------------------------------

    assert processed_df.select.return_value.columns == FINAL_COLUMNS


# ============================================================
# RUN - EMPTY RAW DATA
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_empty_raw_data(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    raw_df = MagicMock()

    raw_df.columns = RAW_COLUMNS.copy()
    raw_df.count.return_value = 0

    transform_job.spark.read.json.return_value = raw_df

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            ValueError,
            match="Raw input contains no records",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.schema_inferer.infer.assert_not_called()
    transform_job.schema_manager.normalize.assert_not_called()
    transform_job.data_validator.validate.assert_not_called()
    transform_job.parquet_writer.write.assert_not_called()


# ============================================================
# RUN - SOURCE COLUMN MISSING
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_source_column_missing(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    raw_df = MagicMock()

    raw_df.columns = RAW_COLUMNS[:-1]
    raw_df.count.return_value = 100

    transform_job.spark.read.json.return_value = raw_df

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            ValueError,
            match="Columns found in original NDJSON but missing",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.schema_inferer.infer.assert_not_called()
    transform_job.schema_manager.normalize.assert_not_called()
    transform_job.parquet_writer.write.assert_not_called()


# ============================================================
# RUN - SCHEMA INFERENCE FAILURE
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_schema_inference_failure(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    raw_df = MagicMock()

    raw_df.columns = RAW_COLUMNS.copy()
    raw_df.count.return_value = 100

    row = MagicMock()

    row.asDict.return_value = {
        "id": "bitcoin",
    }

    raw_df.collect.return_value = [row]

    transform_job.spark.read.json.return_value = raw_df

    transform_job.schema_inferer.infer.side_effect = ValueError(
        "schema inference failed"
    )

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            ValueError,
            match="schema inference failed",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.schema_inferer.infer.assert_called_once()
    transform_job.schema_manager.normalize.assert_not_called()
    transform_job.schema_manager.validate.assert_not_called()
    transform_job.parquet_writer.write.assert_not_called()


# ============================================================
# RUN - SCHEMA VALIDATION FAILURE
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_schema_validation_failure(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    configure_pipeline(transform_job)

    transform_job.schema_manager.validate.side_effect = ValueError(
        "schema validation failed"
    )

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            ValueError,
            match="schema validation failed",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.schema_inferer.infer.assert_called_once()
    transform_job.schema_manager.normalize.assert_called_once()
    transform_job.schema_manager.validate.assert_called_once()

    transform_job.schema_manager.apply_schema.assert_not_called()
    transform_job.data_validator.validate.assert_not_called()
    transform_job.parquet_writer.write.assert_not_called()


# ============================================================
# RUN - DATA VALIDATION FAILURE
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_data_validation_failure(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    configure_pipeline(transform_job)

    validation_result = SimpleNamespace(
        is_valid=False,
        invalid_count=10,
    )

    transform_job.data_validator.validate.return_value = validation_result

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            ValueError,
            match=r"Data validation failed.*10",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.data_validator.validate.assert_called_once()

    transform_job.data_normalizer.normalize.assert_not_called()
    transform_job.feature_engineer.transform.assert_not_called()
    transform_job.parquet_writer.write.assert_not_called()


# ============================================================
# RUN - NORMALIZATION FAILURE
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_normalization_failure(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    configure_pipeline(transform_job)

    transform_job.data_normalizer.normalize.side_effect = RuntimeError(
        "Normalization failed"
    )

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            RuntimeError,
            match="Normalization failed",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.data_normalizer.normalize.assert_called_once()

    transform_job.feature_engineer.transform.assert_not_called()
    transform_job.parquet_writer.write.assert_not_called()


# ============================================================
# RUN - FEATURE ENGINEERING FAILURE
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_feature_engineering_failure(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    configure_pipeline(transform_job)

    transform_job.feature_engineer.transform.side_effect = RuntimeError(
        "Feature engineering failed"
    )

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            RuntimeError,
            match="Feature engineering failed",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.data_normalizer.normalize.assert_called_once()
    transform_job.feature_engineer.transform.assert_called_once()

    transform_job.parquet_writer.write.assert_not_called()


# ============================================================
# RUN - WRITE FAILURE
# ============================================================


@patch("src.transform.transform_job.TransformJob." "_get_original_json_column_order")
def test_run_write_failure(
    mock_get_column_order: MagicMock,
    transform_job: Any,
    mock_config: dict[str, dict[str, str]],
) -> None:

    mock_get_column_order.return_value = RAW_COLUMNS.copy()

    configure_pipeline(transform_job)

    transform_job.parquet_writer.write.side_effect = RuntimeError(
        "parquet write failed"
    )

    with (
        patch(
            "src.transform.transform_job.CONFIG",
            mock_config,
        ),
        pytest.raises(
            RuntimeError,
            match="parquet write failed",
        ),
    ):
        transform_job.run(INPUT_PATH)

    transform_job.parquet_writer.write.assert_called_once_with(
        df=transform_job.feature_engineer.transform.return_value.select.return_value,
        output_path=PROCESSED_PATH,
        mode="append",
    )


# ============================================================
# FACTORY
# ============================================================


@patch("src.transform.transform_job.ParquetWriter")
@patch("src.transform.transform_job.PathBuilder")
@patch("src.transform.transform_job.FeatureEngineer")
@patch("src.transform.transform_job.DataNormalizer")
@patch("src.transform.transform_job.DataValidator")
@patch("src.transform.transform_job.SchemaManager")
@patch("src.transform.transform_job.SchemaInferer")
def test_create_transform_job(
    mock_schema_inferer: MagicMock,
    mock_schema_manager: MagicMock,
    mock_data_validator: MagicMock,
    mock_data_normalizer: MagicMock,
    mock_feature_engineer: MagicMock,
    mock_path_builder: MagicMock,
    mock_parquet_writer: MagicMock,
) -> None:

    spark = MagicMock()

    result = create_transform_job(
        spark=spark,
    )

    assert isinstance(result, TransformJob)

    assert result.spark is spark

    mock_schema_inferer.assert_called_once()
    mock_schema_manager.assert_called_once()
    mock_data_validator.assert_called_once()
    mock_data_normalizer.assert_called_once()
    mock_feature_engineer.assert_called_once()
    mock_path_builder.assert_called_once()
    mock_parquet_writer.assert_called_once()


# ============================================================
# RUNNER
# ============================================================


@patch("src.transform.transform_job.create_transform_job")
def test_run_transform_job(
    mock_create_transform_job: MagicMock,
) -> None:

    spark = MagicMock()

    job = MagicMock()

    job.run.return_value = PROCESSED_PATH

    mock_create_transform_job.return_value = job

    result = run_transform_job(
        spark=spark,
        input_path=INPUT_PATH,
    )

    assert result == PROCESSED_PATH

    mock_create_transform_job.assert_called_once_with(
        spark=spark,
    )

    job.run.assert_called_once_with(
        input_path=INPUT_PATH,
    )
