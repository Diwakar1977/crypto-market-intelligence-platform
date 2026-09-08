from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.load.load_job import (
    LoadJob,
    LoadResult,
    create_load_job,
    run_load_job,
)
from src.load.redshift_storage import RedshiftStorage


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def spark() -> MagicMock:
    """Return a mocked SparkSession."""
    return MagicMock()


@pytest.fixture
def redshift_storage() -> MagicMock:
    """Return a mocked RedshiftStorage."""
    return MagicMock(spec=RedshiftStorage)


@pytest.fixture
def processed_schema() -> StructType:
    """Return a representative processed Parquet schema."""
    return StructType(
        [
            StructField("id", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("market_cap", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("rank", LongType(), True),
        ]
    )


@pytest.fixture
def load_job(
    spark: MagicMock,
    redshift_storage: MagicMock,
) -> LoadJob:
    """Return a valid LoadJob."""
    return LoadJob(
        spark=spark,
        redshift_storage=redshift_storage,
        redshift_schema="analytics",
        redshift_table="crypto_market",
        processed_spark_path=(
            "s3a://crypto-bucket/processed/"
        ),
        processed_s3_path=(
            "s3://crypto-bucket/processed/"
        ),
        redshift_iam_role=(
            "arn:aws:iam::123456789012:role/"
            "CryptoETL-Redshift-Role"
        ),
    )


# ============================================================
# INITIALIZATION
# ============================================================


def test_load_job_initialization(
    load_job: LoadJob,
) -> None:
    """Test LoadJob stores configuration correctly."""

    assert load_job.redshift_schema == "analytics"
    assert load_job.redshift_table == "crypto_market"

    assert (
        load_job.processed_spark_path
        == "s3a://crypto-bucket/processed/"
    )

    assert (
        load_job.processed_s3_path
        == "s3://crypto-bucket/processed/"
    )

    assert (
        load_job.redshift_iam_role
        == (
            "arn:aws:iam::123456789012:role/"
            "CryptoETL-Redshift-Role"
        )
    )


# ============================================================
# READ PROCESSED SCHEMA
# ============================================================


def test_read_processed_schema(
    load_job: LoadJob,
    spark: MagicMock,
    processed_schema: StructType,
) -> None:
    """Test reading and validating the processed Parquet schema."""

    dataframe = MagicMock()
    dataframe.schema = processed_schema

    spark.read.parquet.return_value = dataframe

    with patch(
        "src.load.load_job.RedshiftSchemaMapper.validate_schema"
    ) as mock_validate:
        result = load_job._read_processed_schema()

    spark.read.parquet.assert_called_once_with(
        "s3a://crypto-bucket/processed/"
    )

    mock_validate.assert_called_once_with(
        processed_schema
    )

    assert result == processed_schema


# ============================================================
# CREATE TABLE SQL
# ============================================================


def test_generate_create_table_sql(
    load_job: LoadJob,
    processed_schema: StructType,
) -> None:
    """Test CREATE TABLE SQL generation."""

    sql = load_job._generate_create_table_sql(
        processed_schema
    )

    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert '"analytics"."crypto_market"' in sql

    assert '"id"' in sql
    assert '"price"' in sql
    assert '"market_cap"' in sql
    assert '"volume"' in sql
    assert '"rank"' in sql

    assert "VARCHAR" in sql
    assert "DOUBLE PRECISION" in sql
    assert "BIGINT" in sql


# ============================================================
# COPY SQL
# ============================================================


def test_generate_copy_sql(
    load_job: LoadJob,
) -> None:
    """Test Redshift COPY SQL generation."""

    sql = load_job._generate_copy_sql()

    assert (
        'COPY "analytics"."crypto_market"'
        in sql
    )

    assert (
        "FROM 's3://crypto-bucket/processed/'"
        in sql
    )

    assert (
        "IAM_ROLE "
        "'arn:aws:iam::123456789012:role/"
        "CryptoETL-Redshift-Role'"
        in sql
    )

    assert "FORMAT AS PARQUET;" in sql


# ============================================================
# COMPLETE RUN
# ============================================================


def test_run_success(
    load_job: LoadJob,
    spark: MagicMock,
    redshift_storage: MagicMock,
    processed_schema: StructType,
) -> None:
    """Test complete successful Redshift load."""

    dataframe = MagicMock()
    dataframe.schema = processed_schema

    spark.read.parquet.return_value = dataframe

    with patch(
        "src.load.load_job.RedshiftSchemaMapper.validate_schema"
    ):
        result = load_job.run()

    assert isinstance(result, LoadResult)
    assert result.success is True
    assert result.schema_name == "analytics"
    assert result.table_name == "crypto_market"
    assert (
        result.source_path
        == "s3://crypto-bucket/processed/"
    )

    spark.read.parquet.assert_called_once_with(
        "s3a://crypto-bucket/processed/"
    )

    assert redshift_storage.execute.call_count == 2


# ============================================================
# RUN FAILURE
# ============================================================


def test_run_fails_when_schema_read_fails(
    load_job: LoadJob,
    spark: MagicMock,
) -> None:
    """Test run propagates schema-read failures."""

    spark.read.parquet.side_effect = RuntimeError(
        "Unable to read processed Parquet data."
    )

    with pytest.raises(
        RuntimeError,
        match="Unable to read processed Parquet data",
    ):
        load_job.run()


def test_run_fails_when_create_table_fails(
    load_job: LoadJob,
    spark: MagicMock,
    redshift_storage: MagicMock,
    processed_schema: StructType,
) -> None:
    """Test run propagates CREATE TABLE failures."""

    dataframe = MagicMock()
    dataframe.schema = processed_schema

    spark.read.parquet.return_value = dataframe

    redshift_storage.execute.side_effect = RuntimeError(
        "CREATE TABLE failed."
    )

    with patch(
        "src.load.load_job.RedshiftSchemaMapper.validate_schema"
    ):
        with pytest.raises(
            RuntimeError,
            match="CREATE TABLE failed",
        ):
            load_job.run()

    assert redshift_storage.execute.call_count == 1


def test_run_fails_when_copy_fails(
    load_job: LoadJob,
    spark: MagicMock,
    redshift_storage: MagicMock,
    processed_schema: StructType,
) -> None:
    """Test run propagates COPY failures."""

    dataframe = MagicMock()
    dataframe.schema = processed_schema

    spark.read.parquet.return_value = dataframe

    redshift_storage.execute.side_effect = [
        None,
        RuntimeError("Redshift COPY failed."),
    ]

    with patch(
        "src.load.load_job.RedshiftSchemaMapper.validate_schema"
    ):
        with pytest.raises(
            RuntimeError,
            match="Redshift COPY failed",
        ):
            load_job.run()

    assert redshift_storage.execute.call_count == 2


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_message"),
    [
        (
            "redshift_schema",
            "",
            "Redshift schema name cannot be empty",
        ),
        (
            "redshift_schema",
            "   ",
            "Redshift schema name cannot be empty",
        ),
        (
            "redshift_table",
            "",
            "Redshift table name cannot be empty",
        ),
        (
            "redshift_table",
            "   ",
            "Redshift table name cannot be empty",
        ),
        (
            "processed_spark_path",
            "",
            "Processed Spark path cannot be empty",
        ),
        (
            "processed_spark_path",
            "   ",
            "Processed Spark path cannot be empty",
        ),
        (
            "processed_s3_path",
            "",
            "Processed S3 path cannot be empty",
        ),
        (
            "processed_s3_path",
            "   ",
            "Processed S3 path cannot be empty",
        ),
        (
            "redshift_iam_role",
            "",
            "Redshift IAM role cannot be empty",
        ),
        (
            "redshift_iam_role",
            "   ",
            "Redshift IAM role cannot be empty",
        ),
    ],
)
def test_invalid_empty_configuration(
    spark: MagicMock,
    redshift_storage: MagicMock,
    field_name: str,
    field_value: str,
    error_message: str,
) -> None:
    """Test rejection of empty configuration values."""

    configuration = {
        "redshift_schema": "analytics",
        "redshift_table": "crypto_market",
        "processed_spark_path": (
            "s3a://crypto-bucket/processed/"
        ),
        "processed_s3_path": (
            "s3://crypto-bucket/processed/"
        ),
        "redshift_iam_role": (
            "arn:aws:iam::123456789012:role/"
            "CryptoETL-Redshift-Role"
        ),
    }

    configuration[field_name] = field_value

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        LoadJob(
            spark=spark,
            redshift_storage=redshift_storage,
            **configuration,
        )


def test_invalid_processed_spark_path(
    spark: MagicMock,
    redshift_storage: MagicMock,
) -> None:
    """Test processed Spark path must use s3a://."""

    with pytest.raises(
        ValueError,
        match="Processed Spark path must start with 's3a://'",
    ):
        LoadJob(
            spark=spark,
            redshift_storage=redshift_storage,
            redshift_schema="analytics",
            redshift_table="crypto_market",
            processed_spark_path="/local/processed/",
            processed_s3_path=(
                "s3://crypto-bucket/processed/"
            ),
            redshift_iam_role=(
                "arn:aws:iam::123456789012:role/"
                "CryptoETL-Redshift-Role"
            ),
        )


def test_invalid_processed_s3_path(
    spark: MagicMock,
    redshift_storage: MagicMock,
) -> None:
    """Test processed S3 path must use s3://."""

    with pytest.raises(
        ValueError,
        match="Processed S3 path must start with 's3://'",
    ):
        LoadJob(
            spark=spark,
            redshift_storage=redshift_storage,
            redshift_schema="analytics",
            redshift_table="crypto_market",
            processed_spark_path=(
                "s3a://crypto-bucket/processed/"
            ),
            processed_s3_path="/local/processed/",
            redshift_iam_role=(
                "arn:aws:iam::123456789012:role/"
                "CryptoETL-Redshift-Role"
            ),
        )


def test_invalid_iam_role(
    spark: MagicMock,
    redshift_storage: MagicMock,
) -> None:
    """Test IAM role must be an AWS IAM role ARN."""

    with pytest.raises(
        ValueError,
        match="Redshift IAM role must be a valid IAM role ARN",
    ):
        LoadJob(
            spark=spark,
            redshift_storage=redshift_storage,
            redshift_schema="analytics",
            redshift_table="crypto_market",
            processed_spark_path=(
                "s3a://crypto-bucket/processed/"
            ),
            processed_s3_path=(
                "s3://crypto-bucket/processed/"
            ),
            redshift_iam_role="invalid-role",
        )


# ============================================================
# FACTORY
# ============================================================


def test_create_load_job(
    spark: MagicMock,
    redshift_storage: MagicMock,
) -> None:
    """Test create_load_job factory."""

    job = create_load_job(
        spark=spark,
        redshift_storage=redshift_storage,
        redshift_schema="analytics",
        redshift_table="crypto_market",
        processed_spark_path=(
            "s3a://crypto-bucket/processed/"
        ),
        processed_s3_path=(
            "s3://crypto-bucket/processed/"
        ),
        redshift_iam_role=(
            "arn:aws:iam::123456789012:role/"
            "CryptoETL-Redshift-Role"
        ),
    )

    assert isinstance(job, LoadJob)
    assert job.redshift_schema == "analytics"
    assert job.redshift_table == "crypto_market"
    assert (
        job.processed_spark_path
        == "s3a://crypto-bucket/processed/"
    )
    assert (
        job.processed_s3_path
        == "s3://crypto-bucket/processed/"
    )


# ============================================================
# RUN LOAD JOB
# ============================================================


def test_run_load_job() -> None:
    """Test top-level run_load_job orchestration."""

    mock_spark = MagicMock()
    mock_storage = MagicMock()

    expected_result = LoadResult(
        schema_name="analytics",
        table_name="crypto_market",
        source_path=(
            "s3://crypto-bucket/processed/"
        ),
        success=True,
    )

    mock_job = MagicMock()
    mock_job.run.return_value = expected_result

    config = {
        "aws": {
            "region": "ap-south-1",
        },
        "s3": {
            "bucket": "crypto-bucket",
            "processed_prefix": "processed/",
        },
        "redshift": {
            "host": "example.redshift.amazonaws.com",
            "port": 5439,
            "database": "dev",
            "workgroup": "crypto-workgroup",
            "schema": "analytics",
            "table": "crypto_market",
            "iam_role": (
                "arn:aws:iam::123456789012:role/"
                "CryptoETL-Redshift-Role"
            ),
        },
    }

    with (
        patch(
            "src.load.load_job.CONFIG",
            config,
        ),
        patch(
            "src.load.load_job.SparkSessionFactory.create",
            return_value=mock_spark,
        ) as mock_create_spark,
        patch(
            "src.load.load_job.RedshiftStorage",
            return_value=mock_storage,
        ) as mock_storage_class,
        patch(
            "src.load.load_job.create_load_job",
            return_value=mock_job,
        ) as mock_create_job,
    ):
        result = run_load_job()

    assert result == expected_result

    mock_create_spark.assert_called_once()

    mock_storage_class.assert_called_once_with(
        host="example.redshift.amazonaws.com",
        port=5439,
        database="dev",
        aws_region="ap-south-1",
        workgroup="crypto-workgroup",
    )

    mock_create_job.assert_called_once_with(
        spark=mock_spark,
        redshift_storage=mock_storage,
        redshift_schema="analytics",
        redshift_table="crypto_market",
        processed_spark_path=(
            "s3a://crypto-bucket/processed/"
        ),
        processed_s3_path=(
            "s3://crypto-bucket/processed/"
        ),
        redshift_iam_role=(
            "arn:aws:iam::123456789012:role/"
            "CryptoETL-Redshift-Role"
        ),
    )

    mock_job.run.assert_called_once_with()
    mock_storage.close.assert_called_once_with()
    mock_spark.stop.assert_called_once_with()
