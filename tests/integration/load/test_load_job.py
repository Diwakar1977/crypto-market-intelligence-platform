from __future__ import annotations

from collections.abc import Generator
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.load.load_job import LoadJob
from src.load.redshift_storage import RedshiftStorage

# =====================================================================
# SPARK FIXTURE
# =====================================================================


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Create one local Spark session for integration tests."""

    session = (
        SparkSession.builder.master("local[2]")
        .appName("LoadJobIntegrationTest")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    yield session

    session.stop()


# =====================================================================
# PROCESSED DATA FIXTURE
# =====================================================================


@pytest.fixture
def processed_dataframe(
    spark: SparkSession,
) -> DataFrame:
    """Create a realistic processed crypto-market DataFrame."""

    schema = StructType(
        [
            # ---------------------------------------------------------
            # RAW COINGECKO COLUMNS
            # ---------------------------------------------------------
            StructField(
                "id",
                StringType(),
                nullable=False,
            ),
            StructField(
                "symbol",
                StringType(),
                nullable=False,
            ),
            StructField(
                "name",
                StringType(),
                nullable=False,
            ),
            StructField(
                "current_price",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "market_cap",
                LongType(),
                nullable=True,
            ),
            StructField(
                "market_cap_rank",
                LongType(),
                nullable=True,
            ),
            StructField(
                "high_24h",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "low_24h",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "price_change_percentage_24h",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "total_volume",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "last_updated",
                StringType(),
                nullable=True,
            ),
            # ---------------------------------------------------------
            # FEATURE-ENGINEERED COLUMNS
            # ---------------------------------------------------------
            StructField(
                "price_range_24h",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "price_change_direction",
                StringType(),
                nullable=True,
            ),
        ]
    )

    data = [
        (
            "bitcoin",
            "btc",
            "Bitcoin",
            105000.50,
            2100000000000,
            1,
            106000.00,
            103000.00,
            2.50,
            45000000000.00,
            "2026-09-08T10:00:00Z",
            3000.00,
            "UP",
        ),
        (
            "ethereum",
            "eth",
            "Ethereum",
            4200.75,
            500000000000,
            2,
            4300.00,
            4100.00,
            -1.25,
            18000000000.00,
            "2026-09-08T10:00:00Z",
            200.00,
            "DOWN",
        ),
    ]

    return spark.createDataFrame(
        data,
        schema=schema,
    )


# =====================================================================
# REDSHIFT STORAGE FIXTURE
# =====================================================================


@pytest.fixture
def redshift_storage() -> MagicMock:
    """
    Create a mocked RedshiftStorage.

    No real Redshift connection is created.
    """

    storage = MagicMock(
        spec=RedshiftStorage,
    )

    storage.execute = MagicMock()

    return storage


# =====================================================================
# LOAD JOB FIXTURE
# =====================================================================


@pytest.fixture
def load_job(
    spark: SparkSession,
    redshift_storage: MagicMock,
) -> LoadJob:
    """Create LoadJob with realistic configuration."""

    return LoadJob(
        spark=spark,
        redshift_storage=redshift_storage,
        redshift_schema="public",
        redshift_table="crypto_market",
        processed_spark_path=("s3a://test-bucket/" "processed_data/crypto_market/"),
        processed_s3_path=("s3://test-bucket/" "processed_data/crypto_market/"),
        redshift_iam_role=("arn:aws:iam::123456789012:" "role/CryptoETL-Redshift-Role"),
    )


# =====================================================================
# HELPER - CREATE MOCK DATAFRAME READER
# =====================================================================


def create_mock_reader(
    dataframe: DataFrame,
) -> MagicMock:
    """
    Create a mocked Spark DataFrameReader.

    Only parquet() is mocked. The DataFrame itself
    remains a real Spark DataFrame.
    """

    reader = MagicMock()

    reader.parquet.return_value = dataframe

    return reader


# =====================================================================
# END-TO-END LOAD JOB
# =====================================================================


def test_load_job_end_to_end(
    load_job: LoadJob,
    processed_dataframe: DataFrame,
) -> None:
    """
    Test the complete LoadJob orchestration.

    Real:
        - SparkSession
        - Spark DataFrame
        - LoadJob
        - RedshiftSchemaMapper
        - schema validation
        - CREATE TABLE SQL generation
        - COPY SQL generation
        - configuration validation

    Mocked:
        - Spark DataFrameReader.parquet()
        - RedshiftStorage.execute()

    No real AWS, S3, or Redshift resources are required.
    """

    mock_reader = create_mock_reader(
        processed_dataframe,
    )

    with patch.object(
        SparkSession,
        "read",
        new_callable=lambda: property(
            lambda self: mock_reader,
        ),
    ):
        execute_mock = cast(
            MagicMock,
            load_job.redshift_storage.execute,
        )

        execute_calls: list[str] = []

        def fake_execute(
            sql: str,
        ) -> None:
            """Capture SQL instead of executing Redshift."""

            execute_calls.append(sql)

        execute_mock.side_effect = fake_execute

        result = load_job.run()

    # =================================================================
    # VERIFY SPARK PARQUET READ
    # =================================================================

    mock_reader.parquet.assert_called_once_with(
        "s3a://test-bucket/" "processed_data/crypto_market/"
    )

    # =================================================================
    # VERIFY RESULT
    # =================================================================

    assert result.success is True

    assert result.schema_name == "public"

    assert result.table_name == "crypto_market"

    assert result.source_path == ("s3://test-bucket/" "processed_data/crypto_market/")

    # =================================================================
    # VERIFY REDSHIFT EXECUTION COUNT
    # =================================================================

    assert execute_mock.call_count == 2

    assert len(execute_calls) == 2

    create_table_sql = execute_calls[0]

    copy_sql = execute_calls[1]

    # =================================================================
    # VERIFY CREATE TABLE SQL
    # =================================================================

    assert "CREATE TABLE IF NOT EXISTS" in create_table_sql.upper()

    assert '"public"."crypto_market"' in create_table_sql

    # =================================================================
    # VERIFY RAW COLUMNS
    # =================================================================

    expected_raw_columns = [
        "id",
        "symbol",
        "name",
        "current_price",
        "market_cap",
        "market_cap_rank",
        "high_24h",
        "low_24h",
        "price_change_percentage_24h",
        "total_volume",
        "last_updated",
    ]

    for column in expected_raw_columns:
        assert f'"{column}"' in create_table_sql

    # =================================================================
    # VERIFY FEATURE COLUMNS
    # =================================================================

    expected_feature_columns = [
        "price_range_24h",
        "price_change_direction",
    ]

    for column in expected_feature_columns:
        assert f'"{column}"' in create_table_sql

    # =================================================================
    # VERIFY REDSHIFT TYPES
    # =================================================================

    assert '"id" VARCHAR' in create_table_sql

    assert '"symbol" VARCHAR' in create_table_sql

    assert '"name" VARCHAR' in create_table_sql

    assert '"current_price" DOUBLE PRECISION' in create_table_sql

    assert '"market_cap" BIGINT' in create_table_sql

    assert '"market_cap_rank" BIGINT' in create_table_sql

    assert '"high_24h" DOUBLE PRECISION' in create_table_sql

    assert '"low_24h" DOUBLE PRECISION' in create_table_sql

    assert '"price_change_percentage_24h" ' "DOUBLE PRECISION" in create_table_sql

    assert '"total_volume" DOUBLE PRECISION' in create_table_sql

    assert '"last_updated" VARCHAR' in create_table_sql

    assert '"price_range_24h" DOUBLE PRECISION' in create_table_sql

    assert '"price_change_direction" VARCHAR' in create_table_sql

    # =================================================================
    # VERIFY COPY SQL
    # =================================================================

    assert 'COPY "public"."crypto_market"' in copy_sql

    assert "FROM " "'s3://test-bucket/" "processed_data/crypto_market/'" in copy_sql

    assert (
        "IAM_ROLE "
        "'arn:aws:iam::123456789012:"
        "role/CryptoETL-Redshift-Role'" in copy_sql
    )

    assert "FORMAT AS PARQUET" in copy_sql

    assert copy_sql.endswith(";")

    # =================================================================
    # VERIFY EXECUTION ORDER
    # =================================================================

    assert execute_calls[0] == create_table_sql

    assert execute_calls[1] == copy_sql


# =====================================================================
# TEST PROCESSED SCHEMA READING
# =====================================================================


def test_read_processed_schema(
    load_job: LoadJob,
    processed_dataframe: DataFrame,
) -> None:
    """Test reading and validating the processed schema."""

    mock_reader = create_mock_reader(
        processed_dataframe,
    )

    with patch.object(
        SparkSession,
        "read",
        new_callable=lambda: property(
            lambda self: mock_reader,
        ),
    ):
        schema = load_job._read_processed_schema()

    mock_reader.parquet.assert_called_once_with(
        "s3a://test-bucket/" "processed_data/crypto_market/"
    )

    assert schema == processed_dataframe.schema

    assert len(schema.fields) == 13

    assert schema.fieldNames() == [
        "id",
        "symbol",
        "name",
        "current_price",
        "market_cap",
        "market_cap_rank",
        "high_24h",
        "low_24h",
        "price_change_percentage_24h",
        "total_volume",
        "last_updated",
        "price_range_24h",
        "price_change_direction",
    ]


# =====================================================================
# TEST CREATE TABLE SQL GENERATION
# =====================================================================


def test_generate_create_table_sql(
    load_job: LoadJob,
    processed_dataframe: DataFrame,
) -> None:
    """Test real Spark-to-Redshift schema mapping."""

    sql = load_job._generate_create_table_sql(
        processed_dataframe.schema,
    )

    assert isinstance(sql, str)

    assert "CREATE TABLE IF NOT EXISTS" in sql.upper()

    assert '"public"."crypto_market"' in sql

    assert '"id" VARCHAR' in sql

    assert '"symbol" VARCHAR' in sql

    assert '"name" VARCHAR' in sql

    assert '"current_price" DOUBLE PRECISION' in sql

    assert '"market_cap" BIGINT' in sql

    assert '"market_cap_rank" BIGINT' in sql

    assert '"high_24h" DOUBLE PRECISION' in sql

    assert '"low_24h" DOUBLE PRECISION' in sql

    assert '"price_change_percentage_24h" ' "DOUBLE PRECISION" in sql

    assert '"total_volume" DOUBLE PRECISION' in sql

    assert '"last_updated" VARCHAR' in sql

    assert '"price_range_24h" DOUBLE PRECISION' in sql

    assert '"price_change_direction" VARCHAR' in sql


# =====================================================================
# TEST COPY SQL GENERATION
# =====================================================================


def test_generate_copy_sql(
    load_job: LoadJob,
) -> None:
    """Test Redshift COPY SQL generation."""

    sql = load_job._generate_copy_sql()

    assert isinstance(sql, str)

    assert 'COPY "public"."crypto_market"' in sql

    assert "FROM " "'s3://test-bucket/" "processed_data/crypto_market/'" in sql

    assert (
        "IAM_ROLE " "'arn:aws:iam::123456789012:" "role/CryptoETL-Redshift-Role'" in sql
    )

    assert "FORMAT AS PARQUET;" in sql


# =====================================================================
# TEST TABLE CREATION
# =====================================================================


def test_create_target_table(
    load_job: LoadJob,
) -> None:
    """Test target table creation delegates SQL to storage."""

    sql = """
    CREATE TABLE IF NOT EXISTS
    "public"."crypto_market"
    (
        "id" VARCHAR
    );
    """

    load_job._create_target_table(sql)

    execute_mock = cast(
        MagicMock,
        load_job.redshift_storage.execute,
    )

    execute_mock.assert_called_once_with(sql)


# =====================================================================
# TEST DATA LOAD
# =====================================================================


def test_load_data(
    load_job: LoadJob,
) -> None:
    """Test COPY SQL execution."""

    sql = """
    COPY "public"."crypto_market"
    FROM 's3://test-bucket/processed_data/'
    IAM_ROLE 'arn:aws:iam::123456789012:role/TestRole'
    FORMAT AS PARQUET;
    """

    load_job._load_data(sql)

    execute_mock = cast(
        MagicMock,
        load_job.redshift_storage.execute,
    )

    execute_mock.assert_called_once_with(sql)


# =====================================================================
# CONFIGURATION VALIDATION
# =====================================================================


@pytest.mark.parametrize(
    (
        "field_name",
        "field_value",
        "expected_message",
    ),
    [
        (
            "redshift_schema",
            "",
            "Redshift schema name cannot be empty.",
        ),
        (
            "redshift_table",
            "",
            "Redshift table name cannot be empty.",
        ),
        (
            "processed_spark_path",
            "",
            "Processed Spark path cannot be empty.",
        ),
        (
            "processed_s3_path",
            "",
            "Processed S3 path cannot be empty.",
        ),
        (
            "redshift_iam_role",
            "",
            "Redshift IAM role cannot be empty.",
        ),
    ],
)
def test_load_job_configuration_empty_values(
    spark: SparkSession,
    redshift_storage: MagicMock,
    field_name: str,
    field_value: str,
    expected_message: str,
) -> None:
    """Reject empty required configuration values."""

    if field_name == "redshift_schema":
        with pytest.raises(
            ValueError,
            match=expected_message,
        ):
            LoadJob(
                spark=spark,
                redshift_storage=redshift_storage,
                redshift_schema=field_value,
                redshift_table="crypto_market",
                processed_spark_path=("s3a://test-bucket/processed/"),
                processed_s3_path=("s3://test-bucket/processed/"),
                redshift_iam_role=("arn:aws:iam::123456789012:" "role/TestRole"),
            )

    elif field_name == "redshift_table":
        with pytest.raises(
            ValueError,
            match=expected_message,
        ):
            LoadJob(
                spark=spark,
                redshift_storage=redshift_storage,
                redshift_schema="public",
                redshift_table=field_value,
                processed_spark_path=("s3a://test-bucket/processed/"),
                processed_s3_path=("s3://test-bucket/processed/"),
                redshift_iam_role=("arn:aws:iam::123456789012:" "role/TestRole"),
            )

    elif field_name == "processed_spark_path":
        with pytest.raises(
            ValueError,
            match=expected_message,
        ):
            LoadJob(
                spark=spark,
                redshift_storage=redshift_storage,
                redshift_schema="public",
                redshift_table="crypto_market",
                processed_spark_path=field_value,
                processed_s3_path=("s3://test-bucket/processed/"),
                redshift_iam_role=("arn:aws:iam::123456789012:" "role/TestRole"),
            )

    elif field_name == "processed_s3_path":
        with pytest.raises(
            ValueError,
            match=expected_message,
        ):
            LoadJob(
                spark=spark,
                redshift_storage=redshift_storage,
                redshift_schema="public",
                redshift_table="crypto_market",
                processed_spark_path=("s3a://test-bucket/processed/"),
                processed_s3_path=field_value,
                redshift_iam_role=("arn:aws:iam::123456789012:" "role/TestRole"),
            )

    elif field_name == "redshift_iam_role":
        with pytest.raises(
            ValueError,
            match=expected_message,
        ):
            LoadJob(
                spark=spark,
                redshift_storage=redshift_storage,
                redshift_schema="public",
                redshift_table="crypto_market",
                processed_spark_path=("s3a://test-bucket/processed/"),
                processed_s3_path=("s3://test-bucket/processed/"),
                redshift_iam_role=field_value,
            )

    else:
        pytest.fail(f"Unsupported field name: {field_name}")


# =====================================================================
# INVALID SPARK PATH
# =====================================================================


def test_invalid_processed_spark_path(
    spark: SparkSession,
    redshift_storage: MagicMock,
) -> None:
    """Reject a processed Spark path without s3a://."""

    with pytest.raises(
        ValueError,
        match=("Processed Spark path must start " "with 's3a://'."),
    ):
        LoadJob(
            spark=spark,
            redshift_storage=redshift_storage,
            redshift_schema="public",
            redshift_table="crypto_market",
            processed_spark_path=("s3://test-bucket/processed/"),
            processed_s3_path=("s3://test-bucket/processed/"),
            redshift_iam_role=("arn:aws:iam::123456789012:" "role/TestRole"),
        )


# =====================================================================
# INVALID S3 PATH
# =====================================================================


def test_invalid_processed_s3_path(
    spark: SparkSession,
    redshift_storage: MagicMock,
) -> None:
    """Reject a processed S3 path without s3://."""

    with pytest.raises(
        ValueError,
        match=("Processed S3 path must start " "with 's3://'."),
    ):
        LoadJob(
            spark=spark,
            redshift_storage=redshift_storage,
            redshift_schema="public",
            redshift_table="crypto_market",
            processed_spark_path=("s3a://test-bucket/processed/"),
            processed_s3_path=("s3a://test-bucket/processed/"),
            redshift_iam_role=("arn:aws:iam::123456789012:" "role/TestRole"),
        )


# =====================================================================
# INVALID IAM ROLE
# =====================================================================


def test_invalid_redshift_iam_role(
    spark: SparkSession,
    redshift_storage: MagicMock,
) -> None:
    """Reject an invalid Redshift IAM role ARN."""

    with pytest.raises(
        ValueError,
        match=("Redshift IAM role must be a valid " "IAM role ARN."),
    ):
        LoadJob(
            spark=spark,
            redshift_storage=redshift_storage,
            redshift_schema="public",
            redshift_table="crypto_market",
            processed_spark_path=("s3a://test-bucket/processed/"),
            processed_s3_path=("s3://test-bucket/processed/"),
            redshift_iam_role="invalid-role",
        )


# =====================================================================
# SQL ESCAPING
# =====================================================================


def test_copy_sql_escapes_single_quotes(
    spark: SparkSession,
    redshift_storage: MagicMock,
) -> None:
    """Verify COPY SQL safely escapes single quotes."""

    job = LoadJob(
        spark=spark,
        redshift_storage=redshift_storage,
        redshift_schema="public",
        redshift_table="crypto_market",
        processed_spark_path=("s3a://test-bucket/processed/"),
        processed_s3_path=("s3://test-bucket/data/it's-safe/"),
        redshift_iam_role=("arn:aws:iam::123456789012:" "role/Test'Role"),
    )

    sql = job._generate_copy_sql()

    assert "data/it''s-safe/" in sql

    assert "role/Test''Role" in sql
