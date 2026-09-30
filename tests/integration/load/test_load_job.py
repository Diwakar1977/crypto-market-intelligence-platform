from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from moto import mock_aws
from mypy_boto3_s3.client import S3Client

from src.load.load_job import LoadJob

BUCKET = "crypto-etl-prod-data-ap-south-1"

PROCESSED_PREFIX = "processed_data/crypto_market/"

RUN_PREFIX = "processed_data/crypto_market/" "year=2026/month=09/day=30/" "time=090000/"

PARQUET_KEY = f"{RUN_PREFIX}part-00000-test.parquet"


@pytest.fixture
def aws_s3() -> Generator[None, None, None]:
    with mock_aws():
        yield


@pytest.fixture
def s3_client(aws_s3: None) -> S3Client:
    client = boto3.client(
        "s3",
        region_name="ap-south-1",
    )

    client.create_bucket(
        Bucket=BUCKET,
        CreateBucketConfiguration={
            "LocationConstraint": "ap-south-1",
        },
    )

    return client


def create_test_parquet() -> bytes:
    table = pa.table(
        {
            "id": pa.array(
                ["bitcoin", "ethereum"],
                type=pa.string(),
            ),
            "symbol": pa.array(
                ["btc", "eth"],
                type=pa.string(),
            ),
            "current_price": pa.array(
                [65000.50, 3200.75],
                type=pa.float64(),
            ),
            "market_cap": pa.array(
                [1200000000000, 400000000000],
                type=pa.int64(),
            ),
            "last_updated": pa.array(
                [
                    datetime(
                        2026,
                        9,
                        30,
                        9,
                        0,
                    ),
                    datetime(
                        2026,
                        9,
                        30,
                        9,
                        0,
                    ),
                ],
                type=pa.timestamp("us"),
            ),
        }
    )

    buffer = BytesIO()

    pq.write_table(
        table,
        buffer,
    )

    return buffer.getvalue()


def create_load_job(
    s3_client: S3Client,
) -> LoadJob:
    redshift_storage = MagicMock()

    job = LoadJob.__new__(LoadJob)

    job.redshift_storage = redshift_storage
    job.redshift_schema = "public"
    job.redshift_table = "crypto_market"
    job.bucket = BUCKET
    job.processed_prefix = PROCESSED_PREFIX.rstrip("/")
    job.redshift_iam_role = "arn:aws:iam::123456789012:" "role/CryptoETL-Redshift-Role"
    job.aws_region = "ap-south-1"
    job._s3_client = s3_client

    return job


def test_load_job_discovers_latest_parquet_and_reads_schema(
    s3_client: S3Client,
) -> None:
    parquet_data = create_test_parquet()

    s3_client.put_object(
        Bucket=BUCKET,
        Key=PARQUET_KEY,
        Body=parquet_data,
    )

    job = create_load_job(s3_client)

    latest_run = job._find_latest_processed_run()

    assert latest_run == RUN_PREFIX

    parquet_key = job._find_parquet_file(
        latest_run,
    )

    assert parquet_key == PARQUET_KEY

    schema = job._read_processed_schema(
        latest_run,
    )

    assert len(schema) == 5

    assert schema.field("id").type == pa.string()
    assert schema.field("symbol").type == pa.string()
    assert schema.field("current_price").type == pa.float64()
    assert schema.field("market_cap").type == pa.int64()
    assert schema.field("last_updated").type == pa.timestamp("us")


def test_load_job_generates_create_table_from_real_parquet_schema(
    s3_client: S3Client,
) -> None:
    parquet_data = create_test_parquet()

    s3_client.put_object(
        Bucket=BUCKET,
        Key=PARQUET_KEY,
        Body=parquet_data,
    )

    job = create_load_job(s3_client)

    run_prefix = job._find_latest_processed_run()

    schema = job._read_processed_schema(
        run_prefix,
    )

    create_sql = job._generate_create_table_sql(
        schema,
    )

    assert "CREATE TABLE IF NOT EXISTS " '"public"."crypto_market"' in create_sql

    assert '"id" VARCHAR(65535)' in create_sql
    assert '"symbol" VARCHAR(65535)' in create_sql
    assert '"current_price" DOUBLE PRECISION' in create_sql
    assert '"market_cap" BIGINT' in create_sql
    assert '"last_updated" TIMESTAMP' in create_sql


def test_load_job_generates_copy_for_latest_run(
    s3_client: S3Client,
) -> None:
    parquet_data = create_test_parquet()

    s3_client.put_object(
        Bucket=BUCKET,
        Key=PARQUET_KEY,
        Body=parquet_data,
    )

    job = create_load_job(s3_client)

    run_prefix = job._find_latest_processed_run()

    copy_sql = job._generate_copy_sql(
        run_prefix,
    )

    assert "COPY " '"public"."crypto_market"' in copy_sql

    assert "FROM " f"'s3://{BUCKET}/{RUN_PREFIX}'" in copy_sql

    assert "IAM_ROLE" in copy_sql
    assert "FORMAT AS PARQUET" in copy_sql
