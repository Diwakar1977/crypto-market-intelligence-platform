from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from src.load.load_job import LoadJob, LoadResult


@pytest.fixture
def redshift_storage() -> MagicMock:
    return MagicMock()


@pytest.fixture
def load_job(redshift_storage: MagicMock) -> LoadJob:
    job = LoadJob.__new__(LoadJob)

    job.redshift_storage = redshift_storage
    job.redshift_schema = "public"
    job.redshift_table = "crypto_market"
    job.bucket = "crypto-etl-prod-data-ap-south-1"
    job.processed_prefix = "processed_data"
    job.redshift_iam_role = "arn:aws:iam::123456789012:role/CryptoETL-Redshift-Role"
    job.aws_region = "ap-south-1"

    job._s3_client = MagicMock()

    return job


# ------------------------------------------------------------------
# _find_latest_processed_run
# ------------------------------------------------------------------


def test_find_latest_processed_run_returns_latest_run(
    load_job: LoadJob,
) -> None:
    first_modified = datetime(
        2026,
        9,
        30,
        8,
        30,
        tzinfo=timezone.utc,
    )

    latest_modified = datetime(
        2026,
        9,
        30,
        9,
        30,
        tzinfo=timezone.utc,
    )

    cast(MagicMock, load_job._s3_client.list_objects_v2).return_value = {
        "Contents": [
            {
                "Key": (
                    "processed_data/crypto_market/"
                    "year=2026/month=09/day=30/"
                    "time=083000/"
                    "part-00000.parquet"
                ),
                "LastModified": first_modified,
            },
            {
                "Key": (
                    "processed_data/crypto_market/"
                    "year=2026/month=09/day=30/"
                    "time=093000/"
                    "part-00000.parquet"
                ),
                "LastModified": latest_modified,
            },
            {
                "Key": (
                    "processed_data/crypto_market/"
                    "year=2026/month=09/day=30/"
                    "time=093000/"
                    "_SUCCESS"
                ),
                "LastModified": latest_modified,
            },
        ],
        "IsTruncated": False,
    }

    result = load_job._find_latest_processed_run()

    assert result == (
        "processed_data/crypto_market/" "year=2026/month=09/day=30/" "time=093000/"
    )


def test_find_latest_processed_run_ignores_non_parquet_files(
    load_job: LoadJob,
) -> None:
    cast(MagicMock, load_job._s3_client.list_objects_v2).return_value = {
        "Contents": [
            {
                "Key": (
                    "processed_data/crypto_market/"
                    "year=2026/month=09/day=30/"
                    "time=083000/"
                    "_SUCCESS"
                ),
                "LastModified": datetime(
                    2026,
                    9,
                    30,
                    8,
                    30,
                    tzinfo=timezone.utc,
                ),
            },
        ],
        "IsTruncated": False,
    }

    with pytest.raises(
        FileNotFoundError,
        match="No processed Parquet files",
    ):
        load_job._find_latest_processed_run()


def test_find_latest_processed_run_raises_when_no_objects(
    load_job: LoadJob,
) -> None:
    cast(MagicMock, load_job._s3_client.list_objects_v2).return_value = {
        "Contents": [],
        "IsTruncated": False,
    }

    with pytest.raises(
        FileNotFoundError,
        match="No processed Parquet files",
    ):
        load_job._find_latest_processed_run()


def test_find_latest_processed_run_handles_pagination(
    load_job: LoadJob,
) -> None:
    first_response = {
        "Contents": [
            {
                "Key": (
                    "processed_data/crypto_market/"
                    "year=2026/month=09/day=30/"
                    "time=080000/"
                    "part-00000.parquet"
                ),
                "LastModified": datetime(
                    2026,
                    9,
                    30,
                    8,
                    0,
                    tzinfo=timezone.utc,
                ),
            },
        ],
        "IsTruncated": True,
        "NextContinuationToken": "TOKEN-1",
    }

    second_response = {
        "Contents": [
            {
                "Key": (
                    "processed_data/crypto_market/"
                    "year=2026/month=09/day=30/"
                    "time=090000/"
                    "part-00000.parquet"
                ),
                "LastModified": datetime(
                    2026,
                    9,
                    30,
                    9,
                    0,
                    tzinfo=timezone.utc,
                ),
            },
        ],
        "IsTruncated": False,
    }

    cast(MagicMock, load_job._s3_client.list_objects_v2).side_effect = [
        first_response,
        second_response,
    ]

    result = load_job._find_latest_processed_run()

    assert result == (
        "processed_data/crypto_market/" "year=2026/month=09/day=30/" "time=090000/"
    )

    assert (
        cast(
            MagicMock,
            load_job._s3_client.list_objects_v2,
        ).call_count
        == 2
    )

    second_call = cast(
        MagicMock,
        load_job._s3_client.list_objects_v2,
    ).call_args_list[1]

    assert second_call.kwargs["ContinuationToken"] == "TOKEN-1"


# ------------------------------------------------------------------
# _find_parquet_file
# ------------------------------------------------------------------


def test_find_parquet_file_returns_parquet_key(
    load_job: LoadJob,
) -> None:
    run_prefix = (
        "processed_data/crypto_market/" "year=2026/month=09/day=30/" "time=090000/"
    )

    cast(MagicMock, load_job._s3_client.list_objects_v2).return_value = {
        "Contents": [
            {
                "Key": f"{run_prefix}_SUCCESS",
            },
            {
                "Key": f"{run_prefix}part-00001.parquet",
            },
            {
                "Key": f"{run_prefix}part-00000.parquet",
            },
        ],
    }

    result = load_job._find_parquet_file(run_prefix)

    assert result == f"{run_prefix}part-00000.parquet"


def test_find_parquet_file_raises_when_missing(
    load_job: LoadJob,
) -> None:
    run_prefix = (
        "processed_data/crypto_market/" "year=2026/month=09/day=30/" "time=090000/"
    )

    cast(MagicMock, load_job._s3_client.list_objects_v2).return_value = {
        "Contents": [
            {
                "Key": f"{run_prefix}_SUCCESS",
            },
        ],
    }

    with pytest.raises(
        FileNotFoundError,
        match="No Parquet file found",
    ):
        load_job._find_parquet_file(run_prefix)


# ------------------------------------------------------------------
# _generate_create_table_sql
# ------------------------------------------------------------------


def test_generate_create_table_sql(
    load_job: LoadJob,
) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("market_cap", pa.int64()),
            pa.field("current_price", pa.float64()),
            pa.field(
                "last_updated",
                pa.timestamp("us"),
            ),
        ]
    )

    result = load_job._generate_create_table_sql(schema)

    assert result == (
        "CREATE TABLE IF NOT EXISTS "
        '"public"."crypto_market" (\n'
        '    "id" VARCHAR(65535),\n'
        '    "market_cap" BIGINT,\n'
        '    "current_price" DOUBLE PRECISION,\n'
        '    "last_updated" TIMESTAMP\n'
        ");"
    )


# ------------------------------------------------------------------
# _generate_copy_sql
# ------------------------------------------------------------------


def test_generate_copy_sql(
    load_job: LoadJob,
) -> None:
    run_prefix = (
        "processed_data/crypto_market/" "year=2026/month=09/day=30/" "time=090000/"
    )

    result = load_job._generate_copy_sql(run_prefix)

    expected = (
        'COPY "public"."crypto_market"\n'
        "FROM "
        "'s3://crypto-etl-prod-data-ap-south-1/"
        "processed_data/crypto_market/"
        "year=2026/month=09/day=30/"
        "time=090000/'\n"
        "IAM_ROLE "
        "'arn:aws:iam::123456789012:"
        "role/CryptoETL-Redshift-Role'\n"
        "FORMAT AS PARQUET;"
    )

    assert result == expected


# ------------------------------------------------------------------
# _create_target_table
# ------------------------------------------------------------------


def test_create_target_table(
    load_job: LoadJob,
    redshift_storage: MagicMock,
) -> None:
    sql = "CREATE TABLE IF NOT EXISTS " '"public"."crypto_market" ' '("id" BIGINT);'

    load_job._create_target_table(sql)

    redshift_storage.execute.assert_called_once_with(sql)


# ------------------------------------------------------------------
# _load_data
# ------------------------------------------------------------------


def test_load_data(
    load_job: LoadJob,
    redshift_storage: MagicMock,
) -> None:
    sql = (
        'COPY "public"."crypto_market" '
        "FROM 's3://bucket/path/' "
        "FORMAT AS PARQUET;"
    )

    load_job._load_data(sql)

    redshift_storage.execute.assert_called_once_with(sql)


# ------------------------------------------------------------------
# _validate_configuration
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        (
            "redshift_schema",
            "",
            "Redshift schema name cannot be empty",
        ),
        (
            "redshift_table",
            "",
            "Redshift table name cannot be empty",
        ),
        (
            "bucket",
            "",
            "S3 bucket cannot be empty",
        ),
        (
            "processed_prefix",
            "",
            "Processed S3 prefix cannot be empty",
        ),
        (
            "redshift_iam_role",
            "",
            "Redshift IAM role cannot be empty",
        ),
        (
            "aws_region",
            "",
            "AWS region cannot be empty",
        ),
    ],
)
def test_invalid_configuration(
    load_job: LoadJob,
    attribute: str,
    value: str,
    message: str,
) -> None:
    setattr(load_job, attribute, value)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        load_job._validate_configuration()


def test_invalid_iam_role_arn(
    load_job: LoadJob,
) -> None:
    load_job.redshift_iam_role = "invalid-role"

    with pytest.raises(
        ValueError,
        match="valid IAM role ARN",
    ):
        load_job._validate_configuration()


# ------------------------------------------------------------------
# run()
# ------------------------------------------------------------------


def test_run_executes_complete_load_flow(
    load_job: LoadJob,
    redshift_storage: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_prefix = (
        "processed_data/crypto_market/" "year=2026/month=09/day=30/" "time=090000/"
    )

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("current_price", pa.float64()),
        ]
    )

    monkeypatch.setattr(
        load_job,
        "_find_latest_processed_run",
        lambda: run_prefix,
    )

    monkeypatch.setattr(
        load_job,
        "_read_processed_schema",
        lambda prefix: schema,
    )

    monkeypatch.setattr(
        load_job,
        "_generate_create_table_sql",
        lambda received_schema: "CREATE TABLE TEST;",
    )

    monkeypatch.setattr(
        load_job,
        "_generate_copy_sql",
        lambda prefix: "COPY TEST;",
    )

    result = load_job.run()

    assert isinstance(result, LoadResult)

    assert result.schema_name == "public"
    assert result.table_name == "crypto_market"

    assert result.source_path == (
        "s3://crypto-etl-prod-data-ap-south-1/"
        "processed_data/crypto_market/"
        "year=2026/month=09/day=30/"
        "time=090000/"
    )

    assert result.success is True

    assert redshift_storage.execute.call_count == 2

    calls = redshift_storage.execute.call_args_list

    assert calls[0].args[0] == "CREATE TABLE TEST;"
    assert calls[1].args[0] == "COPY TEST;"


# ------------------------------------------------------------------
# _download_s3_object
# ------------------------------------------------------------------


def test_download_s3_object(
    load_job: LoadJob,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "processed.parquet"

    load_job._download_s3_object(
        key="processed_data/test.parquet",
        destination=destination,
    )

    cast(
        MagicMock,
        load_job._s3_client.download_file,
    ).assert_called_once_with(
        "crypto-etl-prod-data-ap-south-1",
        "processed_data/test.parquet",
        str(destination),
    )
