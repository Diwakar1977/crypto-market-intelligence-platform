from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from airflow.providers.standard.operators.python import PythonOperator

from dags.crypto_etl_dag import (
    dag,
    execute_extract,
    execute_load,
    execute_local_transform,
)

# ============================================================
# EXECUTE EXTRACT
# ============================================================


@patch("dags.crypto_etl_dag.run_extract_job")
def test_execute_extract(
    mock_run_extract_job: MagicMock,
) -> None:
    """Test extraction task wrapper."""

    mock_result = MagicMock()
    mock_result.s3_key = "raw/2026-09-05/crypto.json"
    mock_result.record_count = 100

    mock_run_extract_job.return_value = mock_result

    result = execute_extract()

    assert result == {
        "s3_key": "raw/2026-09-05/crypto.json",
        "record_count": 100,
    }

    mock_run_extract_job.assert_called_once_with()


# ============================================================
# EXECUTE LOCAL TRANSFORM
# ============================================================


@patch(
    "dags.crypto_etl_dag.S3_CONFIG",
    {"bucket": "crypto-bucket"},
)
@patch("dags.crypto_etl_dag.SparkSessionFactory.create")
@patch("dags.crypto_etl_dag.run_transform_job")
def test_execute_local_transform(
    mock_run_transform_job: MagicMock,
    mock_spark_create: MagicMock,
) -> None:
    """Test local Spark transformation task."""

    mock_spark = MagicMock()
    mock_spark_create.return_value = mock_spark

    mock_run_transform_job.return_value = "processed/2026-09-05/"

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = {
        "s3_key": "raw/2026-09-05/crypto.json",
        "record_count": 100,
    }

    context = {
        "ti": task_instance,
    }

    result = execute_local_transform(**context)

    assert result == "processed/2026-09-05/"

    task_instance.xcom_pull.assert_called_once_with(
        task_ids="extract",
    )

    mock_spark_create.assert_called_once_with()

    mock_run_transform_job.assert_called_once_with(
        spark=mock_spark,
        input_path=("s3a://crypto-bucket/" "raw/2026-09-05/crypto.json"),
    )

    mock_spark.stop.assert_called_once_with()


# ============================================================
# LOCAL TRANSFORM - MISSING EXTRACT RESULT
# ============================================================


def test_execute_local_transform_missing_extract_result() -> None:
    """Test failure when extract task returns no result."""

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = None

    context = {
        "ti": task_instance,
    }

    with pytest.raises(
        ValueError,
        match="Extract task did not return a result",
    ):
        execute_local_transform(**context)


# ============================================================
# LOCAL TRANSFORM - EMPTY S3 KEY
# ============================================================


def test_execute_local_transform_empty_s3_key() -> None:
    """Test failure when extract task returns an empty S3 key."""

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = {
        "s3_key": "",
        "record_count": 100,
    }

    context = {
        "ti": task_instance,
    }

    with pytest.raises(
        ValueError,
        match="Extract task returned an empty S3 key",
    ):
        execute_local_transform(**context)


# ============================================================
# LOCAL TRANSFORM - SPARK CLEANUP ON FAILURE
# ============================================================


@patch(
    "dags.crypto_etl_dag.S3_CONFIG",
    {"bucket": "crypto-bucket"},
)
@patch("dags.crypto_etl_dag.SparkSessionFactory.create")
@patch("dags.crypto_etl_dag.run_transform_job")
def test_execute_local_transform_stops_spark_on_failure(
    mock_run_transform_job: MagicMock,
    mock_spark_create: MagicMock,
) -> None:
    """Test Spark is stopped when transformation fails."""

    mock_spark = MagicMock()
    mock_spark_create.return_value = mock_spark

    mock_run_transform_job.side_effect = RuntimeError("Transformation failed")

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = {
        "s3_key": "raw/2026-09-05/crypto.json",
        "record_count": 100,
    }

    context = {
        "ti": task_instance,
    }

    with pytest.raises(
        RuntimeError,
        match="Transformation failed",
    ):
        execute_local_transform(**context)

    mock_spark.stop.assert_called_once_with()


# ============================================================
# EXECUTE LOAD
# ============================================================


@patch("dags.crypto_etl_dag.run_load_job")
def test_execute_load(
    mock_run_load_job: MagicMock,
) -> None:
    """Test load task wrapper."""

    mock_result = MagicMock()

    mock_run_load_job.return_value = mock_result

    result = execute_load()

    assert result is mock_result

    mock_run_load_job.assert_called_once_with()


# ============================================================
# DAG TASKS
# ============================================================


def test_dag_contains_expected_tasks() -> None:
    """Test that the local DAG contains expected tasks."""

    assert set(dag.task_ids) == {
        "extract",
        "transform",
        "load",
    }


# ============================================================
# DAG DEPENDENCIES
# ============================================================


def test_dag_task_dependencies() -> None:
    """Test extract -> transform -> load dependency chain."""

    extract_task = dag.get_task("extract")
    transform_task = dag.get_task("transform")
    load_task = dag.get_task("load")

    assert extract_task.downstream_task_ids == {
        "transform",
    }

    assert transform_task.upstream_task_ids == {
        "extract",
    }

    assert transform_task.downstream_task_ids == {
        "load",
    }

    assert load_task.upstream_task_ids == {
        "transform",
    }

    assert load_task.downstream_task_ids == set()


# ============================================================
# TASK CALLABLES
# ============================================================


def test_dag_task_callables() -> None:
    """Test PythonOperator callables."""

    extract_task = dag.get_task("extract")
    transform_task = dag.get_task("transform")
    load_task = dag.get_task("load")

    # Narrow Airflow's BaseOperator | MappedOperator union
    # to PythonOperator for mypy.
    assert isinstance(extract_task, PythonOperator)
    assert isinstance(transform_task, PythonOperator)
    assert isinstance(load_task, PythonOperator)

    assert extract_task.python_callable == execute_extract
    assert transform_task.python_callable == execute_local_transform
    assert load_task.python_callable == execute_load
