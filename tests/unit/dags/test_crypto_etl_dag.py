from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from airflow.providers.standard.operators.python import PythonOperator

from dags.crypto_etl_dag import (
    COMPUTE_MODE,
    dag,
    execute_extract,
    execute_load,
    execute_local_transform,
)

# ============================================================
# EXECUTE EXTRACT
# ============================================================


@patch("src.extract.extract_job.run_extract_job")
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
@patch("spark.spark_session.SparkSessionFactory.create")
@patch("src.transform.transform_job.run_transform_job")
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
        input_path="s3a://crypto-bucket/raw/2026-09-05/crypto.json",
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
@patch("spark.spark_session.SparkSessionFactory.create")
@patch("src.transform.transform_job.run_transform_job")
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


@patch("src.load.load_job.run_load_job")
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
    """Test that the DAG contains the expected tasks."""

    if COMPUTE_MODE == "local":
        assert set(dag.task_ids) == {
            "extract",
            "transform",
            "load",
        }

    else:
        assert set(dag.task_ids) == {
            "extract",
            "create_emr_cluster",
            "wait_for_emr_cluster",
            "add_transform_step",
            "wait_for_transform_step",
            "terminate_emr_cluster",
            "load",
        }


# ============================================================
# DAG DEPENDENCIES
# ============================================================


def test_dag_task_dependencies() -> None:
    """Test DAG task dependency chain."""

    extract_task = dag.get_task("extract")
    load_task = dag.get_task("load")

    if COMPUTE_MODE == "local":
        transform_task = dag.get_task("transform")

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

    else:
        create_emr_task = dag.get_task("create_emr_cluster")
        wait_emr_task = dag.get_task("wait_for_emr_cluster")
        transform_step_task = dag.get_task("add_transform_step")
        wait_transform_task = dag.get_task("wait_for_transform_step")
        terminate_emr_task = dag.get_task("terminate_emr_cluster")

        assert extract_task.downstream_task_ids == {
            "create_emr_cluster",
        }

        assert create_emr_task.upstream_task_ids == {
            "extract",
        }

        assert create_emr_task.downstream_task_ids == {
            "wait_for_emr_cluster",
        }

        assert wait_emr_task.downstream_task_ids == {
            "add_transform_step",
        }

        assert transform_step_task.downstream_task_ids == {
            "wait_for_transform_step",
        }

        assert wait_transform_task.downstream_task_ids == {
            "terminate_emr_cluster",
        }

        assert terminate_emr_task.downstream_task_ids == {
            "load",
        }

        assert load_task.upstream_task_ids == {
            "terminate_emr_cluster",
        }

        assert load_task.downstream_task_ids == set()


# ============================================================
# TASK CALLABLES
# ============================================================


def test_dag_task_callables() -> None:
    """Test PythonOperator callables in local mode."""

    extract_task = dag.get_task("extract")
    load_task = dag.get_task("load")

    assert isinstance(extract_task, PythonOperator)
    assert isinstance(load_task, PythonOperator)

    assert extract_task.python_callable == execute_extract
    assert load_task.python_callable == execute_load

    if COMPUTE_MODE == "local":
        transform_task = dag.get_task("transform")

        assert isinstance(transform_task, PythonOperator)
        assert transform_task.python_callable == execute_local_transform
