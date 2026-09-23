from __future__ import annotations

from typing import Any

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from callbacks.dag_callbacks import (
    dag_failure_callback,
    dag_success_callback,
)
from dag_config import (
    DAG_CATCHUP,
    DAG_DEFAULT_ARGS,
    DAG_DESCRIPTION,
    DAG_ID,
    DAG_MAX_ACTIVE_RUNS,
    DAG_MAX_ACTIVE_TASKS,
    DAG_SCHEDULE,
    DAG_START_DATE,
    DAG_TAGS,
)

from config.config import CONFIG
from spark.spark_session import SparkSessionFactory
from src.extract.extract_job import run_extract_job
from src.load.load_job import run_load_job
from src.transform.transform_job import run_transform_job

# ============================================================
# CONFIGURATION
# ============================================================

S3_CONFIG = CONFIG["s3"]
RUNTIME_CONFIG = CONFIG["runtime"]


# ============================================================
# COMPUTE MODE
# ============================================================

COMPUTE_MODE = str(RUNTIME_CONFIG["compute_mode"]).strip().lower()

if COMPUTE_MODE not in {"local", "emr"}:
    raise ValueError(
        "Invalid compute_mode. " "Expected 'local' or 'emr'. " f"Got: {COMPUTE_MODE}"
    )


# ============================================================
# EXTRACT JOB
# ============================================================


def execute_extract() -> dict[str, Any]:
    """
    Execute the extraction job.

    CoinGecko data is extracted and raw data
    is stored in S3.
    """

    result = run_extract_job()

    return {
        "s3_key": result.s3_key,
        "record_count": result.record_count,
    }


# ============================================================
# LOCAL TRANSFORM JOB
# ============================================================


def execute_local_transform(**context: Any) -> str:
    """
    Execute the Spark transformation locally.

    Raw data is read from S3 and processed.
    Parquet data is written to S3.
    """

    task_instance = context["ti"]

    extract_result = task_instance.xcom_pull(task_ids="extract")

    if not extract_result:
        raise ValueError("Extract task did not return a result.")

    raw_s3_key = str(extract_result["s3_key"]).strip()

    if not raw_s3_key:
        raise ValueError("Extract task returned an empty S3 key.")

    bucket = str(S3_CONFIG["bucket"]).strip()

    if not bucket:
        raise ValueError("S3 bucket cannot be empty.")

    input_path = f"s3a://{bucket}/{raw_s3_key}"

    spark = SparkSessionFactory.create()

    try:
        return run_transform_job(
            spark=spark,
            input_path=input_path,
        )
    finally:
        spark.stop()


# ============================================================
# LOAD JOB
# ============================================================


def execute_load() -> Any:
    """
    Execute the Redshift load job.

    All Redshift implementation details are handled
    internally by run_load_job().
    """

    return run_load_job()


# ============================================================
# DAG DEFINITION
# ============================================================

with DAG(
    dag_id=DAG_ID,
    description=DAG_DESCRIPTION,
    schedule=DAG_SCHEDULE,
    start_date=DAG_START_DATE,
    catchup=DAG_CATCHUP,
    default_args=DAG_DEFAULT_ARGS,
    max_active_runs=DAG_MAX_ACTIVE_RUNS,
    max_active_tasks=DAG_MAX_ACTIVE_TASKS,
    tags=DAG_TAGS,
    on_success_callback=dag_success_callback,
    on_failure_callback=dag_failure_callback,
) as dag:

    # ========================================================
    # TASK 1 - EXTRACT
    # ========================================================

    extract_task = PythonOperator(
        task_id="extract",
        python_callable=execute_extract,
    )

    # ========================================================
    # TASK 2 - LOAD
    # ========================================================

    load_task = PythonOperator(
        task_id="load",
        python_callable=execute_load,
    )

    # ========================================================
    # TASK 3 - TRANSFORM
    # ========================================================

    if COMPUTE_MODE == "local":

        # ----------------------------------------------------
        # LOCAL SPARK
        # ----------------------------------------------------

        transform_task = PythonOperator(
            task_id="transform",
            python_callable=execute_local_transform,
        )

        # ----------------------------------------------------
        # LOCAL ORCHESTRATION
        # ----------------------------------------------------

        extract_task >> transform_task >> load_task

    else:

        # ----------------------------------------------------
        # PRODUCTION - EMR
        # ----------------------------------------------------

        from tasks.emr_tasks import (
            add_transform_step,
            create_emr_cluster,
            terminate_emr_cluster,
            wait_for_emr_cluster,
            wait_for_transform_step,
        )

        create_emr_task = create_emr_cluster()

        wait_emr_task = wait_for_emr_cluster()

        transform_step_task = add_transform_step()

        wait_transform_task = wait_for_transform_step()

        terminate_emr_task = terminate_emr_cluster()

        # ----------------------------------------------------
        # EMR ORCHESTRATION
        # ----------------------------------------------------

        (
            extract_task
            >> create_emr_task
            >> wait_emr_task
            >> transform_step_task
            >> wait_transform_task
            >> terminate_emr_task
            >> load_task
        )
