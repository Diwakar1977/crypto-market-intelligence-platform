from __future__ import annotations

from typing import Any, cast

from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import (
    EmrAddStepsOperator,
    EmrCreateJobFlowOperator,
    EmrTerminateJobFlowOperator,
)
from airflow.providers.amazon.aws.sensors.emr import (
    EmrJobFlowSensor,
    EmrStepSensor,
)
from airflow.sdk import TriggerRule

from dags.tasks.emr_tasks import (
    AWS_REGION,
    EMR_CLUSTER_NAME,
    EMR_CORE_INSTANCE_COUNT,
    EMR_CORE_INSTANCE_TYPE,
    EMR_INSTANCE_PROFILE,
    EMR_LOG_URI,
    EMR_MASTER_INSTANCE_TYPE,
    EMR_RELEASE,
    EMR_SERVICE_ROLE,
    EMR_SUBNET_ID,
    S3_BUCKET,
    S3_PROCESSED_PREFIX,
    S3_RAW_PREFIX,
    add_transform_step,
    create_emr_cluster,
    create_emr_tasks,
    terminate_emr_cluster,
    wait_for_emr_cluster,
    wait_for_transform_step,
)

# ============================================================
# CREATE EMR CLUSTER
# ============================================================


def test_create_emr_cluster() -> None:
    """Test EMR cluster creation configuration."""

    task = create_emr_cluster()

    assert isinstance(task, EmrCreateJobFlowOperator)

    assert task.task_id == "create_emr_cluster"
    assert task.aws_conn_id == "aws_default"
    assert task.region_name == AWS_REGION

    # Airflow's provider typing is broader than the actual
    # EMR configuration returned by our task factory.
    overrides = cast(
        dict[str, Any],
        task.job_flow_overrides,
    )

    assert overrides["Name"] == EMR_CLUSTER_NAME
    assert overrides["ReleaseLabel"] == EMR_RELEASE

    assert overrides["Applications"] == [
        {"Name": "Spark"},
    ]

    instances = cast(
        dict[str, Any],
        overrides["Instances"],
    )

    assert instances["Ec2SubnetId"] == EMR_SUBNET_ID
    assert instances["KeepJobFlowAliveWhenNoSteps"] is True
    assert instances["TerminationProtected"] is False

    instance_groups = cast(
        list[dict[str, Any]],
        instances["InstanceGroups"],
    )

    assert instance_groups[0] == {
        "Name": "Master",
        "Market": "ON_DEMAND",
        "InstanceRole": "MASTER",
        "InstanceType": EMR_MASTER_INSTANCE_TYPE,
        "InstanceCount": 1,
    }

    assert instance_groups[1] == {
        "Name": "Core",
        "Market": "ON_DEMAND",
        "InstanceRole": "CORE",
        "InstanceType": EMR_CORE_INSTANCE_TYPE,
        "InstanceCount": EMR_CORE_INSTANCE_COUNT,
    }

    assert overrides["JobFlowRole"] == EMR_INSTANCE_PROFILE
    assert overrides["ServiceRole"] == EMR_SERVICE_ROLE
    assert overrides["LogUri"] == EMR_LOG_URI
    assert overrides["VisibleToAllUsers"] is True


# ============================================================
# WAIT FOR EMR CLUSTER
# ============================================================


def test_wait_for_emr_cluster() -> None:
    """Test EMR cluster waiting sensor configuration."""

    task = wait_for_emr_cluster()

    assert isinstance(task, EmrJobFlowSensor)

    assert task.task_id == "wait_for_emr_cluster"
    assert task.aws_conn_id == "aws_default"
    assert task.region_name == AWS_REGION

    assert task.job_flow_id == (
        "{{ ti.xcom_pull(" "task_ids='create_emr_cluster', " "key='return_value'" ") }}"
    )

    assert task.target_states == ["WAITING"]

    assert task.failed_states == [
        "TERMINATED",
        "TERMINATED_WITH_ERRORS",
    ]


# ============================================================
# ADD TRANSFORM STEP
# ============================================================


def test_add_transform_step() -> None:
    """Test Spark transformation step configuration."""

    task = add_transform_step()

    assert isinstance(task, EmrAddStepsOperator)

    assert task.task_id == "add_transform_step"
    assert task.aws_conn_id == "aws_default"
    assert task.region_name == AWS_REGION

    assert task.job_flow_id == (
        "{{ ti.xcom_pull(" "task_ids='create_emr_cluster', " "key='return_value'" ") }}"
    )

    assert len(task.steps) == 1

    step = cast(
        dict[str, Any],
        task.steps[0],
    )

    assert step["Name"] == "crypto-market-transform"

    # Must match production emr_tasks.py.
    assert step["ActionOnFailure"] == "CANCEL_AND_WAIT"

    hadoop_step = cast(
        dict[str, Any],
        step["HadoopJarStep"],
    )

    assert hadoop_step["Jar"] == "command-runner.jar"

    assert hadoop_step["Args"] == [
        "spark-submit",
        "--deploy-mode",
        "cluster",
        (f"s3://{S3_BUCKET}/" "src/transform/transform_job.py"),
        "--input",
        (f"s3://{S3_BUCKET}/" f"{S3_RAW_PREFIX}"),
        "--output",
        (f"s3://{S3_BUCKET}/" f"{S3_PROCESSED_PREFIX}"),
    ]


# ============================================================
# WAIT FOR TRANSFORM STEP
# ============================================================


def test_wait_for_transform_step() -> None:
    """Test Spark transformation step sensor configuration."""

    task = wait_for_transform_step()

    assert isinstance(task, EmrStepSensor)

    assert task.task_id == "wait_for_transform_step"
    assert task.aws_conn_id == "aws_default"
    assert task.region_name == AWS_REGION

    assert task.job_flow_id == (
        "{{ ti.xcom_pull(" "task_ids='create_emr_cluster', " "key='return_value'" ") }}"
    )

    assert task.step_id == (
        "{{ ti.xcom_pull("
        "task_ids='add_transform_step', "
        "key='return_value'"
        ")[0] }}"
    )

    assert task.target_states == ["COMPLETED"]

    assert task.failed_states == [
        "CANCEL_PENDING",
        "CANCELLED",
        "FAILED",
        "INTERRUPTED",
    ]


# ============================================================
# TERMINATE EMR CLUSTER
# ============================================================


def test_terminate_emr_cluster() -> None:
    """Test EMR cluster termination configuration."""

    task = terminate_emr_cluster()

    assert isinstance(
        task,
        EmrTerminateJobFlowOperator,
    )

    assert task.task_id == "terminate_emr_cluster"
    assert task.aws_conn_id == "aws_default"
    assert task.region_name == AWS_REGION

    assert task.job_flow_id == (
        "{{ ti.xcom_pull(" "task_ids='create_emr_cluster', " "key='return_value'" ") }}"
    )

    assert task.trigger_rule == TriggerRule.ALL_DONE


# ============================================================
# CREATE EMR TASK CHAIN
# ============================================================


def test_create_emr_tasks() -> None:
    """Test complete EMR orchestration chain."""

    with DAG(
        dag_id="test_emr_tasks",
        schedule=None,
        start_date=None,
        catchup=False,
    ):
        tasks = create_emr_tasks()

    assert len(tasks) == 5

    create_cluster = tasks[0]
    wait_cluster = tasks[1]
    transform_step = tasks[2]
    wait_transform = tasks[3]
    terminate_cluster = tasks[4]

    assert create_cluster.task_id == "create_emr_cluster"
    assert wait_cluster.task_id == "wait_for_emr_cluster"
    assert transform_step.task_id == "add_transform_step"
    assert wait_transform.task_id == "wait_for_transform_step"
    assert terminate_cluster.task_id == "terminate_emr_cluster"

    # --------------------------------------------------------
    # create -> wait
    # --------------------------------------------------------

    assert create_cluster.downstream_task_ids == {
        "wait_for_emr_cluster",
    }

    # --------------------------------------------------------
    # wait -> transform
    # --------------------------------------------------------

    assert wait_cluster.downstream_task_ids == {
        "add_transform_step",
    }

    # --------------------------------------------------------
    # transform -> wait
    # --------------------------------------------------------

    assert transform_step.downstream_task_ids == {
        "wait_for_transform_step",
    }

    # --------------------------------------------------------
    # wait -> terminate
    # --------------------------------------------------------

    assert wait_transform.downstream_task_ids == {
        "terminate_emr_cluster",
    }

    # --------------------------------------------------------
    # terminate -> nothing
    # --------------------------------------------------------

    assert terminate_cluster.downstream_task_ids == set()
