from __future__ import annotations

from typing import Any

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

from src.config.config import CONFIG

# ------------------------------------
# CONFIGURATION
# ------------------------------------

AWS_CONFIG = CONFIG["aws"]
EMR_CONFIG = CONFIG["emr"]
S3_CONFIG = CONFIG["s3"]

AWS_REGION = str(AWS_CONFIG["region"])

EMR_CLUSTER_NAME = str(EMR_CONFIG["cluster_name"])
EMR_RELEASE = str(EMR_CONFIG["release"])
EMR_SUBNET_ID = str(EMR_CONFIG["subnet_id"])

EMR_MASTER_INSTANCE_TYPE = str(EMR_CONFIG["master_instance_type"])
EMR_CORE_INSTANCE_TYPE = str(EMR_CONFIG["core_instance_type"])

EMR_CORE_INSTANCE_COUNT = int(EMR_CONFIG["core_instance_count"])

EMR_INSTANCE_PROFILE = str(EMR_CONFIG["instance_profile"])

EMR_SERVICE_ROLE = str(EMR_CONFIG["service_role"])

EMR_LOG_URI = str(EMR_CONFIG["log_uri"])

S3_BUCKET = str(S3_CONFIG["bucket"])
S3_RAW_PREFIX = str(S3_CONFIG["raw_prefix"])
S3_PROCESSED_PREFIX = str(S3_CONFIG["processed_prefix"])

EMR_SPARK_ZIP = f"s3://{S3_BUCKET}/src/transform/tranform"

# ------------------------------------
# CREATE EMR CLUSTER
# ------------------------------------


def create_emr_cluster() -> EmrCreateJobFlowOperator:
    """Create an EMR cluster for Spark transformation."""

    job_flow_overrides: dict[str, Any] = {
        "Name": EMR_CLUSTER_NAME,
        "ReleaseLabel": EMR_RELEASE,
        "Applications": [
            {
                "Name": "Spark",
            },
        ],
        "Instances": {
            "Ec2SubnetId": EMR_SUBNET_ID,
            "KeepJobFlowAliveWhenNoSteps": True,
            "TerminationProtected": False,
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                    "InstanceType": EMR_MASTER_INSTANCE_TYPE,
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                    "InstanceType": EMR_CORE_INSTANCE_TYPE,
                    "InstanceCount": EMR_CORE_INSTANCE_COUNT,
                },
            ],
        },
        "JobFlowRole": EMR_INSTANCE_PROFILE,
        "ServiceRole": EMR_SERVICE_ROLE,
        "LogUri": EMR_LOG_URI,
        "VisibleToAllUsers": True,
    }

    return EmrCreateJobFlowOperator(
        task_id="create_emr_cluster",
        job_flow_overrides=job_flow_overrides,
        aws_conn_id="aws_default",
        region_name=AWS_REGION,
    )


# ------------------------------------
# WAIT FOR EMR CLUSTER
# ------------------------------------


def wait_for_emr_cluster() -> EmrJobFlowSensor:
    """Wait until EMR cluster reaches WAITING state."""

    return EmrJobFlowSensor(
        task_id="wait_for_emr_cluster",
        job_flow_id=(
            "{{ ti.xcom_pull("
            "task_ids='create_emr_cluster', "
            "key='return_value'"
            ") }}"
        ),
        target_states=[
            "WAITING",
        ],
        failed_states=[
            "TERMINATED",
            "TERMINATED_WITH_ERRORS",
        ],
        aws_conn_id="aws_default",
        region_name=AWS_REGION,
    )


# ------------------------------------
# ADD TRANSFORM STEP
# ------------------------------------


def add_transform_step() -> EmrAddStepsOperator:
    """Submit the Spark transformation step to EMR."""

    transform_step: dict[str, Any] = {
        "Name": "crypto-market-transform",
        "ActionOnFailure": "CANCEL_AND_WAIT",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "bash",
                "-c",
                (
                    "set -euo pipefail; "
                    "rm -rf /tmp/crypto_etl_spark; "
                    "mkdir -p /tmp/crypto_etl_spark; "
                    f"aws s3 cp {EMR_SPARK_ZIP} "
                    "/tmp/crypto_etl_spark.zip; "
                    "unzip -q /tmp/crypto_etl_spark.zip "
                    "-d /tmp/crypto_etl_spark; "
                    "spark-submit "
                    "--deploy-mode cluster "
                    "/tmp/crypto_etl_spark/src/transform/transform_job.py "
                    f"--input s3://{S3_BUCKET}/{S3_RAW_PREFIX} "
                    f"--output s3://{S3_BUCKET}/{S3_PROCESSED_PREFIX}"
                ),
            ],
        },
    }

    return EmrAddStepsOperator(
        task_id="add_transform_step",
        job_flow_id=(
            "{{ ti.xcom_pull("
            "task_ids='create_emr_cluster', "
            "key='return_value'"
            ") }}"
        ),
        steps=[
            transform_step,
        ],
        aws_conn_id="aws_default",
        region_name=AWS_REGION,
    )


# ------------------------------------
# WAIT FOR TRANSFORM STEP
# ------------------------------------


def wait_for_transform_step() -> EmrStepSensor:
    """Wait until the Spark transformation step completes."""

    return EmrStepSensor(
        task_id="wait_for_transform_step",
        job_flow_id=(
            "{{ ti.xcom_pull("
            "task_ids='create_emr_cluster', "
            "key='return_value'"
            ") }}"
        ),
        step_id=(
            "{{ ti.xcom_pull("
            "task_ids='add_transform_step', "
            "key='return_value'"
            ")[0] }}"
        ),
        target_states=[
            "COMPLETED",
        ],
        failed_states=[
            "CANCEL_PENDING",
            "CANCELLED",
            "FAILED",
            "INTERRUPTED",
        ],
        aws_conn_id="aws_default",
        region_name=AWS_REGION,
    )


# ------------------------------------
# TERMINATE EMR CLUSTER
# ------------------------------------


def terminate_emr_cluster() -> EmrTerminateJobFlowOperator:
    """
    Terminate the EMR cluster after transformation.

    ALL_DONE ensures the cluster is terminated even when
    the transformation step fails.
    """

    return EmrTerminateJobFlowOperator(
        task_id="terminate_emr_cluster",
        job_flow_id=(
            "{{ ti.xcom_pull("
            "task_ids='create_emr_cluster', "
            "key='return_value'"
            ") }}"
        ),
        aws_conn_id="aws_default",
        region_name=AWS_REGION,
        trigger_rule=TriggerRule.ALL_DONE,
    )


# ------------------------------------
# CREATE EMR TASKS
# ------------------------------------


def create_emr_tasks() -> list[Any]:
    """Create and chain all EMR orchestration tasks."""

    create_cluster = create_emr_cluster()

    wait_cluster = wait_for_emr_cluster()

    transform_step = add_transform_step()

    wait_transform = wait_for_transform_step()

    terminate_cluster = terminate_emr_cluster()

    (
        create_cluster
        >> wait_cluster
        >> transform_step
        >> wait_transform
        >> terminate_cluster
    )

    return [
        create_cluster,
        wait_cluster,
        transform_step,
        wait_transform,
        terminate_cluster,
    ]
