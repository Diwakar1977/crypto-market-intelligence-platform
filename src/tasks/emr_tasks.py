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

EMR_SECURITY_GROUP_ID = str(EMR_CONFIG["security_group_id"])
EMR_SERVICE_ACCESS_SECURITY_GROUP_ID = str(
    EMR_CONFIG["service_access_security_group_id"]
)

EMR_MASTER_INSTANCE_TYPE = str(EMR_CONFIG["master_instance_type"])
EMR_CORE_INSTANCE_TYPE = str(EMR_CONFIG["core_instance_type"])

EMR_CORE_INSTANCE_COUNT = int(EMR_CONFIG["core_instance_count"])

EMR_INSTANCE_PROFILE = str(EMR_CONFIG["instance_profile"])

EMR_SERVICE_ROLE = str(EMR_CONFIG["service_role"])

EMR_LOG_URI = str(EMR_CONFIG["log_uri"])

S3_BUCKET = str(S3_CONFIG["bucket"])
S3_RAW_PREFIX = str(S3_CONFIG["raw_prefix"])


# ------------------------------------
# CREATE EMR CLUSTER
# ------------------------------------


def create_emr_cluster() -> EmrCreateJobFlowOperator:
    """Create an EMR cluster for Spark transformation."""

    job_flow_overrides: dict[str, Any] = {
        "Name": EMR_CLUSTER_NAME,
        "Tags": [{"Key": "for-use-with-amazon-emr-managed-policies", "Value": "true"}],
        "ReleaseLabel": EMR_RELEASE,
        "Applications": [
            {
                "Name": "Spark",
            },
        ],
        "Instances": {
            "Ec2SubnetId": EMR_SUBNET_ID,
            "EmrManagedMasterSecurityGroup": EMR_SECURITY_GROUP_ID,
            "EmrManagedSlaveSecurityGroup": EMR_SECURITY_GROUP_ID,
            "ServiceAccessSecurityGroup": EMR_SERVICE_ACCESS_SECURITY_GROUP_ID,
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
                    # --------------------------------------------------
                    # 1. Download EMR Python requirements
                    # --------------------------------------------------
                    f"aws s3 cp "
                    f"s3://{S3_BUCKET}/requirements.txt "
                    "/tmp/requirements.txt; "
                    # --------------------------------------------------
                    # 2. Set EMR configuration
                    # --------------------------------------------------
                    "export ENV=emr; "
                    # --------------------------------------------------
                    # 3. Create temporary Python dependency directory
                    # --------------------------------------------------
                    "rm -rf /tmp/python_deps; "
                    "mkdir -p /tmp/python_deps; "
                    # --------------------------------------------------
                    # 4. Install Python dependencies
                    # --------------------------------------------------
                    "/usr/bin/python3.11 -m pip install "
                    "--target /tmp/python_deps "
                    "-r /tmp/requirements.txt; "
                    # --------------------------------------------------
                    # 5. Remove boto3 and botocore from dependency ZIP
                    #
                    # EMR already provides the AWS SDK.
                    # We do not package boto3/botocore inside
                    # dependencies.zip because botocore package-data
                    # cannot be loaded correctly through this ZIP.
                    # --------------------------------------------------
                    "rm -rf /tmp/python_deps/boto3; "
                    "rm -rf /tmp/python_deps/boto3-*.dist-info; "
                    "rm -rf /tmp/python_deps/botocore; "
                    "rm -rf /tmp/python_deps/botocore-*.dist-info; "
                    # --------------------------------------------------
                    # 6. Create dependencies.zip
                    # --------------------------------------------------
                    "rm -f /tmp/dependencies.zip; "
                    "cd /tmp/python_deps; "
                    "zip -r /tmp/dependencies.zip .; "
                    # --------------------------------------------------
                    # 7. DEBUG: Verify boto3/botocore were removed
                    # --------------------------------------------------
                    'echo "===== DEPENDENCY ZIP CHECK ====="; '
                    'echo "boto3 in ZIP:"; '
                    "unzip -l /tmp/dependencies.zip "
                    '| grep -E "^.*boto3/" || true; '
                    'echo "botocore in ZIP:"; '
                    "unzip -l /tmp/dependencies.zip "
                    '| grep -E "^.*botocore/" || true; '
                    'echo "ZIP SIZE:"; '
                    "ls -lh /tmp/dependencies.zip; "
                    'echo "===== END DEPENDENCY ZIP CHECK ====="; '
                    # --------------------------------------------------
                    # 8. Download src/ from S3
                    # --------------------------------------------------
                    "rm -rf /tmp/src; "
                    "rm -f /tmp/src.zip; "
                    "cd /tmp; "
                    f"aws s3 cp "
                    f"s3://{S3_BUCKET}/dags/src/ "
                    "/tmp/src/ "
                    "--recursive; "
                    # --------------------------------------------------
                    # 9. Create src.zip
                    # --------------------------------------------------
                    "cd /tmp; "
                    "zip -r /tmp/src.zip src; "
                    # --------------------------------------------------
                    # 10. Run Spark transformation
                    # --------------------------------------------------
                    "spark-submit "
                    "--deploy-mode cluster "
                    "--conf spark.yarn.appMasterEnv.ENV=emr "
                    "--py-files /tmp/src.zip,/tmp/dependencies.zip "
                    f"s3://{S3_BUCKET}/dags/src/transform/transform_job.py "
                    f"s3a://{S3_BUCKET}/{S3_RAW_PREFIX}"
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
        steps=[transform_step],
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
