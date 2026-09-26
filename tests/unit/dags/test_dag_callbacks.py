from __future__ import annotations

from unittest.mock import MagicMock, patch

import pendulum
from airflow.models.taskinstance import TaskInstance
from airflow.sdk import Context

from src.callbacks.dag_callbacks import (
    _get_execution_date,
    _get_task_instance,
    dag_failure_callback,
    dag_success_callback,
)

# ============================================================
# TEST CONSTANTS
# ============================================================


EXECUTION_DATE = "2026-09-04T09:00:00+00:00"

SUCCESS_SUBJECT = "[SUCCESS] crypto_etl_pipeline"
SUCCESS_MESSAGE = "Pipeline completed successfully."

FAILURE_SUBJECT = "[FAILED] crypto_etl_pipeline"
FAILURE_MESSAGE = "Pipeline failed."


# Airflow Context requires pendulum.DateTime for logical_date.
LOGICAL_DATE = pendulum.datetime(
    2026,
    9,
    4,
    9,
    0,
    0,
    tz="UTC",
)


# ============================================================
# EXECUTION DATE
# ============================================================


def test_get_execution_date_success() -> None:
    """Return logical date as ISO string."""

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    result = _get_execution_date(context)

    assert result == EXECUTION_DATE


def test_get_execution_date_missing() -> None:
    """Return Unknown when logical date is missing."""

    context: Context = {}

    result = _get_execution_date(context)

    assert result == "Unknown"


# ============================================================
# TASK INSTANCE
# ============================================================


def test_get_task_instance_success() -> None:
    """Return TaskInstance when present in context."""

    task_instance = object.__new__(TaskInstance)

    context: Context = {
        "ti": task_instance,
    }

    result = _get_task_instance(context)

    assert result is task_instance


def test_get_task_instance_missing() -> None:
    """Return None when TaskInstance is missing."""

    context: Context = {}

    result = _get_task_instance(context)

    assert result is None


def test_get_task_instance_invalid_object() -> None:
    """Return None when context contains an invalid object."""

    context: Context = {
        "ti": MagicMock(),
    }

    result = _get_task_instance(context)

    assert result is None


# ============================================================
# SUCCESS CALLBACK
# ============================================================


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_success_callback(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Send success notification with record count."""

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = {
        "s3_key": (
            "raw_data/crypto_market/"
            "year=2026/month=09/day=04/"
            "run_time=090000.ndjson"
        ),
        "record_count": 100,
    }

    mock_get_task_instance.return_value = task_instance

    mock_email_template.pipeline_success.return_value = (
        SUCCESS_SUBJECT,
        SUCCESS_MESSAGE,
    )

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    dag_success_callback(context)

    task_instance.xcom_pull.assert_called_once_with(
        task_ids="extract",
    )

    mock_email_template.pipeline_success.assert_called_once_with(
        pipeline_name="crypto_etl_pipeline",
        execution_date=EXECUTION_DATE,
        record_count=100,
    )

    mock_sns.return_value.publish.assert_called_once_with(
        subject=SUCCESS_SUBJECT,
        message=SUCCESS_MESSAGE,
    )


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_success_callback_without_task_instance(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Use zero record count when TaskInstance is unavailable."""

    mock_get_task_instance.return_value = None

    mock_email_template.pipeline_success.return_value = (
        SUCCESS_SUBJECT,
        SUCCESS_MESSAGE,
    )

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    dag_success_callback(context)

    mock_email_template.pipeline_success.assert_called_once_with(
        pipeline_name="crypto_etl_pipeline",
        execution_date=EXECUTION_DATE,
        record_count=0,
    )

    mock_sns.return_value.publish.assert_called_once_with(
        subject=SUCCESS_SUBJECT,
        message=SUCCESS_MESSAGE,
    )


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_success_callback_empty_xcom(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Use zero record count when extract XCom is empty."""

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = None

    mock_get_task_instance.return_value = task_instance

    mock_email_template.pipeline_success.return_value = (
        SUCCESS_SUBJECT,
        SUCCESS_MESSAGE,
    )

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    dag_success_callback(context)

    mock_email_template.pipeline_success.assert_called_once_with(
        pipeline_name="crypto_etl_pipeline",
        execution_date=EXECUTION_DATE,
        record_count=0,
    )

    mock_sns.return_value.publish.assert_called_once_with(
        subject=SUCCESS_SUBJECT,
        message=SUCCESS_MESSAGE,
    )


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_success_callback_invalid_xcom(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Use zero record count when extract XCom is invalid."""

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = "invalid-result"

    mock_get_task_instance.return_value = task_instance

    mock_email_template.pipeline_success.return_value = (
        SUCCESS_SUBJECT,
        SUCCESS_MESSAGE,
    )

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    dag_success_callback(context)

    mock_email_template.pipeline_success.assert_called_once_with(
        pipeline_name="crypto_etl_pipeline",
        execution_date=EXECUTION_DATE,
        record_count=0,
    )

    mock_sns.return_value.publish.assert_called_once_with(
        subject=SUCCESS_SUBJECT,
        message=SUCCESS_MESSAGE,
    )


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_success_callback_invalid_record_count(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Use zero when record_count cannot be converted to int."""

    task_instance = MagicMock()

    task_instance.xcom_pull.return_value = {
        "s3_key": "raw_data/test.ndjson",
        "record_count": "invalid",
    }

    mock_get_task_instance.return_value = task_instance

    mock_email_template.pipeline_success.return_value = (
        SUCCESS_SUBJECT,
        SUCCESS_MESSAGE,
    )

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    dag_success_callback(context)

    mock_email_template.pipeline_success.assert_called_once_with(
        pipeline_name="crypto_etl_pipeline",
        execution_date=EXECUTION_DATE,
        record_count=0,
    )

    mock_sns.return_value.publish.assert_called_once_with(
        subject=SUCCESS_SUBJECT,
        message=SUCCESS_MESSAGE,
    )


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_success_callback_notification_disabled(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Do nothing when success notification is disabled."""

    mock_get_task_instance.return_value = None

    with patch(
        "src.callbacks.dag_callbacks.NOTIFICATION_ON_SUCCESS",
        False,
    ):
        context: Context = {}
        dag_success_callback(context)

    mock_email_template.pipeline_success.assert_not_called()

    mock_sns.assert_not_called()


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_success_callback_sns_failure(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Do not raise when SNS publishing fails."""

    mock_get_task_instance.return_value = None

    mock_email_template.pipeline_success.return_value = (
        SUCCESS_SUBJECT,
        SUCCESS_MESSAGE,
    )

    mock_sns.return_value.publish.side_effect = RuntimeError("SNS failed")

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    dag_success_callback(context)

    mock_sns.return_value.publish.assert_called_once_with(
        subject=SUCCESS_SUBJECT,
        message=SUCCESS_MESSAGE,
    )


# ============================================================
# FAILURE CALLBACK
# ============================================================


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_failure_callback(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Send failure notification with exception message."""

    task_instance = MagicMock()

    mock_get_task_instance.return_value = task_instance

    mock_email_template.pipeline_failure.return_value = (
        FAILURE_SUBJECT,
        FAILURE_MESSAGE,
    )

    context: Context = {
        "logical_date": LOGICAL_DATE,
        "exception": RuntimeError("DAG failed"),
    }

    dag_failure_callback(context)

    mock_email_template.pipeline_failure.assert_called_once_with(
        pipeline_name="crypto_etl_pipeline",
        execution_date=EXECUTION_DATE,
        error_message="DAG failed",
    )

    mock_sns.return_value.publish.assert_called_once_with(
        subject=FAILURE_SUBJECT,
        message=FAILURE_MESSAGE,
    )


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_failure_callback_without_exception(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Use default error message when exception is missing."""

    mock_get_task_instance.return_value = MagicMock()

    mock_email_template.pipeline_failure.return_value = (
        FAILURE_SUBJECT,
        FAILURE_MESSAGE,
    )

    context: Context = {
        "logical_date": LOGICAL_DATE,
    }

    dag_failure_callback(context)

    mock_email_template.pipeline_failure.assert_called_once_with(
        pipeline_name="crypto_etl_pipeline",
        execution_date=EXECUTION_DATE,
        error_message="Unknown DAG failure",
    )

    mock_sns.return_value.publish.assert_called_once_with(
        subject=FAILURE_SUBJECT,
        message=FAILURE_MESSAGE,
    )


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
def test_dag_failure_callback_notification_disabled(
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Do nothing when failure notification is disabled."""

    with patch(
        "src.callbacks.dag_callbacks.NOTIFICATION_ON_FAILURE",
        False,
    ):
        context: Context = {}
        dag_failure_callback(context)

    mock_email_template.pipeline_failure.assert_not_called()

    mock_sns.assert_not_called()


@patch("src.callbacks.dag_callbacks.SNSNotification")
@patch("src.callbacks.dag_callbacks.EmailTemplate")
@patch("src.callbacks.dag_callbacks._get_task_instance")
def test_dag_failure_callback_sns_failure(
    mock_get_task_instance: MagicMock,
    mock_email_template: MagicMock,
    mock_sns: MagicMock,
) -> None:
    """Do not raise when SNS publishing fails."""

    mock_get_task_instance.return_value = MagicMock()

    mock_email_template.pipeline_failure.return_value = (
        FAILURE_SUBJECT,
        FAILURE_MESSAGE,
    )

    mock_sns.return_value.publish.side_effect = RuntimeError("SNS failed")

    context: Context = {
        "logical_date": LOGICAL_DATE,
        "exception": RuntimeError("DAG failed"),
    }

    dag_failure_callback(context)

    mock_sns.return_value.publish.assert_called_once_with(
        subject=FAILURE_SUBJECT,
        message=FAILURE_MESSAGE,
    )
