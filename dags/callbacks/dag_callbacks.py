import logging

from airflow.models.taskinstance import TaskInstance
from airflow.sdk import Context
from dag_config import (
    DAG_ID,
    NOTIFICATION_ON_FAILURE,
    NOTIFICATION_ON_SUCCESS,
)
from notifications.email_template import EmailTemplate
from notifications.sns_notification import SNSNotification

logger = logging.getLogger(__name__)


def _get_execution_date(context: Context) -> str:
    """Get execution date from Airflow context."""

    logical_date = context.get("logical_date")

    if logical_date is None:
        return "Unknown"

    return logical_date.isoformat()


def _get_task_instance(
    context: Context,
) -> TaskInstance | None:
    """Get TaskInstance from Airflow context."""

    task_instance = context.get("ti")

    if isinstance(task_instance, TaskInstance):
        return task_instance

    return None


def dag_success_callback(
    context: Context,
) -> None:
    """Send notification when the DAG succeeds."""

    if not NOTIFICATION_ON_SUCCESS:
        logger.info("Success notification disabled.")
        return

    execution_date = _get_execution_date(context)

    task_instance = _get_task_instance(context)

    record_count = 0

    if task_instance is not None:
        try:
            extract_result = task_instance.xcom_pull(task_ids="extract")

            if isinstance(extract_result, dict):
                record_count = int(extract_result.get("record_count", 0) or 0)

        except (TypeError, ValueError):
            record_count = 0

    subject, message = EmailTemplate.pipeline_success(
        pipeline_name=DAG_ID,
        execution_date=execution_date,
        record_count=record_count,
    )

    try:
        notification = SNSNotification()

        notification.publish(
            subject=subject,
            message=message,
        )

        logger.info("DAG success notification sent.")

    except Exception:
        logger.exception("Failed to send DAG success notification.")


def dag_failure_callback(
    context: Context,
) -> None:
    """Send notification when the DAG fails."""

    if not NOTIFICATION_ON_FAILURE:
        logger.info("Failure notification disabled.")
        return

    execution_date = _get_execution_date(context)

    task_instance = _get_task_instance(context)

    error_message = "Unknown DAG failure"

    if task_instance is not None:
        exception = context.get("exception")

        if exception is not None:
            error_message = str(exception)

            logger.error(
                "DAG failed. task_id=%s",
                task_instance.task_id,
            )

    subject, message = EmailTemplate.pipeline_failure(
        pipeline_name=DAG_ID,
        execution_date=execution_date,
        error_message=error_message,
    )

    try:
        notification = SNSNotification()

        notification.publish(
            subject=subject,
            message=message,
        )

        logger.info("DAG failure notification sent.")

    except Exception:
        logger.exception("Failed to send DAG failure notification.")
