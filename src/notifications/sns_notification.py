from __future__ import annotations

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config.config import CONFIG
from src.utils.logger import Logger

logger = Logger.get_logger(
    "sns_notification",
    "sns_notification.log",
)


class SNSNotification:
    """AWS SNS notification utility."""

    def __init__(self) -> None:
        """Initialize SNS client and logger."""

        try:
            aws_config = CONFIG["aws"]
            sns_config = CONFIG["sns"]

            self.region = str(aws_config["region"])
            self.topic_arn = str(sns_config["topic_arn"])

            self.client = boto3.client(
                "sns",
                region_name=self.region,
            )

            logger.info(
                "SNS client initialized successfully. region=%s",
                self.region,
            )

        except (BotoCoreError, ClientError) as exc:
            logger.exception(
                "Failed to initialize SNS client.",
            )

            raise RuntimeError("Failed to initialize SNS client.") from exc

    def publish(
        self,
        subject: str,
        message: str,
    ) -> str:
        """
        Publish a notification to the configured SNS topic.

        Args:
            subject: Notification subject.
            message: Notification message.

        Returns:
            SNS message ID.

        Raises:
            ValueError: If configuration or input is invalid.
            RuntimeError: If SNS publishing fails.
        """

        if not self.topic_arn:
            raise ValueError("SNS topic ARN is not configured.")

        if not subject.strip():
            raise ValueError("SNS notification subject cannot be empty.")

        if not message.strip():
            raise ValueError("SNS notification message cannot be empty.")

        logger.info(
            "Publishing SNS notification. subject=%s",
            subject,
        )

        try:
            response = self.client.publish(
                TopicArn=self.topic_arn,
                Subject=subject,
                Message=message,
            )

            message_id = response["MessageId"]

            logger.info(
                "SNS notification published successfully. " "message_id=%s",
                message_id,
            )

            return str(message_id)

        except (ClientError, BotoCoreError) as exc:
            logger.exception(
                "Failed to publish SNS notification. subject=%s",
                subject,
            )

            raise RuntimeError("Failed to publish SNS notification.") from exc
