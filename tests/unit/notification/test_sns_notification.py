from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from dags.notifications.sns_notification import SNSNotification

# ============================================================
# CONSTANTS
# ============================================================

TOPIC_ARN = "arn:aws:sns:ap-south-1:" "123456789012:crypto-etl"

AWS_CONFIG = {
    "aws": {
        "region": "ap-south-1",
    },
    "sns": {
        "topic_arn": TOPIC_ARN,
    },
}


# ============================================================
# FIXTURE
# ============================================================


@pytest.fixture
def mock_boto_client() -> MagicMock:
    """Return a mocked boto3 SNS client."""
    return MagicMock()


# ============================================================
# INITIALIZATION
# ============================================================


def test_sns_client_initialization(
    mock_boto_client: MagicMock,
) -> None:
    """Test SNS client initialization."""

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            return_value=mock_boto_client,
        ) as mock_client,
        patch(
            "dags.notifications.sns_notification.CONFIG",
            AWS_CONFIG,
        ),
    ):
        sns = SNSNotification()

    mock_client.assert_called_once_with(
        "sns",
        region_name="ap-south-1",
    )

    assert sns.region == "ap-south-1"
    assert sns.topic_arn == TOPIC_ARN
    assert sns.client is mock_boto_client


# ============================================================
# PUBLISH SUCCESS
# ============================================================


def test_publish_success(
    mock_boto_client: MagicMock,
) -> None:
    """Test successful SNS publishing."""

    mock_boto_client.publish.return_value = {
        "MessageId": "test-message-id",
    }

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            return_value=mock_boto_client,
        ),
        patch(
            "dags.notifications.sns_notification.CONFIG",
            AWS_CONFIG,
        ),
    ):
        sns = SNSNotification()

        message_id = sns.publish(
            subject="ETL Pipeline Success",
            message=("Crypto ETL pipeline completed successfully."),
        )

    assert message_id == "test-message-id"

    mock_boto_client.publish.assert_called_once_with(
        TopicArn=TOPIC_ARN,
        Subject="ETL Pipeline Success",
        Message=("Crypto ETL pipeline completed successfully."),
    )


# ============================================================
# VALIDATION
# ============================================================


def test_publish_without_topic_arn(
    mock_boto_client: MagicMock,
) -> None:
    """Test publishing when SNS topic ARN is missing."""

    config = {
        "aws": {
            "region": "ap-south-1",
        },
        "sns": {
            "topic_arn": "",
        },
    }

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            return_value=mock_boto_client,
        ),
        patch(
            "dags.notifications.sns_notification.CONFIG",
            config,
        ),
    ):
        sns = SNSNotification()

        with pytest.raises(
            ValueError,
            match="SNS topic ARN is not configured",
        ):
            sns.publish(
                subject="Test",
                message="Test message",
            )

    mock_boto_client.publish.assert_not_called()


@pytest.mark.parametrize(
    "subject",
    [
        "",
        "   ",
    ],
)
def test_publish_empty_subject(
    mock_boto_client: MagicMock,
    subject: str,
) -> None:
    """Test publishing with an empty subject."""

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            return_value=mock_boto_client,
        ),
        patch(
            "dags.notifications.sns_notification.CONFIG",
            AWS_CONFIG,
        ),
    ):
        sns = SNSNotification()

        with pytest.raises(
            ValueError,
            match="SNS notification subject cannot be empty",
        ):
            sns.publish(
                subject=subject,
                message="Test message",
            )

    mock_boto_client.publish.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
    ],
)
def test_publish_empty_message(
    mock_boto_client: MagicMock,
    message: str,
) -> None:
    """Test publishing with an empty message."""

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            return_value=mock_boto_client,
        ),
        patch(
            "dags.notifications.sns_notification.CONFIG",
            AWS_CONFIG,
        ),
    ):
        sns = SNSNotification()

        with pytest.raises(
            ValueError,
            match="SNS notification message cannot be empty",
        ):
            sns.publish(
                subject="Test",
                message=message,
            )

    mock_boto_client.publish.assert_not_called()


# ============================================================
# AWS ERRORS
# ============================================================


def test_publish_client_error(
    mock_boto_client: MagicMock,
) -> None:
    """Test SNS ClientError is converted to RuntimeError."""

    mock_boto_client.publish.side_effect = ClientError(
        {
            "Error": {
                "Code": "InternalError",
                "Message": "SNS service error",
            },
        },
        "Publish",
    )

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            return_value=mock_boto_client,
        ),
        patch(
            "dags.notifications.sns_notification.CONFIG",
            AWS_CONFIG,
        ),
    ):
        sns = SNSNotification()

        with pytest.raises(
            RuntimeError,
            match="Failed to publish SNS notification",
        ):
            sns.publish(
                subject="Test",
                message="Test message",
            )

    mock_boto_client.publish.assert_called_once_with(
        TopicArn=TOPIC_ARN,
        Subject="Test",
        Message="Test message",
    )


def test_publish_botocore_error(
    mock_boto_client: MagicMock,
) -> None:
    """Test BotoCoreError is converted to RuntimeError."""

    mock_boto_client.publish.side_effect = BotoCoreError()

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            return_value=mock_boto_client,
        ),
        patch(
            "dags.notifications.sns_notification.CONFIG",
            AWS_CONFIG,
        ),
    ):
        sns = SNSNotification()

        with pytest.raises(
            RuntimeError,
            match="Failed to publish SNS notification",
        ):
            sns.publish(
                subject="Test",
                message="Test message",
            )


# ============================================================
# INITIALIZATION ERROR
# ============================================================


def test_sns_client_initialization_error() -> None:
    """Test SNS client initialization failure."""

    with (
        patch(
            "dags.notifications.sns_notification.boto3.client",
            side_effect=ClientError(
                {
                    "Error": {
                        "Code": "AccessDenied",
                        "Message": "Access denied",
                    },
                },
                "CreateClient",
            ),
        ),
        patch(
            "dags.notifications.sns_notification.CONFIG",
            AWS_CONFIG,
        ),
        pytest.raises(
            RuntimeError,
            match="Failed to initialize SNS client",
        ),
    ):
        SNSNotification()
