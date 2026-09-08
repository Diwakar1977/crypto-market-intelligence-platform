from pathlib import Path

from src.notifications.email_template import EmailTemplate
from src.utils.logger import Logger


def test_pipeline_success_email_template():
    """Test successful pipeline email template."""

    # Arrange
    pipeline_name = "CryptoFlow-ETL"
    execution_date = "2026-08-21"
    record_count = 250

    logger = Logger.get_logger(
        "test_email_template_success",
        "test_email_template_success.log",
    )

    try:
        # Act
        subject, body = EmailTemplate.pipeline_success(
            pipeline_name=pipeline_name,
            execution_date=execution_date,
            record_count=record_count,
        )

        # Log generated email for test evidence
        logger.info("EMAIL SUBJECT: %s", subject)
        logger.info("EMAIL BODY:\n%s", body)

        # Assert
        assert subject == "[SUCCESS] CryptoFlow-ETL"

        assert "Crypto ETL Pipeline Notification" in body
        assert "Pipeline       : CryptoFlow-ETL" in body
        assert "Status         : SUCCESS" in body
        assert "Execution Date : 2026-08-21" in body
        assert "Records        : 250" in body
        assert "The ETL pipeline completed successfully." in body

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Verify log file
        project_root = Path(__file__).resolve().parents[3]
        log_file = (
            project_root
            / "logs"
            / "test_email_template_success.log"
        )

        assert log_file.exists()

        log_context = log_file.read_text(encoding="utf-8")

        assert subject in log_context
        assert "EMAIL BODY:" in log_context
        assert "Status         : SUCCESS" in log_context
        assert "Records        : 250" in log_context

    finally:
        for handler in logger.handlers:
            handler.close()

        logger.handlers.clear()


def test_pipeline_failure_email_template():
    """Test failed pipeline email template."""

    # Arrange
    pipeline_name = "CryptoFlow-ETL"
    execution_date = "2026-08-21"
    error_message = "EMR Spark step failed"

    logger = Logger.get_logger(
        "test_email_template_failure",
        "test_email_template_failure.log",
    )

    try:
        # Act
        subject, body = EmailTemplate.pipeline_failure(
            pipeline_name=pipeline_name,
            execution_date=execution_date,
            error_message=error_message,
        )

        # Log generated email for test evidence
        logger.info("EMAIL SUBJECT: %s", subject)
        logger.info("EMAIL BODY:\n%s", body)

        # Assert
        assert subject == "[FAILED] CryptoFlow-ETL"

        assert "Crypto ETL Pipeline Notification" in body
        assert "Pipeline       : CryptoFlow-ETL" in body
        assert "Status         : FAILED" in body
        assert "Execution Date : 2026-08-21" in body
        assert "Error          : EMR Spark step failed" in body
        assert "Please check the Airflow logs for more details." in body

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Verify log file
        project_root = Path(__file__).resolve().parents[3]
        log_file = (
            project_root
            / "logs"
            / "test_email_template_failure.log"
        )

        assert log_file.exists()

        log_context = log_file.read_text(encoding="utf-8")

        assert subject in log_context
        assert "EMAIL BODY:" in log_context
        assert "Status         : FAILED" in log_context
        assert error_message in log_context

    finally:
        for handler in logger.handlers:
            handler.close()

        logger.handlers.clear()