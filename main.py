from __future__ import annotations

from datetime import datetime, timezone

from pyspark.sql import SparkSession

from config.config import CONFIG
from dags.notifications.email_template import EmailTemplate
from dags.notifications.sns_notification import SNSNotification
from spark.spark_session import SparkSessionFactory
from src.extract.extract_job import ExtractResult, create_extract_job
from src.load.load_job import LoadResult, create_load_job
from src.load.redshift_storage import RedshiftStorage
from src.transform.transform_job import create_transform_job
from src.utils.logger import Logger

logger = Logger.get_logger(
    "main",
    "main.log",
)


class CryptoETLPipeline:
    """Orchestrate the complete Crypto Market ETL pipeline."""

    def __init__(self) -> None:
        """Initialize pipeline configuration and runtime resources."""

        self.config = CONFIG

        self.application_config = self.config["application"]
        self.aws_config = self.config["aws"]
        self.s3_config = self.config["s3"]
        self.redshift_config = self.config["redshift"]

        self.pipeline_name = str(self.application_config["pipeline_name"])

        self.spark: SparkSession | None = None

        self.redshift_storage: RedshiftStorage | None = None

        self.extract_result: ExtractResult | None = None

    def run(self) -> LoadResult:
        """Run the complete Crypto Market ETL pipeline."""

        Logger.log_banner(
            logger,
            "CRYPTO ETL PIPELINE STARTED",
        )

        execution_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        try:
            # ==========================================================
            # STEP 1 - EXTRACT
            # ==========================================================

            Logger.log_banner(
                logger,
                "STEP 1 - EXTRACT",
            )

            extract_job = create_extract_job()

            extract_result = extract_job.run()

            self.extract_result = extract_result

            logger.info("Extraction completed successfully.")

            logger.info(
                "Raw S3 key: %s",
                extract_result.s3_key,
            )

            logger.info(
                "Extracted record count: %d",
                extract_result.record_count,
            )

            raw_bucket = str(self.s3_config["bucket"])

            raw_spark_path = f"s3a://{raw_bucket}/{extract_result.s3_key}"

            logger.info(
                "Raw Spark input path: %s",
                raw_spark_path,
            )

            # ==========================================================
            # STEP 2 - CREATE SPARK SESSION
            # ==========================================================

            Logger.log_banner(
                logger,
                "STEP 2 - CREATE SPARK SESSION",
            )

            self.spark = SparkSessionFactory.create()

            logger.info("Spark session created successfully.")

            # ==========================================================
            # STEP 3 - TRANSFORM
            # ==========================================================

            Logger.log_banner(
                logger,
                "STEP 3 - TRANSFORM",
            )

            if self.spark is None:
                raise RuntimeError("Spark session was not initialized.")

            transform_job = create_transform_job(
                spark=self.spark,
            )

            processed_spark_path = transform_job.run(
                input_path=raw_spark_path,
            )

            logger.info("Transform completed successfully.")

            logger.info(
                "Processed Spark path: %s",
                processed_spark_path,
            )

            # ==========================================================
            # STEP 4 - CONVERT S3A PATH TO S3 PATH
            # ==========================================================

            Logger.log_banner(
                logger,
                "STEP 4 - PREPARE REDSHIFT S3 PATH",
            )

            processed_s3_path = self._to_redshift_s3_path(
                processed_spark_path,
            )

            logger.info(
                "Processed Redshift COPY path: %s",
                processed_s3_path,
            )

            # ==========================================================
            # STEP 5 - INITIALIZE REDSHIFT
            # ==========================================================

            Logger.log_banner(
                logger,
                "STEP 5 - INITIALIZE REDSHIFT",
            )

            self.redshift_storage = RedshiftStorage(
                host=str(self.redshift_config["host"]),
                port=int(self.redshift_config["port"]),
                database=str(self.redshift_config["database"]),
                aws_region=str(self.aws_config["region"]),
                workgroup=str(self.redshift_config["workgroup"]),
            )

            logger.info("Redshift storage initialized successfully.")

            # ==========================================================
            # STEP 6 - LOAD
            # ==========================================================

            Logger.log_banner(
                logger,
                "STEP 6 - LOAD",
            )

            load_job = create_load_job(
                spark=self.spark,
                redshift_storage=self.redshift_storage,
                redshift_schema=str(self.redshift_config["schema"]),
                redshift_table=str(self.redshift_config["table"]),
                processed_spark_path=processed_spark_path,
                processed_s3_path=processed_s3_path,
                redshift_iam_role=str(self.redshift_config["iam_role"]),
            )

            load_result = load_job.run()

            logger.info("Redshift load completed successfully.")

            logger.info(
                "Target table: %s.%s",
                load_result.schema_name,
                load_result.table_name,
            )

            # ==========================================================
            # STEP 7 - SUCCESS NOTIFICATION
            # ==========================================================

            Logger.log_banner(
                logger,
                "STEP 7 - SUCCESS NOTIFICATION",
            )

            self._send_success_notification(
                execution_date=execution_date,
                record_count=extract_result.record_count,
            )

            # ==========================================================
            # PIPELINE SUCCESS
            # ==========================================================

            Logger.log_banner(
                logger,
                "CRYPTO ETL PIPELINE COMPLETED SUCCESSFULLY",
            )

            return load_result

        except Exception as exc:
            logger.exception("Crypto ETL pipeline failed.")

            # ==========================================================
            # FAILURE NOTIFICATION
            # ==========================================================

            self._send_failure_notification(
                execution_date=execution_date,
                error_message=str(exc),
            )

            Logger.log_banner(
                logger,
                "CRYPTO ETL PIPELINE FAILED",
            )

            raise

        finally:
            # ==========================================================
            # CLEANUP
            # ==========================================================

            self._close_redshift()

            self._stop_spark()

            Logger.log_banner(
                logger,
                "CRYPTO ETL PIPELINE FINISHED",
            )

    @staticmethod
    def _to_redshift_s3_path(
        processed_spark_path: str,
    ) -> str:
        """Convert a Spark S3A path to a standard S3 path."""

        if not processed_spark_path.strip():
            raise ValueError("Processed Spark path cannot be empty.")

        if processed_spark_path.startswith("s3a://"):
            return processed_spark_path.replace(
                "s3a://",
                "s3://",
                1,
            )

        if processed_spark_path.startswith("s3://"):
            return processed_spark_path

        raise ValueError("Processed path must start with " "'s3a://' or 's3://'.")

    def _send_success_notification(
        self,
        execution_date: str,
        record_count: int,
    ) -> None:
        """Send a successful pipeline notification through SNS."""

        sns_config = self.config.get("sns", {})

        topic_arn = str(
            sns_config.get(
                "topic_arn",
                "",
            )
            or ""
        ).strip()

        if not topic_arn:
            logger.info(
                "SNS topic ARN is not configured. " "Skipping success notification."
            )

            return

        try:
            subject, message = EmailTemplate.pipeline_success(
                pipeline_name=self.pipeline_name,
                execution_date=execution_date,
                record_count=record_count,
            )

            notifier = SNSNotification()

            message_id = notifier.publish(
                subject=subject,
                message=message,
            )

            logger.info(
                "Success notification sent successfully. " "MessageId: %s",
                message_id,
            )

        except Exception:
            logger.exception("Failed to send success notification.")

    def _send_failure_notification(
        self,
        execution_date: str,
        error_message: str,
    ) -> None:
        """Send a failed pipeline notification through SNS."""

        sns_config = self.config.get("sns", {})

        topic_arn = str(
            sns_config.get(
                "topic_arn",
                "",
            )
            or ""
        ).strip()

        if not topic_arn:
            logger.info(
                "SNS topic ARN is not configured. " "Skipping failure notification."
            )

            return

        try:
            subject, message = EmailTemplate.pipeline_failure(
                pipeline_name=self.pipeline_name,
                execution_date=execution_date,
                error_message=error_message,
            )

            notifier = SNSNotification()

            message_id = notifier.publish(
                subject=subject,
                message=message,
            )

            logger.info(
                "Failure notification sent successfully. " "MessageId: %s",
                message_id,
            )

        except Exception:
            logger.exception("Failed to send failure notification.")

    def _close_redshift(self) -> None:
        """Close Redshift resources."""

        if self.redshift_storage is None:
            return

        try:
            logger.info("Closing Redshift connection.")

            self.redshift_storage.close()

            logger.info("Redshift connection closed successfully.")

        except Exception:
            logger.exception("Failed to close Redshift connection.")

        finally:
            self.redshift_storage = None

    def _stop_spark(self) -> None:
        """Stop the Spark session."""

        if self.spark is None:
            return

        try:
            logger.info("Stopping Spark session.")

            self.spark.stop()

            logger.info("Spark session stopped successfully.")

        except Exception:
            logger.exception("Failed to stop Spark session.")

        finally:
            self.spark = None


def main() -> None:
    """Application entry point."""

    pipeline = CryptoETLPipeline()

    pipeline.run()


if __name__ == "__main__":
    main()
