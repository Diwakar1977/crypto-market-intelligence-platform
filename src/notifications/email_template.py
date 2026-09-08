class EmailTemplate:
    """Build standardized email notifications for the ETL pipeline."""

    @staticmethod
    def pipeline_success(
        pipeline_name: str,
        execution_date: str,
        record_count: int,
    ) -> tuple[str, str]:
        """Build a successful pipeline notification."""

        subject = f"[SUCCESS] {pipeline_name}"

        body = (
            "Crypto ETL Pipeline Notification\n"
            "=================================\n\n"
            f"Pipeline       : {pipeline_name}\n"
            "Status         : SUCCESS\n"
            f"Execution Date : {execution_date}\n"
            f"Records        : {record_count}\n\n"
            "The ETL pipeline completed successfully."
        )

        return subject, body

    @staticmethod
    def pipeline_failure(
        pipeline_name: str,
        execution_date: str,
        error_message: str,
    ) -> tuple[str, str]:
        """Build a failed pipeline notification."""

        subject = f"[FAILED] {pipeline_name}"

        body = (
            "Crypto ETL Pipeline Notification\n"
            "=================================\n\n"
            f"Pipeline       : {pipeline_name}\n"
            "Status         : FAILED\n"
            f"Execution Date : {execution_date}\n"
            f"Error          : {error_message}\n\n"
            "Please check the Airflow logs for more details."
        )

        return subject, body