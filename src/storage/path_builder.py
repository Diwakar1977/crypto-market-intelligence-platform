from __future__ import annotations

from datetime import datetime, timezone

from config.config import CONFIG


class PathBuilder:
    """Build standardized paths for the Crypto ETL Pipeline."""

    @staticmethod
    def _get_utc_now() -> datetime:
        """Return the current UTC timestamp."""

        return datetime.now(timezone.utc)

    @staticmethod
    def build_raw_path(
        dataset_name: str,
        run_time: datetime | None = None,
    ) -> str:
        """Build the S3 key for raw NDJSON data."""

        run_time = run_time or PathBuilder._get_utc_now()

        s3_config = CONFIG["s3"]
        raw_prefix = str(s3_config["raw_prefix"]).rstrip("/")

        return (
            f"{raw_prefix}/"
            f"{dataset_name}/"
            f"year={run_time:%Y}/"
            f"month={run_time:%m}/"
            f"day={run_time:%d}/"
            f"run_time={run_time:%H%M%S}.ndjson"
        )

    @staticmethod
    def build_processed_path(
        dataset_name: str,
        run_time: datetime | None = None,
    ) -> str:
        """Build the S3 prefix for processed Parquet data."""

        run_time = run_time or PathBuilder._get_utc_now()

        s3_config = CONFIG["s3"]
        processed_prefix = str(s3_config["processed_prefix"]).rstrip("/")

        return (
            f"{processed_prefix}/"
            f"{dataset_name}/"
            f"year={run_time:%Y}/"
            f"month={run_time:%m}/"
            f"day={run_time:%d}/"
            f"time={run_time:%H%M%S}/"
        )
