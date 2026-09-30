from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import boto3
import pyarrow.parquet as pq

from src.config.config import CONFIG
from src.load.pyarrow_schema_mapper import PyArrowSchemaMapper
from src.load.redshift_storage import RedshiftStorage
from src.utils.logger import Logger

logger = Logger.get_logger(
    "load_job",
    "load_job.log",
)


@dataclass(frozen=True)
class LoadResult:
    """Represent the result of the Redshift load job."""

    schema_name: str
    table_name: str
    source_path: str
    success: bool


class LoadJob:
    """
    Load processed Parquet data from S3 into Redshift.

    Responsibilities:
        1. Find the latest processed Parquet run in S3.
        2. Read the Parquet schema using PyArrow.
        3. Generate Redshift CREATE TABLE SQL.
        4. Create the target table when it does not exist.
        5. COPY only the latest processed run into Redshift.

    This class does not use Spark.
    """

    PARQUET_FORMAT = "PARQUET"

    def __init__(
        self,
        redshift_storage: RedshiftStorage,
        redshift_schema: str,
        redshift_table: str,
        bucket: str,
        processed_prefix: str,
        redshift_iam_role: str,
        aws_region: str,
    ) -> None:
        """Initialize the Redshift load job."""

        self.redshift_storage = redshift_storage
        self.redshift_schema = redshift_schema
        self.redshift_table = redshift_table
        self.bucket = bucket
        self.processed_prefix = processed_prefix.strip("/")
        self.redshift_iam_role = redshift_iam_role
        self.aws_region = aws_region

        self._s3_client = boto3.client(
            "s3",
            region_name=self.aws_region,
        )

        self._validate_configuration()

    # ------------------------------------------------------------------
    # Main Job
    # ------------------------------------------------------------------

    def run(self) -> LoadResult:
        """Execute the complete Redshift load process."""

        Logger.log_banner(
            logger,
            "START REDSHIFT LOAD JOB",
        )

        logger.info(
            "Target table: %s.%s",
            self.redshift_schema,
            self.redshift_table,
        )

        logger.info(
            "Processed S3 prefix: s3://%s/%s/",
            self.bucket,
            self.processed_prefix,
        )

        try:
            # ----------------------------------------------------------
            # STEP 1: FIND LATEST PROCESSED RUN
            # ----------------------------------------------------------

            latest_run_prefix = self._find_latest_processed_run()

            latest_run_s3_path = f"s3://{self.bucket}/{latest_run_prefix}"

            logger.info(
                "Latest processed run detected: %s",
                latest_run_s3_path,
            )

            # ----------------------------------------------------------
            # STEP 2: READ PROCESSED PARQUET SCHEMA
            # ----------------------------------------------------------

            schema = self._read_processed_schema(
                latest_run_prefix,
            )

            # ----------------------------------------------------------
            # STEP 3: GENERATE CREATE TABLE SQL
            # ----------------------------------------------------------

            create_table_sql = self._generate_create_table_sql(
                schema,
            )

            logger.info(
                "Generated Redshift CREATE TABLE SQL:\n%s",
                create_table_sql,
            )

            # ----------------------------------------------------------
            # STEP 4: CREATE REDSHIFT TARGET TABLE
            # ----------------------------------------------------------

            self._create_target_table(
                create_table_sql,
            )

            # ----------------------------------------------------------
            # STEP 5: GENERATE COPY SQL
            # ----------------------------------------------------------

            copy_sql = self._generate_copy_sql(
                latest_run_prefix,
            )

            logger.info(
                "Generated Redshift COPY SQL:\n%s",
                copy_sql,
            )

            # ----------------------------------------------------------
            # STEP 6: COPY LATEST RUN TO REDSHIFT
            # ----------------------------------------------------------

            self._load_data(
                copy_sql,
            )

            Logger.log_banner(
                logger,
                "REDSHIFT LOAD JOB COMPLETED",
            )

            logger.info(
                "Redshift load completed successfully: %s.%s",
                self.redshift_schema,
                self.redshift_table,
            )

            return LoadResult(
                schema_name=self.redshift_schema,
                table_name=self.redshift_table,
                source_path=latest_run_s3_path,
                success=True,
            )

        except Exception:
            logger.exception(
                "Redshift load failed: %s.%s",
                self.redshift_schema,
                self.redshift_table,
            )
            raise

    # ------------------------------------------------------------------
    # Find Latest Processed Run
    # ------------------------------------------------------------------

    def _find_latest_processed_run(self) -> str:
        """
        Find the latest processed Parquet run.

        Example S3 structure:

            processed_data/
                crypto_market/
                    year=2026/
                        month=09/
                            day=30/
                                time=083612/
                                    part-00000-xxxx.parquet

        Returns:
            S3 prefix for the latest run, ending with '/'.
        """

        Logger.log_banner(
            logger,
            "FIND LATEST PROCESSED RUN",
        )

        prefix = f"{self.processed_prefix}/"

        logger.info(
            "Searching processed Parquet files under: s3://%s/%s",
            self.bucket,
            prefix,
        )

        latest_object: dict[str, Any] | None = None

        continuation_token: str | None = None

        while True:
            request: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
            }

            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token

            response = self._s3_client.list_objects_v2(
                **request,
            )

            for obj in response.get("Contents", []):
                key = str(obj.get("Key", ""))

                if not key.endswith(".parquet"):
                    continue

                if latest_object is None:
                    latest_object = cast(dict[str, Any], obj)
                    continue

                current_modified = obj.get("LastModified")
                latest_modified = latest_object.get("LastModified")

                if (
                    current_modified is not None
                    and latest_modified is not None
                    and current_modified > latest_modified
                ):
                    latest_object = cast(dict[str, Any], obj)

            if not response.get("IsTruncated", False):
                break

            continuation_token = response.get(
                "NextContinuationToken",
            )

            if not continuation_token:
                break

        if latest_object is None:
            raise FileNotFoundError(
                "No processed Parquet files were found under "
                f"s3://{self.bucket}/{prefix}"
            )

        latest_key = str(latest_object["Key"])

        logger.info(
            "Latest processed Parquet file: s3://%s/%s",
            self.bucket,
            latest_key,
        )

        # Remove the file name and retain only the run folder.
        run_prefix = f"{latest_key.rsplit('/', 1)[0]}/"

        logger.info(
            "Latest processed run prefix: s3://%s/%s",
            self.bucket,
            run_prefix,
        )

        return run_prefix

    # ------------------------------------------------------------------
    # Read Parquet Schema
    # ------------------------------------------------------------------

    def _read_processed_schema(
        self,
        run_prefix: str,
    ) -> Any:
        """
        Read the latest Parquet schema using PyArrow.

        Spark is intentionally not used here.
        """

        Logger.log_banner(
            logger,
            "READ PROCESSED PARQUET SCHEMA",
        )

        parquet_key = self._find_parquet_file(
            run_prefix,
        )

        logger.info(
            "Reading Parquet schema from: s3://%s/%s",
            self.bucket,
            parquet_key,
        )

        with TemporaryDirectory(
            prefix="crypto-redshift-schema-",
        ) as temp_dir:

            local_file = Path(temp_dir) / "processed.parquet"

            self._download_s3_object(
                key=parquet_key,
                destination=local_file,
            )

            schema = pq.read_schema(
                local_file,
            )

        # Validate that every PyArrow type can be mapped.
        PyArrowSchemaMapper.validate_schema(
            schema,
        )

        logger.info(
            "Processed PyArrow schema validated successfully.",
        )

        logger.info(
            "Detected processed columns: %d",
            len(schema),
        )

        logger.info(
            "Processed columns: %s",
            ", ".join(field.name for field in schema),
        )

        for field in schema:
            logger.info(
                "Detected column: %s | PyArrow type: %s | nullable=%s",
                field.name,
                field.type,
                field.nullable,
            )

        return schema

    # ------------------------------------------------------------------
    # Find Parquet File
    # ------------------------------------------------------------------

    def _find_parquet_file(
        self,
        run_prefix: str,
    ) -> str:
        """Find one Parquet file inside the latest run."""

        response = self._s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=run_prefix,
        )

        parquet_keys = [
            str(obj["Key"])
            for obj in response.get("Contents", [])
            if str(obj["Key"]).endswith(".parquet")
        ]

        if not parquet_keys:
            raise FileNotFoundError(
                "No Parquet file found in processed run: "
                f"s3://{self.bucket}/{run_prefix}"
            )

        parquet_key = sorted(parquet_keys)[0]

        logger.info(
            "Parquet schema source file selected: s3://%s/%s",
            self.bucket,
            parquet_key,
        )

        return parquet_key

    # ------------------------------------------------------------------
    # Download S3 Object
    # ------------------------------------------------------------------

    def _download_s3_object(
        self,
        key: str,
        destination: Path,
    ) -> None:
        """Download one S3 object to a temporary local file."""

        logger.info(
            "Downloading S3 object for schema inspection: %s",
            key,
        )

        self._s3_client.download_file(
            self.bucket,
            key,
            str(destination),
        )

        logger.info(
            "Parquet schema source downloaded successfully.",
        )

    # ------------------------------------------------------------------
    # Generate CREATE TABLE SQL
    # ------------------------------------------------------------------

    def _generate_create_table_sql(
        self,
        schema: Any,
    ) -> str:
        """Generate Redshift CREATE TABLE SQL from PyArrow schema."""

        Logger.log_banner(
            logger,
            "GENERATE REDSHIFT TABLE SCHEMA",
        )

        logger.info(
            "Mapping PyArrow schema to Redshift schema.",
        )

        sql = PyArrowSchemaMapper.generate_create_table_sql(
            schema=schema,
            schema_name=self.redshift_schema,
            table_name=self.redshift_table,
            if_not_exists=True,
        )

        logger.info(
            "Redshift CREATE TABLE SQL generated successfully.",
        )

        return sql

    # ------------------------------------------------------------------
    # Create Target Table
    # ------------------------------------------------------------------

    def _create_target_table(
        self,
        sql: str,
    ) -> None:
        """Create the target Redshift table."""

        Logger.log_banner(
            logger,
            "CREATE REDSHIFT TARGET TABLE",
        )

        logger.info(
            "Preparing target table: %s.%s",
            self.redshift_schema,
            self.redshift_table,
        )

        self.redshift_storage.execute(
            sql,
        )

        logger.info(
            "Redshift target table is ready: %s.%s",
            self.redshift_schema,
            self.redshift_table,
        )

    # ------------------------------------------------------------------
    # Generate COPY SQL
    # ------------------------------------------------------------------

    def _generate_copy_sql(
        self,
        run_prefix: str,
    ) -> str:
        """
        Generate Redshift COPY SQL for the latest run only.
        """

        Logger.log_banner(
            logger,
            "GENERATE REDSHIFT COPY SQL",
        )

        table_name = (
            f"{PyArrowSchemaMapper.quote_identifier(self.redshift_schema)}."
            f"{PyArrowSchemaMapper.quote_identifier(self.redshift_table)}"
        )

        s3_path = f"s3://{self.bucket}/{run_prefix}"

        escaped_s3_path = s3_path.replace(
            "'",
            "''",
        )

        escaped_iam_role = self.redshift_iam_role.replace(
            "'",
            "''",
        )

        sql = (
            f"COPY {table_name}\n"
            f"FROM '{escaped_s3_path}'\n"
            f"IAM_ROLE '{escaped_iam_role}'\n"
            f"FORMAT AS {self.PARQUET_FORMAT};"
        )

        logger.info(
            "Redshift COPY SQL generated successfully.",
        )

        return sql

    # ------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------

    def _load_data(
        self,
        sql: str,
    ) -> None:
        """Execute Redshift COPY command."""

        Logger.log_banner(
            logger,
            "COPY PROCESSED DATA TO REDSHIFT",
        )

        logger.info(
            "Loading processed data into: %s.%s",
            self.redshift_schema,
            self.redshift_table,
        )

        self.redshift_storage.execute(
            sql,
        )

        logger.info(
            "Processed data copied successfully to Redshift.",
        )

    # ------------------------------------------------------------------
    # Configuration Validation
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate load job configuration."""

        if not self.redshift_schema.strip():
            raise ValueError("Redshift schema name cannot be empty.")

        if not self.redshift_table.strip():
            raise ValueError("Redshift table name cannot be empty.")

        if not self.bucket.strip():
            raise ValueError("S3 bucket cannot be empty.")

        if not self.processed_prefix.strip():
            raise ValueError("Processed S3 prefix cannot be empty.")

        if not self.redshift_iam_role.strip():
            raise ValueError("Redshift IAM role cannot be empty.")

        if not self.redshift_iam_role.startswith(
            "arn:aws:iam::",
        ):
            raise ValueError("Redshift IAM role must be a valid IAM role ARN.")

        if not self.aws_region.strip():
            raise ValueError("AWS region cannot be empty.")


# ======================================================================
# Factory
# ======================================================================


def create_load_job(
    redshift_storage: RedshiftStorage,
    redshift_schema: str,
    redshift_table: str,
    bucket: str,
    processed_prefix: str,
    redshift_iam_role: str,
    aws_region: str,
) -> LoadJob:
    """Create a configured LoadJob."""

    return LoadJob(
        redshift_storage=redshift_storage,
        redshift_schema=redshift_schema,
        redshift_table=redshift_table,
        bucket=bucket,
        processed_prefix=processed_prefix,
        redshift_iam_role=redshift_iam_role,
        aws_region=aws_region,
    )


# ======================================================================
# Entry Point
# ======================================================================


def run_load_job() -> LoadResult:
    """Create configuration, execute the load job, and return the result."""

    redshift_config = CONFIG["redshift"]
    s3_config = CONFIG["s3"]
    aws_config = CONFIG["aws"]

    bucket = str(
        s3_config["bucket"],
    ).strip()

    processed_prefix = str(
        s3_config["processed_prefix"],
    ).strip()

    aws_region = str(
        aws_config["region"],
    ).strip()

    redshift_storage: RedshiftStorage | None = None

    try:
        redshift_storage = RedshiftStorage(
            host=str(
                redshift_config["host"],
            ),
            port=int(
                redshift_config["port"],
            ),
            database=str(
                redshift_config["database"],
            ),
            aws_region=aws_region,
            workgroup=str(
                redshift_config["workgroup"],
            ),
        )

        job = create_load_job(
            redshift_storage=redshift_storage,
            redshift_schema=str(
                redshift_config["schema"],
            ),
            redshift_table=str(
                redshift_config["table"],
            ),
            bucket=bucket,
            processed_prefix=processed_prefix,
            redshift_iam_role=str(
                redshift_config["iam_role"],
            ),
            aws_region=aws_region,
        )

        return job.run()

    finally:
        if redshift_storage is not None:
            redshift_storage.close()
