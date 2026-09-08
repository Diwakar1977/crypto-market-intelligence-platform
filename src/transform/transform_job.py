from __future__ import annotations

import json
import sys

import boto3
from pyspark.sql import DataFrame, SparkSession

from config.config import CONFIG
from src.schema.schema_inferer import SchemaInferer
from src.schema.schema_manager import SchemaManager
from src.spark.spark_session import SparkSessionFactory
from src.storage.parquet_writer import ParquetWriter
from src.storage.path_builder import PathBuilder
from src.transform.data_normalizer import DataNormalizer
from src.transform.data_validator import (
    DataValidator,
    ValidationResult,
)
from src.transform.feature_engineer import FeatureEngineer
from src.utils.logger import Logger


logger = Logger.get_logger(
    "transform_job",
    "transform_job.log",
)


class TransformJob:
    """Orchestrate the complete cryptocurrency transformation pipeline."""

    def __init__(
        self,
        spark: SparkSession,
        schema_inferer: SchemaInferer,
        schema_manager: SchemaManager,
        data_validator: DataValidator,
        data_normalizer: DataNormalizer,
        feature_engineer: FeatureEngineer,
        path_builder: PathBuilder,
        parquet_writer: ParquetWriter,
    ) -> None:

        self.spark = spark
        self.schema_inferer = schema_inferer
        self.schema_manager = schema_manager
        self.data_validator = data_validator
        self.data_normalizer = data_normalizer
        self.feature_engineer = feature_engineer
        self.path_builder = path_builder
        self.parquet_writer = parquet_writer

    # =============================================================
    # RUN
    # =============================================================

    def run(
        self,
        input_path: str,
    ) -> str:

        Logger.log_banner(
            logger,
            "TRANSFORM JOB STARTED",
        )

        try:

            if not input_path or not input_path.strip():
                raise ValueError(
                    "Input path cannot be empty."
                )

            logger.info(
                "Raw input path: %s",
                input_path,
            )

            application_config = CONFIG["application"]
            s3_config = CONFIG["s3"]

            raw_dataset = str(
                application_config["raw_dataset"]
            )

            s3_bucket = str(
                s3_config["bucket"]
            )

            # =====================================================
            # STEP 1 - ORIGINAL JSON COLUMN ORDER
            # =====================================================

            original_json_column_order = (
                self._get_original_json_column_order(
                    input_path=input_path,
                )
            )

            logger.info(
                "ORIGINAL NDJSON column order: %s",
                ", ".join(
                    original_json_column_order
                ),
            )

            # =====================================================
            # STEP 2 - READ RAW DATA
            # =====================================================

            raw_df = self._read_raw_data(
                input_path=input_path,
            )

            raw_record_count = raw_df.count()

            if raw_record_count == 0:
                raise ValueError(
                    "Raw input contains no records."
                )

            logger.info(
                "Raw data loaded successfully. "
                "records=%d columns=%d",
                raw_record_count,
                len(raw_df.columns),
            )

            # =====================================================
            # STEP 3 - SOURCE COLUMN VALIDATION
            # =====================================================

            self._validate_source_columns(
                original_json_column_order=(
                    original_json_column_order
                ),
                raw_df=raw_df,
            )

            # =====================================================
            # STEP 4 - COLLECT RECORDS FOR SCHEMA INFERENCE
            # =====================================================

            records = [
                row.asDict(
                    recursive=True
                )
                for row in raw_df.collect()
            ]

            # =====================================================
            # STEP 5 - SCHEMA INFERENCE
            # =====================================================

            inferred_schema = (
                self.schema_inferer.infer(
                    records,
                )
            )

            # =====================================================
            # STEP 6 - SCHEMA MANAGEMENT
            # =====================================================

            managed_schema = (
                self.schema_manager.normalize(
                    inferred_schema,
                )
            )

            self.schema_manager.validate(
                managed_schema,
            )

            # =====================================================
            # STEP 7 - APPLY MANAGED SCHEMA
            # =====================================================

            typed_df = self.schema_manager.apply_schema(
                df=raw_df,
                normalized_schema=managed_schema,
            )

            # =====================================================
            # STEP 8 - DATA VALIDATION
            # =====================================================

            validation_result = (
                self.data_validator.validate(
                    typed_df,
                )
            )

            self._handle_validation_result(
                validation_result,
            )

            logger.info(
                "Data validation completed successfully."
            )

            # =====================================================
            # STEP 9 - DATA NORMALIZATION
            # =====================================================

            normalized_df = (
                self.data_normalizer.normalize(
                    typed_df,
                )
            )

            # =====================================================
            # STEP 10 - SURVIVING RAW COLUMNS
            # =====================================================

            normalized_column_set = set(
                normalized_df.columns
            )

            surviving_raw_columns = [
                column_name
                for column_name
                in original_json_column_order
                if column_name
                in normalized_column_set
            ]

            removed_raw_columns = [
                column_name
                for column_name
                in original_json_column_order
                if column_name
                not in normalized_column_set
            ]

            logger.info(
                "RAW columns surviving normalization: %d",
                len(surviving_raw_columns),
            )

            if removed_raw_columns:
                logger.info(
                    "RAW columns removed: %s",
                    ", ".join(
                        removed_raw_columns
                    ),
                )

            # =====================================================
            # STEP 11 - NORMALIZED COLUMN VALIDATION
            # =====================================================

            self._validate_normalized_columns(
                raw_column_order=(
                    surviving_raw_columns
                ),
                normalized_df=normalized_df,
            )

            # =====================================================
            # STEP 12 - FEATURE ENGINEERING
            # =====================================================

            processed_df = (
                self.feature_engineer.transform(
                    normalized_df,
                )
            )

            # =====================================================
            # STEP 13 - FINAL COLUMN ORDER
            # =====================================================

            processed_df = (
                self._order_processed_columns(
                    raw_column_order=(
                        surviving_raw_columns
                    ),
                    processed_df=processed_df,
                )
            )

            logger.info(
                "Final processed column order: %s",
                ", ".join(
                    processed_df.columns
                ),
            )

            # =====================================================
            # STEP 14 - FINAL VALIDATION
            # =====================================================

            self._validate_final_columns(
                raw_column_order=(
                    surviving_raw_columns
                ),
                processed_df=processed_df,
            )

            # =====================================================
            # STEP 15 - WRITE PARQUET
            # =====================================================

            processed_key = (
                self.path_builder.build_processed_path(
                    dataset_name=raw_dataset,
                )
            )

            processed_path = (
                f"s3a://{s3_bucket}/{processed_key}"
            )

            logger.info(
                "Processed S3 destination: %s",
                processed_path,
            )

            self.parquet_writer.write(
                df=processed_df,
                output_path=processed_path,
                mode="append",
            )

            logger.info(
                "Processed Parquet data written successfully."
            )

            Logger.log_banner(
                logger,
                "TRANSFORM JOB COMPLETED",
            )

            return processed_path

        except Exception:

            logger.exception(
                "Transform job failed."
            )

            Logger.log_banner(
                logger,
                "TRANSFORM JOB FAILED",
            )

            raise

    # =============================================================
    # GET ORIGINAL JSON COLUMN ORDER
    # =============================================================

    @staticmethod
    def _get_original_json_column_order(
        input_path: str,
    ) -> list[str]:

        if not input_path or not input_path.strip():
            raise ValueError(
                "Input path cannot be empty."
            )

        if not input_path.startswith(
            "s3a://"
        ):
            raise ValueError(
                "Expected s3a:// input path, "
                f"got: {input_path}"
            )

        s3_path = input_path.replace(
            "s3a://",
            "",
            1,
        )

        if "/" not in s3_path:
            raise ValueError(
                "Invalid S3A path. "
                "Expected s3a://bucket/key."
            )

        bucket, key = s3_path.split(
            "/",
            1,
        )

        if not bucket:
            raise ValueError(
                "S3 bucket cannot be empty."
            )

        if not key:
            raise ValueError(
                "S3 object key cannot be empty."
            )

        s3 = boto3.client(
            "s3",
        )

        response = s3.get_object(
            Bucket=bucket,
            Key=key,
        )

        body = response["Body"]

        try:
            first_line_bytes = body.readline()
        finally:
            body.close()

        if not first_line_bytes:
            raise ValueError(
                "Raw NDJSON file is empty."
            )

        first_line = (
            first_line_bytes
            .decode("utf-8")
            .strip()
        )

        if not first_line:
            raise ValueError(
                "First NDJSON line is empty."
            )

        try:
            first_record = json.loads(
                first_line,
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "First NDJSON line is not valid JSON."
            ) from exc

        if not isinstance(
            first_record,
            dict,
        ):
            raise ValueError(
                "First NDJSON record must be a JSON object."
            )

        column_order = list(
            first_record.keys()
        )

        if not column_order:
            raise ValueError(
                "First NDJSON record contains no columns."
            )

        return column_order

    # =============================================================
    # READ RAW
    # =============================================================

    def _read_raw_data(
        self,
        input_path: str,
    ) -> DataFrame:

        if not input_path or not input_path.strip():
            raise ValueError(
                "Input path cannot be empty."
            )

        return self.spark.read.json(
            input_path,
        )

    # =============================================================
    # SOURCE VALIDATION
    # =============================================================

    @staticmethod
    def _validate_source_columns(
        original_json_column_order: list[str],
        raw_df: DataFrame,
    ) -> None:

        spark_columns = set(
            raw_df.columns
        )

        missing_columns = [
            column_name
            for column_name
            in original_json_column_order
            if column_name
            not in spark_columns
        ]

        if missing_columns:
            raise ValueError(
                "Columns found in original NDJSON but missing "
                f"from Spark DataFrame: {missing_columns}"
            )

    # =============================================================
    # NORMALIZED COLUMN VALIDATION
    # =============================================================

    @staticmethod
    def _validate_normalized_columns(
        raw_column_order: list[str],
        normalized_df: DataFrame,
    ) -> None:

        normalized_columns = set(
            normalized_df.columns
        )

        missing_columns = [
            column_name
            for column_name
            in raw_column_order
            if column_name
            not in normalized_columns
        ]

        if missing_columns:
            raise ValueError(
                "Required RAW columns are missing after "
                f"normalization: {missing_columns}"
            )

    # =============================================================
    # FINAL COLUMN ORDER
    # =============================================================

    @staticmethod
    def _order_processed_columns(
        raw_column_order: list[str],
        processed_df: DataFrame,
    ) -> DataFrame:

        processed_columns = list(
            processed_df.columns
        )

        original_columns = [
            column_name
            for column_name
            in raw_column_order
            if column_name
            in processed_columns
        ]

        derived_columns = [
            column_name
            for column_name
            in processed_columns
            if column_name
            not in original_columns
        ]

        final_columns = (
            original_columns
            + derived_columns
        )

        if not final_columns:
            raise ValueError(
                "No columns available after transformation."
            )

        if len(final_columns) != len(
            processed_columns
        ):
            raise ValueError(
                "Final column ordering does not match "
                "processed DataFrame column count."
            )

        if len(final_columns) != len(
            set(final_columns)
        ):
            raise ValueError(
                "Duplicate columns detected in final order."
            )

        return processed_df.select(
            *final_columns,
        )

    # =============================================================
    # FINAL VALIDATION
    # =============================================================

    @staticmethod
    def _validate_final_columns(
        raw_column_order: list[str],
        processed_df: DataFrame,
    ) -> None:

        processed_columns = list(
            processed_df.columns
        )

        raw_count = len(
            raw_column_order
        )

        if len(processed_columns) < raw_count:
            raise ValueError(
                "Processed DataFrame contains fewer columns "
                "than required RAW columns."
            )

        actual_raw_columns = (
            processed_columns[:raw_count]
        )

        if actual_raw_columns != (
            raw_column_order
        ):
            raise ValueError(
                "RAW column order validation failed. "
                f"Expected={raw_column_order}, "
                f"Actual={actual_raw_columns}"
            )

        if len(processed_columns) != len(
            set(processed_columns)
        ):
            raise ValueError(
                "Duplicate columns detected in final "
                "processed DataFrame."
            )

        logger.info(
            "Final processed schema validation passed. "
            "Total columns=%d",
            len(processed_columns),
        )

    # =============================================================
    # VALIDATION RESULT
    # =============================================================

    @staticmethod
    def _handle_validation_result(
        validation_result: ValidationResult,
    ) -> None:

        if not validation_result.is_valid:
            raise ValueError(
                "Data validation failed. "
                f"Invalid record count: "
                f"{validation_result.invalid_count}"
            )


# =================================================================
# FACTORY
# =================================================================


def create_transform_job(
    spark: SparkSession,
) -> TransformJob:

    return TransformJob(
        spark=spark,
        schema_inferer=SchemaInferer(),
        schema_manager=SchemaManager(),
        data_validator=DataValidator(),
        data_normalizer=DataNormalizer(),
        feature_engineer=FeatureEngineer(),
        path_builder=PathBuilder(),
        parquet_writer=ParquetWriter(),
    )


# =================================================================
# RUNNER
# =================================================================


def run_transform_job(
    spark: SparkSession,
    input_path: str,
) -> str:

    job = create_transform_job(
        spark=spark,
    )

    return job.run(
        input_path=input_path,
    )


# =================================================================
# MAIN
# =================================================================


def main() -> None:

    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: transform_job.py <raw_input_path>"
        )

    input_path = sys.argv[1]

    Logger.log_banner(
        logger,
        "SPARK TRANSFORM APPLICATION STARTED",
    )

    spark = SparkSessionFactory.create()

    try:

        run_transform_job(
            spark=spark,
            input_path=input_path,
        )

    finally:

        logger.info(
            "Stopping Spark session."
        )

        spark.stop()

        Logger.log_banner(
            logger,
            "SPARK TRANSFORM APPLICATION FINISHED",
        )


if __name__ == "__main__":
    main()
