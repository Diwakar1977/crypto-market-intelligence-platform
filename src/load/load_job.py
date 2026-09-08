from dataclasses import dataclass

from config.config import CONFIG

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from src.load.redshift_schema_mapper import RedshiftSchemaMapper
from src.load.redshift_storage import RedshiftStorage
from src.spark.spark_session import SparkSessionFactory
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
    """Load processed Parquet data from S3 into Redshift."""

    PARQUET_FORMAT = "PARQUET"

    def __init__(
        self,
        spark: SparkSession,
        redshift_storage: RedshiftStorage,
        redshift_schema: str,
        redshift_table: str,
        processed_spark_path: str,
        processed_s3_path: str,
        redshift_iam_role: str,
    ) -> None:
        self.spark = spark
        self.redshift_storage = redshift_storage
        self.redshift_schema = redshift_schema
        self.redshift_table = redshift_table
        self.processed_spark_path = processed_spark_path
        self.processed_s3_path = processed_s3_path
        self.redshift_iam_role = redshift_iam_role

        self._validate_configuration()

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
            "Processed Spark path: %s",
            self.processed_spark_path,
        )

        logger.info(
            "Processed Redshift COPY path: %s",
            self.processed_s3_path,
        )

        try:
            # ---------------------------------------------------------
            # STEP 1: READ PROCESSED SCHEMA
            # ---------------------------------------------------------
            
            schema = self._read_processed_schema()

            # ---------------------------------------------------------
            # STEP 2: GENERATE CREATE TABLE SQL
            # ---------------------------------------------------------
            
            create_table_sql = (
                self._generate_create_table_sql(
                    schema
                )
            )

            # ---------------------------------------------------------
            # STEP 3: CREATE REDSHIFT TABLE
            # ---------------------------------------------------------
            
            self._create_target_table(
                create_table_sql
            )

            # ---------------------------------------------------------
            # STEP 4: GENERATE COPY SQL
            # ---------------------------------------------------------
            
            copy_sql = self._generate_copy_sql()

            # ---------------------------------------------------------
            # STEP 5: COPY DATA INTO REDSHIFT
            # ---------------------------------------------------------
            
            self._load_data(copy_sql)

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
                source_path=self.processed_s3_path,
                success=True,
            )

        except Exception:
            logger.exception(
                "Redshift load failed: %s.%s",
                self.redshift_schema,
                self.redshift_table,
            )
            raise

    def _read_processed_schema(self) -> StructType:
        """Read processed Parquet schema using Spark."""

        Logger.log_banner(
            logger,
            "READ PROCESSED PARQUET SCHEMA",
        )

        logger.info(
            "Reading processed data from: %s",
            self.processed_spark_path,
        )

        dataframe = self.spark.read.parquet(
            self.processed_spark_path
        )

        schema = dataframe.schema

        RedshiftSchemaMapper.validate_schema(
            schema
        )

        logger.info(
            "Processed schema validated successfully."
        )

        logger.info(
            "Detected processed columns: %d",
            len(schema.fields),
        )

        logger.info(
            "Processed columns: %s",
            ", ".join(
                field.name
                for field in schema.fields
            ),
        )

        return schema

    def _generate_create_table_sql(
        self,
        schema: StructType,
    ) -> str:
        """Generate Redshift CREATE TABLE SQL."""

        Logger.log_banner(
            logger,
            "GENERATE REDSHIFT TABLE SCHEMA",
        )

        logger.info(
            "Mapping Spark schema to Redshift schema."
        )

        sql = (
            RedshiftSchemaMapper.generate_create_table_sql(
                schema=schema,
                schema_name=self.redshift_schema,
                table_name=self.redshift_table,
                if_not_exists=True,
            )
        )

        logger.info(
            "Redshift CREATE TABLE SQL generated successfully."
        )

        return sql

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

        self.redshift_storage.execute(sql)

        logger.info(
            "Redshift target table is ready: %s.%s",
            self.redshift_schema,
            self.redshift_table,
        )

    def _generate_copy_sql(self) -> str:
        """Generate Redshift COPY SQL."""

        Logger.log_banner(
            logger,
            "GENERATE REDSHIFT COPY SQL",
        )

        table_name = (
            f"{RedshiftSchemaMapper.quote_identifier(self.redshift_schema)}."
            f"{RedshiftSchemaMapper.quote_identifier(self.redshift_table)}"
        )

        s3_path = self.processed_s3_path.replace(
            "'",
            "''",
        )

        iam_role = self.redshift_iam_role.replace(
            "'",
            "''",
        )

        sql = (
            f"COPY {table_name}\n"
            f"FROM '{s3_path}'\n"
            f"IAM_ROLE '{iam_role}'\n"
            f"FORMAT AS {self.PARQUET_FORMAT};"
        )

        logger.info(
            "Redshift COPY SQL generated successfully."
        )

        return sql

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

        self.redshift_storage.execute(sql)

        logger.info(
            "Processed data copied successfully to Redshift."
        )

    def _validate_configuration(self) -> None:
        """Validate load job configuration."""

        if not self.redshift_schema.strip():
            raise ValueError(
                "Redshift schema name cannot be empty."
            )

        if not self.redshift_table.strip():
            raise ValueError(
                "Redshift table name cannot be empty."
            )

        if not self.processed_spark_path.strip():
            raise ValueError(
                "Processed Spark path cannot be empty."
            )

        if not self.processed_spark_path.startswith(
            "s3a://"
        ):
            raise ValueError(
                "Processed Spark path must start with 's3a://'."
            )

        if not self.processed_s3_path.strip():
            raise ValueError(
                "Processed S3 path cannot be empty."
            )

        if not self.processed_s3_path.startswith(
            "s3://"
        ):
            raise ValueError(
                "Processed S3 path must start with 's3://'."
            )

        if not self.redshift_iam_role.strip():
            raise ValueError(
                "Redshift IAM role cannot be empty."
            )

        if not self.redshift_iam_role.startswith(
            "arn:aws:iam::"
        ):
            raise ValueError(
                "Redshift IAM role must be a valid IAM role ARN."
            )


def create_load_job(
    spark: SparkSession,
    redshift_storage: RedshiftStorage,
    redshift_schema: str,
    redshift_table: str,
    processed_spark_path: str,
    processed_s3_path: str,
    redshift_iam_role: str,
) -> LoadJob:
    """Create a configured LoadJob."""

    return LoadJob(
        spark=spark,
        redshift_storage=redshift_storage,
        redshift_schema=redshift_schema,
        redshift_table=redshift_table,
        processed_spark_path=processed_spark_path,
        processed_s3_path=processed_s3_path,
        redshift_iam_role=redshift_iam_role,
    )

def run_load_job() -> LoadResult:
    """Create configuration, execute, and return the load result."""

    redshift_config = CONFIG["redshift"]
    s3_config = CONFIG["s3"]
    aws_config = CONFIG["aws"]

    bucket = str(
        s3_config["bucket"]
    ).strip()

    processed_prefix = str(
        s3_config["processed_prefix"]
    ).strip()

    processed_spark_path = (
        f"s3a://{bucket}/{processed_prefix}"
    )

    processed_s3_path = (
        f"s3://{bucket}/{processed_prefix}"
    )

    spark = SparkSessionFactory.create()
    redshift_storage: RedshiftStorage | None = None

    try:
        redshift_storage = RedshiftStorage(
            host=str(
                redshift_config["host"]
            ),
            port=int(
                redshift_config["port"]
            ),
            database=str(
                redshift_config["database"]
            ),
            aws_region=str(
                aws_config["region"]
            ),
            workgroup=str(
                redshift_config["workgroup"]
            ),
        )

        job = create_load_job(
            spark=spark,
            redshift_storage=redshift_storage,
            redshift_schema=str(
                redshift_config["schema"]
            ),
            redshift_table=str(
                redshift_config["table"]
            ),
            processed_spark_path=processed_spark_path,
            processed_s3_path=processed_s3_path,
            redshift_iam_role=str(
                redshift_config["iam_role"]
            ),
        )

        return job.run()

    finally:
        if redshift_storage is not None:
            redshift_storage.close()

        spark.stop()