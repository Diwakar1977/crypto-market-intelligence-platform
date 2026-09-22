from __future__ import annotations

from pyspark.sql import SparkSession

from config.config import CONFIG
from src.utils.logger import Logger

logger = Logger.get_logger(
    "spark_session",
    "spark_session.log",
)


class SparkSessionFactory:
    """Factory for creating the application SparkSession."""

    @classmethod
    def create(cls) -> SparkSession:
        """Create and return a configured SparkSession."""

        try:
            application_config = CONFIG["application"]
            spark_config = CONFIG["spark"]
            hadoop_config = CONFIG["hadoop"]

            application_name = str(application_config["name"])
            master = str(spark_config["master"])
            driver_memory = str(spark_config["driver_memory"])
            executor_memory = str(spark_config["executor_memory"])
            shuffle_partitions = str(spark_config["shuffle_partitions"])
            aws_package = str(hadoop_config["aws_package"])

            spark_builder = (
                SparkSession.builder.appName(application_name)
                .master(master)
                .config(
                    "spark.driver.memory",
                    driver_memory,
                )
                .config(
                    "spark.executor.memory",
                    executor_memory,
                )
                .config(
                    "spark.sql.shuffle.partitions",
                    shuffle_partitions,
                )
                .config(
                    "spark.jars.packages",
                    aws_package,
                )
                .config(
                    "spark.hadoop.fs.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3AFileSystem",
                )
                .config(
                    "spark.hadoop.fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                )
            )

            spark_session = spark_builder.getOrCreate()

            logger.info(
                "SparkSession created successfully. "
                "application=%s master=%s version=%s",
                application_name,
                spark_session.sparkContext.master,
                spark_session.version,
            )

            return spark_session

        except Exception as exc:
            logger.exception(
                "Failed to create SparkSession.",
            )

            raise RuntimeError("Failed to create SparkSession.") from exc
