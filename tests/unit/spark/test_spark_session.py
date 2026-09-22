from __future__ import annotations

import logging

from pyspark.sql import SparkSession

from spark.spark_session import SparkSessionFactory

# Suppress unnecessary Py4J INFO/DEBUG logs during tests."
py4j_logger = logging.getLogger("py4j")
py4j_logger.setLevel(logging.WARNING)
py4j_logger.propagate = False


def test_create_spark_session() -> None:
    """Test that a configured SparkSession is created."""

    spark = SparkSessionFactory.create()

    try:
        assert isinstance(spark, SparkSession)

        assert spark.sparkContext.master == "local[*]"
        assert spark.conf.get("spark.driver.memory") == "4g"
        assert spark.conf.get("spark.executor.memory") == "4g"
        assert spark.conf.get("spark.sql.shuffle.partitions") == "200"

    finally:
        spark.stop()
