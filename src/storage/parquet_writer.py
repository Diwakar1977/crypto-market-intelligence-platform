from __future__ import annotations

from pyspark.sql import DataFrame

from src.utils.logger import Logger

logger = Logger.get_logger(
    "parquet_writer",
    "parquet_writer.log"
)

class ParquetWriter:
    """Write Spark DataFrame to Parquet storage."""

    def write(
        self,
        df: DataFrame,
        output_path: str,
        mode: str = "append"
    ) -> None:
        """
        Write a DataFrame to Parquet storage.
        
        Parameters:
            df: Spark DataFrame
            ouput_path: Target storage path.
            mode: Spark write mode
                (append, overwrite, ignore, error)
        """

        logger.info(
            "Starting Parquet write. path=%s mode=%s",
            output_path,
            mode
        )

        try:
            if not output_path:
                raise ValueError("Output path cannot be empty.")

            (
                df.write
                .mode(mode)
                .parquet(output_path)
            )

            logger.info(
                "Parquet write completed successully. path=%s",
                output_path
            )

        except Exception:
            logger.exception(
                "Parquet write failed. path=%s",
                output_path
            )
            raise