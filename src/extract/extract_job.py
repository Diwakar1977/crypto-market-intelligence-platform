import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from config.config import CONFIG
from src.extract.coingecko_client import CoinGeckoClient
from src.storage.path_builder import PathBuilder
from src.storage.s3_storage import S3Storage
from src.utils.logger import Logger


logger = Logger.get_logger(
    "extract_job",
    "extract_job.log",
)


@dataclass(frozen=True)
class ExtractResult:
    """Represent the result of the extraction job."""

    s3_key: str
    record_count: int


class ExtractJob:
    """Extract CoinGecko data and store it in S3 raw storage."""

    def __init__(
        self,
        client: CoinGeckoClient,
        storage: S3Storage,
    ) -> None:
        self.client = client
        self.storage = storage

    def run(self) -> ExtractResult:
        """Execute extraction and upload raw data to S3."""

        Logger.log_banner(
            logger,
            "EXTRACT JOB STARTED",
        )

        try:
            # ---------------------------------------------------------
            # STEP 1: FETCH DATA FROM COINGECKO
            # ---------------------------------------------------------
            
            records = self.client.fetch_market_data()

            record_count = len(records)

            logger.info(
                "Extracted %d records from CoinGecko.",
                record_count,
            )

            if record_count == 0:
                raise ValueError(
                    "CoinGecko returned zero records."
                )

            # ---------------------------------------------------------
            # STEP 2: READ CONFIGURATION
            # ---------------------------------------------------------
            
            raw_dataset = str(
                CONFIG["application"]["raw_dataset"]
            )

            s3_config = CONFIG["s3"]

            s3_bucket = str(
                s3_config["bucket"]
            )

            # ---------------------------------------------------------
            # STEP 3: BUILD RAW S3 KEY
            # ---------------------------------------------------------
            
            s3_key = PathBuilder.build_raw_path(
                dataset_name=raw_dataset,
            )

            logger.info(
                "Raw S3 destination: s3://%s/%s",
                s3_bucket,
                s3_key,
            )

            # ---------------------------------------------------------
            # STEP 4: CREATE TEMPORARY NDJSON
            # ---------------------------------------------------------
            
            with TemporaryDirectory() as temp_dir:

                local_path = (
                    Path(temp_dir)
                    / f"{raw_dataset}.ndjson"
                )

                self._write_ndjson(
                    records=records,
                    local_path=local_path,
                )

                logger.info(
                    "Raw NDJSON file created: %s",
                    local_path,
                )

                # -----------------------------------------------------
                # STEP 5: UPLOAD TO S3
                # -----------------------------------------------------
                
                self.storage.upload_file(
                    local_path=local_path,
                    s3_key=s3_key,
                )

                logger.info(
                    "Raw data uploaded successfully."
                )

            # ---------------------------------------------------------
            # STEP 6: RETURN EXTRACTION RESULT
            # ---------------------------------------------------------
            
            result = ExtractResult(
                s3_key=s3_key,
                record_count=record_count,
            )

            logger.info(
                "Extraction result: s3_key=%s, record_count=%d",
                result.s3_key,
                result.record_count,
            )

            Logger.log_banner(
                logger,
                "EXTRACT JOB COMPLETED",
            )

            return result

        except Exception:
            logger.exception(
                "Extract job failed."
            )

            Logger.log_banner(
                logger,
                "EXTRACT JOB FAILED",
            )

            raise

    @staticmethod
    def _write_ndjson(
        records: list[dict[str, Any]],
        local_path: Path,
    ) -> None:
        """Write records to an NDJSON file."""

        with local_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:

            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )
                file.write("\n")


def create_extract_job() -> ExtractJob:
    """Create a configured ExtractJob instance."""

    aws_config = CONFIG["aws"]
    s3_config = CONFIG["s3"]

    client = CoinGeckoClient()

    storage = S3Storage(
        bucket_name=str(
            s3_config["bucket"]
        ),
        region_name=str(
            aws_config["region"]
        ),
    )

    return ExtractJob(
        client=client,
        storage=storage,
    )


def run_extract_job() -> ExtractResult:
    """Create and execute the extraction job."""

    job = create_extract_job()

    return job.run()