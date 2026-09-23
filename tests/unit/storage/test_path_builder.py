from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.config.config import CONFIG
from src.storage.path_builder import PathBuilder
from src.utils.logger import Logger


def test_build_raw_path() -> None:
    """Test that the expected raw S3 path is generated."""

    run_time = datetime(
        2026,
        8,
        21,
        12,
        30,
        45,
        tzinfo=timezone.utc,
    )

    raw_prefix = str(
        CONFIG["s3"]["raw_prefix"],
    ).rstrip("/")

    raw_path = PathBuilder.build_raw_path(
        dataset_name="crypto_market",
        run_time=run_time,
    )

    expected_path = (
        f"{raw_prefix}/"
        "crypto_market/"
        "year=2026/"
        "month=08/"
        "day=21/"
        "run_time=123045.ndjson"
    )

    assert raw_path == expected_path


def test_build_processed_path() -> None:
    """Test that the expected processed S3 path is generated."""

    run_time = datetime(
        2026,
        8,
        21,
        12,
        30,
        45,
        tzinfo=timezone.utc,
    )

    processed_prefix = str(
        CONFIG["s3"]["processed_prefix"],
    ).rstrip("/")

    processed_path = PathBuilder.build_processed_path(
        dataset_name="crypto_market",
        run_time=run_time,
    )

    expected_path = (
        f"{processed_prefix}/"
        "crypto_market/"
        "year=2026/"
        "month=08/"
        "day=21/"
        "time=123045/"
    )

    assert processed_path == expected_path


def test_path_builder_logs() -> None:
    """Test that generated paths are written to the log file."""

    logger = Logger.get_logger(
        "test_path_builder",
        "test_path_builder.log",
    )

    run_time = datetime(
        2026,
        8,
        21,
        12,
        30,
        45,
        tzinfo=timezone.utc,
    )

    raw_path = PathBuilder.build_raw_path(
        dataset_name="crypto_market",
        run_time=run_time,
    )

    processed_path = PathBuilder.build_processed_path(
        dataset_name="crypto_market",
        run_time=run_time,
    )

    logger.info(
        "RAW PATH: %s",
        raw_path,
    )

    logger.info(
        "PROCESSED PATH: %s",
        processed_path,
    )

    for handler in logger.handlers:
        handler.flush()

    project_root = Path(__file__).resolve().parents[3]

    log_file = project_root / "logs" / "test_path_builder.log"

    assert log_file.exists()

    content = log_file.read_text(
        encoding="utf-8",
    )

    assert "RAW PATH:" in content
    assert "PROCESSED PATH:" in content
    assert raw_path in content
    assert processed_path in content

    for handler in logger.handlers:
        handler.close()

    logger.handlers.clear()
