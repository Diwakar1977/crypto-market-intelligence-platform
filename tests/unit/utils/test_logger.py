from __future__ import annotations

from pathlib import Path

from config.config import CONFIG
from src.utils.logger import Logger


def test_logger_creates_log_file() -> None:
    """Test that logger creates and writes to a log file."""

    log_directory = Path(str(CONFIG["runtime"]["log_dir"]))
    log_directory.mkdir(parents=True, exist_ok=True)

    log_file = log_directory / "test_logger.log"

    # Remove previous test log if it exists.
    if log_file.exists():
        log_file.unlink()

    logger = Logger.get_logger(
        "test_log",
        "test_logger.log",
    )

    Logger.log_banner(
        logger,
        "CRYPTO ETL PIPELINE STARTED",
    )

    logger.info("Application started successfully")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Flush file handlers so all messages are written.
    for handler in logger.handlers:
        handler.flush()

    # Verify log file exists.
    assert log_file.exists()

    content = log_file.read_text(encoding="utf-8")

    assert "CRYPTO ETL PIPELINE STARTED" in content
    assert "Application started successfully" in content
    assert "This is a warning" in content
    assert "This is an error" in content

    # Close handlers.
    for handler in logger.handlers:
        handler.close()

    logger.handlers.clear()
