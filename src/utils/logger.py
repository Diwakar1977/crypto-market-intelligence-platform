from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.config import CONFIG


class Logger:
    """Centralized application logging utility."""

    _FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def get_logger(name: str, log_file: str):
        """
        Create and return a configured application logger.

        Logs are written to:
        1. Console
        2. Rotating log file
        """

        logger = logging.getLogger(name)

        # Prevent dupplicte handlers
        if logger.handlers:
            return logger

        runtime_config = CONFIG["runtime"]

        log_level = runtime_config["log_level"]
        log_dir = runtime_config["log_dir"]

        level = getattr(logging, str(log_level).upper(), logging.INFO)

        logger.setLevel(level)
        logger.propagate = False

        formatter = logging.Formatter(fmt=Logger._FORMAT, datefmt=Logger._DATE_FORMAT)

        # Log directory
        log_directory = Path(str(log_dir))
        log_directory.mkdir(parents=True, exist_ok=True)

        log_path = log_directory / log_file

        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def log_banner(logger: logging.Logger, message: str) -> None:
        """ "Write a formatted banner to the log."""

        separator = "=" * 70

        logger.info(separator)
        logger.info(message)
        logger.info(separator)
