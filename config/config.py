from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import boto3
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml_file(
    config_path: Path,
) -> dict[str, Any]:
    """Load configuration from a local YAML file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Invalid configuration format: {config_path}")

    return config


def _load_s3_config() -> dict[str, Any]:
    """Load production configuration from S3."""

    bucket = os.getenv("CONFIG_S3_BUCKET")
    key = os.getenv(
        "CONFIG_S3_KEY",
        "config/production.yaml",
    )

    if not bucket:
        raise ValueError(
            "CONFIG_S3_BUCKET environment variable " "is required for production."
        )

    region = os.getenv(
        "AWS_REGION",
        "ap-south-1",
    )

    s3_client = boto3.client(
        "s3",
        region_name=region,
    )

    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
    )

    content = response["Body"].read().decode("utf-8")

    config = yaml.safe_load(content)

    if not isinstance(config, dict):
        raise TypeError(f"Invalid configuration format: " f"s3://{bucket}/{key}")

    return config


def load_config() -> dict[str, Any]:
    """Load configuration based on the ENV variable."""

    environment = (
        os.getenv(
            "ENV",
            "local",
        )
        .strip()
        .lower()
    )

    if environment == "local":
        config_path = CONFIG_DIR / "local.yaml"

        return _load_yaml_file(config_path)

    if environment == "ci":
        config_path = CONFIG_DIR / "ci.yaml"

        return _load_yaml_file(config_path)

    if environment == "production":
        return _load_s3_config()

    raise ValueError(f"Unsupported ENV: {environment}")


CONFIG = load_config()
