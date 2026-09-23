from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _load_airflow_config() -> dict[str, Any]:
    """Load production configuration from Airflow Variable."""

    try:
        from airflow.sdk import Variable

    except ImportError as exc:
        raise ImportError(
            "Airflow is required to load production configuration "
            "from Airflow Variables."
        ) from exc

    config = Variable.get(
        "crypto_etl_config",
        deserialize_json=True,
    )

    if not isinstance(config, dict):
        raise TypeError(
            "Airflow Variable 'crypto_etl_config' " "must contain a JSON object."
        )

    return config


def load_config() -> dict[str, Any]:
    """Load configuration based on the ENV variable."""

    environment = (
        os.getenv(
            "ENV",
            "",
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

    if environment == "":
        return _load_airflow_config()

    raise ValueError(
        "Unsupported ENV. Expected 'local' or 'ci', "
        "or leave ENV unset for MWAA. "
        f"Got: {environment}"
    )


CONFIG = load_config()
