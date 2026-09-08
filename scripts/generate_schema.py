from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schema.schema_inferer import SchemaInferer
from src.schema.schema_manager import SchemaManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_FILE = PROJECT_ROOT / "data" / "sample" / "crypto_market_sample.ndjson"

SCHEMA_FILE = PROJECT_ROOT / "schemas" / "crypto_market_schema.json"

def load_raw_data(
    file_path: Path
) -> list[dict[str, Any]]:
    """Load raw NDJSON records."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {file_path}"
        )

    records: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue
            
            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line"
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Record on line {line_number}"
                    "must be a JSON object."
                )

            records.append(record)

        return records

def generate_schema() -> None:
    """Generate and store the normalized dataset schema."""

    print("Starting schema generation....")

    records = load_raw_data(RAW_DATA_FILE)

    print(f"Loaded {len(records)} raw records.")

    inferer = SchemaInferer()
    manager = SchemaManager()

    inferred_schema = inferer.infer(records)

    normalized_schema = manager.normalize(
        inferred_schema
    )

    manager.validate(normalized_schema)

    SCHEMA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with SCHEMA_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            normalized_schema,
            file,
            indent=4,
            sort_keys=False
        )

    print("Schema generation completed.")
    print(f"Schema saved to:: {SCHEMA_FILE}")

if __name__ == "__main__":
    generate_schema()