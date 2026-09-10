import json
from pathlib import Path

from src.extract.coingecko_client import CoinGeckoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "crypto_market_sample.ndjson"


def main() -> None:
    """Fetch market data and save it as a local sample dataset."""

    client = CoinGeckoClient()

    data = client.fetch_market_data()

    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SAMPLE_DATA_PATH.open("w", encoding="utf-8") as file:
        for record in data:
            file.write(json.dumps(record) + "\n")

        print(f"Sample data save to :{SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()
