from typing import Any

import requests

from config.config import CONFIG
from src.utils.logger import Logger

logger = Logger.get_logger(
    "coingecko_client",
    "coingecko_client.log",
)


class CoinGeckoClient:
    """Client for extracting market data from CoinGecko API."""

    def fetch_market_data(self) -> list[dict[str, Any]]:
        """Retrieve market data from CoinGecko API."""

        coingecko_config = CONFIG["coingecko"]

        base_url = str(coingecko_config["base_url"])
        vs_currency = str(coingecko_config["vs_currency"])

        params = {
            "vs_currency": vs_currency,
        }

        logger.info("Calling CoinGecko market API.")

        try:
            response = requests.get(
                base_url,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise TypeError(
                    "Unexpected CoinGecko API response format"
                )

            if not data:
                raise ValueError(
                    "CoinGecko API returned empty response"
                )

            logger.info(
                "CoinGecko API extraction successful: %d records",
                len(data),
            )

            return data

        except requests.RequestException:
            logger.exception(
                "CoinGecko API request failed"
            )
            raise

        except (TypeError, ValueError):
            logger.exception(
                "Invalid CoinGecko API response"
            )
            raise