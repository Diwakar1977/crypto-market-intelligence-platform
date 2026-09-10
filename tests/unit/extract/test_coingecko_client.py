from unittest.mock import MagicMock, patch

import pytest
import requests

from src.extract.coingecko_client import CoinGeckoClient


@patch("src.extract.coingecko_client.requests.get")
def test_fetch_market_data_success(
    mock_get: MagicMock,
) -> None:
    """Test successful CoinGecko market data extraction."""

    mock_response = MagicMock()

    mock_response.json.return_value = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 100000.0,
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "name": "Ethereum",
            "current_price": 4000.0,
        },
    ]

    mock_get.return_value = mock_response

    client = CoinGeckoClient()

    result = client.fetch_market_data()

    mock_get.assert_called_once_with(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={"vs_currency": "usd"},
        timeout=30,
    )

    mock_response.raise_for_status.assert_called_once()

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["id"] == "bitcoin"
    assert result[1]["id"] == "ethereum"


@patch("src.extract.coingecko_client.requests.get")
def test_fetch_market_data_request_error(
    mock_get: MagicMock,
) -> None:
    """Test CoinGecko API request failure."""

    mock_get.side_effect = requests.RequestException("Connection error")

    client = CoinGeckoClient()

    with pytest.raises(
        requests.RequestException,
        match="Connection error",
    ):
        client.fetch_market_data()


@patch("src.extract.coingecko_client.requests.get")
def test_fetch_market_data_invalid_response(
    mock_get: MagicMock,
) -> None:
    """Test invalid CoinGecko API response format."""

    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "Invalid response"}
    mock_get.return_value = mock_response

    client = CoinGeckoClient()

    with pytest.raises(
        TypeError,
        match="Unexpected CoinGecko API response format",
    ):
        client.fetch_market_data()


@patch("src.extract.coingecko_client.requests.get")
def test_fetch_market_data_empty_response(
    mock_get: MagicMock,
) -> None:
    """Test empty CoinGecko API response."""

    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    client = CoinGeckoClient()

    with pytest.raises(
        ValueError,
        match="CoinGecko API returned empty response",
    ):
        client.fetch_market_data()
