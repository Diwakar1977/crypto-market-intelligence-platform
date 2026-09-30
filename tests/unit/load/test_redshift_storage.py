from __future__ import annotations

from collections.abc import Generator
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from src.load.redshift_storage import RedshiftStorage

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def storage() -> Generator[RedshiftStorage, None, None]:
    """Create RedshiftStorage with mocked AWS client."""

    with patch("src.load.redshift_storage.boto3.client") as mock_client:
        storage = RedshiftStorage(
            host="example.redshift-serverless.amazonaws.com",
            port=5439,
            database="dev",
            aws_region="ap-south-1",
            workgroup="crypto-etl-prod-rs-workgroup",
        )

        storage._redshift_client = mock_client.return_value

        yield storage


@pytest.fixture
def mock_connection(storage: RedshiftStorage) -> MagicMock:
    """Attach a mocked Redshift connection."""

    connection = MagicMock()
    storage._connection = connection

    return connection


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


def test_initialization(storage: RedshiftStorage) -> None:
    assert storage.host == "example.redshift-serverless.amazonaws.com"
    assert storage.port == 5439
    assert storage.database == "dev"
    assert storage.aws_region == "ap-south-1"
    assert storage.workgroup == "crypto-etl-prod-rs-workgroup"
    assert storage._connection is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"host": ""}, "host must not be empty."),
        ({"port": 0}, "port must be greater than zero."),
        ({"database": ""}, "database must not be empty."),
        ({"aws_region": ""}, "aws_region must not be empty."),
        ({"workgroup": ""}, "workgroup must not be empty."),
    ],
)
def test_initialization_validation(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "host": "example.redshift-serverless.amazonaws.com",
        "port": 5439,
        "database": "dev",
        "aws_region": "ap-south-1",
        "workgroup": "crypto-etl-prod-rs-workgroup",
    }

    values.update(kwargs)

    with patch("src.load.redshift_storage.boto3.client"):
        with pytest.raises(ValueError, match=message):
            RedshiftStorage(**values)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# IAM Credentials
# ------------------------------------------------------------------


def test_get_iam_credentials(storage: RedshiftStorage) -> None:
    storage._redshift_client.get_credentials.return_value = {
        "dbUser": "IAMR:crypto-user",
        "dbPassword": "temporary-password",
    }

    username, password = storage._get_iam_credentials()

    assert username == "IAMR:crypto-user"
    assert password == "temporary-password"

    storage._redshift_client.get_credentials.assert_called_once_with(
        workgroupName="crypto-etl-prod-rs-workgroup",
        dbName="dev",
        durationSeconds=900,
    )


def test_get_iam_credentials_failure(storage: RedshiftStorage) -> None:
    storage._redshift_client.get_credentials.side_effect = RuntimeError(
        "AWS credentials error"
    )

    with pytest.raises(RuntimeError, match="AWS credentials error"):
        storage._get_iam_credentials()


# ------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------


@patch("src.load.redshift_storage.redshift_connector.connect")
def test_connect(
    mock_connect: MagicMock,
    storage: RedshiftStorage,
) -> None:
    storage._redshift_client.get_credentials.return_value = {
        "dbUser": "IAMR:crypto-user",
        "dbPassword": "temporary-password",
    }

    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection

    storage.connect()

    assert storage._connection is mock_connection

    mock_connect.assert_called_once_with(
        host="example.redshift-serverless.amazonaws.com",
        port=5439,
        database="dev",
        user="IAMR:crypto-user",
        password="temporary-password",
        ssl=True,
        timeout=60,
        tcp_keepalive=True,
    )


@patch("src.load.redshift_storage.redshift_connector.connect")
def test_connect_reuses_existing_connection(
    mock_connect: MagicMock,
    storage: RedshiftStorage,
) -> None:
    existing_connection = MagicMock()
    storage._connection = existing_connection

    storage.connect()

    assert storage._connection is existing_connection
    mock_connect.assert_not_called()


@patch("src.load.redshift_storage.redshift_connector.connect")
def test_connect_failure(
    mock_connect: MagicMock,
    storage: RedshiftStorage,
) -> None:
    storage._redshift_client.get_credentials.return_value = {
        "dbUser": "IAMR:crypto-user",
        "dbPassword": "temporary-password",
    }

    mock_connect.side_effect = RuntimeError("connection failed")

    with pytest.raises(RuntimeError, match="connection failed"):
        storage.connect()

    assert storage._connection is None


# ------------------------------------------------------------------
# get_connection
# ------------------------------------------------------------------


def test_get_connection(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    result = storage.get_connection()

    assert result is mock_connection


@patch.object(RedshiftStorage, "connect")
def test_get_connection_creates_connection(
    mock_connect: MagicMock,
    storage: RedshiftStorage,
) -> None:
    connection = MagicMock()
    storage._connection = connection

    result = storage.get_connection()

    assert result is connection
    mock_connect.assert_not_called()


# ------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------


def test_execute(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    cursor = MagicMock()
    mock_connection.cursor.return_value = cursor

    storage.execute(
        "CREATE TABLE test_table (id INTEGER)",
    )

    cursor.execute.assert_called_once_with(
        "CREATE TABLE test_table (id INTEGER)",
        None,
    )

    mock_connection.commit.assert_called_once()
    cursor.close.assert_called_once()


def test_execute_with_parameters(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    cursor = MagicMock()
    mock_connection.cursor.return_value = cursor

    parameters = (123, "bitcoin")

    storage.execute(
        "INSERT INTO crypto_market VALUES (%s, %s)",
        parameters,
    )

    cursor.execute.assert_called_once_with(
        "INSERT INTO crypto_market VALUES (%s, %s)",
        parameters,
    )

    mock_connection.commit.assert_called_once()
    cursor.close.assert_called_once()


def test_execute_empty_sql(storage: RedshiftStorage) -> None:
    with pytest.raises(ValueError, match="SQL statement cannot be empty"):
        storage.execute("")


def test_execute_failure_retries(
    storage: RedshiftStorage,
) -> None:
    first_connection = MagicMock()
    first_cursor = MagicMock()

    first_connection.cursor.return_value = first_cursor
    first_cursor.execute.side_effect = RuntimeError("temporary failure")

    second_connection = MagicMock()
    second_cursor = MagicMock()

    second_connection.cursor.return_value = second_cursor

    storage._connection = first_connection

    def get_connection_side_effect() -> MagicMock:
        if storage._connection is None:
            storage._connection = second_connection

        return cast(MagicMock, storage._connection)

    with patch.object(
        storage,
        "get_connection",
        side_effect=get_connection_side_effect,
    ):
        storage.execute("SELECT 1")

    first_connection.rollback.assert_called_once()
    first_connection.close.assert_called_once()
    second_connection.commit.assert_called_once()


# ------------------------------------------------------------------
# Execute Many
# ------------------------------------------------------------------


def test_execute_many(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    cursor = MagicMock()
    mock_connection.cursor.return_value = cursor

    parameters = [
        (1, "bitcoin"),
        (2, "ethereum"),
    ]

    storage.execute_many(
        "INSERT INTO crypto_market VALUES (%s, %s)",
        parameters,
    )

    cursor.executemany.assert_called_once_with(
        "INSERT INTO crypto_market VALUES (%s, %s)",
        parameters,
    )

    mock_connection.commit.assert_called_once()
    cursor.close.assert_called_once()


def test_execute_many_empty_sql(
    storage: RedshiftStorage,
) -> None:
    with pytest.raises(ValueError, match="SQL statement cannot be empty"):
        storage.execute_many("", [(1,)])


def test_execute_many_empty_parameters(
    storage: RedshiftStorage,
) -> None:
    with pytest.raises(ValueError, match="SQL parameters cannot be empty"):
        storage.execute_many("SELECT 1", [])


# ------------------------------------------------------------------
# Fetch One
# ------------------------------------------------------------------


def test_fetch_one(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    cursor = MagicMock()
    mock_connection.cursor.return_value = cursor

    cursor.fetchone.return_value = (
        1,
        "bitcoin",
    )

    result = storage.fetch_one(
        "SELECT id, symbol FROM crypto_market WHERE id = %s",
        (1,),
    )

    assert result == (
        1,
        "bitcoin",
    )

    cursor.execute.assert_called_once_with(
        "SELECT id, symbol FROM crypto_market WHERE id = %s",
        (1,),
    )

    cursor.close.assert_called_once()


def test_fetch_one_no_result(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    cursor = MagicMock()
    mock_connection.cursor.return_value = cursor

    cursor.fetchone.return_value = None

    result = storage.fetch_one("SELECT 1")

    assert result is None
    cursor.close.assert_called_once()


def test_fetch_one_empty_sql(
    storage: RedshiftStorage,
) -> None:
    with pytest.raises(ValueError, match="SQL query cannot be empty"):
        storage.fetch_one("")


# ------------------------------------------------------------------
# Fetch All
# ------------------------------------------------------------------


def test_fetch_all(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    cursor = MagicMock()
    mock_connection.cursor.return_value = cursor

    cursor.fetchall.return_value = [
        (1, "bitcoin"),
        (2, "ethereum"),
    ]

    result = storage.fetch_all(
        "SELECT id, symbol FROM crypto_market",
    )

    assert result == [
        (1, "bitcoin"),
        (2, "ethereum"),
    ]

    cursor.execute.assert_called_once_with(
        "SELECT id, symbol FROM crypto_market",
        None,
    )

    cursor.close.assert_called_once()


def test_fetch_all_empty_sql(
    storage: RedshiftStorage,
) -> None:
    with pytest.raises(ValueError, match="SQL query cannot be empty"):
        storage.fetch_all("")


# ------------------------------------------------------------------
# Commit / Rollback
# ------------------------------------------------------------------


def test_commit(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    storage.commit()

    mock_connection.commit.assert_called_once()


def test_commit_without_connection(
    storage: RedshiftStorage,
) -> None:
    storage.commit()

    assert storage._connection is None


def test_rollback(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    storage.rollback()

    mock_connection.rollback.assert_called_once()


def test_rollback_without_connection(
    storage: RedshiftStorage,
) -> None:
    storage.rollback()

    assert storage._connection is None


# ------------------------------------------------------------------
# Reset Connection
# ------------------------------------------------------------------


def test_reset_connection(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    storage._reset_connection()

    mock_connection.close.assert_called_once()
    assert storage._connection is None


def test_reset_connection_without_connection(
    storage: RedshiftStorage,
) -> None:
    storage._reset_connection()

    assert storage._connection is None


# ------------------------------------------------------------------
# Close
# ------------------------------------------------------------------


def test_close(
    storage: RedshiftStorage,
    mock_connection: MagicMock,
) -> None:
    storage.close()

    mock_connection.close.assert_called_once()
    assert storage._connection is None


def test_close_without_connection(
    storage: RedshiftStorage,
) -> None:
    storage.close()

    assert storage._connection is None


# ------------------------------------------------------------------
# Context Manager
# ------------------------------------------------------------------


@patch.object(RedshiftStorage, "connect")
def test_context_manager_enter(
    mock_connect: MagicMock,
    storage: RedshiftStorage,
) -> None:
    with storage as result:
        assert result is storage

    mock_connect.assert_called_once()


@patch.object(RedshiftStorage, "close")
def test_context_manager_exit(
    mock_close: MagicMock,
    storage: RedshiftStorage,
) -> None:
    storage.__exit__(
        None,
        None,
        None,
    )

    mock_close.assert_called_once()
