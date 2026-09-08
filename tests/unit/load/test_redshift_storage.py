from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.load.redshift_storage import RedshiftStorage


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def redshift_storage() -> RedshiftStorage:
    """Create RedshiftStorage with mocked AWS client."""
    with patch(
        "src.load.redshift_storage.boto3.client"
    ) as mock_boto3_client:

        mock_boto3_client.return_value = MagicMock()

        storage = RedshiftStorage(
            host="example.redshift-serverless.amazonaws.com",
            port=5439,
            database="dev",
            aws_region="ap-south-1",
            workgroup="crypto-etl-workgroup",
        )

        return storage


@pytest.fixture
def mock_redshift_connection() -> MagicMock:
    """Create a mocked Redshift connection."""
    connection = MagicMock()
    return connection


# =====================================================================
# INITIALIZATION
# =====================================================================


def test_init_success() -> None:
    """Initialize RedshiftStorage successfully."""

    with patch(
        "src.load.redshift_storage.boto3.client"
    ) as mock_boto3_client:

        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        storage = RedshiftStorage(
            host="redshift.example.com",
            port=5439,
            database="dev",
            aws_region="ap-south-1",
            workgroup="crypto-workgroup",
        )

        assert storage.host == "redshift.example.com"
        assert storage.port == 5439
        assert storage.database == "dev"
        assert storage.aws_region == "ap-south-1"
        assert storage.workgroup == "crypto-workgroup"
        assert storage._connection is None

        mock_boto3_client.assert_called_once()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host", ""),
        ("database", ""),
        ("aws_region", ""),
        ("workgroup", ""),
    ],
)
def test_init_rejects_empty_string(
    field: str,
    value: str,
) -> None:
    """Reject empty required string configuration."""

    kwargs = {
        "host": "redshift.example.com",
        "port": 5439,
        "database": "dev",
        "aws_region": "ap-south-1",
        "workgroup": "crypto-workgroup",
    }

    kwargs[field] = value

    with patch(
        "src.load.redshift_storage.boto3.client"
    ):
        with pytest.raises(ValueError):
            RedshiftStorage(**kwargs)


def test_init_rejects_invalid_port() -> None:
    """Reject zero or negative Redshift port."""

    with patch(
        "src.load.redshift_storage.boto3.client"
    ):
        with pytest.raises(
            ValueError,
            match="port must be greater than zero.",
        ):
            RedshiftStorage(
                host="redshift.example.com",
                port=0,
                database="dev",
                aws_region="ap-south-1",
                workgroup="crypto-workgroup",
            )


# =====================================================================
# IAM CREDENTIALS
# =====================================================================


def test_get_iam_credentials(
    redshift_storage: RedshiftStorage,
) -> None:
    """Return temporary Redshift IAM credentials."""

    redshift_storage._redshift_client.get_credentials.return_value = {
        "dbUser": "iam_user",
        "dbPassword": "temporary_password",
    }

    username, password = redshift_storage._get_iam_credentials()

    assert username == "iam_user"
    assert password == "temporary_password"

    redshift_storage._redshift_client.get_credentials.assert_called_once_with(
        workgroupName="crypto-etl-workgroup",
        durationSeconds=900,
    )


def test_get_iam_credentials_failure(
    redshift_storage: RedshiftStorage,
) -> None:
    """Raise exception when IAM credentials cannot be obtained."""

    redshift_storage._redshift_client.get_credentials.side_effect = (
        RuntimeError("AWS credentials error")
    )

    with pytest.raises(
        RuntimeError,
        match="AWS credentials error",
    ):
        redshift_storage._get_iam_credentials()


# =====================================================================
# CONNECTION
# =====================================================================


@patch("src.load.redshift_storage.redshift_connector.connect")
def test_connect_success(
    mock_connect: MagicMock,
    redshift_storage: RedshiftStorage,
) -> None:
    """Create Redshift IAM connection successfully."""

    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection

    redshift_storage._redshift_client.get_credentials.return_value = {
        "dbUser": "iam_user",
        "dbPassword": "temporary_password",
    }

    redshift_storage.connect()

    assert redshift_storage._connection is mock_connection

    mock_connect.assert_called_once_with(
        host="example.redshift-serverless.amazonaws.com",
        port=5439,
        database="dev",
        user="iam_user",
        password="temporary_password",
        timeout=60,
    )


@patch("src.load.redshift_storage.redshift_connector.connect")
def test_connect_reuses_existing_connection(
    mock_connect: MagicMock,
    redshift_storage: RedshiftStorage,
) -> None:
    """Reuse an existing connection."""

    existing_connection = MagicMock()

    redshift_storage._connection = existing_connection

    redshift_storage.connect()

    assert redshift_storage._connection is existing_connection
    mock_connect.assert_not_called()


@patch("src.load.redshift_storage.redshift_connector.connect")
def test_connect_failure_resets_connection(
    mock_connect: MagicMock,
    redshift_storage: RedshiftStorage,
) -> None:
    """Reset connection when connection creation fails."""

    mock_connect.side_effect = RuntimeError(
        "Connection failed"
    )

    redshift_storage._redshift_client.get_credentials.return_value = {
        "dbUser": "iam_user",
        "dbPassword": "temporary_password",
    }

    with pytest.raises(
        RuntimeError,
        match="Connection failed",
    ):
        redshift_storage.connect()

    assert redshift_storage._connection is None


# =====================================================================
# GET CONNECTION
# =====================================================================


@patch("src.load.redshift_storage.redshift_connector.connect")
def test_get_connection_creates_connection(
    mock_connect: MagicMock,
    redshift_storage: RedshiftStorage,
) -> None:
    """get_connection creates a connection when needed."""

    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection

    redshift_storage._redshift_client.get_credentials.return_value = {
        "dbUser": "iam_user",
        "dbPassword": "temporary_password",
    }

    result = redshift_storage.get_connection()

    assert result is mock_connection


def test_get_connection_returns_existing_connection(
    redshift_storage: RedshiftStorage,
) -> None:
    """get_connection returns existing connection."""

    existing_connection = MagicMock()

    redshift_storage._connection = existing_connection

    result = redshift_storage.get_connection()

    assert result is existing_connection


# =====================================================================
# RESET CONNECTION
# =====================================================================


def test_reset_connection(
    redshift_storage: RedshiftStorage,
) -> None:
    """Close and reset current connection."""

    connection = MagicMock()
    redshift_storage._connection = connection

    redshift_storage._reset_connection()

    connection.close.assert_called_once()
    assert redshift_storage._connection is None


def test_reset_connection_when_none(
    redshift_storage: RedshiftStorage,
) -> None:
    """Reset does nothing when connection is None."""

    redshift_storage._connection = None

    redshift_storage._reset_connection()

    assert redshift_storage._connection is None


def test_reset_connection_handles_close_error(
    redshift_storage: RedshiftStorage,
) -> None:
    """Reset connection even when close raises an exception."""

    connection = MagicMock()
    connection.close.side_effect = RuntimeError(
        "close failed"
    )

    redshift_storage._connection = connection

    redshift_storage._reset_connection()

    assert redshift_storage._connection is None


# =====================================================================
# EXECUTE
# =====================================================================


def test_execute_success(
    redshift_storage: RedshiftStorage,
) -> None:
    """Execute SQL and commit successfully."""

    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value = cursor
    redshift_storage._connection = connection

    sql = "CREATE TABLE test_table (id INTEGER)"

    redshift_storage.execute(sql)

    cursor.execute.assert_called_once_with(
        sql,
        None,
    )

    connection.commit.assert_called_once()
    cursor.close.assert_called_once()


def test_execute_with_parameters(
    redshift_storage: RedshiftStorage,
) -> None:
    """Execute SQL with parameters."""

    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value = cursor
    redshift_storage._connection = connection

    sql = "INSERT INTO test_table (id) VALUES (%s)"
    parameters = (1,)

    redshift_storage.execute(
        sql,
        parameters,
    )

    cursor.execute.assert_called_once_with(
        sql,
        parameters,
    )

    connection.commit.assert_called_once()


def test_execute_rejects_empty_sql(
    redshift_storage: RedshiftStorage,
) -> None:
    """Reject empty SQL."""

    with pytest.raises(
        ValueError,
        match="SQL statement cannot be empty.",
    ):
        redshift_storage.execute("")


def test_execute_retries_after_failure(
    redshift_storage: RedshiftStorage,
) -> None:
    """Retry SQL execution after connection failure."""

    first_connection = MagicMock()
    first_cursor = MagicMock()

    second_connection = MagicMock()
    second_cursor = MagicMock()

    first_connection.cursor.return_value = first_cursor
    second_connection.cursor.return_value = second_cursor

    first_cursor.execute.side_effect = RuntimeError(
        "Broken pipe"
    )

    redshift_storage._connection = first_connection

    connections = [
        first_connection,
        second_connection,
    ]

    def get_connection_side_effect() -> MagicMock:
        connection = connections.pop(0)
        redshift_storage._connection = connection
        return connection

    with patch.object(
        redshift_storage,
        "get_connection",
        side_effect=get_connection_side_effect,
    ):
        with patch.object(
            redshift_storage,
            "_reset_connection",
            side_effect=lambda: setattr(
                redshift_storage,
                "_connection",
                None,
            ),
        ):
            redshift_storage.execute(
                "SELECT 1"
            )

    first_connection.rollback.assert_called_once()
    second_connection.commit.assert_called_once()
    second_cursor.execute.assert_called_once_with(
        "SELECT 1",
        None,
    )


# =====================================================================
# EXECUTE MANY
# =====================================================================


def test_execute_many_success(
    redshift_storage: RedshiftStorage,
) -> None:
    """Execute batch SQL successfully."""

    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value = cursor
    redshift_storage._connection = connection

    sql = "INSERT INTO test_table (id) VALUES (%s)"

    parameters = [
        (1,),
        (2,),
        (3,),
    ]

    redshift_storage.execute_many(
        sql,
        parameters,
    )

    cursor.executemany.assert_called_once_with(
        sql,
        parameters,
    )

    connection.commit.assert_called_once()
    cursor.close.assert_called_once()


def test_execute_many_rejects_empty_sql(
    redshift_storage: RedshiftStorage,
) -> None:
    """Reject empty batch SQL."""

    with pytest.raises(
        ValueError,
        match="SQL statement cannot be empty.",
    ):
        redshift_storage.execute_many(
            "",
            [(1,)],
        )


def test_execute_many_rejects_empty_parameters(
    redshift_storage: RedshiftStorage,
) -> None:
    """Reject empty parameter list."""

    with pytest.raises(
        ValueError,
        match="SQL parameters cannot be empty.",
    ):
        redshift_storage.execute_many(
            "INSERT INTO test_table VALUES (%s)",
            [],
        )


# =====================================================================
# FETCH ONE
# =====================================================================


def test_fetch_one_returns_row(
    redshift_storage: RedshiftStorage,
) -> None:
    """Fetch one row successfully."""

    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value = cursor
    cursor.fetchone.return_value = (
        1,
        "bitcoin",
    )

    redshift_storage._connection = connection

    result = redshift_storage.fetch_one(
        "SELECT id, name FROM crypto_market"
    )

    assert result == (
        1,
        "bitcoin",
    )

    cursor.execute.assert_called_once_with(
        "SELECT id, name FROM crypto_market",
        None,
    )

    cursor.close.assert_called_once()


def test_fetch_one_returns_none(
    redshift_storage: RedshiftStorage,
) -> None:
    """Return None when query has no rows."""

    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value = cursor
    cursor.fetchone.return_value = None

    redshift_storage._connection = connection

    result = redshift_storage.fetch_one(
        "SELECT * FROM crypto_market"
    )

    assert result is None


def test_fetch_one_rejects_empty_sql(
    redshift_storage: RedshiftStorage,
) -> None:
    """Reject empty SELECT query."""

    with pytest.raises(
        ValueError,
        match="SQL query cannot be empty.",
    ):
        redshift_storage.fetch_one("")


# =====================================================================
# FETCH ALL
# =====================================================================


def test_fetch_all_returns_rows(
    redshift_storage: RedshiftStorage,
) -> None:
    """Fetch all rows successfully."""

    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value = cursor

    cursor.fetchall.return_value = [
        (1, "bitcoin"),
        (2, "ethereum"),
    ]

    redshift_storage._connection = connection

    result = redshift_storage.fetch_all(
        "SELECT id, name FROM crypto_market"
    )

    assert result == [
        (1, "bitcoin"),
        (2, "ethereum"),
    ]

    cursor.execute.assert_called_once_with(
        "SELECT id, name FROM crypto_market",
        None,
    )

    cursor.close.assert_called_once()


def test_fetch_all_returns_empty_list(
    redshift_storage: RedshiftStorage,
) -> None:
    """Return empty list when query has no rows."""

    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = []

    redshift_storage._connection = connection

    result = redshift_storage.fetch_all(
        "SELECT * FROM crypto_market"
    )

    assert result == []


def test_fetch_all_rejects_empty_sql(
    redshift_storage: RedshiftStorage,
) -> None:
    """Reject empty SELECT query."""

    with pytest.raises(
        ValueError,
        match="SQL query cannot be empty.",
    ):
        redshift_storage.fetch_all("")


# =====================================================================
# COMMIT
# =====================================================================


def test_commit_success(
    redshift_storage: RedshiftStorage,
) -> None:
    """Commit current transaction."""

    connection = MagicMock()
    redshift_storage._connection = connection

    redshift_storage.commit()

    connection.commit.assert_called_once()


def test_commit_without_connection(
    redshift_storage: RedshiftStorage,
) -> None:
    """Commit does nothing when no connection exists."""

    redshift_storage._connection = None

    redshift_storage.commit()

    assert redshift_storage._connection is None


def test_commit_failure(
    redshift_storage: RedshiftStorage,
) -> None:
    """Raise exception when commit fails."""

    connection = MagicMock()
    connection.commit.side_effect = RuntimeError(
        "commit failed"
    )

    redshift_storage._connection = connection

    with pytest.raises(
        RuntimeError,
        match="commit failed",
    ):
        redshift_storage.commit()


# =====================================================================
# ROLLBACK
# =====================================================================


def test_rollback_success(
    redshift_storage: RedshiftStorage,
) -> None:
    """Rollback current transaction."""

    connection = MagicMock()
    redshift_storage._connection = connection

    redshift_storage.rollback()

    connection.rollback.assert_called_once()


def test_rollback_without_connection(
    redshift_storage: RedshiftStorage,
) -> None:
    """Rollback does nothing when no connection exists."""

    redshift_storage._connection = None

    redshift_storage.rollback()

    assert redshift_storage._connection is None


def test_rollback_failure(
    redshift_storage: RedshiftStorage,
) -> None:
    """Raise exception when rollback fails."""

    connection = MagicMock()
    connection.rollback.side_effect = RuntimeError(
        "rollback failed"
    )

    redshift_storage._connection = connection

    with pytest.raises(
        RuntimeError,
        match="rollback failed",
    ):
        redshift_storage.rollback()


# =====================================================================
# CLOSE
# =====================================================================


def test_close_success(
    redshift_storage: RedshiftStorage,
) -> None:
    """Close Redshift connection successfully."""

    connection = MagicMock()
    redshift_storage._connection = connection

    redshift_storage.close()

    connection.close.assert_called_once()
    assert redshift_storage._connection is None


def test_close_without_connection(
    redshift_storage: RedshiftStorage,
) -> None:
    """Close does nothing when connection does not exist."""

    redshift_storage._connection = None

    redshift_storage.close()

    assert redshift_storage._connection is None


def test_close_handles_exception(
    redshift_storage: RedshiftStorage,
) -> None:
    """Connection is reset even when close fails."""

    connection = MagicMock()
    connection.close.side_effect = RuntimeError(
        "close failed"
    )

    redshift_storage._connection = connection

    redshift_storage.close()

    assert redshift_storage._connection is None


# =====================================================================
# CONTEXT MANAGER
# =====================================================================


@patch.object(RedshiftStorage, "connect")
def test_context_manager_enter(
    mock_connect: MagicMock,
    redshift_storage: RedshiftStorage,
) -> None:
    """__enter__ should connect and return self."""

    result = redshift_storage.__enter__()

    mock_connect.assert_called_once()
    assert result is redshift_storage


@patch.object(RedshiftStorage, "close")
def test_context_manager_exit(
    mock_close: MagicMock,
    redshift_storage: RedshiftStorage,
) -> None:
    """__exit__ should close the connection."""

    redshift_storage.__exit__(
        None,
        None,
        None,
    )

    mock_close.assert_called_once()


def test_context_manager_usage(
    redshift_storage: RedshiftStorage,
) -> None:
    """Verify context manager behavior."""

    with patch.object(
        redshift_storage,
        "connect",
    ) as mock_connect:
        with patch.object(
            redshift_storage,
            "close",
        ) as mock_close:

            with redshift_storage as result:
                assert result is redshift_storage

            mock_connect.assert_called_once()
            mock_close.assert_called_once()
