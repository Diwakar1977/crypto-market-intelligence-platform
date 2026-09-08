from types import TracebackType
from typing import Any, Self

import boto3
import redshift_connector
from botocore.config import Config
from redshift_connector import Connection, Cursor

from src.utils.logger import Logger


logger = Logger.get_logger(
    "redshift_storage",
    "redshift_storage.log",
)


class RedshiftStorage:
    """Manage IAM-authenticated Amazon Redshift Serverless operations."""

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    REDSHIFT_CONNECT_TIMEOUT = 60
    AWS_CONNECT_TIMEOUT = 10
    AWS_READ_TIMEOUT = 60
    IAM_CREDENTIAL_DURATION = 900
    MAX_SQL_ATTEMPTS = 2

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        aws_region: str,
        workgroup: str,
    ) -> None:
        """Initialize Redshift Serverless configuration."""

        if not host:
            raise ValueError("host must not be empty.")

        if port <= 0:
            raise ValueError("port must be greater than zero.")

        if not database:
            raise ValueError("database must not be empty.")

        if not aws_region:
            raise ValueError("aws_region must not be empty.")

        if not workgroup:
            raise ValueError("workgroup must not be empty.")

        self.host = host
        self.port = port
        self.database = database
        self.aws_region = aws_region
        self.workgroup = workgroup

        self._connection: Connection | None = None

        self._redshift_client = boto3.client(
            "redshift-serverless",
            region_name=self.aws_region,
            config=Config(
                retries={
                    "max_attempts": 3,
                    "mode": "standard",
                },
                connect_timeout=self.AWS_CONNECT_TIMEOUT,
                read_timeout=self.AWS_READ_TIMEOUT,
            ),
        )

        logger.info(
            "RedshiftStorage initialized. "
            "workgroup=%s database=%s region=%s",
            self.workgroup,
            self.database,
            self.aws_region,
        )

    # ------------------------------------------------------------------
    # IAM Credentials
    # ------------------------------------------------------------------

    def _get_iam_credentials(self) -> tuple[str, str]:
        """
        Obtain temporary Redshift IAM database credentials.

        Returns:
            Tuple containing database username and password.
        """

        try:
            logger.info(
                "Requesting temporary Redshift IAM credentials "
                "for workgroup: %s",
                self.workgroup,
            )

            credentials = self._redshift_client.get_credentials(
                workgroupName=self.workgroup,
                durationSeconds=self.IAM_CREDENTIAL_DURATION,
            )

            username = credentials["dbUser"]
            password = credentials["dbPassword"]

            logger.info(
                "Temporary Redshift IAM credentials "
                "obtained successfully."
            )

            return username, password

        except Exception:
            logger.exception(
                "Failed to obtain temporary Redshift IAM credentials."
            )
            raise

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Create an IAM-authenticated Redshift connection.

        If a connection already exists, it is reused.
        """

        if self._connection is not None:
            return

        try:
            logger.info(
                "Connecting to Redshift Serverless: %s:%s/%s",
                self.host,
                self.port,
                self.database,
            )

            username, password = self._get_iam_credentials()

            self._connection = redshift_connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=username,
                password=password,
                timeout=self.REDSHIFT_CONNECT_TIMEOUT,
            )

            logger.info(
                "Redshift Serverless IAM connection "
                "established successfully."
            )

        except Exception:
            self._connection = None

            logger.exception(
                "Failed to connect to Redshift Serverless "
                "using IAM authentication."
            )
            raise

    def get_connection(self) -> Connection:
        """
        Return the active Redshift connection.

        Creates a connection when one does not already exist.
        """

        if self._connection is None:
            self.connect()

        if self._connection is None:
            raise RuntimeError(
                "Redshift connection could not be established."
            )

        return self._connection

    def _reset_connection(self) -> None:
        """
        Close and reset the current Redshift connection.

        This is used when the socket becomes stale, times out,
        or the server closes the connection.
        """

        if self._connection is None:
            return

        logger.warning(
            "Resetting Redshift connection."
        )

        try:
            self._connection.close()

        except Exception:
            logger.warning(
                "Failed to close broken Redshift connection.",
                exc_info=True,
            )

        finally:
            self._connection = None

    # ------------------------------------------------------------------
    # Execute SQL
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] | None = None,
    ) -> None:
        """
        Execute one SQL statement and commit the transaction.

        The statement is retried once if the Redshift connection
        fails or times out.
        """

        if not sql.strip():
            raise ValueError(
                "SQL statement cannot be empty."
            )

        last_exception: Exception | None = None

        for attempt in range(1, self.MAX_SQL_ATTEMPTS + 1):
            connection = self.get_connection()
            cursor: Cursor | None = None

            try:
                logger.info(
                    "Executing Redshift SQL statement "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                cursor = connection.cursor()

                cursor.execute(
                    sql,
                    parameters,
                )

                connection.commit()

                logger.info(
                    "Redshift SQL statement "
                    "executed successfully."
                )

                return

            except Exception as exc:
                last_exception = exc

                logger.exception(
                    "Redshift SQL execution failed "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                try:
                    connection.rollback()

                except Exception:
                    logger.warning(
                        "Failed to roll back Redshift transaction.",
                        exc_info=True,
                    )

                if attempt < self.MAX_SQL_ATTEMPTS:
                    logger.warning(
                        "Redshift SQL execution failed. "
                        "Resetting connection before retry."
                    )

                    self._reset_connection()

            finally:
                if cursor is not None:
                    try:
                        cursor.close()

                    except Exception:
                        logger.warning(
                            "Failed to close Redshift cursor.",
                            exc_info=True,
                        )

        if last_exception is not None:
            raise last_exception

        raise RuntimeError(
            "Redshift SQL execution failed."
        )

    # ------------------------------------------------------------------
    # Execute Many
    # ------------------------------------------------------------------

    def execute_many(
        self,
        sql: str,
        parameters: list[tuple[Any, ...]],
    ) -> None:
        """
        Execute a SQL statement for multiple parameter sets.

        The batch is retried once if the connection fails.

        Note:
            For non-idempotent INSERT operations, retrying after an
            ambiguous network failure can potentially duplicate data.
            Prefer idempotent SQL/load strategies for production.
        """

        if not sql.strip():
            raise ValueError(
                "SQL statement cannot be empty."
            )

        if not parameters:
            raise ValueError(
                "SQL parameters cannot be empty."
            )

        last_exception: Exception | None = None

        for attempt in range(1, self.MAX_SQL_ATTEMPTS + 1):
            connection = self.get_connection()
            cursor: Cursor | None = None

            try:
                logger.info(
                    "Executing batch Redshift SQL statement "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                cursor = connection.cursor()

                cursor.executemany(
                    sql,
                    parameters,
                )

                connection.commit()

                logger.info(
                    "Batch Redshift SQL execution "
                    "completed successfully."
                )

                return

            except Exception as exc:
                last_exception = exc

                logger.exception(
                    "Batch Redshift SQL execution failed "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                try:
                    connection.rollback()

                except Exception:
                    logger.warning(
                        "Failed to roll back batch transaction.",
                        exc_info=True,
                    )

                if attempt < self.MAX_SQL_ATTEMPTS:
                    logger.warning(
                        "Batch SQL execution failed. "
                        "Resetting connection before retry."
                    )

                    self._reset_connection()

            finally:
                if cursor is not None:
                    try:
                        cursor.close()

                    except Exception:
                        logger.warning(
                            "Failed to close Redshift cursor.",
                            exc_info=True,
                        )

        if last_exception is not None:
            raise last_exception

        raise RuntimeError(
            "Batch Redshift SQL execution failed."
        )

    # ------------------------------------------------------------------
    # Fetch One
    # ------------------------------------------------------------------

    def fetch_one(
        self,
        sql: str,
        parameters: tuple[Any, ...] | None = None,
    ) -> tuple[Any, ...] | None:
        """
        Execute a SELECT query and return one row.

        The query is retried once if the Redshift connection fails.
        """

        if not sql.strip():
            raise ValueError(
                "SQL query cannot be empty."
            )

        last_exception: Exception | None = None

        for attempt in range(1, self.MAX_SQL_ATTEMPTS + 1):
            connection = self.get_connection()
            cursor: Cursor | None = None

            try:
                logger.info(
                    "Executing Redshift SELECT query "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                cursor = connection.cursor()

                cursor.execute(
                    sql,
                    parameters,
                )

                result = cursor.fetchone()

                logger.info(
                    "Redshift SELECT query "
                    "completed successfully."
                )

                return (
                    tuple(result)
                    if result is not None
                    else None
                )

            except Exception as exc:
                last_exception = exc

                logger.exception(
                    "Redshift SELECT query failed "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                if attempt < self.MAX_SQL_ATTEMPTS:
                    self._reset_connection()

            finally:
                if cursor is not None:
                    try:
                        cursor.close()

                    except Exception:
                        logger.warning(
                            "Failed to close Redshift cursor.",
                            exc_info=True,
                        )

        if last_exception is not None:
            raise last_exception

        raise RuntimeError(
            "Redshift SELECT query failed."
        )

    # ------------------------------------------------------------------
    # Fetch All
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        sql: str,
        parameters: tuple[Any, ...] | None = None,
    ) -> list[tuple[Any, ...]]:
        """
        Execute a SELECT query and return all rows.

        The query is retried once if the Redshift connection fails.
        """

        if not sql.strip():
            raise ValueError(
                "SQL query cannot be empty."
            )

        last_exception: Exception | None = None

        for attempt in range(1, self.MAX_SQL_ATTEMPTS + 1):
            connection = self.get_connection()
            cursor: Cursor | None = None

            try:
                logger.info(
                    "Executing Redshift SELECT query "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                cursor = connection.cursor()

                cursor.execute(
                    sql,
                    parameters,
                )

                result = cursor.fetchall()

                logger.info(
                    "Redshift SELECT query returned %d rows.",
                    len(result),
                )

                return list(result)

            except Exception as exc:
                last_exception = exc

                logger.exception(
                    "Redshift SELECT query failed "
                    "(attempt %d/%d).",
                    attempt,
                    self.MAX_SQL_ATTEMPTS,
                )

                if attempt < self.MAX_SQL_ATTEMPTS:
                    self._reset_connection()

            finally:
                if cursor is not None:
                    try:
                        cursor.close()

                    except Exception:
                        logger.warning(
                            "Failed to close Redshift cursor.",
                            exc_info=True,
                        )

        if last_exception is not None:
            raise last_exception

        raise RuntimeError(
            "Redshift SELECT query failed."
        )

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """Commit the current Redshift transaction."""

        if self._connection is None:
            logger.warning(
                "Commit requested but no Redshift connection exists."
            )
            return

        try:
            self._connection.commit()

            logger.info(
                "Redshift transaction committed successfully."
            )

        except Exception:
            logger.exception(
                "Failed to commit Redshift transaction."
            )
            raise

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> None:
        """Roll back the current Redshift transaction."""

        if self._connection is None:
            logger.warning(
                "Rollback requested but no Redshift connection exists."
            )
            return

        try:
            self._connection.rollback()

            logger.warning(
                "Redshift transaction rolled back."
            )

        except Exception:
            logger.exception(
                "Failed to roll back Redshift transaction."
            )
            raise

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Redshift connection safely."""

        if self._connection is None:
            return

        try:
            self._connection.close()

            logger.info(
                "Redshift connection closed successfully."
            )

        except Exception:
            logger.exception(
                "Failed to close Redshift connection."
            )

        finally:
            self._connection = None

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Open the Redshift connection."""

        self.connect()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the Redshift connection."""

        self.close()
