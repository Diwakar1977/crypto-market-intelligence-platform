from __future__ import annotations

from datetime import timedelta

from dags import dag_config

# =====================================================================
# DAG IDENTITY
# =====================================================================


def test_dag_id() -> None:
    """DAG ID must be correctly configured."""

    assert dag_config.DAG_ID == "crypto_etl_pipeline"


def test_dag_description() -> None:
    """DAG description must describe the complete ETL pipeline."""

    assert dag_config.DAG_DESCRIPTION == (
        "Crypto Market ETL Pipeline: "
        "CoinGecko API -> S3 RAW -> EMR Spark ->"
        "S3 PROCESSED -> Redshift"
    )


def test_dag_owner() -> None:
    """DAG owner must be configured."""

    assert dag_config.DAG_OWNER == "DIWAKAR K"


def test_dag_tags() -> None:
    """DAG must contain the required tags."""

    assert dag_config.DAG_TAGS == [
        "crypto",
        "etl",
        "emr",
        "spark",
        "redshift",
    ]


# =====================================================================
# SCHEDULE
# =====================================================================


def test_dag_schedule() -> None:
    """DAG must run once every day at 09:00."""

    assert dag_config.DAG_SCHEDULE == "0 9 * * *"


# =====================================================================
# START DATE
# =====================================================================


def test_dag_start_date() -> None:
    """DAG start date must be September 1, 2026."""

    assert dag_config.DAG_START_DATE.year == 2026
    assert dag_config.DAG_START_DATE.month == 9
    assert dag_config.DAG_START_DATE.day == 23


# =====================================================================
# DAG BEHAVIOUR
# =====================================================================


def test_dag_catchup() -> None:
    """DAG catchup must be disabled."""

    assert dag_config.DAG_CATCHUP is False


def test_dag_max_active_runs() -> None:
    """Only one DAG run should be active at a time."""

    assert dag_config.DAG_MAX_ACTIVE_RUNS == 1


def test_dag_max_active_tasks() -> None:
    """Maximum active tasks must be configured."""

    assert dag_config.DAG_MAX_ACTIVE_TASKS == 5


# =====================================================================
# DEFAULT TASK ARGUMENTS
# =====================================================================


def test_dag_default_args() -> None:
    """Default task arguments must be correctly configured."""

    assert dag_config.DAG_DEFAULT_ARGS["owner"] == "DIWAKAR K"

    assert dag_config.DAG_DEFAULT_ARGS["depends_on_past"] is False

    assert dag_config.DAG_DEFAULT_ARGS["retries"] == 2

    assert dag_config.DAG_DEFAULT_ARGS["retry_delay"] == timedelta(minutes=5)


def test_dag_default_args_keys() -> None:
    """Default task arguments must contain required keys."""

    expected_keys = {
        "owner",
        "depends_on_past",
        "retries",
        "retry_delay",
    }

    assert set(dag_config.DAG_DEFAULT_ARGS.keys()) == expected_keys


# =====================================================================
# AIRFLOW CONNECTIONS
# =====================================================================


def test_aws_connection_id() -> None:
    """AWS Airflow connection ID must be configured."""

    assert dag_config.AWS_CONNECTION_ID == "aws_default"


def test_redshift_connection_id() -> None:
    """Redshift Airflow connection ID must be configured."""

    assert dag_config.REDSHIFT_CONNECTION_ID == "redshift_default"


# =====================================================================
# NOTIFICATIONS
# =====================================================================


def test_notification_on_failure() -> None:
    """Failure notification must be enabled."""

    assert dag_config.NOTIFICATION_ON_FAILURE is True


def test_notification_on_success() -> None:
    """Success notification must be enabled."""

    assert dag_config.NOTIFICATION_ON_SUCCESS is True


# =====================================================================
# CONFIGURATION COMPLETENESS
# =====================================================================


def test_required_dag_configuration_exists() -> None:
    """All required DAG configuration values must exist."""

    required_attributes = [
        "DAG_ID",
        "DAG_DESCRIPTION",
        "DAG_OWNER",
        "DAG_TAGS",
        "DAG_SCHEDULE",
        "DAG_START_DATE",
        "DAG_CATCHUP",
        "DAG_MAX_ACTIVE_RUNS",
        "DAG_MAX_ACTIVE_TASKS",
        "DAG_DEFAULT_ARGS",
        "AWS_CONNECTION_ID",
        "REDSHIFT_CONNECTION_ID",
        "NOTIFICATION_ON_FAILURE",
        "NOTIFICATION_ON_SUCCESS",
    ]

    for attribute in required_attributes:
        assert hasattr(dag_config, attribute), f"Missing DAG configuration: {attribute}"


# =====================================================================
# CONFIGURATION TYPE VALIDATION
# =====================================================================


def test_dag_configuration_types() -> None:
    """DAG configuration values must have expected types."""

    assert isinstance(dag_config.DAG_ID, str)

    assert isinstance(dag_config.DAG_DESCRIPTION, str)

    assert isinstance(dag_config.DAG_OWNER, str)

    assert isinstance(dag_config.DAG_TAGS, list)

    assert isinstance(dag_config.DAG_SCHEDULE, str)

    assert dag_config.DAG_START_DATE is not None

    assert isinstance(dag_config.DAG_CATCHUP, bool)

    assert isinstance(
        dag_config.DAG_MAX_ACTIVE_RUNS,
        int,
    )

    assert isinstance(
        dag_config.DAG_MAX_ACTIVE_TASKS,
        int,
    )

    assert isinstance(
        dag_config.DAG_DEFAULT_ARGS,
        dict,
    )

    assert isinstance(
        dag_config.AWS_CONNECTION_ID,
        str,
    )

    assert isinstance(
        dag_config.REDSHIFT_CONNECTION_ID,
        str,
    )

    assert isinstance(
        dag_config.NOTIFICATION_ON_FAILURE,
        bool,
    )

    assert isinstance(
        dag_config.NOTIFICATION_ON_SUCCESS,
        bool,
    )
