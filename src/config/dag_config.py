from __future__ import annotations

from datetime import timedelta

from airflow.sdk import timezone

# ---------------------------------------------------
# DAG CONFIGURATION
# ---------------------------------------------------

DAG_ID = "crypto_etl_pipeline"

DAG_DESCRIPTION = (
    "Crypto Market ETL Pipeline: "
    "CoinGecko API -> S3 RAW -> EMR Spark ->"
    "S3 PROCESSED -> Redshift"
)

DAG_OWNER = "DIWAKAR K"

DAG_TAGS = ["crypto", "etl", "emr", "spark", "redshift"]

# ---------------------------------------------------
# SCHEDULE
# ---------------------------------------------------

DAG_SCHEDULE = "0 9 * * *"

# ---------------------------------------------------
# START DATE
# ---------------------------------------------------

DAG_START_DATE = timezone.datetime(2026, 9, 25)

# ---------------------------------------------------
# DAG BEHAVIOUR
# ---------------------------------------------------

DAG_CATCHUP = False

DAG_MAX_ACTIVE_RUNS = 1

DAG_MAX_ACTIVE_TASKS = 5

# ---------------------------------------------------
# DEFAULT TASK ARGUMENTS
# ---------------------------------------------------

DAG_DEFAULT_ARGS = {
    "owner": DAG_OWNER,
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------
# AIRFLOW CONNECTIONS
# ---------------------------------------------------

AWS_CONNECTION_ID = "aws_default"

REDSHIFT_CONNECTION_ID = "redshift_default"

# ---------------------------------------------------
# NOTIFICATIONS
# ---------------------------------------------------

NOTIFICATION_ON_FAILURE = True

NOTIFICATION_ON_SUCCESS = True
