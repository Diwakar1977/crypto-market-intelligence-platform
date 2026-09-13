#!/bin/bash

set -e

CONFIG_FILE="crypto_etl_config.json"
VARIABLE_NAME="crypto_etl_config"

while true; do
    inotifywait -e close_write,moved_to,create "$CONFIG_FILE"

    echo "Configuration changed. Updating Airflow Variable..."

    python -m json.tool "$CONFIG_FILE" > /dev/null

    airflow variables set \
        "$VARIABLE_NAME" \
        "$(cat "$CONFIG_FILE")"

    echo "Airflow Variable updated successfully."
done