"""
dashboard/db.py
MySQL connection helper for the Streamlit dashboard. Reads vw_ml_features,
vw_latest_prices, vw_portfolio_detail, vw_prediction_accuracy, Predictions.

Credentials come from environment variables, same convention as
ingest_and_indicators.py. Set DB_PASSWORD before running:

    PowerShell:  $env:DB_PASSWORD="your_password"
    CMD:         set DB_PASSWORD=your_password
    bash:        export DB_PASSWORD=your_password
"""

import os

import mysql.connector

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "stockdb"),
)


def get_connection():
    if not DB["password"]:
        raise SystemExit(
            "DB_PASSWORD is not set.\n"
            '  PowerShell:  $env:DB_PASSWORD="your_password"\n'
            "  CMD:         set DB_PASSWORD=your_password\n"
            "  bash:        export DB_PASSWORD=your_password"
        )
    return mysql.connector.connect(**DB)
