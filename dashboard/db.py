"""
dashboard/db.py
MySQL connection helper for the Streamlit dashboard. Reads vw_ml_features
and vw_prediction_accuracy plus the Models, Predictions, Companies and
DailyPrices tables.

Credentials come from a .env file in the project root, same convention as
ingest_and_indicators.py. Set DB_PASSWORD before running:

    PowerShell:  $env:DB_PASSWORD="your_password"
    CMD:         set DB_PASSWORD=your_password
    bash:        export DB_PASSWORD=your_password
"""

import os

import mysql.connector

from dotenv import load_dotenv
load_dotenv()

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
    # autocommit so every query opens a fresh snapshot. The connection is
    # cached for the life of the process, and under REPEATABLE READ it would
    # otherwise keep serving the view of the data it saw on its first query.
    return mysql.connector.connect(**DB, autocommit=True)
