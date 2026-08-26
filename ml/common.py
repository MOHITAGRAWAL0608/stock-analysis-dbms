"""
ml/common.py
Model-agnostic shared logic for the ML training layer: feature loading,
labeling, train/test splitting, and persistence helpers (model file,
Models row, Predictions rows, outcome backfill).

Algorithm-specific training scripts (e.g. ml/train_rf.py) import from here
and supply their own model, hyperparams, and constants.
"""

import json
import os
from datetime import date

import numpy as np
import pandas as pd
import joblib

FORWARD_DAYS = 20       # trading days ahead used to build the label
BACKFILL_DAYS = 27      # sp_backfill_prediction_outcomes counts calendar days;
                        # 27 of them land on the 20th trading day

INDICATOR_COLS = ["rsi_14", "macd", "macd_hist", "sma_20", "bb_position", "sma_ratio"]
FEATURE_COLS = ["rsi_14", "bb_position", "sma_ratio", "close_to_sma20",
                "macd_to_close", "macd_hist_to_close", "volume_ratio"]


def label_signal(forward_return):
    return "BUY" if forward_return > 0 else "SELL"


# ---------------------------------------------------------------------
# 1. Load vw_ml_features
# ---------------------------------------------------------------------

def load_features(conn):
    print("[1] Reading vw_ml_features ...")
    cols = ["company_id", "ticker", "trade_date", "close_price", "volume"] + INDICATOR_COLS
    cur = conn.cursor()
    cur.execute(f"SELECT {', '.join(cols)} FROM vw_ml_features")
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    cur.close()
    print(f"    {len(df):,} rows read")

    numeric_cols = ["close_price", "volume"] + INDICATOR_COLS
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["company_id", "trade_date"]).reset_index(drop=True)
    vol_avg = df.groupby("company_id")["volume"].transform(lambda s: s.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / vol_avg
    df["close_to_sma20"] = df["close_price"] / df["sma_20"]
    df["macd_to_close"] = df["macd"] / df["close_price"]
    df["macd_hist_to_close"] = df["macd_hist"] / df["close_price"]

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"    {len(df):,} rows remain after dropping warm-up (NULL indicator) rows")
    return df


# ---------------------------------------------------------------------
# 2. Labels — forward return per company
# ---------------------------------------------------------------------

def build_labels(df):
    print(f"[2] Building {FORWARD_DAYS}-trading-day forward-return direction labels ...")
    df = df.sort_values(["company_id", "trade_date"]).reset_index(drop=True)

    future_close = df.groupby("company_id")["close_price"].shift(-FORWARD_DAYS)
    df["forward_return"] = (future_close - df["close_price"]) / df["close_price"]

    before = len(df)
    df = df.dropna(subset=["forward_return"])
    print(f"    dropped {before - len(df):,} rows (last {FORWARD_DAYS} trading days per company, no forward data)")

    df["signal_type"] = df["forward_return"].apply(label_signal)
    return df


# ---------------------------------------------------------------------
# 3. Time-based train/test split
# ---------------------------------------------------------------------

def _date_cutoff_split(df, fraction):
    unique_dates = np.sort(df["trade_date"].unique())
    cutoff_date = unique_dates[int(len(unique_dates) * fraction)]
    part1 = df[df["trade_date"] < cutoff_date]
    part2 = df[df["trade_date"] >= cutoff_date]
    return part1, part2, cutoff_date


def time_split(df):
    print("[3] Splitting train/test by date (80/20, no shuffling) ...")
    train_df, test_df, cutoff_date = _date_cutoff_split(df, 0.8)
    print(f"    cutoff date: {cutoff_date}  (train < cutoff, test >= cutoff)")
    print(f"    train: {len(train_df):,} rows, {train_df['trade_date'].min()} to {train_df['trade_date'].max()}")
    print(f"    test:  {len(test_df):,} rows, {test_df['trade_date'].min()} to {test_df['trade_date'].max()}")
    return train_df, test_df


def val_split(train_df, fraction=0.85):
    """Carve a time-based validation slice out of a training set only —
    never call this on the test set. Same date-cutoff approach as
    time_split, generalized to an arbitrary split fraction."""
    print(f"    Splitting validation slice by date ({int(fraction*100)}/{int(round((1 - fraction) * 100))}, no shuffling) ...")
    fit_df, val_df, cutoff_date = _date_cutoff_split(train_df, fraction)
    print(f"    cutoff date: {cutoff_date}  (fit < cutoff, val >= cutoff)")
    print(f"    fit: {len(fit_df):,} rows, {fit_df['trade_date'].min()} to {fit_df['trade_date'].max()}")
    print(f"    val: {len(val_df):,} rows, {val_df['trade_date'].min()} to {val_df['trade_date'].max()}")
    return fit_df, val_df


# ---------------------------------------------------------------------
# Persist model to disk
# ---------------------------------------------------------------------

def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"[6] Model saved to {os.path.abspath(filename)}")


# ---------------------------------------------------------------------
# Write Models row
# ---------------------------------------------------------------------

def insert_model_row(conn, model_name, algorithm, train_df, hyperparams, accuracy, mae, rmse):
    print("[7] Inserting Models row ...")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO Models
            (model_name, algorithm, trained_on_date, train_start, train_end,
             hyperparams, test_accuracy, test_mae, test_rmse)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            model_id = LAST_INSERT_ID(model_id),
            hyperparams = VALUES(hyperparams),
            test_accuracy = VALUES(test_accuracy),
            test_mae = VALUES(test_mae),
            test_rmse = VALUES(test_rmse)
        """,
        (
            model_name,
            algorithm,
            date.today(),
            train_df["trade_date"].min(),
            train_df["trade_date"].max(),
            json.dumps(hyperparams),
            round(accuracy, 4),
            round(mae, 6),
            round(rmse, 6),
        ),
    )
    conn.commit()
    model_id = cur.lastrowid
    cur.close()
    print(f"    model_id = {model_id}")
    return model_id


# ---------------------------------------------------------------------
# Write Predictions (test set only)
# ---------------------------------------------------------------------

def insert_predictions(conn, model_id, test_df, y_pred, confidence):
    print("[8] Inserting Predictions (test set only) ...")
    # Every row here already excludes the last FORWARD_DAYS trading days per
    # company (dropped in build_labels while computing forward_return), so
    # each one is guaranteed to have >= FORWARD_DAYS DailyPrices rows after
    # it — sp_backfill_prediction_outcomes can score all of them immediately.
    rows = [
        (
            int(company_id),
            model_id,
            trade_date,
            signal,
            round(float(conf), 4),
        )
        for company_id, trade_date, signal, conf in zip(
            test_df["company_id"], test_df["trade_date"], y_pred, confidence
        )
    ]

    cur = conn.cursor()
    cur.execute("DELETE FROM Predictions WHERE model_id = %s", (model_id,))
    cur.executemany(
        """
        INSERT INTO Predictions (company_id, model_id, trade_date, signal_type, confidence)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            signal_type = VALUES(signal_type),
            confidence  = VALUES(confidence)
        """,
        rows,
    )
    conn.commit()
    cur.close()
    print(f"    {len(rows):,} predictions inserted/updated")


# ---------------------------------------------------------------------
# Backfill outcomes
# ---------------------------------------------------------------------

def backfill(conn):
    print(f"[9] CALL sp_backfill_prediction_outcomes({BACKFILL_DAYS}) ...")
    cur = conn.cursor()
    cur.callproc("sp_backfill_prediction_outcomes", (BACKFILL_DAYS,))
    rows_backfilled = None
    for result in cur.stored_results():
        row = result.fetchone()
        rows_backfilled = row[0] if row else None
    conn.commit()
    cur.close()
    print(f"    rows_backfilled = {rows_backfilled}")
