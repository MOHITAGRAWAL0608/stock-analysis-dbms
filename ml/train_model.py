"""
ml/train_model.py
Trains a RandomForest BUY/SELL/HOLD signal classifier on vw_ml_features,
evaluates it with a time-based holdout split, and writes results back to
the DB per the "Interface with the ML layer" contract in README.md:

    read vw_ml_features -> train -> write Models -> write Predictions
    -> CALL sp_backfill_prediction_outcomes(5)

Run:
    python ml/train_model.py
"""

import json
import os
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_connection  # noqa: E402  (ml/db.py)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
FORWARD_DAYS = 5        # trading days ahead used to build the label
HOLD_THRESHOLD = 0.01   # +-1%, must match sp_backfill_prediction_outcomes
                         # in 03_views_procedures_triggers.sql — keep in sync

INDICATOR_COLS = [
    "rsi_14", "macd", "macd_hist", "sma_20", "sma_50", "ema_12",
    "bb_position", "sma_ratio",
]
FEATURE_COLS = INDICATOR_COLS + ["volume"]

MODEL_NAME = "rf_v1"
ALGORITHM = "RandomForest"
HYPERPARAMS = dict(
    n_estimators=300, max_depth=12, class_weight="balanced", random_state=42
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "rf_model.joblib")


def label_signal(forward_return):
    if forward_return > HOLD_THRESHOLD:
        return "BUY"
    if forward_return < -HOLD_THRESHOLD:
        return "SELL"
    return "HOLD"


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

    df = df.dropna(subset=INDICATOR_COLS)
    print(f"    {len(df):,} rows remain after dropping warm-up (NULL indicator) rows")
    return df


# ---------------------------------------------------------------------
# 2. Labels — 5-trading-day forward return per company
# ---------------------------------------------------------------------

def build_labels(df):
    print(f"[2] Building {FORWARD_DAYS}-trading-day forward-return labels "
          f"(HOLD band +-{HOLD_THRESHOLD:.0%}) ...")
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

def time_split(df):
    print("[3] Splitting train/test by date (80/20, no shuffling) ...")
    unique_dates = np.sort(df["trade_date"].unique())
    cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]

    train_df = df[df["trade_date"] < cutoff_date]
    test_df = df[df["trade_date"] >= cutoff_date]
    print(f"    cutoff date: {cutoff_date}  (train < cutoff, test >= cutoff)")
    print(f"    train: {len(train_df):,} rows, {train_df['trade_date'].min()} to {train_df['trade_date'].max()}")
    print(f"    test:  {len(test_df):,} rows, {test_df['trade_date'].min()} to {test_df['trade_date'].max()}")
    return train_df, test_df


# ---------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------

def train_model(train_df):
    print("[4] Class distribution in training set:")
    for cls, count in train_df["signal_type"].value_counts().items():
        print(f"    {cls:<5} {count:>7,}")

    model = RandomForestClassifier(**HYPERPARAMS)
    model.fit(train_df[FEATURE_COLS], train_df["signal_type"])
    print("    model trained")
    return model


# ---------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------

def evaluate(model, test_df):
    print("[5] Evaluating on test set ...")
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["signal_type"]

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    confidence = proba.max(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"    accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # The model outputs a class, not a return, so there is no native
    # regression target. test_mae/test_rmse are populated per the Models
    # schema by comparing predicted class confidence against the realised
    # forward_return magnitude — a calibration check, not a return forecast.
    diff = confidence - np.abs(test_df["forward_return"].to_numpy())
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    print(f"    confidence-vs-|forward_return| MAE:  {mae:.6f}")
    print(f"    confidence-vs-|forward_return| RMSE: {rmse:.6f}")

    return y_pred, confidence, accuracy, mae, rmse


# ---------------------------------------------------------------------
# 6. Persist model to disk
# ---------------------------------------------------------------------

def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[6] Model saved to {os.path.abspath(MODEL_PATH)}")


# ---------------------------------------------------------------------
# 7. Write Models row
# ---------------------------------------------------------------------

def insert_model_row(conn, train_df, accuracy, mae, rmse):
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
            MODEL_NAME,
            ALGORITHM,
            date.today(),
            train_df["trade_date"].min(),
            train_df["trade_date"].max(),
            json.dumps(HYPERPARAMS),
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
# 8. Write Predictions (test set only)
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
# 9. Backfill outcomes
# ---------------------------------------------------------------------

def backfill(conn):
    print(f"[9] CALL sp_backfill_prediction_outcomes({FORWARD_DAYS}) ...")
    cur = conn.cursor()
    cur.callproc("sp_backfill_prediction_outcomes", (FORWARD_DAYS,))
    rows_backfilled = None
    for result in cur.stored_results():
        row = result.fetchone()
        rows_backfilled = row[0] if row else None
    conn.commit()
    cur.close()
    print(f"    rows_backfilled = {rows_backfilled}")


# ---------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------

def print_summary(conn, model_id):
    print("[10] Summary")
    cur = conn.cursor()
    cur.execute(
        "SELECT SUM(total_predictions), SUM(correct_predictions) "
        "FROM vw_prediction_accuracy WHERE model_name = %s",
        (MODEL_NAME,),
    )
    total_scored, correct = cur.fetchone()
    overall_accuracy = (correct / total_scored) if total_scored else None

    cur.execute("SELECT COUNT(*) FROM Predictions WHERE model_id = %s", (model_id,))
    total_predictions = cur.fetchone()[0]
    cur.close()

    print(f"    total predictions made (this model):        {total_predictions:,}")
    print(f"    total predictions scored (was_correct set):  {total_scored or 0:,}")
    if overall_accuracy is not None:
        print(f"    overall accuracy (vw_prediction_accuracy):   {overall_accuracy:.4f}")
    else:
        print("    overall accuracy (vw_prediction_accuracy):   n/a (nothing scored yet)")


# ---------------------------------------------------------------------

def main():
    conn = get_connection()

    df = load_features(conn)
    df = build_labels(df)
    train_df, test_df = time_split(df)

    model = train_model(train_df)
    y_pred, confidence, accuracy, mae, rmse = evaluate(model, test_df)
    save_model(model)

    model_id = insert_model_row(conn, train_df, accuracy, mae, rmse)
    insert_predictions(conn, model_id, test_df, y_pred, confidence)
    backfill(conn)
    print_summary(conn, model_id)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
