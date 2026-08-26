"""
ml/train_ensemble.py
Confidence-weighted voting ensemble over the 5 already-trained models'
stored Predictions rows. Does not retrain or reload any model — RF/XGBoost
need raw features, LogReg/SVM need their saved scalers, and LSTM needs
15-day sequences, so combining their *already-scored predictions* avoids
reconciling four different feature-preparation paths.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_connection  # noqa: E402  (ml/db.py)
from common import insert_model_row, insert_predictions, backfill  # noqa: E402  (ml/common.py)

MODEL_NAME = "ensemble_v1"
ALGORITHM = "WeightedVotingEnsemble"
COMPONENT_MODEL_NAMES = ["rf_v1", "logreg_v1", "xgb_v1", "svm_v1", "lstm_v1"]


# ---------------------------------------------------------------------
# 1. Load each component model's latest Predictions
# ---------------------------------------------------------------------

def load_component_predictions(conn):
    print("[1] Resolving latest model_id per component model_name ...")
    cur = conn.cursor()
    placeholders = ", ".join(["%s"] * len(COMPONENT_MODEL_NAMES))
    cur.execute(
        f"""
        SELECT model_name, model_id, test_accuracy
        FROM (
            SELECT model_name, model_id, test_accuracy,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name ORDER BY trained_on_date DESC, model_id DESC
                   ) AS rn
            FROM Models
            WHERE model_name IN ({placeholders})
        ) latest
        WHERE rn = 1
        """,
        COMPONENT_MODEL_NAMES,
    )
    rows = cur.fetchall()
    model_ids = {name: int(mid) for name, mid, _ in rows}
    individual_test_accuracy = {name: float(acc) if acc is not None else None for name, _, acc in rows}
    for name in COMPONENT_MODEL_NAMES:
        print(f"    {name:<10} model_id={model_ids[name]}  test_accuracy={individual_test_accuracy[name]}")
    missing = set(COMPONENT_MODEL_NAMES) - set(model_ids)
    if missing:
        raise SystemExit(f"Missing trained model(s), run their training scripts first: {sorted(missing)}")

    print("[2] Reading Predictions for those 5 model_ids ...")
    id_placeholders = ", ".join(["%s"] * len(model_ids))
    cur.execute(
        f"""
        SELECT company_id, trade_date, model_id, signal_type, confidence
        FROM Predictions
        WHERE model_id IN ({id_placeholders})
        """,
        list(model_ids.values()),
    )
    cols = ["company_id", "trade_date", "model_id", "signal_type", "confidence"]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    cur.close()
    df["confidence"] = df["confidence"].astype(float)
    print(f"    {len(df):,} prediction rows read across all 5 models")
    return df, model_ids, individual_test_accuracy


# ---------------------------------------------------------------------
# 2/3. Intersect to pairs where all 5 models predicted, then vote
# ---------------------------------------------------------------------

def build_ensemble(df, model_ids):
    print("[3] Intersecting to (company_id, trade_date) pairs where all 5 models have a prediction ...")
    n_models = len(model_ids)
    present_counts = df.groupby(["company_id", "trade_date"])["model_id"].nunique()
    total_pairs = len(present_counts)
    complete_pairs = present_counts[present_counts == n_models].index
    dropped_pairs = total_pairs - len(complete_pairs)
    print(f"    {len(complete_pairs):,} pairs have all {n_models} models present")
    print(f"    {dropped_pairs:,} pairs dropped (missing at least one model's prediction)")

    complete_df = df.set_index(["company_id", "trade_date"]).loc[complete_pairs].reset_index()

    print("[4] Computing confidence-weighted vote per pair ...")
    sums = (
        complete_df.groupby(["company_id", "trade_date", "signal_type"])["confidence"]
        .sum()
        .unstack(fill_value=0.0)
    )
    sums.columns.name = None
    for col in ("BUY", "SELL"):
        if col not in sums.columns:
            sums[col] = 0.0

    ensemble_signal = np.where(sums["BUY"] >= sums["SELL"], "BUY", "SELL")
    winning_sum = sums[["BUY", "SELL"]].max(axis=1)
    total_sum = sums["BUY"] + sums["SELL"]
    ensemble_confidence = (winning_sum / total_sum).to_numpy()

    ensemble_df = sums.reset_index()[["company_id", "trade_date"]].copy()
    ensemble_df["signal_type"] = ensemble_signal
    ensemble_df["confidence"] = ensemble_confidence
    return ensemble_df, len(complete_pairs), dropped_pairs


# ---------------------------------------------------------------------

def main():
    conn = get_connection()

    df, model_ids, individual_test_accuracy = load_component_predictions(conn)
    ensemble_df, complete_count, dropped_count = build_ensemble(df, model_ids)

    hyperparams = dict(
        voting_method="confidence_weighted_sum",
        combined_models=model_ids,
    )

    print("[5] Inserting placeholder Models row ...")
    model_id = insert_model_row(conn, MODEL_NAME, ALGORITHM, ensemble_df, hyperparams, 0.0, 0.0, 0.0)

    print("[6] Inserting ensemble Predictions ...")
    insert_predictions(
        conn, model_id, ensemble_df,
        ensemble_df["signal_type"].to_numpy(), ensemble_df["confidence"].to_numpy(),
    )

    backfill(conn)

    print("[7] Computing final accuracy/MAE/RMSE from backfilled outcomes ...")
    cur = conn.cursor()
    cur.execute(
        "SELECT signal_type, confidence, actual_return, was_correct "
        "FROM Predictions WHERE model_id = %s AND was_correct IS NOT NULL",
        (model_id,),
    )
    scored_cols = ["signal_type", "confidence", "actual_return", "was_correct"]
    scored_df = pd.DataFrame(cur.fetchall(), columns=scored_cols)
    cur.close()

    scored_df["confidence"] = scored_df["confidence"].astype(float)
    scored_df["was_correct"] = scored_df["was_correct"].astype(int)

    accuracy = float(scored_df["was_correct"].mean())

    if scored_df["actual_return"].notna().all() and len(scored_df):
        actual_return = scored_df["actual_return"].astype(float).to_numpy()
        diff = scored_df["confidence"].to_numpy() - np.abs(actual_return)
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
    else:
        mae = None
        rmse = None

    opposite = np.where(scored_df["signal_type"] == "BUY", "SELL", "BUY")
    true_label = np.where(scored_df["was_correct"] == 1, scored_df["signal_type"], opposite)
    balanced_acc = balanced_accuracy_score(true_label, scored_df["signal_type"])

    print(f"    scored predictions: {len(scored_df):,}")
    print(f"    accuracy: {accuracy:.4f}")
    print(f"    balanced accuracy: {balanced_acc:.4f}")
    print(classification_report(true_label, scored_df["signal_type"], digits=4, zero_division=0))
    if mae is not None:
        print(f"    confidence-vs-|actual_return| MAE:  {mae:.6f}")
        print(f"    confidence-vs-|actual_return| RMSE: {rmse:.6f}")
    else:
        print("    actual_return not cleanly available for all scored rows — MAE/RMSE left NULL")

    print("[8] Updating Models row with final metrics ...")
    insert_model_row(conn, MODEL_NAME, ALGORITHM, ensemble_df, hyperparams, accuracy, mae, rmse)

    print("[9] Comparison vs individual models")
    print(f"    {'model_name':<12} {'test_accuracy':>14}")
    beats_all = True
    for name in COMPONENT_MODEL_NAMES:
        acc = individual_test_accuracy[name]
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        marker = ""
        if acc is not None:
            if accuracy > acc:
                marker = "  (ensemble beats this)"
            else:
                marker = "  (ensemble does NOT beat this)"
                beats_all = False
        print(f"    {name:<12} {acc_s:>14}{marker}")
    print(f"    {'ensemble_v1':<12} {accuracy:>14.4f}")
    print(f"    balanced accuracy: {balanced_acc:.4f}")
    if beats_all:
        print("    RESULT: ensemble_v1 beats every individual model's test_accuracy.")
    else:
        print("    RESULT: ensemble_v1 does NOT beat every individual model's test_accuracy.")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
