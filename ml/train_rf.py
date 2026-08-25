import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_connection  # noqa: E402  (ml/db.py)
from common import (  # noqa: E402  (ml/common.py)
    FEATURE_COLS,
    load_features,
    build_labels,
    time_split,
    save_model,
    insert_model_row,
    insert_predictions,
    backfill,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MODEL_NAME = "rf_v1"
ALGORITHM = "RandomForest"
HYPERPARAMS = dict(
    n_estimators=300, max_depth=12, class_weight="balanced", random_state=42
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "rf_model.joblib")


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
    baseline = y_test.value_counts(normalize=True).max()
    print(f"    accuracy: {accuracy:.4f}  (majority-class baseline {baseline:.4f})")
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
    save_model(model, MODEL_PATH)

    model_id = insert_model_row(conn, MODEL_NAME, ALGORITHM, train_df, HYPERPARAMS, accuracy, mae, rmse)
    insert_predictions(conn, model_id, test_df, y_pred, confidence)
    backfill(conn)
    print_summary(conn, model_id)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
