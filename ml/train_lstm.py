"""
ml/train_lstm.py
Sequence-based LSTM signal classifier.

This does NOT reuse common.py's time_split — an LSTM needs overlapping
15-day windows per company, not single feature rows, so the split has to
operate on one row per *sequence* (keyed by the label's trade_date) rather
than one row per trading day. Feature loading and the 20-day forward-return
labeling logic are still the shared common.py functions; only the windowing,
scaling, and split-application steps are LSTM-specific.
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_connection  # noqa: E402  (ml/db.py)
from common import (  # noqa: E402  (ml/common.py)
    FEATURE_COLS,
    load_features,
    build_labels,
    time_split,
    insert_model_row,
    insert_predictions,
    backfill,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MODEL_NAME = "lstm_v1"
ALGORITHM = "LSTM"
SEQUENCE_LENGTH = 15
LSTM_UNITS = 64
DROPOUT = 0.3
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 256

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm_model.keras")

SEED = 42


# ---------------------------------------------------------------------
# 3. Reshape into overlapping per-company sequences
# ---------------------------------------------------------------------

def build_sequences(df):
    print(f"[3] Building {SEQUENCE_LENGTH}-day sequences per company (label = the day right after each window) ...")
    n_features = len(FEATURE_COLS)
    min_rows = SEQUENCE_LENGTH + 1

    X_parts = []
    meta_parts = []
    dropped_companies = 0

    for company_id, g in df.sort_values(["company_id", "trade_date"]).groupby("company_id", sort=True):
        if len(g) < min_rows:
            dropped_companies += 1
            continue

        arr = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        # windows[i] covers rows [i : i+SEQUENCE_LENGTH); the label comes
        # from the row right after the window, i.e. row (i + SEQUENCE_LENGTH)
        windows_all = np.lib.stride_tricks.sliding_window_view(
            arr, window_shape=SEQUENCE_LENGTH, axis=0
        ).transpose(0, 2, 1)
        windows = windows_all[:-1]  # drop the final window (no "day after" it)

        target_idx = np.arange(SEQUENCE_LENGTH, len(g))
        targets = g.iloc[target_idx]

        X_parts.append(windows)
        meta_parts.append(pd.DataFrame({
            "company_id": targets["company_id"].to_numpy(),
            "trade_date": targets["trade_date"].to_numpy(),
            "signal_type": targets["signal_type"].to_numpy(),
            "forward_return": targets["forward_return"].to_numpy(),
        }))

    X = np.concatenate(X_parts, axis=0)
    seq_meta = pd.concat(meta_parts, ignore_index=True)

    print(f"    {dropped_companies} companies dropped (fewer than {min_rows} valid rows)")
    print(f"    {len(seq_meta):,} sequences built, shape {X.shape}")
    assert X.shape == (len(seq_meta), SEQUENCE_LENGTH, n_features)
    return X, seq_meta


# ---------------------------------------------------------------------
# 5. Scale features (fit on flattened train windows only)
# ---------------------------------------------------------------------

def scale_sequences(X_train, X_test):
    print("[5] Scaling features (StandardScaler fit on flattened train windows) ...")
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------

def train_model(X_train, train_meta):
    print("[6] Class distribution in training set:")
    for cls, count in train_meta["signal_type"].value_counts().items():
        print(f"    {cls:<5} {count:>7,}")

    # y encodes BUY=1, SELL=0 (see below), so in Keras's class_weight dict
    # key 0 -> SELL and key 1 -> BUY. Standard inverse-frequency balancing.
    total = len(train_meta)
    buy_count = int((train_meta["signal_type"] == "BUY").sum())
    sell_count = int((train_meta["signal_type"] == "SELL").sum())
    weight_sell = total / (2 * sell_count)
    weight_buy = total / (2 * buy_count)
    class_weight = {0: weight_sell, 1: weight_buy}
    print(f"    class_weight = {{0 (SELL): {weight_sell:.4f}, 1 (BUY): {weight_buy:.4f}}}")

    # Sort chronologically by label date so Keras's validation_split (which
    # takes the tail of the arrays as given, unshuffled) carves out the
    # chronologically LAST 15% as validation rather than a random slice.
    order = np.argsort(train_meta["trade_date"].to_numpy(), kind="stable")
    X_train_sorted = X_train[order]
    y_train_sorted = (train_meta["signal_type"].to_numpy()[order] == "BUY").astype(np.float32)

    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, len(FEATURE_COLS))),
        tf.keras.layers.LSTM(LSTM_UNITS),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
    )

    history = model.fit(
        X_train_sorted, y_train_sorted,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=2,
    )

    epochs_run = len(history.history["loss"])
    print(f"    model trained ({epochs_run} epochs run)")

    hyperparams = dict(
        sequence_length=SEQUENCE_LENGTH,
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT,
        epochs=epochs_run,
        class_weight_applied=True,
        class_weight={"SELL_0": round(weight_sell, 4), "BUY_1": round(weight_buy, 4)},
    )
    return model, hyperparams


# ---------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------

def evaluate(model, X_test, test_meta):
    print("[7] Evaluating on test set ...")
    y_test = test_meta["signal_type"].to_numpy()

    prob_buy = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred = np.where(prob_buy >= 0.5, "BUY", "SELL")
    confidence = np.where(prob_buy >= 0.5, prob_buy, 1.0 - prob_buy)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    baseline = pd.Series(y_test).value_counts(normalize=True).max()
    print(f"    accuracy: {accuracy:.4f}  (majority-class baseline {baseline:.4f})")
    print(f"    balanced accuracy: {balanced_acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Same calibration check as the other models: predicted-class confidence
    # vs. the realised |forward_return| magnitude.
    diff = confidence - np.abs(test_meta["forward_return"].to_numpy())
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    print(f"    confidence-vs-|forward_return| MAE:  {mae:.6f}")
    print(f"    confidence-vs-|forward_return| RMSE: {rmse:.6f}")

    return y_pred, confidence, accuracy, mae, rmse


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(conn, model_id):
    print("[Summary]")
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

    X, seq_meta = build_sequences(df)

    print("[4] Splitting sequences into train/test by the label's trade_date (80/20, no shuffling) ...")
    train_meta, test_meta = time_split(seq_meta)
    X_train = X[train_meta.index.to_numpy()]
    X_test = X[test_meta.index.to_numpy()]

    X_train_scaled, X_test_scaled, scaler = scale_sequences(X_train, X_test)

    model, hyperparams = train_model(X_train_scaled, train_meta)
    y_pred, confidence, accuracy, mae, rmse = evaluate(model, X_test_scaled, test_meta)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[8] Model saved to {os.path.abspath(MODEL_PATH)}")

    model_id = insert_model_row(conn, MODEL_NAME, ALGORITHM, train_meta, hyperparams, accuracy, mae, rmse)
    insert_predictions(conn, model_id, test_meta, y_pred, confidence)
    backfill(conn)
    print_summary(conn, model_id)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
