"""
ml/train_all.py
Runs all five training scripts in sequence, then prints a final comparison
table across their latest Models rows (test_accuracy DESC).

The Models table can carry stale rows from earlier experimentation (a
retired baseline, an old hyperparameter attempt) that share a model_name
with a current script but sit under an earlier trained_on_date — those are
excluded in favor of each model_name's most recent row.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_connection  # noqa: E402  (ml/db.py)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "train_rf.py",
    "train_logreg.py",
    "train_xgboost.py",
    "train_svm.py",
    "train_lstm.py",
]

MODEL_NAMES = ["rf_v1", "logreg_v1", "xgb_v1", "svm_v1", "lstm_v1"]


def run_script(script_name):
    print(f"\n{'=' * 70}\nRunning {script_name}\n{'=' * 70}")
    result = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, script_name)], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise SystemExit(f"{script_name} failed with exit code {result.returncode}")


def print_comparison():
    conn = get_connection()
    cur = conn.cursor()
    placeholders = ", ".join(["%s"] * len(MODEL_NAMES))
    cur.execute(
        f"""
        SELECT model_name, algorithm, test_accuracy, test_mae, test_rmse
        FROM (
            SELECT model_name, algorithm, test_accuracy, test_mae, test_rmse,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name ORDER BY trained_on_date DESC, model_id DESC
                   ) AS rn
            FROM Models
            WHERE model_name IN ({placeholders}) AND model_name != 'rf_baseline_v1'
        ) latest
        WHERE rn = 1
        ORDER BY test_accuracy DESC
        """,
        MODEL_NAMES,
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"\n{'=' * 70}\nFinal comparison (latest row per model, by test_accuracy DESC)\n{'=' * 70}")
    header = f"{'model_name':<12} {'algorithm':<20} {'test_accuracy':>14} {'test_mae':>10} {'test_rmse':>10}"
    print(header)
    print("-" * len(header))
    for model_name, algorithm, accuracy, mae, rmse in rows:
        acc_s = f"{float(accuracy):.4f}" if accuracy is not None else "n/a"
        mae_s = f"{float(mae):.6f}" if mae is not None else "n/a"
        rmse_s = f"{float(rmse):.6f}" if rmse is not None else "n/a"
        print(f"{model_name:<12} {algorithm:<20} {acc_s:>14} {mae_s:>10} {rmse_s:>10}")


def main():
    for script in SCRIPTS:
        run_script(script)
    print_comparison()


if __name__ == "__main__":
    main()
