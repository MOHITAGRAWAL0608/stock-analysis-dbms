"""
dashboard/app.py
Streamlit dashboard for the stock analysis platform. Reads exclusively from
the views/tables documented in README.md ("Interface with the ML layer" and
the schema section) — vw_ml_features, vw_prediction_accuracy, Models,
Predictions, Companies, DailyPrices.

Credentials come from environment variables, same convention as
ingest_and_indicators.py / ml/train_model.py. Set DB_PASSWORD before running:

    PowerShell:  $env:DB_PASSWORD="your_password"
    CMD:         set DB_PASSWORD=your_password
    bash:        export DB_PASSWORD=your_password

Run:
    streamlit run dashboard/app.py
"""

import os
import re
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import DB, get_connection  # noqa: E402  (dashboard/db.py)

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

SIGNAL_HORIZON_DAYS = 20   # must match FORWARD_DAYS in ml/train_model.py


# ---------------------------------------------------------------------
# Connection + query helpers
# ---------------------------------------------------------------------

if not DB["password"]:
    st.error(
        "DB_PASSWORD is not set.\n\n"
        "PowerShell:  `$env:DB_PASSWORD=\"your_password\"`  \n"
        "CMD:         `set DB_PASSWORD=your_password`  \n"
        "bash:        `export DB_PASSWORD=your_password`\n\n"
        "Then restart Streamlit."
    )
    st.stop()


@st.cache_resource
def get_conn():
    return get_connection()


@st.cache_data(ttl=60)
def run_query(query, params=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(query, params or ())
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=cols)


try:
    companies_df = run_query(
        "SELECT company_id, ticker FROM Companies WHERE is_active = TRUE ORDER BY ticker"
    )
except Exception as exc:
    st.error(f"Could not connect to MySQL: {exc}")
    st.stop()

st.title("Stock Analysis Dashboard")

tickers = companies_df["ticker"].tolist()
selected = st.sidebar.selectbox("Ticker", ["All"] + tickers)


# ---------------------------------------------------------------------
# 1. Header — latest trained model
# ---------------------------------------------------------------------

model_df = run_query(
    "SELECT model_name, algorithm, trained_on_date, test_accuracy, test_mae, test_rmse "
    "FROM Models ORDER BY trained_on_date DESC, model_id DESC LIMIT 1"
)

if model_df.empty:
    latest_model = None
    st.info("No trained model found yet — run `python ml/train_model.py` first.")
else:
    m = model_df.iloc[0]
    latest_model = m["model_name"]
    st.caption(f"{m['model_name']} ({m['algorithm']}) — trained {m['trained_on_date']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{float(m['test_accuracy']):.2%}" if m["test_accuracy"] is not None else "n/a")
    c2.metric("Test MAE", f"{float(m['test_mae']):.4f}" if m["test_mae"] is not None else "n/a")
    c3.metric("Test RMSE", f"{float(m['test_rmse']):.4f}" if m["test_rmse"] is not None else "n/a")

st.divider()


# ---------------------------------------------------------------------
# 2. Prediction accuracy
# ---------------------------------------------------------------------

st.subheader("Prediction Accuracy")

if selected == "All":
    acc_df = run_query(
        """
        SELECT month,
               SUM(total_predictions)   AS total_predictions,
               SUM(correct_predictions) AS correct_predictions,
               ROUND(SUM(correct_predictions) / SUM(total_predictions), 4) AS accuracy
        FROM vw_prediction_accuracy
        WHERE model_name = %s
        GROUP BY month
        ORDER BY month
        """,
        (latest_model,),
    )
else:
    acc_df = run_query(
        "SELECT month, total_predictions, correct_predictions, accuracy "
        "FROM vw_prediction_accuracy WHERE model_name = %s AND ticker = %s "
        "ORDER BY month",
        (latest_model, selected),
    )

if acc_df.empty:
    st.info("No scored predictions yet for this selection — run `python ml/train_model.py` first.")
else:
    total_predictions = int(acc_df["total_predictions"].sum())
    total_correct = int(acc_df["correct_predictions"].sum())
    overall_accuracy = total_correct / total_predictions if total_predictions else None

    c1, c2 = st.columns(2)
    c1.metric("Overall accuracy", f"{overall_accuracy:.1%}" if overall_accuracy is not None else "n/a")
    c2.metric("Total predictions", f"{total_predictions:,}")

    fig = px.line(acc_df, x="month", y="accuracy", markers=True)
    fig.update_yaxes(tickformat=".0%", title="Accuracy")
    fig.update_xaxes(title="Month")
    st.plotly_chart(fig, width="stretch")

st.divider()


# ---------------------------------------------------------------------
# 3. Price + technical indicators (requires a specific ticker)
# ---------------------------------------------------------------------

st.subheader("Price & Technical Indicators")

if selected == "All":
    st.info("Select a specific ticker in the sidebar to view price and indicator charts.")
    feat_df = pd.DataFrame()
else:
    feat_df = run_query(
        "SELECT trade_date, close_price, sma_20, sma_50, rsi_14 "
        "FROM vw_ml_features WHERE ticker = %s ORDER BY trade_date",
        (selected,),
    )

    if feat_df.empty:
        st.info(f"No feature data available for {selected} yet.")
    else:
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=feat_df["trade_date"], y=feat_df["close_price"], name="Close"))
        price_fig.add_trace(go.Scatter(x=feat_df["trade_date"], y=feat_df["sma_20"], name="SMA 20"))
        price_fig.add_trace(go.Scatter(x=feat_df["trade_date"], y=feat_df["sma_50"], name="SMA 50"))
        price_fig.update_layout(title=f"{selected} close price", height=400, margin=dict(t=40))
        st.plotly_chart(price_fig, width="stretch")

        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=feat_df["trade_date"], y=feat_df["rsi_14"], name="RSI 14"))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(title="RSI (14)", height=250, yaxis_range=[0, 100], margin=dict(t=40))
        st.plotly_chart(rsi_fig, width="stretch")

st.divider()


# ---------------------------------------------------------------------
# 4. Portfolio simulation — buy & hold vs. model signals (headline result)
#    Each actual_return spans SIGNAL_HORIZON_DAYS, so only every Nth
#    prediction is compounded. Overlapping windows would count the same
#    price move up to SIGNAL_HORIZON_DAYS times over.
# ---------------------------------------------------------------------

st.subheader("Portfolio Simulation: Buy & Hold vs. Model Signals")

if selected == "All":
    st.info("Select a specific ticker in the sidebar to view the portfolio simulation.")
elif feat_df.empty:
    st.info("No price data to simulate for this ticker.")
else:
    first_close = float(feat_df.iloc[0]["close_price"])
    last_close = float(feat_df.iloc[-1]["close_price"])
    buy_hold_return = (last_close - first_close) / first_close

    pred_df = run_query(
        """
        SELECT p.trade_date, p.signal_type, p.actual_return
        FROM Predictions p
        JOIN Companies c ON c.company_id = p.company_id
        JOIN Models    m ON m.model_id   = p.model_id
        WHERE c.ticker = %s AND m.model_name = %s AND p.was_correct IS NOT NULL
        ORDER BY p.trade_date
        """,
        (selected, latest_model),
    )

    if pred_df.empty:
        model_return = None
    else:
        def leg_return(row):
            if row["signal_type"] == "BUY":
                return float(row["actual_return"])
            if row["signal_type"] == "SELL":
                return -float(row["actual_return"])
            return 0.0

        legs = pred_df.iloc[::SIGNAL_HORIZON_DAYS].apply(leg_return, axis=1)
        model_return = float((1 + legs).prod() - 1)

    c1, c2 = st.columns(2)
    c1.metric("Buy & Hold return", f"{buy_hold_return:.2%}")
    if model_return is not None:
        c2.metric(
            "Model-signal return",
            f"{model_return:.2%}",
            delta=f"{(model_return - buy_hold_return):.2%} vs. buy & hold",
        )
        st.caption(f"{len(legs)} non-overlapping {SIGNAL_HORIZON_DAYS}-day positions")
    else:
        c2.metric("Model-signal return", "n/a")
        st.caption("No scored predictions yet for this ticker — run `python ml/train_model.py` first.")

st.divider()


# ---------------------------------------------------------------------
# 5. Query performance — indexed plan on DailyPrices(company_id, trade_date)
# ---------------------------------------------------------------------

st.subheader("Query Performance")

if selected == "All":
    st.info("Select a specific ticker in the sidebar to view its query plan.")
else:
    company_id = int(companies_df.loc[companies_df["ticker"] == selected, "company_id"].iloc[0])

    plan_df = run_query(
        "EXPLAIN ANALYZE SELECT * FROM DailyPrices "
        "WHERE company_id = %s AND trade_date > '2024-01-01'",
        (company_id,),
    )
    plan_text = "\n".join(str(v) for v in plan_df.iloc[:, 0])

    total_rows = int(run_query("SELECT COUNT(*) FROM DailyPrices").iloc[0, 0])

    measured = re.search(r"actual time=[\d.]+\.\.([\d.]+) rows=(\d+)", plan_text)
    if measured:
        elapsed_ms, scanned = float(measured.group(1)), int(measured.group(2))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows examined", f"{scanned:,}")
        c2.metric("Rows in table", f"{total_rows:,}")
        c3.metric("Execution time", f"{elapsed_ms:.2f} ms")
        st.caption(f"The index narrowed the scan to {scanned / total_rows:.3%} of the table.")
