"""
ingest_and_indicators.py
Phases 1 and 3: data acquisition into MySQL, then indicator computation
written back into MySQL.

    pip install yfinance pandas mysql-connector-python

Credentials come from environment variables so nothing secret is committed.
Set DB_PASSWORD before running:

    PowerShell:  $env:DB_PASSWORD="your_password"
    CMD:         set DB_PASSWORD=your_password
    bash:        export DB_PASSWORD=your_password

Run:
    python ingest_and_indicators.py --years 10          # full historical load
    python ingest_and_indicators.py --years 10 --indicators-only
"""

import argparse
import json
import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
import mysql.connector

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "stockdb"),
)


# ---------------------------------------------------------------------
# Phase 1 — acquisition and validation
# ---------------------------------------------------------------------

def fetch_companies(cur):
    cur.execute("SELECT company_id, ticker FROM Companies WHERE is_active = TRUE")
    return cur.fetchall()


def log_error(cur, ticker, trade_date, reason, payload=None):
    cur.execute(
        "INSERT INTO IngestionErrors (ticker, trade_date, reason, raw_payload) "
        "VALUES (%s, %s, %s, %s)",
        (ticker, trade_date, reason, json.dumps(payload) if payload else None),
    )


def validate(row, ticker, trade_date, cur):
    """Return True if the row is fit to insert, otherwise log and return False."""
    if any(pd.isna(row[c]) for c in ("Open", "High", "Low", "Close")):
        log_error(cur, ticker, trade_date, "null OHLC value")
        return False
    if row["High"] < row["Low"]:
        log_error(cur, ticker, trade_date, "high < low")
        return False
    if row["Volume"] < 0:
        log_error(cur, ticker, trade_date, "negative volume")
        return False
    return True


UPSERT_PRICE = """
INSERT INTO DailyPrices
    (company_id, trade_date, open_price, high_price, low_price,
     close_price, adj_close, volume)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    open_price  = VALUES(open_price),
    high_price  = VALUES(high_price),
    low_price   = VALUES(low_price),
    close_price = VALUES(close_price),
    adj_close   = VALUES(adj_close),
    volume      = VALUES(volume)
"""


def ingest_prices(conn, years):
    """Idempotent load. Re-running updates in place instead of duplicating,
    because of the UNIQUE (company_id, trade_date) constraint."""
    cur = conn.cursor()
    start = date.today() - timedelta(days=365 * years)

    for company_id, ticker in fetch_companies(cur):
        symbol = f"{ticker}.NS"
        df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
        if df.empty:
            log_error(cur, ticker, None, "yfinance returned no rows")
            conn.commit()
            print(f"  {ticker:<10} no data")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        batch = []
        for idx, row in df.iterrows():
            trade_date = idx.date()
            if not validate(row, ticker, trade_date, cur):
                continue
            batch.append((
                company_id, trade_date,
                float(row["Open"]), float(row["High"]), float(row["Low"]),
                float(row["Close"]),
                float(row.get("Adj Close", row["Close"])),
                int(row["Volume"]),
            ))

        cur.executemany(UPSERT_PRICE, batch)
        conn.commit()
        print(f"  {ticker:<10} {len(batch):>6} rows")

    cur.close()


# ---------------------------------------------------------------------
# Phase 3 — indicator engine (read from DB, compute, write back to DB)
# ---------------------------------------------------------------------

def compute_indicators(df):
    close = df["close_price"].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["ema_12"] = ema12
    df["ema_26"] = ema26
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["sma_20"] = sma20
    df["sma_50"] = close.rolling(50).mean()
    df["bb_mid"] = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # Warm-up rows stay NULL. Do not fillna(0) — a zeroed RSI is a fake
    # oversold signal the model will happily learn from.
    return df


IND_COLS = ["rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_mid",
            "bb_lower", "sma_20", "sma_50", "ema_12", "ema_26"]

INSERT_IND = f"""
INSERT INTO TechnicalIndicators (company_id, trade_date, {', '.join(IND_COLS)})
VALUES ({', '.join(['%s'] * (2 + len(IND_COLS)))})
"""


def build_indicators(conn):
    """Delete-then-insert per ticker. Simpler and faster than upserting
    eleven nullable columns, and safe because the table is fully derived."""
    cur = conn.cursor()
    read_cur = conn.cursor()
    for company_id, ticker in fetch_companies(cur):
        read_cur.execute(
            "SELECT trade_date, close_price FROM DailyPrices "
            "WHERE company_id = %s ORDER BY trade_date",
            (company_id,),
        )
        df = pd.DataFrame(read_cur.fetchall(), columns=["trade_date", "close_price"])

        if len(df) < 50:
            print(f"  {ticker:<10} skipped ({len(df)} rows, need 50+)")
            continue

        df = compute_indicators(df)
        cur.execute("DELETE FROM TechnicalIndicators WHERE company_id = %s", (company_id,))

        rows = [
            tuple([company_id, r.trade_date] +
                  [None if pd.isna(getattr(r, c)) else float(getattr(r, c)) for c in IND_COLS])
            for r in df.itertuples()
        ]
        cur.executemany(INSERT_IND, rows)
        conn.commit()
        print(f"  {ticker:<10} {len(rows):>6} indicator rows")

    cur.close()
    read_cur.close()


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--indicators-only", action="store_true")
    args = ap.parse_args()

    if not DB["password"]:
        raise SystemExit(
            "DB_PASSWORD is not set.\n"
            '  PowerShell:  $env:DB_PASSWORD="your_password"\n'
            "  CMD:         set DB_PASSWORD=your_password\n"
            "  bash:        export DB_PASSWORD=your_password"
        )

    conn = mysql.connector.connect(**DB)
    if not args.indicators_only:
        print("Phase 1 — ingesting prices")
        ingest_prices(conn, args.years)
    print("Phase 3 — computing indicators")
    build_indicators(conn)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()