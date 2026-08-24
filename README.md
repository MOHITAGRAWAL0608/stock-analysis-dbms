# Database-Driven Stock Analysis Platform — DBMS Layer

A normalized MySQL platform that stores historical NSE price data, computes technical indicators back into the database, and serves feature views to a machine-learning prediction layer. Built as a DBMS course project.

**Current dataset:** 138 NSE tickers, ~334,000 daily price rows, 10 years of history.

## Contents

| File | Purpose |
|---|---|
| `01_schema.sql` | 12 tables, constraints, indexes |
| `02_seed_data.sql` | Base companies, users, portfolios, model registry row |
| `02b_expand_tickers.sql` | Extends to ~140 NSE tickers |
| `ingest_and_indicators.py` | Fetches prices via yfinance, computes RSI/MACD/Bollinger/SMA/EMA |
| `03_views_procedures_triggers.sql` | 4 views, 3 stored procedures, 2 triggers |
| `04_index_demo.sql` | Query optimization demonstration |
| `ml/train_model.py` | Trains the signal classifier, writes `Models` and `Predictions` |
| `dashboard/app.py` | Streamlit dashboard over the views |

---

## Setup

```bash
pip install -r requirements.txt
```

Credentials come from environment variables — nothing secret is stored in the code.

**Primary method:** copy `.env.example` to `.env` in the project root and fill in your real password. `python-dotenv` loads it automatically, so `DB_PASSWORD` (and any other variables in the file) are set before scripts run — no manual `$env:`/`export` needed.

```bash
cp .env.example .env
# then edit .env and set DB_PASSWORD (and DB_HOST/DB_USER/DB_NAME if needed)
```

**Fallback:** if you'd rather not use a `.env` file, set the environment variables manually before running any script.

```powershell
# PowerShell
$env:DB_PASSWORD="your_password"
```

```bash
# bash
export DB_PASSWORD=your_password
```

Optional overrides: `DB_HOST` (default `localhost`), `DB_USER` (default `root`), `DB_NAME` (default `stockdb`).

Requires **MySQL 8.0.20 or later** — `CHECK` constraints and window functions are both used.

---

## Execution order

Order matters. The trigger created in step 5 must not exist during the bulk load in step 4.

| # | Command | Notes |
|---|---|---|
| 1 | `Get-Content 01_schema.sql \| mysql -u root -p` | Drops and recreates `stockdb`. **Destructive.** |
| 2 | `Get-Content 02_seed_data.sql \| mysql -u root -p stockdb` | Companies, users, portfolios, one model row |
| 3 | `Get-Content 02b_expand_tickers.sql \| mysql -u root -p stockdb` | Expands to ~140 tickers |
| 4 | `python ingest_and_indicators.py --years 10` | 15–25 minutes. Set `DB_PASSWORD` first. |
| 5 | `Get-Content 03_views_procedures_triggers.sql \| mysql -u root -p stockdb` | **After** the load, not before |
| 6 | `04_index_demo.sql` | Run interactively in MySQL Workbench or the client |
| 7 | `python ml/train_model.py` | Trains, writes `Predictions`, backfills outcomes |
| 8 | `streamlit run dashboard/app.py` | Dashboard on http://localhost:8501 |

PowerShell reserves `<` for future use, hence `Get-Content | mysql` rather than `mysql < file`. In bash or CMD, `mysql -u root -p stockdb < file.sql` works normally.

Re-running the loader is safe. `ON DUPLICATE KEY UPDATE` against `UNIQUE (company_id, trade_date)` updates in place rather than duplicating.

**If the trigger already exists before a reload**, drop it first — it fires per inserted row, turning a 20-minute load into hours:

```sql
DROP TRIGGER IF EXISTS trg_price_after_insert;
```

Then re-run step 5 afterwards. It rebuilds the trigger and calls `sp_rebuild_latest_prices()` to reseed the cache from the new data.

---

## Verification

```sql
SELECT COUNT(*) FROM DailyPrices;          -- ~334,000
SELECT COUNT(*) FROM TechnicalIndicators;  -- same
SELECT COUNT(*) FROM vw_ml_features;       -- same
SELECT COUNT(*) FROM IngestionErrors;      -- one row per delisted symbol
SELECT TRIGGER_NAME FROM information_schema.TRIGGERS
WHERE TRIGGER_SCHEMA = 'stockdb';          -- 2 rows
```

### Demonstrating the trigger

It only fires on new inserts, so use a fresh row:

```sql
SELECT * FROM PortfolioSummary WHERE portfolio_id = 1;

INSERT INTO DailyPrices
    (company_id, trade_date, open_price, high_price, low_price,
     close_price, adj_close, volume)
VALUES (1, '2026-08-20', 1300, 1330, 1295, 1325, 1325, 5000000);

SELECT * FROM PortfolioSummary WHERE portfolio_id = 1;
```

`total_value` and `last_updated` both change. That before/after pair is the entire trigger demo.

---

## Design decisions

**`LatestPrices` cache table.** MySQL raises error 1442 if a trigger queries the table it is defined on. The trigger therefore maintains a small cache of the most recent close per company and reads portfolio valuations from that, never from `DailyPrices` directly.

**`Models` split from `Predictions`.** Algorithm and hyperparameters depend on the model, not on the individual prediction. Storing them inline would be a transitive dependency and a 3NF violation.

**`TechnicalIndicators` separate from `DailyPrices`** despite being 1:1. The two are populated by different processes at different times, and indicator columns are NULL for the first ~50 trading days per ticker. Merging them produces a wide table half-full of NULLs.

**`PortfolioSummary` as a separate 1:1 table.** Deliberate denormalization — a materialized summary maintained by the trigger, kept out of `Portfolios` so trigger writes never lock the base entity row.

**Both `close_price` and `adj_close` stored.** Portfolio valuation needs the raw traded price; return calculations spanning a split or bonus issue need the back-adjusted series. RELIANCE and HDFCBANK both had bonus issues within this dataset's range.

**Validation rule corrected during development.** The original loader rejected weekend dates as invalid. This wrongly discarded NSE Muhurat trading sessions (Diwali) and the 1 February 2025 Union Budget Saturday session — 30 legitimate rows, surfaced by the `IngestionErrors` log.

---

## Troubleshooting

**`Error 1419: You do not have the SUPER privilege`** — trigger creation blocked. As root:
```sql
SET GLOBAL log_bin_trust_function_creators = 1;
```

**`Error 1442: Can't update table ... already used by statement`** — a trigger is trying to read its own subject table. Read from `LatestPrices` instead.

**`VALUES()` deprecation warnings** — harmless on MySQL 8.0.x. On 8.4+ the function is removed; switch the loader's upsert to row-alias syntax.

**`CHECK` constraints silently ignored** — you are on MariaDB or MySQL 5.7. Confirm with `SELECT VERSION();`.

**Index demo shows no measurable difference** — the table is too small. Below ~250,000 rows MySQL scans in single-digit milliseconds either way. Load more tickers.

**Some tickers return no data** — expected. Symbols change after demergers and delistings. Failures are logged to `IngestionErrors`; mark them inactive:
```sql
UPDATE Companies SET is_active = FALSE WHERE ticker IN ('TATAMOTORS','PEL');
```

---

## Interface with the ML layer

The database layer and the prediction layer share exactly two touch points.

**Read from `vw_ml_features`** — one row per company per trading day:

`company_id`, `ticker`, `trade_date`, `close_price`, `volume`, `rsi_14`, `macd`, `macd_hist`, `sma_20`, `sma_50`, `ema_12`, `bb_position`, `sma_ratio`

The first ~50 rows per ticker have NULL indicators (warm-up period) and should be dropped before training. The view carries no target variable — the label is constructed from forward returns.

**Write to `Models`, then `Predictions`:**

```sql
INSERT INTO Models (model_name, algorithm, trained_on_date,
                    train_start, train_end, hyperparams, test_accuracy)
VALUES ('rf_v1', 'RandomForest', CURDATE(), '2016-01-01', '2024-12-31',
        JSON_OBJECT('n_estimators', 300, 'max_depth', 12), 0.5612);

INSERT INTO Predictions (company_id, model_id, trade_date, signal_type, confidence)
VALUES (%s, %s, %s, %s, %s);
```

`signal_type` is an ENUM accepting only `'BUY'`, `'SELL'`, or `'HOLD'`. Leave `actual_return` and `was_correct` NULL — they are backfilled by:

```sql
CALL sp_backfill_prediction_outcomes(5);
```

No other tables are written to by the ML layer. If a feature is needed that the view does not expose, the view is altered rather than the base tables being queried directly.

---

## Report sections these files support

| Rubric item | Evidence |
|---|---|
| ER diagram and schema justification | `01_schema.sql` comments, Chen and crow's-foot diagrams |
| Normalization to 3NF | `Models` split; transitive dependency walkthrough |
| Integrity constraints | FK actions, `CHECK` constraints, `trg_holding_before_insert` |
| Stored procedures | `sp_portfolio_value`, `sp_backfill_prediction_outcomes`, `sp_rebuild_latest_prices` |
| Triggers | `trg_price_after_insert`, `trg_holding_before_insert` |
| Views | `vw_latest_prices`, `vw_ml_features`, `vw_prediction_accuracy`, `vw_portfolio_detail` |
| Query optimization | `04_index_demo.sql` before/after execution plans |
