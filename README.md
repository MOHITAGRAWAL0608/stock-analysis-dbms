# Execution order

Run these in sequence. Order matters — the trigger in step 4 must not exist during the bulk load in step 3.

| # | File | Command | Notes |
|---|---|---|---|
| 1 | `01_schema.sql` | `mysql -u root -p < 01_schema.sql` | Drops and recreates `stockdb`. Destructive. |
| 2 | `02_seed_data.sql` | `mysql -u root -p < 02_seed_data.sql` | Companies, users, portfolios, one model row. |
| 3 | `ingest_and_indicators.py` | `python ingest_and_indicators.py --years 10` | Set `DB["password"]` first. Takes several minutes. |
| 4 | `03_views_procedures_triggers.sql` | `mysql -u root -p < 03_views_procedures_triggers.sql` | **After** the load, not before. |
| 5 | `04_index_demo.sql` | Run interactively in Workbench | Screenshot the plans as you go. |

Re-running the loader later is safe — `ON DUPLICATE KEY UPDATE` against `UNIQUE (company_id, trade_date)` updates in place instead of duplicating.

---

## Verifying the trigger actually works

The trigger only fires on new inserts, so demonstrate it with a fresh row:

```sql
CALL sp_portfolio_value(1);
SELECT * FROM PortfolioSummary WHERE portfolio_id = 1;

INSERT INTO DailyPrices
    (company_id, trade_date, open_price, high_price, low_price,
     close_price, adj_close, volume)
VALUES (1, '2026-08-14', 2900, 2950, 2890, 2940, 2940, 5000000);

SELECT * FROM PortfolioSummary WHERE portfolio_id = 1;
```

`total_value` and `last_updated` should both change. That single before/after pair is the whole trigger demo — you don't need slides for it.

---

## Things that will go wrong

**`Error 1419: You do not have the SUPER privilege`** — trigger creation is blocked. Fix as root:

```sql
SET GLOBAL log_bin_trust_function_creators = 1;
```

**`Error 1442: Can't update table ... already used by statement`** — you tried to make the trigger read `DailyPrices`. That's exactly why `LatestPrices` exists. Read from the cache table, never the subject table.

**`VALUES()` deprecation warnings** — harmless on MySQL 8.0.x. On MySQL 8.4+ the function is removed; switch the loader's upsert to row-alias syntax:

```sql
INSERT INTO DailyPrices (...) VALUES (...) AS new
ON DUPLICATE KEY UPDATE close_price = new.close_price;
```

**`CHECK` constraints silently ignored** — you're on MariaDB or MySQL 5.7, not MySQL 8. Check with `SELECT VERSION();`. Window functions in `vw_latest_prices` also need 8.0+.

**pandas `read_sql` warning about DBAPI2** — cosmetic with mysql-connector. Silence it by switching to SQLAlchemy if it bothers you.

**Index demo shows no difference** — your table is too small. 10 tickers × 10 years ≈ 25,000 rows, which MySQL scans in under 10ms. Extend the `Companies` insert to 100+ NSE tickers and reload.

---

## What to hand your teammate

One sentence, and hold him to it:

> Read features from `vw_ml_features`. Write predictions to `Predictions` with columns `(company_id, model_id, trade_date, signal_type, confidence)`. Register the model in `Models` first and use its `model_id`. Do not write to any other table.

Then you run `CALL sp_backfill_prediction_outcomes(5);` to fill in `actual_return` and `was_correct`, and the dashboard reads `vw_prediction_accuracy`. Neither of you touches the other's tables.

Lock this contract in week one. Schema reconciliation in the final week is how these projects die.

---

## Report sections these files support

| Rubric item | Evidence |
|---|---|
| ER diagram and schema justification | `01_schema.sql` comments + the Chen diagrams |
| Normalization to 3NF | The `Models` split; the `StockData` decomposition walkthrough |
| Integrity constraints | FK actions, `CHECK` constraints, `trg_holding_before_insert` |
| Stored procedures | `sp_portfolio_value`, `sp_backfill_prediction_outcomes` |
| Triggers | `trg_price_after_insert` |
| Views | Four views in section A |
| Query optimization | `04_index_demo.sql` before/after plans |
