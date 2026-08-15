-- =====================================================================
-- 04_index_demo.sql
-- Phase 6: the live query-optimization demonstration.
--
-- WHY A CLONE TABLE:
-- DailyPrices already carries UNIQUE (company_id, trade_date), and in
-- InnoDB a UNIQUE constraint IS a B-tree index. There is no "before"
-- state to demonstrate on that table. This script builds an unindexed
-- copy so the before/after comparison is real rather than staged.
--
-- SCALE WARNING: with only 10 tickers you have ~25k rows and MySQL will
-- scan them in single-digit milliseconds — the demo shows nothing.
-- Load 100+ tickers (or 10+ years) to clear ~250k rows first.
-- =====================================================================

USE stockdb;

-- Confirm you have enough data before continuing
SELECT COUNT(*) AS row_count,
       IF(COUNT(*) >= 200000, 'OK for demo', 'TOO SMALL — load more tickers') AS verdict
FROM DailyPrices;


-- =====================================================================
-- STEP 1 — Build the unindexed clone
-- =====================================================================
DROP TABLE IF EXISTS DailyPrices_NoIndex;

CREATE TABLE DailyPrices_NoIndex (
    price_id        BIGINT,
    company_id      INT,
    trade_date      DATE,
    open_price      DECIMAL(12,4),
    high_price      DECIMAL(12,4),
    low_price       DECIMAL(12,4),
    close_price     DECIMAL(12,4),
    adj_close       DECIMAL(12,4),
    volume          BIGINT
) ENGINE=InnoDB;
-- Deliberately: no PRIMARY KEY, no UNIQUE, no secondary index.

INSERT INTO DailyPrices_NoIndex
SELECT price_id, company_id, trade_date, open_price, high_price,
       low_price, close_price, adj_close, volume
FROM DailyPrices;

ANALYZE TABLE DailyPrices_NoIndex;


-- =====================================================================
-- STEP 2 — BEFORE: the same query against the unindexed table
-- Expect: type = ALL, a full table scan, rows examined = the whole table.
-- =====================================================================
SET profiling = 1;

EXPLAIN
SELECT trade_date, close_price, volume
FROM DailyPrices_NoIndex
WHERE company_id = 3
  AND trade_date BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY trade_date;

-- Classic EXPLAIN too — easier to screenshot for the report
EXPLAIN
SELECT trade_date, close_price, volume
FROM DailyPrices_NoIndex
WHERE company_id = 3
  AND trade_date BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY trade_date;


-- =====================================================================
-- STEP 3 — Add the composite index
-- Column order matters: company_id first because it is the equality
-- predicate, trade_date second because it is the range predicate.
-- Reverse that order and the index cannot serve the range efficiently.
-- =====================================================================
ALTER TABLE DailyPrices_NoIndex
    ADD INDEX idx_demo_company_date (company_id, trade_date);

ANALYZE TABLE DailyPrices_NoIndex;


-- =====================================================================
-- STEP 4 — AFTER: identical query, indexed table
-- Expect: type = range, key = idx_demo_company_date, rows examined
-- drops from hundreds of thousands to a few hundred.
-- The ORDER BY should also disappear from the plan — the index already
-- returns rows in trade_date order, so no filesort is needed.
-- =====================================================================
EXPLAIN
SELECT trade_date, close_price, volume
FROM DailyPrices_NoIndex
WHERE company_id = 3
  AND trade_date BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY trade_date;

EXPLAIN
SELECT trade_date, close_price, volume
FROM DailyPrices_NoIndex
WHERE company_id = 3
  AND trade_date BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY trade_date;

SHOW PROFILES;
SET profiling = 0;


-- =====================================================================
-- STEP 5 — Second demo: a JOIN that benefits even more
-- Without indexes this is a nested-loop scan; with them it is two
-- index range scans. The gap is far more dramatic than the single-table
-- case, so use this one if the first is unconvincing on your data size.
-- =====================================================================
DROP TABLE IF EXISTS TechnicalIndicators_NoIndex;
CREATE TABLE TechnicalIndicators_NoIndex AS
SELECT company_id, trade_date, rsi_14, macd, sma_20, sma_50
FROM TechnicalIndicators;

EXPLAIN
SELECT p.trade_date, p.close_price, t.rsi_14, t.macd
FROM DailyPrices_NoIndex p
JOIN TechnicalIndicators_NoIndex t
      ON t.company_id = p.company_id
     AND t.trade_date = p.trade_date
WHERE p.company_id = 3
  AND t.rsi_14 < 30;

ALTER TABLE TechnicalIndicators_NoIndex
    ADD INDEX idx_demo_ind (company_id, trade_date);

ANALYZE TABLE TechnicalIndicators_NoIndex;

EXPLAIN
SELECT p.trade_date, p.close_price, t.rsi_14, t.macd
FROM DailyPrices_NoIndex p
JOIN TechnicalIndicators_NoIndex t
      ON t.company_id = p.company_id
     AND t.trade_date = p.trade_date
WHERE p.company_id = 3
  AND t.rsi_14 < 30;


-- =====================================================================
-- STEP 6 — The counter-example, worth showing if you want the extra mark
-- Indexes are not free. This proves you understand the trade-off rather
-- than treating "add index" as a universal fix.
-- =====================================================================

-- 6a. Wrong column order: trade_date first makes the index useless
--     for a company_id equality lookup across all dates.
ALTER TABLE DailyPrices_NoIndex DROP INDEX idx_demo_company_date;
ALTER TABLE DailyPrices_NoIndex ADD INDEX idx_demo_wrong_order (trade_date, company_id);


-- Compare against the correct order:
ALTER TABLE DailyPrices_NoIndex DROP INDEX idx_demo_wrong_order;
ALTER TABLE DailyPrices_NoIndex ADD INDEX idx_demo_company_date (company_id, trade_date);
EXPLAIN
SELECT COUNT(*) FROM DailyPrices_NoIndex WHERE company_id = 3;

-- 6b. Function on an indexed column defeats the index entirely.
EXPLAIN
SELECT * FROM DailyPrices_NoIndex WHERE YEAR(trade_date) = 2023;   -- full scan
EXPLAIN
SELECT * FROM DailyPrices_NoIndex
WHERE trade_date BETWEEN '2023-01-01' AND '2023-12-31';            -- range scan


-- =====================================================================
-- Cleanup (run only after you have captured your screenshots)
-- =====================================================================
-- DROP TABLE DailyPrices_NoIndex;
-- DROP TABLE TechnicalIndicators_NoIndex;
