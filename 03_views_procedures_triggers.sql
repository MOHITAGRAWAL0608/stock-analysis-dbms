-- =====================================================================
-- 03_views_procedures_triggers.sql
-- Phases 4 and part of 5: SQL views, stored procedures, triggers.
--
-- IMPORTANT: run this AFTER the historical bulk load. The trigger fires
-- per row; creating it before loading 250k rows means 250k portfolio
-- recalculations and a load that takes hours instead of minutes.
-- =====================================================================

USE stockdb;


-- =====================================================================
-- SECTION A — VIEWS
-- =====================================================================

-- ---------------------------------------------------------------------
-- A1. vw_latest_prices — most recent close per company.
--     This is the contract the stored procedure depends on.
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS vw_latest_prices;
CREATE VIEW vw_latest_prices AS
SELECT company_id, trade_date, close_price
FROM (
    SELECT company_id, trade_date, close_price,
           ROW_NUMBER() OVER (PARTITION BY company_id
                              ORDER BY trade_date DESC) AS rn
    FROM DailyPrices
) ranked
WHERE rn = 1;


-- ---------------------------------------------------------------------
-- A2. vw_ml_features — THE interface between the DB side and the ML side.
--     Your teammate reads only from here. Lock this contract early:
--     if the column list changes, tell them before you deploy it.
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS vw_ml_features;
CREATE VIEW vw_ml_features AS
SELECT
    c.company_id,
    c.ticker,
    p.trade_date,
    p.close_price,
    p.volume,
    t.rsi_14,
    t.macd,
    t.macd_hist,
    t.sma_20,
    t.sma_50,
    t.ema_12,
    CASE WHEN t.bb_upper > t.bb_lower
         THEN (p.close_price - t.bb_lower) / (t.bb_upper - t.bb_lower)
    END AS bb_position,
    CASE WHEN t.sma_50 > 0
         THEN (t.sma_20 - t.sma_50) / t.sma_50
    END AS sma_ratio
FROM DailyPrices p
JOIN TechnicalIndicators t
      ON t.company_id = p.company_id
     AND t.trade_date = p.trade_date
JOIN Companies c
      ON c.company_id = p.company_id;


-- ---------------------------------------------------------------------
-- A3. vw_prediction_accuracy — feeds the dashboard accuracy chart.
--     Only counts predictions whose outcome has been backfilled.
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS vw_prediction_accuracy;
CREATE VIEW vw_prediction_accuracy AS
SELECT
    m.model_name,
    c.ticker,
    DATE_FORMAT(p.trade_date, '%Y-%m') AS month,
    COUNT(*)                            AS total_predictions,
    SUM(p.was_correct = TRUE)           AS correct_predictions,
    ROUND(AVG(p.was_correct = TRUE), 4) AS accuracy,
    ROUND(AVG(p.confidence), 4)         AS avg_confidence
FROM Predictions p
JOIN Models    m ON m.model_id   = p.model_id
JOIN Companies c ON c.company_id = p.company_id
WHERE p.was_correct IS NOT NULL
GROUP BY m.model_name, c.ticker, DATE_FORMAT(p.trade_date, '%Y-%m');


-- ---------------------------------------------------------------------
-- A4. vw_portfolio_detail — per-holding breakdown for the dashboard table
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS vw_portfolio_detail;
CREATE VIEW vw_portfolio_detail AS
SELECT
    pf.portfolio_id,
    pf.portfolio_name,
    u.username,
    c.ticker,
    h.quantity,
    h.avg_buy_price,
    lp.close_price                                    AS current_price,
    ROUND(h.quantity * lp.close_price, 2)             AS market_value,
    ROUND(h.quantity * (lp.close_price - h.avg_buy_price), 2) AS unrealized_pnl,
    ROUND(100 * (lp.close_price - h.avg_buy_price) / h.avg_buy_price, 2) AS pct_return
FROM PortfolioHoldings h
JOIN Portfolios        pf ON pf.portfolio_id = h.portfolio_id
JOIN Users             u  ON u.user_id       = pf.user_id
JOIN Companies         c  ON c.company_id    = h.company_id
JOIN vw_latest_prices  lp ON lp.company_id   = h.company_id;


-- =====================================================================
-- SECTION B — STORED PROCEDURES
-- =====================================================================

-- ---------------------------------------------------------------------
-- B1. sp_portfolio_value — the headline procedure for the demo.
-- ---------------------------------------------------------------------
DROP PROCEDURE IF EXISTS sp_portfolio_value;
DELIMITER //
CREATE PROCEDURE sp_portfolio_value(IN p_portfolio_id INT)
BEGIN
    DECLARE v_exists INT DEFAULT 0;

    SELECT COUNT(*) INTO v_exists
    FROM Portfolios WHERE portfolio_id = p_portfolio_id;

    IF v_exists = 0 THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'sp_portfolio_value: portfolio_id does not exist';
    END IF;

    SELECT
        pf.portfolio_id,
        pf.portfolio_name,
        pf.cash_balance,
        COALESCE(SUM(h.quantity * lp.close_price), 0)                        AS market_value,
        COALESCE(SUM(h.quantity * h.avg_buy_price), 0)                       AS cost_basis,
        COALESCE(SUM(h.quantity * (lp.close_price - h.avg_buy_price)), 0)    AS unrealized_pnl,
        pf.cash_balance + COALESCE(SUM(h.quantity * lp.close_price), 0)      AS total_account_value
    FROM Portfolios pf
    LEFT JOIN PortfolioHoldings h  ON h.portfolio_id = pf.portfolio_id
    LEFT JOIN vw_latest_prices  lp ON lp.company_id  = h.company_id
    WHERE pf.portfolio_id = p_portfolio_id
    GROUP BY pf.portfolio_id, pf.portfolio_name, pf.cash_balance;
END //
DELIMITER ;


-- ---------------------------------------------------------------------
-- B2. sp_backfill_prediction_outcomes — closes the prediction loop.
--     Compares each prediction against the realised N-day forward return.
-- ---------------------------------------------------------------------
DROP PROCEDURE IF EXISTS sp_backfill_prediction_outcomes;
DELIMITER //
CREATE PROCEDURE sp_backfill_prediction_outcomes(IN p_horizon_days INT)
BEGIN
    UPDATE Predictions p
    JOIN DailyPrices d0
          ON d0.company_id = p.company_id
         AND d0.trade_date = p.trade_date
    JOIN DailyPrices d1
          ON d1.company_id = p.company_id
         AND d1.trade_date = (
              SELECT MIN(trade_date) FROM DailyPrices
              WHERE company_id = p.company_id
                AND trade_date > DATE_ADD(p.trade_date, INTERVAL p_horizon_days DAY)
         )
    SET p.actual_return = (d1.close_price - d0.close_price) / d0.close_price,
        p.was_correct = CASE
            WHEN p.signal_type = 'BUY'  AND d1.close_price > d0.close_price THEN TRUE
            WHEN p.signal_type = 'SELL' AND d1.close_price < d0.close_price THEN TRUE
            WHEN p.signal_type = 'HOLD' AND ABS((d1.close_price - d0.close_price)
                                                / d0.close_price) < 0.01 THEN TRUE
            ELSE FALSE
        END
    WHERE p.actual_return IS NULL;

    SELECT ROW_COUNT() AS rows_backfilled;
END //
DELIMITER ;


-- ---------------------------------------------------------------------
-- B3. sp_rebuild_latest_prices — seeds the cache table after a bulk load,
--     since the trigger only maintains it going forward.
-- ---------------------------------------------------------------------
DROP PROCEDURE IF EXISTS sp_rebuild_latest_prices;
DELIMITER //
CREATE PROCEDURE sp_rebuild_latest_prices()
BEGIN
    DELETE FROM LatestPrices;
    INSERT INTO LatestPrices (company_id, trade_date, close_price)
    SELECT company_id, trade_date, close_price FROM vw_latest_prices;

    REPLACE INTO PortfolioSummary
        (portfolio_id, total_value, cost_basis, unrealized_pnl, last_updated)
    SELECT h.portfolio_id,
           SUM(h.quantity * lp.close_price),
           SUM(h.quantity * h.avg_buy_price),
           SUM(h.quantity * (lp.close_price - h.avg_buy_price)),
           NOW()
    FROM PortfolioHoldings h
    JOIN LatestPrices lp ON lp.company_id = h.company_id
    GROUP BY h.portfolio_id;

    SELECT COUNT(*) AS cached_companies FROM LatestPrices;
END //
DELIMITER ;


-- =====================================================================
-- SECTION C — TRIGGERS
-- =====================================================================

-- ---------------------------------------------------------------------
-- C1. trg_price_after_insert
--     Two jobs: refresh the LatestPrices cache, then recompute
--     PortfolioSummary for every portfolio holding that company.
--
--     Note it reads LatestPrices, never DailyPrices. MySQL forbids a
--     trigger from querying the table it is defined on, which is the
--     whole reason the cache table exists.
-- ---------------------------------------------------------------------
DROP TRIGGER IF EXISTS trg_price_after_insert;
DELIMITER //
CREATE TRIGGER trg_price_after_insert
AFTER INSERT ON DailyPrices
FOR EACH ROW
BEGIN
    INSERT INTO LatestPrices (company_id, trade_date, close_price)
    VALUES (NEW.company_id, NEW.trade_date, NEW.close_price)
    ON DUPLICATE KEY UPDATE
        trade_date  = IF(NEW.trade_date >= LatestPrices.trade_date,
                         NEW.trade_date,  LatestPrices.trade_date),
        close_price = IF(NEW.trade_date >= LatestPrices.trade_date,
                         NEW.close_price, LatestPrices.close_price);

    REPLACE INTO PortfolioSummary
        (portfolio_id, total_value, cost_basis, unrealized_pnl, last_updated)
    SELECT h.portfolio_id,
           SUM(h.quantity * lp.close_price),
           SUM(h.quantity * h.avg_buy_price),
           SUM(h.quantity * (lp.close_price - h.avg_buy_price)),
           NOW()
    FROM PortfolioHoldings h
    JOIN LatestPrices lp ON lp.company_id = h.company_id
    WHERE h.portfolio_id IN (
        SELECT portfolio_id FROM PortfolioHoldings
        WHERE company_id = NEW.company_id
    )
    GROUP BY h.portfolio_id;
END //
DELIMITER ;


-- ---------------------------------------------------------------------
-- C2. trg_holding_before_insert — integrity trigger.
--     Blocks a holding on a delisted company. CHECK constraints cannot
--     reference another table, so this has to be a trigger.
-- ---------------------------------------------------------------------
DROP TRIGGER IF EXISTS trg_holding_before_insert;
DELIMITER //
CREATE TRIGGER trg_holding_before_insert
BEFORE INSERT ON PortfolioHoldings
FOR EACH ROW
BEGIN
    DECLARE v_active BOOLEAN;
    SELECT is_active INTO v_active
    FROM Companies WHERE company_id = NEW.company_id;

    IF v_active IS NULL OR v_active = FALSE THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'Cannot hold a delisted or unknown company';
    END IF;
END //
DELIMITER ;


-- =====================================================================
-- Post-install: seed the cache, then verify
-- =====================================================================
CALL sp_rebuild_latest_prices();
CALL sp_portfolio_value(1);

SELECT * FROM PortfolioSummary;
SHOW TRIGGERS;
