-- =====================================================================
-- 02_seed_data.sql
-- Reference data and demo users. Run AFTER 01_schema.sql,
-- BEFORE the Python ingestion script.
--
-- NOTE: Companies insert removed from this run — the 10 base tickers
-- already loaded successfully before this file failed partway through
-- on the first attempt, and 02b_expand_tickers.sql has since brought
-- the table to 140 rows. Re-inserting them here would just hit
-- duplicate-key errors on ticker.
-- =====================================================================

USE stockdb;

-- ---------------------------------------------------------------------
-- Users. password_hash is a bcrypt placeholder — never store plaintext.
-- Trimmed to exactly 60 chars to fit CHAR(60) (original was 61 — that
-- off-by-one is what caused the first run to fail here).
-- ---------------------------------------------------------------------
INSERT INTO Users (username, email, password_hash) VALUES       
('mohit',    'mohit@example.com',    '$2b$12$abcdefghijklmnopqrstuvABCDEFGHIJKLMNOPQRSTUVWXYZ01234'),
('gurirath', 'gurirath@example.com', '$2b$12$abcdefghijklmnopqrstuvABCDEFGHIJKLMNOPQRSTUVWXYZ01234');


-- ---------------------------------------------------------------------
-- Portfolios and holdings for the stored-procedure demo
-- ---------------------------------------------------------------------
INSERT INTO Portfolios (user_id, portfolio_name, cash_balance) VALUES
(1, 'Core Long Term', 150000.00),
(1, 'Momentum Swing',  50000.00),
(2, 'Balanced',       100000.00);

INSERT INTO PortfolioHoldings (portfolio_id, company_id, quantity, avg_buy_price) VALUES
(1, 1,  40, 2450.5000),
(1, 2,  25, 3610.0000),
(1, 4,  60, 1520.7500),
(2, 3, 100, 1480.0000),
(2, 5, 200,  415.2500),
(3, 1,  15, 2600.0000),
(3, 6,  80,  790.0000);

INSERT INTO Watchlists (user_id, company_id) VALUES
(1, 7), (1, 8), (2, 9), (2, 10);


-- ---------------------------------------------------------------------
-- Model registry entry. The ML side inserts its own rows after training;
-- this one exists so Predictions has a valid FK target during testing.
-- ---------------------------------------------------------------------
INSERT INTO Models (model_name, algorithm, trained_on_date, train_start, train_end,
                    hyperparams, test_accuracy)
VALUES ('rf_baseline_v1', 'RandomForest', CURRENT_DATE, '2019-01-01', '2024-12-31',
        JSON_OBJECT('n_estimators', 300, 'max_depth', 12, 'random_state', 42),
        0.5620);

SELECT 'Seed complete' AS status,
       (SELECT COUNT(*) FROM Companies) AS companies,
       (SELECT COUNT(*) FROM Users)     AS users,
       (SELECT COUNT(*) FROM Portfolios) AS portfolios;