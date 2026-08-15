-- =====================================================================
-- 01_schema.sql
-- Database-Driven AI-Based Quantitative Stock Analysis Platform
-- Phase 2: Database Design (schema, constraints, indexes)
-- Target: MySQL 8.0.20 or later
-- =====================================================================

DROP DATABASE IF EXISTS stockdb;
CREATE DATABASE stockdb
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_0900_ai_ci;
USE stockdb;


-- ---------------------------------------------------------------------
-- 1. Companies  (parent entity of the market-data side)
-- ---------------------------------------------------------------------
CREATE TABLE Companies (
    company_id      INT AUTO_INCREMENT PRIMARY KEY,
    ticker          VARCHAR(20)  NOT NULL,
    company_name    VARCHAR(120) NOT NULL,
    sector          VARCHAR(60),
    exchange        VARCHAR(20)  NOT NULL DEFAULT 'NSE',
    currency        CHAR(3)      NOT NULL DEFAULT 'INR',
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    CONSTRAINT uq_companies_ticker UNIQUE (ticker)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 2. DailyPrices  (raw OHLCV, one row per company per trading day)
--    Composite candidate key (company_id, trade_date) enforces idempotent
--    re-ingestion and doubles as the primary access-path index.
-- ---------------------------------------------------------------------
CREATE TABLE DailyPrices (
    price_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    company_id      INT           NOT NULL,
    trade_date      DATE          NOT NULL,
    open_price      DECIMAL(12,4) NOT NULL,
    high_price      DECIMAL(12,4) NOT NULL,
    low_price       DECIMAL(12,4) NOT NULL,
    close_price     DECIMAL(12,4) NOT NULL,
    adj_close       DECIMAL(12,4) NOT NULL,
    volume          BIGINT        NOT NULL DEFAULT 0,
    CONSTRAINT uq_prices_company_date UNIQUE (company_id, trade_date),
    CONSTRAINT fk_prices_company FOREIGN KEY (company_id)
        REFERENCES Companies(company_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT chk_prices_range CHECK (high_price >= low_price),
    CONSTRAINT chk_prices_volume CHECK (volume >= 0)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 3. TechnicalIndicators  (derived, 1:1 with DailyPrices on the same key)
--    NULLs are expected for the first ~50 rows per ticker: SMA50 has no
--    value until 50 trading days of history exist. Do not zero-fill.
-- ---------------------------------------------------------------------
CREATE TABLE TechnicalIndicators (
    indicator_id    BIGINT AUTO_INCREMENT PRIMARY KEY,
    company_id      INT           NOT NULL,
    trade_date      DATE          NOT NULL,
    rsi_14          DECIMAL(8,4),
    macd            DECIMAL(12,6),
    macd_signal     DECIMAL(12,6),
    macd_hist       DECIMAL(12,6),
    bb_upper        DECIMAL(12,4),
    bb_mid          DECIMAL(12,4),
    bb_lower        DECIMAL(12,4),
    sma_20          DECIMAL(12,4),
    sma_50          DECIMAL(12,4),
    ema_12          DECIMAL(12,4),
    ema_26          DECIMAL(12,4),
    CONSTRAINT uq_ind_company_date UNIQUE (company_id, trade_date),
    CONSTRAINT fk_ind_company FOREIGN KEY (company_id)
        REFERENCES Companies(company_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 4. Models  (split out of Predictions to remove a transitive dependency:
--    algorithm and hyperparams depend on the model, not on the prediction)
-- ---------------------------------------------------------------------
CREATE TABLE Models (
    model_id        INT AUTO_INCREMENT PRIMARY KEY,
    model_name      VARCHAR(80)  NOT NULL,
    algorithm       VARCHAR(40)  NOT NULL,
    trained_on_date DATE         NOT NULL,
    train_start     DATE,
    train_end       DATE,
    hyperparams     JSON,
    test_accuracy   DECIMAL(6,4),
    test_mae        DECIMAL(10,6),
    test_rmse       DECIMAL(10,6),
    CONSTRAINT uq_models_name UNIQUE (model_name, trained_on_date)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 5. Predictions  (closed-loop: actual_return is backfilled after the fact)
-- ---------------------------------------------------------------------
CREATE TABLE Predictions (
    prediction_id   BIGINT AUTO_INCREMENT PRIMARY KEY,
    company_id      INT          NOT NULL,
    model_id        INT          NOT NULL,
    trade_date      DATE         NOT NULL,
    signal_type     ENUM('BUY','SELL','HOLD') NOT NULL,
    confidence      DECIMAL(6,4),
    predicted_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actual_return   DECIMAL(10,6) DEFAULT NULL,
    was_correct     BOOLEAN       DEFAULT NULL,
    CONSTRAINT uq_pred_company_model_date UNIQUE (company_id, model_id, trade_date),
    CONSTRAINT fk_pred_company FOREIGN KEY (company_id)
        REFERENCES Companies(company_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_pred_model FOREIGN KEY (model_id)
        REFERENCES Models(model_id)
        ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT chk_pred_conf CHECK (confidence IS NULL OR (confidence BETWEEN 0 AND 1))
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 6. Users
-- ---------------------------------------------------------------------
CREATE TABLE Users (
    user_id         INT AUTO_INCREMENT PRIMARY KEY,
    username        VARCHAR(50)  NOT NULL,
    email           VARCHAR(120) NOT NULL,
    password_hash   CHAR(60)     NOT NULL,
    created_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_users_username UNIQUE (username),
    CONSTRAINT uq_users_email    UNIQUE (email)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 7. Portfolios
-- ---------------------------------------------------------------------
CREATE TABLE Portfolios (
    portfolio_id    INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT           NOT NULL,
    portfolio_name  VARCHAR(80)   NOT NULL,
    cash_balance    DECIMAL(16,2) NOT NULL DEFAULT 0.00,
    created_on      DATE          NOT NULL DEFAULT (CURRENT_DATE),
    CONSTRAINT uq_portfolio_per_user UNIQUE (user_id, portfolio_name),
    CONSTRAINT fk_portfolio_user FOREIGN KEY (user_id)
        REFERENCES Users(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT chk_portfolio_cash CHECK (cash_balance >= 0)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 8. PortfolioHoldings  (associative entity resolving Portfolios M:N Companies)
-- ---------------------------------------------------------------------
CREATE TABLE PortfolioHoldings (
    holding_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id    INT           NOT NULL,
    company_id      INT           NOT NULL,
    quantity        INT           NOT NULL,
    avg_buy_price   DECIMAL(12,4) NOT NULL,
    CONSTRAINT uq_holding UNIQUE (portfolio_id, company_id),
    CONSTRAINT fk_holding_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES Portfolios(portfolio_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_holding_company FOREIGN KEY (company_id)
        REFERENCES Companies(company_id)
        ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT chk_holding_qty CHECK (quantity > 0)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 9. Watchlists  (associative entity resolving Users M:N Companies)
-- ---------------------------------------------------------------------
CREATE TABLE Watchlists (
    watchlist_id    BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT       NOT NULL,
    company_id      INT       NOT NULL,
    added_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_watchlist UNIQUE (user_id, company_id),
    CONSTRAINT fk_watch_user FOREIGN KEY (user_id)
        REFERENCES Users(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_watch_company FOREIGN KEY (company_id)
        REFERENCES Companies(company_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 10. LatestPrices  (materialized cache, maintained by trigger)
--     Exists so the trigger never has to SELECT from DailyPrices, which
--     MySQL restricts inside a trigger defined on that same table.
-- ---------------------------------------------------------------------
CREATE TABLE LatestPrices (
    company_id      INT           PRIMARY KEY,
    trade_date      DATE          NOT NULL,
    close_price     DECIMAL(12,4) NOT NULL,
    CONSTRAINT fk_latest_company FOREIGN KEY (company_id)
        REFERENCES Companies(company_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 11. PortfolioSummary  (derived 1:1 with Portfolios, maintained by trigger)
--     Deliberate denormalization: separated from Portfolios so trigger
--     writes never lock the base entity row.
-- ---------------------------------------------------------------------
CREATE TABLE PortfolioSummary (
    portfolio_id    INT PRIMARY KEY,
    total_value     DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    cost_basis      DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    unrealized_pnl  DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    last_updated    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_summary_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES Portfolios(portfolio_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- 12. IngestionErrors  (rejected rows from the loader; integrity evidence)
-- ---------------------------------------------------------------------
CREATE TABLE IngestionErrors (
    error_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker          VARCHAR(20),
    trade_date      DATE,
    reason          VARCHAR(255) NOT NULL,
    raw_payload     JSON,
    logged_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------
-- Secondary indexes
-- NOTE: (company_id, trade_date) already exists on DailyPrices and
-- TechnicalIndicators via their UNIQUE constraints. Do not add it again.
-- ---------------------------------------------------------------------
CREATE INDEX idx_prices_date        ON DailyPrices (trade_date);
CREATE INDEX idx_pred_date_signal   ON Predictions (trade_date, signal_type);
CREATE INDEX idx_holdings_company   ON PortfolioHoldings (company_id);
CREATE INDEX idx_companies_sector   ON Companies (sector);

SHOW TABLES;
