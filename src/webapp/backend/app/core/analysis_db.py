from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional, Sequence

from sqlalchemy import (
    BIGINT,
    BOOLEAN,
    DATE,
    DOUBLE_PRECISION,
    INTEGER,
    JSON,
    TEXT,
    TIMESTAMP,
    Column,
    Engine,
    Index,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert

from app.settings import get_analysis_database_url


SCHEMAS = ("pm_raw", "pm_mart", "pm_research")
metadata = MetaData()


pm_raw_weekly_markets = Table(
    "weekly_markets",
    metadata,
    Column("run_id", String(140), primary_key=True),
    Column("market_id", String(64), primary_key=True),
    Column("event_id", String(64), nullable=False),
    Column("event_slug", Text, nullable=True),
    Column("event_title", Text, nullable=True),
    Column("event_end_date", DATE, nullable=True),
    Column("condition_id", Text, nullable=True),
    Column("market_slug", Text, nullable=True),
    Column("market_question", Text, nullable=True),
    Column("ticker", String(32), nullable=False),
    Column("stock_name", Text, nullable=True),
    Column("ticker_source", String(32), nullable=True),
    Column("threshold", Numeric(18, 6), nullable=True),
    Column("week_monday", DATE, nullable=True),
    Column("week_friday", DATE, nullable=True),
    Column("week_sunday", DATE, nullable=True),
    Column("expiry_date_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("resolution_time_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("yes_token_id", Text, nullable=True),
    Column("no_token_id", Text, nullable=True),
    Column("enable_order_book", BOOLEAN, nullable=True),
    Column("active", BOOLEAN, nullable=True),
    Column("closed", BOOLEAN, nullable=True),
    Column("gamma_yes_price", DOUBLE_PRECISION, nullable=True),
    Column("market_subtype", String(64), nullable=False, server_default=text("'weekly_finish_above'")),
    Column("source_file", Text, nullable=True),
    Column("source_run_created_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("ingested_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    Column("schema_version", String(64), nullable=True),
    schema="pm_raw",
)
Index("ix_pm_raw_weekly_markets_ticker_week", pm_raw_weekly_markets.c.ticker, pm_raw_weekly_markets.c.week_friday)


pm_raw_weekly_events = Table(
    "weekly_events",
    metadata,
    Column("run_id", String(140), primary_key=True),
    Column("event_id", String(64), primary_key=True),
    Column("event_slug", Text, nullable=True),
    Column("event_title", Text, nullable=True),
    Column("event_end_date", DATE, nullable=True),
    Column("ticker", String(32), nullable=True),
    Column("stock_name", Text, nullable=True),
    Column("source_file", Text, nullable=True),
    Column("source_run_created_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("ingested_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_raw",
)


pm_raw_price_history = Table(
    "price_history",
    metadata,
    Column("run_id", String(140), primary_key=True),
    Column("timestamp_utc", TIMESTAMP(timezone=True), primary_key=True),
    Column("market_id", String(64), primary_key=True),
    Column("token_id", Text, primary_key=True),
    Column("token_role", String(16), primary_key=True),
    Column("ticker", String(32), nullable=True),
    Column("threshold", Numeric(18, 6), nullable=True),
    Column("price", DOUBLE_PRECISION, nullable=True),
    Column("price_raw", DOUBLE_PRECISION, nullable=True),
    Column("fidelity_min", INTEGER, nullable=True),
    Column("trade_date_ny", DATE, nullable=True),
    Column("source_file", Text, nullable=True),
    Column("source_run_created_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("ingested_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    Column("schema_version", String(64), nullable=True),
    schema="pm_raw",
)
Index("ix_pm_raw_price_history_market_time", pm_raw_price_history.c.market_id, pm_raw_price_history.c.timestamp_utc)
Index("ix_pm_raw_price_history_trade_date", pm_raw_price_history.c.trade_date_ny)


pm_raw_fact_trade = Table(
    "fact_trade",
    metadata,
    Column("trade_id", Text, primary_key=True),
    Column("market_id", String(64), nullable=False),
    Column("outcome_token_id", Text, nullable=True),
    Column("timestamp_utc", TIMESTAMP(timezone=True), nullable=False),
    Column("trade_date_ny", DATE, nullable=True),
    Column("block_number", BIGINT, nullable=True),
    Column("price", DOUBLE_PRECISION, nullable=True),
    Column("size", DOUBLE_PRECISION, nullable=True),
    Column("side", String(16), nullable=True),
    Column("tx_hash", Text, nullable=True),
    Column("ticker", String(32), nullable=True),
    Column("threshold", Numeric(18, 6), nullable=True),
    Column("source_type", String(32), nullable=False, server_default=text("'subgraph'")),
    Column("source_run_id", String(140), nullable=True),
    Column("ingested_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    Column("schema_version", String(64), nullable=True),
    schema="pm_raw",
)
Index("ix_pm_raw_fact_trade_market_time", pm_raw_fact_trade.c.market_id, pm_raw_fact_trade.c.timestamp_utc)
Index("ix_pm_raw_fact_trade_trade_date", pm_raw_fact_trade.c.trade_date_ny)


pm_raw_markets_prn_hourly = Table(
    "markets_prn_hourly",
    metadata,
    Column("run_id", String(140), primary_key=True),
    Column("timestamp_utc", TIMESTAMP(timezone=True), primary_key=True),
    Column("market_id", String(64), primary_key=True),
    Column("week_monday", DATE, nullable=True),
    Column("week_friday", DATE, nullable=True),
    Column("week_sunday", DATE, nullable=True),
    Column("ticker", String(32), nullable=True),
    Column("threshold", Numeric(18, 6), nullable=True),
    Column("expiry_date", DATE, nullable=True),
    Column("event_id", String(64), nullable=True),
    Column("event_end_date", DATE, nullable=True),
    Column("prn_asof_time", TIMESTAMP(timezone=True), nullable=True),
    Column("snapshot_date", DATE, nullable=True),
    Column("pRN", DOUBLE_PRECISION, nullable=True),
    Column("qRN", DOUBLE_PRECISION, nullable=True),
    Column("pRN_raw", DOUBLE_PRECISION, nullable=True),
    Column("qRN_raw", DOUBLE_PRECISION, nullable=True),
    Column("rv20", DOUBLE_PRECISION, nullable=True),
    Column("log_m", DOUBLE_PRECISION, nullable=True),
    Column("abs_log_m", DOUBLE_PRECISION, nullable=True),
    Column("log_m_fwd", DOUBLE_PRECISION, nullable=True),
    Column("abs_log_m_fwd", DOUBLE_PRECISION, nullable=True),
    Column("T_days", DOUBLE_PRECISION, nullable=True),
    Column("S_asof_close", DOUBLE_PRECISION, nullable=True),
    Column("forward_price", DOUBLE_PRECISION, nullable=True),
    Column("dividend_yield", DOUBLE_PRECISION, nullable=True),
    Column("spot", DOUBLE_PRECISION, nullable=True),
    Column("rn_method", String(64), nullable=True),
    Column("spot_source", String(64), nullable=True),
    Column("spot_asof_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("rn_asof_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("polymarket_buy", DOUBLE_PRECISION, nullable=True),
    Column("polymarket_mid", DOUBLE_PRECISION, nullable=True),
    Column("polymarket_ask", DOUBLE_PRECISION, nullable=True),
    Column("polymarket_bid", DOUBLE_PRECISION, nullable=True),
    Column("trade_date_ny", DATE, nullable=True),
    Column("source_run_created_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("ingested_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    Column("schema_version", String(64), nullable=True),
    schema="pm_raw",
)
Index("ix_pm_raw_markets_prn_hourly_market_time", pm_raw_markets_prn_hourly.c.market_id, pm_raw_markets_prn_hourly.c.timestamp_utc)


pm_raw_decision_features_daily = Table(
    "decision_features_daily",
    metadata,
    Column("run_id", String(140), primary_key=True),
    Column("timestamp_utc", TIMESTAMP(timezone=True), primary_key=True),
    Column("market_id", String(64), primary_key=True),
    Column("trade_date_ny", DATE, nullable=True),
    Column("ticker", String(32), nullable=True),
    Column("threshold", Numeric(18, 6), nullable=True),
    Column("condition_id", Text, nullable=True),
    Column("expiry_date_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("resolution_time_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("open", DOUBLE_PRECISION, nullable=True),
    Column("high", DOUBLE_PRECISION, nullable=True),
    Column("low", DOUBLE_PRECISION, nullable=True),
    Column("close", DOUBLE_PRECISION, nullable=True),
    Column("volume", DOUBLE_PRECISION, nullable=True),
    Column("trade_count", DOUBLE_PRECISION, nullable=True),
    Column("pm_last", DOUBLE_PRECISION, nullable=True),
    Column("pm_mid", DOUBLE_PRECISION, nullable=True),
    Column("pm_bid", DOUBLE_PRECISION, nullable=True),
    Column("pm_ask", DOUBLE_PRECISION, nullable=True),
    Column("pm_spread", DOUBLE_PRECISION, nullable=True),
    Column("pm_liquidity_proxy", DOUBLE_PRECISION, nullable=True),
    Column("pm_momentum_1h", DOUBLE_PRECISION, nullable=True),
    Column("pm_volatility", DOUBLE_PRECISION, nullable=True),
    Column("pm_momentum_5m", DOUBLE_PRECISION, nullable=True),
    Column("pm_momentum_1d", DOUBLE_PRECISION, nullable=True),
    Column("pm_time_to_resolution", DOUBLE_PRECISION, nullable=True),
    Column("snapshot_date", DATE, nullable=True),
    Column("prn_asof_time", TIMESTAMP(timezone=True), nullable=True),
    Column("pRN", DOUBLE_PRECISION, nullable=True),
    Column("qRN", DOUBLE_PRECISION, nullable=True),
    Column("pRN_raw", DOUBLE_PRECISION, nullable=True),
    Column("qRN_raw", DOUBLE_PRECISION, nullable=True),
    Column("rv20", DOUBLE_PRECISION, nullable=True),
    Column("log_m", DOUBLE_PRECISION, nullable=True),
    Column("abs_log_m", DOUBLE_PRECISION, nullable=True),
    Column("log_m_fwd", DOUBLE_PRECISION, nullable=True),
    Column("abs_log_m_fwd", DOUBLE_PRECISION, nullable=True),
    Column("T_days", DOUBLE_PRECISION, nullable=True),
    Column("S_asof_close", DOUBLE_PRECISION, nullable=True),
    Column("forward_price", DOUBLE_PRECISION, nullable=True),
    Column("dividend_yield", DOUBLE_PRECISION, nullable=True),
    Column("label", DOUBLE_PRECISION, nullable=True),
    Column("leak_check_passed", BOOLEAN, nullable=True),
    Column("source_run_created_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("ingested_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    Column("schema_version", String(64), nullable=True),
    schema="pm_raw",
)
Index("ix_pm_raw_decision_features_market_date", pm_raw_decision_features_daily.c.market_id, pm_raw_decision_features_daily.c.trade_date_ny)


pm_raw_run_manifest = Table(
    "run_manifest",
    metadata,
    Column("run_id", String(140), primary_key=True),
    Column("run_dir", Text, nullable=True),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("finished_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("status", String(32), nullable=True),
    Column("markets", INTEGER, nullable=True),
    Column("price_rows", INTEGER, nullable=True),
    Column("tickers_json", JSONB, nullable=True),
    Column("pipeline_args_json", JSONB, nullable=True),
    Column("artifacts_json", JSONB, nullable=True),
    Column("backfill_trade_ok", BOOLEAN, nullable=True),
    Column("backfill_trade_error", Text, nullable=True),
    Column("backfill_trade_start_date", DATE, nullable=True),
    Column("backfill_trade_end_date", DATE, nullable=True),
    Column("backfill_trade_rows", INTEGER, nullable=True),
    Column("last_refreshed_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_raw",
)


pm_mart_dim_stock = Table(
    "dim_stock",
    metadata,
    Column("ticker", String(32), primary_key=True),
    Column("stock_name", Text, nullable=True),
    Column("market_subtype", String(64), nullable=False, server_default=text("'weekly_finish_above'")),
    Column("first_listed_date", DATE, nullable=True),
    Column("last_listed_date", DATE, nullable=True),
    Column("market_count", INTEGER, nullable=True),
    Column("active_trading_days", INTEGER, nullable=True),
    Column("latest_refresh_id", String(64), nullable=True),
    Column("updated_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_mart",
)


pm_mart_fact_market_day = Table(
    "fact_market_day",
    metadata,
    Column("trade_date_ny", DATE, primary_key=True),
    Column("market_id", String(64), primary_key=True),
    Column("ticker", String(32), nullable=False),
    Column("stock_name", Text, nullable=True),
    Column("condition_id", Text, nullable=True),
    Column("event_id", String(64), nullable=True),
    Column("yes_token_id", Text, nullable=True),
    Column("no_token_id", Text, nullable=True),
    Column("market_subtype", String(64), nullable=False),
    Column("threshold", Numeric(18, 6), nullable=True),
    Column("week_monday", DATE, nullable=True),
    Column("week_friday", DATE, nullable=True),
    Column("resolution_time_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("days_to_resolution", DOUBLE_PRECISION, nullable=True),
    Column("dtr_bucket", String(32), nullable=True),
    Column("listed_flag", BOOLEAN, nullable=False),
    Column("observed_flag", BOOLEAN, nullable=False),
    Column("traded_flag", BOOLEAN, nullable=False),
    Column("trade_count", DOUBLE_PRECISION, nullable=True),
    Column("trade_count_clean", DOUBLE_PRECISION, nullable=True),
    Column("contract_volume", DOUBLE_PRECISION, nullable=True),
    Column("contract_volume_clean", DOUBLE_PRECISION, nullable=True),
    Column("notional_volume", DOUBLE_PRECISION, nullable=True),
    Column("notional_volume_clean", DOUBLE_PRECISION, nullable=True),
    Column("buy_volume", DOUBLE_PRECISION, nullable=True),
    Column("sell_volume", DOUBLE_PRECISION, nullable=True),
    Column("avg_trade_size", DOUBLE_PRECISION, nullable=True),
    Column("avg_trade_size_clean", DOUBLE_PRECISION, nullable=True),
    Column("hours_observed", DOUBLE_PRECISION, nullable=True),
    Column("turnover_intensity", DOUBLE_PRECISION, nullable=True),
    Column("open_prob", DOUBLE_PRECISION, nullable=True),
    Column("high_prob", DOUBLE_PRECISION, nullable=True),
    Column("low_prob", DOUBLE_PRECISION, nullable=True),
    Column("close_prob", DOUBLE_PRECISION, nullable=True),
    Column("close_prob_1600_et", DOUBLE_PRECISION, nullable=True),
    Column("range_prob", DOUBLE_PRECISION, nullable=True),
    Column("rv_logit_intraday", DOUBLE_PRECISION, nullable=True),
    Column("rv_logit_intraday_clean", DOUBLE_PRECISION, nullable=True),
    Column("gap_logit", DOUBLE_PRECISION, nullable=True),
    Column("distance_to_boundary", DOUBLE_PRECISION, nullable=True),
    Column("buy_sell_imbalance", DOUBLE_PRECISION, nullable=True),
    Column("price_impact_proxy", DOUBLE_PRECISION, nullable=True),
    Column("has_price_data", BOOLEAN, nullable=False),
    Column("has_trade_data", BOOLEAN, nullable=False),
    Column("volume_complete_flag", BOOLEAN, nullable=False),
    Column("price_complete_flag", BOOLEAN, nullable=False),
    Column("sparse_flag", BOOLEAN, nullable=False),
    Column("coverage_class", String(32), nullable=True),
    Column("quality_flag_json", JSONB, nullable=True),
    Column("latest_refresh_id", String(64), nullable=True),
    Column("updated_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_mart",
)
Index("ix_pm_mart_fact_market_day_ticker_date", pm_mart_fact_market_day.c.ticker, pm_mart_fact_market_day.c.trade_date_ny)


pm_mart_fact_stock_day = Table(
    "fact_stock_day",
    metadata,
    Column("trade_date_ny", DATE, primary_key=True),
    Column("ticker", String(32), primary_key=True),
    Column("stock_name", Text, nullable=True),
    Column("active_market_count", INTEGER, nullable=True),
    Column("traded_market_count", INTEGER, nullable=True),
    Column("listed_market_count", INTEGER, nullable=True),
    Column("trade_count", DOUBLE_PRECISION, nullable=True),
    Column("contract_volume", DOUBLE_PRECISION, nullable=True),
    Column("notional_volume", DOUBLE_PRECISION, nullable=True),
    Column("notional_volume_clean", DOUBLE_PRECISION, nullable=True),
    Column("buy_volume", DOUBLE_PRECISION, nullable=True),
    Column("sell_volume", DOUBLE_PRECISION, nullable=True),
    Column("avg_trade_size", DOUBLE_PRECISION, nullable=True),
    Column("turnover_intensity", DOUBLE_PRECISION, nullable=True),
    Column("volume_hhi_within_stock", DOUBLE_PRECISION, nullable=True),
    Column("volume_hhi_within_stock_clean", DOUBLE_PRECISION, nullable=True),
    Column("top_market_share", DOUBLE_PRECISION, nullable=True),
    Column("close_prob_dispersion", DOUBLE_PRECISION, nullable=True),
    Column("median_days_to_resolution", DOUBLE_PRECISION, nullable=True),
    Column("rv_logit_intraday_mean", DOUBLE_PRECISION, nullable=True),
    Column("missingness_rate", DOUBLE_PRECISION, nullable=True),
    Column("active_market_count_clean", DOUBLE_PRECISION, nullable=True),
    Column("sparsity_ratio", DOUBLE_PRECISION, nullable=True),
    Column("volume_complete_flag", BOOLEAN, nullable=False),
    Column("coverage_class", String(32), nullable=True),
    Column("threshold_scope_used", String(64), nullable=True),
    Column("latest_refresh_id", String(64), nullable=True),
    Column("updated_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_mart",
)


pm_mart_fact_global_day = Table(
    "fact_global_day",
    metadata,
    Column("trade_date_ny", DATE, primary_key=True),
    Column("active_stock_count", INTEGER, nullable=True),
    Column("active_market_count", INTEGER, nullable=True),
    Column("traded_market_count", INTEGER, nullable=True),
    Column("trade_count", DOUBLE_PRECISION, nullable=True),
    Column("contract_volume", DOUBLE_PRECISION, nullable=True),
    Column("notional_volume", DOUBLE_PRECISION, nullable=True),
    Column("stock_volume_hhi", DOUBLE_PRECISION, nullable=True),
    Column("top_stock_share", DOUBLE_PRECISION, nullable=True),
    Column("cross_sectional_close_dispersion", DOUBLE_PRECISION, nullable=True),
    Column("missingness_rate", DOUBLE_PRECISION, nullable=True),
    Column("volume_complete_flag", BOOLEAN, nullable=False),
    Column("latest_refresh_id", String(64), nullable=True),
    Column("updated_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_mart",
)


pm_mart_fact_market_lifecycle_day = Table(
    "fact_market_lifecycle_day",
    metadata,
    Column("trade_date_ny", DATE, primary_key=True),
    Column("market_id", String(64), primary_key=True),
    Column("ticker", String(32), nullable=False),
    Column("stock_name", Text, nullable=True),
    Column("lifecycle_day_index", INTEGER, nullable=True),
    Column("lifecycle_total_days", INTEGER, nullable=True),
    Column("lifecycle_progress", DOUBLE_PRECISION, nullable=True),
    Column("days_to_resolution", DOUBLE_PRECISION, nullable=True),
    Column("dtr_bucket", String(32), nullable=True),
    Column("trade_count", DOUBLE_PRECISION, nullable=True),
    Column("contract_volume", DOUBLE_PRECISION, nullable=True),
    Column("notional_volume", DOUBLE_PRECISION, nullable=True),
    Column("close_prob", DOUBLE_PRECISION, nullable=True),
    Column("rv_logit_intraday", DOUBLE_PRECISION, nullable=True),
    Column("traded_flag", BOOLEAN, nullable=False),
    Column("volume_complete_flag", BOOLEAN, nullable=False),
    Column("latest_refresh_id", String(64), nullable=True),
    Column("updated_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_mart",
)


pm_research_variable_profile = Table(
    "variable_profile",
    metadata,
    Column("profile_id", BIGINT, primary_key=True, autoincrement=True),
    Column("refresh_id", String(64), nullable=False),
    Column("scope_level", String(32), nullable=False),
    Column("scope_key", String(128), nullable=False),
    Column("window_label", String(32), nullable=False),
    Column("variable_name", String(64), nullable=False),
    Column("analysis_date", DATE, nullable=True),
    Column("sample_start_date", DATE, nullable=True),
    Column("sample_end_date", DATE, nullable=True),
    Column("observation_count", INTEGER, nullable=True),
    Column("date_count", INTEGER, nullable=True),
    Column("mean", DOUBLE_PRECISION, nullable=True),
    Column("median", DOUBLE_PRECISION, nullable=True),
    Column("stddev", DOUBLE_PRECISION, nullable=True),
    Column("min_value", DOUBLE_PRECISION, nullable=True),
    Column("max_value", DOUBLE_PRECISION, nullable=True),
    Column("p01", DOUBLE_PRECISION, nullable=True),
    Column("p05", DOUBLE_PRECISION, nullable=True),
    Column("p10", DOUBLE_PRECISION, nullable=True),
    Column("p25", DOUBLE_PRECISION, nullable=True),
    Column("p50", DOUBLE_PRECISION, nullable=True),
    Column("p75", DOUBLE_PRECISION, nullable=True),
    Column("p90", DOUBLE_PRECISION, nullable=True),
    Column("p95", DOUBLE_PRECISION, nullable=True),
    Column("p99", DOUBLE_PRECISION, nullable=True),
    Column("iqr", DOUBLE_PRECISION, nullable=True),
    Column("mad", DOUBLE_PRECISION, nullable=True),
    Column("skewness", DOUBLE_PRECISION, nullable=True),
    Column("kurtosis", DOUBLE_PRECISION, nullable=True),
    Column("hill_tail_index", DOUBLE_PRECISION, nullable=True),
    Column("metrics_json", JSONB, nullable=True),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_research",
)
Index("ix_pm_research_variable_profile_scope", pm_research_variable_profile.c.scope_level, pm_research_variable_profile.c.scope_key, pm_research_variable_profile.c.variable_name)


pm_research_threshold_rule = Table(
    "threshold_rule",
    metadata,
    Column("rule_id", BIGINT, primary_key=True, autoincrement=True),
    Column("refresh_id", String(64), nullable=False),
    Column("trade_date_ny", DATE, nullable=False),
    Column("variable_name", String(64), nullable=False),
    Column("scope_level", String(32), nullable=False),
    Column("scope_key", String(128), nullable=False),
    Column("dtr_bucket", String(32), nullable=True),
    Column("window_days", INTEGER, nullable=False),
    Column("sample_count", INTEGER, nullable=True),
    Column("date_count", INTEGER, nullable=True),
    Column("coverage_class", String(32), nullable=True),
    Column("scope_used", String(64), nullable=False),
    Column("q95", DOUBLE_PRECISION, nullable=True),
    Column("q99", DOUBLE_PRECISION, nullable=True),
    Column("q995", DOUBLE_PRECISION, nullable=True),
    Column("median", DOUBLE_PRECISION, nullable=True),
    Column("mad", DOUBLE_PRECISION, nullable=True),
    Column("sigma_robust", DOUBLE_PRECISION, nullable=True),
    Column("cap_robust", DOUBLE_PRECISION, nullable=True),
    Column("cap_quant", DOUBLE_PRECISION, nullable=True),
    Column("upper_cap", DOUBLE_PRECISION, nullable=True),
    Column("authoritative_flag", BOOLEAN, nullable=False),
    Column("sparse_flag", BOOLEAN, nullable=False),
    Column("metrics_json", JSONB, nullable=True),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_research",
)
Index("ix_pm_research_threshold_rule_lookup", pm_research_threshold_rule.c.trade_date_ny, pm_research_threshold_rule.c.variable_name, pm_research_threshold_rule.c.scope_level, pm_research_threshold_rule.c.scope_key)


pm_research_outlier_flag = Table(
    "outlier_flag",
    metadata,
    Column("flag_id", BIGINT, primary_key=True, autoincrement=True),
    Column("refresh_id", String(64), nullable=False),
    Column("trade_date_ny", DATE, nullable=False),
    Column("market_id", String(64), nullable=True),
    Column("ticker", String(32), nullable=True),
    Column("scope_level", String(32), nullable=False),
    Column("scope_key", String(128), nullable=False),
    Column("dtr_bucket", String(32), nullable=True),
    Column("variable_name", String(64), nullable=False),
    Column("raw_value", DOUBLE_PRECISION, nullable=True),
    Column("clean_value", DOUBLE_PRECISION, nullable=True),
    Column("upper_cap", DOUBLE_PRECISION, nullable=True),
    Column("rule_scope_used", String(64), nullable=True),
    Column("clipped_flag", BOOLEAN, nullable=False),
    Column("diagnostic_flags_json", JSONB, nullable=True),
    Column("authoritative_flag", BOOLEAN, nullable=False),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_research",
)
Index("ix_pm_research_outlier_flag_lookup", pm_research_outlier_flag.c.trade_date_ny, pm_research_outlier_flag.c.variable_name, pm_research_outlier_flag.c.ticker)


pm_research_drift_monitor = Table(
    "drift_monitor",
    metadata,
    Column("drift_id", BIGINT, primary_key=True, autoincrement=True),
    Column("refresh_id", String(64), nullable=False),
    Column("scope_level", String(32), nullable=False),
    Column("scope_key", String(128), nullable=False),
    Column("variable_name", String(64), nullable=False),
    Column("window_label", String(32), nullable=False),
    Column("baseline_start_date", DATE, nullable=True),
    Column("baseline_end_date", DATE, nullable=True),
    Column("recent_start_date", DATE, nullable=True),
    Column("recent_end_date", DATE, nullable=True),
    Column("baseline_count", INTEGER, nullable=True),
    Column("recent_count", INTEGER, nullable=True),
    Column("psi", DOUBLE_PRECISION, nullable=True),
    Column("ks_stat", DOUBLE_PRECISION, nullable=True),
    Column("ks_pvalue", DOUBLE_PRECISION, nullable=True),
    Column("median_shift", DOUBLE_PRECISION, nullable=True),
    Column("iqr_ratio", DOUBLE_PRECISION, nullable=True),
    Column("quantile_drift_json", JSONB, nullable=True),
    Column("drift_flag", BOOLEAN, nullable=False),
    Column("authoritative_flag", BOOLEAN, nullable=False),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_research",
)
Index("ix_pm_research_drift_monitor_lookup", pm_research_drift_monitor.c.scope_level, pm_research_drift_monitor.c.scope_key, pm_research_drift_monitor.c.variable_name)


pm_research_break_event = Table(
    "break_event",
    metadata,
    Column("break_id", BIGINT, primary_key=True, autoincrement=True),
    Column("refresh_id", String(64), nullable=False),
    Column("scope_level", String(32), nullable=False),
    Column("scope_key", String(128), nullable=False),
    Column("variable_name", String(64), nullable=False),
    Column("break_date", DATE, nullable=False),
    Column("method", String(32), nullable=False),
    Column("segment_start_date", DATE, nullable=True),
    Column("segment_end_date", DATE, nullable=True),
    Column("before_mean", DOUBLE_PRECISION, nullable=True),
    Column("after_mean", DOUBLE_PRECISION, nullable=True),
    Column("penalty_value", DOUBLE_PRECISION, nullable=True),
    Column("score", DOUBLE_PRECISION, nullable=True),
    Column("metrics_json", JSONB, nullable=True),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_research",
)
Index("ix_pm_research_break_event_lookup", pm_research_break_event.c.scope_level, pm_research_break_event.c.scope_key, pm_research_break_event.c.variable_name, pm_research_break_event.c.break_date)


pm_research_research_note = Table(
    "research_note",
    metadata,
    Column("note_id", BIGINT, primary_key=True, autoincrement=True),
    Column("title", String(240), nullable=False),
    Column("body", Text, nullable=False),
    Column("author", String(120), nullable=True),
    Column("tags_json", JSONB, nullable=True),
    Column("pinned", BOOLEAN, nullable=False, server_default=text("false")),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    Column("updated_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_research",
)


pm_research_refresh_audit = Table(
    "refresh_audit",
    metadata,
    Column("refresh_id", String(64), primary_key=True),
    Column("started_at_utc", TIMESTAMP(timezone=True), nullable=False),
    Column("finished_at_utc", TIMESTAMP(timezone=True), nullable=True),
    Column("ok", BOOLEAN, nullable=False),
    Column("volume_authoritative", BOOLEAN, nullable=False),
    Column("rows_loaded_json", JSONB, nullable=True),
    Column("coverage_json", JSONB, nullable=True),
    Column("params_json", JSONB, nullable=True),
    Column("warning_text", Text, nullable=True),
    Column("error_text", Text, nullable=True),
    Column("created_at_utc", TIMESTAMP(timezone=True), nullable=False, server_default=text("timezone('utc', now())")),
    schema="pm_research",
)


ANALYSIS_TABLES = {
    "pm_raw.weekly_markets": pm_raw_weekly_markets,
    "pm_raw.weekly_events": pm_raw_weekly_events,
    "pm_raw.price_history": pm_raw_price_history,
    "pm_raw.fact_trade": pm_raw_fact_trade,
    "pm_raw.markets_prn_hourly": pm_raw_markets_prn_hourly,
    "pm_raw.decision_features_daily": pm_raw_decision_features_daily,
    "pm_raw.run_manifest": pm_raw_run_manifest,
    "pm_mart.dim_stock": pm_mart_dim_stock,
    "pm_mart.fact_market_day": pm_mart_fact_market_day,
    "pm_mart.fact_stock_day": pm_mart_fact_stock_day,
    "pm_mart.fact_global_day": pm_mart_fact_global_day,
    "pm_mart.fact_market_lifecycle_day": pm_mart_fact_market_lifecycle_day,
    "pm_research.variable_profile": pm_research_variable_profile,
    "pm_research.threshold_rule": pm_research_threshold_rule,
    "pm_research.outlier_flag": pm_research_outlier_flag,
    "pm_research.drift_monitor": pm_research_drift_monitor,
    "pm_research.break_event": pm_research_break_event,
    "pm_research.research_note": pm_research_research_note,
    "pm_research.refresh_audit": pm_research_refresh_audit,
}


def require_analysis_database_url() -> str:
    url = get_analysis_database_url()
    if not url:
        raise RuntimeError(
            "POLYMARKET_ANALYSIS_DATABASE_URL is not configured. "
            "Add it to .env or config/polymarket_analysis.env.sample."
        )
    return url


@lru_cache(maxsize=1)
def get_analysis_engine() -> Engine:
    return create_engine(
        require_analysis_database_url(),
        pool_pre_ping=True,
        future=True,
    )


def ensure_analysis_schemas(engine: Optional[Engine] = None) -> Engine:
    engine = engine or get_analysis_engine()
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
    metadata.create_all(engine)
    return engine


def reset_analysis_cache() -> None:
    get_analysis_engine.cache_clear()


def upsert_rows(
    conn,
    table: Table,
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_columns: Optional[Sequence[str]] = None,
    update_columns: Optional[Sequence[str]] = None,
    chunk_size: int = 1000,
) -> int:
    if not rows:
        return 0

    if conflict_columns is None:
        conflict_columns = [column.name for column in table.primary_key.columns]

    if update_columns is None:
        update_columns = [
            column.name
            for column in table.columns
            if column.name not in set(conflict_columns)
            and not column.primary_key
        ]

    total = 0
    for start in range(0, len(rows), chunk_size):
        batch = rows[start : start + chunk_size]
        stmt = pg_insert(table).values(list(batch))
        if conflict_columns:
            update_map = {column: getattr(stmt.excluded, column) for column in update_columns}
            if update_map:
                stmt = stmt.on_conflict_do_update(
                    index_elements=list(conflict_columns),
                    set_=update_map,
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_columns))
        conn.execute(stmt)
        total += len(batch)
    return total


def replace_table_rows(conn, table: Table, rows: Iterable[Mapping[str, Any]]) -> int:
    conn.execute(table.delete())
    data = list(rows)
    if not data:
        return 0
    conn.execute(table.insert(), data)
    return len(data)


def list_analysis_tables() -> dict[str, Table]:
    return ANALYSIS_TABLES.copy()
