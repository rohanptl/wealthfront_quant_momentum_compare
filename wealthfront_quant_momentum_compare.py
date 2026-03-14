#!/usr/bin/env python3
"""
Compare Wealthfront Quantitative Momentum strategy performance across
weekly, monthly, and quarterly rebalance schedules.

This reuses the same logic as wealthfront_quant_momentum_backtest.py:
1) ETF universe constrained to WealthfrontETFs.txt
2) 12-1 momentum (12-month return skipping the latest month)
3) Frog-in-the-Pan (FIP) quality filter over 252 trading days
4) SPY vs SGOV trailing 12-month regime filter
5) Equal-weight top-k ETFs with 20% cap, remainder to SGOV

The comparison runs the same strategy under multiple rebalance rules and
writes:
- summary CSV with side-by-side stats
- one equity curve CSV per schedule
- one rebalance log CSV per schedule
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("yfinance is required. Install with: pip install yfinance pandas numpy") from exc


TICKER_RE = re.compile(r"^[A-Z0-9.-]{1,10}$")
DEFAULT_RULES = ["W-FRI", "ME", "QE"]


def parse_universe_file(path: Path) -> Tuple[Dict[str, str], List[str]]:
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in raw_lines if line]

    name_by_ticker: Dict[str, str] = {}
    tickers: List[str] = []
    pending_name: str | None = None

    for line in lines:
        if TICKER_RE.fullmatch(line) and line.upper() == line:
            ticker = line.strip()
            if ticker not in name_by_ticker:
                name_by_ticker[ticker] = pending_name or ticker
                tickers.append(ticker)
            pending_name = None
        else:
            pending_name = line

    if "SGOV" not in name_by_ticker:
        name_by_ticker["SGOV"] = "iShares 0-3 Month Treasury Bond ETF"
        tickers.append("SGOV")

    return name_by_ticker, sorted(set(tickers))


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if data.empty:
        raise RuntimeError("No price data downloaded. Check internet connection, dates, and tickers.")

    if isinstance(data.columns, pd.MultiIndex):
        closes = {}
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                try:
                    closes[ticker] = data[ticker]["Close"]
                except Exception:
                    continue
        price_df = pd.DataFrame(closes)
    else:
        price_df = pd.DataFrame({tickers[0]: data["Close"]})

    return price_df.sort_index().ffill().dropna(how="all")


def total_return_skip_recent_month(series: pd.Series, month_days: int = 21, year_days: int = 252) -> float:
    s = series.dropna()
    if len(s) < year_days + 1:
        return np.nan
    end_idx = len(s) - 1 - month_days
    start_idx = len(s) - 1 - year_days
    if start_idx < 0 or end_idx <= start_idx:
        return np.nan
    start_price = s.iloc[start_idx]
    end_price = s.iloc[end_idx]
    if start_price <= 0 or end_price <= 0:
        return np.nan
    return end_price / start_price - 1.0


def fip_score(series: pd.Series, lookback_days: int = 252) -> float:
    s = series.dropna()
    if len(s) < lookback_days + 1:
        return np.nan
    window = s.iloc[-(lookback_days + 1):]
    rets = window.pct_change().dropna()
    if rets.empty:
        return np.nan
    past_return = window.iloc[-1] / window.iloc[0] - 1.0
    sign = 1.0 if past_return >= 0 else -1.0
    pct_negative = float((rets < 0).mean())
    pct_positive = float((rets > 0).mean())
    return sign * (pct_negative - pct_positive)


def regime_is_risk_on(window_prices: pd.DataFrame, market_ticker: str = "SPY", cash_ticker: str = "SGOV") -> bool:
    market_ret = total_return_skip_recent_month(window_prices[market_ticker], month_days=0, year_days=252)
    cash_ret = total_return_skip_recent_month(window_prices[cash_ticker], month_days=0, year_days=252)
    if math.isnan(market_ret) or math.isnan(cash_ret):
        return False
    return market_ret > cash_ret


def compute_scores(window_prices: pd.DataFrame, universe: List[str], cash_ticker: str = "SGOV") -> pd.DataFrame:
    rows = []
    for ticker in universe:
        if ticker == cash_ticker or ticker not in window_prices.columns:
            continue
        series = window_prices[ticker].dropna()
        if len(series) < 300:
            continue
        mom = total_return_skip_recent_month(series, month_days=21, year_days=252)
        fip = fip_score(series, lookback_days=252)
        latest = float(series.iloc[-1])
        rows.append({
            "ticker": ticker,
            "momentum_12_1": mom,
            "fip": fip,
            "latest_price": latest,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["momentum_12_1", "fip", "latest_price"])
    df = df[df["momentum_12_1"] > 0].copy()
    if df.empty:
        return df

    df["mom_rank"] = df["momentum_12_1"].rank(ascending=False, method="first")
    df["fip_rank"] = df["fip"].rank(ascending=True, method="first")
    df["composite_rank"] = df["mom_rank"] + 0.5 * df["fip_rank"]
    df = df.sort_values(["composite_rank", "momentum_12_1", "fip"], ascending=[True, False, True])
    return df.reset_index(drop=True)


def build_target_map(scores: pd.DataFrame, top_k: int, max_allocation: float, cash_ticker: str = "SGOV") -> Dict[str, float]:
    if scores.empty:
        return {cash_ticker: 1.0}

    max_positions_from_cap = max(1, int(math.floor(1.0 / max_allocation)))
    count = min(top_k, len(scores), max_positions_from_cap)
    selected = scores.head(count)

    per_weight = min(max_allocation, 1.0 / count)
    target = {row["ticker"]: per_weight for _, row in selected.iterrows()}
    residual = max(0.0, 1.0 - sum(target.values()))
    target[cash_ticker] = residual
    total = sum(target.values())
    return {k: v / total for k, v in target.items()}


def make_rebalance_dates(price_index: pd.DatetimeIndex, rule: str) -> List[pd.Timestamp]:
    # Normalize deprecated pandas offset aliases
    normalized_rule = rule.replace('M', 'ME').replace('Q', 'QE')
    s = pd.Series(1, index=price_index)
    return list(s.resample(normalized_rule).last().dropna().index)


def align_to_next_trading_day(price_index: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp | None:
    later = price_index[price_index > dt]
    if len(later) == 0:
        return None
    return later[0]


def portfolio_turnover(old_w: Dict[str, float], new_w: Dict[str, float], universe: List[str]) -> float:
    tickers = set(universe) | set(old_w) | set(new_w)
    abs_change = sum(abs(new_w.get(t, 0.0) - old_w.get(t, 0.0)) for t in tickers)
    return 0.5 * abs_change


def compute_performance_stats(equity_curve: pd.Series, daily_returns: pd.Series) -> Dict[str, float]:
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    days = max(1, len(daily_returns))
    cagr = equity_curve.iloc[-1] ** (252.0 / days) - 1.0
    vol = float(daily_returns.std(ddof=0) * np.sqrt(252))
    sharpe = float((daily_returns.mean() * 252) / vol) if vol > 0 else np.nan
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def backtest(
    prices: pd.DataFrame,
    universe: List[str],
    start_date: str,
    end_date: str,
    rebalance_rule: str,
    top_k: int,
    max_allocation: float,
    cash_ticker: str,
    market_ticker: str,
    transaction_cost_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    prices = prices.loc[(prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))].copy()
    prices = prices.sort_index().ffill().dropna(how="all")
    prices = prices[[c for c in prices.columns if c in set(universe) | {cash_ticker, market_ticker}]]
    if prices.empty:
        raise RuntimeError("No prices in requested backtest date range.")

    rebal_dates = make_rebalance_dates(prices.index, rebalance_rule)
    warmup_cutoff = prices.index[300] if len(prices.index) > 300 else prices.index[-1]
    rebal_dates = [d for d in rebal_dates if d >= warmup_cutoff]
    if not rebal_dates:
        raise RuntimeError(f"Not enough data after warmup to run rule {rebalance_rule}.")

    current_weights: Dict[str, float] = {cash_ticker: 1.0}
    daily_records = []
    rebalance_records = []
    portfolio_value = 1.0
    rebalance_map: Dict[pd.Timestamp, Dict[str, float]] = {}
    rebalance_meta: Dict[pd.Timestamp, Dict[str, object]] = {}

    for dt in rebal_dates:
        window_prices = prices.loc[:dt]
        risk_on = regime_is_risk_on(window_prices, market_ticker=market_ticker, cash_ticker=cash_ticker)
        if risk_on:
            scores = compute_scores(window_prices, universe=universe, cash_ticker=cash_ticker)
            target = build_target_map(scores, top_k=top_k, max_allocation=max_allocation, cash_ticker=cash_ticker)
            leaders = list(scores.head(min(top_k, len(scores)))["ticker"]) if not scores.empty else []
        else:
            target = {cash_ticker: 1.0}
            leaders = [cash_ticker]

        effective_dt = align_to_next_trading_day(prices.index, dt)
        if effective_dt is None:
            continue
        rebalance_map[effective_dt] = target
        rebalance_meta[effective_dt] = {
            "signal_date": dt,
            "risk_on": risk_on,
            "leaders": ",".join(leaders),
        }

    if not rebalance_map:
        raise RuntimeError(f"No effective rebalance dates were generated for {rebalance_rule}.")

    daily_ret = prices.pct_change().fillna(0.0)

    for dt in prices.index:
        if dt in rebalance_map:
            new_weights = rebalance_map[dt]
            turnover = portfolio_turnover(current_weights, new_weights, universe=universe)
            cost = turnover * (transaction_cost_bps / 10000.0)
            portfolio_value *= (1.0 - cost)
            current_weights = new_weights.copy()
            rebalance_records.append(
                {
                    "effective_date": dt,
                    "signal_date": rebalance_meta[dt]["signal_date"],
                    "risk_on": rebalance_meta[dt]["risk_on"],
                    "leaders": rebalance_meta[dt]["leaders"],
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "weights": dict(sorted(current_weights.items())),
                }
            )

        day_return = sum(weight * float(daily_ret.at[dt, ticker]) for ticker, weight in current_weights.items() if ticker in daily_ret.columns)
        portfolio_value *= (1.0 + day_return)
        row = {
            "date": dt,
            "portfolio_value": portfolio_value,
            "portfolio_return": day_return,
        }
        for ticker in sorted(current_weights):
            row[f"w_{ticker}"] = current_weights[ticker]
        daily_records.append(row)

    equity_df = pd.DataFrame(daily_records).set_index("date")
    equity_df["drawdown"] = equity_df["portfolio_value"] / equity_df["portfolio_value"].cummax() - 1.0

    rebalance_df = pd.DataFrame(rebalance_records)
    stats = compute_performance_stats(equity_df["portfolio_value"], equity_df["portfolio_return"])
    stats["avg_turnover_per_rebalance"] = float(rebalance_df["turnover"].mean()) if not rebalance_df.empty else np.nan
    stats["rebalances"] = int(len(rebalance_df))
    stats["start_date"] = str(equity_df.index.min().date())
    stats["end_date"] = str(equity_df.index.max().date())
    return equity_df, rebalance_df, stats


def rule_label(rule: str) -> str:
    mapping = {"W-FRI": "weekly", "M": "monthly", "ME": "monthly", "Q": "quarterly", "QE": "quarterly"}
    return mapping.get(rule, rule.lower().replace("-", "_"))


def print_comparison(summary_df: pd.DataFrame) -> None:
    display = summary_df.copy()
    pct_cols = ["total_return", "cagr", "annual_volatility", "max_drawdown", "avg_turnover_per_rebalance"]
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "nan")
    if "sharpe" in display.columns:
        display["sharpe"] = display["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
    print("\n=== REBALANCE FREQUENCY COMPARISON ===")
    print(display.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare weekly, monthly, quarterly backtests for Wealthfront Quantitative Momentum")
    parser.add_argument("--universe", default="WealthfrontETFs.txt", help="Path to Wealthfront ETF universe file")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date, e.g. 2020-01-01")
    parser.add_argument("--end", default=None, help="Backtest end date, e.g. 2026-03-13")
    parser.add_argument("--rules", nargs="+", default=DEFAULT_RULES, help="Rebalance rules to compare, e.g. W-FRI ME QE (or deprecated M Q)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of ETFs to hold in risk-on mode")
    parser.add_argument("--max-allocation", type=float, default=0.20, help="Max allocation per ETF except SGOV")
    parser.add_argument("--cash-ticker", default="SGOV", help="Ticker used as cash sleeve")
    parser.add_argument("--market-ticker", default="SPY", help="Ticker used for regime filter")
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0, help="Transaction cost in bps per dollar traded")
    parser.add_argument("--summary-output", default="backtest_frequency_comparison.csv", help="CSV output for schedule comparison summary")
    parser.add_argument("--output-dir", default=".", help="Directory for per-schedule CSV outputs")
    args = parser.parse_args()

    if args.max_allocation <= 0 or args.max_allocation > 1:
        raise SystemExit("--max-allocation must be between 0 and 1.")

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, universe = parse_universe_file(Path(args.universe))
    required = {args.cash_ticker, args.market_ticker}
    missing_required = required - set(universe)
    if missing_required:
        raise SystemExit(f"Required tickers missing from universe file: {sorted(missing_required)}")

    warmup_start = (pd.Timestamp(args.start) - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
    price_tickers = sorted(set(universe) | required)
    prices = download_prices(price_tickers, start=warmup_start, end=end)

    summary_rows = []
    for rule in args.rules:
        label = rule_label(rule)
        equity_df, rebalance_df, stats = backtest(
            prices=prices,
            universe=universe,
            start_date=args.start,
            end_date=end,
            rebalance_rule=rule,
            top_k=args.top_k,
            max_allocation=args.max_allocation,
            cash_ticker=args.cash_ticker,
            market_ticker=args.market_ticker,
            transaction_cost_bps=args.transaction_cost_bps,
        )

        equity_path = output_dir / f"backtest_equity_{label}.csv"
        rebalance_path = output_dir / f"backtest_rebalances_{label}.csv"
        equity_df.to_csv(equity_path)
        rebalance_df.to_csv(rebalance_path, index=False)

        summary_rows.append({
            "schedule": label,
            "rule": rule,
            **stats,
            "equity_curve_file": str(equity_path),
            "rebalance_file": str(rebalance_path),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("cagr", ascending=False)
    summary_df.to_csv(output_dir / args.summary_output, index=False)
    print_comparison(summary_df[[
        "schedule", "rule", "total_return", "cagr", "annual_volatility", "sharpe",
        "max_drawdown", "avg_turnover_per_rebalance", "rebalances", "start_date", "end_date"
    ]])
    print(f"\nSaved comparison summary to: {output_dir / args.summary_output}")


if __name__ == "__main__":
    main()
