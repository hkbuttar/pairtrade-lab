"""Event-driven, walk-forward pairs-trading backtest.

Steps through trading days chronologically (not vectorized over the whole
range at once), matching the no-lookahead discipline in alpha-signal-lab's
backtest/engine.py:

    for each trading day:
        1. fill any orders decided the previous day, at today's open
        2. mark the portfolio to market at today's close
        3. on days a pair's signal transitions (entry/exit), queue orders
           for tomorrow's open

Walk-forward structure, per the project plan: pair selection
(pairs.cointegration.test_sector_pairs) and hedge ratio estimation are
re-run at each refit date using only data available strictly before that
date, and the resulting pairs/estimates are then traded purely
out-of-sample until the next refit. A pair that continues to qualify across
a refit boundary keeps its existing position and its hedge ratio/signal
state (recomputed from the same tenure start, not reset to flat); a pair
that drops out of selection is flattened.

Baskets (pairs.johansen) are not wired into this engine yet: multi-leg
position/fill bookkeeping adds real complexity, and the basket search so
far has found very few FDR-significant baskets to trade anyway (see
pairs/README.md). Disclosed as a scope limitation, not an oversight.

Position sizing: each active pair is allocated a fixed notional
(``notional_per_pair``) split across its two legs in the *exact share ratio*
implied by its hedge ratio (1 share of y : -hedge_ratio shares of x), sized
so the y-leg's notional equals ``notional_per_pair``. This is deliberately
not dollar-neutral: dollar-neutral would trade a different linear
combination than the one actually tested for cointegration and would no
longer be the stationary spread the whole strategy is betting on reverting.
Share counts are locked in at entry and held fixed until exit (not
continuously re-targeted to notional every day, which would trade every
single day purely to chase a moving target and is not how pairs trading
works in practice).

Risk layer (risk/): every new entry's notional is clipped by
risk.limits.clip_new_pair_notional (a per-pair cap, then whatever gross
exposure budget remains once already-open positions are counted, both as a
fraction of *current* equity) before it becomes an order. Stop-loss on
spread divergence is signals.spread.generate_signals's own
stop_loss_threshold, already wired in via _compute_pair_series. A portfolio-
level risk.kill_switch.KillSwitch checks equity every bar; once triggered
(sticky, no auto-resume) every open position is flattened and no further
selection or signal processing happens for the rest of the run, the same
manual-reset-only design used across this portfolio's other backtesters.
"""

from __future__ import annotations

import pandas as pd

from backtest import fills
from backtest import metrics as backtest_metrics
from backtest.portfolio import Portfolio
from config.universe import SECTOR_TICKERS, UNIVERSE
from data.prices import load_prices, to_price_panel
from monitor.structural_break import cusum_detect, monitor_pair_status, rolling_cointegration_pvalue
from pairs.cointegration import test_sector_pairs
from risk.kill_switch import KillSwitch
from risk.limits import clip_new_pair_notional
from signals.hedge_ratio import kalman_hedge_ratio_series, static_hedge_ratio_series
from signals.spread import generate_signals, pair_spread, rolling_zscore

ORDER_EPSILON = 1e-6
DEFAULT_HISTORY_BUFFER_DAYS = 1095  # ~3 years lookback for the first refit's selection test


def _refit_dates(trading_dates: pd.DatetimeIndex, refit_every_days: int) -> list[pd.Timestamp]:
    """Snap [start, start+refit_every_days, ...] to the next actual trading date."""
    if len(trading_dates) == 0:
        return []
    dates = []
    candidate = trading_dates[0]
    while candidate <= trading_dates[-1]:
        snapped = trading_dates[trading_dates >= candidate]
        if len(snapped) == 0:
            break
        dates.append(snapped[0])
        candidate = snapped[0] + pd.Timedelta(days=refit_every_days)
    return dates


def _compute_pair_series(
    close_panel: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    tenure_start: pd.Timestamp,
    window_end: pd.Timestamp,
    hedge_ratio_method: str,
    zscore_window: int,
    entry_threshold: float,
    exit_threshold: float,
    stop_loss_threshold: float,
    rolling_cointegration_window: int,
    rolling_cointegration_step: int,
    cusum_k: float,
    cusum_h: float,
    pvalue_threshold: float,
    requalify_bars: int,
) -> dict:
    y = close_panel.loc[tenure_start:window_end, ticker_y]
    x = close_panel.loc[tenure_start:window_end, ticker_x]

    if hedge_ratio_method == "kalman":
        estimate = kalman_hedge_ratio_series(y, x)
    else:
        estimate = static_hedge_ratio_series(y, x)

    spread = pair_spread(y, x, estimate["hedge_ratio"], estimate["intercept"])
    zscore = rolling_zscore(spread, window=zscore_window)
    rolling_pvalue = rolling_cointegration_pvalue(
        y, x, window=rolling_cointegration_window, step=rolling_cointegration_step
    )
    cusum = cusum_detect(zscore, k=cusum_k, h=cusum_h)
    monitor_status = monitor_pair_status(
        rolling_pvalue,
        cusum["break_flagged"],
        pvalue_threshold=pvalue_threshold,
        requalify_bars=requalify_bars,
    )
    positions = generate_signals(
        zscore, entry_threshold, exit_threshold, stop_loss_threshold, monitor_status=monitor_status
    )

    return {
        "hedge_ratio": estimate["hedge_ratio"],
        "zscore": zscore,
        "monitor_status": monitor_status,
        "positions": positions,
    }


def _run_backtest_on_panel(
    close_panel: pd.DataFrame,
    open_panel: pd.DataFrame,
    sector_tickers: dict[str, list[str]],
    start: str,
    end: str,
    refit_every_days: int = 365,
    max_pairs: int = 3,
    hedge_ratio_method: str = "kalman",
    zscore_window: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_threshold: float = 3.5,
    rolling_cointegration_window: int = 90,
    rolling_cointegration_step: int = 5,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
    pvalue_threshold: float = 0.05,
    requalify_bars: int = 20,
    fdr_alpha: float = 0.05,
    cost_bps: float = 5.0,
    starting_cash: float = 1_000_000.0,
    notional_per_pair: float | None = None,
    max_notional_per_pair_fraction: float = 0.5,
    max_gross_exposure_fraction: float = 1.0,
    kill_switch_drawdown: float = 0.15,
) -> dict:
    trading_dates = close_panel.index[
        (close_panel.index >= pd.Timestamp(start)) & (close_panel.index < pd.Timestamp(end))
    ]
    refit_dates = _refit_dates(trading_dates, refit_every_days)
    notional_per_pair = notional_per_pair or starting_cash / max_pairs

    portfolio = Portfolio(starting_cash)
    kill_switch = KillSwitch(kill_switch_drawdown)
    halted = False
    tracked: dict[tuple[str, str], dict] = {}
    trades: list[dict] = []
    open_trade_meta: dict[tuple[str, str], dict] = {}
    pending_ticker_orders: dict[str, float] = {}
    pending_pair_shares: dict[tuple[str, str], dict[str, float]] = {}

    for cycle_idx, refit_date in enumerate(refit_dates):
        window_end = (
            refit_dates[cycle_idx + 1] if cycle_idx + 1 < len(refit_dates) else trading_dates[-1]
        )

        if not halted:
            selection_panel = close_panel.loc[close_panel.index < refit_date]
            selection = test_sector_pairs(selection_panel, sector_tickers, fdr_alpha=fdr_alpha)
            significant = selection[selection["cointegrated"]].sort_values("p_value_fdr")
            selected_pairs = [
                (row.ticker_y, row.ticker_x) for row in significant.head(max_pairs).itertuples()
            ]

            for pair in list(tracked):
                if pair not in selected_pairs:
                    shares = tracked[pair]["shares"]
                    y_t, x_t = pair
                    if abs(shares["y"]) > ORDER_EPSILON or abs(shares["x"]) > ORDER_EPSILON:
                        pending_ticker_orders[y_t] = (
                            pending_ticker_orders.get(y_t, 0.0) - shares["y"]
                        )
                        pending_ticker_orders[x_t] = (
                            pending_ticker_orders.get(x_t, 0.0) - shares["x"]
                        )
                        meta = open_trade_meta.pop(pair, None)
                        if meta is not None:
                            trades.append(
                                {**meta, "exit_date": refit_date, "exit_reason": "dropped"}
                            )
                    del tracked[pair]

            for pair in selected_pairs:
                ticker_y, ticker_x = pair
                tenure_start = tracked[pair]["tenure_start"] if pair in tracked else refit_date
                try:
                    series = _compute_pair_series(
                        close_panel,
                        ticker_y,
                        ticker_x,
                        tenure_start,
                        window_end,
                        hedge_ratio_method,
                        zscore_window,
                        entry_threshold,
                        exit_threshold,
                        stop_loss_threshold,
                        rolling_cointegration_window,
                        rolling_cointegration_step,
                        cusum_k,
                        cusum_h,
                        pvalue_threshold,
                        requalify_bars,
                    )
                except ValueError:
                    # Not enough observations yet in this tenure (e.g. a pair
                    # just selected with a short history since its own start).
                    # Skip it this cycle rather than crash the whole backtest;
                    # it can be picked up again at the next refit once it has
                    # enough history, or dropped from `selected_pairs` entirely
                    # if it stops testing significant by then.
                    continue
                existing = tracked.get(pair, {})
                tracked[pair] = {
                    "tenure_start": tenure_start,
                    "shares": existing.get("shares", {"y": 0.0, "x": 0.0}),
                    "notional": existing.get("notional", 0.0),
                    **series,
                }

        cycle_dates = trading_dates[(trading_dates >= refit_date) & (trading_dates < window_end)]
        for date in cycle_dates:
            if pending_ticker_orders:
                today_open = open_panel.loc[date]
                for ticker, shares_delta in pending_ticker_orders.items():
                    next_open = today_open.get(ticker)
                    if pd.isna(next_open):
                        continue
                    fill_price, cost = fills.compute_fill(shares_delta, next_open, cost_bps)
                    portfolio.apply_fill(ticker, shares_delta, fill_price, cost)
                for pair, new_shares in pending_pair_shares.items():
                    if pair in tracked:
                        tracked[pair]["shares"] = new_shares
                pending_ticker_orders = {}
                pending_pair_shares = {}

            equity = portfolio.mark_to_market(date, close_panel.loc[date].to_dict())

            if not halted and kill_switch.check(equity):
                halted = True
                pending_ticker_orders = portfolio.flatten_orders()
                continue
            if halted:
                continue

            for pair, state in tracked.items():
                if date not in state["positions"].index:
                    continue
                pos_today = state["positions"].loc[date]
                prior = state["positions"].shift(1).loc[date]
                pos_yesterday = 0 if pd.isna(prior) else prior
                if pos_today == pos_yesterday:
                    continue

                ticker_y, ticker_x = pair
                hedge_ratio_today = state["hedge_ratio"].loc[date]
                y_price = close_panel.loc[date, ticker_y]

                if pos_today == 0:
                    delta_y, delta_x = -state["shares"]["y"], -state["shares"]["x"]
                    new_shares = {"y": 0.0, "x": 0.0}
                    tracked[pair]["notional"] = 0.0
                    meta = open_trade_meta.pop(pair, None)
                    if meta is not None:
                        z = state["zscore"].loc[date]
                        if state["monitor_status"].loc[date] == "HALTED":
                            reason = "halted"
                        elif pd.notna(z) and abs(z) <= exit_threshold:
                            reason = "reversion"
                        else:
                            reason = "stop_loss"
                        trades.append({**meta, "exit_date": date, "exit_reason": reason})
                elif pd.notna(hedge_ratio_today) and pd.notna(y_price) and y_price != 0:
                    raw_notional = pos_today * notional_per_pair
                    existing_notional = {
                        p: s["notional"] for p, s in tracked.items() if p != pair
                    }
                    clipped_notional = clip_new_pair_notional(
                        raw_notional,
                        existing_notional,
                        equity,
                        max_notional_per_pair_fraction,
                        max_gross_exposure_fraction,
                    )
                    if clipped_notional == 0.0:
                        continue

                    target_y = clipped_notional / y_price
                    target_x = -hedge_ratio_today * clipped_notional / y_price
                    delta_y = target_y - state["shares"]["y"]
                    delta_x = target_x - state["shares"]["x"]
                    new_shares = {"y": target_y, "x": target_x}
                    tracked[pair]["notional"] = clipped_notional
                    open_trade_meta[pair] = {
                        "side": "long" if pos_today == 1 else "short",
                        "entry_date": date,
                        "pair": f"{ticker_y}/{ticker_x}",
                    }
                else:
                    continue

                pending_ticker_orders[ticker_y] = pending_ticker_orders.get(ticker_y, 0.0) + delta_y
                pending_ticker_orders[ticker_x] = pending_ticker_orders.get(ticker_x, 0.0) + delta_x
                pending_pair_shares[pair] = new_shares

    for meta in open_trade_meta.values():
        trades.append({**meta, "exit_date": trading_dates[-1], "exit_reason": "open"})

    trades_df = pd.DataFrame(
        trades, columns=["pair", "side", "entry_date", "exit_date", "exit_reason"]
    )
    if not trades_df.empty:
        trades_df["bars_held"] = trades_df.apply(
            lambda r: trading_dates.searchsorted(r["exit_date"])
            - trading_dates.searchsorted(r["entry_date"]),
            axis=1,
        )

    equity_curve = portfolio.equity_series()
    metrics = backtest_metrics.compute_metrics(equity_curve, trades_df)

    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": metrics,
        "kill_switch_triggered": kill_switch.triggered,
    }


def run_backtest(
    start: str,
    end: str,
    universe: list[str] | None = None,
    sector_tickers: dict[str, list[str]] | None = None,
    price_source: str = "yfinance",
    history_buffer_days: int = DEFAULT_HISTORY_BUFFER_DAYS,
    **kwargs,
) -> dict:
    """Load data and run the walk-forward pairs backtest over [start, end).

    Args:
        start: Backtest start date (out-of-sample trading begins here).
        end: Backtest end date, exclusive.
        universe: Tickers to load; defaults to config.universe.UNIVERSE.
        sector_tickers: Sector groupings for selection; defaults to
            config.universe.SECTOR_TICKERS.
        price_source: "yfinance" or "alpaca".
        history_buffer_days: Calendar days of price history loaded before
            ``start``, so the first refit's selection test has a real
            lookback rather than starting from nothing.
        **kwargs: Passed through to _run_backtest_on_panel (refit_every_days,
            max_pairs, hedge_ratio_method, zscore_window, entry/exit/
            stop_loss thresholds, rolling_cointegration_window/step, cusum_k/
            h, pvalue_threshold, requalify_bars, fdr_alpha, cost_bps,
            starting_cash, notional_per_pair, max_notional_per_pair_fraction,
            max_gross_exposure_fraction, kill_switch_drawdown).

    Returns:
        Dict with "equity_curve" (pd.Series), "trades" (pd.DataFrame),
        "metrics" (dict), and "kill_switch_triggered" (bool).
    """
    tickers = universe or UNIVERSE
    sectors = sector_tickers or SECTOR_TICKERS
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=history_buffer_days)).strftime(
        "%Y-%m-%d"
    )

    long_prices = load_prices(tickers, load_start, end, source=price_source)
    close_panel = to_price_panel(long_prices, field="close")
    open_panel = to_price_panel(long_prices, field="open")

    return _run_backtest_on_panel(close_panel, open_panel, sectors, start, end, **kwargs)
