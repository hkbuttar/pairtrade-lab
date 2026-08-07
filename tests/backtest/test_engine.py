import numpy as np
import pandas as pd

from backtest.engine import _refit_dates, _run_backtest_on_panel

N = 700


def _synthetic_panel(seed: int = 0):
    """AAA/BBB genuinely cointegrated throughout; CCC/DDD independent random
    walks that never should qualify. All four in one sector, so selection has
    something to reject as well as something to find.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=N)

    w = np.cumsum(rng.normal(0, 1, N))
    aaa = w + rng.normal(0, 0.3, N) + 100
    bbb = 0.5 * w + rng.normal(0, 0.3, N) + 50
    ccc = np.cumsum(rng.normal(0, 1, N)) + 80
    ddd = np.cumsum(rng.normal(0, 1, N)) + 60

    close = pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc, "DDD": ddd}, index=dates)
    # Approximate next-bar open as the following close, adequate for a
    # backtest-mechanics smoke test that isn't trying to validate real
    # open/close microstructure.
    open_ = close.shift(-1).bfill()
    return close, open_


def _run(cost_bps=5.0, **overrides):
    close, open_ = _synthetic_panel()
    sector_tickers = {"S": ["AAA", "BBB", "CCC", "DDD"]}
    kwargs = dict(
        start="2016-06-01",
        end="2018-01-01",
        refit_every_days=180,
        max_pairs=2,
        rolling_cointegration_step=5,
        cost_bps=cost_bps,
    )
    kwargs.update(overrides)
    return _run_backtest_on_panel(close, open_, sector_tickers, **kwargs)


def test_refit_dates_are_spaced_and_snap_to_trading_calendar():
    trading_dates = pd.bdate_range("2020-01-01", periods=400)

    dates = _refit_dates(trading_dates, refit_every_days=90)

    assert dates[0] == trading_dates[0]
    assert all(d in trading_dates for d in dates)
    assert list(dates) == sorted(dates)
    assert len(dates) > 1


def test_run_backtest_produces_a_full_equity_curve():
    result = _run()

    equity = result["equity_curve"]
    assert not equity.empty
    assert equity.notna().all()
    assert (equity > 0).all()  # starting cash 1e6, no leverage blowup expected here


def test_run_backtest_trades_table_is_well_formed():
    result = _run()
    trades = result["trades"]

    assert set(trades.columns) >= {"pair", "side", "entry_date", "exit_date", "bars_held"}
    assert trades["side"].isin(["long", "short"]).all()
    assert (trades["exit_date"] >= trades["entry_date"]).all()
    assert (trades["bars_held"] >= 0).all()


def test_run_backtest_only_trades_the_cointegrated_pair():
    result = _run()
    trades = result["trades"]

    if not trades.empty:
        assert set(trades["pair"]) <= {"AAA/BBB"}


def test_transaction_costs_reduce_terminal_equity():
    cheap = _run(cost_bps=0.0)
    expensive = _run(cost_bps=100.0)

    n_trades = len(cheap["trades"])
    if n_trades == 0:
        return  # nothing to compare if the synthetic run had no trades

    assert expensive["equity_curve"].iloc[-1] <= cheap["equity_curve"].iloc[-1]


def test_run_backtest_metrics_present_and_finite_where_expected():
    result = _run()

    metrics = result["metrics"]
    assert metrics["n_trades"] == len(result["trades"])
    assert np.isfinite(metrics["max_drawdown"])


def test_kill_switch_reported_untriggered_under_normal_conditions():
    result = _run()

    assert result["kill_switch_triggered"] is False


def test_kill_switch_triggers_and_halts_further_trading():
    # An extremely tight drawdown threshold combined with a high transaction
    # cost guarantees the very first entry's fees alone breach it, so the
    # trigger is deterministic regardless of which way the trade would have
    # gone.
    result = _run(kill_switch_drawdown=1e-5, cost_bps=100.0)

    assert result["kill_switch_triggered"] is True
    equity = result["equity_curve"]
    # Once halted, nothing further should move equity: the tail of the
    # curve should be flat.
    assert equity.iloc[-10:].nunique() == 1


def test_zero_notional_limit_suppresses_all_entries():
    result = _run(max_notional_per_pair_fraction=0.0, max_gross_exposure_fraction=0.0)

    assert result["trades"].empty


def test_use_monitor_false_never_produces_halted_exits():
    result = _run(use_monitor=False)

    trades = result["trades"]
    assert "halted" not in set(trades["exit_reason"])
