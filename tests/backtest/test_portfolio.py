import pandas as pd

from backtest.portfolio import Portfolio


def test_apply_fill_updates_cash_and_position():
    p = Portfolio(starting_cash=1000.0)

    p.apply_fill("AAA", 10.0, 5.0, cost=1.0)

    assert p.cash == 1000.0 - 50.0 - 1.0
    assert p.positions["AAA"] == 10.0


def test_apply_fill_closes_position_below_epsilon():
    p = Portfolio(starting_cash=1000.0)
    p.apply_fill("AAA", 10.0, 5.0, cost=0.0)

    p.apply_fill("AAA", -10.0, 5.0, cost=0.0)

    assert "AAA" not in p.positions


def test_mark_to_market_returns_cash_plus_position_value():
    p = Portfolio(starting_cash=1000.0)
    p.apply_fill("AAA", 10.0, 5.0, cost=0.0)

    equity = p.mark_to_market(pd.Timestamp("2020-01-01"), {"AAA": 6.0})

    assert equity == 950.0 + 60.0  # cash after buying 10@5 (950) + 10 shares @6


def test_equity_series_sorted_by_date():
    p = Portfolio(starting_cash=100.0)
    p.mark_to_market(pd.Timestamp("2020-01-02"), {})
    p.mark_to_market(pd.Timestamp("2020-01-01"), {})

    series = p.equity_series()

    assert list(series.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_equity_series_empty_when_never_marked():
    p = Portfolio(starting_cash=100.0)

    assert p.equity_series().empty


def test_flatten_orders_negates_all_positions():
    p = Portfolio(starting_cash=1000.0)
    p.apply_fill("AAA", 10.0, 5.0, cost=0.0)
    p.apply_fill("BBB", -4.0, 2.0, cost=0.0)

    orders = p.flatten_orders()

    assert orders == {"AAA": -10.0, "BBB": 4.0}
