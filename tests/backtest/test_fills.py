from backtest.fills import compute_fill


def test_compute_fill_returns_next_open_price_unchanged():
    price, cost = compute_fill(10.0, 100.0, cost_bps=5.0)

    assert price == 100.0


def test_compute_fill_cost_scales_with_notional_and_bps():
    price, cost = compute_fill(10.0, 100.0, cost_bps=5.0)

    assert cost == 10.0 * 100.0 * 5.0 / 10_000


def test_compute_fill_cost_is_direction_independent():
    _, buy_cost = compute_fill(10.0, 100.0, cost_bps=5.0)
    _, sell_cost = compute_fill(-10.0, 100.0, cost_bps=5.0)

    assert buy_cost == sell_cost


def test_compute_fill_zero_order_is_free():
    price, cost = compute_fill(0.0, 100.0, cost_bps=5.0)

    assert price == 100.0
    assert cost == 0.0
