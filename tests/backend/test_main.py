import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_significant_pairs(client, monkeypatch):
    fake_table = pd.DataFrame(
        [{"sector": "Financials", "ticker_y": "BAC", "ticker_x": "PNC", "p_value_fdr": 0.015}]
    )
    monkeypatch.setattr(main, "get_significant_pairs", lambda start, end: fake_table)

    response = client.get("/api/pairs/significant")

    assert response.status_code == 200
    body = response.json()
    assert body["pairs"] == [
        {"sector": "Financials", "ticker_y": "BAC", "ticker_x": "PNC", "p_value_fdr": 0.015}
    ]


def _fake_monitor_data():
    dates = pd.bdate_range("2020-01-01", periods=5)
    return {
        "status": pd.Series(["ACTIVE", "ACTIVE", "HALTED", "HALTED", "ACTIVE"], index=dates),
        "zscore": pd.Series([0.5, 1.0, 2.5, 2.6, 0.2], index=dates),
        "rolling_pvalue": pd.Series([0.01, 0.02, 0.2, 0.2, 0.01], index=dates),
    }


def test_pair_status(client, monkeypatch):
    monkeypatch.setattr(
        main, "get_pair_monitor_data", lambda ticker_y, ticker_x, start, end: _fake_monitor_data()
    )

    response = client.get("/api/pairs/BAC/PNC/status")

    assert response.status_code == 200
    body = response.json()
    assert body["pair"] == "BAC/PNC"
    assert body["status"] == "ACTIVE"
    assert body["current_zscore"] == pytest.approx(0.2)
    assert body["days_since_last_break"] is not None


def test_pair_status_returns_400_on_value_error(client, monkeypatch):
    def _raise(ticker_y, ticker_x, start, end):
        raise ValueError("not enough observations")

    monkeypatch.setattr(main, "get_pair_monitor_data", _raise)

    response = client.get("/api/pairs/AAA/BBB/status")

    assert response.status_code == 400
    assert "not enough observations" in response.json()["detail"]


def test_pair_alerts(client, monkeypatch):
    monkeypatch.setattr(
        main, "get_pair_monitor_data", lambda ticker_y, ticker_x, start, end: _fake_monitor_data()
    )

    response = client.get("/api/pairs/BAC/PNC/alerts")

    assert response.status_code == 200
    body = response.json()
    assert body["pair"] == "BAC/PNC"
    assert len(body["halt_events"]) == 1


def test_comparison_endpoint_returns_committed_snapshot(client):
    response = client.get("/api/comparison")

    assert response.status_code == 200
    body = response.json()
    assert "baseline" in body
    assert "static_hedge_ratio" in body


def test_comparison_endpoint_404_when_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "COMPARISON_RESULTS_PATH", tmp_path / "missing.json")

    response = client.get("/api/comparison")

    assert response.status_code == 404
