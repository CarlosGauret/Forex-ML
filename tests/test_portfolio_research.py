import unittest

import numpy as np
import pandas as pd

from src.demo_portfolio import config_id, portfolio_decision
from src.portfolio_research import (
    currency_exposure_allowed,
    currency_legs,
    passes_confirmation,
    passes_selection,
    server_to_utc,
    simulate_portfolio,
    simulate_trades,
    trend_allows,
)


def _dataset(closes, highs=None, lows=None, atr=1.0, spread=0.0):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "fecha": pd.date_range("2020-01-01", periods=len(closes), freq="h"),
        "Close": closes,
        "High": closes if highs is None else np.asarray(highs, dtype=float),
        "Low": closes if lows is None else np.asarray(lows, dtype=float),
        "ATR": atr,
        "SPREAD": spread,
    })


class ServerTimeTests(unittest.TestCase):
    def test_xm_offset_follows_new_york_dst(self):
        # Julio: GMT+3. Enero: GMT+2.
        utc = server_to_utc(pd.DatetimeIndex(["2025-07-01 12:00", "2025-01-15 12:00"]))
        self.assertEqual(list(utc), [pd.Timestamp("2025-07-01 09:00"), pd.Timestamp("2025-01-15 10:00")])


class SimulateTradesTests(unittest.TestCase):
    def test_long_take_profit_is_two_r(self):
        data = _dataset([100, 100.5, 101], highs=[100, 100.5, 102.5], lows=[100, 99.5, 100.5])
        trades = simulate_trades(data, [True, False, False], "LONG")
        self.assertEqual(list(trades["outcome"]), ["TP"])
        self.assertAlmostEqual(trades["r"].iloc[0], 2.0)

    def test_stop_wins_when_both_hit_same_bar(self):
        data = _dataset([100, 100], highs=[100, 103], lows=[100, 98])
        trades = simulate_trades(data, [True, False], "LONG")
        self.assertEqual(list(trades["outcome"]), ["SL"])
        self.assertAlmostEqual(trades["r"].iloc[0], -1.0)

    def test_spread_cost_reduces_r(self):
        data = _dataset([100, 100], highs=[100, 103], lows=[100, 99.9], spread=0.1)
        trades = simulate_trades(data, [True, False], "LONG")
        # entrada al ask 100.15 (spread x1.5), TP a 102.15 => sigue siendo +2R desde la entrada
        self.assertAlmostEqual(trades["r"].iloc[0], 2.0)
        short = simulate_trades(data, [True, False], "SHORT")
        # SHORT: el ask (high + spread) toca el stop
        self.assertEqual(list(short["outcome"]), ["SL"])

    def test_timeout_exits_at_close_and_blocks_overlap(self):
        closes = [100] + [100.2] * 30
        data = _dataset(closes)
        trades = simulate_trades(data, [True, True] + [False] * 29, "LONG")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades["outcome"].iloc[0], "TIME")
        self.assertAlmostEqual(trades["r"].iloc[0], 0.2)


class CriteriaTests(unittest.TestCase):
    def test_selection_requires_sample_edge_and_consistency(self):
        good = {"trades": 100, "expectancy_r": 0.1, "profit_factor": 1.3, "positive_years": 0.7}
        self.assertTrue(passes_selection(good))
        self.assertFalse(passes_selection(dict(good, trades=10)))
        self.assertFalse(passes_selection(dict(good, expectancy_r=0.01)))
        self.assertFalse(passes_selection(dict(good, positive_years=0.4)))

    def test_confirmation_requires_positive_out_of_sample(self):
        self.assertTrue(passes_confirmation({"trades": 30, "expectancy_r": 0.05}))
        self.assertFalse(passes_confirmation({"trades": 30, "expectancy_r": -0.05}))
        self.assertFalse(passes_confirmation({"trades": 5, "expectancy_r": 0.5}))


class CurrencyExposureTests(unittest.TestCase):
    def test_legs(self):
        self.assertEqual(currency_legs("GBPJPY"), ("GBP", "JPY"))
        self.assertEqual(currency_legs("GOLD"), ("XAU", "USD"))
        self.assertEqual(currency_legs("BTCUSD"), ("BTC", "USD"))

    def test_third_jpy_short_is_blocked(self):
        open_legs = [("EURJPY", "BUY"), ("GBPJPY", "BUY")]
        self.assertFalse(currency_exposure_allowed(open_legs, "AUDJPY", "LONG", 2))
        # Una venta de un cruce JPY reduce la exposicion: permitida
        self.assertTrue(currency_exposure_allowed(open_legs, "AUDJPY", "SHORT", 2))

    def test_portfolio_simulation_respects_limits(self):
        t = pd.Timestamp("2024-01-01")
        trade = lambda r: pd.DataFrame([{"entry_time": t, "exit_time": t + pd.Timedelta(hours=5), "r": r, "outcome": "TP"}])
        sim = simulate_portfolio({("EURJPY", "LONG"): trade(2.0), ("GBPJPY", "LONG"): trade(2.0),
                                  ("AUDJPY", "LONG"): trade(2.0)}, currency_limit=2)
        self.assertEqual(sim["trades"], 2)
        self.assertEqual(sim["skipped"], 1)
        self.assertAlmostEqual(sim["final_equity"], 1.04)


class PortfolioDecisionTests(unittest.TestCase):
    SELECTION = {
        "AUDNZD": {
            "LONG": {"enabled": False, "threshold": 0.60, "trend_filter": False},
            "SHORT": {"enabled": True, "threshold": 0.55, "trend_filter": True},
        },
    }

    def test_disabled_direction_never_trades(self):
        self.assertEqual(portfolio_decision("AUDNZD", 0.95, 0.10, 0, self.SELECTION)[0], "WAIT")

    def test_per_direction_threshold_and_trend_filter(self):
        self.assertEqual(portfolio_decision("AUDNZD", 0.10, 0.56, -1, self.SELECTION), ("SHORT", 0.55))
        # Filtro de tendencia: SHORT solo con tendencia bajista
        self.assertEqual(portfolio_decision("AUDNZD", 0.10, 0.56, 1, self.SELECTION)[0], "WAIT")
        self.assertTrue(trend_allows("SHORT", -1) and not trend_allows("LONG", -1))

    def test_asset_without_rules_waits(self):
        self.assertEqual(portfolio_decision("EURUSD", 0.90, 0.10, 1, self.SELECTION)[0], "WAIT")

    def test_without_selection_falls_back_to_global_threshold(self):
        self.assertEqual(portfolio_decision("USDJPY", 0.70, 0.20, 0, None), ("LONG", 0.60))

    def test_rule_strategy_trades_only_on_signal(self):
        selection = {"EURAUD": {
            "LONG": {"enabled": False},
            "SHORT": {"enabled": True, "strategy": "BREAKOUT_LONDON", "threshold": 0.0, "trend_filter": True},
        }}
        # Una probabilidad ML cualquiera NO basta para una direccion de reglas
        self.assertEqual(portfolio_decision("EURAUD", 0.1, 0.9, -1, selection, {})[0], "WAIT")
        self.assertEqual(portfolio_decision("EURAUD", 0.1, 0.1, -1, selection, {"SHORT": True})[0], "SHORT")
        # El filtro de tendencia tambien aplica a las reglas
        self.assertEqual(portfolio_decision("EURAUD", 0.1, 0.1, 1, selection, {"SHORT": True})[0], "WAIT")

    def test_conflicting_directions_wait(self):
        selection = {"X": {
            "LONG": {"enabled": True, "threshold": 0.55, "trend_filter": False},
            "SHORT": {"enabled": True, "strategy": "BREAKOUT_5D", "threshold": 0.0, "trend_filter": False},
        }}
        self.assertEqual(portfolio_decision("X", 0.9, 0.1, 0, selection, {"SHORT": True})[0], "WAIT")

    def test_config_id_reflects_threshold(self):
        self.assertTrue(config_id("AUDNZD", "SHORT", 0.55).endswith("_T055"))


if __name__ == "__main__":
    unittest.main()
