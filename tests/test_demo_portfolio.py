import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.demo_portfolio import (
    decide,
    load_local_prices,
    load_portfolio_models,
    model_path,
    run_portfolio_live,
    send_test_order,
)
from src.live_executor import (
    MAGIC_NUMBER,
    STATUS_CLOSED_MANUAL,
    STATUS_DRY_RUN,
    STATUS_SENT,
    close_all_positions,
    notify_closed_deals,
)
from src.risk import BUY, SELL


ROOT = Path(__file__).resolve().parents[1]
SERVER_OFFSET_HOURS = 3

SYMBOL_SPECS = {
    "EURUSD": {"digits": 5, "point": 0.00001, "contract": 100000.0},
    "USDJPY": {"digits": 3, "point": 0.001, "contract": 100000.0},
    "GOLD": {"digits": 2, "point": 0.01, "contract": 100.0},
}


def _mt5_rates(asset, end, bars=1000):
    """Velas del CSV local hasta end, como las entrega MT5: hora servidor + vela en formacion."""
    prices = load_local_prices(ROOT, asset)
    prices = prices[prices.index <= end].iloc[-bars:]
    rates = [
        {
            "time": int((ts + pd.Timedelta(hours=SERVER_OFFSET_HOURS)).timestamp()),
            "open": row.Open, "high": row.High, "low": row.Low, "close": row.Close, "tick_volume": 0,
        }
        for ts, row in prices.iterrows()
    ]
    forming = dict(rates[-1], time=rates[-1]["time"] + 3600)
    return rates + [forming], prices.index.max()


class MultiFakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    TIMEFRAME_H1 = 16385
    DEAL_ENTRY_OUT = 1

    def __init__(self, assets=("EURUSD", "USDJPY", "GOLD")):
        self.account = SimpleNamespace(trade_mode=0, balance=10000.0)
        self.infos, self.rates, self.ticks = {}, {}, {}
        # Misma ultima vela cerrada para todos los activos (como en vivo)
        self.last_closed = min(load_local_prices(ROOT, asset).index.max() for asset in assets)
        for asset in assets:
            spec = SYMBOL_SPECS[asset]
            name = "XAUUSD" if asset == "GOLD" else asset
            self.infos[name] = SimpleNamespace(
                name=name, description=name, visible=True, digits=spec["digits"], point=spec["point"],
                volume_min=0.01, volume_max=100.0, volume_step=0.01,
                trade_contract_size=spec["contract"], trade_tick_size=spec["point"],
                trade_tick_value=spec["point"] * spec["contract"], trade_stops_level=0, filling_mode=2,
            )
            self.rates[name], _ = _mt5_rates(asset, self.last_closed)
            close = self.rates[name][-1]["close"]
            spread = spec["point"] * 10
            self.ticks[name] = SimpleNamespace(bid=close, ask=close + spread, time=self.rates[name][-1]["time"] + 300)
        self.positions, self.deals, self.sent = [], [], []

    def account_info(self):
        return self.account

    def symbols_get(self):
        return list(self.infos.values())

    def symbol_info(self, name):
        return self.infos.get(name)

    def symbol_info_tick(self, name):
        return self.ticks.get(name)

    def symbol_select(self, name, enable):
        return True

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return self.rates.get(symbol, [])[-count:]

    def positions_get(self):
        return self.positions

    def history_deals_get(self, date_from, date_to):
        return self.deals

    def order_calc_profit(self, order_type, symbol, volume, entry, exit_price):
        sign = 1 if order_type == self.ORDER_TYPE_BUY else -1
        profit = (exit_price - entry) * self.infos[symbol].trade_contract_size * volume * sign
        # USDJPY: la ganancia sale en JPY; se pasa a USD con el precio
        return profit / entry if symbol == "USDJPY" else profit

    def order_send(self, request):
        self.sent.append(dict(request))
        return SimpleNamespace(retcode=10009, order=7000 + len(self.sent), price=request["price"], comment="ok")

    def last_error(self):
        return (0, "fake")


class FixedModel:
    classes_ = [0, 1]

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, x):
        return [[1 - self.probability, self.probability]]


def _factory(mt5):
    @contextmanager
    def factory():
        yield mt5

    return factory


def _models(assets, long_prob, short_prob):
    models = {}
    for asset in assets:
        models[(asset, "LONG")] = FixedModel(long_prob)
        models[(asset, "SHORT")] = FixedModel(short_prob)
    return models


class FreshBarTests(unittest.TestCase):
    def setUp(self):
        self.mt5 = MultiFakeMT5(assets=("EURUSD", "USDJPY"))
        self.now = (self.mt5.last_closed + pd.Timedelta(hours=1, minutes=5)).tz_localize("UTC")

    def test_server_offset_comes_from_latest_tick(self):
        from src.demo_portfolio import infer_server_offset

        self.assertEqual(infer_server_offset(self.mt5, ["EURUSD", "USDJPY"], self.now), SERVER_OFFSET_HOURS)

    def test_market_closed_has_no_valid_offset(self):
        from src.demo_portfolio import infer_server_offset

        weekend = self.now + pd.Timedelta(hours=50)
        self.assertIsNone(infer_server_offset(self.mt5, ["EURUSD", "USDJPY"], weekend))

    def test_frozen_symbol_without_current_bar_is_rejected(self):
        from src.demo_portfolio import closed_h1_features

        self.mt5.rates["EURUSD"] = self.mt5.rates["EURUSD"][:-2]  # EURUSD congelado 1 hora

        self.assertIsNone(closed_h1_features(self.mt5, "EURUSD", self.now, SERVER_OFFSET_HOURS))
        self.assertIsNotNone(closed_h1_features(self.mt5, "USDJPY", self.now, SERVER_OFFSET_HOURS))


class DecideTests(unittest.TestCase):
    def test_highest_direction_above_threshold_wins(self):
        self.assertEqual(decide(0.70, 0.40, threshold=0.60), "LONG")
        self.assertEqual(decide(0.40, 0.65, threshold=0.60), "SHORT")
        self.assertEqual(decide(0.59, 0.55, threshold=0.60), "WAIT")


class PortfolioModelsTests(unittest.TestCase):
    def test_all_trained_models_load_with_valid_hash(self):
        models = load_portfolio_models(ROOT)

        self.assertEqual(len(models), 16)

    def test_tampered_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = model_path(ROOT, "EURUSD", "LONG")
            dst = model_path(root, "EURUSD", "LONG")
            dst.parent.mkdir(parents=True)
            dst.write_bytes(src.read_bytes() + b"x")
            dst.with_suffix(".json").write_text(src.with_suffix(".json").read_text())

            with self.assertRaises(RuntimeError):
                load_portfolio_models(root, assets=["EURUSD"])


class PortfolioRunTests(unittest.TestCase):
    ASSETS = ["EURUSD", "USDJPY", "GOLD"]

    def _run(self, mt5, root, models=None, now=None, dry_run=True):
        now = now or (mt5.last_closed + pd.Timedelta(hours=1, minutes=5)).tz_localize("UTC")
        with patch("src.demo_portfolio.DEMO_PORTFOLIO_ASSETS", self.ASSETS):
            return run_portfolio_live(root, dry_run=dry_run, mt5_factory=_factory(mt5), now=now,
                                      models=models or load_portfolio_models(ROOT, self.ASSETS))

    def test_real_models_on_mt5_style_bars_produce_decisions(self):
        mt5 = MultiFakeMT5()
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(mt5, Path(tmp))
            signals = pd.read_csv(Path(tmp) / "live" / "portfolio_signals.csv")

        decisions = {item["asset"]: item for item in result["results"]}
        self.assertEqual(set(decisions), set(self.ASSETS))
        for item in decisions.values():
            self.assertIn(item["decision"], ("LONG", "SHORT", "WAIT"))
            self.assertTrue(0 <= item["prob_long"] <= 1 and 0 <= item["prob_short"] <= 1)
            # La vela evaluada es la ultima cerrada, ya normalizada a UTC
            self.assertEqual(pd.to_datetime(item["fecha"]), mt5.last_closed)
        self.assertEqual(len(signals), 3)
        self.assertEqual(mt5.sent, [])

    def test_buy_and_sell_signals_become_orders(self):
        mt5 = MultiFakeMT5()
        models = _models(["EURUSD", "GOLD"], long_prob=0.70, short_prob=0.30)
        models.update(_models(["USDJPY"], long_prob=0.30, short_prob=0.72))
        with tempfile.TemporaryDirectory() as tmp, patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
            result = self._run(mt5, Path(tmp), models=models, dry_run=False)

        executions = {item["asset"]: item["execution"] for item in result["results"]}
        self.assertEqual(executions["EURUSD"].status, STATUS_SENT)
        self.assertEqual(executions["EURUSD"].direction, BUY)
        self.assertEqual(executions["USDJPY"].status, STATUS_SENT)
        self.assertEqual(executions["USDJPY"].direction, SELL)
        self.assertEqual(executions["GOLD"].symbol, "XAUUSD")
        sell = next(r for r in mt5.sent if r["symbol"] == "USDJPY")
        self.assertEqual(sell["type"], MultiFakeMT5.ORDER_TYPE_SELL)
        self.assertGreater(sell["sl"], sell["price"])
        self.assertLess(sell["tp"], sell["price"])

    def test_below_threshold_waits_without_orders(self):
        mt5 = MultiFakeMT5()
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(mt5, Path(tmp), models=_models(self.ASSETS, 0.52, 0.48))

        self.assertEqual({item["decision"] for item in result["results"]}, {"WAIT"})
        self.assertEqual(mt5.sent, [])

    def test_market_closed_or_stale_bars_do_nothing(self):
        mt5 = MultiFakeMT5()
        late = (mt5.last_closed + pd.Timedelta(hours=6)).tz_localize("UTC")
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(mt5, Path(tmp), models=_models(self.ASSETS, 0.9, 0.1), now=late)

        self.assertEqual({item["decision"] for item in result["results"]}, {"NO_DATA"})

    def test_second_run_same_hour_does_not_duplicate(self):
        mt5 = MultiFakeMT5()
        models = _models(self.ASSETS, 0.70, 0.30)
        with tempfile.TemporaryDirectory() as tmp, patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
            self._run(mt5, Path(tmp), models=models, dry_run=False)
            second = self._run(mt5, Path(tmp), models=models, dry_run=False)
            signals = pd.read_csv(Path(tmp) / "live" / "portfolio_signals.csv")

        self.assertEqual(len(mt5.sent), 3)
        self.assertEqual({item["execution"].reason for item in second["results"]}, {"DUPLICATE_SIGNAL"})
        self.assertEqual(len(signals), 3)

    def test_position_limit_is_respected(self):
        mt5 = MultiFakeMT5()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True), \
                patch("src.demo_portfolio.DEMO_PORTFOLIO_MAX_POSITIONS", 1):
            mt5.positions = [SimpleNamespace(ticket=1, symbol="GBPUSD", magic=MAGIC_NUMBER, comment="x",
                                             profit=0.0, time=0, type=0, volume=0.01)]
            result = self._run(mt5, Path(tmp), models=_models(self.ASSETS, 0.70, 0.30), dry_run=False)

        self.assertEqual({item["execution"].reason for item in result["results"]}, {"MAXIMUM_POSITIONS_REACHED"})
        self.assertEqual(mt5.sent, [])


class ManualCommandsTests(unittest.TestCase):
    def test_test_order_uses_mt5_atr_and_all_gates(self):
        mt5 = MultiFakeMT5(assets=("EURUSD",))
        with tempfile.TemporaryDirectory() as tmp:
            dry = send_test_order(Path(tmp), "EURUSD", "SELL", mt5_factory=_factory(mt5))
            with patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True), \
                    patch("src.demo_portfolio.DEMO_EXECUTION_ENABLED", True):
                sent = send_test_order(Path(tmp), "EURUSD", "SELL", mt5_factory=_factory(mt5))

        self.assertEqual(dry.status, STATUS_DRY_RUN)
        self.assertEqual(sent.status, STATUS_SENT)
        self.assertEqual(sent.direction, SELL)
        self.assertGreater(sent.sl, sent.price)
        self.assertEqual(len(mt5.sent), 1)

    def test_close_all_closes_only_bot_positions(self):
        mt5 = MultiFakeMT5(assets=("EURUSD",))
        mt5.positions = [
            SimpleNamespace(ticket=1, symbol="EURUSD", magic=MAGIC_NUMBER, comment="FXML-a", type=0, volume=0.1),
            SimpleNamespace(ticket=2, symbol="EURUSD", magic=0, comment="manual", type=0, volume=0.1),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
            results = close_all_positions(mt5, Path(tmp), dry_run=False)

        self.assertEqual([r.status for r in results], [STATUS_CLOSED_MANUAL])
        self.assertEqual(mt5.sent[0]["position"], 1)

    def test_closed_deals_are_announced_with_position_event_id(self):
        mt5 = MultiFakeMT5(assets=("EURUSD",))
        mt5.deals = [
            SimpleNamespace(ticket=10, position_id=7001, magic=MAGIC_NUMBER, entry=1, symbol="EURUSD",
                            volume=0.1, price=1.1, reason=5, profit=20.0, commission=0.0, swap=0.0, fee=0.0),
            SimpleNamespace(ticket=11, position_id=7001, magic=MAGIC_NUMBER, entry=0, symbol="EURUSD",
                            volume=0.1, price=1.098, reason=3, profit=0.0, commission=0.0, swap=0.0, fee=0.0),
        ]
        sent = []

        closed = notify_closed_deals(mt5, lambda event_id, message: sent.append((event_id, message)))

        self.assertEqual(len(closed), 1)
        self.assertEqual(sent[0][0], "LIVE_CLOSED|7001")
        self.assertIn("Motivo: TP", sent[0][1])


if __name__ == "__main__":
    unittest.main()
