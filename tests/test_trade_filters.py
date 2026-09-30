import unittest
from types import SimpleNamespace

from src.trade_filters import (
    DECISION_SKIP_BROKER,
    DECISION_SKIP_RISK,
    DECISION_WAIT,
    evaluate_trade_filters,
    final_decision_from_signal,
)


def info(**overrides):
    data = {
        "point": 0.0001,
        "digits": 5,
        "volume_min": 0.01,
        "volume_max": 10.0,
        "volume_step": 0.01,
        "trade_stops_level": 20,
        "trade_freeze_level": 10,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def tick(bid=1.1000, ask=1.1002):
    return SimpleNamespace(bid=bid, ask=ask)


class TradeFilterTests(unittest.TestCase):
    def test_spread_filter(self):
        result = evaluate_trade_filters(
            info(),
            tick(1.1000, 1.1010),
            max_spread_points=5,
            model_validated=True,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.decision, DECISION_WAIT)
        self.assertIn("SPREAD_EXCESIVO", result.reason)

    def test_stops_level_violation(self):
        result = evaluate_trade_filters(
            info(trade_stops_level=20, trade_freeze_level=0),
            tick(),
            volume=0.01,
            entry=1.1002,
            stop_loss=1.0990,
            take_profit=1.1030,
            model_validated=True,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.decision, DECISION_SKIP_BROKER)
        self.assertIn("STOPS_LEVEL_VIOLATION", result.reason)

    def test_freeze_level_violation(self):
        result = evaluate_trade_filters(
            info(trade_stops_level=0, trade_freeze_level=20),
            tick(1.1000, 1.1002),
            volume=0.01,
            entry=1.1002,
            stop_loss=1.1001,
            take_profit=1.1030,
            model_validated=True,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.decision, DECISION_SKIP_BROKER)
        self.assertIn("FREEZE_LEVEL_VIOLATION", result.reason)

    def test_volume_min_max_step(self):
        below = evaluate_trade_filters(info(), tick(), volume=0.001, model_validated=True)
        above = evaluate_trade_filters(info(), tick(), volume=11.0, model_validated=True)
        bad_step = evaluate_trade_filters(info(), tick(), volume=0.015, model_validated=True)

        self.assertIn("VOLUME_BELOW_MIN", below.reason)
        self.assertIn("VOLUME_ABOVE_MAX", above.reason)
        self.assertIn("VOLUME_STEP_INVALID", bad_step.reason)

    def test_market_closed_or_invalid_bid_ask(self):
        result = evaluate_trade_filters(info(), tick(0, 0), model_validated=True)

        self.assertFalse(result.passed)
        self.assertEqual(result.decision, DECISION_WAIT)
        self.assertIn("MERCADO_CERRADO_O_PRECIO_INVALIDO", result.reason)

    def test_model_not_validated_waits(self):
        result = evaluate_trade_filters(info(), tick(), model_validated=False)

        self.assertFalse(result.passed)
        self.assertEqual(result.decision, DECISION_WAIT)
        self.assertIn("MODELO_VALIDADO_NO", result.reason)
        self.assertEqual(final_decision_from_signal("LONG", result), DECISION_WAIT)

    def test_risk_limits(self):
        result = evaluate_trade_filters(info(), tick(), daily_loss_ok=False)

        self.assertFalse(result.passed)
        self.assertEqual(result.decision, DECISION_SKIP_RISK)
        self.assertIn("MAXIMUM_DAILY_LOSS_REACHED", result.reason)


if __name__ == "__main__":
    unittest.main()
