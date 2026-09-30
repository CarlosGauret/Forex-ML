import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.risk import BUY, DECISION_SKIP, RiskLimits, RiskManager, normalize_volume_down


SYMBOL_INFO = {
    "volume_min": 0.01,
    "volume_max": 10.0,
    "volume_step": 0.01,
    "trade_tick_size": 0.01,
    "trade_tick_value": 1.0,
    "trade_contract_size": 100.0,
    "point": 0.01,
    "digits": 2,
}


def _order_calc_profit(order_type, symbol, volume, entry, stop_loss):
    return -(abs(entry - stop_loss) / 0.01) * 1.0 * volume


class RiskManagerTests(unittest.TestCase):
    def test_position_sizing_uses_mt5_order_calc_profit(self):
        manager = RiskManager(RiskLimits(risk_per_trade=0.01))

        result = manager.calculate_position_size(
            symbol="GOLD",
            direction=BUY,
            entry=100.0,
            stop_loss=90.0,
            balance=10000.0,
            symbol_info=SYMBOL_INFO,
            order_calc_profit=_order_calc_profit,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.volume, 0.1)
        self.assertEqual(result.estimated_loss, 100.0)
        self.assertEqual(result.loss_source, "MT5_ORDER_CALC_PROFIT")

    def test_volume_step_rounds_down(self):
        manager = RiskManager(RiskLimits(risk_per_trade=0.0105))

        result = manager.calculate_position_size(
            symbol="GOLD",
            direction=BUY,
            entry=100.0,
            stop_loss=90.0,
            balance=10000.0,
            symbol_info=SYMBOL_INFO,
            order_calc_profit=_order_calc_profit,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.theoretical_volume, 0.105)
        self.assertEqual(result.volume, 0.1)

    def test_minimum_volume_exceeds_risk_returns_skip(self):
        manager = RiskManager(RiskLimits(risk_per_trade=0.0001))

        result = manager.calculate_position_size(
            symbol="GOLD",
            direction=BUY,
            entry=100.0,
            stop_loss=90.0,
            balance=1000.0,
            symbol_info=SYMBOL_INFO,
            order_calc_profit=_order_calc_profit,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.decision, DECISION_SKIP)
        self.assertEqual(result.reason, "MINIMUM_VOLUME_EXCEEDS_RISK")
        self.assertIsNone(result.volume)

    def test_tick_value_fallback_without_order_calc_profit(self):
        manager = RiskManager(RiskLimits(risk_per_trade=0.01))

        result = manager.calculate_position_size(
            symbol="GOLD",
            direction=BUY,
            entry=100.0,
            stop_loss=90.0,
            balance=10000.0,
            symbol_info=SYMBOL_INFO,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.volume, 0.1)
        self.assertEqual(result.loss_source, "SYMBOL_TICK_VALUE")

    def test_kill_switch_blocks_trade_permission(self):
        manager = RiskManager(RiskLimits(kill_switch=True))

        result = manager.evaluate_trade_permission(balance=10000.0)

        self.assertFalse(result.allowed)
        self.assertEqual(result.decision, DECISION_SKIP)
        self.assertIn(result.reason, ["TRADING_DISABLED_FAIL_CLOSED", "KILL_SWITCH_ACTIVE"])

    def test_global_trading_disabled_blocks_before_risk_limits(self):
        manager = RiskManager()

        result = manager.evaluate_trade_permission(balance=10000.0)

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "TRADING_DISABLED_FAIL_CLOSED")

    def test_symbol_position_already_open_blocks_when_global_gate_is_enabled(self):
        manager = RiskManager()

        with patch(
            "src.risk.evaluate_trade_gate",
            return_value=SimpleNamespace(allowed=True, reason="OK"),
        ):
            result = manager.evaluate_trade_permission(
                balance=10000.0,
                symbol="EURUSD",
                open_symbols=["eurusd"],
            )

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "SYMBOL_POSITION_ALREADY_OPEN")

    def test_normalize_volume_down(self):
        self.assertEqual(normalize_volume_down(0.109, 0.01, 10.0), 0.10)
        self.assertEqual(normalize_volume_down(1.234, 0.1, 10.0), 1.2)


if __name__ == "__main__":
    unittest.main()
