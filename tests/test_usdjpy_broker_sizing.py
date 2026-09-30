import unittest

from src.risk import BUY, SELL, DECISION_EXECUTABLE, DECISION_SKIP, RiskLimits, RiskManager


USDJPY_INFO = {
    "volume_min": 0.01,
    "volume_max": 100.0,
    "volume_step": 0.01,
    "trade_contract_size": 100000.0,
    "trade_tick_size": 0.001,
    "trade_tick_value": 0.636,
    "digits": 3,
    "point": 0.001,
}


def usdjpy_order_calc_profit(order_type, symbol, volume, entry, stop_loss):
    del symbol
    multiplier = 1 if order_type in (BUY, 0) else -1
    profit_jpy = (stop_loss - entry) * multiplier * 100000.0 * volume
    return profit_jpy / stop_loss


class UsdJpyBrokerSizingTests(unittest.TestCase):
    def setUp(self):
        self.manager = RiskManager(RiskLimits(risk_per_trade=0.01))

    def _size(self, direction, balance):
        entry = 153.300995 if direction == BUY else 153.004707
        stop = 153.004707 if direction == BUY else 153.300995
        return self.manager.calculate_position_size(
            symbol="USDJPY",
            direction=direction,
            entry=entry,
            stop_loss=stop,
            balance=balance,
            symbol_info=USDJPY_INFO,
            order_calc_profit=usdjpy_order_calc_profit,
            order_type=direction,
        )

    def test_usdjpy_buy_risk_uses_order_calc_profit(self):
        result = self._size(BUY, 500)

        self.assertTrue(result.ok)
        self.assertEqual(result.decision, DECISION_EXECUTABLE)
        self.assertEqual(result.volume, 0.02)
        self.assertAlmostEqual(result.min_volume_loss, 1.936463, places=5)
        self.assertEqual(result.loss_source, "MT5_ORDER_CALC_PROFIT")

    def test_usdjpy_buy_accepts_mt5_numeric_order_type_zero(self):
        result = self.manager.calculate_position_size(
            symbol="USDJPY",
            direction=BUY,
            entry=153.300995,
            stop_loss=153.004707,
            balance=500,
            symbol_info=USDJPY_INFO,
            order_calc_profit=usdjpy_order_calc_profit,
            order_type=0,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.loss_source, "MT5_ORDER_CALC_PROFIT")
        self.assertAlmostEqual(result.min_volume_loss, 1.936463, places=5)

    def test_usdjpy_sell_risk_uses_order_calc_profit(self):
        result = self._size(SELL, 500)

        self.assertTrue(result.ok)
        self.assertEqual(result.volume, 0.02)
        self.assertAlmostEqual(result.min_volume_loss, 1.933, places=3)
        self.assertEqual(result.loss_source, "MT5_ORDER_CALC_PROFIT")

    def test_capital_100_does_not_round_up_to_volume_min(self):
        result = self._size(BUY, 100)

        self.assertFalse(result.ok)
        self.assertEqual(result.decision, DECISION_SKIP)
        self.assertEqual(result.reason, "MINIMUM_VOLUME_EXCEEDS_RISK")
        self.assertLess(result.theoretical_volume, USDJPY_INFO["volume_min"])
        self.assertIsNone(result.volume)

    def test_capitals_500_1000_5000_are_executable_with_step_round_down(self):
        expected = {
            500: 0.02,
            1000: 0.05,
            5000: 0.25,
        }

        for balance, volume in expected.items():
            with self.subTest(balance=balance):
                result = self._size(BUY, balance)
                self.assertTrue(result.ok)
                self.assertEqual(result.volume, volume)
                self.assertLessEqual(result.estimated_loss, result.risk_amount)

    def test_volume_step_rounds_down_not_up(self):
        result = self._size(BUY, 500)

        self.assertGreater(result.theoretical_volume, 0.02)
        self.assertLess(result.theoretical_volume, 0.03)
        self.assertEqual(result.volume, 0.02)

    def test_tick_value_alias_fallback_uses_broker_aware_keys(self):
        alias_info = {
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "contract_size": 100000.0,
            "tick_size": 0.001,
            "tick_value": 0.636,
        }

        result = self.manager.calculate_position_size(
            symbol="USDJPY",
            direction=BUY,
            entry=153.300995,
            stop_loss=153.004707,
            balance=500,
            symbol_info=alias_info,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.loss_source, "SYMBOL_TICK_VALUE")
        self.assertEqual(result.volume, 0.02)


if __name__ == "__main__":
    unittest.main()
