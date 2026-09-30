import unittest

from config import TRADING_ENABLED
from src.safety import evaluate_trade_gate, trading_enabled


class SafetyGateTests(unittest.TestCase):
    def test_trading_enabled_is_false_by_default(self):
        self.assertIs(TRADING_ENABLED, False)
        self.assertFalse(trading_enabled())

    def test_trade_gate_blocks_when_trading_disabled(self):
        result = evaluate_trade_gate("ORDER_SEND")

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "TRADING_DISABLED_FAIL_CLOSED")
        self.assertFalse(result.trading_enabled)


if __name__ == "__main__":
    unittest.main()
