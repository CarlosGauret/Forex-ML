import unittest
from types import SimpleNamespace

from config import DEMO_EXECUTION_ENABLED
from src.execution_demo import (
    DEMO_DISABLED_REASON,
    DEMO_REQUIRED_REASON,
    evaluate_demo_account,
    evaluate_demo_order_gate,
)


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_REAL = 2

    def __init__(self, trade_mode):
        self._account = SimpleNamespace(trade_mode=trade_mode)

    def account_info(self):
        return self._account


class DemoExecutionTests(unittest.TestCase):
    def test_demo_execution_flag_is_false_by_default(self):
        self.assertIs(DEMO_EXECUTION_ENABLED, False)

    def test_real_account_is_blocked(self):
        gate = evaluate_demo_account(FakeMT5(FakeMT5.ACCOUNT_TRADE_MODE_REAL))

        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, DEMO_REQUIRED_REASON)

    def test_demo_account_still_blocked_when_execution_flag_disabled(self):
        gate = evaluate_demo_order_gate(FakeMT5(FakeMT5.ACCOUNT_TRADE_MODE_DEMO))

        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, DEMO_DISABLED_REASON)


if __name__ == "__main__":
    unittest.main()
