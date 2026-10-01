import unittest
from pathlib import Path

from config import DEMO_EXECUTION_ENABLED, TRADING_ENABLED


class RunnerBatTests(unittest.TestCase):
    def test_run_paper_order_gold_eurusd_auto(self):
        text = Path("run_paper.bat").read_text(encoding="utf-8")

        gold = text.index("@('PAPER') $goldLog")
        eurusd = text.index("@('PAPER', 'EURUSD') $eurusdLog")
        auto = text.index("@('MT5', 'DEMO', 'AUTO', 'RUN') $autoLog")

        self.assertLess(gold, eurusd)
        self.assertLess(eurusd, auto)

    def test_run_paper_uses_auto_log_and_not_portfolio(self):
        text = Path("run_paper.bat").read_text(encoding="utf-8")

        self.assertIn("mt5_demo_auto_runner.log", text)
        self.assertNotIn("live_portfolio_runner.log", text)
        self.assertNotIn("@('LIVE', 'GOLD')", text)
        self.assertNotIn("@('LIVE', 'EURUSD')", text)
        self.assertNotIn("@('LIVE', 'PORTFOLIO')", text)

    def test_flags_remain_safe_by_default(self):
        self.assertIs(TRADING_ENABLED, False)
        self.assertIs(DEMO_EXECUTION_ENABLED, False)


if __name__ == "__main__":
    unittest.main()
