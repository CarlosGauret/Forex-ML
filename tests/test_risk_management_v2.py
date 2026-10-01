import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.exit_modes import (
    EXIT_BREAK_EVEN_TP,
    EXIT_FIXED_TP,
    EXIT_TRAILING,
    simulate_exit_mode,
)
from src.live_executor import (
    MAGIC_NUMBER,
    STATUS_DRY_RUN_SL_UPDATE,
    manage_protective_exits,
    read_journal,
)
from src.risk import BUY, SELL, RiskLimits, RiskManager
from src.risk_state import load_or_roll_daily_risk_state


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


class RiskCapitalizationTests(unittest.TestCase):
    def test_position_size_uses_equity_when_available(self):
        manager = RiskManager(RiskLimits(risk_per_trade=0.01))

        low = manager.calculate_position_size(
            symbol="GOLD",
            direction=BUY,
            entry=100.0,
            stop_loss=90.0,
            balance=1000.0,
            equity=1100.0,
            symbol_info=SYMBOL_INFO,
            order_calc_profit=_order_calc_profit,
        )
        high = manager.calculate_position_size(
            symbol="GOLD",
            direction=BUY,
            entry=100.0,
            stop_loss=90.0,
            balance=1000.0,
            equity=2100.0,
            symbol_info=SYMBOL_INFO,
            order_calc_profit=_order_calc_profit,
        )

        self.assertEqual(low.risk_amount, 11.0)
        self.assertEqual(high.risk_amount, 21.0)
        self.assertLess(low.volume, high.volume)

    def test_daily_reference_equity_persists_and_rolls_by_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day1 = pd.Timestamp("2026-10-01T12:00:00Z")
            first = load_or_roll_daily_risk_state(root, equity=1000, now=day1)
            second = load_or_roll_daily_risk_state(root, equity=800, now=day1)
            rolled = load_or_roll_daily_risk_state(root, equity=900, now=day1 + pd.Timedelta(days=1))

        self.assertEqual(first.starting_equity, 1000)
        self.assertEqual(second.starting_equity, 1000)
        self.assertEqual(rolled.starting_equity, 900)

    def test_total_open_risk_limit_blocks_new_trade(self):
        manager = RiskManager(RiskLimits(maximum_total_open_risk=0.03))
        gate = SimpleNamespace(allowed=True, reason="OK")

        result = manager.evaluate_trade_permission(
            balance=1000,
            equity=1000,
            open_risk_amount=25,
            new_trade_risk_amount=10,
            safety_gate=gate,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "MAX_TOTAL_OPEN_RISK")


class ExitModeTests(unittest.TestCase):
    def test_fixed_tp_closes_at_take_profit(self):
        entry = {"direction": BUY, "entry": 100.0, "sl": 99.0, "tp": 102.0}
        bars = pd.DataFrame([{"High": 102.1, "Low": 100.2, "Close": 102.0}])

        result = simulate_exit_mode(entry, bars, EXIT_FIXED_TP)

        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(result["profit_r"], 2.0)

    def test_break_even_tp_moves_stop_to_entry(self):
        entry = {"direction": BUY, "entry": 100.0, "sl": 99.0, "tp": 103.0}
        bars = pd.DataFrame([
            {"High": 101.2, "Low": 100.4, "Close": 101.0},
            {"High": 100.5, "Low": 99.9, "Close": 100.0},
        ])

        result = simulate_exit_mode(entry, bars, EXIT_BREAK_EVEN_TP)

        self.assertEqual(result["exit_reason"], "BREAK_EVEN")
        self.assertEqual(result["profit_r"], 0.0)

    def test_trailing_sell_stop_only_moves_favorably(self):
        entry = {"direction": SELL, "entry": 100.0, "sl": 101.0, "tp": 98.0}
        bars = pd.DataFrame([
            {"High": 99.6, "Low": 97.8, "Close": 98.0},
            {"High": 98.9, "Low": 97.9, "Close": 98.5},
        ])

        result = simulate_exit_mode(entry, bars, EXIT_TRAILING)

        self.assertEqual(result["exit_reason"], "TRAILING_STOP")
        self.assertGreater(result["profit_r"], 0)


class FakeMT5ForExit:
    ACCOUNT_TRADE_MODE_DEMO = 0
    POSITION_TYPE_BUY = 0
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(self):
        self.tick = SimpleNamespace(bid=101.2, ask=101.3)
        self.info = SimpleNamespace(digits=2, point=0.01)
        self.positions = [
            SimpleNamespace(
                ticket=10,
                symbol="EURUSD",
                time=0,
                type=0,
                volume=0.1,
                magic=MAGIC_NUMBER,
                comment="FXML-x",
                profit=0.0,
                price_open=100.0,
                sl=99.0,
                tp=102.0,
            )
        ]

    def account_info(self):
        return SimpleNamespace(trade_mode=0)

    def positions_get(self):
        return self.positions

    def symbol_info_tick(self, symbol):
        return self.tick

    def symbol_info(self, symbol):
        return self.info


class ProtectiveExitTests(unittest.TestCase):
    def test_break_even_update_is_journaled_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mt5 = FakeMT5ForExit()
            results = manage_protective_exits(mt5, root, exit_mode=EXIT_BREAK_EVEN_TP, dry_run=True)
            journal = read_journal(root)

        self.assertEqual(results[0].status, STATUS_DRY_RUN_SL_UPDATE)
        self.assertEqual(results[0].reason, "BREAK_EVEN")
        self.assertEqual(results[0].sl, 100.0)
        self.assertEqual(journal[0]["REASON"], "BREAK_EVEN")


if __name__ == "__main__":
    unittest.main()
