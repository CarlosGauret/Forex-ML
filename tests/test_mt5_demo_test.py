import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config
from src.execution_demo import DEMO_DISABLED_REASON
from src.mt5_demo_test import (
    DEMO_TEST_COMMENT,
    DEMO_TEST_MAGIC,
    DEMO_TEST_VOLUME,
    close_demo_test,
    open_demo_test,
    preview_demo_test,
)


SERVER_TIME = 1_790_000_000


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_REAL = 2
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2

    def __init__(self, trade_mode=0, check_retcode=10009):
        self.account = SimpleNamespace(
            trade_mode=trade_mode,
            balance=10000.0,
            server="XM Demo",
        )
        self.info = SimpleNamespace(
            name="EURUSD",
            description="Euro vs US Dollar",
            visible=True,
            digits=5,
            point=0.00001,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_contract_size=100000.0,
            trade_stops_level=0,
            filling_mode=2,
        )
        self.tick = SimpleNamespace(bid=1.10000, ask=1.10010, time=SERVER_TIME)
        self.positions = []
        self.calls = []
        self.check_retcode = check_retcode

    def account_info(self):
        return self.account

    def symbols_get(self):
        return [self.info]

    def symbol_info(self, name):
        return self.info if name == self.info.name else None

    def symbol_info_tick(self, name):
        return self.tick

    def symbol_select(self, name, enable):
        return True

    def positions_get(self):
        return self.positions

    def order_calc_profit(self, order_type, symbol, volume, entry, exit_price):
        sign = 1 if order_type == self.ORDER_TYPE_BUY else -1
        return (exit_price - entry) * self.info.trade_contract_size * volume * sign

    def order_check(self, request):
        self.calls.append(("check", dict(request)))
        return SimpleNamespace(retcode=self.check_retcode, comment="check ok")

    def order_send(self, request):
        self.calls.append(("send", dict(request)))
        if "position" in request:
            return SimpleNamespace(retcode=10009, order=request["position"], price=request["price"], profit=1.25)
        return SimpleNamespace(retcode=10009, order=7001, deal=8001, price=request["price"], comment="sent")

    def last_error(self):
        return (0, "fake")


class ConfirmingFakeMT5(FakeMT5):
    def order_send(self, request):
        result = super().order_send(request)
        if "position" not in request:
            self.positions.append(_test_position(ticket=result.order))
        return result


def _test_position(ticket=7001, magic=DEMO_TEST_MAGIC, comment=DEMO_TEST_COMMENT):
    return SimpleNamespace(
        ticket=ticket,
        symbol="EURUSD",
        type=0,
        volume=DEMO_TEST_VOLUME,
        magic=magic,
        comment=comment,
        time=SERVER_TIME - 60,
        price_open=1.10010,
        profit=0.75,
    )


class Mt5DemoTestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_preview_demo_account_allowed_without_sending(self):
        mt5 = FakeMT5()

        result = preview_demo_test(mt5, self.root)

        self.assertEqual(result.status, "PREVIEW")
        self.assertEqual(result.account, "DEMO")
        self.assertEqual(result.orders_sent, 0)
        self.assertEqual([name for name, _ in mt5.calls], ["check"])

    def test_real_account_is_blocked(self):
        mt5 = FakeMT5(trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_REAL)
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = open_demo_test(mt5, self.root)

        self.assertEqual(result.reason, "REAL ACCOUNT BLOCKED")
        self.assertEqual(mt5.calls, [])

    def test_flag_false_blocks_order_send(self):
        mt5 = FakeMT5()

        result = open_demo_test(mt5, self.root)

        self.assertEqual(result.reason, DEMO_DISABLED_REASON)
        self.assertEqual(mt5.calls, [])

    def test_order_check_runs_before_order_send(self):
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = open_demo_test(mt5, self.root)

        self.assertEqual(result.status, "OPENED")
        self.assertEqual([name for name, _ in mt5.calls], ["check", "send"])

    def test_open_without_confirmed_position_does_not_notify(self):
        mt5 = FakeMT5()
        events = []
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = open_demo_test(mt5, self.root, notify=lambda event, msg: events.append((event, msg)))

        self.assertEqual(result.status, "OPENED")
        self.assertEqual(events, [])

    def test_confirmed_open_notifies_mock_without_network(self):
        mt5 = ConfirmingFakeMT5()
        events = []
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = open_demo_test(mt5, self.root, notify=lambda event, msg: events.append((event, msg)))

        self.assertEqual(result.status, "OPENED")
        self.assertEqual(len(events), 1)
        self.assertIn("MT5 DEMO - OPERACION ABIERTA", events[0][1])

    def test_single_test_order_uses_dedicated_magic_and_comment(self):
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            open_demo_test(mt5, self.root)

        request = mt5.calls[1][1]
        self.assertEqual(request["volume"], DEMO_TEST_VOLUME)
        self.assertEqual(request["magic"], DEMO_TEST_MAGIC)
        self.assertEqual(request["comment"], DEMO_TEST_COMMENT)

    def test_existing_test_position_blocks_second_open(self):
        mt5 = FakeMT5()
        mt5.positions = [_test_position()]
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = open_demo_test(mt5, self.root)

        self.assertEqual(result.reason, "DEMO_TEST_POSITION_ALREADY_OPEN")
        self.assertEqual(mt5.calls, [])

    def test_close_only_demo_test_position(self):
        mt5 = FakeMT5()
        mt5.positions = [
            _test_position(ticket=1, magic=999, comment="MANUAL"),
            _test_position(ticket=2),
        ]
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = close_demo_test(mt5, self.root)

        self.assertEqual(result.status, "CLOSED")
        close_request = mt5.calls[1][1]
        self.assertEqual(close_request["position"], 2)
        self.assertEqual(close_request["magic"], DEMO_TEST_MAGIC)
        self.assertEqual(close_request["comment"], DEMO_TEST_COMMENT)

    def test_secrets_are_not_logged(self):
        mt5 = FakeMT5()
        mt5.account.login = 123456
        mt5.account.password = "secret-password"
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            open_demo_test(mt5, self.root)

        log_text = (self.root / "logs" / "mt5_demo_execution.csv").read_text(encoding="utf-8")
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("123456", log_text)

    def test_telegram_failure_does_not_stop_close(self):
        mt5 = FakeMT5()
        mt5.positions = [_test_position()]

        def broken_notify(event_id, message):
            raise RuntimeError("telegram down")

        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = close_demo_test(mt5, self.root, notify=broken_notify)

        self.assertEqual(result.status, "CLOSED")

    def test_trading_enabled_stays_false(self):
        self.assertIs(config.TRADING_ENABLED, False)

    def test_log_has_minimum_columns(self):
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            open_demo_test(mt5, self.root)

        with (self.root / "logs" / "mt5_demo_execution.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, [
                "timestamp", "symbol", "side", "volume", "entry", "sl", "tp",
                "ticket", "retcode", "event", "profit",
            ])


if __name__ == "__main__":
    unittest.main()
