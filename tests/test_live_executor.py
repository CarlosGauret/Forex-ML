import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.execution_demo import DEMO_DISABLED_REASON, DEMO_REQUIRED_REASON
from src.live_executor import (
    MAGIC_NUMBER,
    STATUS_CLOSED_TIME,
    STATUS_DRY_RUN,
    STATUS_DRY_RUN_CLOSE,
    STATUS_ERROR,
    STATUS_SENT,
    STATUS_SKIP,
    OrderIntent,
    close_expired_positions,
    execute_signal,
    kill_switch_path,
    order_comment,
    read_journal,
    reconcile,
)
from src.live_runner import (
    GOLD_LIVE_CONFIG_ID,
    STALE_REASON,
    fresh_pending_signal,
    run_eurusd_live,
    run_gold_live,
)
from src.risk import BUY, SELL


SERVER_TIME = 1_790_000_000


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_REAL = 2
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    TIMEFRAME_H1 = 16385
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2

    def __init__(self, trade_mode=0, balance=10000.0, bid=1.10000, ask=1.10010, retcodes=None):
        self.account = SimpleNamespace(trade_mode=trade_mode, balance=balance)
        self.tick = SimpleNamespace(bid=bid, ask=ask, time=SERVER_TIME)
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
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            trade_stops_level=0,
            filling_mode=2,
        )
        self.positions = []
        self.deals = []
        self.sent = []
        self.retcodes = list(retcodes or [10009])
        self.rates = _hourly_rates(SERVER_TIME - 100 * 3600, SERVER_TIME)

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

    def history_deals_get(self, date_from, date_to):
        return self.deals

    def order_calc_profit(self, order_type, symbol, volume, entry, exit_price):
        sign = 1 if order_type == self.ORDER_TYPE_BUY else -1
        return (exit_price - entry) * self.info.trade_contract_size * volume * sign

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return self.rates[-count:] if self.rates is not None else None

    def order_send(self, request):
        self.sent.append(dict(request))
        retcode = self.retcodes.pop(0) if len(self.retcodes) > 1 else self.retcodes[0]
        return SimpleNamespace(retcode=retcode, order=5000 + len(self.sent), price=request["price"], comment="fake")

    def last_error(self):
        return (0, "fake")


def _hourly_rates(first_time, server_now):
    """Velas H1 desde first_time hasta la vela en formacion de server_now."""
    start = first_time - first_time % 3600
    return [{"time": t} for t in range(start, server_now - server_now % 3600 + 1, 3600)]


def _intent(signal_id="EURUSD_TEST|2026-09-30T12:00:00", direction=BUY, atr=0.0010):
    return OrderIntent(signal_id=signal_id, config_id="EURUSD_TEST", asset="EURUSD", direction=direction, atr=atr)


def _position(ticket=1, symbol="EURUSD", opened=SERVER_TIME, comment="FXML-x", profit=0.0, magic=MAGIC_NUMBER):
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, time=opened, type=0, volume=0.1, magic=magic,
        comment=comment, profit=profit, price_open=1.1, sl=1.099, tp=1.102,
    )


class LiveExecutorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _enable_demo(self):
        return patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True)

    def test_dry_run_computes_order_without_sending(self):
        mt5 = FakeMT5()

        result = execute_signal(mt5, _intent(), self.root, dry_run=True)

        self.assertEqual(result.status, STATUS_DRY_RUN)
        self.assertEqual(mt5.sent, [])
        self.assertEqual(result.price, 1.10010)
        self.assertAlmostEqual(result.sl, 1.09910)
        self.assertAlmostEqual(result.tp, 1.10210)
        # 1% de 10000 = 100 USD; 10 pips a 10 USD/pip/lote => ~1.0 lote.
        # El redondeo del SL deja la distancia apenas sobre 10 pips: el RiskManager
        # baja un step para no superar nunca el riesgo.
        self.assertIn(result.volume, (0.99, 1.0))
        self.assertEqual(read_journal(self.root)[0]["STATUS"], STATUS_DRY_RUN)

    def test_real_account_is_always_blocked(self):
        for dry_run in (True, False):
            mt5 = FakeMT5(trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_REAL)
            with self._enable_demo():
                result = execute_signal(mt5, _intent(), self.root, dry_run=dry_run)
            self.assertEqual(result.status, STATUS_SKIP)
            self.assertEqual(result.reason, DEMO_REQUIRED_REASON)
            self.assertEqual(mt5.sent, [])

    def test_demo_flag_disabled_blocks_sending(self):
        mt5 = FakeMT5()

        result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.reason, DEMO_DISABLED_REASON)
        self.assertEqual(mt5.sent, [])

    def test_demo_sends_order_with_server_side_sl_tp(self):
        mt5 = FakeMT5()
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.status, STATUS_SENT)
        self.assertEqual(len(mt5.sent), 1)
        request = mt5.sent[0]
        self.assertEqual(request["type"], FakeMT5.ORDER_TYPE_BUY)
        self.assertAlmostEqual(request["sl"], 1.09910)
        self.assertAlmostEqual(request["tp"], 1.10210)
        self.assertEqual(request["magic"], MAGIC_NUMBER)
        self.assertEqual(request["comment"], order_comment(_intent().signal_id))
        self.assertLessEqual(len(request["comment"]), 31)
        self.assertEqual(request["type_filling"], FakeMT5.ORDER_FILLING_IOC)
        self.assertEqual(read_journal(self.root)[0]["STATUS"], STATUS_SENT)

    def test_sell_levels_are_mirrored(self):
        mt5 = FakeMT5()

        result = execute_signal(mt5, _intent(direction=SELL), self.root, dry_run=True)

        self.assertEqual(result.price, 1.10000)
        self.assertAlmostEqual(result.sl, 1.10100)
        self.assertAlmostEqual(result.tp, 1.09800)

    def test_same_signal_is_never_sent_twice(self):
        mt5 = FakeMT5()
        with self._enable_demo():
            first = execute_signal(mt5, _intent(), self.root, dry_run=False)
            second = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(first.status, STATUS_SENT)
        self.assertEqual(second.reason, "DUPLICATE_SIGNAL")
        self.assertEqual(len(mt5.sent), 1)

    def test_duplicate_detected_from_mt5_when_journal_lost(self):
        mt5 = FakeMT5()
        mt5.deals = [SimpleNamespace(magic=MAGIC_NUMBER, comment=order_comment(_intent().signal_id),
                                     time=SERVER_TIME, profit=0.0, commission=0.0, swap=0.0, fee=0.0)]
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.reason, "DUPLICATE_SIGNAL")
        self.assertEqual(mt5.sent, [])

    def test_dry_run_does_not_block_later_real_send(self):
        mt5 = FakeMT5()
        execute_signal(mt5, _intent(), self.root, dry_run=True)
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.status, STATUS_SENT)

    def test_kill_switch_blocks_new_orders(self):
        kill_switch_path(self.root).write_text("stop")
        mt5 = FakeMT5()
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.reason, "KILL_SWITCH_ACTIVE")
        self.assertEqual(mt5.sent, [])

    def test_daily_loss_limit_blocks(self):
        mt5 = FakeMT5()
        mt5.deals = [SimpleNamespace(magic=MAGIC_NUMBER, comment="FXML-old", time=SERVER_TIME - 3600,
                                     profit=-250.0, commission=-5.0, swap=0.0, fee=0.0)]
        mt5.positions = [_position(symbol="GBPUSD", profit=-50.0)]
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.reason, "MAXIMUM_DAILY_LOSS_REACHED")
        self.assertEqual(mt5.sent, [])

    def test_losses_older_than_24h_do_not_count(self):
        mt5 = FakeMT5()
        mt5.deals = [SimpleNamespace(magic=MAGIC_NUMBER, comment="FXML-old", time=SERVER_TIME - 90000,
                                     profit=-500.0, commission=0.0, swap=0.0, fee=0.0)]
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.status, STATUS_SENT)

    def test_symbol_already_open_blocks(self):
        mt5 = FakeMT5()
        mt5.positions = [_position()]
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.reason, "SYMBOL_POSITION_ALREADY_OPEN")

    def test_manual_positions_are_ignored(self):
        mt5 = FakeMT5()
        mt5.positions = [_position(magic=0)]
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.status, STATUS_SENT)

    def test_wide_spread_blocks(self):
        mt5 = FakeMT5(bid=1.10000, ask=1.10040)

        result = execute_signal(mt5, _intent(), self.root, dry_run=True)

        self.assertEqual(result.reason, "SPREAD_TOO_WIDE")

    def test_minimum_volume_exceeding_risk_skips(self):
        mt5 = FakeMT5(balance=50.0)

        result = execute_signal(mt5, _intent(), self.root, dry_run=True)

        self.assertEqual(result.reason, "MINIMUM_VOLUME_EXCEEDS_RISK")

    def test_requote_is_retried(self):
        mt5 = FakeMT5(retcodes=[10004, 10009])
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False, sleep=lambda _: None)

        self.assertEqual(result.status, STATUS_SENT)
        self.assertEqual(len(mt5.sent), 2)

    def test_rejected_order_reports_error(self):
        mt5 = FakeMT5(retcodes=[10019])  # NO_MONEY
        notified = []
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False,
                                    notify=lambda event_id, msg: notified.append(event_id))

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(len(mt5.sent), 1)
        self.assertEqual(len(notified), 1)

    def test_algo_trading_disabled_has_clear_message(self):
        mt5 = FakeMT5(retcodes=[10027])
        with self._enable_demo():
            result = execute_signal(mt5, _intent(), self.root, dry_run=False)

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertIn("Algo Trading", result.reason)


class ClosePositionsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_expired_position_is_closed_and_fresh_one_kept(self):
        mt5 = FakeMT5()
        mt5.positions = [
            # 25h despues de la entrada: 24 velas cerradas desde la vela de entrada
            _position(ticket=1, opened=SERVER_TIME - 25 * 3600),
            _position(ticket=2, opened=SERVER_TIME - 3600),
        ]
        with patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
            results = close_expired_positions(mt5, self.root, {"EURUSD"}, max_hold_bars=24, dry_run=False)

        self.assertEqual([r.status for r in results], [STATUS_CLOSED_TIME])
        self.assertEqual(mt5.sent[0]["position"], 1)
        self.assertEqual(mt5.sent[0]["type"], FakeMT5.ORDER_TYPE_SELL)
        self.assertEqual(mt5.sent[0]["price"], 1.10000)

    def test_dry_run_close_does_not_send(self):
        mt5 = FakeMT5()
        mt5.positions = [_position(opened=SERVER_TIME - 30 * 3600)]

        results = close_expired_positions(mt5, self.root, {"EURUSD"}, max_hold_bars=24, dry_run=True)

        self.assertEqual(results[0].status, STATUS_DRY_RUN_CLOSE)
        self.assertEqual(mt5.sent, [])

    def test_close_works_with_kill_switch_active(self):
        kill_switch_path(self.root).write_text("stop")
        mt5 = FakeMT5()
        mt5.positions = [_position(opened=SERVER_TIME - 30 * 3600)]
        with patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
            results = close_expired_positions(mt5, self.root, {"EURUSD"}, max_hold_bars=24, dry_run=False)

        self.assertEqual(results[0].status, STATUS_CLOSED_TIME)


    def test_weekend_hours_do_not_count_as_bars(self):
        current_bar = SERVER_TIME - SERVER_TIME % 3600
        entry_bar = current_bar - 60 * 3600
        mt5 = FakeMT5()
        # 3 velas el viernes despues de la entrada, 48h sin mercado, 5 velas cerradas el lunes
        mt5.rates = (
            [{"time": entry_bar + h * 3600} for h in range(0, 4)]
            + [{"time": current_bar - h * 3600} for h in range(5, -1, -1)]
        )
        mt5.positions = [_position(opened=entry_bar + 300)]
        with patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
            results = close_expired_positions(mt5, self.root, {"EURUSD"}, max_hold_bars=24, dry_run=False)

        self.assertEqual(results, [])
        self.assertEqual(mt5.sent, [])

    def test_missing_rates_never_force_a_close(self):
        mt5 = FakeMT5()
        mt5.rates = None
        mt5.positions = [_position(opened=SERVER_TIME - 30 * 3600)]
        with patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
            results = close_expired_positions(mt5, self.root, {"EURUSD"}, max_hold_bars=24, dry_run=False)

        self.assertEqual(results, [])


class ReconcileTests(unittest.TestCase):
    def test_detects_missing_and_orphan_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mt5 = FakeMT5()
            with patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
                execute_signal(mt5, _intent("A"), root, dry_run=False)
            # A no aparece en MT5 y hay una posicion del bot que el journal no conoce
            mt5.positions = [_position(ticket=999, comment="FXML-desconocido")]

            report = reconcile(mt5, root)

        self.assertFalse(report["ok"])
        self.assertEqual(len(report["missing_in_mt5"]), 1)
        self.assertEqual(len(report["orphans_in_mt5"]), 1)

    def test_closed_position_found_in_deals_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mt5 = FakeMT5()
            with patch("src.execution_demo.DEMO_EXECUTION_ENABLED", True):
                result = execute_signal(mt5, _intent("A"), root, dry_run=False)
            mt5.deals = [SimpleNamespace(magic=MAGIC_NUMBER, position_id=result.ticket, comment="x",
                                         time=SERVER_TIME, profit=0.0)]

            report = reconcile(mt5, root)

        self.assertTrue(report["ok"])


class LiveRunnerTests(unittest.TestCase):
    PENDING = {"SIGNAL_ID": "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065|2026-09-30T12:00:00",
               "FECHA_SIGNAL": "2026-09-30T12:00:00", "PROBABILIDAD": 0.7, "ATR": 0.0010}

    def test_only_signal_from_last_closed_bar_is_fresh(self):
        fresh = {"pending_signal": self.PENDING, "last_closed_bar": "2026-09-30T12:00:00"}
        stale = {"pending_signal": self.PENDING, "last_closed_bar": "2026-09-30T15:00:00"}

        self.assertEqual(fresh_pending_signal(fresh), self.PENDING)
        self.assertIsNone(fresh_pending_signal(stale))
        self.assertIsNone(fresh_pending_signal({"last_closed_bar": "2026-09-30T12:00:00"}))

    def test_run_executes_fresh_paper_signal_in_dry_run_by_default(self):
        from contextlib import contextmanager

        mt5 = FakeMT5()

        @contextmanager
        def factory():
            yield mt5

        state = {"pending_signal": self.PENDING, "last_closed_bar": "2026-09-30T12:00:00"}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("src.eurusd_forward.process_eurusd_forward", return_value={"processed": 1}), \
                patch("src.eurusd_forward._read_json", return_value=state):
            result = run_eurusd_live(Path(tmp), mt5_factory=factory)

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["execution"].status, STATUS_DRY_RUN)
        self.assertEqual(result["execution"].direction, BUY)
        self.assertEqual(mt5.sent, [])


def _gold_mt5():
    mt5 = FakeMT5(bid=2650.00, ask=2650.30)
    mt5.info.name = "XAUUSD"
    mt5.info.description = "Gold vs US Dollar"
    mt5.info.digits = 2
    mt5.info.point = 0.01
    mt5.info.trade_contract_size = 100.0
    mt5.info.trade_tick_size = 0.01
    mt5.info.trade_tick_value = 1.0
    return mt5


def _factory(mt5):
    from contextlib import contextmanager

    @contextmanager
    def factory():
        yield mt5

    return factory


class GoldRunnerTests(unittest.TestCase):
    NOW = "2026-09-30T13:05:00+00:00"

    def _state(self, fecha_signal):
        return {"pending_signals": [
            {"CONFIG_ID": "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T055", "FECHA_SIGNAL": fecha_signal,
             "ATR_SIGNAL": 99.0, "PROBABILIDAD": 0.57},
            {"CONFIG_ID": GOLD_LIVE_CONFIG_ID, "FECHA_SIGNAL": fecha_signal,
             "ATR_SIGNAL": 10.0, "PROBABILIDAD": 0.63},
        ]}

    def _run(self, root, mt5, state, paper_side_effect=None):
        with patch("src.paper_trader.ejecutar_paper", side_effect=paper_side_effect, return_value={}), \
                patch("src.paper_trader._leer_json", return_value=state):
            return run_gold_live(root, mt5_factory=_factory(mt5), now=self.NOW)

    def test_fresh_t060_signal_executes_on_mt5_gold(self):
        mt5 = _gold_mt5()
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(Path(tmp), mt5, self._state("2026-09-30T12:00:00"))

        execution = result["execution"]
        self.assertTrue(result["fresh"])
        self.assertEqual(execution.status, STATUS_DRY_RUN)
        self.assertEqual(execution.symbol, "XAUUSD")
        self.assertEqual(execution.config_id, GOLD_LIVE_CONFIG_ID)
        # ATR de T060 (10), no el de T055: SL 1 ATR, TP 2 ATR sobre el ask
        self.assertAlmostEqual(execution.sl, 2640.30)
        self.assertAlmostEqual(execution.tp, 2670.30)
        self.assertIn(execution.volume, (0.09, 0.1))

    def test_stale_signal_is_logged_once_and_not_executed(self):
        mt5 = _gold_mt5()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._run(root, mt5, self._state("2026-09-30T10:00:00"))
            second = self._run(root, mt5, self._state("2026-09-30T10:00:00"))
            journal = read_journal(root)

        self.assertFalse(first["fresh"])
        self.assertEqual(first["execution"].reason, STALE_REASON)
        self.assertIsNone(second["execution"])
        self.assertEqual([row["REASON"] for row in journal], [STALE_REASON])
        self.assertEqual(mt5.sent, [])

    def test_paper_failure_still_runs_time_exits(self):
        mt5 = _gold_mt5()
        mt5.positions = [_position(symbol="XAUUSD", opened=SERVER_TIME - 30 * 3600)]
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(Path(tmp), mt5, {}, paper_side_effect=RuntimeError("yfinance caido"))

        self.assertEqual(result["paper_error"], "yfinance caido")
        self.assertIsNone(result["execution"])
        self.assertEqual([c.status for c in result["closed"]], [STATUS_DRY_RUN_CLOSE])


if __name__ == "__main__":
    unittest.main()
