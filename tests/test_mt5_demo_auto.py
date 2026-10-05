import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config
from src.mt5_demo_auto import (
    AUTHORIZED_EURUSD_CONFIG,
    COMMENT_BY_ASSET,
    EURUSD_EXPECTED_HASH,
    MAGIC_BY_ASSET,
    close_due_positions,
    run_auto,
    sync_broker_closes,
    build_decisions,
)


NOW = datetime(2026, 10, 1, 11, 5, tzinfo=timezone.utc)
FRESH_BAR = "2026-10-01T10:00:00"
OLD_BAR = "2026-10-01T08:00:00"


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_REAL = 2
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_RETURN = 2
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_FOK = 0
    TIMEFRAME_H1 = 16385

    def __init__(self, trade_mode=0, balance=10000.0, server="XM Demo"):
        self.account = SimpleNamespace(trade_mode=trade_mode, balance=balance, equity=balance, server=server)
        self.calls = []
        self.positions = []
        self.deals = []
        self.info = {
            "EURUSD": SimpleNamespace(
                name="EURUSD", description="Euro vs Dollar", visible=True, digits=5, point=0.00001,
                volume_min=0.01, volume_max=100.0, volume_step=0.01, trade_contract_size=100000.0,
                trade_tick_size=0.00001, trade_tick_value=1.0, trade_stops_level=0, filling_mode=2,
            ),
            "GOLD": SimpleNamespace(
                name="GOLD", description="Gold", visible=True, digits=2, point=0.01,
                volume_min=0.01, volume_max=50.0, volume_step=0.01, trade_contract_size=100.0,
                trade_tick_size=0.01, trade_tick_value=1.0, trade_stops_level=0, filling_mode=2,
            ),
        }
        self.ticks = {
            "EURUSD": SimpleNamespace(bid=1.10000, ask=1.10010, time=1_790_000_000),
            "GOLD": SimpleNamespace(bid=4200.00, ask=4200.50, time=1_790_000_000),
        }
        self.calc_calls = []

    def account_info(self):
        return self.account

    def symbols_get(self):
        return list(self.info.values())

    def symbol_info(self, name):
        return self.info.get(name)

    def symbol_info_tick(self, name):
        return self.ticks.get(name)

    def symbol_select(self, name, enable):
        return True

    def positions_get(self):
        return self.positions

    def history_deals_get(self, date_from, date_to):
        return self.deals

    def order_calc_profit(self, order_type, symbol, volume, entry, stop_loss):
        self.calc_calls.append({
            "order_type": order_type,
            "symbol": symbol,
            "volume": volume,
            "entry": entry,
            "stop_loss": stop_loss,
        })
        contract = self.info[symbol].trade_contract_size
        sign = 1 if order_type == self.ORDER_TYPE_BUY else -1
        return (stop_loss - entry) * contract * volume * sign

    def order_check(self, request):
        self.calls.append(("check", dict(request)))
        return SimpleNamespace(retcode=0, comment="Done")

    def order_send(self, request):
        self.calls.append(("send", dict(request)))
        if "position" in request:
            return SimpleNamespace(retcode=10009, order=request["position"], price=request["price"], profit=0.84)
        ticket = 9000 + len([c for c in self.calls if c[0] == "send"])
        self.positions.append(SimpleNamespace(
            ticket=ticket,
            symbol=request["symbol"],
            type=0,
            volume=request["volume"],
            magic=request["magic"],
            comment=request["comment"],
            time=self.ticks[request["symbol"]].time,
            price_open=request["price"],
            profit=0.0,
        ))
        return SimpleNamespace(retcode=10009, order=ticket, deal=ticket + 100, price=request["price"], comment="sent")

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        base = self.ticks[symbol].time - (count * 3600)
        return [{"time": base + i * 3600} for i in range(count)]

    def last_error(self):
        return (0, "fake")


class FailedOrderMT5(FakeMT5):
    def order_send(self, request):
        self.calls.append(("send", dict(request)))
        return SimpleNamespace(retcode=10019, order=None, deal=None, price=request["price"], comment="no money")


class UnconfirmedOrderMT5(FakeMT5):
    def order_send(self, request):
        self.calls.append(("send", dict(request)))
        ticket = 9000 + len([c for c in self.calls if c[0] == "send"])
        return SimpleNamespace(retcode=10009, order=ticket, deal=ticket + 100, price=request["price"], comment="sent")


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_root(root, eurusd=True, gold=False, signal="LONG", execution="OPEN", bar=FRESH_BAR):
    metadata = root / "models" / "paper" / "eurusd_long_rf_actual_t065_metadata.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({"hash_modelo": EURUSD_EXPECTED_HASH}), encoding="utf-8")
    eurusd_path = root / "paper" / "eurusd" / "signals.csv"
    gold_path = root / "paper" / "signals.csv"
    if eurusd:
        _write_csv(eurusd_path, [{
            "SIGNAL_ID": f"{AUTHORIZED_EURUSD_CONFIG}|{bar}",
            "CONFIG_ID": AUTHORIZED_EURUSD_CONFIG,
            "ACTIVO": "EURUSD",
            "DIRECCION": "LONG",
            "MODELO": "RF_ACTUAL",
            "SISTEMA": "ML_ONLY",
            "THRESHOLD": "0.65",
            "FECHA_SIGNAL": bar,
            "SIGNAL": signal,
            "EXECUTION": execution,
            "PROBABILIDAD": "0.8",
            "ENTRY": "1.10010",
            "STOP": "1.09910",
            "TP": "1.10210",
            "ATR": "0.001",
            "VOLUME": "",
            "VOLUME_MIN": "",
            "RISK_USD": "",
            "REASON": "OK",
        }], ["SIGNAL_ID", "CONFIG_ID", "ACTIVO", "DIRECCION", "MODELO", "SISTEMA", "THRESHOLD",
             "FECHA_SIGNAL", "SIGNAL", "EXECUTION", "PROBABILIDAD", "ENTRY", "STOP", "TP",
             "ATR", "VOLUME", "VOLUME_MIN", "RISK_USD", "REASON"])
    elif eurusd_path.exists():
        eurusd_path.unlink()
    if gold:
        rows = []
        for threshold, suffix in (("0.55", "T055"), ("0.60", "T060")):
            config_id = f"GOLD_LONG_LOGISTIC_BASE_PLUS_ML_{suffix}"
            rows.append({
                "CONFIG_ID": config_id,
                "ACTIVO": "GOLD",
                "DIRECCION": "LONG",
                "MODELO": "LOGISTIC",
                "SISTEMA": "BASE_PLUS_ML",
                "THRESHOLD": threshold,
                "FECHA_SIGNAL": bar,
                "SIGNAL": signal,
                "PROBABILIDAD": "0.8",
                "VALIDATED": "True",
                "ENTRY": "4200.5",
                "RSI": "50",
                "ATR_SIGNAL": "10",
                "MA_20": "4200",
                "STOP": "4000.0",
                "TP": "5000.0",
            })
        _write_csv(gold_path, rows, [
            "CONFIG_ID", "ACTIVO", "DIRECCION", "MODELO", "SISTEMA", "THRESHOLD", "FECHA_SIGNAL",
            "SIGNAL", "PROBABILIDAD", "VALIDATED", "ENTRY", "RSI", "ATR_SIGNAL", "MA_20", "STOP", "TP",
        ])
    elif gold_path.exists():
        gold_path.unlink()


class Mt5DemoAutoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _write_root(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_account_blocked(self):
        mt5 = FakeMT5(trade_mode=FakeMT5.ACCOUNT_TRADE_MODE_REAL)
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = build_decisions(mt5, self.root, send=True, now=NOW)[0]
        self.assertEqual(result.reason, "REAL ACCOUNT BLOCKED")
        self.assertEqual(mt5.calls, [])

    def test_gold_order_send_uses_broker_gold_not_gcf(self):
        _write_root(self.root, eurusd=False, gold=True)
        mt5 = FakeMT5()

        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            build_decisions(mt5, self.root, send=True, now=NOW)

        sends = [payload for name, payload in mt5.calls if name == "send"]
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0]["symbol"], "GOLD")
        self.assertNotEqual(sends[0]["symbol"], "GC=F")

    def test_gold_entry_uses_broker_ask_for_buy(self):
        _write_root(self.root, eurusd=False, gold=True)
        mt5 = FakeMT5()

        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]

        self.assertEqual(result.price, mt5.ticks["GOLD"].ask)

    def test_gold_sl_tp_rebuilt_from_broker_price_not_gcf_absolute(self):
        _write_root(self.root, eurusd=False, gold=True)
        mt5 = FakeMT5()

        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]

        self.assertEqual(result.sl, 4190.50)
        self.assertEqual(result.tp, 4220.50)
        self.assertNotEqual(result.sl, 4000.0)
        self.assertNotEqual(result.tp, 5000.0)
        self.assertEqual(result.sl_distance, 10.0)
        self.assertEqual(result.tp_distance, 20.0)

    def test_gold_order_calc_profit_uses_broker_gold(self):
        _write_root(self.root, eurusd=False, gold=True)
        mt5 = FakeMT5()

        build_decisions(mt5, self.root, send=False, now=NOW)

        self.assertTrue(mt5.calc_calls)
        self.assertTrue(all(call["symbol"] == "GOLD" for call in mt5.calc_calls))

    def test_gold_volume_min_exceeds_risk_skip(self):
        _write_root(self.root, eurusd=False, gold=True)
        mt5 = FakeMT5(balance=100)

        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]

        self.assertEqual(result.reason, "SKIP_RISK")
        self.assertIsNone(result.volume)

    def test_demo_allowed_in_preview(self):
        result = build_decisions(FakeMT5(), self.root, send=False, now=NOW)[0]
        self.assertEqual(result.status, "PREVIEW")
        self.assertEqual(result.reason, "READY")

    def test_flag_false_blocks_send(self):
        result = build_decisions(FakeMT5(), self.root, send=True, now=NOW)[0]
        self.assertEqual(result.reason, "DEMO_EXECUTION_DISABLED")

    def test_fresh_signal_allowed(self):
        result = build_decisions(FakeMT5(), self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "READY")
        self.assertLessEqual(result.signal_age_minutes, 10)

    def test_stale_signal_blocked(self):
        _write_root(self.root, bar=OLD_BAR)
        result = build_decisions(FakeMT5(), self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "SKIP_STALE_SIGNAL")

    def test_catchup_old_signal_no_send(self):
        _write_root(self.root, bar=OLD_BAR)
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = build_decisions(mt5, self.root, send=True, now=NOW)[0]
        self.assertEqual(result.reason, "SKIP_STALE_SIGNAL")
        self.assertEqual(mt5.calls, [])

    def test_eurusd_hash_correct(self):
        self.assertEqual(build_decisions(FakeMT5(), self.root, send=False, now=NOW)[0].reason, "READY")

    def test_eurusd_hash_incorrect_blocks(self):
        path = self.root / "models" / "paper" / "eurusd_long_rf_actual_t065_metadata.json"
        path.write_text(json.dumps({"hash_modelo": "bad"}), encoding="utf-8")
        result = build_decisions(FakeMT5(), self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "EURUSD_HASH_MISMATCH_ABORT")

    def test_gold_t055_t060_dedupes_one_decision(self):
        _write_root(self.root, eurusd=False, gold=True)
        result = build_decisions(FakeMT5(), self.root, send=False, now=NOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0].config_ids), {
            "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T055",
            "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060",
        })

    def test_one_position_per_symbol(self):
        mt5 = FakeMT5()
        mt5.positions = [SimpleNamespace(ticket=1, symbol="EURUSD", magic=MAGIC_BY_ASSET["EURUSD"],
                                         comment=COMMENT_BY_ASSET["EURUSD"], profit=0.0)]
        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "SYMBOL_POSITION_ALREADY_OPEN")

    def test_max_two_positions(self):
        mt5 = FakeMT5()
        mt5.positions = [
            SimpleNamespace(ticket=1, symbol="GBPUSD", magic=MAGIC_BY_ASSET["EURUSD"], comment=COMMENT_BY_ASSET["EURUSD"], profit=0.0),
            SimpleNamespace(ticket=2, symbol="GOLD", magic=MAGIC_BY_ASSET["GOLD"], comment=COMMENT_BY_ASSET["GOLD"], profit=0.0),
        ]
        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "MAXIMUM_POSITIONS_REACHED")

    def test_daily_loss_three_percent_blocks(self):
        mt5 = FakeMT5(balance=10000)
        mt5.deals = [SimpleNamespace(magic=MAGIC_BY_ASSET["EURUSD"], profit=-300, commission=0, swap=0, fee=0)]
        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "MAXIMUM_DAILY_LOSS_REACHED")

    def test_skip_volume_min_exceeds_risk(self):
        mt5 = FakeMT5(balance=10)
        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "SKIP_RISK")

    def test_cap_volume_to_001(self):
        result = build_decisions(FakeMT5(balance=10000), self.root, send=False, now=NOW)[0]
        self.assertEqual(result.volume, 0.01)

    def test_no_round_up_when_theoretical_below_minimum(self):
        mt5 = FakeMT5(balance=100)
        mt5.ticks["EURUSD"] = SimpleNamespace(bid=1.10000, ask=1.10010, time=1_790_000_000)
        _write_root(self.root, bar=FRESH_BAR)
        path = self.root / "paper" / "eurusd" / "signals.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["STOP"] = "1.00010"
        _write_csv(path, rows, rows[0].keys())
        result = build_decisions(mt5, self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "SKIP_RISK")

    def test_order_check_before_order_send(self):
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            build_decisions(mt5, self.root, send=True, now=NOW)
        self.assertEqual([name for name, _ in mt5.calls[:2]], ["check", "send"])

    def test_reconciliation_blocks_missing_ticket(self):
        live = self.root / "live"
        live.mkdir()
        _write_csv(live / "forward_demo_orders.csv", [{
            "timestamp": NOW.isoformat(), "status": "SENT", "reason": "OK", "execution_id": "old",
            "config_ids": AUTHORIZED_EURUSD_CONFIG, "asset": "EURUSD", "symbol": "EURUSD",
            "direction": "BUY", "bar_timestamp": OLD_BAR, "volume": "0.01", "price": "1.1",
            "sl": "1.09", "tp": "1.12", "risk_usd": "1", "ticket": "999", "deal": "",
            "retcode": "10009", "comment": COMMENT_BY_ASSET["EURUSD"], "event": "OPEN",
            "profit": "", "commission": "", "swap": "",
        }], ["timestamp", "status", "reason", "execution_id", "config_ids", "asset", "symbol",
             "direction", "bar_timestamp", "volume", "price", "sl", "tp", "risk_usd",
             "ticket", "deal", "retcode", "comment", "event", "profit", "commission", "swap"])
        result = build_decisions(FakeMT5(), self.root, send=False, now=NOW)[0]
        self.assertEqual(result.reason, "ERROR / RECONCILIATION_REQUIRED")

    def test_idempotence_second_run_no_second_send(self):
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            build_decisions(mt5, self.root, send=True, now=NOW)
            build_decisions(mt5, self.root, send=True, now=NOW)
        sends = [call for call in mt5.calls if call[0] == "send"]
        self.assertEqual(len(sends), 1)

    def test_telegram_open(self):
        events = []
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            build_decisions(FakeMT5(), self.root, send=True, now=NOW, notify=lambda event, msg: events.append((event, msg)))
        self.assertIn("OPERACION ABIERTA", events[0][1])
        self.assertIn("Ticket: 9001", events[0][1])
        self.assertIn("Deal: 9101", events[0][1])

    def test_failed_order_does_not_generate_open_telegram(self):
        events = []
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = build_decisions(
                FailedOrderMT5(),
                self.root,
                send=True,
                now=NOW,
                notify=lambda event, msg: events.append((event, msg)),
            )[0]

        self.assertEqual(result.status, "ERROR")
        self.assertIn("ORDER_SEND_FAILED", result.reason)
        self.assertFalse(any("OPERACION ABIERTA" in msg for _, msg in events))
        self.assertEqual(len(events), 1)
        self.assertIn("ERROR DE EJECUCION", events[0][1])

    def test_successful_order_without_reconciliation_does_not_generate_open_telegram(self):
        events = []
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = build_decisions(
                UnconfirmedOrderMT5(),
                self.root,
                send=True,
                now=NOW,
                notify=lambda event, msg: events.append((event, msg)),
            )[0]

        self.assertEqual(result.status, "ERROR")
        self.assertEqual(result.reason, "ORDER_SENT_BUT_POSITION_NOT_CONFIRMED")
        self.assertFalse(any("OPERACION ABIERTA" in msg for _, msg in events))
        self.assertIn("ERROR DE EJECUCION", events[0][1])

    def test_telegram_close(self):
        events = []
        mt5 = FakeMT5()
        mt5.positions = [SimpleNamespace(ticket=7, symbol="EURUSD", type=0, volume=0.01,
                                         magic=MAGIC_BY_ASSET["EURUSD"], comment=COMMENT_BY_ASSET["EURUSD"],
                                         time=1_790_000_000 - 30 * 3600, price_open=1.1, profit=0.2)]
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            close_due_positions(mt5, self.root, notify=lambda event, msg: events.append((event, msg)), now=NOW)
        self.assertIn("OPERACION CERRADA", events[0][1])

    def test_telegram_no_wait(self):
        _write_root(self.root, signal="WAIT", execution="WAIT")
        events = []
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            build_decisions(FakeMT5(), self.root, send=True, now=NOW, notify=lambda event, msg: events.append((event, msg)))
        self.assertEqual(events, [])

    def test_telegram_failure_does_not_stop_paper_layer(self):
        def broken(event, msg):
            raise RuntimeError("telegram down")
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = build_decisions(FakeMT5(), self.root, send=True, now=NOW, notify=broken)[0]
        self.assertEqual(result.status, "SENT")

    def test_no_touch_foreign_positions(self):
        mt5 = FakeMT5()
        mt5.positions = [SimpleNamespace(ticket=99, symbol="EURUSD", type=0, volume=0.01,
                                         magic=123, comment="MANUAL", time=1_790_000_000 - 30 * 3600,
                                         price_open=1.1, profit=0.2)]
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            close_due_positions(mt5, self.root, now=NOW)
        self.assertEqual(mt5.calls, [])

    def _close_deal(self, ticket, reason=5, profit=2.0):
        return SimpleNamespace(ticket=ticket + 500, position_id=ticket, entry=1, magic=MAGIC_BY_ASSET["EURUSD"],
                               symbol="EURUSD", volume=0.01, price=1.102, reason=reason, profit=profit,
                               commission=0.0, swap=0.0, fee=0.0)

    def test_broker_close_does_not_block_next_signal(self):
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            first = build_decisions(mt5, self.root, send=True, now=NOW)[0]
            ticket = mt5.positions[0].ticket
            mt5.positions = []
            mt5.deals = [self._close_deal(ticket)]
            next_now = NOW + timedelta(hours=1)
            _write_root(self.root, bar="2026-10-01T11:00:00")
            second = build_decisions(mt5, self.root, send=True, now=next_now)[0]
        self.assertEqual(first.status, "SENT")
        self.assertEqual(second.status, "SENT")

    def test_daily_trade_limit_blocks_forward_send(self):
        live = self.root / "live"
        live.mkdir()
        with (live / "orders.csv").open("w", encoding="utf-8") as handle:
            handle.write("TIMESTAMP_UTC,STATUS,SIGNAL_ID\n")
            handle.writelines(f"{NOW.isoformat()},SENT,P{i}\n" for i in range(10))
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = build_decisions(mt5, self.root, send=True, now=NOW)[0]
        self.assertEqual(result.reason, "DAILY_TRADE_LIMIT")
        self.assertEqual([c for c in mt5.calls if c[0] == "send"], [])

    def test_kill_switch_blocks_send(self):
        (self.root / "STOP_TRADING").touch()
        mt5 = FakeMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            result = build_decisions(mt5, self.root, send=True, now=NOW)[0]
        self.assertEqual(result.reason, "KILL_SWITCH_ACTIVE")
        self.assertEqual([c for c in mt5.calls if c[0] == "send"], [])

    def test_unconfirmed_order_is_never_resent(self):
        mt5 = UnconfirmedOrderMT5()
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            build_decisions(mt5, self.root, send=True, now=NOW)
            second = build_decisions(mt5, self.root, send=True, now=NOW)[0]
        self.assertEqual(second.reason, "DUPLICATE_EXECUTION_ID")
        self.assertEqual(len([c for c in mt5.calls if c[0] == "send"]), 1)

    def test_broker_sl_tp_close_is_logged_and_notified_once(self):
        events = []
        mt5 = FakeMT5()
        mt5.deals = [self._close_deal(4242, reason=4, profit=-1.0)]
        notify = lambda event, msg: events.append((event, msg))
        sync_broker_closes(mt5, self.root, notify=notify, now=NOW)
        sync_broker_closes(mt5, self.root, notify=notify, now=NOW)
        self.assertEqual(len(events), 1)
        self.assertIn("STOP LOSS", events[0][1])
        self.assertIn("-1.00", events[0][1])
        rows = list(csv.DictReader((self.root / "live" / "forward_demo_orders.csv").open(encoding="utf-8")))
        self.assertEqual([(r["event"], r["ticket"]) for r in rows], [("CLOSE", "4242")])

    def test_max_hold_ignores_bar_still_forming(self):
        mt5 = FakeMT5()
        tick_time = mt5.ticks["EURUSD"].time
        # Abierta hace 24h + 10 min: solo 23 velas cerradas despues de la entrada + 1 en curso.
        mt5.ticks["EURUSD"] = SimpleNamespace(bid=1.1, ask=1.1001, time=tick_time + 600)
        mt5.copy_rates_from_pos = lambda symbol, tf, start, count: [
            {"time": tick_time - (24 - i) * 3600} for i in range(25)
        ]
        mt5.positions = [SimpleNamespace(ticket=8, symbol="EURUSD", type=0, volume=0.01,
                                         magic=MAGIC_BY_ASSET["EURUSD"], comment=COMMENT_BY_ASSET["EURUSD"],
                                         time=tick_time - 24 * 3600, price_open=1.1, profit=0.0)]
        with patch.object(config, "DEMO_EXECUTION_ENABLED", True):
            closed = close_due_positions(mt5, self.root, now=NOW)
        self.assertEqual(closed, [])


if __name__ == "__main__":
    unittest.main()
