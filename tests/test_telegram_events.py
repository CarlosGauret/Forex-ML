import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from src import telegram_events as te


class TelegramEventsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sent = []

    def tearDown(self):
        self.tmp.cleanup()

    def sender(self, message):
        self.sent.append(message)
        return True

    def signal_row(self, signal="LONG"):
        return {
            "CONFIG_ID": "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065",
            "ACTIVO": "EURUSD",
            "DIRECCION": "LONG",
            "THRESHOLD": 0.65,
            "FECHA_SIGNAL": "2026-09-30T15:00:00",
            "SIGNAL": signal,
            "PROBABILIDAD": 0.6842,
        }

    def open_row(self):
        return {
            "CONFIG_ID": "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065",
            "ACTIVO": "EURUSD",
            "DIRECCION": "LONG",
            "FECHA_ENTRADA": "2026-09-30T16:00:00",
            "ENTRY": 1.1752,
            "STOP": 1.1728,
            "TP": 1.18,
            "VOLUME": 0.01,
            "RISK_USD": 1.0,
            "CAPITAL_ANTES": 100.0,
        }

    def close_row(self, pnl=1.82, reason="TP"):
        before = 100.0
        return {
            "TRADE_ID": f"TRD-{pnl}",
            "CONFIG_ID": "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065",
            "ACTIVO": "EURUSD",
            "DIRECCION": "LONG",
            "FECHA_ENTRADA": "2026-09-30T16:00:00",
            "FECHA_SALIDA": "2026-09-30T22:00:00",
            "ENTRY": 1.1752,
            "EXIT": 1.1798,
            "R_NETO": pnl / 1.0,
            "CAPITAL_ANTES": before,
            "CAPITAL_DESPUES": before + pnl,
            "MOTIVO_SALIDA": reason,
        }

    def test_signal_new(self):
        result = te.notify_signal(self.root, self.signal_row(), send_func=self.sender)

        self.assertTrue(result["sent"])
        self.assertIn("NUEVA SEÑAL PAPER", self.sent[0])
        self.assertIn("EVALUANDO RIESGO", self.sent[0])

    def test_open(self):
        te.notify_open(self.root, self.open_row(), send_func=self.sender)

        self.assertIn("OPERACION PAPER ABIERTA", self.sent[0])
        self.assertIn("PAPER / SIMULACION", self.sent[0])
        self.assertIn("Volumen simulado: 0.01", self.sent[0])

    def test_mock_open_can_be_tested_without_network(self):
        fake_requests = SimpleNamespace(post=Mock())
        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = te.notify_open(self.root, self.open_row(), send_func=self.sender)

        self.assertTrue(result["sent"])
        self.assertIn("OPERACION PAPER ABIERTA", self.sent[0])
        fake_requests.post.assert_not_called()

    def test_close_winner(self):
        te.notify_close(self.root, self.close_row(1.82, "TAKE PROFIT"), send_func=self.sender)

        self.assertIn("OPERACION PAPER CERRADA", self.sent[0])
        self.assertIn("+$1.82", self.sent[0])
        self.assertIn("+1.82R", self.sent[0])

    def test_mock_close_can_be_tested_without_network(self):
        fake_requests = SimpleNamespace(post=Mock())
        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = te.notify_close(self.root, self.close_row(1.82, "TAKE PROFIT"), send_func=self.sender)

        self.assertTrue(result["sent"])
        self.assertIn("OPERACION PAPER CERRADA", self.sent[0])
        fake_requests.post.assert_not_called()

    def test_close_loser(self):
        te.notify_close(self.root, self.close_row(-0.96, "STOP LOSS"), send_func=self.sender)

        self.assertIn("-$0.96", self.sent[0])
        self.assertIn("-0.96R", self.sent[0])

    def test_skip_risk(self):
        row = self.signal_row()
        row.update(
            {
                "SIGNAL_ID": "S1",
                "REASON": "MINIMUM_VOLUME_EXCEEDS_RISK",
                "VOLUME_THEORETICAL": 0.006,
                "VOLUME_MIN": 0.01,
            }
        )

        te.notify_skip_risk(self.root, row, send_func=self.sender)

        self.assertIn("NO EJECUTADA", self.sent[0])
        self.assertIn("0.006", self.sent[0])
        self.assertIn("0.01", self.sent[0])

    def test_mock_skip_risk_can_be_tested_without_network(self):
        row = self.signal_row()
        row.update(
            {
                "SIGNAL_ID": "S1",
                "REASON": "MINIMUM_VOLUME_EXCEEDS_RISK",
                "VOLUME_THEORETICAL": 0.006,
                "VOLUME_MIN": 0.01,
            }
        )
        fake_requests = SimpleNamespace(post=Mock())

        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = te.notify_skip_risk(self.root, row, send_func=self.sender)

        self.assertTrue(result["sent"])
        self.assertIn("NO EJECUTADA", self.sent[0])
        fake_requests.post.assert_not_called()

    def test_error_critico(self):
        te.notify_critical_error(self.root, "EURUSD", "STALE", "DATA_STALE", send_func=self.sender)

        self.assertIn("FOREX ML - ERROR", self.sent[0])
        self.assertIn("DATA STATUS: STALE", self.sent[0])

    def test_daily_report(self):
        paper = self.root / "paper" / "eurusd"
        paper.mkdir(parents=True)
        pd.DataFrame([self.close_row(1.82)]).to_csv(paper / "trades.csv", index=False)
        pd.DataFrame(
            [
                {
                    "FECHA": "2026-09-30T22:00:00",
                    "CONFIG_ID": "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065",
                    "CAPITAL": 101.82,
                }
            ]
        ).to_csv(paper / "equity.csv", index=False)
        pd.DataFrame([self.signal_row()]).to_csv(paper / "signals.csv", index=False)

        message = te.build_daily_report(self.root, now="2026-10-01T03:30:00+00:00")

        self.assertIn("30/09/2026", message)
        self.assertIn("Operaciones cerradas: 1", message)
        self.assertIn("+$1.82", message)

    def test_daily_report_without_trades(self):
        message = te.build_daily_report(self.root, now="2026-10-01T03:30:00+00:00")

        self.assertIn("Operaciones cerradas: 0", message)
        self.assertIn("Resultado neto: $0.00", message)
        self.assertIn("MT5 DEMO", message)

    def test_daily_report_separates_mt5_demo_real_execution(self):
        live = self.root / "live"
        live.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-09-30T22:00:00+00:00",
                    "status": "CLOSED",
                    "reason": "TP",
                    "execution_id": "E1",
                    "config_ids": "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065",
                    "asset": "EURUSD",
                    "symbol": "EURUSD",
                    "direction": "BUY",
                    "bar_timestamp": "2026-09-30T20:00:00",
                    "volume": 0.01,
                    "price": 1.1,
                    "sl": 1.09,
                    "tp": 1.12,
                    "risk_usd": 1.0,
                    "ticket": 1,
                    "deal": 2,
                    "retcode": 10009,
                    "comment": "FOREX_ML_EURUSD_DEMO",
                    "event": "CLOSE",
                    "profit": 0.84,
                    "commission": 0,
                    "swap": 0,
                }
            ]
        ).to_csv(live / "forward_demo_orders.csv", index=False)

        message = te.build_daily_report(self.root, now="2026-10-01T03:30:00+00:00")

        self.assertIn("MT5 DEMO", message)
        self.assertIn("Operaciones: 1", message)
        self.assertIn("Resultado neto: +$0.84", message)

    def test_timezone_america_lima(self):
        paper = self.root / "paper" / "eurusd"
        paper.mkdir(parents=True)
        pd.DataFrame([self.close_row(1.0)]).to_csv(paper / "trades.csv", index=False)

        message = te.build_daily_report(self.root, now="2026-10-01T03:30:00+00:00")

        self.assertIn("30/09/2026", message)
        self.assertIn("Operaciones cerradas: 1", message)

    def test_antiduplicado(self):
        first = te.notify_signal(self.root, self.signal_row(), send_func=self.sender)
        second = te.notify_signal(self.root, self.signal_row(), send_func=self.sender)

        self.assertTrue(first["sent"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(self.sent), 1)

    def test_telegram_failure_no_detiene_paper(self):
        def fail(_message):
            raise RuntimeError("network")

        result = te.notify_open(self.root, self.open_row(), send_func=fail)

        self.assertFalse(result["sent"])
        self.assertTrue((self.root / "logs" / "telegram_errors.log").exists())

    def test_unit_tests_never_call_real_telegram_sender(self):
        from src.notifier import enviar_telegram

        fake_requests = SimpleNamespace(post=Mock(return_value=SimpleNamespace(ok=True, status_code=200)))
        with patch.dict(
            os.environ,
            {
                "FOREX_ML_TESTING": "True",
                "TELEGRAM_BOT_TOKEN": "TEST_TOKEN",
                "TELEGRAM_CHAT_ID": "TEST_CHAT",
            },
            clear=False,
        ), patch.dict(sys.modules, {"requests": fake_requests}):
            result = enviar_telegram("no network")

        self.assertFalse(result)
        fake_requests.post.assert_not_called()

    def test_send_telegram_event_blocks_real_sender_in_tests(self):
        fake_requests = SimpleNamespace(post=Mock(return_value=SimpleNamespace(ok=True, status_code=200)))
        with patch.dict(
            os.environ,
            {
                "FOREX_ML_TESTING": "True",
                "TELEGRAM_BOT_TOKEN": "TEST_TOKEN",
                "TELEGRAM_CHAT_ID": "TEST_CHAT",
            },
            clear=False,
        ), patch.dict(sys.modules, {"requests": fake_requests}):
            result = te.send_telegram_event(self.root, "E1", "OPEN", "no network")

        self.assertFalse(result["sent"])
        self.assertEqual(result["error"], "TELEGRAM_DISABLED_IN_TEST")
        fake_requests.post.assert_not_called()

    def test_numpy_importing_unittest_does_not_block_production_telegram(self):
        import numpy.testing  # noqa: F401  (importa unittest, como ocurre en produccion)
        from src.notifier import testing_mode_enabled

        fake_main = SimpleNamespace(__spec__=None)
        env = {k: v for k, v in os.environ.items() if k != "FOREX_ML_TESTING"}
        with patch.dict(os.environ, env, clear=True), \
                patch.dict(sys.modules, {"__main__": fake_main}), \
                patch.dict(sys.modules, {"pytest": None}):
            sys.modules.pop("pytest", None)
            self.assertIn("unittest", sys.modules)
            self.assertFalse(testing_mode_enabled())

    def test_real_demo_telegram_integration_is_preserved_outside_test_mode(self):
        with patch.dict(os.environ, {"FOREX_ML_TESTING": "False"}, clear=False), \
                patch("src.notifier.enviar_telegram", return_value=True) as sender:
            result = te.send_telegram_event(self.root, "DEMO_REAL", "MT5_DEMO", "demo message")

        self.assertTrue(result["sent"])
        sender.assert_called_once_with("demo message")

    def test_secretos_no_impresos(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "SECRET_TOKEN", "TELEGRAM_CHAT_ID": "SECRET_CHAT"},
            clear=False,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = te.telegram_status(self.root)
                print(f"Telegram configurado: {'SI' if status['configured'] else 'NO'}")
                print(f"Ultimo evento enviado: {status['last_event_id'] or 'N/A'}")

        output = buffer.getvalue()
        self.assertNotIn("SECRET_TOKEN", output)
        self.assertNotIn("SECRET_CHAT", output)

    def test_no_wait_spam(self):
        result = te.notify_signal(self.root, self.signal_row(signal="WAIT"), send_func=self.sender)

        self.assertFalse(result["sent"])
        self.assertEqual(result["error"], "WAIT_IGNORED")
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
