import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

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

    def test_close_winner(self):
        te.notify_close(self.root, self.close_row(1.82, "TAKE PROFIT"), send_func=self.sender)

        self.assertIn("OPERACION PAPER CERRADA", self.sent[0])
        self.assertIn("+$1.82", self.sent[0])
        self.assertIn("+1.82R", self.sent[0])

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

        self.assertIn("SEÑAL NO EJECUTADA", self.sent[0])
        self.assertIn("0.006", self.sent[0])
        self.assertIn("0.01", self.sent[0])

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
