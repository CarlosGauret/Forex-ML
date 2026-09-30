import csv
import tempfile
import unittest
from pathlib import Path

from src.analytics_logger import DRYRUN_LOG_COLUMNS, write_dryrun_log


class AnalyticsLoggerTests(unittest.TestCase):
    def test_write_dryrun_log_uses_separate_v2_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_dryrun_log(
                [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "activo_logico": "EURUSD",
                        "simbolo_mt5": "EURUSDm",
                        "decision": "ESPERAR",
                        "reason": "MODELO_VALIDADO_NO",
                    }
                ],
                root,
            )

            self.assertEqual(path, root / "logs" / "mt5_dryrun_v2.csv")
            self.assertFalse((root / "paper" / "signals.csv").exists())

            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["activo_logico"], "EURUSD")
            self.assertEqual(rows[0]["decision"], "ESPERAR")
            self.assertEqual(list(rows[0].keys()), DRYRUN_LOG_COLUMNS)


if __name__ == "__main__":
    unittest.main()
