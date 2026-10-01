import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.gold_data_check import run_gold_data_check
from src.yfinance_cache import ensure_yfinance_cache


class GoldDataCheckTests(unittest.TestCase):
    def test_yfinance_cache_is_project_local_and_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ensure_yfinance_cache(tmp)

            self.assertEqual(cache, Path(tmp) / "data" / "yfinance_cache")
            self.assertTrue(cache.exists())
            probe = cache / "probe.txt"
            probe.write_text("ok", encoding="utf-8")
            self.assertEqual(probe.read_text(encoding="utf-8"), "ok")

    def test_yfinance_cache_is_gitignored(self):
        text = Path(".gitignore").read_text(encoding="utf-8")

        self.assertIn("data/yfinance_cache/", text)

    def test_gold_data_check_does_not_create_paper_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = pd.date_range("2026-10-01T00:00:00Z", periods=3, freq="h")
            data = pd.DataFrame(
                {
                    "Open": [1.0, 1.1, 1.2],
                    "High": [1.2, 1.3, 1.4],
                    "Low": [0.9, 1.0, 1.1],
                    "Close": [1.1, 1.2, 1.3],
                    "Volume": [10, 11, 12],
                },
                index=index,
            )
            with patch("yfinance.download", return_value=data):
                result = run_gold_data_check(root)

            self.assertIn(result["data_status"], {"LIVE", "STALE"})
            self.assertFalse((root / "paper" / "signals.csv").exists())
            self.assertFalse((root / "paper" / "state.json").exists())


if __name__ == "__main__":
    unittest.main()
