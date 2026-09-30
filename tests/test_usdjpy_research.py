import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import usdjpy_research as ur


class UsdJpyResearchTests(unittest.TestCase):
    def test_research_outputs_only_usdjpy_v2_and_does_not_create_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = pd.DataFrame(
                {
                    "fecha": pd.date_range("2026-01-01", periods=3, freq="h"),
                    "Open": [150.0, 150.1, 150.2],
                    "High": [150.2, 150.3, 150.4],
                    "Low": [149.9, 150.0, 150.1],
                    "Close": [150.1, 150.2, 150.3],
                    "ATR": [0.1, 0.1, 0.1],
                }
            )
            audit = pd.DataFrame(
                [
                    {
                        "ACTIVO": "USDJPY",
                        "DIRECCION": "LONG",
                        "EXISTE": False,
                    }
                ]
            )
            walkforward_raw = pd.DataFrame(
                [{"DIRECCION": "LONG", "MODELO": "LOGISTIC", "FOLD": 1, "ROC_AUC": 0.5}]
            )
            oof = pd.DataFrame(
                [{"DIRECCION": "LONG", "MODELO": "LOGISTIC", "FOLD": 1, "FECHA": data["fecha"].iloc[0]}]
            )
            oof_summary = pd.DataFrame(
                [
                    {
                        "COST_BPS": 2,
                        "CLASIFICACION": "NO APTO",
                        "DIRECCION": "LONG",
                        "MODELO": "LOGISTIC",
                        "SISTEMA": "ML_ONLY",
                        "THRESHOLD": 0.55,
                        "TRADES": 0,
                        "PROFIT_FACTOR": 0,
                        "EXPECTANCY_R": 0,
                        "R_TOTAL": 0,
                        "MAX_DRAWDOWN_PCT": 0,
                    }
                ]
            )
            trades = pd.DataFrame(columns=["COST_BPS"])
            robustness = pd.DataFrame(
                [
                    {
                        "COST_BPS": 2,
                        "CLASIFICACION": "NO APTO",
                        "DIRECCION": "LONG",
                        "MODELO": "LOGISTIC",
                        "SISTEMA": "ML_ONLY",
                        "THRESHOLD": 0.55,
                        "TRADES": 0,
                        "MOTIVO": "test",
                    }
                ]
            )
            walkforward = pd.DataFrame(
                [{"COST_BPS": 2, "DIRECCION": "LONG", "MODELO": "LOGISTIC"}]
            )
            broker = pd.DataFrame(columns=["CAPITAL", "DECISION", "REASON"])

            with (
                patch.object(ur, "_read_dataset", return_value=data),
                patch.object(ur, "_audit_existing_models", return_value=audit),
                patch.object(
                    ur,
                    "_run_with_usdjpy_asset",
                    return_value=(walkforward_raw, oof, oof_summary, trades, robustness),
                ),
                patch.object(ur, "_walkforward_economics", return_value=walkforward),
                patch.object(ur, "_broker_aware", return_value=broker),
            ):
                result = ur.run_usdjpy_research(root)

            out = root / "results" / "v2" / "usdjpy"
            self.assertEqual(result["output_dir"], out)
            self.assertTrue((out / "usdjpy_existing_models_audit.csv").exists())
            self.assertTrue((out / "usdjpy_walkforward_summary.csv").exists())
            self.assertTrue((out / "usdjpy_oof_summary.csv").exists())
            self.assertFalse((root / "paper" / "usdjpy").exists())
            self.assertFalse((root / "models" / "paper").exists())
            self.assertFalse((root / "results" / "v2" / "gbpusd").exists())
            self.assertFalse((root / "results" / "v2" / "eurusd").exists())


if __name__ == "__main__":
    unittest.main()
