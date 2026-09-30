import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import gbpusd_research as gr


class GbpUsdResearchTests(unittest.TestCase):
    def test_research_outputs_only_gbpusd_v2_and_does_not_create_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = pd.DataFrame(
                {
                    "fecha": pd.date_range("2026-01-01", periods=3, freq="h"),
                    "Open": [1.2, 1.21, 1.22],
                    "High": [1.21, 1.22, 1.23],
                    "Low": [1.19, 1.20, 1.21],
                    "Close": [1.205, 1.215, 1.225],
                    "ATR": [0.001, 0.001, 0.001],
                }
            )
            audit = pd.DataFrame(
                [
                    {
                        "ACTIVO": "GBPUSD",
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
                patch.object(gr, "_read_dataset", return_value=data),
                patch.object(gr, "_audit_existing_models", return_value=audit),
                patch.object(
                    gr,
                    "_run_with_gbpusd_asset",
                    return_value=(walkforward_raw, oof, oof_summary, trades, robustness),
                ),
                patch.object(gr, "_walkforward_economics", return_value=walkforward),
                patch.object(gr, "_broker_aware", return_value=broker),
            ):
                result = gr.run_gbpusd_research(root)

            out = root / "results" / "v2" / "gbpusd"
            self.assertEqual(result["output_dir"], out)
            self.assertTrue((out / "gbpusd_existing_models_audit.csv").exists())
            self.assertTrue((out / "gbpusd_walkforward_summary.csv").exists())
            self.assertTrue((out / "gbpusd_oof_summary.csv").exists())
            self.assertFalse((root / "paper" / "gbpusd").exists())
            self.assertFalse((root / "models" / "paper").exists())
            self.assertFalse((root / "results" / "v2" / "eurusd").exists())


if __name__ == "__main__":
    unittest.main()
