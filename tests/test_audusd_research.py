import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import audusd_research as ar


class AudUsdResearchTests(unittest.TestCase):
    def test_research_outputs_only_audusd_v2_and_does_not_create_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = pd.DataFrame(
                {
                    "fecha": pd.date_range("2026-01-01", periods=3, freq="h"),
                    "Open": [0.70, 0.701, 0.702],
                    "High": [0.701, 0.702, 0.703],
                    "Low": [0.699, 0.700, 0.701],
                    "Close": [0.7005, 0.7015, 0.7025],
                    "ATR": [0.001, 0.001, 0.001],
                }
            )
            source = {
                "source": "test",
                "interval": "1h",
                "downloaded_period": "N/A",
                "created_now": False,
            }
            audit = pd.DataFrame(
                [
                    {
                        "ACTIVO": "AUDUSD",
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
            sizing_validation = pd.DataFrame(columns=["RESULTADO"])

            with (
                patch.object(ar, "_read_or_create_dataset", return_value=(data, source)),
                patch.object(ar, "_audit_existing_models", return_value=audit),
                patch.object(
                    ar,
                    "_run_with_audusd_asset",
                    return_value=(walkforward_raw, oof, oof_summary, trades, robustness),
                ),
                patch.object(ar, "_walkforward_economics", return_value=walkforward),
                patch.object(ar, "_broker_aware", return_value=(broker, sizing_validation)),
            ):
                result = ar.run_audusd_research(root)

            out = root / "results" / "v2" / "audusd"
            self.assertEqual(result["output_dir"], out)
            self.assertTrue((out / "audusd_existing_models_audit.csv").exists())
            self.assertTrue((out / "audusd_walkforward_summary.csv").exists())
            self.assertTrue((out / "audusd_oof_summary.csv").exists())
            self.assertFalse((root / "paper" / "audusd").exists())
            self.assertFalse((root / "models" / "paper").exists())
            self.assertFalse((root / "results" / "v2" / "gbpusd").exists())
            self.assertFalse((root / "results" / "v2" / "usdjpy").exists())
            self.assertFalse((root / "results" / "v2" / "eurusd").exists())


if __name__ == "__main__":
    unittest.main()
