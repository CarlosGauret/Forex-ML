import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import usdcad_research as ur


class UsdCadResearchTests(unittest.TestCase):
    def test_candidate_is_downgraded_when_expectancy_ci_crosses_zero(self):
        robustness = pd.DataFrame(
            [
                {
                    "CLASIFICACION": "CANDIDATO A FORWARD TEST",
                    "MOTIVO": "old",
                    "BOOTSTRAP_EXPECTANCY_CI95_LOW": -0.1,
                    "BOOTSTRAP_EXPECTANCY_CI95_HIGH": 0.2,
                }
            ]
        )

        result = ur._apply_candidate_safety_rules(robustness)

        self.assertEqual(result.loc[0, "CLASIFICACION"], "MIXTO")
        self.assertIn("VENTAJA NO CONFIRMADA", result.loc[0, "MOTIVO"])

    def test_research_outputs_only_usdcad_v2_and_does_not_create_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = pd.DataFrame(
                {
                    "fecha": pd.date_range("2026-01-01", periods=3, freq="h"),
                    "Open": [1.35, 1.351, 1.352],
                    "High": [1.351, 1.352, 1.353],
                    "Low": [1.349, 1.350, 1.351],
                    "Close": [1.3505, 1.3515, 1.3525],
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
                        "ACTIVO": "USDCAD",
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
                        "BOOTSTRAP_EXPECTANCY_CI95_LOW": None,
                        "BOOTSTRAP_EXPECTANCY_CI95_HIGH": None,
                    }
                ]
            )
            walkforward = pd.DataFrame(
                [{"COST_BPS": 2, "DIRECCION": "LONG", "MODELO": "LOGISTIC"}]
            )
            broker = pd.DataFrame(columns=["CAPITAL", "DECISION", "REASON"])
            sizing_validation = pd.DataFrame(columns=["RESULTADO"])

            with (
                patch.object(ur, "_read_or_create_dataset", return_value=(data, source)),
                patch.object(ur, "_audit_existing_models", return_value=audit),
                patch.object(
                    ur,
                    "_run_with_usdcad_asset",
                    return_value=(walkforward_raw, oof, oof_summary, trades, robustness),
                ),
                patch.object(ur, "_walkforward_economics", return_value=walkforward),
                patch.object(ur, "_broker_aware", return_value=(broker, sizing_validation)),
            ):
                result = ur.run_usdcad_research(root)

            out = root / "results" / "v2" / "usdcad"
            self.assertEqual(result["output_dir"], out)
            self.assertTrue((out / "usdcad_existing_models_audit.csv").exists())
            self.assertTrue((out / "usdcad_walkforward_summary.csv").exists())
            self.assertTrue((out / "usdcad_oof_summary.csv").exists())
            self.assertFalse((root / "paper" / "usdcad").exists())
            self.assertFalse((root / "models" / "paper").exists())
            self.assertFalse((root / "results" / "v2" / "audusd").exists())
            self.assertFalse((root / "results" / "v2" / "gbpusd").exists())
            self.assertFalse((root / "results" / "v2" / "usdjpy").exists())
            self.assertFalse((root / "results" / "v2" / "eurusd").exists())


if __name__ == "__main__":
    unittest.main()
