import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import nzdusd_research as nr


class NzdUsdResearchTests(unittest.TestCase):
    def test_create_dataset_does_not_persist_data_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = pd.DataFrame(
                {
                    "Open": [0.61, 0.611],
                    "High": [0.612, 0.613],
                    "Low": [0.609, 0.610],
                    "Close": [0.611, 0.612],
                    "Volume": [0, 0],
                },
                index=pd.date_range("2026-01-01", periods=2, freq="h"),
            )
            raw.attrs["intervalo"] = "1h"
            raw.attrs["periodo_descargado"] = "2y"
            dataset = pd.DataFrame({"fecha": raw.index, "Open": raw["Open"]})

            with (
                patch("src.data._descargar_y_preparar", return_value=raw),
                patch("src.indicators.calcular_indicadores", return_value=raw),
                patch("src.features.crear_dataset_ml", return_value=dataset),
            ):
                created, source = nr._create_dataset(root)

            self.assertEqual(len(created), 2)
            self.assertEqual(source["source"], "Yahoo Finance NZDUSD=X")
            self.assertFalse((root / "data" / "nzdusd").exists())

    def test_candidate_is_downgraded_when_expectancy_ci_crosses_zero(self):
        robustness = pd.DataFrame(
            [
                {
                    "CLASIFICACION": "CANDIDATO A FORWARD TEST",
                    "MOTIVO": "expectancy > 0 y PF > 1",
                    "BOOTSTRAP_EXPECTANCY_CI95_LOW": -0.01,
                    "BOOTSTRAP_EXPECTANCY_CI95_HIGH": 0.2,
                }
            ]
        )

        result = nr._apply_candidate_safety_rules(robustness)

        self.assertEqual(result.loc[0, "CLASIFICACION"], "MIXTO")
        self.assertIn("VENTAJA NO CONFIRMADA", result.loc[0, "MOTIVO"])

    def test_research_outputs_only_nzdusd_v2_and_does_not_create_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = pd.DataFrame(
                {
                    "fecha": pd.date_range("2026-01-01", periods=3, freq="h"),
                    "Open": [0.61, 0.611, 0.612],
                    "High": [0.611, 0.612, 0.613],
                    "Low": [0.609, 0.610, 0.611],
                    "Close": [0.6105, 0.6115, 0.6125],
                    "ATR": [0.001, 0.001, 0.001],
                }
            )
            source = {
                "source": "test",
                "interval": "1h",
                "downloaded_period": "N/A",
                "created_now": False,
            }
            audit = pd.DataFrame([{"ACTIVO": "NZDUSD", "DIRECCION": "LONG", "EXISTE": False}])
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
                patch.object(nr, "_read_or_create_dataset", return_value=(data, source)),
                patch.object(nr, "_audit_existing_models", return_value=audit),
                patch.object(
                    nr.shared_research,
                    "_run_with_usdcad_asset",
                    return_value=(walkforward_raw, oof, oof_summary, trades, robustness),
                ),
                patch.object(nr.shared_research, "_walkforward_economics", return_value=walkforward),
                patch.object(nr.shared_research, "_broker_aware", return_value=(broker, sizing_validation)),
            ):
                result = nr.run_nzdusd_research(root)

            out = root / "results" / "v2" / "nzdusd"
            self.assertEqual(result["output_dir"], out)
            self.assertTrue((out / "nzdusd_existing_models_audit.csv").exists())
            self.assertTrue((out / "nzdusd_walkforward_summary.csv").exists())
            self.assertTrue((out / "nzdusd_oof_summary.csv").exists())
            self.assertFalse((root / "paper" / "nzdusd").exists())
            self.assertFalse((root / "models" / "paper").exists())
            self.assertFalse((root / "data" / "nzdusd").exists())
            self.assertFalse((root / "results" / "v2" / "usdchf").exists())
            self.assertFalse((root / "results" / "v2" / "usdcad").exists())
            self.assertFalse((root / "results" / "v2" / "audusd").exists())
            self.assertFalse((root / "results" / "v2" / "usdjpy").exists())


if __name__ == "__main__":
    unittest.main()
