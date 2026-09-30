import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.multi_asset import inspect_asset_pipeline


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


class MultiAssetPipelineTests(unittest.TestCase):
    def test_models_are_separate_by_asset_and_direction(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            touch(root / "models" / "eurusd" / "random_forest_long.pkl")
            touch(root / "models" / "eurusd" / "random_forest_short.pkl")

            eurusd = inspect_asset_pipeline("EURUSD", root)
            gbpusd = inspect_asset_pipeline("GBPUSD", root)

            self.assertTrue(eurusd.long_model.available)
            self.assertTrue(eurusd.short_model.available)
            self.assertFalse(gbpusd.long_model.available)
            self.assertFalse(gbpusd.short_model.available)

    def test_backtest_alone_does_not_validate_model(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            touch(root / "models" / "gold" / "random_forest_long.pkl")
            touch(root / "models" / "gold" / "random_forest_short.pkl")
            touch(root / "results" / "gold" / "summary.csv")
            touch(root / "results" / "gold" / "trades.csv")

            state = inspect_asset_pipeline("GOLD", root)

            self.assertTrue(state.backtest_ready)
            self.assertFalse(state.validated)
            self.assertIn("forward test", state.missing_validation)
            self.assertIn("modelo congelado", state.missing_validation)

    def test_costs_require_oof_summary_with_cost_and_threshold_columns(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / "results" / "gold" / "oof_backtest_summary.csv"
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text("ACTIVO,THRESHOLD,COST_BPS\nGOLD,0.6,1\n", encoding="utf-8")

            state = inspect_asset_pipeline("GOLD", root)

            self.assertTrue(state.costs_ready)


if __name__ == "__main__":
    unittest.main()
