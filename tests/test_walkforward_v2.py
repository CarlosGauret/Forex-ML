import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtest_v2 import BrokerMetadata
from src.sl_tp import ATR_ONLY
from src.walkforward_v2 import (
    EMBARGO_BARS,
    generar_folds_expanding,
    ejecutar_walkforward_v2_gold,
)


def _sample_oof(rows=150):
    fechas = pd.date_range("2025-01-01", periods=rows, freq="h")
    data = []
    for i, fecha in enumerate(fechas):
        base = 100.0 + (i * 0.08)
        wave = 2.0 if (i // 7) % 2 == 0 else -2.0
        price = base + wave
        high = price + 1.0
        low = price - 1.0
        if i % 17 == 5:
            low = price - 4.0
        if i % 19 == 9:
            high = price + 4.0
        base_signal = "BUY" if i % 9 == 0 else "WAIT"
        data.append(
            {
                "FECHA": fecha,
                "ACTIVO": "GOLD",
                "DIRECCION": "LONG",
                "MODELO": "LOGISTIC",
                "FOLD": 1,
                "CLOSE": price,
                "OPEN": price,
                "HIGH": high,
                "LOW": low,
                "ATR": 1.0,
                "PROBABILIDAD": 0.70,
                "TARGET_REAL": 1,
                "BASE_SIGNAL": base_signal,
            }
        )
    return pd.DataFrame(data)


def _metadata(volume_min=0.01):
    return BrokerMetadata(
        symbol="XAUUSD",
        contract_size=100.0,
        volume_min=volume_min,
        volume_max=50.0,
        volume_step=0.01,
        point=0.01,
        digits=2,
        source="TEST",
    )


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WalkForwardV2Tests(unittest.TestCase):
    def test_temporal_order_no_shuffle_and_embargo(self):
        fechas = pd.date_range("2025-01-01", periods=150, freq="h")
        folds = generar_folds_expanding(fechas, n_folds=3, embargo_bars=EMBARGO_BARS)

        self.assertEqual([fold["FOLD"] for fold in folds], [1, 2, 3])
        for fold in folds:
            self.assertLess(fold["TRAIN_END"], fold["EMBARGO_START"])
            self.assertLessEqual(fold["EMBARGO_START"], fold["EMBARGO_END"])
            self.assertLess(fold["EMBARGO_END"], fold["TEST_START"])
            self.assertGreaterEqual(fold["EMBARGO_BARS"], 24)
        self.assertLess(folds[0]["TEST_START"], folds[1]["TEST_START"])
        self.assertLess(folds[1]["TRAIN_END"], folds[2]["TRAIN_END"])

    def test_same_entries_between_sl_tp_modes_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ejecutar_walkforward_v2_gold(
                tmp,
                oof_predictions=_sample_oof(),
                broker_metadata=_metadata(),
                costs=[2],
                capitals=[1000],
                n_folds=3,
            )

        trades = result["trades"]
        theoretical = trades[~trades["BROKER_AWARE"]]
        for (_, config_id, fold), group in theoretical.groupby(["COST_BPS", "CONFIG_ID", "FOLD"]):
            sets = {
                mode: set(mode_group["ENTRY_ID"])
                for mode, mode_group in group.groupby("SL_TP_MODE")
            }
            self.assertEqual(len({tuple(sorted(values)) for values in sets.values()}), 1)
            self.assertTrue((group["FECHA_SIGNAL"] >= group["TEST_START"]).all())
            self.assertTrue((group["FECHA_SIGNAL"] <= group["TEST_END"]).all())

    def test_costs_reduce_expectancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ejecutar_walkforward_v2_gold(
                tmp,
                oof_predictions=_sample_oof(),
                broker_metadata=_metadata(),
                costs=[0, 5],
                capitals=[1000],
                n_folds=3,
            )

        summary = result["summary"]
        base = summary[
            (~summary["BROKER_AWARE"])
            & (summary["CAPITAL_INICIAL"] == 1000)
            & (summary["SL_TP_MODE"] == ATR_ONLY)
        ]
        cost0 = float(base[base["COST_BPS"] == 0]["MEAN_EXPECTANCY"].iloc[0])
        cost5 = float(base[base["COST_BPS"] == 5]["MEAN_EXPECTANCY"].iloc[0])
        self.assertGreater(cost0, cost5)

    def test_broker_skip_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            result_1 = ejecutar_walkforward_v2_gold(
                tmp1,
                oof_predictions=_sample_oof(),
                broker_metadata=_metadata(volume_min=1.0),
                costs=[2],
                capitals=[100],
                n_folds=3,
            )
            result_2 = ejecutar_walkforward_v2_gold(
                tmp2,
                oof_predictions=_sample_oof(),
                broker_metadata=_metadata(volume_min=1.0),
                costs=[2],
                capitals=[100],
                n_folds=3,
            )

        pd.testing.assert_frame_equal(result_1["summary"], result_2["summary"])
        broker = result_1["summary"][result_1["summary"]["BROKER_AWARE"]]
        self.assertTrue((broker["NO_EJECUTABLES_BROKER_MIN"] > 0).all())

    def test_paper_files_are_not_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "paper"
            paper.mkdir()
            files = [
                paper / "state.json",
                paper / "trades.csv",
                paper / "signals.csv",
            ]
            for file in files:
                file.write_text(f"frozen:{file.name}", encoding="utf-8")
            before = {file: _hash(file) for file in files}

            ejecutar_walkforward_v2_gold(
                root,
                oof_predictions=_sample_oof(),
                broker_metadata=_metadata(),
                costs=[2],
                capitals=[1000],
                n_folds=3,
            )

            after = {file: _hash(file) for file in files}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
