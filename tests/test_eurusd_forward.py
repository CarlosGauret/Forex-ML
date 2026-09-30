import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import DEMO_EXECUTION_ENABLED
from src import eurusd_forward as ef


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability=0.1):
        self.probability = probability

    def predict_proba(self, x):
        return np.array([[1.0 - self.probability, self.probability]] * len(x))


def make_history(root, periods=340):
    data_dir = Path(root) / "data" / "eurusd"
    data_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-01-01 00:00:00+00:00", periods=periods, freq="h")
    rng = np.random.default_rng(7)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0007, periods))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.0012
    low = np.minimum(open_, close) - 0.0012
    data = pd.DataFrame(
        {
            "fecha": index,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 100,
        }
    )
    data.to_csv(data_dir / "eurusd_h1.csv", index=False)
    return data


def write_start(root, start="2026-01-13T00:00:00+00:00"):
    paper_dir = Path(root) / "paper" / "eurusd"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "start.json").write_text(
        json.dumps(
            {
                "asset": "EURUSD",
                "timeframe": "H1",
                "direction": "LONG",
                "model": "RF_ACTUAL",
                "system": "ML_ONLY",
                "threshold": 0.65,
                "start_time_utc": start,
            }
        ),
        encoding="utf-8",
    )


def fake_features(start="2026-01-13 00:00:00", periods=8):
    dates = pd.date_range(start, periods=periods, freq="h")
    data = pd.DataFrame(
        {
            "fecha": dates,
            "Open": np.linspace(1.10, 1.11, periods),
            "High": np.linspace(1.101, 1.111, periods),
            "Low": np.linspace(1.099, 1.109, periods),
            "Close": np.linspace(1.1005, 1.1105, periods),
            "ATR": [0.001] * periods,
        }
    )
    for feature in ef.FEATURES_ML:
        data[feature] = 0.1
    return data


def fake_context(features=None, status="LIVE"):
    features = features if features is not None else fake_features()
    return {
        "ok": status == "LIVE",
        "prices": pd.DataFrame(),
        "features": features if status == "LIVE" else pd.DataFrame(),
        "broker": ef._fallback_broker_metadata() if status == "LIVE" else None,
        "diagnostics": {
            "data_source": "MT5 XM",
            "data_status": status,
            "utc_now": "2026-01-13T09:30:00+00:00",
            "last_mt5_bar": "2026-01-13T09:00:00",
            "last_closed_bar": "2026-01-13T08:00:00",
            "first_full_forward_bar": "2026-01-13T00:00:00",
            "pending_bars": len(features) if status == "LIVE" else 0,
            "error": "" if status == "LIVE" else "DATA_STALE",
            "symbol": "EURUSD",
        },
    }


class EurusdForwardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_history(self.root)
        write_start(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_freeze_creates_separate_model_and_metadata_without_lookahead(self):
        dates = pd.date_range("2026-01-01", periods=80, freq="h")
        dataset = pd.DataFrame({feature: np.linspace(0.1, 1.0, len(dates)) for feature in ef.FEATURES_ML})
        dataset["fecha"] = dates
        dataset["TARGET_LONG"] = [0, 1] * 40

        with patch.object(ef, "_training_dataset", return_value=dataset):
            result = ef.freeze_eurusd_forward_model(self.root)
        metadata = result["metadata"]

        self.assertTrue(result["path"].exists())
        self.assertEqual(result["path"].name, "eurusd_long_rf_actual_t065.pkl")
        self.assertEqual(metadata["activo"], "EURUSD")
        self.assertEqual(metadata["direccion"], "LONG")
        self.assertEqual(metadata["threshold"], 0.65)
        self.assertEqual(metadata["features"], ef.FEATURES_ML)
        self.assertLess(
            pd.to_datetime(metadata["fin_datos_entrenamiento"], utc=True),
            pd.to_datetime(metadata["forward_start"], utc=True),
        )
        self.assertFalse((self.root / "paper" / "start.json").exists())

    def test_threshold_and_demo_flags_are_fixed(self):
        self.assertEqual(ef.THRESHOLD, 0.65)
        self.assertIs(DEMO_EXECUTION_ENABLED, False)

    def test_closed_bars_excludes_current_forming_bar(self):
        now = "2026-01-15T03:30:00+00:00"
        prices = pd.DataFrame(
            {
                "Open": [1.0] * 260,
                "High": [1.1] * 260,
                "Low": [0.9] * 260,
                "Close": [1.0] * 260,
                "Volume": [1] * 260,
            },
            index=pd.date_range("2026-01-04 08:00:00", periods=260, freq="h"),
        )

        with patch.object(ef, "_mt5_h1_prices", return_value=(prices, ef._fallback_broker_metadata(), "EURUSD")):
            context = ef._mt5_data_context(self.root, now=now)

        self.assertEqual(context["diagnostics"]["last_mt5_bar"], "2026-01-15T03:00:00")
        self.assertEqual(context["diagnostics"]["last_closed_bar"], "2026-01-15T02:00:00")
        self.assertTrue((context["features"]["fecha"] <= pd.Timestamp("2026-01-15 02:00:00")).all())

    def test_catchup_is_idempotent_without_duplicate_waits(self):
        with (
            patch.object(ef, "_assert_frozen_model", return_value={"path": Path("x"), "metadata": {}}),
            patch.object(ef, "_load_model", return_value=FakeModel(probability=0.1)),
            patch.object(ef, "_mt5_data_context", return_value=fake_context()),
        ):
            first = ef.process_eurusd_forward(self.root, now="2026-01-13T06:00:00+00:00")
            signals_before = pd.read_csv(self.root / "paper" / "eurusd" / "signals.csv")
            second = ef.process_eurusd_forward(self.root, now="2026-01-13T06:00:00+00:00")
            signals_after = pd.read_csv(self.root / "paper" / "eurusd" / "signals.csv")

        self.assertGreater(first["processed"], 0)
        self.assertEqual(second["processed"], 0)
        self.assertEqual(len(signals_before), len(signals_after))
        self.assertTrue((signals_after["SIGNAL"] == "WAIT").all())

    def test_missing_middle_bars_are_not_hidden_by_last_processed(self):
        features = fake_features(periods=5)
        paths = ef._ensure_files(self.root)
        existing = []
        for index in (0, 4):
            row = features.iloc[index]
            existing.append(
                {
                    "SIGNAL_ID": ef._signal_id(row["fecha"]),
                    "CONFIG_ID": ef.CONFIG_ID,
                    "ACTIVO": ef.ASSET,
                    "DIRECCION": ef.DIRECTION,
                    "MODELO": ef.MODEL_NAME,
                    "SISTEMA": ef.SYSTEM,
                    "THRESHOLD": ef.THRESHOLD,
                    "FECHA_SIGNAL": pd.to_datetime(row["fecha"]).isoformat(),
                    "SIGNAL": "WAIT",
                    "EXECUTION": "WAIT",
                    "PROBABILIDAD": 0.1,
                    "ENTRY": None,
                    "STOP": None,
                    "TP": None,
                    "ATR": row["ATR"],
                    "VOLUME": None,
                    "VOLUME_MIN": None,
                    "RISK_USD": None,
                    "REASON": "PROBABILITY_BELOW_THRESHOLD",
                }
            )
        pd.DataFrame(existing, columns=ef.SIGNALS_COLUMNS).to_csv(paths["signals"], index=False)
        state = ef._read_json(paths["state"], {})
        state["last_processed_bar"] = pd.to_datetime(features.iloc[4]["fecha"]).isoformat()
        ef._write_json(paths["state"], state)

        with (
            patch.object(ef, "_assert_frozen_model", return_value={"path": Path("x"), "metadata": {}}),
            patch.object(ef, "_load_model", return_value=FakeModel(probability=0.1)),
            patch.object(ef, "_mt5_data_context", return_value=fake_context(features)),
        ):
            result = ef.process_eurusd_forward(self.root)

        signals = pd.read_csv(paths["signals"])
        processed = ef._read_json(paths["state"], {})["processed_bar_timestamps"]

        self.assertEqual(result["pending_found"], 3)
        self.assertEqual(result["processed_bar_timestamps"], [
            pd.to_datetime(features.iloc[i]["fecha"]).isoformat() for i in (1, 2, 3)
        ])
        self.assertEqual(len(signals), 5)
        self.assertEqual(len(processed), 5)

    def test_first_full_forward_bar_ceil(self):
        result = ef._first_full_forward_bar("2026-09-29T12:54:32+00:00")

        self.assertEqual(result.isoformat(), "2026-09-29T13:00:00")

    def test_mt5_server_time_is_normalized_to_utc(self):
        prices = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-09-29 16:00:00")]),
        )

        normalized, offset = ef._normalize_mt5_server_time(
            prices,
            now="2026-09-29T13:04:00+00:00",
        )

        self.assertEqual(offset, 3)
        self.assertEqual(normalized.index.max().isoformat(), "2026-09-29T13:00:00")

    def test_warmup_rows_before_forward_do_not_generate_signals(self):
        write_start(self.root, start="2026-01-13T03:30:00+00:00")
        features = fake_features(start="2026-01-13 00:00:00", periods=8)
        with (
            patch.object(ef, "_assert_frozen_model", return_value={"path": Path("x"), "metadata": {}}),
            patch.object(ef, "_load_model", return_value=FakeModel(probability=0.1)),
            patch.object(ef, "_mt5_data_context", return_value=fake_context(features)),
        ):
            result = ef.process_eurusd_forward(self.root)

        signals = pd.read_csv(self.root / "paper" / "eurusd" / "signals.csv")
        self.assertEqual(result["processed"], 4)
        self.assertTrue((pd.to_datetime(signals["FECHA_SIGNAL"]) >= pd.Timestamp("2026-01-13 04:00:00")).all())

    def test_stale_data_fails_closed(self):
        with (
            patch.object(ef, "_assert_frozen_model", return_value={"path": Path("x"), "metadata": {}}),
            patch.object(ef, "_load_model", return_value=FakeModel(probability=0.9)),
            patch.object(ef, "_mt5_data_context", return_value=fake_context(status="STALE")),
        ):
            result = ef.process_eurusd_forward(self.root)

        self.assertEqual(result["processed"], 0)
        self.assertEqual(result["data_status"], "STALE")

    def test_hash_mismatch_aborts(self):
        paths = ef._paths(self.root)
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        paths["model"].write_bytes(b"changed")
        paths["metadata"].write_text(
            json.dumps(
                {
                    "hash_modelo": ef.EXPECTED_MODEL_HASH,
                    "activo": "EURUSD",
                    "direccion": "LONG",
                    "modelo": "RF_ACTUAL",
                    "sistema": "ML_ONLY",
                    "threshold": 0.65,
                    "features": ef.FEATURES_ML,
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            ef._assert_frozen_model(self.root)

    def test_broker_sizing_skips_when_min_volume_exceeds_one_percent(self):
        broker = ef._fallback_broker_metadata()
        broker["volume_min"] = 1.0
        result = ef._sizing(1.1000, 1.0990, 100.0, broker)

        self.assertFalse(result.ok)
        self.assertEqual(result.decision, "SKIP")
        self.assertEqual(result.reason, "MINIMUM_VOLUME_EXCEEDS_RISK")
        self.assertIsNone(result.volume)

    def test_no_order_send_in_eurusd_forward_module(self):
        source = inspect.getsource(ef)

        self.assertNotIn("order_send(", source)


if __name__ == "__main__":
    unittest.main()
