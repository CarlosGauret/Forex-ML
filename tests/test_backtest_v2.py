import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtest_v2 import (
    BrokerMetadata,
    _aplicar_costos,
    _base_temporal,
    _calcular_sl_tp_para_entrada,
    _max_drawdown,
    _resolver_salida,
    construir_entradas_base,
    ejecutar_backtest_v2_gold,
)
from src.sl_tp import ATR_ONLY, SWING_ATR_RR


CONFIG = {
    "CONFIG_ID": "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060",
    "ACTIVO": "GOLD",
    "DIRECCION": "LONG",
    "MODELO": "LOGISTIC",
    "SISTEMA": "BASE_PLUS_ML",
    "THRESHOLD": 0.60,
    "MAX_HOLD_BARS": 24,
}


def _sample_oof(rows=50):
    fechas = pd.date_range("2025-01-01", periods=rows, freq="h")
    data = []
    for i, fecha in enumerate(fechas):
        price = 100.0 + (i * 0.2)
        high = price + 0.6
        low = price - 0.6
        if i == 4:
            low = 90.0
        if i == 7:
            high = 120.0
        if i == 12:
            high = 113.5
            low = 111.0
        if i == 36:
            low = 108.0
            high = 118.0
        base_signal = "BUY" if i in (10, 35) else "WAIT"
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


class BacktestV2Tests(unittest.TestCase):
    def test_same_entries_between_modes(self):
        datos = _base_temporal(_sample_oof(), CONFIG)
        entradas = construir_entradas_base(datos, CONFIG)

        self.assertEqual([item["ENTRY_ID"] for item in entradas], [1, 2])
        self.assertEqual(entradas[0]["FECHA_ENTRADA"], datos.iloc[11]["FECHA"])

    def test_swing_confirmado_without_lookahead(self):
        datos = _base_temporal(_sample_oof(), CONFIG)
        entrada = construir_entradas_base(datos, CONFIG)[0]

        sltp, swing_low, swing_high, structural_target, anti_ok = (
            _calcular_sl_tp_para_entrada(datos, entrada, SWING_ATR_RR)
        )

        self.assertTrue(sltp.ok)
        self.assertEqual(swing_low, 90.0)
        self.assertEqual(swing_high, 120.0)
        self.assertEqual(structural_target, 120.0)
        self.assertTrue(anti_ok)

    def test_same_bar_sl_tp_uses_stop_first(self):
        datos = _base_temporal(_sample_oof(), CONFIG)
        entrada = construir_entradas_base(datos, CONFIG)[0]
        sltp, *_ = _calcular_sl_tp_para_entrada(datos, entrada, ATR_ONLY)
        datos.loc[entrada["ENTRY_INDEX"], "LOW"] = sltp.stop_loss - 0.1
        datos.loc[entrada["ENTRY_INDEX"], "HIGH"] = sltp.take_profit + 0.1

        _, exit_price, motivo = _resolver_salida(datos, entrada, sltp, 24)

        self.assertEqual(motivo, "SL")
        self.assertEqual(exit_price, sltp.stop_loss)

    def test_costs_reduce_r(self):
        r_neto, cost_r = _aplicar_costos(100.0, 102.0, 2.0, 1.0, 1)

        self.assertAlmostEqual(cost_r, 0.0202)
        self.assertAlmostEqual(r_neto, 1.9798)

    def test_drawdown(self):
        self.assertAlmostEqual(_max_drawdown([1000, 1100, 990, 1200]), 10.0)

    def test_broker_minimum_capitals_and_reproducibility(self):
        metadata = BrokerMetadata(
            symbol="XAUUSD",
            contract_size=100.0,
            volume_min=0.10,
            volume_max=50.0,
            volume_step=0.01,
            point=0.01,
            digits=2,
            source="TEST",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_1 = ejecutar_backtest_v2_gold(
                root,
                oof_predictions=_sample_oof(),
                broker_metadata=metadata,
                costs=[0, 2],
                capitals=[100, 5000],
            )
            result_2 = ejecutar_backtest_v2_gold(
                root,
                oof_predictions=_sample_oof(),
                broker_metadata=metadata,
                costs=[0, 2],
                capitals=[100, 5000],
            )
            self.assertTrue(result_1["ruta_summary"].exists())
            self.assertTrue(result_1["ruta_trades"].exists())

            summary_1 = result_1["summary"]
            summary_2 = result_2["summary"]
        pd.testing.assert_frame_equal(summary_1, summary_2)

        broker_100 = summary_1[
            (summary_1["BROKER_AWARE"])
            & (summary_1["CAPITAL_INICIAL"] == 100)
            & (summary_1["SL_TP_MODE"] == ATR_ONLY)
        ]
        self.assertTrue((broker_100["NO_EJECUTABLES_BROKER_MIN"] > 0).any())

        no_broker = summary_1[
            (~summary_1["BROKER_AWARE"])
            & (summary_1["CAPITAL_INICIAL"] == 5000)
            & (summary_1["SL_TP_MODE"] == ATR_ONLY)
        ]
        self.assertTrue((no_broker["TOTAL_TRADES"] > 0).all())


if __name__ == "__main__":
    unittest.main()
