import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import paper_trader as pt


CONFIG = {
    "CONFIG_ID": "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060",
    "ACTIVO": "GOLD",
    "DIRECCION": "LONG",
    "MODELO": "LOGISTIC",
    "SISTEMA": "BASE_PLUS_ML",
    "THRESHOLD": 0.60,
    "TIMEFRAME": "1h",
    "STOP_ATR": 1,
    "TAKE_PROFIT_ATR": 2,
    "MAX_HOLD_BARS": 24,
    "RIESGO_POR_OPERACION": 0.01,
}


def _datos(n=10, signal_at=None):
    index = pd.date_range("2026-01-01 00:00:00", periods=n, freq="h")
    datos = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [105.0] * n,
            "Low": [95.0] * n,
            "Close": [100.0] * n,
            "Volume": [1.0] * n,
            "EMA20": [100.0] * n,
            "EMA50": [95.0] * n,
            "EMA200": [90.0] * n,
            "RSI": [40.0] * n,
            "ATR": [10.0] * n,
            "RET_1H": [0.0] * n,
            "RET_4H": [0.0] * n,
            "RET_24H": [0.0] * n,
            "VOLATILIDAD": [0.01] * n,
            "TEST_PROB": [0.1] * n,
        },
        index=index,
    )

    if signal_at is not None:
        datos.iloc[signal_at, datos.columns.get_loc("RSI")] = 60.0
        datos.iloc[signal_at, datos.columns.get_loc("TEST_PROB")] = 0.9

    return datos


class PaperCatchupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)
        self.rutas = pt._asegurar_archivos(self.raiz, inicializar_start=False)
        pd.DataFrame(
            [
                {
                    "FECHA": "2026-01-01T00:00:00",
                    "CONFIG_ID": CONFIG["CONFIG_ID"],
                    "CAPITAL": 1000.0,
                }
            ],
            columns=pt.EQUITY_COLUMNS,
        ).to_csv(self.rutas["equity"], index=False)

        self.state = {"last_processed_bar": {}, "pending_signals": []}
        self.start_time = pd.Timestamp("2026-01-01 00:00:00")
        self.patches = [
            patch.object(pt, "PAPER_CONFIGS", [CONFIG]),
            patch.object(
                pt,
                "_modelo_estado",
                return_value={"cargado": True, "ruta": "mock", "causa": ""},
            ),
            patch.object(
                pt,
                "_probabilidad_long",
                side_effect=lambda raiz, config, fila, start_time: (
                    float(fila.get("TEST_PROB", 0.1)),
                    "",
                ),
            ),
            patch.object(pt, "_enviar_telegram_unico", return_value=None),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tmp.cleanup()

    def _run(self, datos):
        return pt._procesar_catchup_config(
            CONFIG,
            datos,
            self.rutas,
            self.state,
            self.start_time,
            self.raiz,
        )

    def test_10_pending_bars_without_signals(self):
        resultado = self._run(_datos(10))

        signals = pd.read_csv(self.rutas["signals"])
        trades = pd.read_csv(self.rutas["trades"])

        self.assertEqual(resultado["velas_pendientes"], 10)
        self.assertEqual(resultado["velas_procesadas_catchup"], 10)
        self.assertEqual(len(signals), 10)
        self.assertTrue((signals["SIGNAL"] == "WAIT").all())
        self.assertTrue(trades.empty)

    def test_1_pending_bar_without_signal(self):
        resultado = self._run(_datos(1))
        signals = pd.read_csv(self.rutas["signals"])

        self.assertEqual(resultado["velas_pendientes"], 1)
        self.assertEqual(resultado["velas_procesadas_catchup"], 1)
        self.assertEqual(len(signals), 1)

    def test_5_pending_bars_without_signals(self):
        resultado = self._run(_datos(5))
        signals = pd.read_csv(self.rutas["signals"])

        self.assertEqual(resultado["velas_pendientes"], 5)
        self.assertEqual(resultado["velas_procesadas_catchup"], 5)
        self.assertEqual(len(signals), 5)

    def test_missing_middle_bars_are_not_hidden_by_last_processed(self):
        datos = _datos(5)
        existing = []
        for index in (0, 4):
            fecha = datos.index[index]
            fila = datos.iloc[index]
            existing.append(
                pt._crear_signal_row(
                    CONFIG,
                    fecha,
                    fila,
                    "WAIT",
                    0.1,
                    False,
                )
            )
        pd.DataFrame(existing, columns=pt.SIGNALS_COLUMNS).to_csv(
            self.rutas["signals"],
            index=False,
        )
        self.state["last_processed_bar"][CONFIG["CONFIG_ID"]] = datos.index[4].isoformat()

        resultado = self._run(datos)
        signals = pd.read_csv(self.rutas["signals"])
        processed = self.state["processed_bar_timestamps"][CONFIG["CONFIG_ID"]]

        self.assertEqual(resultado["velas_pendientes"], 3)
        self.assertEqual(resultado["timestamps_procesados_catchup"], [
            datos.index[i].isoformat() for i in (1, 2, 3)
        ])
        self.assertEqual(len(signals), 5)
        self.assertEqual(len(processed), 5)

    def test_signal_entry_close_and_continue(self):
        datos = _datos(10, signal_at=1)
        datos.iloc[4, datos.columns.get_loc("High")] = 121.0

        resultado = self._run(datos)
        trades = pd.read_csv(self.rutas["trades"])
        signals = pd.read_csv(self.rutas["signals"])

        self.assertEqual(resultado["velas_procesadas_catchup"], 10)
        self.assertEqual(resultado["trades_abiertos_catchup"], 1)
        self.assertEqual(resultado["trades_cerrados_catchup"], 1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["RESULTADO"], "WIN")
        self.assertEqual(signals.iloc[-1]["FECHA_SIGNAL"], datos.index[-1].isoformat())

    def test_open_position_closes_by_tp_and_continue(self):
        datos = _datos(10)
        datos.iloc[4, datos.columns.get_loc("High")] = 121.0
        self.state["last_processed_bar"][CONFIG["CONFIG_ID"]] = datos.index[1].isoformat()
        pd.DataFrame(
            [
                {
                    "CONFIG_ID": CONFIG["CONFIG_ID"],
                    "ACTIVO": "GOLD",
                    "DIRECCION": "LONG",
                    "MODELO": "LOGISTIC",
                    "SISTEMA": "BASE_PLUS_ML",
                    "THRESHOLD": 0.60,
                    "FECHA_SIGNAL": datos.index[0].isoformat(),
                    "FECHA_ENTRADA": datos.index[0].isoformat(),
                    "ENTRY": 100.0,
                    "STOP": 90.0,
                    "TP": 120.0,
                    "ATR_SIGNAL": 10.0,
                    "PROBABILIDAD": 0.9,
                    "COST_BPS": 2,
                    "CAPITAL_ANTES": 1000.0,
                }
            ],
            columns=pt.OPEN_COLUMNS,
        ).to_csv(self.rutas["open_positions"], index=False)

        resultado = self._run(datos)
        trades = pd.read_csv(self.rutas["trades"])

        self.assertEqual(resultado["trades_cerrados_catchup"], 1)
        self.assertEqual(resultado["velas_procesadas_catchup"], 8)
        self.assertEqual(trades.iloc[0]["FECHA_SALIDA"], datos.index[4].isoformat())
        self.assertEqual(
            self.state["last_processed_bar"][CONFIG["CONFIG_ID"]],
            datos.index[-1].isoformat(),
        )

    def test_open_position_closes_by_max_hold(self):
        datos = _datos(30)
        self.state["last_processed_bar"][CONFIG["CONFIG_ID"]] = datos.index[0].isoformat()
        pd.DataFrame(
            [
                {
                    "CONFIG_ID": CONFIG["CONFIG_ID"],
                    "ACTIVO": "GOLD",
                    "DIRECCION": "LONG",
                    "MODELO": "LOGISTIC",
                    "SISTEMA": "BASE_PLUS_ML",
                    "THRESHOLD": 0.60,
                    "FECHA_SIGNAL": datos.index[0].isoformat(),
                    "FECHA_ENTRADA": datos.index[0].isoformat(),
                    "ENTRY": 100.0,
                    "STOP": 80.0,
                    "TP": 140.0,
                    "ATR_SIGNAL": 10.0,
                    "PROBABILIDAD": 0.9,
                    "COST_BPS": 2,
                    "CAPITAL_ANTES": 1000.0,
                }
            ],
            columns=pt.OPEN_COLUMNS,
        ).to_csv(self.rutas["open_positions"], index=False)

        self._run(datos)
        trades = pd.read_csv(self.rutas["trades"])

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["FECHA_SALIDA"], datos.index[23].isoformat())

    def test_signal_on_last_bar_stays_pending(self):
        datos = _datos(10, signal_at=9)

        resultado = self._run(datos)
        trades = pd.read_csv(self.rutas["trades"])

        self.assertEqual(resultado["trades_abiertos_catchup"], 0)
        self.assertTrue(trades.empty)
        self.assertEqual(len(self.state["pending_signals"]), 1)
        self.assertEqual(
            self.state["pending_signals"][0]["FECHA_SIGNAL"],
            datos.index[-1].isoformat(),
        )

    def test_second_run_without_new_bar_is_idempotent(self):
        datos = _datos(10)
        self._run(datos)
        sizes_before = {
            name: self.rutas[name].stat().st_size
            for name in ("signals", "trades", "equity", "open_positions")
        }

        resultado = self._run(datos)
        sizes_after = {
            name: self.rutas[name].stat().st_size
            for name in ("signals", "trades", "equity", "open_positions")
        }

        self.assertEqual(resultado["velas_pendientes"], 0)
        self.assertEqual(resultado["velas_procesadas_catchup"], 0)
        self.assertEqual(sizes_before, sizes_after)

    def test_stop_loss_is_conservative_when_sl_and_tp_same_bar(self):
        datos = _datos(5)
        datos.iloc[2, datos.columns.get_loc("High")] = 121.0
        datos.iloc[2, datos.columns.get_loc("Low")] = 89.0
        self.state["last_processed_bar"][CONFIG["CONFIG_ID"]] = datos.index[0].isoformat()
        pd.DataFrame(
            [
                {
                    "CONFIG_ID": CONFIG["CONFIG_ID"],
                    "ACTIVO": "GOLD",
                    "DIRECCION": "LONG",
                    "MODELO": "LOGISTIC",
                    "SISTEMA": "BASE_PLUS_ML",
                    "THRESHOLD": 0.60,
                    "FECHA_SIGNAL": datos.index[0].isoformat(),
                    "FECHA_ENTRADA": datos.index[0].isoformat(),
                    "ENTRY": 100.0,
                    "STOP": 90.0,
                    "TP": 120.0,
                    "ATR_SIGNAL": 10.0,
                    "PROBABILIDAD": 0.9,
                    "COST_BPS": 2,
                    "CAPITAL_ANTES": 1000.0,
                }
            ],
            columns=pt.OPEN_COLUMNS,
        ).to_csv(self.rutas["open_positions"], index=False)

        self._run(datos)
        trades = pd.read_csv(self.rutas["trades"])

        self.assertEqual(trades.iloc[0]["RESULTADO"], "LOSS")
        self.assertEqual(float(trades.iloc[0]["R_BRUTO"]), -1.0)


if __name__ == "__main__":
    unittest.main()
