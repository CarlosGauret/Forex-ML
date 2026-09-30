import unittest

import pandas as pd

from src.market_structure import (
    LAST_SWING_HIGH,
    LAST_SWING_LOW,
    SWING_HIGH_CONFIRMED,
    SWING_HIGH_PRICE,
    SWING_LOW_CONFIRMED,
    SWING_LOW_PRICE,
    detectar_swings_confirmados,
)


def _df(highs, lows):
    return pd.DataFrame(
        {
            "High": highs,
            "Low": lows,
            "Open": highs,
            "Close": lows,
        },
        index=pd.date_range("2026-01-01", periods=len(highs), freq="h"),
    )


class MarketStructureTests(unittest.TestCase):
    def test_swing_high_confirmado_en_vela_de_confirmacion(self):
        datos = detectar_swings_confirmados(
            _df([1, 2, 5, 3, 2], [5, 4, 3, 4, 5]),
            left_bars=1,
            right_bars=2,
        )

        self.assertFalse(bool(datos.iloc[2][SWING_HIGH_CONFIRMED]))
        self.assertFalse(bool(datos.iloc[3][SWING_HIGH_CONFIRMED]))
        self.assertTrue(bool(datos.iloc[4][SWING_HIGH_CONFIRMED]))
        self.assertEqual(float(datos.iloc[4][SWING_HIGH_PRICE]), 5.0)

    def test_swing_low_confirmado_en_vela_de_confirmacion(self):
        datos = detectar_swings_confirmados(
            _df([5, 4, 3, 4, 5], [5, 4, 1, 3, 4]),
            left_bars=1,
            right_bars=2,
        )

        self.assertFalse(bool(datos.iloc[2][SWING_LOW_CONFIRMED]))
        self.assertFalse(bool(datos.iloc[3][SWING_LOW_CONFIRMED]))
        self.assertTrue(bool(datos.iloc[4][SWING_LOW_CONFIRMED]))
        self.assertEqual(float(datos.iloc[4][SWING_LOW_PRICE]), 1.0)

    def test_no_lookahead_no_expone_ultimo_swing_antes_de_confirmacion(self):
        datos = detectar_swings_confirmados(
            _df([1, 2, 5, 3, 2], [5, 4, 3, 4, 5]),
            left_bars=1,
            right_bars=2,
        )

        self.assertTrue(pd.isna(datos.iloc[2][LAST_SWING_HIGH]))
        self.assertTrue(pd.isna(datos.iloc[3][LAST_SWING_HIGH]))
        self.assertEqual(float(datos.iloc[4][LAST_SWING_HIGH]), 5.0)

    def test_requiere_pivote_estricto(self):
        datos = detectar_swings_confirmados(
            _df([1, 5, 5, 3, 2], [5, 3, 3, 4, 5]),
            left_bars=1,
            right_bars=1,
        )

        self.assertFalse(datos[SWING_HIGH_CONFIRMED].any())
        self.assertFalse(datos[SWING_LOW_CONFIRMED].any())

    def test_last_swing_low_se_propaga_solo_despues_de_confirmacion(self):
        datos = detectar_swings_confirmados(
            _df([5, 4, 3, 4, 5, 6], [5, 4, 1, 3, 4, 5]),
            left_bars=1,
            right_bars=2,
        )

        self.assertTrue(pd.isna(datos.iloc[3][LAST_SWING_LOW]))
        self.assertEqual(float(datos.iloc[4][LAST_SWING_LOW]), 1.0)
        self.assertEqual(float(datos.iloc[5][LAST_SWING_LOW]), 1.0)


if __name__ == "__main__":
    unittest.main()
