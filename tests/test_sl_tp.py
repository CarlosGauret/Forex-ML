import unittest

from src.sl_tp import (
    ATR_ONLY,
    LONG,
    SHORT,
    SWING_ATR_RR,
    SWING_STRUCTURAL_TARGET,
    calcular_sl_tp,
)


class SLTPTests(unittest.TestCase):
    def test_atr_only_long_sl_tp(self):
        result = calcular_sl_tp(100.0, LONG, 2.0, mode=ATR_ONLY)

        self.assertTrue(result.ok)
        self.assertEqual(result.stop_loss, 98.0)
        self.assertEqual(result.take_profit, 104.0)
        self.assertEqual(result.rr, 2.0)

    def test_atr_only_short_sl_tp(self):
        result = calcular_sl_tp(100.0, SHORT, 2.0, mode=ATR_ONLY)

        self.assertTrue(result.ok)
        self.assertEqual(result.stop_loss, 102.0)
        self.assertEqual(result.take_profit, 96.0)
        self.assertEqual(result.rr, 2.0)

    def test_swing_atr_rr_long(self):
        result = calcular_sl_tp(
            100.0,
            LONG,
            1.0,
            mode=SWING_ATR_RR,
            swing_low=97.0,
            rr=2.0,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.stop_loss, 96.0)
        self.assertEqual(result.risk_distance, 4.0)
        self.assertEqual(result.take_profit, 108.0)
        self.assertEqual(result.rr, 2.0)

    def test_swing_atr_rr_short(self):
        result = calcular_sl_tp(
            100.0,
            SHORT,
            1.0,
            mode=SWING_ATR_RR,
            swing_high=103.0,
            rr=2.0,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.stop_loss, 104.0)
        self.assertEqual(result.risk_distance, 4.0)
        self.assertEqual(result.take_profit, 92.0)

    def test_structural_target_long(self):
        result = calcular_sl_tp(
            100.0,
            LONG,
            1.0,
            mode=SWING_STRUCTURAL_TARGET,
            swing_low=97.0,
            structural_target=107.0,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.stop_loss, 96.0)
        self.assertEqual(result.take_profit, 107.0)
        self.assertEqual(result.rr, 1.75)

    def test_structural_target_short(self):
        result = calcular_sl_tp(
            100.0,
            SHORT,
            1.0,
            mode=SWING_STRUCTURAL_TARGET,
            swing_high=103.0,
            structural_target=93.0,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.stop_loss, 104.0)
        self.assertEqual(result.take_profit, 93.0)
        self.assertEqual(result.rr, 1.75)

    def test_missing_structural_stop_waits(self):
        result = calcular_sl_tp(100.0, LONG, 1.0, mode=SWING_ATR_RR)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "STRUCTURAL_STOP_NOT_AVAILABLE")


if __name__ == "__main__":
    unittest.main()
