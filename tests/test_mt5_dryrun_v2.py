import unittest
from types import SimpleNamespace

from src.mt5_connector import evaluar_dryrun_activo_v2
from src.symbols import ResolvedSymbol


def resolved(found=True):
    return ResolvedSymbol(
        logical_symbol="EURUSD",
        mt5_symbol="EURUSDm" if found else "",
        description="Euro vs US Dollar",
        digits=5,
        point=0.00001,
        contract_size=100000,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        found=found,
        reason="OK" if found else "SYMBOL_NOT_FOUND",
    )


def info():
    return SimpleNamespace(
        name="EURUSDm",
        description="Euro vs US Dollar",
        digits=5,
        point=0.00001,
        trade_contract_size=100000,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_stops_level=0,
        trade_freeze_level=0,
    )


class MT5DryrunV2Tests(unittest.TestCase):
    def test_model_not_validated_waits_and_sends_zero_orders(self):
        result = evaluar_dryrun_activo_v2(
            activo="EURUSD",
            resolved_symbol=resolved(),
            info=info(),
            tick=SimpleNamespace(bid=1.1, ask=1.1002),
            balance=1000.0,
            atr_data={"atr": 0.001, "fuente": "TEST", "fecha": "2026-01-01"},
            timestamp="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(result["decision"], "ESPERAR")
        self.assertEqual(result["reason"], "MODELO_VALIDADO_NO")
        self.assertFalse(result["modelo_validado"])
        self.assertEqual(result["ordenes_enviadas"], 0)
        self.assertFalse(result["trading_enabled"])

    def test_missing_symbol_is_data_skip(self):
        result = evaluar_dryrun_activo_v2(
            activo="EURUSD",
            resolved_symbol=resolved(found=False),
            info=None,
            tick=None,
            balance=1000.0,
            timestamp="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(result["decision"], "SKIP - DATOS INSUFICIENTES")
        self.assertEqual(result["reason"], "SYMBOL_NOT_FOUND")
        self.assertEqual(result["ordenes_enviadas"], 0)


if __name__ == "__main__":
    unittest.main()
