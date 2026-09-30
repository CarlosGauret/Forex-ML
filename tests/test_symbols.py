import unittest
from types import SimpleNamespace

from src.symbols import resolve_symbol


def sym(name, description="", **extra):
    data = {
        "name": name,
        "description": description,
        "digits": 5,
        "point": 0.00001,
        "trade_contract_size": 100000,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "visible": True,
    }
    data.update(extra)
    return SimpleNamespace(**data)


class SymbolMappingTests(unittest.TestCase):
    def test_exact_symbol_mapping(self):
        result = resolve_symbol("EURUSD", [sym("EURUSD")])

        self.assertTrue(result.found)
        self.assertEqual(result.logical_symbol, "EURUSD")
        self.assertEqual(result.mt5_symbol, "EURUSD")
        self.assertEqual(result.volume_step, 0.01)

    def test_suffix_symbol_mapping(self):
        result = resolve_symbol("EURUSD", [sym("GBPUSDm"), sym("EURUSDm")])

        self.assertTrue(result.found)
        self.assertEqual(result.mt5_symbol, "EURUSDm")

    def test_gold_mapping_by_xauusd_or_description(self):
        result = resolve_symbol(
            "GOLD",
            [sym("XAUUSDm", "Gold vs US Dollar", digits=2, point=0.01)],
        )

        self.assertTrue(result.found)
        self.assertEqual(result.mt5_symbol, "XAUUSDm")
        self.assertEqual(result.digits, 2)
        self.assertEqual(result.point, 0.01)

    def test_missing_symbol(self):
        result = resolve_symbol("NZDUSD", [sym("EURUSDm")])

        self.assertFalse(result.found)
        self.assertEqual(result.reason, "SYMBOL_NOT_FOUND")
        self.assertEqual(result.mt5_symbol, "")


if __name__ == "__main__":
    unittest.main()
