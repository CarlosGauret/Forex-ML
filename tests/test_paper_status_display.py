import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

import main
from config_paper import PAPER_CONFIGS
from src.display import formatear_direccion


class PaperStatusDisplayTests(unittest.TestCase):
    def test_config_id_is_not_human_formatted_in_paper_status(self):
        config = PAPER_CONFIGS[0].copy()
        self.assertEqual(config["CONFIG_ID"], "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060")

        status = {
            "iniciado": True,
            "fecha_inicio": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
            "dias_forward": 1.0,
            "configs": [
                {
                    "config": config,
                    "trades_cerrados": 0,
                    "trades_abiertos": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "expectancy": 0,
                    "r_acumulado": 0,
                    "capital": 1000,
                    "max_drawdown": 0,
                }
            ],
        }

        with patch("src.paper_trader.paper_status", return_value=status):
            salida = io.StringIO()
            with redirect_stdout(salida):
                main.ejecutar_paper_status()

        texto = salida.getvalue()
        self.assertIn("CONFIG_ID: GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060", texto)
        self.assertIn("Direccion: LONG (COMPRAR)", texto)
        self.assertNotIn("GOLD_LONG (COMPRAR)_LOGISTIC", texto)
        self.assertNotIn("(COMPRAR)_LOGISTIC", texto)
        self.assertNotIn("(VENDER)", config["CONFIG_ID"])
        self.assertNotIn("(COMPRAR)", config["CONFIG_ID"])

    def test_direction_labels_are_visual_only(self):
        self.assertEqual(formatear_direccion("LONG"), "LONG (COMPRAR)")
        self.assertEqual(formatear_direccion("SHORT"), "SHORT (VENDER)")

        for config in PAPER_CONFIGS:
            self.assertNotIn("(COMPRAR)", config["CONFIG_ID"])
            self.assertNotIn("(VENDER)", config["CONFIG_ID"])
            self.assertIn(config["DIRECCION"], ["LONG", "SHORT"])


if __name__ == "__main__":
    unittest.main()
