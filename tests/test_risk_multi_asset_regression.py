import unittest

from src.risk import BUY, SELL, DECISION_SKIP, RiskLimits, RiskManager


ASSET_CASES = {
    "GOLD": {
        "entry": 2350.00,
        "buy_stop": 2345.00,
        "sell_stop": 2355.00,
        "tick_size": 0.01,
        "tick_value": 0.01,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 50.0,
    },
    "EURUSD": {
        "entry": 1.10000,
        "buy_stop": 1.09800,
        "sell_stop": 1.10200,
        "tick_size": 0.00001,
        "tick_value": 1.0,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
    },
    "GBPUSD": {
        "entry": 1.27000,
        "buy_stop": 1.26750,
        "sell_stop": 1.27250,
        "tick_size": 0.00001,
        "tick_value": 1.0,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
    },
    "USDJPY": {
        "entry": 153.300995,
        "buy_stop": 153.004707,
        "sell_stop": 153.597283,
        "tick_size": 0.001,
        "tick_value": 0.636,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
    },
}


def _symbol_info(case, aliases=False):
    if aliases:
        return {
            "volume_min": case["volume_min"],
            "volume_max": case["volume_max"],
            "volume_step": case["volume_step"],
            "tick_size": case["tick_size"],
            "tick_value": case["tick_value"],
        }
    return {
        "volume_min": case["volume_min"],
        "volume_max": case["volume_max"],
        "volume_step": case["volume_step"],
        "trade_tick_size": case["tick_size"],
        "trade_tick_value": case["tick_value"],
    }


def _fake_order_calc_profit_factory(case):
    def fake_order_calc_profit(order_type, symbol, volume, entry, stop_loss):
        del symbol
        is_buy = order_type in (BUY, 0)
        signed_distance = (stop_loss - entry) if is_buy else (entry - stop_loss)
        money = abs(signed_distance) / case["tick_size"] * case["tick_value"] * volume
        return -money

    return fake_order_calc_profit


class RiskMultiAssetRegressionTests(unittest.TestCase):
    def setUp(self):
        self.manager = RiskManager(RiskLimits(risk_per_trade=0.01))

    def test_order_calc_profit_matches_fallback_for_buy_and_sell(self):
        # The fallback uses tick_size/tick_value, so with static tick values it should
        # match an order_calc_profit implementation based on the same broker metadata.
        for asset, case in ASSET_CASES.items():
            for direction, stop, order_type in (
                (BUY, case["buy_stop"], 0),
                (SELL, case["sell_stop"], 1),
            ):
                with self.subTest(asset=asset, direction=direction):
                    fallback = self.manager.calculate_position_size(
                        symbol=asset,
                        direction=direction,
                        entry=case["entry"],
                        stop_loss=stop,
                        balance=1000,
                        symbol_info=_symbol_info(case),
                    )
                    order_calc = self.manager.calculate_position_size(
                        symbol=asset,
                        direction=direction,
                        entry=case["entry"],
                        stop_loss=stop,
                        balance=1000,
                        symbol_info=_symbol_info(case),
                        order_calc_profit=_fake_order_calc_profit_factory(case),
                        order_type=order_type,
                    )

                    self.assertEqual(order_calc.loss_source, "MT5_ORDER_CALC_PROFIT")
                    self.assertAlmostEqual(order_calc.min_volume_loss, fallback.min_volume_loss, places=9)
                    self.assertAlmostEqual(order_calc.estimated_loss, fallback.estimated_loss, places=9)

    def test_tick_size_tick_value_aliases_are_supported(self):
        for asset, case in ASSET_CASES.items():
            with self.subTest(asset=asset):
                result = self.manager.calculate_position_size(
                    symbol=asset,
                    direction=BUY,
                    entry=case["entry"],
                    stop_loss=case["buy_stop"],
                    balance=1000,
                    symbol_info=_symbol_info(case, aliases=True),
                )

                self.assertEqual(result.loss_source, "SYMBOL_TICK_VALUE")
                self.assertIsNotNone(result.min_volume_loss)

    def test_no_rounding_up_when_volume_required_is_below_minimum(self):
        case = ASSET_CASES["GOLD"]
        result = self.manager.calculate_position_size(
            symbol="GOLD",
            direction=BUY,
            entry=case["entry"],
            stop_loss=case["buy_stop"],
            balance=1,
            symbol_info=_symbol_info(case),
            order_calc_profit=_fake_order_calc_profit_factory(case),
            order_type=0,
        )

        self.assertEqual(result.decision, DECISION_SKIP)
        self.assertEqual(result.reason, "MINIMUM_VOLUME_EXCEEDS_RISK")
        self.assertLess(result.theoretical_volume, case["volume_min"])
        self.assertIsNone(result.volume)

    def test_volume_step_rounds_down_for_all_assets(self):
        for asset, case in ASSET_CASES.items():
            with self.subTest(asset=asset):
                result = self.manager.calculate_position_size(
                    symbol=asset,
                    direction=BUY,
                    entry=case["entry"],
                    stop_loss=case["buy_stop"],
                    balance=1234,
                    symbol_info=_symbol_info(case),
                    order_calc_profit=_fake_order_calc_profit_factory(case),
                    order_type=0,
                )

                self.assertIsNotNone(result.volume)
                ratio = round(result.volume / case["volume_step"])
                self.assertAlmostEqual(result.volume, ratio * case["volume_step"], places=9)
                self.assertLessEqual(result.estimated_loss, result.risk_amount)


if __name__ == "__main__":
    unittest.main()
