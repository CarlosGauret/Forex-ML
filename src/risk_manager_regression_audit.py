from pathlib import Path

import pandas as pd

from src.mt5_connector import _credenciales_mt5, _modo_cuenta, _valor
from src.risk import BUY, SELL, RiskLimits, RiskManager
from src.symbols import discover_symbols


ASSETS = ("GOLD", "EURUSD", "GBPUSD", "USDJPY")
RISK_PER_TRADE = 0.01
BALANCE = 5000.0
VOLUME = 0.01
DISTANCES = {
    "GOLD": 5.0,
    "EURUSD": 0.0020,
    "GBPUSD": 0.0025,
    "USDJPY": 0.296288,
}


def _root(raiz_proyecto=None):
    return Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]


def _symbol_info_dict(info):
    return {
        "trade_contract_size": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
        "contract_size": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
        "volume_min": _valor(info, "volume_min"),
        "volume_max": _valor(info, "volume_max"),
        "volume_step": _valor(info, "volume_step"),
        "trade_tick_size": _valor(info, "trade_tick_size"),
        "tick_size": _valor(info, "trade_tick_size"),
        "trade_tick_value": _valor(info, "trade_tick_value"),
        "tick_value": _valor(info, "trade_tick_value"),
        "trade_tick_value_profit": _valor(info, "trade_tick_value_profit"),
        "trade_tick_value_loss": _valor(info, "trade_tick_value_loss"),
        "digits": _valor(info, "digits"),
        "point": _valor(info, "point"),
        "currency_base": _valor(info, "currency_base"),
        "currency_profit": _valor(info, "currency_profit"),
        "currency_margin": _valor(info, "currency_margin"),
    }


def _metadata_row(asset, symbol, info):
    return {
        "ACTIVO": asset,
        "SYMBOL": symbol,
        "CONTRACT_SIZE": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
        "VOLUME_MIN": _valor(info, "volume_min"),
        "VOLUME_STEP": _valor(info, "volume_step"),
        "VOLUME_MAX": _valor(info, "volume_max"),
        "TICK_SIZE": _valor(info, "trade_tick_size"),
        "TICK_VALUE": _valor(info, "trade_tick_value"),
        "TICK_VALUE_PROFIT": _valor(info, "trade_tick_value_profit"),
        "TICK_VALUE_LOSS": _valor(info, "trade_tick_value_loss"),
        "CURRENCY_BASE": _valor(info, "currency_base"),
        "CURRENCY_PROFIT": _valor(info, "currency_profit"),
        "CURRENCY_MARGIN": _valor(info, "currency_margin"),
        "DIGITS": _valor(info, "digits"),
        "POINT": _valor(info, "point"),
    }


def _price_pair(asset, tick, info):
    bid = _valor(tick, "bid", _valor(info, "bid"))
    ask = _valor(tick, "ask", _valor(info, "ask"))
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None, None
    distance = DISTANCES[asset]
    return {
        "direction": BUY,
        "entry": float(ask),
        "stop": float(ask) - distance,
    }, {
        "direction": SELL,
        "entry": float(bid),
        "stop": float(bid) + distance,
    }


def _pct_diff(reference, value):
    if reference is None or reference == 0 or value is None:
        return None
    return abs(value - reference) / abs(reference) * 100


def _eurusd_paper_state(root):
    base = root / "paper" / "eurusd"
    trades = base / "trades.csv"
    open_positions = base / "open_positions.csv"
    trade_count = 0
    open_count = 0
    if trades.exists():
        data = pd.read_csv(trades)
        trade_count = len(data)
    if open_positions.exists():
        data = pd.read_csv(open_positions)
        open_count = len(data)
    return {
        "trades_file": str(trades),
        "open_positions_file": str(open_positions),
        "trades": trade_count,
        "open_positions": open_count,
        "state_correction_needed": bool(trade_count or open_count),
    }


def run_risk_manager_regression_audit(raiz_proyecto=None):
    root = _root(raiz_proyecto)
    credentials, error = _credenciales_mt5()
    if error:
        raise RuntimeError(error)

    import MetaTrader5 as mt5

    initialized = False
    try:
        initialized = mt5.initialize()
        if not initialized:
            raise RuntimeError(f"No se pudo inicializar MT5: {mt5.last_error()}")
        if not mt5.login(credentials["login"], password=credentials["password"], server=credentials["server"]):
            raise RuntimeError(f"No se pudo conectar MT5: {mt5.last_error()}")
        if _modo_cuenta(mt5, mt5.account_info()) != "DEMO":
            raise RuntimeError("Cuenta MT5 no DEMO; auditoria bloqueada")

        resolved = discover_symbols(mt5, ASSETS)
        manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))
        metadata_rows = []
        comparison_rows = []
        fallback_rows = []

        for asset in ASSETS:
            item = resolved[asset]
            if not item.found:
                comparison_rows.append(
                    {
                        "ACTIVO": asset,
                        "DIRECCION": "",
                        "RIESGO_MT5": None,
                        "RIESGO_RISK_MANAGER": None,
                        "DIFERENCIA": None,
                        "DIFERENCIA_PCT": None,
                        "RESULTADO": f"ERROR: {item.reason}",
                    }
                )
                continue
            mt5.symbol_select(item.mt5_symbol, True)
            info = mt5.symbol_info(item.mt5_symbol)
            tick = mt5.symbol_info_tick(item.mt5_symbol)
            metadata_rows.append(_metadata_row(asset, item.mt5_symbol, info))
            symbol_info = _symbol_info_dict(info)
            for example in _price_pair(asset, tick, info):
                if example is None:
                    continue
                order_type = mt5.ORDER_TYPE_BUY if example["direction"] == BUY else mt5.ORDER_TYPE_SELL
                mt5_loss = abs(
                    float(
                        mt5.order_calc_profit(
                            order_type,
                            item.mt5_symbol,
                            VOLUME,
                            example["entry"],
                            example["stop"],
                        )
                    )
                )
                sizing = manager.calculate_position_size(
                    symbol=item.mt5_symbol,
                    direction=example["direction"],
                    entry=example["entry"],
                    stop_loss=example["stop"],
                    balance=BALANCE,
                    symbol_info=symbol_info,
                    order_calc_profit=mt5.order_calc_profit,
                    order_type=order_type,
                )
                fallback = manager.calculate_position_size(
                    symbol=item.mt5_symbol,
                    direction=example["direction"],
                    entry=example["entry"],
                    stop_loss=example["stop"],
                    balance=BALANCE,
                    symbol_info=symbol_info,
                )
                rm_loss = sizing.min_volume_loss
                diff = abs(rm_loss - mt5_loss) if rm_loss is not None else None
                result = "OK" if diff is not None and diff <= max(0.01, mt5_loss * 0.001) else "ERROR"
                comparison_rows.append(
                    {
                        "ACTIVO": asset,
                        "SYMBOL": item.mt5_symbol,
                        "DIRECCION": example["direction"],
                        "ENTRY": example["entry"],
                        "SL": example["stop"],
                        "DISTANCIA_SL": abs(example["entry"] - example["stop"]),
                        "VOLUME": VOLUME,
                        "RIESGO_MT5": mt5_loss,
                        "RIESGO_RISK_MANAGER": rm_loss,
                        "DIFERENCIA": diff,
                        "DIFERENCIA_PCT": _pct_diff(mt5_loss, rm_loss),
                        "RESULTADO": result,
                        "LOSS_SOURCE": sizing.loss_source,
                    }
                )
                fallback_rows.append(
                    {
                        "ACTIVO": asset,
                        "DIRECCION": example["direction"],
                        "RIESGO_MT5": mt5_loss,
                        "RIESGO_FALLBACK_TICK": fallback.min_volume_loss,
                        "DIFERENCIA": abs(fallback.min_volume_loss - mt5_loss)
                        if fallback.min_volume_loss is not None
                        else None,
                        "DIFERENCIA_PCT": _pct_diff(mt5_loss, fallback.min_volume_loss),
                        "LOSS_SOURCE": fallback.loss_source,
                    }
                )

        return {
            "metadata": pd.DataFrame(metadata_rows),
            "comparison": pd.DataFrame(comparison_rows),
            "fallback": pd.DataFrame(fallback_rows),
            "eurusd_paper_state": _eurusd_paper_state(root),
        }
    finally:
        if initialized:
            mt5.shutdown()


def main():
    result = run_risk_manager_regression_audit()
    print("METADATA XM/MT5")
    print(result["metadata"].to_string(index=False))
    print()
    print("ORDER_CALC_PROFIT VS RISK MANAGER")
    print(result["comparison"].to_string(index=False))
    print()
    print("ORDER_CALC_PROFIT VS FALLBACK TICK_SIZE/TICK_VALUE")
    print(result["fallback"].to_string(index=False))
    print()
    print("EURUSD PAPER STATE")
    print(result["eurusd_paper_state"])


if __name__ == "__main__":
    main()
