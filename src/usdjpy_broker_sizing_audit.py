from pathlib import Path

import pandas as pd

from src.mt5_connector import _credenciales_mt5, _modo_cuenta, _valor
from src.risk import BUY, SELL, RiskLimits, RiskManager
from src.symbols import discover_symbols


CAPITALS = (100, 500, 1000, 5000)
RISK_PER_TRADE = 0.01
MAIN_COST_BPS = 2


def _root(raiz_proyecto=None):
    return Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]


def _trades_path(raiz):
    return raiz / "results" / "v2" / "usdjpy" / "usdjpy_trades.csv"


def _previous_broker_path(raiz):
    return raiz / "results" / "v2" / "usdjpy" / "usdjpy_broker_aware.csv"


def _dataset_path(raiz):
    return raiz / "data" / "usdjpy" / "ml_dataset.csv"


def _load_representative_trades(raiz):
    trades = pd.read_csv(_trades_path(raiz))
    trades = trades[trades["COST_BPS"] == MAIN_COST_BPS].copy()
    trades["DISTANCIA_SL"] = (trades["ENTRY"] - trades["STOP_LOSS"]).abs()
    return trades


def _load_atr_by_signal(raiz):
    data = pd.read_csv(_dataset_path(raiz), usecols=["fecha", "ATR"])
    data["FECHA_SIGNAL"] = pd.to_datetime(data["fecha"], utc=True).dt.tz_convert(None)
    data["ATR"] = pd.to_numeric(data["ATR"], errors="coerce")
    return data[["FECHA_SIGNAL", "ATR"]].dropna()


def _select_examples(trades, atr_data, count=10):
    unique = (
        trades.sort_values("DISTANCIA_SL")
        .drop_duplicates(subset=["DIRECCION", "ENTRY", "STOP_LOSS", "FECHA_SIGNAL"])
        .copy()
    )
    unique["FECHA_SIGNAL"] = pd.to_datetime(unique["FECHA_SIGNAL"])
    merged = unique.merge(atr_data, on="FECHA_SIGNAL", how="left")

    selected = []
    per_direction = max(1, count // 2)
    for direction in ("LONG", "SHORT"):
        subset = merged[merged["DIRECCION"] == direction].sort_values("DISTANCIA_SL")
        if subset.empty:
            continue
        positions = sorted(
            {
                int(round(index))
                for index in pd.Series(range(per_direction))
                * ((len(subset) - 1) / max(per_direction - 1, 1))
            }
        )
        selected.append(subset.iloc[positions])

    examples = pd.concat(selected, ignore_index=True) if selected else merged.head(count)
    if len(examples) < count:
        remaining = merged.drop(examples.index, errors="ignore").sort_values("DISTANCIA_SL")
        examples = pd.concat([examples, remaining.head(count - len(examples))], ignore_index=True)
    return examples.head(count).copy()


def _symbol_info_dict(info):
    return {
        "trade_contract_size": _valor(info, "trade_contract_size"),
        "contract_size": _valor(info, "trade_contract_size"),
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


def _metadata_row(symbol, info, tick):
    return {
        "symbol": symbol,
        "contract_size": _valor(info, "trade_contract_size"),
        "volume_min": _valor(info, "volume_min"),
        "volume_step": _valor(info, "volume_step"),
        "volume_max": _valor(info, "volume_max"),
        "tick_size": _valor(info, "trade_tick_size"),
        "tick_value": _valor(info, "trade_tick_value"),
        "tick_value_profit": _valor(info, "trade_tick_value_profit"),
        "tick_value_loss": _valor(info, "trade_tick_value_loss"),
        "digits": _valor(info, "digits"),
        "point": _valor(info, "point"),
        "currency_base": _valor(info, "currency_base"),
        "currency_profit": _valor(info, "currency_profit"),
        "currency_margin": _valor(info, "currency_margin"),
        "bid": _valor(tick, "bid"),
        "ask": _valor(tick, "ask"),
    }


def _old_broker_lookup(raiz):
    path = _previous_broker_path(raiz)
    if not path.exists():
        return pd.DataFrame()
    previous = pd.read_csv(path)
    previous = previous[previous["CAPITAL"] == 5000].copy()
    previous["KEY"] = (
        previous["DIRECCION"].astype(str)
        + "|"
        + previous["ENTRY"].round(9).astype(str)
        + "|"
        + previous["STOP_LOSS"].round(9).astype(str)
    )
    return previous.drop_duplicates("KEY").set_index("KEY")


def _old_key(row):
    return f"{row['DIRECCION']}|{round(float(row['ENTRY']), 9)}|{round(float(row['STOP_LOSS']), 9)}"


def run_usdjpy_broker_sizing_audit(raiz_proyecto=None, example_capital=5000):
    raiz = _root(raiz_proyecto)
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
        account = mt5.account_info()
        if _modo_cuenta(mt5, account) != "DEMO":
            raise RuntimeError("Cuenta MT5 no DEMO; auditoria bloqueada")

        resolved = discover_symbols(mt5, ["USDJPY"])["USDJPY"]
        if not resolved.found:
            raise RuntimeError(f"No se encontro USDJPY en MT5: {resolved.reason}")
        if not mt5.symbol_select(resolved.mt5_symbol, True):
            raise RuntimeError(f"No se pudo seleccionar simbolo MT5: {resolved.mt5_symbol}")

        info = mt5.symbol_info(resolved.mt5_symbol)
        tick = mt5.symbol_info_tick(resolved.mt5_symbol)
        metadata = _metadata_row(resolved.mt5_symbol, info, tick)
        symbol_info = _symbol_info_dict(info)
        manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))

        trades = _load_representative_trades(raiz)
        atr_data = _load_atr_by_signal(raiz)
        previous = _old_broker_lookup(raiz)

        capital_rows = []
        for capital in CAPITALS:
            executable = 0
            skip = 0
            for _, trade in trades.iterrows():
                direction = BUY if trade["DIRECCION"] == "LONG" else SELL
                order_type = mt5.ORDER_TYPE_BUY if direction == BUY else mt5.ORDER_TYPE_SELL
                sizing = manager.calculate_position_size(
                    symbol=resolved.mt5_symbol,
                    direction=direction,
                    entry=trade["ENTRY"],
                    stop_loss=trade["STOP_LOSS"],
                    balance=capital,
                    symbol_info=symbol_info,
                    order_calc_profit=mt5.order_calc_profit,
                    order_type=order_type,
                )
                executable += int(sizing.ok)
                skip += int(not sizing.ok)
            total = executable + skip
            capital_rows.append(
                {
                    "CAPITAL": capital,
                    "TOTAL_EVALUADO": total,
                    "EJECUTABLES": executable,
                    "SKIP": skip,
                    "PORCENTAJE_EJECUTABLE": (executable / total * 100) if total else 0.0,
                }
            )

        example_rows = []
        examples = _select_examples(trades, atr_data, count=10)
        for _, trade in examples.iterrows():
            direction = BUY if trade["DIRECCION"] == "LONG" else SELL
            order_type = mt5.ORDER_TYPE_BUY if direction == BUY else mt5.ORDER_TYPE_SELL
            risk_amount = example_capital * RISK_PER_TRADE
            risk_min = abs(
                float(
                    mt5.order_calc_profit(
                        order_type,
                        resolved.mt5_symbol,
                        float(symbol_info["volume_min"]),
                        float(trade["ENTRY"]),
                        float(trade["STOP_LOSS"]),
                    )
                )
            )
            sizing = manager.calculate_position_size(
                symbol=resolved.mt5_symbol,
                direction=direction,
                entry=trade["ENTRY"],
                stop_loss=trade["STOP_LOSS"],
                balance=example_capital,
                symbol_info=symbol_info,
                order_calc_profit=mt5.order_calc_profit,
                order_type=order_type,
            )
            old = previous.loc[_old_key(trade)] if not previous.empty and _old_key(trade) in previous.index else {}
            example_rows.append(
                {
                    "FECHA_SIGNAL": trade["FECHA_SIGNAL"],
                    "DIRECCION": trade["DIRECCION"],
                    "ENTRY": trade["ENTRY"],
                    "SL": trade["STOP_LOSS"],
                    "DISTANCIA": abs(float(trade["ENTRY"]) - float(trade["STOP_LOSS"])),
                    "ATR": trade.get("ATR"),
                    "CAPITAL": example_capital,
                    "RIESGO_OBJETIVO_USD": risk_amount,
                    "VOLUME_MIN": symbol_info["volume_min"],
                    "RIESGO_USD_001_ORDER_CALC": risk_min,
                    "RIESGO_PCT_001": (risk_min / example_capital) * 100,
                    "VOLUMEN_TEORICO": sizing.theoretical_volume,
                    "VOLUMEN_PERMITIDO": sizing.volume,
                    "DECISION": sizing.decision,
                    "REASON": sizing.reason,
                    "CALC_ANTERIOR_MIN_LOSS": old.get("MIN_VOLUME_LOSS"),
                    "CALC_ANTERIOR_DECISION": old.get("DECISION"),
                    "CALC_ANTERIOR_REASON": old.get("REASON"),
                }
            )

        return {
            "metadata": pd.DataFrame([metadata]),
            "examples": pd.DataFrame(example_rows),
            "capital_summary": pd.DataFrame(capital_rows),
        }
    finally:
        if initialized:
            mt5.shutdown()


def main():
    result = run_usdjpy_broker_sizing_audit()
    print("METADATA USDJPY XM")
    print(result["metadata"].to_string(index=False))
    print()
    print("10 EJEMPLOS CON ORDER_CALC_PROFIT")
    print(result["examples"].to_string(index=False))
    print()
    print("EJECUTABILIDAD POR CAPITAL")
    print(result["capital_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
