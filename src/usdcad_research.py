from pathlib import Path

import pandas as pd

from config import COST_BPS, PROB_THRESHOLDS
from src import eurusd_research as core
from src import gbpusd_research as shared_research
from src.model import FEATURES_ML, TARGET_HORIZON, _agregar_base_signal
from src.risk import BUY, SELL, RiskLimits, RiskManager


ASSET = "USDCAD"
MAIN_COST_BPS = 2
CAPITALS = (100, 500, 1000, 5000)
RISK_PER_TRADE = 0.01
DIRECTIONS = ("LONG", "SHORT")


def _root(raiz_proyecto=None):
    return Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]


def _research_dir(raiz):
    path = raiz / "results" / "v2" / "usdcad"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_dataset(raiz):
    from src.data import obtener_datos
    from src.features import crear_dataset_ml
    from src.indicators import calcular_indicadores

    raw = obtener_datos(ASSET)
    indicators = calcular_indicadores(raw)
    data_dir = raiz / "data" / "usdcad"
    data_dir.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(data_dir / "usdcad_h1_indicators.csv")
    dataset = crear_dataset_ml(indicators)
    dataset.to_csv(data_dir / "ml_dataset.csv", index=False)
    return (
        dataset,
        {
            "source": "Yahoo Finance CAD=X",
            "interval": raw.attrs.get("intervalo", "1h"),
            "downloaded_period": raw.attrs.get("periodo_descargado", "N/A"),
            "created_now": True,
        },
    )


def _read_or_create_dataset(raiz):
    path = raiz / "data" / "usdcad" / "ml_dataset.csv"
    source = {
        "source": "data/usdcad/ml_dataset.csv",
        "interval": "1h",
        "downloaded_period": "N/A",
        "created_now": False,
    }
    if path.exists():
        data = pd.read_csv(path)
    else:
        data, source = _create_dataset(raiz)

    required = [
        "fecha",
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "TARGET_LONG",
        "TARGET_SHORT",
    ] + FEATURES_ML
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset USDCAD incompleto. Faltan: {', '.join(missing)}")

    data["fecha"] = pd.to_datetime(data["fecha"], utc=True).dt.tz_convert(None)
    data = data.sort_values("fecha").reset_index(drop=True)
    for column in ["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    data = data.dropna(subset=["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML)
    return _agregar_base_signal(data), source


def _audit_existing_models(raiz, data):
    rows = []
    models_dir = raiz / "models" / "usdcad"
    train_end_idx = max(0, int(len(data) * 0.70) - TARGET_HORIZON - 1)
    train_start = data["fecha"].iloc[0] if not data.empty else ""
    train_end = data["fecha"].iloc[train_end_idx] if not data.empty else ""

    for direction in DIRECTIONS:
        path = models_dir / f"random_forest_{direction.lower()}.pkl"
        row = {
            "ACTIVO": ASSET,
            "DIRECCION": direction,
            "RUTA": str(path),
            "EXISTE": path.exists(),
            "FECHA_ARCHIVO": "",
            "ALGORITMO": "",
            "FEATURES": ",".join(FEATURES_ML),
            "N_FEATURES_MODELO": "",
            "THRESHOLDS_DISPONIBLES": ",".join(str(item) for item in PROB_THRESHOLDS),
            "PERIODO_ENTRENAMIENTO_INFERIDO_INICIAL": train_start,
            "PERIODO_ENTRENAMIENTO_INFERIDO_FINAL": train_end,
            "PERIODO_DATASET_INICIAL": data["fecha"].min(),
            "PERIODO_DATASET_FINAL": data["fecha"].max(),
            "COMPATIBLE_METODOLOGIA_ACTUAL": False,
            "LEAKAGE_LOOKAHEAD_CONOCIDO": False,
            "NOTAS": "",
        }
        if path.exists():
            row["FECHA_ARCHIVO"] = pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()
            try:
                import joblib

                model = joblib.load(path)
                row["ALGORITMO"] = type(model).__name__
                n_features = getattr(model, "n_features_in_", "")
                row["N_FEATURES_MODELO"] = n_features
                feature_names = list(getattr(model, "feature_names_in_", []))
                feature_ok = (not feature_names and n_features == len(FEATURES_ML)) or feature_names == FEATURES_ML
                row["COMPATIBLE_METODOLOGIA_ACTUAL"] = bool(
                    row["ALGORITMO"] == "RandomForestClassifier" and feature_ok
                )
                row["NOTAS"] = (
                    "Modelo previo localizado; utilizable como referencia. "
                    "No se sobrescribe ni se declara validado."
                )
            except Exception as error:
                row["NOTAS"] = f"No se pudo cargar modelo: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


def _walkforward_economics(walkforward, oof):
    original_asset = shared_research.ASSET
    try:
        shared_research.ASSET = ASSET
        result = shared_research._walkforward_economics(walkforward, oof)
    finally:
        shared_research.ASSET = original_asset
    result["ACTIVO"] = ASSET
    return result


def _metadata_from_mt5(mt5):
    from src.mt5_connector import _valor
    from src.symbols import discover_symbols

    resolved = discover_symbols(mt5, [ASSET])[ASSET]
    if not resolved.found:
        return None, f"No se encontro USDCAD en MT5: {resolved.reason}"
    if not mt5.symbol_select(resolved.mt5_symbol, True):
        return None, f"No se pudo seleccionar simbolo MT5: {resolved.mt5_symbol}"

    info = mt5.symbol_info(resolved.mt5_symbol)
    tick = mt5.symbol_info_tick(resolved.mt5_symbol)
    bid = _valor(tick, "bid")
    ask = _valor(tick, "ask")
    point = _valor(info, "point")
    spread = (ask - bid) if bid is not None and ask is not None else None
    spread_points = (spread / point) if spread is not None and point else None
    return {
        "symbol": resolved.mt5_symbol,
        "contract_size": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
        "trade_contract_size": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
        "tick_value": _valor(info, "trade_tick_value"),
        "trade_tick_value": _valor(info, "trade_tick_value"),
        "tick_value_profit": _valor(info, "trade_tick_value_profit"),
        "trade_tick_value_profit": _valor(info, "trade_tick_value_profit"),
        "tick_value_loss": _valor(info, "trade_tick_value_loss"),
        "trade_tick_value_loss": _valor(info, "trade_tick_value_loss"),
        "tick_size": _valor(info, "trade_tick_size"),
        "trade_tick_size": _valor(info, "trade_tick_size"),
        "volume_min": _valor(info, "volume_min"),
        "volume_step": _valor(info, "volume_step"),
        "volume_max": _valor(info, "volume_max"),
        "digits": _valor(info, "digits"),
        "point": point,
        "currency_base": _valor(info, "currency_base"),
        "currency_profit": _valor(info, "currency_profit"),
        "currency_margin": _valor(info, "currency_margin"),
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "spread_points": spread_points,
    }, ""


def _broker_context():
    try:
        from src.mt5_connector import _credenciales_mt5, _modo_cuenta
    except Exception as error:
        return None, None, None, f"MT5 no disponible: {error}"

    credentials, error = _credenciales_mt5()
    if error:
        return None, None, None, error

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None, None, None, "MetaTrader5 no esta instalado"

    initialized = mt5.initialize()
    if not initialized:
        return mt5, None, False, f"No se pudo inicializar MT5: {mt5.last_error()}"
    if not mt5.login(credentials["login"], password=credentials["password"], server=credentials["server"]):
        mt5.shutdown()
        return mt5, None, False, f"No se pudo conectar MT5: {mt5.last_error()}"
    account = mt5.account_info()
    if _modo_cuenta(mt5, account) != "DEMO":
        mt5.shutdown()
        return mt5, None, False, "Cuenta MT5 no DEMO; metadata bloqueada"
    metadata, error = _metadata_from_mt5(mt5)
    if error:
        mt5.shutdown()
        return mt5, None, False, error
    return mt5, metadata, True, ""


def _order_type(mt5, direction):
    return mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL


def _validate_sizing(mt5, metadata, trades, max_examples=10):
    rows = []
    sample = trades[trades["COST_BPS"] == MAIN_COST_BPS].copy()
    per_direction = max(1, max_examples // 2)
    sample = (
        sample.sort_values(["DIRECCION", "FECHA_ENTRADA"])
        .groupby("DIRECCION", group_keys=False)
        .head(per_direction)
        .head(max_examples)
    )
    manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))
    volume_min = float(metadata["volume_min"])

    for _, trade in sample.iterrows():
        direction = BUY if trade["DIRECCION"] == "LONG" else SELL
        order_type = _order_type(mt5, trade["DIRECCION"])
        mt5_loss = abs(
            float(
                mt5.order_calc_profit(
                    order_type,
                    metadata["symbol"],
                    volume_min,
                    float(trade["ENTRY"]),
                    float(trade["STOP_LOSS"]),
                )
            )
        )
        sizing = manager.calculate_position_size(
            symbol=metadata["symbol"],
            direction=direction,
            entry=trade["ENTRY"],
            stop_loss=trade["STOP_LOSS"],
            balance=5000,
            symbol_info=metadata,
            order_calc_profit=mt5.order_calc_profit,
            order_type=order_type,
        )
        rm_loss = sizing.min_volume_loss
        diff = abs(rm_loss - mt5_loss) if rm_loss is not None else None
        diff_pct = (diff / mt5_loss * 100) if mt5_loss else 0.0
        rows.append(
            {
                "ACTIVO": ASSET,
                "DIRECCION": trade["DIRECCION"],
                "ENTRY": trade["ENTRY"],
                "STOP_LOSS": trade["STOP_LOSS"],
                "DISTANCIA_SL": abs(float(trade["ENTRY"]) - float(trade["STOP_LOSS"])),
                "VOLUME": volume_min,
                "RIESGO_MT5_ORDER_CALC": mt5_loss,
                "RIESGO_RISK_MANAGER": rm_loss,
                "DIFERENCIA_ABS": diff,
                "DIFERENCIA_PCT": diff_pct,
                "LOSS_SOURCE": sizing.loss_source,
                "RESULTADO": "OK" if diff is not None and diff <= max(0.01, mt5_loss * 0.001) else "ERROR",
            }
        )
    validation = pd.DataFrame(rows)
    if not validation.empty and (validation["RESULTADO"] != "OK").any():
        raise RuntimeError("USDCAD sizing no coincide con mt5.order_calc_profit; auditoria detenida")
    return validation


def _broker_aware(trades):
    mt5, metadata, initialized, error = _broker_context()
    rows = []
    validation = pd.DataFrame()
    representative = trades[trades["COST_BPS"] == MAIN_COST_BPS].copy()

    try:
        if metadata and mt5:
            validation = _validate_sizing(mt5, metadata, trades)

        for capital in CAPITALS:
            for _, trade in representative.iterrows():
                base = {
                    "ACTIVO": ASSET,
                    "CAPITAL": capital,
                    "RISK_PCT": RISK_PER_TRADE * 100,
                    "DIRECCION": trade["DIRECCION"],
                    "MODELO": trade["MODELO"],
                    "SISTEMA": trade["SISTEMA"],
                    "THRESHOLD": trade["THRESHOLD"],
                    "COST_BPS": trade["COST_BPS"],
                    "ENTRY": trade["ENTRY"],
                    "STOP_LOSS": trade["STOP_LOSS"],
                    "BROKER_METADATA_SOURCE": "MT5_DEMO" if metadata else "UNAVAILABLE",
                    "BROKER_METADATA_ERROR": error,
                    "SYMBOL": metadata.get("symbol") if metadata else None,
                    "CONTRACT_SIZE": metadata.get("contract_size") if metadata else None,
                    "TICK_SIZE": metadata.get("tick_size") if metadata else None,
                    "TICK_VALUE": metadata.get("tick_value") if metadata else None,
                    "TICK_VALUE_PROFIT": metadata.get("tick_value_profit") if metadata else None,
                    "TICK_VALUE_LOSS": metadata.get("tick_value_loss") if metadata else None,
                    "DIGITS": metadata.get("digits") if metadata else None,
                    "POINT": metadata.get("point") if metadata else None,
                    "CURRENCY_BASE": metadata.get("currency_base") if metadata else None,
                    "CURRENCY_PROFIT": metadata.get("currency_profit") if metadata else None,
                    "CURRENCY_MARGIN": metadata.get("currency_margin") if metadata else None,
                    "SPREAD": metadata.get("spread") if metadata else None,
                    "SPREAD_POINTS": metadata.get("spread_points") if metadata else None,
                    "VOLUME_MIN": metadata.get("volume_min") if metadata else None,
                    "VOLUME_STEP": metadata.get("volume_step") if metadata else None,
                    "VOLUME_MAX": metadata.get("volume_max") if metadata else None,
                    "VOLUME_THEORETICAL": None,
                    "VOLUME_CALCULATED": None,
                    "ESTIMATED_LOSS": None,
                    "MIN_VOLUME_LOSS": None,
                    "RISK_PCT_REAL": None,
                    "RISK_PCT_MINIMUM": None,
                    "LOSS_SOURCE": "NONE",
                    "DECISION": "SKIP",
                    "REASON": error or "OK",
                }
                if metadata and mt5:
                    manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))
                    direction = BUY if trade["DIRECCION"] == "LONG" else SELL
                    sizing = manager.calculate_position_size(
                        symbol=metadata["symbol"],
                        direction=direction,
                        entry=trade["ENTRY"],
                        stop_loss=trade["STOP_LOSS"],
                        balance=capital,
                        symbol_info=metadata,
                        order_calc_profit=mt5.order_calc_profit,
                        order_type=_order_type(mt5, trade["DIRECCION"]),
                    )
                    base["VOLUME_THEORETICAL"] = sizing.theoretical_volume
                    base["VOLUME_CALCULATED"] = sizing.volume
                    base["ESTIMATED_LOSS"] = sizing.estimated_loss
                    base["MIN_VOLUME_LOSS"] = sizing.min_volume_loss
                    base["RISK_PCT_REAL"] = sizing.risk_pct_real
                    base["RISK_PCT_MINIMUM"] = sizing.risk_pct_minimum
                    base["LOSS_SOURCE"] = sizing.loss_source
                    base["DECISION"] = sizing.decision
                    base["REASON"] = sizing.reason
                rows.append(base)
    finally:
        if initialized and mt5:
            mt5.shutdown()
    return pd.DataFrame(rows), validation


def _run_with_usdcad_asset(data):
    original_asset = core.ASSET
    try:
        core.ASSET = ASSET
        walkforward_raw, oof = core._run_walk_forward(data)
        oof_summary, trades = core._run_oof_backtest(oof)
        robustness = core._robustness(oof_summary, trades)
    finally:
        core.ASSET = original_asset
    return walkforward_raw, oof, oof_summary, trades, robustness


def _apply_candidate_safety_rules(robustness):
    result = robustness.copy()
    crosses_zero = (
        (result["BOOTSTRAP_EXPECTANCY_CI95_LOW"] <= 0)
        & (result["BOOTSTRAP_EXPECTANCY_CI95_HIGH"] >= 0)
    )
    mask = (result["CLASIFICACION"] == "CANDIDATO A FORWARD TEST") & crosses_zero
    result.loc[mask, "CLASIFICACION"] = "MIXTO"
    result.loc[mask, "MOTIVO"] = "CI95 de expectancy cruza cero; VENTAJA NO CONFIRMADA"
    return result


def run_usdcad_research(raiz_proyecto=None):
    raiz = _root(raiz_proyecto)
    out = _research_dir(raiz)
    data, data_source = _read_or_create_dataset(raiz)
    audit = _audit_existing_models(raiz, data)
    walkforward_raw, oof, oof_summary, trades, robustness = _run_with_usdcad_asset(data)
    robustness = _apply_candidate_safety_rules(robustness)
    walkforward = _walkforward_economics(walkforward_raw, oof)
    broker, sizing_validation = _broker_aware(trades)

    audit.to_csv(out / "usdcad_existing_models_audit.csv", index=False)
    walkforward.to_csv(out / "usdcad_walkforward_summary.csv", index=False)
    oof.to_csv(out / "usdcad_oof_predictions.csv", index=False)
    oof_summary.to_csv(out / "usdcad_oof_summary.csv", index=False)
    robustness.to_csv(out / "usdcad_robustness.csv", index=False)
    trades.to_csv(out / "usdcad_trades.csv", index=False)
    broker.to_csv(out / "usdcad_broker_aware.csv", index=False)

    candidates = robustness[
        (robustness["COST_BPS"] == MAIN_COST_BPS)
        & (robustness["CLASIFICACION"] == "CANDIDATO A FORWARD TEST")
    ].copy()
    discarded = robustness[
        (robustness["COST_BPS"] == MAIN_COST_BPS)
        & (robustness["CLASIFICACION"] != "CANDIDATO A FORWARD TEST")
    ].copy()

    return {
        "output_dir": out,
        "data_period": {
            "start": data["fecha"].min(),
            "end": data["fecha"].max(),
            "rows": len(data),
            "source": data_source["source"],
            "interval": data_source["interval"],
            "downloaded_period": data_source["downloaded_period"],
            "created_now": data_source["created_now"],
        },
        "audit": audit,
        "walkforward": walkforward,
        "walkforward_raw": walkforward_raw,
        "oof_summary": oof_summary,
        "robustness": robustness,
        "broker_aware": broker,
        "sizing_validation": sizing_validation,
        "candidates": candidates,
        "discarded": discarded,
        "trades": trades,
    }
