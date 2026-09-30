from pathlib import Path

import pandas as pd

from config import COST_BPS, PROB_THRESHOLDS
from src import eurusd_research as core
from src import gbpusd_research as shared_research
from src.model import FEATURES_ML, TARGET_HORIZON, _agregar_base_signal
from src.risk import RiskLimits, RiskManager


ASSET = "USDJPY"
MAIN_COST_BPS = 2
CAPITALS = (100, 500, 1000, 5000)
RISK_PER_TRADE = 0.01
DIRECTIONS = ("LONG", "SHORT")


def _root(raiz_proyecto=None):
    return Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]


def _research_dir(raiz):
    path = raiz / "results" / "v2" / "usdjpy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_dataset(raiz):
    path = raiz / "data" / "usdjpy" / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe dataset USDJPY: {path}")

    data = pd.read_csv(path)
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
        raise ValueError(f"Dataset USDJPY incompleto. Faltan: {', '.join(missing)}")

    data["fecha"] = pd.to_datetime(data["fecha"], utc=True).dt.tz_convert(None)
    data = data.sort_values("fecha").reset_index(drop=True)
    for column in ["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    data = data.dropna(subset=["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML)
    return _agregar_base_signal(data)


def _audit_existing_models(raiz, data):
    rows = []
    models_dir = raiz / "models" / "usdjpy"
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


def _broker_metadata():
    try:
        from src.mt5_connector import _credenciales_mt5, _modo_cuenta, _valor
        from src.symbols import discover_symbols
    except Exception as error:
        return None, f"MT5 no disponible: {error}"

    credentials, error = _credenciales_mt5()
    if error:
        return None, error

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None, "MetaTrader5 no esta instalado"

    initialized = False
    try:
        initialized = mt5.initialize()
        if not initialized:
            return None, f"No se pudo inicializar MT5: {mt5.last_error()}"
        if not mt5.login(credentials["login"], password=credentials["password"], server=credentials["server"]):
            return None, f"No se pudo conectar MT5: {mt5.last_error()}"
        account = mt5.account_info()
        if _modo_cuenta(mt5, account) != "DEMO":
            return None, "Cuenta MT5 no DEMO; metadata bloqueada"
        resolved = discover_symbols(mt5, [ASSET])[ASSET]
        if not resolved.found:
            return None, f"No se encontro USDJPY en MT5: {resolved.reason}"
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
            "tick_value": _valor(info, "trade_tick_value"),
            "tick_size": _valor(info, "trade_tick_size"),
            "volume_min": _valor(info, "volume_min"),
            "volume_step": _valor(info, "volume_step"),
            "volume_max": _valor(info, "volume_max"),
            "point": point,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "spread_points": spread_points,
        }, ""
    finally:
        if initialized:
            mt5.shutdown()


def _broker_aware(trades):
    metadata, error = _broker_metadata()
    rows = []
    representative = trades[trades["COST_BPS"] == MAIN_COST_BPS].copy()

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
                "SPREAD": metadata.get("spread") if metadata else None,
                "SPREAD_POINTS": metadata.get("spread_points") if metadata else None,
                "VOLUME_MIN": metadata.get("volume_min") if metadata else None,
                "VOLUME_STEP": metadata.get("volume_step") if metadata else None,
                "VOLUME_THEORETICAL": None,
                "VOLUME_CALCULATED": None,
                "ESTIMATED_LOSS": None,
                "MIN_VOLUME_LOSS": None,
                "RISK_PCT_REAL": None,
                "RISK_PCT_MINIMUM": None,
                "DECISION": "SKIP",
                "REASON": error or "OK",
            }
            if metadata:
                manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))
                sizing = manager.calculate_position_size(
                    symbol=metadata["symbol"],
                    direction="BUY" if trade["DIRECCION"] == "LONG" else "SELL",
                    entry=trade["ENTRY"],
                    stop_loss=trade["STOP_LOSS"],
                    balance=capital,
                    symbol_info=metadata,
                )
                base["VOLUME_THEORETICAL"] = sizing.theoretical_volume
                base["VOLUME_CALCULATED"] = sizing.volume
                base["ESTIMATED_LOSS"] = sizing.estimated_loss
                base["MIN_VOLUME_LOSS"] = sizing.min_volume_loss
                base["RISK_PCT_REAL"] = sizing.risk_pct_real
                base["RISK_PCT_MINIMUM"] = sizing.risk_pct_minimum
                base["DECISION"] = sizing.decision
                base["REASON"] = sizing.reason
            rows.append(base)
    return pd.DataFrame(rows)


def _run_with_usdjpy_asset(data):
    original_asset = core.ASSET
    try:
        core.ASSET = ASSET
        walkforward_raw, oof = core._run_walk_forward(data)
        oof_summary, trades = core._run_oof_backtest(oof)
        robustness = core._robustness(oof_summary, trades)
    finally:
        core.ASSET = original_asset
    return walkforward_raw, oof, oof_summary, trades, robustness


def run_usdjpy_research(raiz_proyecto=None):
    raiz = _root(raiz_proyecto)
    out = _research_dir(raiz)
    data = _read_dataset(raiz)
    audit = _audit_existing_models(raiz, data)
    walkforward_raw, oof, oof_summary, trades, robustness = _run_with_usdjpy_asset(data)
    walkforward = _walkforward_economics(walkforward_raw, oof)
    broker = _broker_aware(trades)

    audit.to_csv(out / "usdjpy_existing_models_audit.csv", index=False)
    walkforward.to_csv(out / "usdjpy_walkforward_summary.csv", index=False)
    oof.to_csv(out / "usdjpy_oof_predictions.csv", index=False)
    oof_summary.to_csv(out / "usdjpy_oof_summary.csv", index=False)
    robustness.to_csv(out / "usdjpy_robustness.csv", index=False)
    trades.to_csv(out / "usdjpy_trades.csv", index=False)
    broker.to_csv(out / "usdjpy_broker_aware.csv", index=False)

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
        },
        "audit": audit,
        "walkforward": walkforward,
        "walkforward_raw": walkforward_raw,
        "oof_summary": oof_summary,
        "robustness": robustness,
        "broker_aware": broker,
        "candidates": candidates,
        "discarded": discarded,
        "trades": trades,
    }
