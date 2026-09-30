import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from config import DEMO_EXECUTION_ENABLED, TRADING_ENABLED
from src.features import crear_dataset_ml
from src.indicators import calcular_indicadores
from src.risk import BUY, RiskLimits, RiskManager
from src.telegram_events import (
    format_close_message,
    format_new_signal_message,
    format_open_message,
    format_skip_risk_message,
    notify_critical_error,
)


ASSET = "EURUSD"
DIRECTION = "LONG"
MODEL_NAME = "RF_ACTUAL"
SYSTEM = "ML_ONLY"
THRESHOLD = 0.65
CONFIG_ID = "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065"
TIMEFRAME = "H1"
CAPITAL_INITIAL = 100.0
RISK_PER_TRADE = 0.01
COST_BPS = 2
MAX_HOLD_BARS = 24
EXPECTED_MODEL_HASH = "b7d2136797e433bb3f2a59170b77569570c60c58f0cf7f75ba19f4ae17aa8a6f"
MT5_RATES_COUNT = 5000
STALE_MAX_HOURS = 3

FEATURES_ML = [
    "EMA20_EMA50",
    "PRECIO_EMA20",
    "PRECIO_EMA50",
    "PRECIO_EMA200",
    "RSI",
    "ATR_PCT",
    "RET_1H",
    "RET_4H",
    "RET_24H",
    "VOLATILIDAD",
    "RANGO_VELA",
    "CUERPO_VELA",
    "HORA",
    "DIA_SEMANA",
]

SIGNALS_COLUMNS = [
    "SIGNAL_ID",
    "CONFIG_ID",
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "FECHA_SIGNAL",
    "SIGNAL",
    "EXECUTION",
    "PROBABILIDAD",
    "ENTRY",
    "STOP",
    "TP",
    "ATR",
    "VOLUME",
    "VOLUME_MIN",
    "RISK_USD",
    "REASON",
]
OPEN_COLUMNS = [
    "POSITION_ID",
    "SIGNAL_ID",
    "CONFIG_ID",
    "ACTIVO",
    "DIRECCION",
    "FECHA_SIGNAL",
    "FECHA_ENTRADA",
    "ENTRY",
    "STOP",
    "TP",
    "ATR",
    "PROBABILIDAD",
    "VOLUME",
    "CAPITAL_ANTES",
    "BARS_HELD",
]
TRADES_COLUMNS = [
    "TRADE_ID",
    "POSITION_ID",
    "SIGNAL_ID",
    "CONFIG_ID",
    "ACTIVO",
    "DIRECCION",
    "FECHA_SIGNAL",
    "FECHA_ENTRADA",
    "FECHA_SALIDA",
    "ENTRY",
    "STOP",
    "TP",
    "EXIT",
    "RESULTADO",
    "R_BRUTO",
    "R_NETO",
    "CAPITAL_ANTES",
    "CAPITAL_DESPUES",
    "MOTIVO_SALIDA",
]
EQUITY_COLUMNS = ["FECHA", "CONFIG_ID", "CAPITAL"]


def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_naive_utc(value):
    return pd.to_datetime(value, utc=True).tz_convert(None)


def _paths(root):
    root = Path(root)
    paper_dir = root / "paper" / "eurusd"
    model_dir = root / "models" / "paper"
    return {
        "paper_dir": paper_dir,
        "start": paper_dir / "start.json",
        "state": paper_dir / "state.json",
        "signals": paper_dir / "signals.csv",
        "open_positions": paper_dir / "open_positions.csv",
        "trades": paper_dir / "trades.csv",
        "equity": paper_dir / "equity.csv",
        "model": model_dir / "eurusd_long_rf_actual_t065.pkl",
        "metadata": model_dir / "eurusd_long_rf_actual_t065_metadata.json",
    }


def _read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_csv(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def _write_csv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _append_csv(path, rows, columns):
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=False, index=False)


def _ensure_files(root):
    paths = _paths(root)
    paths["paper_dir"].mkdir(parents=True, exist_ok=True)
    for key, columns in (
        ("signals", SIGNALS_COLUMNS),
        ("open_positions", OPEN_COLUMNS),
        ("trades", TRADES_COLUMNS),
        ("equity", EQUITY_COLUMNS),
    ):
        if not paths[key].exists():
            pd.DataFrame(columns=columns).to_csv(paths[key], index=False)
    if not paths["state"].exists():
        _write_json(
            paths["state"],
            {
                "last_processed_bar": None,
                "pending_signal": None,
                "last_probability_long": None,
                "last_signal": "WAIT",
                "last_execution": "WAIT",
                "last_reason": "NO_PROCESSED_BARS",
                "data_source": "MT5 XM",
                "data_status": "UNKNOWN",
                "utc_now": None,
                "last_mt5_bar": None,
                "last_closed_bar": None,
                "first_full_forward_bar": None,
                "pending_bars": 0,
            },
        )
    return paths


def _ensure_start(root):
    paths = _ensure_files(root)
    if paths["start"].exists():
        return _read_json(paths["start"], {})

    start = _now_utc().isoformat()
    data = {
        "asset": ASSET,
        "timeframe": TIMEFRAME,
        "direction": DIRECTION,
        "model": MODEL_NAME,
        "system": SYSTEM,
        "threshold": THRESHOLD,
        "start_time_utc": start,
    }
    _write_json(paths["start"], data)
    equity = _read_csv(paths["equity"], EQUITY_COLUMNS)
    if equity.empty:
        _append_csv(
            paths["equity"],
            [{"FECHA": start, "CONFIG_ID": CONFIG_ID, "CAPITAL": CAPITAL_INITIAL}],
            EQUITY_COLUMNS,
        )
    return data


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_full_forward_bar(start_time):
    start = pd.to_datetime(start_time, utc=True)
    floored = start.floor("h")
    if start == floored:
        return floored.tz_convert(None)
    return (floored + pd.Timedelta(hours=1)).tz_convert(None)


def _assert_frozen_model(root):
    paths = _paths(root)
    if not paths["model"].exists():
        raise FileNotFoundError("Modelo EURUSD forward congelado no existe.")
    if not paths["metadata"].exists():
        raise FileNotFoundError("Metadata EURUSD forward congelada no existe.")

    current_hash = _sha256(paths["model"])
    if current_hash != EXPECTED_MODEL_HASH:
        raise RuntimeError("ABORTAR: hash del modelo EURUSD congelado cambio.")

    metadata = _read_json(paths["metadata"], {})
    if metadata.get("hash_modelo") != EXPECTED_MODEL_HASH:
        raise RuntimeError("ABORTAR: metadata hash EURUSD no coincide.")
    if metadata.get("activo") != ASSET or metadata.get("direccion") != DIRECTION:
        raise RuntimeError("ABORTAR: metadata EURUSD no corresponde al candidato congelado.")
    if metadata.get("modelo") != MODEL_NAME or metadata.get("sistema") != SYSTEM:
        raise RuntimeError("ABORTAR: metadata modelo/sistema no coincide.")
    if float(metadata.get("threshold", -1)) != THRESHOLD:
        raise RuntimeError("ABORTAR: threshold EURUSD congelado no coincide.")
    if metadata.get("features") != FEATURES_ML:
        raise RuntimeError("ABORTAR: features EURUSD congeladas no coinciden.")

    return {"created": False, "path": paths["model"], "metadata": metadata}


def _closed_bar_limit(now):
    now_ts = pd.to_datetime(now or _now_utc(), utc=True)
    return (now_ts.floor("h") - pd.Timedelta(hours=1)).tz_convert(None)


def _rates_to_prices(rates):
    data = pd.DataFrame(rates)
    if data.empty:
        return data
    data["fecha"] = pd.to_datetime(data["time"], unit="s", utc=True).dt.tz_convert(None)
    data = data.sort_values("fecha").drop_duplicates(subset=["fecha"], keep="last")
    data.index = data["fecha"]
    volume = data["tick_volume"] if "tick_volume" in data.columns else 0
    prices = pd.DataFrame(
        {
            "Open": pd.to_numeric(data["open"], errors="coerce"),
            "High": pd.to_numeric(data["high"], errors="coerce"),
            "Low": pd.to_numeric(data["low"], errors="coerce"),
            "Close": pd.to_numeric(data["close"], errors="coerce"),
            "Volume": volume,
        },
        index=data.index,
    )
    return prices.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def _normalize_mt5_server_time(prices, now):
    if prices.empty:
        return prices, 0
    now_floor = pd.to_datetime(now, utc=True).floor("h").tz_convert(None)
    last_bar = pd.to_datetime(prices.index.max())
    offset_hours = 0
    if last_bar > now_floor:
        raw_offset = (last_bar - now_floor) / pd.Timedelta(hours=1)
        rounded = int(round(raw_offset))
        if 0 < rounded <= 14:
            offset_hours = rounded
    if offset_hours:
        prices = prices.copy()
        prices.index = prices.index - pd.Timedelta(hours=offset_hours)
        prices = prices.sort_index()
    return prices, offset_hours


def _mt5_h1_prices(now=None, rates_count=MT5_RATES_COUNT):
    from src.mt5_connector import _credenciales_mt5, _modo_cuenta, _valor
    from src.symbols import discover_symbols

    credentials, error = _credenciales_mt5()
    if error:
        raise RuntimeError(error)

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 no esta instalado") from exc

    initialized = False
    try:
        initialized = mt5.initialize()
        if not initialized:
            raise RuntimeError(f"No se pudo inicializar MT5: {mt5.last_error()}")
        if not mt5.login(credentials["login"], password=credentials["password"], server=credentials["server"]):
            raise RuntimeError(f"No se pudo conectar MT5: {mt5.last_error()}")
        account = mt5.account_info()
        if _modo_cuenta(mt5, account) != "DEMO":
            raise RuntimeError("Cuenta MT5 no DEMO; forward EURUSD bloqueado")

        resolved = discover_symbols(mt5, [ASSET])[ASSET]
        if not resolved.found:
            raise RuntimeError(f"No se pudo resolver EURUSD en MT5: {resolved.reason}")
        symbol = resolved.mt5_symbol
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"No se pudo seleccionar simbolo MT5: {symbol}")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, rates_count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 no devolvio rates H1 para {symbol}")

        prices = _rates_to_prices(rates)
        prices, server_utc_offset_hours = _normalize_mt5_server_time(prices, now or _now_utc())
        info = mt5.symbol_info(symbol)
        broker = {
            "name": symbol,
            "server_utc_offset_hours": server_utc_offset_hours,
            "volume_min": _valor(info, "volume_min"),
            "volume_max": _valor(info, "volume_max"),
            "volume_step": _valor(info, "volume_step"),
            "trade_contract_size": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
            "trade_tick_size": _valor(info, "trade_tick_size"),
            "trade_tick_value": _valor(info, "trade_tick_value"),
            "point": _valor(info, "point"),
            "digits": _valor(info, "digits"),
        }
        return prices, broker, symbol
    finally:
        if initialized:
            mt5.shutdown()


def _mt5_data_context(root, now=None):
    utc_now = pd.to_datetime(now or _now_utc(), utc=True)
    diagnostics = {
        "data_source": "MT5 XM",
        "data_status": "ERROR",
        "utc_now": utc_now.isoformat(),
        "last_mt5_bar": None,
        "last_closed_bar": None,
        "first_full_forward_bar": None,
        "pending_bars": 0,
        "error": "",
        "symbol": None,
        "server_utc_offset_hours": None,
    }
    try:
        prices, broker, symbol = _mt5_h1_prices(now=utc_now)
    except Exception as error:
        diagnostics["error"] = str(error)
        return {"ok": False, "prices": pd.DataFrame(), "features": pd.DataFrame(), "broker": None, "diagnostics": diagnostics}

    diagnostics["symbol"] = symbol
    diagnostics["server_utc_offset_hours"] = broker.get("server_utc_offset_hours") if broker else None
    if prices.empty:
        diagnostics["error"] = "MT5 devolvio DataFrame vacio"
        return {"ok": False, "prices": prices, "features": pd.DataFrame(), "broker": broker, "diagnostics": diagnostics}

    last_mt5_bar = prices.index.max()
    closed_limit = _closed_bar_limit(utc_now)
    closed_prices = prices[prices.index <= closed_limit].copy()
    diagnostics["last_mt5_bar"] = pd.to_datetime(last_mt5_bar).isoformat()
    if closed_prices.empty:
        diagnostics["data_status"] = "STALE"
        diagnostics["error"] = "No hay velas H1 cerradas disponibles"
        return {"ok": False, "prices": closed_prices, "features": pd.DataFrame(), "broker": broker, "diagnostics": diagnostics}

    last_closed = closed_prices.index.max()
    diagnostics["last_closed_bar"] = pd.to_datetime(last_closed).isoformat()
    stale_age = utc_now.tz_convert(None) - (pd.to_datetime(last_closed) + pd.Timedelta(hours=1))
    if stale_age > pd.Timedelta(hours=STALE_MAX_HOURS):
        diagnostics["data_status"] = "STALE"
        diagnostics["error"] = "DATA_STALE"
        return {"ok": False, "prices": closed_prices, "features": pd.DataFrame(), "broker": broker, "diagnostics": diagnostics}

    start = _ensure_start(root)
    first_full = _first_full_forward_bar(start["start_time_utc"])
    diagnostics["first_full_forward_bar"] = pd.to_datetime(first_full).isoformat()
    try:
        features = crear_dataset_ml(calcular_indicadores(closed_prices))
    except Exception as error:
        diagnostics["error"] = f"FEATURES_ERROR: {error}"
        return {"ok": False, "prices": closed_prices, "features": pd.DataFrame(), "broker": broker, "diagnostics": diagnostics}
    features["fecha"] = pd.to_datetime(features["fecha"], utc=True).dt.tz_convert(None)
    features = features.sort_values("fecha").reset_index(drop=True)
    diagnostics["data_status"] = "LIVE"
    return {"ok": True, "prices": closed_prices, "features": features, "broker": broker, "diagnostics": diagnostics}


def _load_price_history(root):
    path = Path(root) / "data" / "eurusd" / "eurusd_h1.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe historico EURUSD H1: {path}")
    data = pd.read_csv(path)
    date_col = next((col for col in ("fecha", "Datetime", "Date", "index", "Unnamed: 0") if col in data.columns), data.columns[0])
    data[date_col] = pd.to_datetime(data[date_col], utc=True, errors="coerce")
    data = data.dropna(subset=[date_col])
    data.index = data[date_col].dt.tz_convert(None)
    if "Volume" not in data.columns:
        data["Volume"] = 0
    price_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in price_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Historico EURUSD incompleto: {', '.join(missing)}")
    for column in price_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=price_columns).sort_index()
    return data[price_columns]


def _training_dataset(root, forward_start):
    prices = _load_price_history(root)
    prices = prices[prices.index < forward_start].copy()
    if len(prices) <= MAX_HOLD_BARS + 250:
        raise ValueError("No hay suficiente historico anterior al inicio forward.")
    dataset = crear_dataset_ml(calcular_indicadores(prices))
    dataset["fecha"] = pd.to_datetime(dataset["fecha"], utc=True).dt.tz_convert(None)
    dataset = dataset[dataset["fecha"] < forward_start].copy()
    dataset = dataset.dropna(subset=FEATURES_ML + ["TARGET_LONG"])
    dataset["TARGET_LONG"] = dataset["TARGET_LONG"].astype(int)
    if dataset["TARGET_LONG"].nunique() < 2:
        raise ValueError("TARGET_LONG necesita dos clases para congelar RF_ACTUAL.")
    if not dataset.empty and dataset["fecha"].max() >= forward_start:
        raise ValueError("Control de fuga fallido: training end >= forward start.")
    return dataset


def freeze_eurusd_forward_model(root):
    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import RandomForestClassifier

    paths = _ensure_files(root)
    start = _ensure_start(root)
    forward_start = _to_naive_utc(start["start_time_utc"])

    if paths["model"].exists() and paths["metadata"].exists():
        metadata = _read_json(paths["metadata"], {})
        return {"created": False, "path": paths["model"], "metadata": metadata}

    dataset = _training_dataset(root, forward_start)
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(dataset[FEATURES_ML], dataset["TARGET_LONG"])
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths["model"])
    model_hash = _sha256(paths["model"])
    metadata = {
        "activo": ASSET,
        "timeframe": TIMEFRAME,
        "direccion": DIRECTION,
        "algoritmo": "RandomForestClassifier",
        "modelo": MODEL_NAME,
        "sistema": SYSTEM,
        "features": FEATURES_ML,
        "threshold": THRESHOLD,
        "fecha_entrenamiento": _now_utc().isoformat(),
        "inicio_datos_entrenamiento": pd.to_datetime(dataset["fecha"].min()).isoformat(),
        "fin_datos_entrenamiento": pd.to_datetime(dataset["fecha"].max()).isoformat(),
        "forward_start": start["start_time_utc"],
        "hash_modelo": model_hash,
        "sklearn_version": sklearn_version,
        "fecha_congelamiento": _now_utc().isoformat(),
        "capital_inicial": CAPITAL_INITIAL,
        "risk_per_trade": RISK_PER_TRADE,
        "max_hold_bars": MAX_HOLD_BARS,
        "sl_atr": 1,
        "tp_atr": 2,
        "frozen": True,
        "data_posterior_forward_utilizada": False,
    }
    _write_json(paths["metadata"], metadata)
    return {"created": True, "path": paths["model"], "metadata": metadata}


def _load_model(root):
    paths = _paths(root)
    if not paths["model"].exists():
        raise FileNotFoundError("Modelo EURUSD forward no congelado.")
    return joblib.load(paths["model"])


def _closed_feature_data(root, now=None):
    context = _mt5_data_context(root, now=now)
    if not context["ok"]:
        return pd.DataFrame()
    return context["features"]


def _probability_long(model, row):
    x = pd.DataFrame([row[FEATURES_ML].astype(float).to_dict()])
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return 0.0
    return float(probs[0][classes.index(1)])


def _capital(paths):
    equity = _read_csv(paths["equity"], EQUITY_COLUMNS)
    if equity.empty:
        return CAPITAL_INITIAL
    return float(equity.iloc[-1]["CAPITAL"])


def _broker_metadata():
    try:
        from src.mt5_connector import _credenciales_mt5, _modo_cuenta, _valor
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
            return None, "Cuenta MT5 no DEMO; broker-aware bloqueado"
        info = mt5.symbol_info("EURUSD") or mt5.symbol_info("EURUSDm")
        if info is None:
            return None, "No se encontro EURUSD/EURUSDm"
        return {
            "name": _valor(info, "name", "EURUSD"),
            "volume_min": _valor(info, "volume_min"),
            "volume_max": _valor(info, "volume_max"),
            "volume_step": _valor(info, "volume_step"),
            "trade_contract_size": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
            "trade_tick_size": _valor(info, "trade_tick_size"),
            "trade_tick_value": _valor(info, "trade_tick_value"),
            "point": _valor(info, "point"),
            "digits": _valor(info, "digits"),
        }, ""
    finally:
        if initialized:
            mt5.shutdown()


def _fallback_broker_metadata():
    return {
        "name": "EURUSD",
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "trade_contract_size": 100000.0,
        "trade_tick_size": 0.00001,
        "trade_tick_value": 1.0,
        "point": 0.00001,
        "digits": 5,
    }


def _sizing(entry, stop, capital, broker_info=None):
    broker_info = broker_info or _fallback_broker_metadata()
    manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))
    return manager.calculate_position_size(
        symbol=broker_info.get("name", "EURUSD"),
        direction=BUY,
        entry=entry,
        stop_loss=stop,
        balance=capital,
        symbol_info=broker_info,
    )


def _signal_id(fecha_signal):
    return f"{CONFIG_ID}|{pd.to_datetime(fecha_signal).isoformat()}"


def _bar_timestamp(value):
    return pd.to_datetime(value).isoformat()


def _normalize_timestamp_set(values):
    normalized = set()
    for value in values or []:
        if value is None or pd.isna(value):
            continue
        normalized.add(_bar_timestamp(value))
    return normalized


def _processed_bar_timestamps(paths, state):
    state_processed = _normalize_timestamp_set(state.get("processed_bar_timestamps", []))
    signals = _read_csv(paths["signals"], SIGNALS_COLUMNS)
    signal_processed = set()
    if not signals.empty and "FECHA_SIGNAL" in signals.columns:
        signal_processed = _normalize_timestamp_set(signals["FECHA_SIGNAL"].dropna().tolist())
    return state_processed | signal_processed


def _sorted_timestamps(values):
    return sorted(_normalize_timestamp_set(values), key=lambda item: pd.to_datetime(item))


def _mark_bar_processed(paths, state, timestamp):
    processed = _processed_bar_timestamps(paths, state)
    processed.add(_bar_timestamp(timestamp))
    ordered = _sorted_timestamps(processed)
    state["processed_bar_timestamps"] = ordered
    state["last_processed_bar"] = ordered[-1] if ordered else None


def _append_signal(paths, row):
    signals = _read_csv(paths["signals"], SIGNALS_COLUMNS)
    if not signals.empty and row["SIGNAL_ID"] in set(signals["SIGNAL_ID"].astype(str)):
        return
    _append_csv(paths["signals"], [row], SIGNALS_COLUMNS)


def _notify_once(state, event_id, message, root=None, event_type="PAPER", asset=ASSET, config_id=CONFIG_ID):
    sent = state.setdefault("telegram_sent", [])
    if event_id in sent:
        return

    root = root or Path(__file__).resolve().parents[1]
    try:
        from src.telegram_events import send_telegram_event

        result = send_telegram_event(
            root,
            event_id,
            event_type,
            message,
            asset=asset,
            config_id=config_id,
        )
        if result.get("sent") or result.get("duplicate"):
            sent.append(event_id)
    except Exception:
        return


def _has_open_position(paths):
    open_positions = _read_csv(paths["open_positions"], OPEN_COLUMNS)
    return not open_positions.empty


def _open_pending_if_possible(paths, state, row, broker_info):
    pending = state.get("pending_signal")
    if not pending or _has_open_position(paths):
        return None
    entry = float(row["Open"])
    atr = float(pending["ATR"])
    stop = entry - atr
    tp = entry + (2 * atr)
    capital = _capital(paths)
    sizing = _sizing(entry, stop, capital, broker_info)
    signal_row = {
        "SIGNAL_ID": pending["SIGNAL_ID"],
        "CONFIG_ID": CONFIG_ID,
        "ACTIVO": ASSET,
        "DIRECCION": DIRECTION,
        "MODELO": MODEL_NAME,
        "SISTEMA": SYSTEM,
        "THRESHOLD": THRESHOLD,
        "FECHA_SIGNAL": pending["FECHA_SIGNAL"],
        "SIGNAL": "LONG",
        "EXECUTION": "OPEN" if sizing.ok else "SKIP_RISK",
        "PROBABILIDAD": pending["PROBABILIDAD"],
        "ENTRY": entry,
        "STOP": stop,
        "TP": tp,
        "ATR": atr,
        "VOLUME": sizing.volume,
        "VOLUME_MIN": sizing.volume_min,
        "RISK_USD": sizing.risk_amount,
        "REASON": sizing.reason,
    }
    _append_signal(paths, signal_row)
    state["pending_signal"] = None
    state["last_signal"] = "LONG"
    state["last_execution"] = signal_row["EXECUTION"]
    state["last_reason"] = signal_row["REASON"]
    if not sizing.ok:
        _notify_once(
            state,
            f"SKIP_RISK|{pending['SIGNAL_ID']}",
            format_skip_risk_message(signal_row),
            root=paths["paper_dir"].parents[1],
            event_type="SKIP_RISK",
        )
        return "SKIP_RISK"

    position = {
        "POSITION_ID": f"POS|{pending['SIGNAL_ID']}",
        "SIGNAL_ID": pending["SIGNAL_ID"],
        "CONFIG_ID": CONFIG_ID,
        "ACTIVO": ASSET,
        "DIRECCION": DIRECTION,
        "FECHA_SIGNAL": pending["FECHA_SIGNAL"],
        "FECHA_ENTRADA": pd.to_datetime(row["fecha"]).isoformat(),
        "ENTRY": entry,
        "STOP": stop,
        "TP": tp,
        "ATR": atr,
        "PROBABILIDAD": pending["PROBABILIDAD"],
        "VOLUME": sizing.volume,
        "CAPITAL_ANTES": capital,
        "BARS_HELD": 0,
    }
    _write_csv(paths["open_positions"], [position], OPEN_COLUMNS)
    _notify_once(
        state,
        f"OPEN|{pending['SIGNAL_ID']}",
        format_open_message(position),
        root=paths["paper_dir"].parents[1],
        event_type="OPEN",
    )
    return "OPEN"


def _r_net(entry, exit_price, raw_r, atr):
    cost = (abs(entry) + abs(exit_price)) * (COST_BPS / 10000)
    return raw_r - (cost / atr)


def _close_open_positions(paths, row, state=None):
    open_positions = _read_csv(paths["open_positions"], OPEN_COLUMNS)
    if open_positions.empty:
        return 0
    remaining = []
    trades = []
    equity_rows = []
    closed_count = 0
    for _, item in open_positions.iterrows():
        position = item.to_dict()
        entry = float(position["ENTRY"])
        stop = float(position["STOP"])
        tp = float(position["TP"])
        atr = float(position["ATR"])
        bars_held = int(float(position.get("BARS_HELD", 0))) + 1
        high = float(row["High"])
        low = float(row["Low"])
        exit_price = None
        reason = None
        if low <= stop:
            exit_price = stop
            reason = "SL"
        elif high >= tp:
            exit_price = tp
            reason = "TP"
        elif bars_held >= MAX_HOLD_BARS:
            exit_price = float(row["Close"])
            reason = "TIME"

        if exit_price is None:
            position["BARS_HELD"] = bars_held
            remaining.append(position)
            continue

        raw_r = (exit_price - entry) / atr
        net_r = _r_net(entry, exit_price, raw_r, atr)
        capital_before = _capital(paths)
        capital_after = capital_before + (capital_before * RISK_PER_TRADE * net_r)
        trade = {
            "TRADE_ID": f"TRD|{position['POSITION_ID']}|{pd.to_datetime(row['fecha']).isoformat()}",
            "POSITION_ID": position["POSITION_ID"],
            "SIGNAL_ID": position["SIGNAL_ID"],
            "CONFIG_ID": CONFIG_ID,
            "ACTIVO": ASSET,
            "DIRECCION": DIRECTION,
            "FECHA_SIGNAL": position["FECHA_SIGNAL"],
            "FECHA_ENTRADA": position["FECHA_ENTRADA"],
            "FECHA_SALIDA": pd.to_datetime(row["fecha"]).isoformat(),
            "ENTRY": entry,
            "STOP": stop,
            "TP": tp,
            "EXIT": exit_price,
            "RESULTADO": "WIN" if raw_r > 0 else "LOSS",
            "R_BRUTO": raw_r,
            "R_NETO": net_r,
            "CAPITAL_ANTES": capital_before,
            "CAPITAL_DESPUES": capital_after,
            "MOTIVO_SALIDA": reason,
        }
        trades.append(trade)
        equity_rows.append(
            {
                "FECHA": pd.to_datetime(row["fecha"]).isoformat(),
                "CONFIG_ID": CONFIG_ID,
                "CAPITAL": capital_after,
            }
        )
        if state is not None:
            _notify_once(
                state,
                f"CLOSED|{trade['TRADE_ID']}",
                format_close_message(trade),
                root=paths["paper_dir"].parents[1],
                event_type="CLOSE",
            )
        closed_count += 1
    _write_csv(paths["open_positions"], remaining, OPEN_COLUMNS)
    _append_csv(paths["trades"], trades, TRADES_COLUMNS)
    _append_csv(paths["equity"], equity_rows, EQUITY_COLUMNS)
    return closed_count


def _create_wait_signal(paths, row, probability):
    signal_row = {
        "SIGNAL_ID": _signal_id(row["fecha"]),
        "CONFIG_ID": CONFIG_ID,
        "ACTIVO": ASSET,
        "DIRECCION": DIRECTION,
        "MODELO": MODEL_NAME,
        "SISTEMA": SYSTEM,
        "THRESHOLD": THRESHOLD,
        "FECHA_SIGNAL": pd.to_datetime(row["fecha"]).isoformat(),
        "SIGNAL": "WAIT",
        "EXECUTION": "WAIT",
        "PROBABILIDAD": probability,
        "ENTRY": None,
        "STOP": None,
        "TP": None,
        "ATR": row["ATR"],
        "VOLUME": None,
        "VOLUME_MIN": None,
        "RISK_USD": None,
        "REASON": "PROBABILITY_BELOW_THRESHOLD",
    }
    _append_signal(paths, signal_row)


def process_eurusd_forward(root, now=None, broker_info=None):
    paths = _ensure_files(root)
    start = _ensure_start(root)
    freeze = _assert_frozen_model(root)
    state = _read_json(paths["state"], {})
    model = _load_model(root)
    context = _mt5_data_context(root, now=now)
    diagnostics = context["diagnostics"]
    state.update(
        {
            "data_source": diagnostics["data_source"],
            "data_status": diagnostics["data_status"],
            "utc_now": diagnostics["utc_now"],
            "last_mt5_bar": diagnostics["last_mt5_bar"],
            "last_closed_bar": diagnostics["last_closed_bar"],
            "first_full_forward_bar": diagnostics["first_full_forward_bar"],
            "last_data_error": diagnostics["error"],
        }
    )
    if not context["ok"]:
        state["pending_bars"] = 0
        state["last_execution"] = diagnostics["data_status"]
        state["last_reason"] = diagnostics["error"] or diagnostics["data_status"]
        notify_critical_error(
            root,
            ASSET,
            diagnostics["data_status"],
            diagnostics["error"] or diagnostics["data_status"],
        )
        _write_json(paths["state"], state)
        return {
            "processed": 0,
            "processed_bar_timestamps": [],
            "pending_found": 0,
            "eligible_bars_total": 0,
            "processed_bars_before": 0,
            "opened": 0,
            "closed": 0,
            "skipped_risk": 0,
            "freeze": freeze,
            "data_source": diagnostics["data_source"],
            "data_status": diagnostics["data_status"],
            "data_error": diagnostics["error"],
        }

    data = context["features"]
    first_full = _first_full_forward_bar(start["start_time_utc"])
    eligible = data[data["fecha"] >= first_full].copy()
    processed_timestamps = _processed_bar_timestamps(paths, state)
    if not processed_timestamps and state.get("last_processed_bar"):
        last_processed = _to_naive_utc(state["last_processed_bar"])
        processed_timestamps = _normalize_timestamp_set(
            eligible.loc[eligible["fecha"] <= last_processed, "fecha"].tolist()
        )
    processed_bars_before = int(len(processed_timestamps))
    data = eligible[
        ~eligible["fecha"].map(lambda value: _bar_timestamp(value)).isin(processed_timestamps)
    ].copy()
    data = data.sort_values("fecha").reset_index(drop=True)
    state["pending_bars"] = int(len(data))
    state["first_full_forward_bar"] = pd.to_datetime(first_full).isoformat()
    state["eligible_bars_total"] = int(len(eligible))
    state["processed_bars_total"] = int(len(processed_timestamps))
    state["missing_bar_timestamps"] = [
        _bar_timestamp(value) for value in data["fecha"].tolist()
    ]

    metadata = broker_info or context["broker"]
    if metadata is None:
        state["data_status"] = "ERROR"
        state["last_execution"] = "ERROR"
        state["last_reason"] = "BROKER_METADATA_UNAVAILABLE"
        notify_critical_error(root, ASSET, "ERROR", "BROKER_METADATA_UNAVAILABLE")
        _write_json(paths["state"], state)
        return {
            "processed": 0,
            "processed_bar_timestamps": [],
            "pending_found": 0,
            "eligible_bars_total": int(len(eligible)),
            "processed_bars_before": processed_bars_before,
            "opened": 0,
            "closed": 0,
            "skipped_risk": 0,
            "freeze": freeze,
            "data_source": diagnostics["data_source"],
            "data_status": "ERROR",
            "data_error": "BROKER_METADATA_UNAVAILABLE",
        }

    processed = opened = closed = skipped = 0
    processed_this_run = []
    for _, row in data.iterrows():
        closed += _close_open_positions(paths, row, state)
        action = _open_pending_if_possible(paths, state, row, metadata)
        if action == "OPEN":
            opened += 1
        elif action == "SKIP_RISK":
            skipped += 1

        probability = _probability_long(model, row)
        state["last_probability_long"] = probability
        if probability >= THRESHOLD and not _has_open_position(paths) and not state.get("pending_signal"):
            state["pending_signal"] = {
                "SIGNAL_ID": _signal_id(row["fecha"]),
                "FECHA_SIGNAL": pd.to_datetime(row["fecha"]).isoformat(),
                "PROBABILIDAD": probability,
                "ATR": float(row["ATR"]),
            }
            state["last_signal"] = "LONG"
            state["last_execution"] = "LONG"
            state["last_reason"] = "PENDING_NEXT_BAR"
            _notify_once(
                state,
                f"NEW_SIGNAL|{_signal_id(row['fecha'])}",
                format_new_signal_message(
                    {
                        "CONFIG_ID": CONFIG_ID,
                        "ACTIVO": ASSET,
                        "DIRECCION": DIRECTION,
                        "THRESHOLD": THRESHOLD,
                        "FECHA_SIGNAL": pd.to_datetime(row["fecha"]).isoformat(),
                        "SIGNAL": "LONG",
                        "PROBABILIDAD": probability,
                    }
                ),
                root=root,
                event_type="SIGNAL",
            )
        else:
            _create_wait_signal(paths, row, probability)
            if probability < THRESHOLD:
                state["last_signal"] = "WAIT"
                state["last_execution"] = "WAIT"
                state["last_reason"] = "PROBABILITY_BELOW_THRESHOLD"

        _mark_bar_processed(paths, state, row["fecha"])
        processed_this_run.append(_bar_timestamp(row["fecha"]))
        processed += 1

    processed_timestamps = _processed_bar_timestamps(paths, state)
    remaining = eligible[
        ~eligible["fecha"].map(lambda value: _bar_timestamp(value)).isin(processed_timestamps)
    ]
    state["processed_bar_timestamps"] = _sorted_timestamps(processed_timestamps)
    state["pending_bars"] = int(len(remaining))
    state["missing_bar_timestamps"] = [
        _bar_timestamp(value) for value in remaining["fecha"].tolist()
    ]
    state["processed_bars_total"] = len(state.get("processed_bar_timestamps", []))
    _write_json(paths["state"], state)
    return {
        "processed": processed,
        "processed_bar_timestamps": processed_this_run,
        "pending_found": int(len(data)),
        "eligible_bars_total": int(len(eligible)),
        "processed_bars_before": processed_bars_before,
        "opened": opened,
        "closed": closed,
        "skipped_risk": skipped,
        "freeze": freeze,
        "data_source": diagnostics["data_source"],
        "data_status": diagnostics["data_status"],
        "data_error": diagnostics["error"],
        "broker_metadata_source": "MT5_DEMO",
        "broker_metadata_error": "",
    }


def _metrics(paths):
    trades = _read_csv(paths["trades"], TRADES_COLUMNS)
    equity = _read_csv(paths["equity"], EQUITY_COLUMNS)
    capital = CAPITAL_INITIAL if equity.empty else float(equity.iloc[-1]["CAPITAL"])
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "r_total": 0.0,
            "max_drawdown": 0.0,
            "capital": capital,
        }
    r_values = trades["R_NETO"].astype(float)
    wins = int((r_values > 0).sum())
    losses = int((r_values <= 0).sum())
    gross_profit = r_values[r_values > 0].sum()
    gross_loss = abs(r_values[r_values < 0].sum())
    profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else 0.0
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    capital_values = equity["CAPITAL"].astype(float).tolist() if not equity.empty else [CAPITAL_INITIAL]
    peak = capital_values[0]
    max_dd = 0.0
    for value in capital_values:
        peak = max(peak, value)
        if peak:
            max_dd = max(max_dd, (peak - value) / peak)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades)) * 100,
        "profit_factor": profit_factor,
        "expectancy_r": float(r_values.mean()),
        "r_total": float(r_values.sum()),
        "max_drawdown": max_dd * 100,
        "capital": capital,
    }


def eurusd_forward_status(root):
    paths = _ensure_files(root)
    start = _ensure_start(root)
    freeze = _assert_frozen_model(root)
    state = _read_json(paths["state"], {})
    context = _mt5_data_context(root)
    diagnostics = context["diagnostics"]
    data = context["features"] if context["ok"] else pd.DataFrame()
    first_full = _first_full_forward_bar(start["start_time_utc"])
    pending = pd.DataFrame()
    eligible = pd.DataFrame()
    processed_timestamps = _processed_bar_timestamps(paths, state)
    if not data.empty:
        eligible = data[data["fecha"] >= first_full].copy()
        if not processed_timestamps and state.get("last_processed_bar"):
            last_processed = _to_naive_utc(state["last_processed_bar"])
            processed_timestamps = _normalize_timestamp_set(
                eligible.loc[eligible["fecha"] <= last_processed, "fecha"].tolist()
            )
        pending = eligible[
            ~eligible["fecha"].map(lambda value: _bar_timestamp(value)).isin(processed_timestamps)
        ].copy()
    open_positions = _read_csv(paths["open_positions"], OPEN_COLUMNS)
    signals = _read_csv(paths["signals"], SIGNALS_COLUMNS)
    last_signal_row = signals.iloc[-1].to_dict() if not signals.empty else {}
    metrics = _metrics(paths)
    broker_info = context["broker"]
    sample_sizing = None
    if broker_info is not None and not data.empty:
        sample_entry = float(data.iloc[-1]["Close"])
        sample_atr = float(data.iloc[-1]["ATR"])
        sample_sizing = _sizing(sample_entry, sample_entry - sample_atr, CAPITAL_INITIAL, broker_info)
    return {
        "asset": ASSET,
        "direction": DIRECTION,
        "model": MODEL_NAME,
        "system": SYSTEM,
        "threshold": THRESHOLD,
        "forward_start": start["start_time_utc"],
        "forward_start_original": start["start_time_utc"],
        "first_full_forward_bar": diagnostics.get("first_full_forward_bar") or pd.to_datetime(first_full).isoformat(),
        "training_end": freeze["metadata"].get("fin_datos_entrenamiento"),
        "model_path": str(freeze["path"]),
        "model_hash": freeze["metadata"].get("hash_modelo"),
        "data_source": diagnostics["data_source"],
        "data_status": diagnostics["data_status"],
        "data_error": diagnostics["error"],
        "utc_now": diagnostics["utc_now"],
        "last_mt5_bar": diagnostics["last_mt5_bar"],
        "last_closed_bar": diagnostics["last_closed_bar"],
        "server_utc_offset_hours": diagnostics.get("server_utc_offset_hours"),
        "pending_bars": int(len(pending)),
        "eligible_bars_total": int(len(eligible)),
        "processed_bars_total": int(len(processed_timestamps)),
        "pending_bar_timestamps": [
            _bar_timestamp(value) for value in pending["fecha"].tolist()
        ] if not pending.empty else [],
        "processed_bar_timestamps": _sorted_timestamps(processed_timestamps),
        "last_available_bar": diagnostics["last_mt5_bar"],
        "last_processed_bar": state.get("last_processed_bar"),
        "probability_long": state.get("last_probability_long"),
        "signal": state.get("last_signal", last_signal_row.get("SIGNAL", "WAIT")),
        "execution": "OPEN" if not open_positions.empty else state.get("last_execution", last_signal_row.get("EXECUTION", "WAIT")),
        "reason": state.get("last_reason", last_signal_row.get("REASON", "")),
        "sample_insufficient": metrics["trades"] < 30,
        "metrics": metrics,
        "broker": {
            "metadata_error": diagnostics["error"] if broker_info is None else "",
            "volume_min": broker_info.get("volume_min") if broker_info else None,
            "volume_step": broker_info.get("volume_step") if broker_info else None,
            "contract_size": broker_info.get("trade_contract_size") if broker_info else None,
            "tick_size": broker_info.get("trade_tick_size") if broker_info else None,
            "tick_value": broker_info.get("trade_tick_value") if broker_info else None,
            "capital": CAPITAL_INITIAL,
            "risk_pct": RISK_PER_TRADE * 100,
            "sample_volume": sample_sizing.volume if sample_sizing else None,
            "sample_decision": sample_sizing.decision if sample_sizing else "ERROR",
            "sample_reason": sample_sizing.reason if sample_sizing else diagnostics["error"],
        },
        "trading_enabled": TRADING_ENABLED,
        "demo_execution_enabled": DEMO_EXECUTION_ENABLED,
        "orders_sent": 0,
    }
