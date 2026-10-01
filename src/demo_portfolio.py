"""Portafolio DEMO multi-activo: compra y venta automaticas en MT5 demo.

Es un experimento de EXPLORACION, separado de los forward congelados (GOLD T060/T055
y EURUSD T065 siguen intactos). Sirve para probar la mecanica completa en demo con
varias monedas y ambas direcciones. Ningun modelo de este portafolio esta validado
para dinero real.

Por cada activo y cada vela H1 cerrada:
- probabilidad LONG y SHORT con un modelo LOGISTIC por direccion;
- si la mayor supera el threshold, se envia la orden (SL 1 ATR, TP 2 ATR);
- las posiciones se cierran por SL/TP en el broker o por tiempo (24 velas).
"""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from config import (
    DEMO_EXECUTION_ENABLED,
    DEMO_PORTFOLIO_ASSETS,
    DEMO_PORTFOLIO_MAX_POSITIONS,
    DEMO_PORTFOLIO_THRESHOLD,
    MT5_SERVER_UTC_OFFSETS,
)
from src.features import crear_dataset_ml
from src.indicators import calcular_indicadores
from src.live_executor import (
    OrderIntent,
    close_expired_positions,
    execute_signal,
    manage_protective_exits,
    mt5_session,
    notify_closed_deals,
    telegram_notifier,
)
from src.model import FEATURES_ML, _crear_modelo
from src.risk import BUY, SELL, RiskLimits, RiskManager
from src.symbols import discover_symbols


CONFIG_PREFIX = "DEMO_PORTFOLIO"
MODEL_NAME = "LOGISTIC"
STOP_ATR = 1.0
TAKE_PROFIT_ATR = 2.0
MAX_HOLD_BARS = 24
MT5_RATES_COUNT = 1000
HOLDOUT_FRACTION = 0.2
EMBARGO_BARS = 24
DIRECTIONS = {"LONG": ("TARGET_LONG", BUY), "SHORT": ("TARGET_SHORT", SELL)}
SIGNAL_COLUMNS = [
    "FECHA_SIGNAL", "ASSET", "SYMBOL", "PROB_LONG", "PROB_SHORT", "DECISION", "SIGNAL_ID",
    "EVENT_STATUS", "SKIP_REASON",
]
SHADOW_COLUMNS = [
    "FECHA_SIGNAL", "ASSET", "SYMBOL", "DECISION", "SIGNAL_ID", "SKIP_REASON",
    "PROB_LONG", "PROB_SHORT",
]


def model_dir(root):
    return Path(root) / "models" / "demo"


def model_path(root, asset, direction):
    return model_dir(root) / f"{asset.lower()}_{direction.lower()}_logistic.pkl"


def config_id(asset, direction):
    return f"{CONFIG_PREFIX}_{asset}_{direction}_{MODEL_NAME}_T{int(DEMO_PORTFOLIO_THRESHOLD * 100):03d}"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_local_prices(root, asset):
    path = Path(root) / "data" / asset.lower() / f"{asset.lower()}_h1.csv"
    prices = pd.read_csv(path, index_col=0)
    # Todo en UTC naive: en vivo las velas MT5 tambien se normalizan a UTC (HORA coherente).
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    return prices[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()


def _holdout_report(dataset, target):
    cut = int(len(dataset) * (1 - HOLDOUT_FRACTION))
    train = dataset.iloc[: max(0, cut - EMBARGO_BARS)]
    test = dataset.iloc[cut:]
    model = _crear_modelo(MODEL_NAME).fit(train[FEATURES_ML], train[target].astype(int))
    probs = model.predict_proba(test[FEATURES_ML])[:, list(model.classes_).index(1)]
    hits = test[target][probs >= DEMO_PORTFOLIO_THRESHOLD].astype(int)
    test_days = max(1, test["fecha"].dt.date.nunique())
    return {
        "holdout_start": test["fecha"].iloc[0].isoformat(),
        "holdout_signals": int(len(hits)),
        "holdout_signals_per_day": round(len(hits) / test_days, 2),
        "holdout_win_rate": round(float(hits.mean()), 4) if len(hits) else None,
    }


def train_portfolio_models(root, assets=None, data_source="yfinance"):
    """Entrena LONG y SHORT por activo con todo el historico local y guarda metadata."""
    assets = assets or DEMO_PORTFOLIO_ASSETS
    model_dir(root).mkdir(parents=True, exist_ok=True)
    report = []
    for asset in assets:
        dataset = crear_dataset_ml(calcular_indicadores(load_local_prices(root, asset)))
        for direction, (target, _) in DIRECTIONS.items():
            data = dataset.dropna(subset=FEATURES_ML + [target]).reset_index(drop=True)
            holdout = _holdout_report(data, target)
            model = _crear_modelo(MODEL_NAME).fit(data[FEATURES_ML], data[target].astype(int))
            path = model_path(root, asset, direction)
            joblib.dump(model, path)
            metadata = {
                "config_id": config_id(asset, direction),
                "activo": asset,
                "direccion": direction,
                "modelo": MODEL_NAME,
                "threshold": DEMO_PORTFOLIO_THRESHOLD,
                "features": FEATURES_ML,
                "timeframe": "H1",
                "stop_atr": STOP_ATR,
                "take_profit_atr": TAKE_PROFIT_ATR,
                "max_hold_bars": MAX_HOLD_BARS,
                "data_source": data_source,
                "train_start": data["fecha"].iloc[0].isoformat(),
                "train_end": data["fecha"].iloc[-1].isoformat(),
                "train_rows": int(len(data)),
                "hash_modelo": _sha256(path),
                "uso": "EXPLORACION DEMO - no validado para cuenta real",
                **holdout,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            report.append(metadata)
    return report


def load_portfolio_models(root, assets=None):
    models = {}
    for asset in assets or DEMO_PORTFOLIO_ASSETS:
        for direction in DIRECTIONS:
            path = model_path(root, asset, direction)
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if _sha256(path) != metadata["hash_modelo"]:
                raise RuntimeError(f"ABORTAR: hash del modelo demo {asset} {direction} no coincide.")
            if metadata["features"] != FEATURES_ML:
                raise RuntimeError(f"ABORTAR: features del modelo demo {asset} {direction} no coinciden.")
            models[(asset, direction)] = joblib.load(path)
    return models


def audit_portfolio_models(root, assets=None):
    """Inventario de modelos demo disponibles y compatibilidad con el scanner."""
    rows = []
    for asset in assets or DEMO_PORTFOLIO_ASSETS:
        for direction in DIRECTIONS:
            path = model_path(root, asset, direction)
            metadata_path = path.with_suffix(".json")
            row = {
                "asset": asset,
                "direction": direction,
                "model_path": str(path),
                "metadata_path": str(metadata_path),
                "exists": path.exists(),
                "metadata_exists": metadata_path.exists(),
                "compatible": False,
                "reason": "",
                "timeframe": None,
                "features": None,
                "config_id": config_id(asset, direction),
            }
            if not path.exists():
                row["reason"] = "MODEL_FILE_MISSING"
            elif not metadata_path.exists():
                row["reason"] = "METADATA_MISSING"
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                row.update({
                    "timeframe": metadata.get("timeframe"),
                    "features": metadata.get("features"),
                    "config_id": metadata.get("config_id", row["config_id"]),
                })
                if _sha256(path) != metadata.get("hash_modelo"):
                    row["reason"] = "MODEL_HASH_MISMATCH"
                elif metadata.get("features") != FEATURES_ML:
                    row["reason"] = "FEATURES_MISMATCH"
                elif metadata.get("timeframe") != "H1":
                    row["reason"] = "TIMEFRAME_MISMATCH"
                else:
                    row["compatible"] = True
                    row["reason"] = "OK"
            rows.append(row)
    return rows


def _probability(model, row):
    x = pd.DataFrame([row[FEATURES_ML].astype(float).to_dict()])
    classes = list(model.classes_)
    return float(model.predict_proba(x)[0][classes.index(1)]) if 1 in classes else 0.0


def decide(prob_long, prob_short, threshold=None):
    """LONG, SHORT o WAIT: gana la direccion mas probable si supera el threshold."""
    threshold = DEMO_PORTFOLIO_THRESHOLD if threshold is None else threshold
    if max(prob_long, prob_short) < threshold:
        return "WAIT"
    return "LONG" if prob_long >= prob_short else "SHORT"


def infer_server_offset(mt5, symbols, now):
    """Desfase servidor-UTC (horas) usando el tick mas reciente de todos los simbolos.

    None si no hay ticks recientes (mercado cerrado / MT5 desconectado) o si el desfase
    no es uno de MT5_SERVER_UTC_OFFSETS.
    """
    times = [int(getattr(mt5.symbol_info_tick(symbol), "time", 0) or 0) for symbol in symbols]
    times = [t for t in times if t > 0]
    if not times:
        return None
    delta_hours = (max(times) - pd.to_datetime(now, utc=True).timestamp()) / 3600
    offset = round(delta_hours)
    # Tick de hace mas de 15 minutos: mercado quieto o cerrado.
    if abs(delta_hours - offset) > 0.25 or offset not in MT5_SERVER_UTC_OFFSETS:
        return None
    return offset


def closed_h1_features(mt5, symbol, now, server_offset):
    """Features de velas H1 cerradas desde MT5, en UTC. None si no hay vela fresca."""
    from src.eurusd_forward import _rates_to_prices

    if server_offset is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, MT5_RATES_COUNT)
    if rates is None or len(rates) == 0:
        return None
    prices = _rates_to_prices(rates)
    prices.index = prices.index - pd.Timedelta(hours=server_offset)
    current_bar = pd.to_datetime(now, utc=True).floor("h").tz_convert(None)
    last_closed = current_bar - pd.Timedelta(hours=1)
    # Mercado activo: debe existir la vela en formacion de esta hora. Evita operar con
    # un feed congelado o con el mercado cerrado (fin de semana, pausa diaria del oro).
    if prices.empty or prices.index.max() != current_bar:
        return None
    prices = prices[prices.index <= last_closed]
    if prices.empty or prices.index.max() != last_closed:
        return None
    features = crear_dataset_ml(calcular_indicadores(prices))
    return features if not features.empty and features["fecha"].iloc[-1] == last_closed else None


def _append_signal(root, row):
    path = Path(root) / "live" / "portfolio_signals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, usecols=["FECHA_SIGNAL", "ASSET"])
        if ((existing["FECHA_SIGNAL"] == row["FECHA_SIGNAL"]) & (existing["ASSET"] == row["ASSET"])).any():
            return
    pd.DataFrame([row], columns=SIGNAL_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_shadow_signal(root, row):
    path = Path(root) / "live" / "shadow_signals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, usecols=["SIGNAL_ID", "SKIP_REASON"])
        if ((existing["SIGNAL_ID"] == row["SIGNAL_ID"]) & (existing["SKIP_REASON"] == row["SKIP_REASON"])).any():
            return
    pd.DataFrame([row], columns=SHADOW_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_portfolio_live(root, dry_run=None, mt5_factory=mt5_session, now=None, models=None):
    dry_run = (not DEMO_EXECUTION_ENABLED) if dry_run is None else dry_run
    now = pd.to_datetime(now or pd.Timestamp.now(tz="UTC"), utc=True)
    models = models or load_portfolio_models(root)
    risk_manager = RiskManager(RiskLimits(maximum_simultaneous_positions=DEMO_PORTFOLIO_MAX_POSITIONS))
    notify = None if dry_run else telegram_notifier(root, "PORTFOLIO", CONFIG_PREFIX)

    results = []
    with mt5_factory() as mt5:
        closed_deals = notify_closed_deals(mt5, notify)
        resolved = discover_symbols(mt5, DEMO_PORTFOLIO_ASSETS)
        found = [r.mt5_symbol for r in resolved.values() if r.found]
        for symbol in found:
            mt5.symbol_select(symbol, True)
        protective_updates = manage_protective_exits(mt5, root, dry_run=dry_run, notify=notify)
        server_offset = infer_server_offset(mt5, found, now)
        for asset in DEMO_PORTFOLIO_ASSETS:
            symbol = resolved[asset].mt5_symbol
            item = {"asset": asset, "symbol": symbol, "decision": "NO_DATA", "execution": None, "closed": []}
            results.append(item)
            if not resolved[asset].found:
                item["decision"] = "SYMBOL_NOT_FOUND"
                continue
            item["closed"] = close_expired_positions(
                mt5, root, symbols={symbol}, max_hold_bars=MAX_HOLD_BARS, dry_run=dry_run, notify=notify,
            )

            features = closed_h1_features(mt5, symbol, now, server_offset)
            if features is None:
                continue
            row = features.iloc[-1]
            prob_long = _probability(models[(asset, "LONG")], row)
            prob_short = _probability(models[(asset, "SHORT")], row)
            decision = decide(prob_long, prob_short)
            fecha = pd.to_datetime(row["fecha"]).isoformat()
            signal_id = f"{config_id(asset, decision)}|{fecha}" if decision != "WAIT" else ""
            item.update({"decision": decision, "prob_long": prob_long, "prob_short": prob_short, "fecha": fecha})
            _append_signal(root, {
                "FECHA_SIGNAL": fecha, "ASSET": asset, "SYMBOL": symbol, "PROB_LONG": prob_long,
                "PROB_SHORT": prob_short, "DECISION": decision, "SIGNAL_ID": signal_id,
                "EVENT_STATUS": "SIGNAL_GENERATED", "SKIP_REASON": "",
            })
            if decision == "WAIT":
                continue

            intent = OrderIntent(
                signal_id=signal_id,
                config_id=config_id(asset, decision),
                asset=asset,
                direction=DIRECTIONS[decision][1],
                atr=float(row["ATR"]),
                stop_atr=STOP_ATR,
                take_profit_atr=TAKE_PROFIT_ATR,
            )
            item["execution"] = execute_signal(
                mt5, intent, root, dry_run=dry_run, risk_manager=risk_manager, notify=notify,
            )
            if item["execution"].status == "SKIP":
                _append_shadow_signal(root, {
                    "FECHA_SIGNAL": fecha,
                    "ASSET": asset,
                    "SYMBOL": symbol,
                    "DECISION": decision,
                    "SIGNAL_ID": signal_id,
                    "SKIP_REASON": item["execution"].reason,
                    "PROB_LONG": prob_long,
                    "PROB_SHORT": prob_short,
                })

    return {"dry_run": dry_run, "results": results, "closed_deals": closed_deals,
            "server_offset": server_offset, "protective_updates": protective_updates}


def current_atr(mt5, symbol):
    """ATR(14) H1 de las ultimas velas MT5, sin exigir vela fresca (sirve para pruebas)."""
    from src.eurusd_forward import _rates_to_prices

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates is None or len(rates) == 0:
        return None
    atr = calcular_indicadores(_rates_to_prices(rates))["ATR"].dropna()
    return float(atr.iloc[-1]) if not atr.empty else None


def send_test_order(root, asset, direction, dry_run=None, mt5_factory=mt5_session, now=None):
    """Orden manual de prueba (BUY/SELL) con todas las puertas de seguridad del ejecutor."""
    dry_run = (not DEMO_EXECUTION_ENABLED) if dry_run is None else dry_run
    now = pd.to_datetime(now or pd.Timestamp.now(tz="UTC"), utc=True)
    asset = asset.upper()
    direction = direction.upper()
    if direction not in (BUY, SELL):
        raise ValueError("Direccion debe ser BUY o SELL")
    notify = None if dry_run else telegram_notifier(root, asset, "TEST")
    with mt5_factory() as mt5:
        resolved = discover_symbols(mt5, [asset])[asset]
        if not resolved.found:
            raise RuntimeError(f"No se encontro {asset} en MT5: {resolved.reason}")
        mt5.symbol_select(resolved.mt5_symbol, True)
        atr = current_atr(mt5, resolved.mt5_symbol)
        if atr is None:
            raise RuntimeError(f"MT5 no devolvio velas H1 para {resolved.mt5_symbol}")
        intent = OrderIntent(
            signal_id=f"TEST_{asset}_{direction}|{now.isoformat()}",
            config_id=f"TEST_{asset}_{direction}",
            asset=asset,
            direction=direction,
            atr=atr,
            stop_atr=STOP_ATR,
            take_profit_atr=TAKE_PROFIT_ATR,
        )
        return execute_signal(mt5, intent, root, dry_run=dry_run, notify=notify)
