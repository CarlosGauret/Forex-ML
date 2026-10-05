"""Investigacion del portafolio DEMO con datos del broker (MT5) y walk-forward anual.

Corrige las debilidades del holdout unico con yfinance:
- datos H1 del propio broker (XM) desde 2010, con el spread real de cada vela;
- walk-forward anual: cada año se predice con un modelo entrenado solo con años anteriores;
- simulacion de TODAS las operaciones (SL, TP y cierre por tiempo a 24 velas), una posicion
  por simbolo, con costo de spread x SPREAD_COST_MULTIPLIER;
- seleccion de variantes solo con el periodo SELECTION (hasta 2023) y confirmacion en un
  periodo que la seleccion nunca vio (CONFIRMATION, 2024 en adelante).

Una direccion solo queda habilitada si pasa ambos periodos. El resultado se guarda en
models/demo/portfolio_selection.json y el runner en vivo lo respeta.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import _crear_target, crear_dataset_ml
from src.indicators import calcular_indicadores
from src.model import FEATURES_ML, _crear_modelo


TREND_EMA_BARS = 1200  # ~EMA50 diaria en velas H1
TREND_SLOPE_BARS = 120
SPREAD_COST_MULTIPLIER = 1.5  # el spread de la vela MT5 es optimista; margen para slippage
STOP_ATR = 1.0
TAKE_PROFIT_ATR = 2.0
MAX_HOLD_BARS = 24
# Tamaños de stop evaluados (en ATR H1). TP = 2x stop y tiempo maximo = 24 velas x stop:
# stops mas anchos reducen el peso del spread en R.
STOP_MULTIPLIERS = (1.0, 2.0, 3.0)
EMBARGO_BARS = 24
MIN_TRAIN_YEARS = 3
CONFIRMATION_START = pd.Timestamp("2024-01-01")
THRESHOLDS = (0.55, 0.60, 0.65)
TREND_FILTERS = (False, True)
MT5_HISTORY_BARS = 99999

# Criterios de habilitacion (en R, despues de costos)
SELECTION_MIN_TRADES = 60
SELECTION_MIN_EXPECTANCY = 0.05
SELECTION_MIN_PROFIT_FACTOR = 1.15
SELECTION_MIN_POSITIVE_YEARS = 0.6
YEAR_MIN_TRADES = 5
CONFIRMATION_MIN_TRADES = 20
CONFIRMATION_MIN_EXPECTANCY = 0.0


def mt5_data_path(root, asset):
    return Path(root) / "data" / "mt5" / f"{asset.lower()}_h1.csv"


def selection_path(root):
    return Path(root) / "models" / "demo" / "portfolio_selection.json"


def server_to_utc(server_times):
    """XM: GMT+3 con horario de verano de Nueva York, GMT+2 el resto del año."""
    server_times = pd.DatetimeIndex(server_times)
    guess = (server_times - pd.Timedelta(hours=2)).tz_localize("UTC")
    dst = guess.tz_convert("America/New_York").map(lambda t: t.dst() != pd.Timedelta(0))
    offsets = np.where(np.asarray(dst, dtype=bool), 3, 2)
    return (server_times - pd.to_timedelta(offsets, unit="h"))


def download_mt5_history(root, assets, mt5=None):
    """Descarga velas H1 del broker y las guarda en UTC con el spread en unidades de precio."""
    import MetaTrader5

    from src.symbols import discover_symbols

    mt5 = mt5 or MetaTrader5
    if not mt5.initialize():
        raise RuntimeError(f"No se pudo iniciar MT5: {mt5.last_error()}")
    report = {}
    try:
        resolved = discover_symbols(mt5, assets)
        for asset in assets:
            if not resolved[asset].found:
                report[asset] = "SYMBOL_NOT_FOUND"
                continue
            symbol = resolved[asset].mt5_symbol
            mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, MT5_HISTORY_BARS)
            if rates is None or len(rates) == 0:
                report[asset] = f"NO_RATES {mt5.last_error()}"
                continue
            point = float(mt5.symbol_info(symbol).point)
            data = pd.DataFrame(rates)
            server = pd.to_datetime(data["time"], unit="s")
            prices = pd.DataFrame({
                "Open": data["open"].to_numpy(),
                "High": data["high"].to_numpy(),
                "Low": data["low"].to_numpy(),
                "Close": data["close"].to_numpy(),
                "Volume": data["tick_volume"].to_numpy(),
                "Spread": data["spread"].to_numpy() * point,
            }, index=server_to_utc(server))
            prices.index.name = "fecha"
            prices = prices[~prices.index.duplicated(keep="last")].sort_index()
            path = mt5_data_path(root, asset)
            path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_csv(path)
            report[asset] = f"{len(prices)} velas desde {prices.index[0]:%Y-%m-%d}"
    finally:
        mt5.shutdown()
    return report


def load_mt5_prices(root, asset):
    prices = pd.read_csv(mt5_data_path(root, asset), index_col=0, parse_dates=True)
    return prices.sort_index()


def add_trend_columns(dataset, closes):
    """TREND: +1 alcista, -1 bajista, 0 sin tendencia clara (EMA ~D1 y su pendiente)."""
    ema = closes.ewm(span=TREND_EMA_BARS, adjust=False, min_periods=TREND_EMA_BARS).mean()
    slope = ema - ema.shift(TREND_SLOPE_BARS)
    trend = pd.Series(0, index=closes.index)
    trend[(closes > ema) & (slope > 0)] = 1
    trend[(closes < ema) & (slope < 0)] = -1
    trend[ema.isna() | slope.isna()] = 0
    dataset = dataset.copy()
    dataset["TREND"] = trend.reindex(pd.DatetimeIndex(dataset["fecha"])).to_numpy()
    return dataset


def build_dataset(prices):
    dataset = crear_dataset_ml(calcular_indicadores(prices[["Open", "High", "Low", "Close", "Volume"]]))
    dataset = add_strategy_columns(add_trend_columns(dataset, prices["Close"]))
    spread = prices["Spread"] if "Spread" in prices else pd.Series(0.0, index=prices.index)
    dataset["SPREAD"] = spread.reindex(pd.DatetimeIndex(dataset["fecha"])).fillna(0).to_numpy()
    return dataset


def trend_allows(direction, trend):
    return trend == (1 if direction == "LONG" else -1)


def exit_params(stop_atr, tp_ratio=2.0):
    """(stop_atr, take_profit_atr, max_hold_bars) para un multiplicador de stop."""
    return float(stop_atr), float(tp_ratio) * float(stop_atr), int(round(MAX_HOLD_BARS * stop_atr))


# --- Estrategias de reglas clasicas (sin ML) -------------------------------------------
# Todas usan solo velas H1 cerradas y se calculan igual en investigacion y en vivo.
RULE_STRATEGIES = ("MEANREV_RSI2", "PULLBACK_TREND", "BREAKOUT_5D", "BREAKOUT_LONDON")
RULE_STOPS = (1.0, 2.0, 3.0)
RULE_TP_RATIOS = (1.0, 2.0)
BREAKOUT_BARS = 120  # ~5 dias de trading en H1
LONDON_HOURS = (7, 8, 9, 10)  # UTC, apertura de Londres
ASIA_HOURS = range(0, 7)


def add_strategy_columns(dataset):
    """Columnas auxiliares de las estrategias de reglas (sin mirar el futuro)."""
    d = dataset.copy()
    close = d["Close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    d["RSI2"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d["HH_PREV"] = d["High"].astype(float).shift(1).rolling(BREAKOUT_BARS).max()
    d["LL_PREV"] = d["Low"].astype(float).shift(1).rolling(BREAKOUT_BARS).min()
    fecha = pd.to_datetime(d["fecha"])
    day = fecha.dt.normalize()
    asia = fecha.dt.hour.isin(ASIA_HOURS)
    # Rango asiatico del dia: solo se usa en horas de Londres, cuando ya esta completo.
    d["ASIA_HIGH"] = d["High"].where(asia).groupby(day).transform("max")
    d["ASIA_LOW"] = d["Low"].where(asia).groupby(day).transform("min")
    return d


def rule_signals(dataset, strategy, direction):
    """Mascara booleana de entradas (al cierre de la vela) para una estrategia de reglas."""
    d = dataset
    long = direction == "LONG"
    close = d["Close"].astype(float)
    if strategy == "MEANREV_RSI2":
        # Connors: comprar caidas fuertes por encima de la EMA200 (y al reves)
        mask = (d["RSI2"] < 10) & (close > d["EMA200"]) if long else (d["RSI2"] > 90) & (close < d["EMA200"])
    elif strategy == "PULLBACK_TREND":
        # Retroceso dentro de la tendencia ~D1
        mask = (d["TREND"] == 1) & (d["RSI"] < 35) if long else (d["TREND"] == -1) & (d["RSI"] > 65)
    elif strategy == "BREAKOUT_5D":
        # Ruptura del maximo/minimo de ~5 dias (estilo Tortugas en H1)
        mask = close > d["HH_PREV"] if long else close < d["LL_PREV"]
    elif strategy == "BREAKOUT_LONDON":
        hour = pd.to_datetime(d["fecha"]).dt.hour
        level = d["ASIA_HIGH"] if long else d["ASIA_LOW"]
        crossed = (close > level) if long else (close < level)
        first = crossed & ~crossed.shift(1, fill_value=False)
        mask = hour.isin(LONDON_HOURS) & first
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")
    return mask.fillna(False).to_numpy(dtype=bool)


def target_for_stop(dataset, direction, stop_atr):
    """Target TP-antes-que-SL escalado: SL stop_atr ATR, TP 2x, horizonte 24 x stop_atr."""
    scaled = dataset[["Close", "High", "Low"]].copy()
    scaled["ATR"] = dataset["ATR"] * stop_atr
    return pd.array(_crear_target(scaled, direction, exit_params(stop_atr)[2]), dtype="Float64")


def simulate_trades(dataset, signal_mask, direction, stop_atr=STOP_ATR,
                    take_profit_atr=TAKE_PROFIT_ATR, max_hold_bars=MAX_HOLD_BARS):
    """R de cada operacion (despues de costos) con una sola posicion abierta a la vez.

    Velas MT5 = precio bid. LONG compra al ask (close + spread) y sale al bid; SHORT vende al
    bid y recompra al ask. SL/TP a stop_atr/take_profit_atr ATR del precio de entrada y
    cierre por tiempo a las max_hold_bars velas.
    """
    close = dataset["Close"].to_numpy(float)
    high = dataset["High"].to_numpy(float)
    low = dataset["Low"].to_numpy(float)
    atr = dataset["ATR"].to_numpy(float)
    spread = dataset["SPREAD"].to_numpy(float) * SPREAD_COST_MULTIPLIER
    dates = pd.DatetimeIndex(dataset["fecha"])
    signals = np.flatnonzero(np.asarray(signal_mask, dtype=bool))
    n = len(close)
    trades = []
    next_free = 0
    for i in signals:
        if i < next_free or i + 1 >= n or not atr[i] > 0:
            continue
        risk = stop_atr * atr[i]
        last = min(i + max_hold_bars, n - 1)
        exit_idx, outcome = last, "TIME"
        if direction == "LONG":
            entry = close[i] + spread[i]
            sl, tp = entry - risk, entry + take_profit_atr * atr[i]
            exit_price = close[last]
            for j in range(i + 1, last + 1):
                if low[j] <= sl:
                    exit_idx, outcome, exit_price = j, "SL", sl
                    break
                if high[j] >= tp:
                    exit_idx, outcome, exit_price = j, "TP", tp
                    break
            r = (exit_price - entry) / risk
        else:
            entry = close[i]
            sl, tp = entry + risk, entry - take_profit_atr * atr[i]
            exit_price = close[last] + spread[last]
            for j in range(i + 1, last + 1):
                if high[j] + spread[j] >= sl:
                    exit_idx, outcome, exit_price = j, "SL", sl
                    break
                if low[j] + spread[j] <= tp:
                    exit_idx, outcome, exit_price = j, "TP", tp
                    break
            r = (entry - exit_price) / risk
        trades.append({"entry_time": dates[i], "exit_time": dates[exit_idx], "r": float(r), "outcome": outcome})
        next_free = exit_idx + 1
    return pd.DataFrame(trades, columns=["entry_time", "exit_time", "r", "outcome"])


def trade_metrics(trades):
    if trades.empty:
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "profit_factor": None,
                "total_r": 0.0, "max_dd_r": 0.0, "positive_years": None, "timeouts": 0}
    r = trades["r"]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    equity = r.cumsum()
    yearly = trades.groupby(trades["entry_time"].dt.year)["r"].agg(["sum", "count"])
    yearly = yearly[yearly["count"] >= YEAR_MIN_TRADES]
    return {
        "trades": int(len(r)),
        "win_rate": round(float((r > 0).mean()), 4),
        "expectancy_r": round(float(r.mean()), 4),
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
        "total_r": round(float(r.sum()), 2),
        "max_dd_r": round(float((equity.cummax() - equity).max()), 2),
        "positive_years": round(float((yearly["sum"] > 0).mean()), 3) if len(yearly) else None,
        "timeouts": int((trades["outcome"] == "TIME").sum()),
    }


def walk_forward_probabilities(dataset, target):
    """Probabilidad fuera de muestra por vela: cada año con un modelo entrenado en años previos."""
    probs = pd.Series(np.nan, index=dataset.index)
    years = dataset["fecha"].dt.year
    first_year = int(years.min()) + MIN_TRAIN_YEARS
    for year in range(first_year, int(years.max()) + 1):
        test_idx = dataset.index[years == year]
        if len(test_idx) == 0:
            continue
        train = dataset.loc[: test_idx[0] - EMBARGO_BARS - 1]
        train = train.dropna(subset=[target])
        if train[target].nunique() < 2:
            continue
        model = _crear_modelo("LOGISTIC").fit(train[FEATURES_ML], train[target].astype(int))
        classes = list(model.classes_)
        probs.loc[test_idx] = model.predict_proba(dataset.loc[test_idx, FEATURES_ML])[:, classes.index(1)]
    return probs


@dataclass(frozen=True)
class Variant:
    threshold: float
    trend_filter: bool
    stop_atr: float = 1.0
    strategy: str = "ML"
    tp_ratio: float = 2.0

    @property
    def name(self):
        core = f"T{int(round(self.threshold * 100)):03d}" if self.strategy == "ML" else f"TP{self.tp_ratio:g}"
        return f"{self.strategy}_SL{self.stop_atr:g}_{core}{'_TREND' if self.trend_filter else ''}"


def variant_mask(dataset, probs, direction, variant):
    if variant.strategy == "ML":
        mask = probs.to_numpy() >= variant.threshold
    else:
        mask = rule_signals(dataset, variant.strategy, direction).copy()
    if variant.trend_filter:
        mask &= trend_allows(direction, dataset["TREND"].to_numpy())
    return mask


def passes_selection(m):
    return (
        m["trades"] >= SELECTION_MIN_TRADES
        and m["expectancy_r"] is not None and m["expectancy_r"] >= SELECTION_MIN_EXPECTANCY
        and m["profit_factor"] is not None and m["profit_factor"] >= SELECTION_MIN_PROFIT_FACTOR
        and m["positive_years"] is not None and m["positive_years"] >= SELECTION_MIN_POSITIVE_YEARS
    )


def passes_confirmation(m):
    return (
        m["trades"] >= CONFIRMATION_MIN_TRADES
        and m["expectancy_r"] is not None and m["expectancy_r"] > CONFIRMATION_MIN_EXPECTANCY
    )


def research_direction(dataset, probs_by_stop, direction, include_rules=True):
    """Evalua todas las variantes; elige con SELECTION y confirma con CONFIRMATION.

    probs_by_stop: {stop_atr: probabilidades walk-forward del modelo entrenado con ese target}.
    """
    selection = (dataset["fecha"] < CONFIRMATION_START).to_numpy()
    # Mismo periodo fuera de muestra para ML y reglas (el ML no tiene prediccion antes).
    oos = (dataset["fecha"].dt.year >= int(dataset["fecha"].dt.year.min()) + MIN_TRAIN_YEARS).to_numpy()
    rows = []

    def evaluate(variant, probs):
        mask = variant_mask(dataset, probs, direction, variant) & oos
        params = exit_params(variant.stop_atr, variant.tp_ratio)
        sel_trades = simulate_trades(dataset, mask & selection, direction, *params)
        conf_trades = simulate_trades(dataset, mask & ~selection, direction, *params)
        sel, conf = trade_metrics(sel_trades), trade_metrics(conf_trades)
        rows.append({"variant": variant, "selection": sel, "confirmation": conf,
                     "confirmation_trades": conf_trades, "passes_selection": passes_selection(sel)})

    for stop_atr, probs in probs_by_stop.items():
        for threshold in THRESHOLDS:
            for trend_filter in TREND_FILTERS:
                evaluate(Variant(threshold, trend_filter, stop_atr), probs)
    if include_rules:
        for strategy in RULE_STRATEGIES:
            for stop_atr in RULE_STOPS:
                for tp_ratio in RULE_TP_RATIOS:
                    for trend_filter in TREND_FILTERS:
                        evaluate(Variant(0.0, trend_filter, stop_atr, strategy, tp_ratio), None)
    candidates = [r for r in rows if r["passes_selection"]]
    best = max(candidates, key=lambda r: r["selection"]["expectancy_r"]) if candidates else None
    baseline = next((r for r in rows if r["variant"] == Variant(0.60, False, 1.0)), None)
    for row in rows:  # memoria: solo se conservan las operaciones de la mejor y la base
        if row is not best and row is not baseline:
            row.pop("confirmation_trades", None)
    enabled =best is not None and passes_confirmation(best["confirmation"])
    if best is None:
        reason = "FALLA_SELECCION_2016_2023"
    elif not enabled:
        reason = "FALLA_CONFIRMACION_2024_2026"
    else:
        reason = "OK"
    return {"enabled": enabled, "reason": reason, "best": best, "baseline": baseline, "variants": rows}


def research_asset(root, asset):
    dataset = build_dataset(load_mt5_prices(root, asset)).reset_index(drop=True)
    out = {}
    for direction in ("LONG", "SHORT"):
        probs_by_stop = {}
        for stop_atr in STOP_MULTIPLIERS:
            dataset["TARGET"] = target_for_stop(dataset, direction, stop_atr)
            probs_by_stop[stop_atr] = walk_forward_probabilities(dataset, "TARGET")
        out[direction] = research_direction(dataset, probs_by_stop, direction)
    out["first_oos_year"] = int(dataset["fecha"].dt.year.min()) + MIN_TRAIN_YEARS
    out["bars"] = int(len(dataset))
    out["data_start"] = dataset["fecha"].iloc[0].isoformat()
    return out


def currency_legs(asset):
    asset = asset.upper()
    special = {"GOLD": ("XAU", "USD"), "SILVER": ("XAG", "USD")}
    return special.get(asset, (asset[:3], asset[3:6]))


def exposure_after(positions, asset, direction):
    """Exposicion neta por moneda (+1 comprada / -1 vendida) contando una nueva posicion."""
    exposure = {}
    for pos_asset, pos_direction in list(positions) + [(asset, direction)]:
        base, quote = currency_legs(pos_asset)
        sign = 1 if pos_direction in ("LONG", "BUY") else -1
        exposure[base] = exposure.get(base, 0) + sign
        exposure[quote] = exposure.get(quote, 0) - sign
    return exposure


def currency_exposure_allowed(positions, asset, direction, limit):
    """False si la nueva posicion deja alguna moneda con mas de `limit` posiciones netas
    en el mismo sentido (ej. 3 compras de JPY-cruces = 3 ventas de JPY)."""
    exposure = exposure_after(positions, asset, direction)
    base, quote = currency_legs(asset)
    return abs(exposure[base]) <= limit and abs(exposure[quote]) <= limit


def simulate_portfolio(trades_by_key, risk_per_trade=0.01, max_positions=5, currency_limit=2,
                       max_daily_loss=0.03):
    """Curva de capital con interes compuesto: riesgo = % del capital actual, limite de
    posiciones, de exposicion por moneda y tope de perdida diaria (no abre mas ese dia).
    trades_by_key: {(asset, direction): DataFrame de simulate_trades}."""
    events = []
    for (asset, direction), trades in trades_by_key.items():
        for t in trades.itertuples(index=False):
            events.append((t.entry_time, t.exit_time, asset, direction, t.r))
    events.sort(key=lambda e: e[0])
    equity, peak, max_dd = 1.0, 1.0, 0.0
    open_positions = []  # (exit_time, asset, direction, r, risk_amount)
    taken = skipped = daily_stops = 0
    curve = []
    day, day_start_equity, day_realized = None, equity, 0.0
    for entry_time, exit_time, asset, direction, r in events:
        still_open = []
        for pos in sorted(open_positions):
            if pos[0] <= entry_time:
                pnl = pos[3] * pos[4]
                equity += pnl
                if day is not None and pd.Timestamp(pos[0]).normalize() == day:
                    day_realized += pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
                curve.append((pos[0], equity))
            else:
                still_open.append(pos)
        open_positions = still_open
        today = pd.Timestamp(entry_time).normalize()
        if today != day:
            day, day_start_equity, day_realized = today, equity, 0.0
        if day_realized <= -max_daily_loss * day_start_equity:
            skipped += 1
            daily_stops += 1
            continue
        symbols = {p[1] for p in open_positions}
        legs = [(p[1], p[2]) for p in open_positions]
        if (len(open_positions) >= max_positions or asset in symbols
                or not currency_exposure_allowed(legs, asset, direction, currency_limit)):
            skipped += 1
            continue
        open_positions.append((exit_time, asset, direction, r, equity * risk_per_trade))
        taken += 1
    for pos in sorted(open_positions):
        equity += pos[3] * pos[4]
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        curve.append((pos[0], equity))
    return {"final_equity": equity, "max_drawdown": max_dd, "trades": taken, "skipped": skipped,
            "daily_loss_blocks": daily_stops, "curve": pd.DataFrame(curve, columns=["fecha", "equity"])}


def _variant_json(row):
    if row is None:
        return None
    return {"variant": row["variant"].name, "strategy": row["variant"].strategy,
            "threshold": row["variant"].threshold,
            "trend_filter": row["variant"].trend_filter, "stop_atr": row["variant"].stop_atr,
            "tp_ratio": row["variant"].tp_ratio,
            "selection": row["selection"],
            "confirmation": row["confirmation"]}


def save_selection(root, results):
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": ("walk-forward anual MT5, seleccion <2024, confirmacion >=2024, costos spread x1.5, "
                   f"ML stops {list(STOP_MULTIPLIERS)} ATR + reglas {list(RULE_STRATEGIES)}"),
        "assets": {},
    }
    for asset, res in results.items():
        payload["assets"][asset] = {}
        for direction in ("LONG", "SHORT"):
            d = res[direction]
            best = d["best"]
            payload["assets"][asset][direction] = {
                "enabled": bool(d["enabled"]),
                "reason": d["reason"],
                "strategy": best["variant"].strategy if best else None,
                "threshold": best["variant"].threshold if best else None,
                "trend_filter": best["variant"].trend_filter if best else None,
                **dict(zip(("stop_atr", "take_profit_atr", "max_hold_bars"),
                           exit_params(best["variant"].stop_atr, best["variant"].tp_ratio)
                           if best else exit_params(STOP_ATR))),
                "best": _variant_json(best),
                "baseline_T060": _variant_json(d["baseline"]),
            }
    path = selection_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_selection(root):
    path = selection_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["assets"]


def run_full_research(root, assets, log=print):
    """Walk-forward de todos los activos, guarda la seleccion y compara portafolios en el
    periodo de confirmacion: todas las direcciones T060 (como antes) vs solo las habilitadas."""
    results = {}
    for asset in assets:
        if not mt5_data_path(root, asset).exists():
            log(f"{asset}: sin datos MT5, omitido")
            continue
        log(f"Analizando {asset}...")
        results[asset] = research_asset(root, asset)
    path = save_selection(root, results)

    rows = []
    baseline_trades, selected_trades = {}, {}
    for asset, res in results.items():
        for direction in ("LONG", "SHORT"):
            d = res[direction]
            base = d["baseline"]
            best = d["best"]
            baseline_trades[(asset, direction)] = base["confirmation_trades"]
            if d["enabled"]:
                selected_trades[(asset, direction)] = best["confirmation_trades"]
            rows.append({
                "asset": asset, "direction": direction, "enabled": d["enabled"], "reason": d["reason"],
                "variant": best["variant"].name if best else "-",
                "sel_trades": best["selection"]["trades"] if best else base["selection"]["trades"],
                "sel_exp_r": best["selection"]["expectancy_r"] if best else base["selection"]["expectancy_r"],
                "sel_pf": best["selection"]["profit_factor"] if best else base["selection"]["profit_factor"],
                "sel_pos_years": best["selection"]["positive_years"] if best else base["selection"]["positive_years"],
                "conf_trades": best["confirmation"]["trades"] if best else None,
                "conf_exp_r": best["confirmation"]["expectancy_r"] if best else None,
                "conf_win": best["confirmation"]["win_rate"] if best else None,
                "base_T060_conf_trades": base["confirmation"]["trades"],
                "base_T060_conf_exp_r": base["confirmation"]["expectancy_r"],
            })
    table = pd.DataFrame(rows)
    portfolios = {
        "ANTES_todas_T060": simulate_portfolio(baseline_trades),
        "AHORA_seleccion_walkforward": simulate_portfolio(selected_trades),
    }
    report_dir = Path(root) / "reports"
    report_dir.mkdir(exist_ok=True)
    table.to_csv(report_dir / "portfolio_research.csv", index=False)
    for name, sim in portfolios.items():
        sim["curve"].to_csv(report_dir / f"portfolio_curve_{name}.csv", index=False)
    return {"table": table, "portfolios": portfolios, "selection_path": path, "results": results}


def daily_bars(prices):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Spread" in prices:
        agg["Spread"] = "median"
    daily = prices.resample("1D").agg(agg).dropna(subset=["Close"])
    if "Spread" not in daily:
        daily["Spread"] = 0.0
    return daily


def donchian_trades(prices, entry_days=20, exit_days=10, stop_n=2.0):
    """Referencia trend following (Tortugas) en D1 con los mismos datos y costos.

    Breakout del maximo/minimo de `entry_days`, salida en el extremo contrario de
    `exit_days` o en el stop de `stop_n` ATR(20). 1R = distancia al stop.
    """
    d = daily_bars(prices)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - d["Close"].shift()).abs(),
                    (d["Low"] - d["Close"].shift()).abs()], axis=1).max(axis=1)
    n_atr = tr.ewm(alpha=1 / 20, adjust=False, min_periods=20).mean().shift(1)
    hi_entry = d["High"].rolling(entry_days).max().shift(1)
    lo_entry = d["Low"].rolling(entry_days).min().shift(1)
    hi_exit = d["High"].rolling(exit_days).max().shift(1)
    lo_exit = d["Low"].rolling(exit_days).min().shift(1)
    o, h, l = d["Open"].to_numpy(), d["High"].to_numpy(), d["Low"].to_numpy()
    spread = d["Spread"].to_numpy() * SPREAD_COST_MULTIPLIER
    dates = d.index
    trades, pos = [], None
    for i in range(len(d)):
        if pos is not None:
            side, entry, stop, risk, t0 = pos
            exit_price = None
            if side == 1:
                if l[i] <= stop:
                    exit_price = min(o[i], stop)
                elif l[i] <= lo_exit.iloc[i]:
                    exit_price = min(o[i], lo_exit.iloc[i])
                r = None if exit_price is None else (exit_price - entry) / risk
            else:
                if h[i] + spread[i] >= stop:
                    exit_price = max(o[i] + spread[i], stop)
                elif h[i] >= hi_exit.iloc[i]:
                    exit_price = max(o[i], hi_exit.iloc[i]) + spread[i]
                r = None if exit_price is None else (entry - exit_price) / risk
            if exit_price is not None:
                trades.append({"entry_time": t0, "exit_time": dates[i], "r": float(r), "outcome": "EXIT"})
                pos = None
            continue
        if not n_atr.iloc[i] > 0 or pd.isna(hi_entry.iloc[i]):
            continue
        up, down = h[i] >= hi_entry.iloc[i], l[i] <= lo_entry.iloc[i]
        if up == down:
            continue
        risk = stop_n * n_atr.iloc[i]
        if up:
            entry = max(o[i], hi_entry.iloc[i]) + spread[i]
            pos = (1, entry, entry - risk, risk, dates[i])
        else:
            entry = min(o[i], lo_entry.iloc[i])
            pos = (-1, entry, entry + risk, risk, dates[i])
    return pd.DataFrame(trades, columns=["entry_time", "exit_time", "r", "outcome"])


def benchmark_asset(root, asset, first_year):
    prices = load_mt5_prices(root, asset)
    out = {}
    for name, (entry_days, exit_days) in {"TURTLE_20_10": (20, 10), "TURTLE_55_20": (55, 20)}.items():
        trades = donchian_trades(prices, entry_days, exit_days)
        trades = trades[trades["entry_time"].dt.year >= first_year]
        sel = trades[trades["entry_time"] < CONFIRMATION_START]
        conf = trades[trades["entry_time"] >= CONFIRMATION_START]
        out[name] = {"selection": trade_metrics(sel), "confirmation": trade_metrics(conf),
                     "confirmation_trades": conf, "all_trades": trades}
    return out
