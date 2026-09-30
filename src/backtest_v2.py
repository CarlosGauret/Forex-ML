import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import COST_BPS
from config_paper import PAPER_CONFIGS
from src.market_structure import (
    LAST_SWING_HIGH,
    LAST_SWING_HIGH_INDEX,
    LAST_SWING_LOW,
    LAST_SWING_LOW_INDEX,
    detectar_swings_confirmados,
)
from src.risk import BUY, RiskLimits, RiskManager
from src.sl_tp import (
    ATR_ONLY,
    LONG,
    SWING_ATR,
    SWING_ATR_RR,
    SWING_STRUCTURAL_TARGET,
    calcular_sl_tp,
)


SL_TP_MODES = [ATR_ONLY, SWING_ATR, SWING_ATR_RR, SWING_STRUCTURAL_TARGET]
CAPITALS = [100, 500, 1000, 5000]
RISK_PER_TRADE = 0.01
MAX_HOLD_BARS = 24
RESULTS_COLUMNS = [
    "ACTIVO",
    "CONFIG_ID",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "SL_TP_MODE",
    "COST_BPS",
    "CAPITAL_INICIAL",
    "BROKER_AWARE",
    "BROKER_METADATA_SOURCE",
    "TOTAL_ENTRADAS_BASE",
    "TOTAL_TRADES",
    "WINS",
    "LOSSES",
    "WIN_RATE",
    "PROFIT_FACTOR",
    "EXPECTANCY",
    "AVERAGE_R",
    "R_ACUMULADO",
    "NET_PROFIT",
    "RETURN_PCT",
    "MAX_DRAWDOWN",
    "SHARPE",
    "MAX_LOSING_STREAK",
    "CAPITAL_FINAL",
    "NO_EJECUTABLES_BROKER_MIN",
    "SKIPPED_SLTP",
]
TRADES_COLUMNS = [
    "ACTIVO",
    "CONFIG_ID",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "SL_TP_MODE",
    "COST_BPS",
    "CAPITAL_INICIAL",
    "BROKER_AWARE",
    "ENTRY_ID",
    "FECHA_SIGNAL",
    "FECHA_ENTRADA",
    "FECHA_SALIDA",
    "ENTRY",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "EXIT",
    "ATR",
    "RISK_DISTANCE",
    "RR",
    "RESULTADO",
    "MOTIVO_SALIDA",
    "R_BRUTO",
    "R_NETO",
    "COST_R",
    "PROBABILIDAD",
    "CAPITAL_ANTES",
    "CAPITAL_DESPUES",
    "RISK_AMOUNT",
    "VOLUME_THEORETICAL",
    "VOLUME_ALLOWED",
    "ESTIMATED_LOSS",
    "RISK_PCT_REAL",
    "BROKER_REASON",
    "ANTI_LOOKAHEAD_OK",
    "SWING_LOW_USADO",
    "SWING_HIGH_USADO",
    "STRUCTURAL_TARGET",
]


@dataclass(frozen=True)
class BrokerMetadata:
    symbol: str
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    point: float
    digits: int
    source: str


DEFAULT_GOLD_BROKER_METADATA = BrokerMetadata(
    symbol="XAUUSD",
    contract_size=100.0,
    volume_min=0.01,
    volume_max=50.0,
    volume_step=0.01,
    point=0.01,
    digits=2,
    source="CONFIG_LOCAL_FALLBACK_MT5_UNAVAILABLE",
)


def obtener_gold_broker_metadata(default_metadata=DEFAULT_GOLD_BROKER_METADATA):
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return default_metadata

    try:
        from src.mt5_connector import _credenciales_mt5, _modo_cuenta
        from src.symbols import discover_symbols
    except Exception:
        return default_metadata

    credenciales, error = _credenciales_mt5()
    if error or credenciales is None:
        return default_metadata

    inicializado = False
    try:
        if not mt5.initialize():
            return default_metadata
        inicializado = True

        if not mt5.login(
            credenciales["login"],
            password=credenciales["password"],
            server=credenciales["server"],
        ):
            return default_metadata

        account_info = mt5.account_info()
        if _modo_cuenta(mt5, account_info) == "REAL":
            return default_metadata

        resolved = discover_symbols(mt5, ["GOLD"]).get("GOLD")
        if (
            resolved is None
            or not resolved.found
            or resolved.contract_size is None
            or resolved.volume_min is None
            or resolved.volume_max is None
            or resolved.volume_step is None
        ):
            return default_metadata

        return BrokerMetadata(
            symbol=resolved.mt5_symbol,
            contract_size=float(resolved.contract_size),
            volume_min=float(resolved.volume_min),
            volume_max=float(resolved.volume_max),
            volume_step=float(resolved.volume_step),
            point=float(resolved.point or default_metadata.point),
            digits=int(resolved.digits or default_metadata.digits),
            source="MT5_READONLY",
        )
    finally:
        if inicializado:
            mt5.shutdown()


def _to_naive_datetime(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def _normalizar_oof(oof_predictions):
    datos = oof_predictions.copy()
    datos["FECHA"] = _to_naive_datetime(datos["FECHA"])
    for columna in ["OPEN", "HIGH", "LOW", "CLOSE", "ATR", "PROBABILIDAD"]:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
    datos = datos.dropna(
        subset=["FECHA", "OPEN", "HIGH", "LOW", "CLOSE", "ATR", "PROBABILIDAD"]
    )
    datos = datos.sort_values("FECHA").reset_index(drop=True)
    return datos


def cargar_gold_oof(raiz_proyecto):
    ruta = Path(raiz_proyecto) / "results" / "gold" / "oof_predictions.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe OOF requerido para GOLD: {ruta}")
    return _normalizar_oof(pd.read_csv(ruta))


def _base_temporal(oof, config):
    datos = oof[
        (oof["ACTIVO"] == config["ACTIVO"])
        & (oof["DIRECCION"] == config["DIRECCION"])
        & (oof["MODELO"] == config["MODELO"])
    ].copy()
    datos = datos.sort_values("FECHA").drop_duplicates("FECHA", keep="first")
    datos = datos.reset_index(drop=True)
    precios = datos.rename(
        columns={
            "OPEN": "Open",
            "HIGH": "High",
            "LOW": "Low",
            "CLOSE": "Close",
        }
    ).copy()
    precios.index = precios["FECHA"]
    precios = detectar_swings_confirmados(precios, high_col="High", low_col="Low")
    precios = precios.reset_index(drop=True)
    for columna in [
        LAST_SWING_HIGH,
        LAST_SWING_LOW,
        LAST_SWING_HIGH_INDEX,
        LAST_SWING_LOW_INDEX,
    ]:
        datos[columna] = precios[columna].values
    return datos


def construir_entradas_base(datos, config, max_hold_bars=MAX_HOLD_BARS):
    entradas = []
    siguiente_signal_permitida = 0
    threshold = float(config["THRESHOLD"])

    for indice in range(len(datos) - 1):
        if indice < siguiente_signal_permitida:
            continue

        fila = datos.iloc[indice]
        if fila["BASE_SIGNAL"] != "BUY" or float(fila["PROBABILIDAD"]) < threshold:
            continue

        indice_entrada = indice + 1
        fila_entrada = datos.iloc[indice_entrada]
        entradas.append(
            {
                "ENTRY_ID": len(entradas) + 1,
                "SIGNAL_INDEX": indice,
                "ENTRY_INDEX": indice_entrada,
                "FECHA_SIGNAL": fila["FECHA"],
                "FECHA_ENTRADA": fila_entrada["FECHA"],
                "ENTRY": float(fila_entrada["OPEN"]),
                "ATR": float(fila["ATR"]),
                "PROBABILIDAD": float(fila["PROBABILIDAD"]),
            }
        )
        siguiente_signal_permitida = indice + max_hold_bars

    return entradas


def _ultimo_valor_confirmado(fila, precio_columna, indice_columna, entry, direction):
    valor = fila.get(precio_columna)
    indice = fila.get(indice_columna)
    if pd.isna(valor) or pd.isna(indice):
        return None, None

    valor = float(valor)
    if direction == LONG and precio_columna == LAST_SWING_LOW and valor >= entry:
        return None, None
    if direction == LONG and precio_columna == LAST_SWING_HIGH and valor <= entry:
        return None, None

    return valor, pd.to_datetime(indice)


def _anti_lookahead_ok(fila_signal, swing_index):
    if swing_index is None or pd.isna(swing_index):
        return False
    return pd.to_datetime(swing_index) <= pd.to_datetime(fila_signal["FECHA"])


def _calcular_sl_tp_para_entrada(datos, entrada, mode):
    fila_signal = datos.iloc[entrada["SIGNAL_INDEX"]]
    entry = entrada["ENTRY"]
    atr = entrada["ATR"]
    swing_low, swing_low_index = _ultimo_valor_confirmado(
        fila_signal,
        LAST_SWING_LOW,
        LAST_SWING_LOW_INDEX,
        entry,
        LONG,
    )
    swing_high, swing_high_index = _ultimo_valor_confirmado(
        fila_signal,
        LAST_SWING_HIGH,
        LAST_SWING_HIGH_INDEX,
        entry,
        LONG,
    )
    structural_target = swing_high

    resultado = calcular_sl_tp(
        entry=entry,
        direction=LONG,
        atr=atr,
        mode=mode,
        atr_multiplier=1.0,
        rr=2.0,
        take_profit_atr_multiplier=2.0,
        swing_low=swing_low,
        swing_high=swing_high,
        structural_target=structural_target,
    )
    if not resultado.ok:
        return resultado, swing_low, swing_high, structural_target, False

    take_profit = resultado.take_profit
    rr = resultado.rr
    reason = resultado.reason
    if mode == SWING_ATR and take_profit is None:
        take_profit = entry + (2.0 * atr)
        rr = abs(take_profit - entry) / resultado.risk_distance
        reason = "OK"
        resultado = resultado.__class__(
            ok=True,
            direction=resultado.direction,
            mode=resultado.mode,
            entry=resultado.entry,
            stop_loss=resultado.stop_loss,
            take_profit=take_profit,
            risk_distance=resultado.risk_distance,
            rr=rr,
            reason=reason,
        )

    if mode == ATR_ONLY:
        anti_lookahead_ok = True
    elif mode == SWING_STRUCTURAL_TARGET:
        anti_lookahead_ok = _anti_lookahead_ok(
            fila_signal, swing_low_index
        ) and _anti_lookahead_ok(fila_signal, swing_high_index)
    else:
        anti_lookahead_ok = _anti_lookahead_ok(fila_signal, swing_low_index)

    return resultado, swing_low, swing_high, structural_target, anti_lookahead_ok


def _aplicar_costos(entry, exit_price, r_bruto, risk_distance, cost_bps):
    costo_por_unidad = (abs(entry) + abs(exit_price)) * (float(cost_bps) / 10000)
    costo_r = costo_por_unidad / risk_distance
    return r_bruto - costo_r, costo_r


def _resolver_salida(datos, entrada, sltp, max_hold_bars):
    limite = min(entrada["ENTRY_INDEX"] + max_hold_bars - 1, len(datos) - 1)

    for indice in range(entrada["ENTRY_INDEX"], limite + 1):
        fila = datos.iloc[indice]
        low = float(fila["LOW"])
        high = float(fila["HIGH"])

        toca_stop = low <= sltp.stop_loss
        toca_take = high >= sltp.take_profit
        if toca_stop:
            return indice, float(sltp.stop_loss), "SL"
        if toca_take:
            return indice, float(sltp.take_profit), "TP"

    return limite, float(datos.iloc[limite]["CLOSE"]), "TIME"


def _max_drawdown(capitales):
    pico = capitales[0]
    max_dd = 0.0
    for capital in capitales:
        pico = max(pico, capital)
        if pico > 0:
            max_dd = max(max_dd, ((pico - capital) / pico) * 100)
    return max_dd


def _max_losing_streak(r_values):
    actual = 0
    maximo = 0
    for valor in r_values:
        if valor <= 0:
            actual += 1
            maximo = max(maximo, actual)
        else:
            actual = 0
    return maximo


def _sharpe_trade_level(retornos):
    if len(retornos) < 2:
        return None
    serie = pd.Series(retornos, dtype="float64")
    std = serie.std(ddof=1)
    if std == 0 or pd.isna(std):
        return None
    return float((serie.mean() / std) * math.sqrt(len(serie)))


def _metricas_base(filas, capitales, escenario, total_entradas):
    ejecutadas = [fila for fila in filas if fila["RESULTADO"] in ("WIN", "LOSS")]
    r_values = [float(fila["R_NETO"]) for fila in ejecutadas]
    ganancias = [
        float(fila["CAPITAL_DESPUES"]) - float(fila["CAPITAL_ANTES"])
        for fila in ejecutadas
        if float(fila["CAPITAL_DESPUES"]) > float(fila["CAPITAL_ANTES"])
    ]
    perdidas = [
        float(fila["CAPITAL_DESPUES"]) - float(fila["CAPITAL_ANTES"])
        for fila in ejecutadas
        if float(fila["CAPITAL_DESPUES"]) <= float(fila["CAPITAL_ANTES"])
    ]
    total = len(ejecutadas)
    wins = len(ganancias)
    losses = len(perdidas)
    gross_profit = sum(ganancias)
    gross_loss = abs(sum(perdidas))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    capital_inicial = float(escenario["CAPITAL_INICIAL"])
    capital_final = capitales[-1]
    net_profit = capital_final - capital_inicial
    retornos = [
        (float(fila["CAPITAL_DESPUES"]) - float(fila["CAPITAL_ANTES"]))
        / float(fila["CAPITAL_ANTES"])
        for fila in ejecutadas
        if float(fila["CAPITAL_ANTES"]) > 0
    ]

    return {
        **escenario,
        "TOTAL_ENTRADAS_BASE": total_entradas,
        "TOTAL_TRADES": total,
        "WINS": wins,
        "LOSSES": losses,
        "WIN_RATE": (wins / total) * 100 if total else 0.0,
        "PROFIT_FACTOR": profit_factor,
        "EXPECTANCY": sum(r_values) / total if total else 0.0,
        "AVERAGE_R": sum(r_values) / total if total else 0.0,
        "R_ACUMULADO": sum(r_values),
        "NET_PROFIT": net_profit,
        "RETURN_PCT": (net_profit / capital_inicial) * 100 if capital_inicial else 0.0,
        "MAX_DRAWDOWN": _max_drawdown(capitales),
        "SHARPE": _sharpe_trade_level(retornos),
        "MAX_LOSING_STREAK": _max_losing_streak(r_values),
        "CAPITAL_FINAL": capital_final,
        "NO_EJECUTABLES_BROKER_MIN": sum(
            1
            for fila in filas
            if fila["RESULTADO"] == "NO EJECUTABLE POR MINIMO DE BROKER"
        ),
        "SKIPPED_SLTP": sum(1 for fila in filas if fila["RESULTADO"] == "SKIP_SLTP"),
    }


def _metadata_to_symbol_info(metadata):
    return {
        "trade_contract_size": metadata.contract_size,
        "contract_size": metadata.contract_size,
        "volume_min": metadata.volume_min,
        "volume_max": metadata.volume_max,
        "volume_step": metadata.volume_step,
        "point": metadata.point,
        "digits": metadata.digits,
    }


def _fila_skip(escenario, entrada, resultado, reason, capital, sltp=None):
    return {
        **escenario,
        "ENTRY_ID": entrada["ENTRY_ID"],
        "FECHA_SIGNAL": entrada["FECHA_SIGNAL"],
        "FECHA_ENTRADA": entrada["FECHA_ENTRADA"],
        "FECHA_SALIDA": pd.NaT,
        "ENTRY": entrada["ENTRY"],
        "STOP_LOSS": getattr(sltp, "stop_loss", None),
        "TAKE_PROFIT": getattr(sltp, "take_profit", None),
        "EXIT": None,
        "ATR": entrada["ATR"],
        "RISK_DISTANCE": getattr(sltp, "risk_distance", None),
        "RR": getattr(sltp, "rr", None),
        "RESULTADO": resultado,
        "MOTIVO_SALIDA": reason,
        "R_BRUTO": None,
        "R_NETO": None,
        "COST_R": None,
        "PROBABILIDAD": entrada["PROBABILIDAD"],
        "CAPITAL_ANTES": capital,
        "CAPITAL_DESPUES": capital,
        "RISK_AMOUNT": capital * RISK_PER_TRADE,
        "VOLUME_THEORETICAL": None,
        "VOLUME_ALLOWED": None,
        "ESTIMATED_LOSS": None,
        "RISK_PCT_REAL": None,
        "BROKER_REASON": reason,
        "ANTI_LOOKAHEAD_OK": False,
        "SWING_LOW_USADO": None,
        "SWING_HIGH_USADO": None,
        "STRUCTURAL_TARGET": None,
    }


def _ejecutar_escenario(datos, entradas, config, mode, cost_bps, capital_inicial, broker_metadata, broker_aware):
    capital = float(capital_inicial)
    capitales = [capital]
    filas = []
    risk_manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))
    escenario = {
        "ACTIVO": config["ACTIVO"],
        "CONFIG_ID": config["CONFIG_ID"],
        "DIRECCION": config["DIRECCION"],
        "MODELO": config["MODELO"],
        "SISTEMA": config["SISTEMA"],
        "THRESHOLD": float(config["THRESHOLD"]),
        "SL_TP_MODE": mode,
        "COST_BPS": cost_bps,
        "CAPITAL_INICIAL": capital_inicial,
        "BROKER_AWARE": bool(broker_aware),
        "BROKER_METADATA_SOURCE": broker_metadata.source,
    }

    for entrada in entradas:
        sltp, swing_low, swing_high, structural_target, anti_ok = _calcular_sl_tp_para_entrada(
            datos, entrada, mode
        )
        if not sltp.ok or not anti_ok:
            filas.append(
                _fila_skip(
                    escenario,
                    entrada,
                    "SKIP_SLTP",
                    sltp.reason if sltp.ok else sltp.reason,
                    capital,
                    sltp,
                )
            )
            continue

        sizing = None
        risk_amount = capital * RISK_PER_TRADE
        volume_theoretical = None
        volume_allowed = None
        estimated_loss = risk_amount
        risk_pct_real = RISK_PER_TRADE * 100
        broker_reason = "NO_BROKER_AWARE"
        if broker_aware:
            sizing = risk_manager.calculate_position_size(
                symbol=broker_metadata.symbol,
                direction=BUY,
                entry=sltp.entry,
                stop_loss=sltp.stop_loss,
                balance=capital,
                symbol_info=_metadata_to_symbol_info(broker_metadata),
            )
            volume_theoretical = sizing.theoretical_volume
            volume_allowed = sizing.volume
            estimated_loss = sizing.estimated_loss
            risk_pct_real = sizing.risk_pct_real
            broker_reason = sizing.reason
            if not sizing.ok:
                filas.append(
                    {
                        **_fila_skip(
                            escenario,
                            entrada,
                            "NO EJECUTABLE POR MINIMO DE BROKER"
                            if sizing.reason == "MINIMUM_VOLUME_EXCEEDS_RISK"
                            else "SKIP_BROKER",
                            sizing.reason,
                            capital,
                            sltp,
                        ),
                        "VOLUME_THEORETICAL": sizing.theoretical_volume,
                        "ESTIMATED_LOSS": sizing.min_volume_loss,
                        "RISK_PCT_REAL": sizing.risk_pct_minimum,
                        "BROKER_REASON": sizing.reason,
                        "ANTI_LOOKAHEAD_OK": anti_ok,
                        "SWING_LOW_USADO": swing_low,
                        "SWING_HIGH_USADO": swing_high,
                        "STRUCTURAL_TARGET": structural_target,
                    }
                )
                continue

        salida_idx, exit_price, motivo = _resolver_salida(datos, entrada, sltp, MAX_HOLD_BARS)
        r_bruto = (exit_price - sltp.entry) / sltp.risk_distance
        r_neto, cost_r = _aplicar_costos(
            sltp.entry, exit_price, r_bruto, sltp.risk_distance, cost_bps
        )
        capital_antes = capital
        base_riesgo = estimated_loss if broker_aware else risk_amount
        capital += base_riesgo * r_neto
        capitales.append(capital)
        resultado = "WIN" if r_neto > 0 else "LOSS"

        filas.append(
            {
                **escenario,
                "ENTRY_ID": entrada["ENTRY_ID"],
                "FECHA_SIGNAL": entrada["FECHA_SIGNAL"],
                "FECHA_ENTRADA": entrada["FECHA_ENTRADA"],
                "FECHA_SALIDA": datos.iloc[salida_idx]["FECHA"],
                "ENTRY": sltp.entry,
                "STOP_LOSS": sltp.stop_loss,
                "TAKE_PROFIT": sltp.take_profit,
                "EXIT": exit_price,
                "ATR": entrada["ATR"],
                "RISK_DISTANCE": sltp.risk_distance,
                "RR": sltp.rr,
                "RESULTADO": resultado,
                "MOTIVO_SALIDA": motivo,
                "R_BRUTO": r_bruto,
                "R_NETO": r_neto,
                "COST_R": cost_r,
                "PROBABILIDAD": entrada["PROBABILIDAD"],
                "CAPITAL_ANTES": capital_antes,
                "CAPITAL_DESPUES": capital,
                "RISK_AMOUNT": risk_amount,
                "VOLUME_THEORETICAL": volume_theoretical,
                "VOLUME_ALLOWED": volume_allowed,
                "ESTIMATED_LOSS": estimated_loss,
                "RISK_PCT_REAL": risk_pct_real,
                "BROKER_REASON": broker_reason,
                "ANTI_LOOKAHEAD_OK": anti_ok,
                "SWING_LOW_USADO": swing_low,
                "SWING_HIGH_USADO": swing_high,
                "STRUCTURAL_TARGET": structural_target,
            }
        )

    return filas, _metricas_base(filas, capitales, escenario, len(entradas))


def ejecutar_backtest_v2_gold(
    raiz_proyecto,
    oof_predictions=None,
    broker_metadata=None,
    costs=None,
    capitals=None,
):
    raiz = Path(raiz_proyecto)
    oof = cargar_gold_oof(raiz) if oof_predictions is None else _normalizar_oof(oof_predictions)
    broker_metadata = broker_metadata or obtener_gold_broker_metadata()
    costs = COST_BPS if costs is None else costs
    capitals = CAPITALS if capitals is None else capitals

    configs = [
        config
        for config in PAPER_CONFIGS
        if config["ACTIVO"] == "GOLD"
        and config["DIRECCION"] == "LONG"
        and config["MODELO"] == "LOGISTIC"
        and config["SISTEMA"] == "BASE_PLUS_ML"
    ]
    resumen = []
    trades = []
    entradas_rows = []

    for config in configs:
        datos = _base_temporal(oof, config)
        entradas = construir_entradas_base(datos, config, config.get("MAX_HOLD_BARS", MAX_HOLD_BARS))
        for entrada in entradas:
            entradas_rows.append(
                {
                    "CONFIG_ID": config["CONFIG_ID"],
                    "THRESHOLD": config["THRESHOLD"],
                    **entrada,
                }
            )

        for mode in SL_TP_MODES:
            for cost_bps in costs:
                for capital in capitals:
                    for broker_aware in (False, True):
                        filas, metricas = _ejecutar_escenario(
                            datos=datos,
                            entradas=entradas,
                            config=config,
                            mode=mode,
                            cost_bps=cost_bps,
                            capital_inicial=capital,
                            broker_metadata=broker_metadata,
                            broker_aware=broker_aware,
                        )
                        trades.extend(filas)
                        resumen.append(metricas)

    carpeta = raiz / "results" / "v2"
    carpeta.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades, columns=TRADES_COLUMNS)
    resumen_df = pd.DataFrame(resumen, columns=RESULTS_COLUMNS)
    entradas_df = pd.DataFrame(entradas_rows)
    ruta_trades = carpeta / "gold_sl_tp_trades.csv"
    ruta_summary = carpeta / "gold_sl_tp_comparison.csv"
    ruta_entries = carpeta / "gold_sl_tp_entries.csv"
    trades_df.to_csv(ruta_trades, index=False)
    resumen_df.to_csv(ruta_summary, index=False)
    entradas_df.to_csv(ruta_entries, index=False)

    return {
        "summary": resumen_df,
        "trades": trades_df,
        "entries": entradas_df,
        "ruta_summary": ruta_summary,
        "ruta_trades": ruta_trades,
        "ruta_entries": ruta_entries,
        "broker_metadata": broker_metadata,
    }
