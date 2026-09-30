import pandas as pd

from config import COST_BPS, PROB_THRESHOLD, PROB_THRESHOLDS


CAPITAL_INICIAL = 1000
RIESGO_POR_TRADE = 0.01
MAX_DURACION = 24
SISTEMAS = ["BASE", "ML_ONLY", "BASE_PLUS_ML"]
COLUMNAS_REQUERIDAS = [
    "FECHA",
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "FOLD",
    "CLOSE",
    "OPEN",
    "HIGH",
    "LOW",
    "ATR",
    "PROBABILIDAD",
    "TARGET_REAL",
    "BASE_SIGNAL",
]


def _validar_oof(oof):
    faltantes = [columna for columna in COLUMNAS_REQUERIDAS if columna not in oof.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas OOF requeridas: {', '.join(faltantes)}")


def _preparar_oof(oof):
    _validar_oof(oof)

    datos = oof.copy()
    datos["FECHA"] = pd.to_datetime(datos["FECHA"])

    for columna in ["CLOSE", "OPEN", "HIGH", "LOW", "ATR", "PROBABILIDAD"]:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos = datos.dropna(
        subset=["FECHA", "CLOSE", "OPEN", "HIGH", "LOW", "ATR", "PROBABILIDAD"]
    )
    datos = datos.sort_values(["DIRECCION", "MODELO", "FECHA"])

    return datos


def cargar_oof_si_valido(ruta_oof, activo=None):
    if not ruta_oof.exists():
        return None

    try:
        datos = _preparar_oof(pd.read_csv(ruta_oof))
    except Exception:
        return None

    if datos.empty:
        return None

    if activo is not None and not (datos["ACTIVO"] == activo).all():
        return None

    return datos


def _hay_senal(fila, sistema, direccion, threshold):
    prob_ok = fila["PROBABILIDAD"] >= threshold
    base_requerida = "BUY" if direccion == "LONG" else "SELL"
    base_ok = fila["BASE_SIGNAL"] == base_requerida

    if sistema == "BASE":
        return base_ok

    if sistema == "ML_ONLY":
        return prob_ok

    if sistema == "BASE_PLUS_ML":
        return base_ok and prob_ok

    raise ValueError(f"Sistema no soportado: {sistema}")


def _aplicar_costos(entry, exit_price, r_bruto, riesgo_precio, cost_bps):
    costo_por_unidad = (abs(entry) + abs(exit_price)) * (cost_bps / 10000)
    costo_r = costo_por_unidad / riesgo_precio
    return r_bruto - costo_r


def _resolver_trade(datos, indice_signal, direccion, cost_bps):
    indice_entrada = indice_signal + 1

    if indice_entrada >= len(datos):
        return None

    fila_signal = datos.iloc[indice_signal]
    fila_entrada = datos.iloc[indice_entrada]
    entry = float(fila_entrada["OPEN"])
    atr = float(fila_signal["ATR"])

    if atr <= 0:
        return None

    if direccion == "LONG":
        stop_loss = entry - atr
        take_profit = entry + (2 * atr)
    else:
        stop_loss = entry + atr
        take_profit = entry - (2 * atr)

    salida_idx = None
    exit_price = None
    motivo = None
    limite = min(indice_entrada + MAX_DURACION - 1, len(datos) - 1)

    for j in range(indice_entrada, limite + 1):
        fila = datos.iloc[j]
        high = float(fila["HIGH"])
        low = float(fila["LOW"])

        if direccion == "LONG":
            toca_stop = low <= stop_loss
            toca_take = high >= take_profit
        else:
            toca_stop = high >= stop_loss
            toca_take = low <= take_profit

        if toca_stop:
            salida_idx = j
            exit_price = stop_loss
            motivo = "SL"
            break

        if toca_take:
            salida_idx = j
            exit_price = take_profit
            motivo = "TP"
            break

    if salida_idx is None:
        salida_idx = limite
        exit_price = float(datos.iloc[salida_idx]["CLOSE"])
        motivo = "TIME"

    if direccion == "LONG":
        r_bruto = (exit_price - entry) / atr
    else:
        r_bruto = (entry - exit_price) / atr

    r_neto = _aplicar_costos(entry, exit_price, r_bruto, atr, cost_bps)

    return {
        "indice_entrada": indice_entrada,
        "salida_idx": salida_idx,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "exit_price": exit_price,
        "motivo": motivo,
        "r": r_neto,
    }


def _calcular_max_drawdown(capitales):
    pico = capitales[0]
    max_drawdown = 0

    for capital in capitales:
        pico = max(pico, capital)
        if pico == 0:
            continue

        drawdown = (pico - capital) / pico
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown * 100


def _calcular_metricas(trades, capitales, activo, direccion, modelo, sistema, threshold, cost_bps):
    total = len(trades)
    capital_inicial = capitales[0]
    capital_final = capitales[-1]
    ganancias = [trade["GANANCIA_PERDIDA"] for trade in trades if trade["GANANCIA_PERDIDA"] > 0]
    perdidas = [trade["GANANCIA_PERDIDA"] for trade in trades if trade["GANANCIA_PERDIDA"] < 0]
    r_values = [trade["R"] for trade in trades]
    ganadoras = len(ganancias)
    perdedoras = len(perdidas)
    gross_profit = sum(ganancias)
    gross_loss = abs(sum(perdidas))

    win_rate = (ganadoras / total) * 100 if total else 0
    profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else 0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    rentabilidad = (
        ((capital_final - capital_inicial) / capital_inicial) * 100
        if capital_inicial
        else 0
    )
    promedio_r = sum(r_values) / total if total else 0

    return {
        "ACTIVO": activo,
        "DIRECCION": direccion,
        "MODELO": modelo,
        "SISTEMA": sistema,
        "THRESHOLD": threshold,
        "COST_BPS": cost_bps,
        "TRADES": total,
        "WIN RATE": win_rate,
        "PROFIT FACTOR": profit_factor,
        "EXPECTANCY R": promedio_r,
        "R TOTAL": sum(r_values),
        "CAPITAL INICIAL": capital_inicial,
        "CAPITAL FINAL": capital_final,
        "RENTABILIDAD %": rentabilidad,
        "MAX DRAWDOWN %": _calcular_max_drawdown(capitales),
        "GANANCIA MEDIA": sum(ganancias) / len(ganancias) if ganancias else 0,
        "PERDIDA MEDIA": sum(perdidas) / len(perdidas) if perdidas else 0,
        "PROMEDIO R POR TRADE": promedio_r,
    }


def _ejecutar_escenario(datos, activo, direccion, modelo, sistema, threshold, cost_bps):
    capital = float(CAPITAL_INICIAL)
    capitales = [capital]
    trades = []
    i = 0

    while i < len(datos) - 1:
        fila_signal = datos.iloc[i]

        if not _hay_senal(fila_signal, sistema, direccion, threshold):
            i += 1
            continue

        trade = _resolver_trade(datos, i, direccion, cost_bps)

        if trade is None:
            i += 1
            continue

        riesgo_monetario = capital * RIESGO_POR_TRADE
        ganancia_perdida = riesgo_monetario * trade["r"]
        capital += ganancia_perdida
        capitales.append(capital)

        resultado = "WIN" if ganancia_perdida > 0 else "LOSS"
        fila_entrada = datos.iloc[trade["indice_entrada"]]
        fila_salida = datos.iloc[trade["salida_idx"]]

        trades.append(
            {
                "ACTIVO": activo,
                "DIRECCION": direccion,
                "MODELO": modelo,
                "SISTEMA": sistema,
                "THRESHOLD": threshold,
                "COST_BPS": cost_bps,
                "FOLD_SIGNAL": fila_signal["FOLD"],
                "FECHA_SIGNAL": fila_signal["FECHA"],
                "FECHA_ENTRADA": fila_entrada["FECHA"],
                "FECHA_SALIDA": fila_salida["FECHA"],
                "ENTRY": trade["entry"],
                "STOP_LOSS": trade["stop_loss"],
                "TAKE_PROFIT": trade["take_profit"],
                "EXIT": trade["exit_price"],
                "MOTIVO_SALIDA": trade["motivo"],
                "PROBABILIDAD": fila_signal["PROBABILIDAD"],
                "TARGET_REAL": fila_signal["TARGET_REAL"],
                "BASE_SIGNAL": fila_signal["BASE_SIGNAL"],
                "RESULTADO": resultado,
                "R": trade["r"],
                "GANANCIA_PERDIDA": ganancia_perdida,
                "CAPITAL_DESPUES": capital,
            }
        )

        i = trade["salida_idx"] + 1

    metricas = _calcular_metricas(
        trades=trades,
        capitales=capitales,
        activo=activo,
        direccion=direccion,
        modelo=modelo,
        sistema=sistema,
        threshold=threshold,
        cost_bps=cost_bps,
    )

    return trades, metricas


def ejecutar_oof_backtest(oof_predictions, activo, raiz_proyecto, progress_callback=None):
    datos = _preparar_oof(oof_predictions)
    todos_trades = []
    resumen = []

    for direccion in sorted(datos["DIRECCION"].unique()):
        datos_direccion = datos[datos["DIRECCION"] == direccion].copy()
        base_direccion = (
            datos_direccion
            .sort_values("FECHA")
            .drop_duplicates(subset=["FECHA"], keep="first")
            .reset_index(drop=True)
        )

        if not base_direccion.empty:
            for cost_bps in COST_BPS:
                trades, metricas = _ejecutar_escenario(
                    datos=base_direccion,
                    activo=activo,
                    direccion=direccion,
                    modelo="BASE",
                    sistema="BASE",
                    threshold=PROB_THRESHOLD,
                    cost_bps=cost_bps,
                )
                todos_trades.extend(trades)
                resumen.append(metricas)

        for modelo in sorted(datos_direccion["MODELO"].unique()):
            if progress_callback is not None:
                progress_callback(direccion, modelo)

            base = datos[
                (datos["DIRECCION"] == direccion)
                & (datos["MODELO"] == modelo)
            ].copy()
            base = base.sort_values("FECHA").reset_index(drop=True)

            if base.empty:
                continue

            for sistema in ("ML_ONLY", "BASE_PLUS_ML"):
                for threshold in PROB_THRESHOLDS:
                    for cost_bps in COST_BPS:
                        trades, metricas = _ejecutar_escenario(
                            datos=base,
                            activo=activo,
                            direccion=direccion,
                            modelo=modelo,
                            sistema=sistema,
                            threshold=threshold,
                            cost_bps=cost_bps,
                        )
                        todos_trades.extend(trades)
                        resumen.append(metricas)

    trades_df = pd.DataFrame(todos_trades)
    resumen_df = pd.DataFrame(resumen)

    carpeta_resultados = raiz_proyecto / "results" / activo.lower()
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    ruta_trades = carpeta_resultados / "oof_backtest_trades.csv"
    ruta_summary = carpeta_resultados / "oof_backtest_summary.csv"

    trades_df.to_csv(ruta_trades, index=False)
    resumen_df.to_csv(ruta_summary, index=False)

    return {
        "trades": trades_df,
        "summary": resumen_df,
        "ruta_trades": ruta_trades,
        "ruta_summary": ruta_summary,
    }
