import pandas as pd

from config import COSTO_POR_OPERACION


COLUMNAS_REQUERIDAS = ["Open", "High", "Low", "Close", "ATR", "SIGNAL"]


def _validar_columnas(df):
    faltantes = [columna for columna in COLUMNAS_REQUERIDAS if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


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


def _calcular_metricas(trades, capital_inicial, capital_final, capitales):
    total = len(trades)
    ganadoras = sum(1 for trade in trades if trade["RESULTADO"] == "WIN")
    perdedoras = sum(1 for trade in trades if trade["RESULTADO"] == "LOSS")
    ganancias = [trade["GANANCIA_PERDIDA"] for trade in trades if trade["GANANCIA_PERDIDA"] > 0]
    perdidas = [trade["GANANCIA_PERDIDA"] for trade in trades if trade["GANANCIA_PERDIDA"] < 0]
    gross_profit = sum(ganancias)
    gross_loss = abs(sum(perdidas))

    if total == 0:
        win_rate = 0
    else:
        win_rate = (ganadoras / total) * 100

    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else 0
    else:
        profit_factor = gross_profit / gross_loss

    if capital_inicial == 0:
        rentabilidad = 0
    else:
        rentabilidad = ((capital_final - capital_inicial) / capital_inicial) * 100

    return {
        "Total de operaciones": total,
        "Ganadoras": ganadoras,
        "Perdedoras": perdedoras,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "R acumulado": sum(trade["R"] for trade in trades),
        "Capital inicial": capital_inicial,
        "Capital final": capital_final,
        "Rentabilidad %": rentabilidad,
        "Máximo Drawdown %": _calcular_max_drawdown(capitales),
        "Ganancia promedio": sum(ganancias) / len(ganancias) if ganancias else 0,
        "Pérdida promedio": sum(perdidas) / len(perdidas) if perdidas else 0,
    }


def ejecutar_backtest(df, capital_inicial=1000, riesgo_por_trade=0.01):
    _validar_columnas(df)

    datos = df.copy().sort_index()
    activo = datos.attrs.get("activo", "N/A")
    capital = float(capital_inicial)
    capitales = [capital]
    trades = []
    i = 0

    while i < len(datos) - 1:
        fila_signal = datos.iloc[i]
        tipo = fila_signal["SIGNAL"]

        if tipo not in ("BUY", "SELL") or pd.isna(fila_signal["ATR"]) or fila_signal["ATR"] <= 0:
            i += 1
            continue

        fila_entrada = datos.iloc[i + 1]
        entry = float(fila_entrada["Open"])
        atr = float(fila_signal["ATR"])

        if tipo == "BUY":
            stop_loss = entry - (1.5 * atr)
            take_profit = entry + (3.0 * atr)
        else:
            stop_loss = entry + (1.5 * atr)
            take_profit = entry - (3.0 * atr)

        riesgo_monetario = capital * riesgo_por_trade
        fecha_salida = None
        resultado = None
        r = None
        salida_idx = None

        for j in range(i + 1, len(datos)):
            fila_actual = datos.iloc[j]
            high = float(fila_actual["High"])
            low = float(fila_actual["Low"])

            if tipo == "BUY":
                toca_stop = low <= stop_loss
                toca_take = high >= take_profit
            else:
                toca_stop = high >= stop_loss
                toca_take = low <= take_profit

            if toca_stop:
                resultado = "LOSS"
                r = -1
                fecha_salida = datos.index[j]
                salida_idx = j
                break

            if toca_take:
                resultado = "WIN"
                r = 2
                fecha_salida = datos.index[j]
                salida_idx = j
                break

        if resultado is None:
            break

        ganancia_perdida = (riesgo_monetario * r) - COSTO_POR_OPERACION
        capital += ganancia_perdida
        capitales.append(capital)

        trades.append(
            {
                "ACTIVO": activo,
                "TIPO": tipo,
                "FECHA_SIGNAL": datos.index[i],
                "FECHA_ENTRADA": datos.index[i + 1],
                "FECHA_SALIDA": fecha_salida,
                "ENTRY": entry,
                "STOP_LOSS": stop_loss,
                "TAKE_PROFIT": take_profit,
                "RESULTADO": resultado,
                "R": r,
                "GANANCIA_PERDIDA": ganancia_perdida,
                "CAPITAL_DESPUES": capital,
            }
        )

        i = salida_idx + 1

    trades_df = pd.DataFrame(
        trades,
        columns=[
            "ACTIVO",
            "TIPO",
            "FECHA_SIGNAL",
            "FECHA_ENTRADA",
            "FECHA_SALIDA",
            "ENTRY",
            "STOP_LOSS",
            "TAKE_PROFIT",
            "RESULTADO",
            "R",
            "GANANCIA_PERDIDA",
            "CAPITAL_DESPUES",
        ],
    )
    metricas = _calcular_metricas(trades, capital_inicial, capital, capitales)

    return {
        "trades": trades_df,
        "metricas": metricas,
    }
