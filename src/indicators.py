import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


COLUMNAS_BASE = ["Open", "High", "Low", "Close", "Volume"]
COLUMNAS_TENDENCIA = ["Close", "EMA20", "EMA50", "EMA200"]


def _validar_columnas(df, columnas):
    faltantes = [columna for columna in columnas if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


def calcular_indicadores(df):
    _validar_columnas(df, COLUMNAS_BASE)

    datos = df.copy()

    for columna in COLUMNAS_BASE:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos["EMA20"] = EMAIndicator(close=datos["Close"], window=20).ema_indicator()
    datos["EMA50"] = EMAIndicator(close=datos["Close"], window=50).ema_indicator()
    datos["EMA200"] = EMAIndicator(close=datos["Close"], window=200).ema_indicator()
    datos["RSI"] = RSIIndicator(close=datos["Close"], window=14).rsi()
    datos["ATR"] = AverageTrueRange(
        high=datos["High"],
        low=datos["Low"],
        close=datos["Close"],
        window=14,
    ).average_true_range()
    datos["RET_1H"] = datos["Close"].pct_change(periods=1)
    datos["RET_4H"] = datos["Close"].pct_change(periods=4)
    datos["RET_24H"] = datos["Close"].pct_change(periods=24)
    datos["VOLATILIDAD"] = datos["RET_1H"].rolling(window=24).std()

    return datos


def determinar_tendencia(df):
    _validar_columnas(df, COLUMNAS_TENDENCIA)

    if df.empty:
        raise ValueError("No hay datos para determinar la tendencia.")

    ultima = df.iloc[-1]

    if ultima[COLUMNAS_TENDENCIA].isna().any():
        return "LATERAL"

    if ultima["Close"] > ultima["EMA200"] and ultima["EMA20"] > ultima["EMA50"]:
        return "ALCISTA"

    if ultima["Close"] < ultima["EMA200"] and ultima["EMA20"] < ultima["EMA50"]:
        return "BAJISTA"

    return "LATERAL"
