import pandas as pd


COLUMNAS_REQUERIDAS = ["Close", "EMA20", "EMA50", "EMA200", "RSI", "ATR"]


def _validar_columnas(df):
    faltantes = [columna for columna in COLUMNAS_REQUERIDAS if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


def generar_senales(df):
    _validar_columnas(df)

    datos = df.copy()

    condicion_buy = (
        (datos["Close"] > datos["EMA200"])
        & (datos["EMA20"] > datos["EMA50"])
        & (datos["RSI"] > 50)
        & (datos["RSI"] < 70)
    )
    condicion_sell = (
        (datos["Close"] < datos["EMA200"])
        & (datos["EMA20"] < datos["EMA50"])
        & (datos["RSI"] > 30)
        & (datos["RSI"] < 50)
    )

    nueva_buy = condicion_buy & ~condicion_buy.shift(1, fill_value=False)
    nueva_sell = condicion_sell & ~condicion_sell.shift(1, fill_value=False)

    datos["SIGNAL"] = "WAIT"
    datos.loc[nueva_buy, "SIGNAL"] = "BUY"
    datos.loc[nueva_sell, "SIGNAL"] = "SELL"

    datos["ENTRY"] = pd.NA
    datos["STOP_LOSS"] = pd.NA
    datos["TAKE_PROFIT"] = pd.NA

    return datos
