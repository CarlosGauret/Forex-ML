import pandas as pd


COLUMNAS_BASE = ["Open", "High", "Low", "Close", "Volume"]
COLUMNAS_INDICADORES = [
    "EMA20",
    "EMA50",
    "EMA200",
    "RSI",
    "ATR",
    "RET_1H",
    "RET_4H",
    "RET_24H",
    "VOLATILIDAD",
]
COLUMNAS_FEATURES = COLUMNAS_INDICADORES + [
    "EMA20_EMA50",
    "PRECIO_EMA20",
    "PRECIO_EMA50",
    "PRECIO_EMA200",
    "ATR_PCT",
    "RANGO_VELA",
    "CUERPO_VELA",
    "HORA",
    "DIA_SEMANA",
]
COLUMNAS_DATASET = (
    ["fecha"]
    + COLUMNAS_BASE
    + COLUMNAS_FEATURES
    + ["TARGET_LONG", "TARGET_SHORT"]
)


def _validar_columnas(df, columnas):
    faltantes = [columna for columna in columnas if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


def _asegurar_indicadores(df):
    if all(columna in df.columns for columna in COLUMNAS_INDICADORES):
        return df.copy()

    from src.indicators import calcular_indicadores

    return calcular_indicadores(df)


def _crear_target_long(datos, horizonte=24):
    targets = []

    for i in range(len(datos)):
        fila = datos.iloc[i]
        close = fila["Close"]
        atr = fila["ATR"]

        if pd.isna(close) or pd.isna(atr) or atr <= 0:
            targets.append(pd.NA)
            continue

        take_profit = close + (2 * atr)
        stop_loss = close - (1 * atr)
        objetivo = pd.NA

        for j in range(i + 1, min(i + horizonte + 1, len(datos))):
            futura = datos.iloc[j]

            if futura["Low"] <= stop_loss:
                objetivo = 0
                break

            if futura["High"] >= take_profit:
                objetivo = 1
                break

        targets.append(objetivo)

    return targets


def _crear_target_short(datos, horizonte=24):
    targets = []

    for i in range(len(datos)):
        fila = datos.iloc[i]
        close = fila["Close"]
        atr = fila["ATR"]

        if pd.isna(close) or pd.isna(atr) or atr <= 0:
            targets.append(pd.NA)
            continue

        take_profit = close - (2 * atr)
        stop_loss = close + (1 * atr)
        objetivo = pd.NA

        for j in range(i + 1, min(i + horizonte + 1, len(datos))):
            futura = datos.iloc[j]

            if futura["High"] >= stop_loss:
                objetivo = 0
                break

            if futura["Low"] <= take_profit:
                objetivo = 1
                break

        targets.append(objetivo)

    return targets


def crear_dataset_ml(df):
    datos = _asegurar_indicadores(df)
    _validar_columnas(datos, COLUMNAS_BASE + COLUMNAS_INDICADORES)

    datos = datos.copy()
    datos.index = pd.to_datetime(datos.index)

    datos["EMA20_EMA50"] = (datos["EMA20"] / datos["EMA50"]) - 1
    datos["PRECIO_EMA20"] = (datos["Close"] / datos["EMA20"]) - 1
    datos["PRECIO_EMA50"] = (datos["Close"] / datos["EMA50"]) - 1
    datos["PRECIO_EMA200"] = (datos["Close"] / datos["EMA200"]) - 1
    datos["ATR_PCT"] = datos["ATR"] / datos["Close"]
    datos["RANGO_VELA"] = (datos["High"] - datos["Low"]) / datos["Close"]
    datos["CUERPO_VELA"] = (datos["Close"] - datos["Open"]).abs() / datos["Close"]
    datos["HORA"] = datos.index.hour
    datos["DIA_SEMANA"] = datos.index.dayofweek
    datos["TARGET_LONG"] = _crear_target_long(datos)
    datos["TARGET_SHORT"] = _crear_target_short(datos)

    datos = datos.replace([float("inf"), float("-inf")], pd.NA)
    datos = datos.dropna(subset=COLUMNAS_FEATURES)

    dataset = datos.reset_index(names="fecha")
    return dataset[COLUMNAS_DATASET]


def calcular_estadisticas_dataset(activo, dataset):
    if dataset.empty:
        return {
            "ACTIVO": activo,
            "VELAS": 0,
            "FECHA INICIAL": "N/A",
            "FECHA FINAL": "N/A",
            "TARGET LONG 1": 0,
            "TARGET LONG 0": 0,
            "TARGET SHORT 1": 0,
            "TARGET SHORT 0": 0,
        }

    return {
        "ACTIVO": activo,
        "VELAS": len(dataset),
        "FECHA INICIAL": dataset["fecha"].iloc[0],
        "FECHA FINAL": dataset["fecha"].iloc[-1],
        "TARGET LONG 1": int((dataset["TARGET_LONG"] == 1).sum()),
        "TARGET LONG 0": int((dataset["TARGET_LONG"] == 0).sum()),
        "TARGET SHORT 1": int((dataset["TARGET_SHORT"] == 1).sum()),
        "TARGET SHORT 0": int((dataset["TARGET_SHORT"] == 0).sum()),
    }
