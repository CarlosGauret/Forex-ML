import pandas as pd


SWING_HIGH_CONFIRMED = "SWING_HIGH_CONFIRMED"
SWING_LOW_CONFIRMED = "SWING_LOW_CONFIRMED"
SWING_HIGH_PRICE = "SWING_HIGH_PRICE"
SWING_LOW_PRICE = "SWING_LOW_PRICE"
SWING_HIGH_PIVOT_INDEX = "SWING_HIGH_PIVOT_INDEX"
SWING_LOW_PIVOT_INDEX = "SWING_LOW_PIVOT_INDEX"
LAST_SWING_HIGH = "LAST_CONFIRMED_SWING_HIGH"
LAST_SWING_LOW = "LAST_CONFIRMED_SWING_LOW"
LAST_SWING_HIGH_INDEX = "LAST_CONFIRMED_SWING_HIGH_INDEX"
LAST_SWING_LOW_INDEX = "LAST_CONFIRMED_SWING_LOW_INDEX"


def _validar_parametros(df, high_col, low_col, left_bars, right_bars):
    faltantes = [col for col in (high_col, low_col) if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")

    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars y right_bars deben ser mayores o iguales a 1.")


def _es_swing_high(highs, posicion, left_bars, right_bars):
    actual = highs.iloc[posicion]
    izquierda = highs.iloc[posicion - left_bars:posicion]
    derecha = highs.iloc[posicion + 1:posicion + right_bars + 1]
    return actual > izquierda.max() and actual > derecha.max()


def _es_swing_low(lows, posicion, left_bars, right_bars):
    actual = lows.iloc[posicion]
    izquierda = lows.iloc[posicion - left_bars:posicion]
    derecha = lows.iloc[posicion + 1:posicion + right_bars + 1]
    return actual < izquierda.min() and actual < derecha.min()


def detectar_swings_confirmados(
    df,
    left_bars=2,
    right_bars=2,
    high_col="High",
    low_col="Low",
):
    _validar_parametros(df, high_col, low_col, left_bars, right_bars)

    datos = df.copy()
    highs = pd.to_numeric(datos[high_col], errors="coerce")
    lows = pd.to_numeric(datos[low_col], errors="coerce")

    datos[SWING_HIGH_CONFIRMED] = False
    datos[SWING_LOW_CONFIRMED] = False
    datos[SWING_HIGH_PRICE] = pd.NA
    datos[SWING_LOW_PRICE] = pd.NA
    datos[SWING_HIGH_PIVOT_INDEX] = pd.NA
    datos[SWING_LOW_PIVOT_INDEX] = pd.NA

    ultimo_pivot = len(datos) - right_bars
    for posicion in range(left_bars, ultimo_pivot):
        if pd.isna(highs.iloc[posicion]) or pd.isna(lows.iloc[posicion]):
            continue

        confirmacion = posicion + right_bars
        pivot_index = datos.index[posicion]

        if _es_swing_high(highs, posicion, left_bars, right_bars):
            datos.iloc[
                confirmacion,
                datos.columns.get_loc(SWING_HIGH_CONFIRMED),
            ] = True
            datos.iloc[
                confirmacion,
                datos.columns.get_loc(SWING_HIGH_PRICE),
            ] = float(highs.iloc[posicion])
            datos.iloc[
                confirmacion,
                datos.columns.get_loc(SWING_HIGH_PIVOT_INDEX),
            ] = pivot_index

        if _es_swing_low(lows, posicion, left_bars, right_bars):
            datos.iloc[
                confirmacion,
                datos.columns.get_loc(SWING_LOW_CONFIRMED),
            ] = True
            datos.iloc[
                confirmacion,
                datos.columns.get_loc(SWING_LOW_PRICE),
            ] = float(lows.iloc[posicion])
            datos.iloc[
                confirmacion,
                datos.columns.get_loc(SWING_LOW_PIVOT_INDEX),
            ] = pivot_index

    datos[LAST_SWING_HIGH] = datos[SWING_HIGH_PRICE].ffill()
    datos[LAST_SWING_LOW] = datos[SWING_LOW_PRICE].ffill()
    datos[LAST_SWING_HIGH_INDEX] = datos[SWING_HIGH_PIVOT_INDEX].ffill()
    datos[LAST_SWING_LOW_INDEX] = datos[SWING_LOW_PIVOT_INDEX].ffill()

    return datos


def ultimo_swing_low_confirmado(datos, entry):
    if LAST_SWING_LOW not in datos.columns:
        raise ValueError(f"Falta columna requerida: {LAST_SWING_LOW}")

    swings = pd.to_numeric(datos[LAST_SWING_LOW], errors="coerce").dropna()
    swings = swings[swings < entry]
    if swings.empty:
        return None
    return float(swings.iloc[-1])


def ultimo_swing_high_confirmado(datos, entry):
    if LAST_SWING_HIGH not in datos.columns:
        raise ValueError(f"Falta columna requerida: {LAST_SWING_HIGH}")

    swings = pd.to_numeric(datos[LAST_SWING_HIGH], errors="coerce").dropna()
    swings = swings[swings > entry]
    if swings.empty:
        return None
    return float(swings.iloc[-1])
