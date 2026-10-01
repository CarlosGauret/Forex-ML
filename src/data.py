from pathlib import Path

import pandas as pd
import yfinance as yf

from config import ACTIVOS
from src.yfinance_cache import ensure_yfinance_cache


COLUMNAS_PRECIO = ["Open", "High", "Low", "Close", "Volume"]
INTERVALO = "1h"
PERIODOS_H1 = ["2y", "730d", "1y", "60d"]


def _corregir_multiindex(datos):
    if not isinstance(datos.columns, pd.MultiIndex):
        return datos

    for nivel in range(datos.columns.nlevels):
        valores = datos.columns.get_level_values(nivel)
        if set(COLUMNAS_PRECIO).issubset(set(valores)):
            datos = datos.copy()
            datos.columns = valores
            return datos

    datos = datos.copy()
    datos.columns = [
        "_".join(str(parte) for parte in columna if parte)
        for columna in datos.columns.to_flat_index()
    ]
    return datos


def _preparar_datos(datos, activo, ticker, periodo_descargado):
    datos = _corregir_multiindex(datos)

    if "Volume" not in datos.columns:
        datos["Volume"] = 0

    columnas_faltantes = [
        columna for columna in COLUMNAS_PRECIO if columna not in datos.columns
    ]
    if columnas_faltantes:
        faltantes = ", ".join(columnas_faltantes)
        raise ValueError(f"Faltan columnas para {activo}: {faltantes}")

    datos = datos[COLUMNAS_PRECIO]
    datos = datos.dropna()
    datos = datos.sort_index()

    if datos.empty:
        raise ValueError(f"No se descargaron datos validos para {activo} ({ticker}).")

    datos.attrs["periodo_solicitado"] = "2y"
    datos.attrs["periodo_descargado"] = periodo_descargado
    datos.attrs["intervalo"] = INTERVALO

    return datos


def _descargar_y_preparar(ticker, activo):
    errores = []
    ensure_yfinance_cache()

    for periodo in PERIODOS_H1:
        try:
            datos = yf.download(
                ticker,
                interval=INTERVALO,
                period=periodo,
                progress=False,
                auto_adjust=False,
            )

            if datos.empty:
                errores.append(f"{periodo}: descarga vacia")
                continue

            return _preparar_datos(datos, activo, ticker, periodo)
        except Exception as error:
            errores.append(f"{periodo}: {error}")

    detalle = " | ".join(errores)
    raise ValueError(f"No se pudo descargar H1 para {activo} ({ticker}). {detalle}")


def obtener_datos(activo):
    activo = activo.upper()

    if activo not in ACTIVOS:
        disponibles = ", ".join(ACTIVOS.keys())
        raise ValueError(f"Activo no soportado: {activo}. Disponibles: {disponibles}")

    ticker = ACTIVOS[activo]["ticker"]

    datos = _descargar_y_preparar(ticker, activo)

    carpeta_activo = activo.lower()
    ruta_salida = (
        Path(__file__).resolve().parents[1]
        / "data"
        / carpeta_activo
        / f"{carpeta_activo}_h1.csv"
    )
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(ruta_salida)

    return datos
