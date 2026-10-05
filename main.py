import math
from pathlib import Path
import sys
import time

from config import ACTIVOS, DEMO_EXECUTION_ENABLED, TRADING_ENABLED
from src.display import formatear_direccion, formatear_texto_direcciones


RAIZ_PROYECTO = Path(__file__).resolve().parent


def mostrar_activos_disponibles():
    print("Activos disponibles:")
    print()
    for activo in ACTIVOS:
        print(activo)
    print("ALL")


def formatear_numero(valor):
    if valor is None:
        return "N/A"

    if math.isnan(valor):
        return "N/A"

    if math.isinf(valor):
        return "INF"

    if abs(valor) >= 1000:
        return f"{valor:.2f}"

    if abs(valor) >= 10:
        return f"{valor:.3f}"

    return f"{valor:.5f}"


def formatear_fecha(valor):
    if valor is None:
        return "N/A"

    return str(valor)


def procesar_activo(activo):
    from src.data import obtener_datos
    from src.features import calcular_estadisticas_dataset, crear_dataset_ml
    from src.indicators import calcular_indicadores, determinar_tendencia

    datos = obtener_datos(activo)
    periodo_descargado = datos.attrs.get("periodo_descargado", "N/A")
    intervalo = datos.attrs.get("intervalo", "1h")
    fecha_inicial = datos.index[0] if not datos.empty else None
    fecha_final = datos.index[-1] if not datos.empty else None

    datos = calcular_indicadores(datos)
    tendencia = determinar_tendencia(datos)
    carpeta_activo = activo.lower()
    ruta_indicadores = (
        RAIZ_PROYECTO
        / "data"
        / carpeta_activo
        / f"{carpeta_activo}_h1_indicators.csv"
    )
    datos.to_csv(ruta_indicadores)

    dataset_ml = crear_dataset_ml(datos)
    ruta_dataset_ml = (
        RAIZ_PROYECTO
        / "data"
        / carpeta_activo
        / "ml_dataset.csv"
    )
    dataset_ml.to_csv(ruta_dataset_ml, index=False)
    estadisticas_ml = calcular_estadisticas_dataset(activo, dataset_ml)
    ultima = datos.iloc[-1]

    return {
        "activo": activo,
        "datos": datos,
        "dataset_ml": dataset_ml,
        "velas": len(datos),
        "fecha_inicial": fecha_inicial,
        "fecha_final": fecha_final,
        "periodo_descargado": periodo_descargado,
        "intervalo": intervalo,
        "precio": float(ultima["Close"]),
        "ema20": float(ultima["EMA20"]),
        "ema50": float(ultima["EMA50"]),
        "ema200": float(ultima["EMA200"]),
        "rsi": float(ultima["RSI"]),
        "atr": float(ultima["ATR"]),
        "volatilidad": float(ultima["VOLATILIDAD"]),
        "tendencia": tendencia,
        "ruta_indicadores": ruta_indicadores,
        "ruta_dataset_ml": ruta_dataset_ml,
        "estadisticas_ml": estadisticas_ml,
    }


def procesar_backtest_activo(activo):
    import pandas as pd

    from src.backtest import ejecutar_backtest
    from src.strategy import generar_senales

    resultado = procesar_activo(activo)
    datos = generar_senales(resultado["datos"])
    datos.attrs["activo"] = activo

    carpeta_activo = activo.lower()
    ruta_indicadores = (
        RAIZ_PROYECTO
        / "data"
        / carpeta_activo
        / f"{carpeta_activo}_h1_indicators.csv"
    )
    datos.to_csv(ruta_indicadores)

    resultado_backtest = ejecutar_backtest(datos)
    trades = resultado_backtest["trades"]
    metricas = resultado_backtest["metricas"]

    carpeta_resultados = RAIZ_PROYECTO / "results" / carpeta_activo
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    ruta_trades = carpeta_resultados / "trades.csv"
    ruta_summary = carpeta_resultados / "summary.csv"

    trades.to_csv(ruta_trades, index=False)
    pd.DataFrame([metricas]).to_csv(ruta_summary, index=False)

    return {
        "activo": activo,
        "trades": trades,
        "metricas": metricas,
        "velas": resultado["velas"],
        "fecha_inicial": resultado["fecha_inicial"],
        "fecha_final": resultado["fecha_final"],
        "periodo_descargado": resultado["periodo_descargado"],
        "intervalo": resultado["intervalo"],
        "ruta_trades": ruta_trades,
        "ruta_summary": ruta_summary,
    }


def procesar_ml_activo(activo):
    from src.model import entrenar_modelos_activo

    resultado = procesar_activo(activo)
    resultados_ml = entrenar_modelos_activo(
        dataset=resultado["dataset_ml"],
        activo=activo,
        raiz_proyecto=RAIZ_PROYECTO,
    )

    return {
        "activo": activo,
        "resultados_ml": resultados_ml,
        "velas": resultado["velas"],
        "fecha_inicial": resultado["fecha_inicial"],
        "fecha_final": resultado["fecha_final"],
        "periodo_descargado": resultado["periodo_descargado"],
        "intervalo": resultado["intervalo"],
        "ruta_dataset_ml": resultado["ruta_dataset_ml"],
    }


def procesar_wf_activo(activo):
    from src.model import ejecutar_walkforward_activo

    resultado = procesar_activo(activo)
    resultados_wf = ejecutar_walkforward_activo(
        dataset=resultado["dataset_ml"],
        activo=activo,
        raiz_proyecto=RAIZ_PROYECTO,
    )

    return {
        "activo": activo,
        "resultados_wf": resultados_wf,
        "velas": resultado["velas"],
        "fecha_inicial": resultado["fecha_inicial"],
        "fecha_final": resultado["fecha_final"],
        "periodo_descargado": resultado["periodo_descargado"],
        "intervalo": resultado["intervalo"],
        "ruta_dataset_ml": resultado["ruta_dataset_ml"],
    }


def procesar_oof_backtest_activo(activo, show_progress=False):
    from src.ml_backtest import cargar_oof_si_valido, ejecutar_oof_backtest

    carpeta_activo = activo.lower()
    ruta_oof = RAIZ_PROYECTO / "results" / carpeta_activo / "oof_predictions.csv"
    oof = cargar_oof_si_valido(ruta_oof, activo=activo)
    fuente_oof = "EXISTENTE"
    resultado_wf = None

    if oof is None:
        if show_progress:
            print("  generando OOF walk-forward...", flush=True)
        fuente_oof = "GENERADO"
        resultado_wf = procesar_wf_activo(activo)
        oof = resultado_wf["resultados_wf"]["oof"]
        ruta_oof = resultado_wf["resultados_wf"]["ruta_oof"]
    elif show_progress:
        print("  usando OOF existente...", flush=True)

    def progress_callback(direccion, modelo):
        if show_progress:
            print(f"  {formatear_direccion(direccion)} {modelo}...", flush=True)

    resultado_backtest = ejecutar_oof_backtest(
        oof_predictions=oof,
        activo=activo,
        raiz_proyecto=RAIZ_PROYECTO,
        progress_callback=progress_callback,
    )

    if resultado_wf is not None:
        velas = resultado_wf["velas"]
        fecha_inicial = resultado_wf["fecha_inicial"]
        fecha_final = resultado_wf["fecha_final"]
        periodo_descargado = resultado_wf["periodo_descargado"]
        intervalo = resultado_wf["intervalo"]
    else:
        fechas = oof["FECHA"]
        velas = oof["FECHA"].nunique()
        fecha_inicial = fechas.min()
        fecha_final = fechas.max()
        periodo_descargado = "N/A"
        intervalo = "1h"

    return {
        "activo": activo,
        "resultado_backtest": resultado_backtest,
        "ruta_oof": ruta_oof,
        "fuente_oof": fuente_oof,
        "velas": velas,
        "fecha_inicial": fecha_inicial,
        "fecha_final": fecha_final,
        "periodo_descargado": periodo_descargado,
        "intervalo": intervalo,
    }


def ejecutar_descarga_individual(activo):
    resultado = procesar_activo(activo)
    carpeta_activo = activo.lower()

    print("=" * 50)
    print("FOREX ML - SISTEMA MULTI-ACTIVO")
    print("=" * 50)
    print(f"Activo: {activo} - {ACTIVOS[activo]['nombre']}")
    print(f"Tipo: {ACTIVOS[activo]['tipo']}")
    print("Timeframe: H1")
    print(f"Periodo descargado: {resultado['periodo_descargado']}")
    print(f"Intervalo: {resultado['intervalo']}")
    print("Modo: DESCARGA DE DATOS")
    print("Dinero real: DESACTIVADO")
    print(f"Filas descargadas: {resultado['velas']}")
    print(f"Fecha inicial: {formatear_fecha(resultado['fecha_inicial'])}")
    print(f"Fecha final: {formatear_fecha(resultado['fecha_final'])}")
    print(f"Archivo guardado: data/{carpeta_activo}/{carpeta_activo}_h1.csv")
    print(
        "Archivo con indicadores: "
        f"data/{carpeta_activo}/{carpeta_activo}_h1_indicators.csv"
    )
    print(f"Dataset ML: data/{carpeta_activo}/ml_dataset.csv")
    print()
    print(f"Precio actual: {formatear_numero(resultado['precio'])}")
    print(f"EMA20: {formatear_numero(resultado['ema20'])}")
    print(f"EMA50: {formatear_numero(resultado['ema50'])}")
    print(f"EMA200: {formatear_numero(resultado['ema200'])}")
    print(f"RSI: {formatear_numero(resultado['rsi'])}")
    print(f"ATR: {formatear_numero(resultado['atr'])}")
    print(f"Volatilidad: {formatear_numero(resultado['volatilidad'])}")
    print(f"Tendencia: {resultado['tendencia']}")
    print()
    print("Estadisticas dataset ML:")
    print(f"Velas utiles: {resultado['estadisticas_ml']['VELAS']}")
    print(f"Target LONG (COMPRAR) 1: {resultado['estadisticas_ml']['TARGET LONG 1']}")
    print(f"Target LONG (COMPRAR) 0: {resultado['estadisticas_ml']['TARGET LONG 0']}")
    print(f"Target SHORT (VENDER) 1: {resultado['estadisticas_ml']['TARGET SHORT 1']}")
    print(f"Target SHORT (VENDER) 0: {resultado['estadisticas_ml']['TARGET SHORT 0']}")


def ejecutar_descarga_todos():
    resultados = []

    for activo in ACTIVOS:
        try:
            resultado = procesar_activo(activo)
            resultados.append(
                {
                    "activo": activo,
                    "estado": "OK",
                    "precio": resultado["precio"],
                    "rsi": resultado["rsi"],
                    "atr": resultado["atr"],
                    "tendencia": resultado["tendencia"],
                    "periodo": resultado["periodo_descargado"],
                    "estadisticas_ml": resultado["estadisticas_ml"],
                    "error": "",
                }
            )
        except Exception as error:
            resultados.append(
                {
                    "activo": activo,
                    "estado": "ERROR",
                    "precio": None,
                    "rsi": None,
                    "atr": None,
                    "tendencia": "N/A",
                    "periodo": "N/A",
                    "estadisticas_ml": None,
                    "error": str(error),
                }
            )

    print("=" * 40)
    print("DESCARGA MULTI-ACTIVO")
    print("=" * 40)
    print()
    print(
        f"{'ACTIVO':<10} {'PERIODO':<8} {'VELAS':>7} "
        f"{'FECHA INICIAL':<25} {'FECHA FINAL':<25} "
        f"{'TL 1':>6} {'TL 0':>6} {'TS 1':>6} {'TS 0':>6}"
    )
    print("-" * 110)

    for resultado in resultados:
        if resultado["estado"] == "OK":
            estadisticas = resultado["estadisticas_ml"]
            print(
                f"{resultado['activo']:<10} "
                f"{resultado['periodo']:<8} "
                f"{estadisticas['VELAS']:>7} "
                f"{formatear_fecha(estadisticas['FECHA INICIAL']):<25} "
                f"{formatear_fecha(estadisticas['FECHA FINAL']):<25} "
                f"{estadisticas['TARGET LONG 1']:>6} "
                f"{estadisticas['TARGET LONG 0']:>6} "
                f"{estadisticas['TARGET SHORT 1']:>6} "
                f"{estadisticas['TARGET SHORT 0']:>6}"
            )
        else:
            print(
                f"{resultado['activo']:<10} "
                f"{'ERROR':<8} "
                f"{'ERROR':>7} "
                f"{'ERROR':<25} "
                f"{'ERROR':<25} "
                f"{'ERROR':>6} "
                f"{'ERROR':>6} "
                f"{'ERROR':>6} "
                f"{'ERROR':>6}"
            )

    print()
    print("=" * 40)

    errores = [resultado for resultado in resultados if resultado["estado"] == "ERROR"]
    if errores:
        print()
        print("Errores:")
        for resultado in errores:
            print(f"{resultado['activo']}: {resultado['error']}")


def imprimir_resumen_backtest(resultado):
    metricas = resultado["metricas"]
    activo = resultado["activo"]
    carpeta_activo = activo.lower()

    print("=" * 50)
    print("BACKTEST MULTI-ACTIVO")
    print("=" * 50)
    print(f"Activo: {activo} - {ACTIVOS[activo]['nombre']}")
    print("Capital inicial ficticio: 1000")
    print("Riesgo por operacion: 1%")
    print(f"Periodo descargado: {resultado['periodo_descargado']}")
    print(f"Intervalo: {resultado['intervalo']}")
    print(f"Velas descargadas: {resultado['velas']}")
    print(f"Fecha inicial: {formatear_fecha(resultado['fecha_inicial'])}")
    print(f"Fecha final: {formatear_fecha(resultado['fecha_final'])}")
    print("Dinero real: DESACTIVADO")
    print(f"Trades guardados: results/{carpeta_activo}/trades.csv")
    print(f"Resumen guardado: results/{carpeta_activo}/summary.csv")
    print()
    print(f"Total de operaciones: {metricas['Total de operaciones']}")
    print(f"Ganadoras: {metricas['Ganadoras']}")
    print(f"Perdedoras: {metricas['Perdedoras']}")
    print(f"Win Rate: {formatear_numero(metricas['Win Rate'])}%")
    print(f"Profit Factor: {formatear_numero(metricas['Profit Factor'])}")
    print(f"R acumulado: {formatear_numero(metricas['R acumulado'])}")
    print(f"Capital final: {formatear_numero(metricas['Capital final'])}")
    print(f"Rentabilidad: {formatear_numero(metricas['Rentabilidad %'])}%")
    print(f"Max Drawdown: {formatear_numero(metricas['Máximo Drawdown %'])}%")


def ejecutar_backtest_individual(activo):
    resultado = procesar_backtest_activo(activo)
    imprimir_resumen_backtest(resultado)


def ejecutar_backtest_todos():
    resultados = []

    for activo in ACTIVOS:
        try:
            resultado = procesar_backtest_activo(activo)
            resultados.append(
                {
                    "activo": activo,
                    "estado": "OK",
                    "metricas": resultado["metricas"],
                    "error": "",
                }
            )
        except Exception as error:
            resultados.append(
                {
                    "activo": activo,
                    "estado": "ERROR",
                    "metricas": None,
                    "error": str(error),
                }
            )

    print("=" * 95)
    print("BACKTEST MULTI-ACTIVO")
    print("=" * 95)
    print()
    print(
        f"{'ACTIVO':<10} | {'TRADES':>6} | {'WIN RATE':>9} | "
        f"{'PROFIT FACTOR':>13} | {'R':>8} | {'CAPITAL FINAL':>13} | "
        f"{'RENTABILIDAD':>12} | {'MAX DRAWDOWN':>12}"
    )
    print("-" * 95)

    for resultado in resultados:
        if resultado["estado"] != "OK":
            print(
                f"{resultado['activo']:<10} | {'ERROR':>6} | {'ERROR':>9} | "
                f"{'ERROR':>13} | {'ERROR':>8} | {'ERROR':>13} | "
                f"{'ERROR':>12} | {'ERROR':>12}"
            )
            continue

        metricas = resultado["metricas"]
        print(
            f"{resultado['activo']:<10} | "
            f"{metricas['Total de operaciones']:>6} | "
            f"{formatear_numero(metricas['Win Rate']):>8}% | "
            f"{formatear_numero(metricas['Profit Factor']):>13} | "
            f"{formatear_numero(metricas['R acumulado']):>8} | "
            f"{formatear_numero(metricas['Capital final']):>13} | "
            f"{formatear_numero(metricas['Rentabilidad %']):>11}% | "
            f"{formatear_numero(metricas['Máximo Drawdown %']):>11}%"
        )

    print()
    print("=" * 95)

    errores = [resultado for resultado in resultados if resultado["estado"] == "ERROR"]
    if errores:
        print()
        print("Errores:")
        for resultado in errores:
            print(f"{resultado['activo']}: {resultado['error']}")


def formatear_metrica(valor):
    if valor is None:
        return "N/A"

    return formatear_numero(float(valor))


def imprimir_matriz_confusion(matriz):
    print("Matriz de confusion:")
    print("          PRED 0  PRED 1")
    print(f"REAL 0    {matriz[0][0]:>6}  {matriz[0][1]:>6}")
    print(f"REAL 1    {matriz[1][0]:>6}  {matriz[1][1]:>6}")


def imprimir_resumen_ml(resultado):
    activo = resultado["activo"]
    carpeta_activo = activo.lower()

    print("=" * 60)
    print("MACHINE LEARNING MULTI-ACTIVO")
    print("=" * 60)
    print(f"Activo: {activo} - {ACTIVOS[activo]['nombre']}")
    print(f"Periodo descargado: {resultado['periodo_descargado']}")
    print(f"Intervalo: {resultado['intervalo']}")
    print(f"Velas descargadas: {resultado['velas']}")
    print(f"Fecha inicial: {formatear_fecha(resultado['fecha_inicial'])}")
    print(f"Fecha final: {formatear_fecha(resultado['fecha_final'])}")
    print(f"Dataset ML: data/{carpeta_activo}/ml_dataset.csv")
    print("Predicciones: NO conectadas a trading ni backtest")

    for lado, datos_modelo in resultado["resultados_ml"].items():
        metricas = datos_modelo["metricas"]
        print()
        print("-" * 60)
        print(f"Modelo {formatear_direccion(lado)}")
        print(f"Modelo guardado: models/{carpeta_activo}/random_forest_{lado.lower()}.pkl")
        print(f"Test guardado: results/{carpeta_activo}/ml_{lado.lower()}_test.csv")
        print(
            "Importancias: "
            f"results/{carpeta_activo}/feature_importance_{lado.lower()}.csv"
        )
        print(
            f"Filas TRAIN/VALIDATION/TEST: "
            f"{datos_modelo['n_train']} / "
            f"{datos_modelo['n_validation']} / "
            f"{datos_modelo['n_test']}"
        )

        for split in ("TRAIN", "VALIDATION", "TEST"):
            split_metricas = metricas[split]
            print()
            print(f"{split}:")
            print(f"Accuracy: {formatear_metrica(split_metricas['Accuracy'])}")
            print(f"Precision: {formatear_metrica(split_metricas['Precision'])}")
            print(f"Recall: {formatear_metrica(split_metricas['Recall'])}")
            print(f"F1: {formatear_metrica(split_metricas['F1'])}")
            print(f"ROC AUC: {formatear_metrica(split_metricas['ROC AUC'])}")
            imprimir_matriz_confusion(split_metricas["Confusion Matrix"])

        print()
        print("Importancia de variables:")
        for fila in datos_modelo["importancias"].itertuples(index=False):
            print(f"{fila.feature:<15} {formatear_numero(float(fila.importance))}")


def ejecutar_ml_individual(activo):
    resultado = procesar_ml_activo(activo)
    imprimir_resumen_ml(resultado)


def _filas_resumen_ml(resultado):
    filas = []

    for lado, datos_modelo in resultado["resultados_ml"].items():
        metricas = datos_modelo["metricas"]
        filas.append(
            {
                "activo": resultado["activo"],
                "modelo": lado,
                "train_auc": metricas["TRAIN"]["ROC AUC"],
                "validation_auc": metricas["VALIDATION"]["ROC AUC"],
                "test_auc": metricas["TEST"]["ROC AUC"],
                "precision": metricas["TEST"]["Precision"],
                "recall": metricas["TEST"]["Recall"],
                "f1": metricas["TEST"]["F1"],
                "n_test": datos_modelo["n_test"],
            }
        )

    return filas


def ejecutar_ml_todos():
    filas = []
    errores = []

    for activo in ACTIVOS:
        try:
            resultado = procesar_ml_activo(activo)
            filas.extend(_filas_resumen_ml(resultado))
        except Exception as error:
            errores.append({"activo": activo, "error": str(error)})

    print("=" * 100)
    print("MACHINE LEARNING MULTI-ACTIVO")
    print("=" * 100)
    print()
    print(
        f"{'ACTIVO':<10} {'MODELO':<15} {'TRAIN AUC':>10} "
        f"{'VAL AUC':>10} {'TEST AUC':>10} {'PRECISION':>10} "
        f"{'RECALL':>10} {'F1':>10} {'N TEST':>8}"
    )
    print("-" * 100)

    for fila in filas:
        print(
            f"{fila['activo']:<10} "
            f"{formatear_direccion(fila['modelo']):<15} "
            f"{formatear_metrica(fila['train_auc']):>10} "
            f"{formatear_metrica(fila['validation_auc']):>10} "
            f"{formatear_metrica(fila['test_auc']):>10} "
            f"{formatear_metrica(fila['precision']):>10} "
            f"{formatear_metrica(fila['recall']):>10} "
            f"{formatear_metrica(fila['f1']):>10} "
            f"{fila['n_test']:>8}"
        )

    for error in errores:
        print(
            f"{error['activo']:<10} "
            f"{'ERROR':<8} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>8}"
        )

    print()
    print("=" * 100)

    if errores:
        print()
        print("Errores:")
        for error in errores:
            print(f"{error['activo']}: {error['error']}")


def _filas_resumen_wf(resultado):
    filas = []

    for direccion, datos_wf in resultado["resultados_wf"].items():
        for _, fila in datos_wf["resumen"].iterrows():
            filas.append(
                {
                    "activo": resultado["activo"],
                    "direccion": direccion,
                    "modelo": fila["MODELO"],
                    "auc_medio": fila["AUC MEDIO"],
                    "auc_std": fila["AUC STD"],
                    "auc_min": fila["AUC MIN"],
                    "auc_max": fila["AUC MAX"],
                    "f1_medio": fila["F1 MEDIO"],
                    "precision_media": fila["PRECISION MEDIA"],
                    "recall_medio": fila["RECALL MEDIO"],
                }
            )

    return filas


def imprimir_resumen_wf(resultado):
    activo = resultado["activo"]
    carpeta_activo = activo.lower()

    print("=" * 80)
    print("WALK-FORWARD VALIDATION MULTI-ACTIVO")
    print("=" * 80)
    print(f"Activo: {activo} - {ACTIVOS[activo]['nombre']}")
    print(f"Periodo descargado: {resultado['periodo_descargado']}")
    print(f"Intervalo: {resultado['intervalo']}")
    print(f"Velas descargadas: {resultado['velas']}")
    print(f"Fecha inicial: {formatear_fecha(resultado['fecha_inicial'])}")
    print(f"Fecha final: {formatear_fecha(resultado['fecha_final'])}")
    print("Embargo: 24 velas")
    print("Predicciones: NO conectadas a trading ni backtest")
    print()
    print(
        f"{'DIRECCION':<15} {'MODELO':<10} {'AUC MEDIO':>10} "
        f"{'AUC STD':>10} {'AUC MIN':>10} {'AUC MAX':>10} "
        f"{'F1 MEDIO':>10} {'PRECISION':>10} {'RECALL':>10}"
    )
    print("-" * 100)

    for fila in _filas_resumen_wf(resultado):
        print(
            f"{formatear_direccion(fila['direccion']):<15} "
            f"{fila['modelo']:<10} "
            f"{formatear_metrica(fila['auc_medio']):>10} "
            f"{formatear_metrica(fila['auc_std']):>10} "
            f"{formatear_metrica(fila['auc_min']):>10} "
            f"{formatear_metrica(fila['auc_max']):>10} "
            f"{formatear_metrica(fila['f1_medio']):>10} "
            f"{formatear_metrica(fila['precision_media']):>10} "
            f"{formatear_metrica(fila['recall_medio']):>10}"
        )

    print()
    print(f"Detalle LONG (COMPRAR): results/{carpeta_activo}/walkforward_long.csv")
    print(f"Detalle SHORT (VENDER): results/{carpeta_activo}/walkforward_short.csv")


def ejecutar_wf_individual(activo):
    resultado = procesar_wf_activo(activo)
    imprimir_resumen_wf(resultado)


def ejecutar_wf_todos():
    filas = []
    errores = []

    for activo in ACTIVOS:
        try:
            resultado = procesar_wf_activo(activo)
            filas.extend(_filas_resumen_wf(resultado))
        except Exception as error:
            errores.append({"activo": activo, "error": str(error)})

    print("=" * 115)
    print("WALK-FORWARD VALIDATION MULTI-ACTIVO")
    print("=" * 115)
    print()
    print(
        f"{'ACTIVO':<10} {'DIRECCION':<15} {'MODELO':<10} "
        f"{'AUC MEDIO':>10} {'AUC STD':>10} {'AUC MIN':>10} "
        f"{'AUC MAX':>10} {'F1 MEDIO':>10} {'PRECISION':>10} {'RECALL':>10}"
    )
    print("-" * 115)

    for fila in filas:
        print(
            f"{fila['activo']:<10} "
            f"{formatear_direccion(fila['direccion']):<15} "
            f"{fila['modelo']:<10} "
            f"{formatear_metrica(fila['auc_medio']):>10} "
            f"{formatear_metrica(fila['auc_std']):>10} "
            f"{formatear_metrica(fila['auc_min']):>10} "
            f"{formatear_metrica(fila['auc_max']):>10} "
            f"{formatear_metrica(fila['f1_medio']):>10} "
            f"{formatear_metrica(fila['precision_media']):>10} "
            f"{formatear_metrica(fila['recall_medio']):>10}"
        )

    for error in errores:
        print(
            f"{error['activo']:<10} "
            f"{'ERROR':<10} "
            f"{'ERROR':<10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10} "
            f"{'ERROR':>10}"
        )

    print()
    print("=" * 115)

    if errores:
        print()
        print("Errores:")
        for error in errores:
            print(f"{error['activo']}: {error['error']}")


def _filas_resumen_oof_backtest(resultado):
    resumen = resultado["resultado_backtest"]["summary"]
    filas = []

    for _, fila in resumen.iterrows():
        filas.append(
            {
                "ACTIVO": fila["ACTIVO"],
                "DIRECCION": fila["DIRECCION"],
                "MODELO": fila["MODELO"],
                "SISTEMA": fila["SISTEMA"],
                "THRESHOLD": fila["THRESHOLD"],
                "COST_BPS": fila["COST_BPS"],
                "TRADES": fila["TRADES"],
                "WIN RATE": fila["WIN RATE"],
                "PROFIT FACTOR": fila["PROFIT FACTOR"],
                "EXPECTANCY": fila["EXPECTANCY R"],
                "R TOTAL": fila["R TOTAL"],
                "RETURN %": fila["RENTABILIDAD %"],
                "MAX DD %": fila["MAX DRAWDOWN %"],
            }
        )

    return filas


def imprimir_tabla_oof_backtest(filas):
    print(
        f"{'ACTIVO':<10} {'DIRECCION':<15} {'MODELO':<10} {'SISTEMA':<13} "
        f"{'THR':>5} {'COST':>5} {'TRADES':>7} {'WIN RATE':>9} "
        f"{'PF':>8} {'EXPECT':>8} {'R TOTAL':>8} {'RETURN %':>9} {'MAX DD %':>9}"
    )
    print("-" * 125)

    for fila in filas:
        print(
            f"{fila['ACTIVO']:<10} "
            f"{formatear_direccion(fila['DIRECCION']):<15} "
            f"{fila['MODELO']:<10} "
            f"{fila['SISTEMA']:<13} "
            f"{formatear_numero(float(fila['THRESHOLD'])):>5} "
            f"{int(fila['COST_BPS']):>5} "
            f"{int(fila['TRADES']):>7} "
            f"{formatear_metrica(fila['WIN RATE']):>9} "
            f"{formatear_metrica(fila['PROFIT FACTOR']):>8} "
            f"{formatear_metrica(fila['EXPECTANCY']):>8} "
            f"{formatear_metrica(fila['R TOTAL']):>8} "
            f"{formatear_metrica(fila['RETURN %']):>9} "
            f"{formatear_metrica(fila['MAX DD %']):>9}"
        )


def imprimir_resumen_oof_backtest(resultado):
    activo = resultado["activo"]
    carpeta_activo = activo.lower()
    filas = _filas_resumen_oof_backtest(resultado)

    print("=" * 125)
    print("BACKTEST ECONOMICO CON PREDICCIONES OUT-OF-FOLD")
    print("=" * 125)
    print(f"Activo: {activo} - {ACTIVOS[activo]['nombre']}")
    print(f"Periodo descargado: {resultado['periodo_descargado']}")
    print(f"Intervalo: {resultado['intervalo']}")
    print(f"Velas descargadas: {resultado['velas']}")
    print(f"Fecha inicial: {formatear_fecha(resultado['fecha_inicial'])}")
    print(f"Fecha final: {formatear_fecha(resultado['fecha_final'])}")
    print(f"OOF usado: {resultado['fuente_oof']}")
    print(f"OOF: results/{carpeta_activo}/oof_predictions.csv")
    print(f"Trades: results/{carpeta_activo}/oof_backtest_trades.csv")
    print(f"Resumen: results/{carpeta_activo}/oof_backtest_summary.csv")
    print("Predicciones usadas: solo out-of-fold")
    print()
    imprimir_tabla_oof_backtest(filas)
    print()
    print("=" * 125)


def ejecutar_oof_backtest_individual(activo):
    resultado = procesar_oof_backtest_activo(activo)
    imprimir_resumen_oof_backtest(resultado)


def ejecutar_oof_backtest_todos():
    filas = []
    errores = []
    inicio_total = time.perf_counter()
    total_activos = len(ACTIVOS)

    for indice, activo in enumerate(ACTIVOS, start=1):
        inicio_activo = time.perf_counter()
        print(f"[{indice}/{total_activos}] {activo}", flush=True)
        try:
            resultado = procesar_oof_backtest_activo(activo, show_progress=True)
            filas.extend(_filas_resumen_oof_backtest(resultado))
            duracion = time.perf_counter() - inicio_activo
            print(f"  completado en {duracion:.2f} segundos", flush=True)
            print(flush=True)
        except Exception as error:
            errores.append({"activo": activo, "error": str(error)})
            duracion = time.perf_counter() - inicio_activo
            print(f"  ERROR en {duracion:.2f} segundos: {error}", flush=True)
            print(flush=True)

    print("=" * 125)
    print("BACKTEST ECONOMICO CON PREDICCIONES OUT-OF-FOLD")
    print("=" * 125)
    print()
    imprimir_tabla_oof_backtest(filas)

    for error in errores:
        print(
            f"{error['activo']:<10} "
            f"{'ERROR':<10} "
            f"{'ERROR':<10} "
            f"{'ERROR':<13} "
            f"{'ERROR':>5} "
            f"{'ERROR':>5} "
            f"{'ERROR':>7} "
            f"{'ERROR':>9} "
            f"{'ERROR':>8} "
            f"{'ERROR':>8} "
            f"{'ERROR':>8} "
            f"{'ERROR':>9} "
            f"{'ERROR':>9}"
        )

    print()
    print("=" * 125)
    print(f"Tiempo total: {time.perf_counter() - inicio_total:.2f} segundos")

    if errores:
        print()
        print("Errores:")
        for error in errores:
            print(f"{error['activo']}: {error['error']}")


def imprimir_tabla_robustez(summary):
    print(
        f"{'ACTIVO':<8} {'DIRECCION':<15} {'MODELO':<10} {'SISTEMA':<13} "
        f"{'THR':>5} {'TRADES':>7} {'PF_0':>8} {'PF_1':>8} {'PF_2':>8} "
        f"{'PF_5':>8} {'EXPECT_1':>10} {'RETURN_1':>10} {'MAX_DD_1':>10} "
        f"{'S1':>5} {'S2':>5} {'S5':>5} {'MUESTRA':<14}"
    )
    print("-" * 145)

    for _, fila in summary.iterrows():
        print(
            f"{fila['ACTIVO']:<8} "
            f"{formatear_direccion(fila['DIRECCION']):<15} "
            f"{fila['MODELO']:<10} "
            f"{fila['SISTEMA']:<13} "
            f"{formatear_metrica(fila['THRESHOLD']):>5} "
            f"{int(fila['TRADES']):>7} "
            f"{formatear_metrica(fila['PF_0']):>8} "
            f"{formatear_metrica(fila['PF_1']):>8} "
            f"{formatear_metrica(fila['PF_2']):>8} "
            f"{formatear_metrica(fila['PF_5']):>8} "
            f"{formatear_metrica(fila['EXPECT_1']):>10} "
            f"{formatear_metrica(fila['RETURN_1']):>10} "
            f"{formatear_metrica(fila['MAX_DD_1']):>10} "
            f"{str(bool(fila['SURVIVES_1_BPS'])):>5} "
            f"{str(bool(fila['SURVIVES_2_BPS'])):>5} "
            f"{str(bool(fila['SURVIVES_5_BPS'])):>5} "
            f"{fila['TAMANO_MUESTRA']:<14}"
        )


def ejecutar_robustez():
    from src.robustness import analizar_robustez

    resultado = analizar_robustez(RAIZ_PROYECTO, ACTIVOS.keys())
    summary = resultado["summary"]

    print("=" * 145)
    print("ANALISIS DE ROBUSTEZ - OOF BACKTEST")
    print("=" * 145)
    print("No usar resultados para trading real.")
    print(f"Archivo guardado: results/robust_summary.csv")
    print()

    if summary.empty:
        print("No hay configuraciones que cumplan los criterios.")
    else:
        imprimir_tabla_robustez(summary)

    print()
    print("=" * 145)
    print(f"Numero total de configuraciones analizadas: {resultado['total_configuraciones']}")
    print(f"Numero que sobreviven 1 bps: {resultado['survives_1']}")
    print(f"Numero que sobreviven 2 bps: {resultado['survives_2']}")
    print(f"Numero que sobreviven 5 bps: {resultado['survives_5']}")

    if resultado["archivos_faltantes"]:
        print()
        print("Archivos faltantes:")
        for ruta in resultado["archivos_faltantes"]:
            print(ruta)


def imprimir_tabla_estabilidad(summary):
    print(
        f"{'ACTIVO':<8} {'DIRECCION':<15} {'MODELO':<10} {'SISTEMA':<13} "
        f"{'THR':>5} {'TRADES':>7} {'PF_2BPS':>8} {'EXPECT':>8} "
        f"{'P_POS':>6} {'N_PER':>6} {'%POS':>7} {'MEJOR_R':>9} "
        f"{'PEOR_R':>9} {'CONC':>8} {'MAX_LOSS':>8} {'CI_LOW':>8} "
        f"{'CI_HIGH':>8} {'DD95':>8}"
    )
    print("-" * 155)

    for _, fila in summary.iterrows():
        print(
            f"{fila['ACTIVO']:<8} "
            f"{formatear_direccion(fila['DIRECCION']):<15} "
            f"{fila['MODELO']:<10} "
            f"{fila['SISTEMA']:<13} "
            f"{formatear_metrica(fila['THRESHOLD']):>5} "
            f"{int(fila['TRADES']):>7} "
            f"{formatear_metrica(fila['PF_2BPS']):>8} "
            f"{formatear_metrica(fila['EXPECT_2BPS']):>8} "
            f"{int(fila['PERIODOS_POSITIVOS']):>6} "
            f"{int(fila['NUM_PERIODOS']):>6} "
            f"{formatear_metrica(fila['% PERIODOS POSITIVOS']):>7} "
            f"{formatear_metrica(fila['MEJOR PERIODO R']):>9} "
            f"{formatear_metrica(fila['PEOR PERIODO R']):>9} "
            f"{formatear_metrica(fila['PROFIT_CONCENTRATION']):>8} "
            f"{int(fila['MAX_LOSS_STREAK']):>8} "
            f"{formatear_metrica(fila['EXPECTANCY_CI_LOW']):>8} "
            f"{formatear_metrica(fila['EXPECTANCY_CI_HIGH']):>8} "
            f"{formatear_metrica(fila['BOOTSTRAP_DD_95']):>8}"
        )


def ejecutar_estabilidad():
    from src.stability import analizar_estabilidad

    resultado = analizar_estabilidad(RAIZ_PROYECTO)
    summary = resultado["summary"]

    print("=" * 155)
    print("ANALISIS DE ESTABILIDAD TEMPORAL - OOF")
    print("=" * 155)
    print("No usar resultados para trading real.")
    print("No se entrenaron modelos ni se recalcularon backtests.")
    print("Archivo guardado: results/stability_summary.csv")
    print("Detalle guardado: results/stability_by_period.csv")
    print()

    if summary.empty:
        print("No hay estrategias para analizar con los criterios solicitados.")
    else:
        imprimir_tabla_estabilidad(summary)

    print()
    print("=" * 155)
    print(f"Estrategias analizadas: {len(summary)}")

    if resultado["archivos_faltantes"]:
        print()
        print("Archivos faltantes:")
        for ruta in resultado["archivos_faltantes"]:
            print(ruta)


def ejecutar_paper():
    from src.paper_trader import ejecutar_paper as ejecutar_paper_forward
    from src.paper_trader import paper_status

    resultado = ejecutar_paper_forward(RAIZ_PROYECTO)
    status = paper_status(RAIZ_PROYECTO)

    print("=" * 48)
    print("FOREX ML - PAPER TRADING")
    print("=" * 48)
    print(f"Fecha: {formatear_fecha(resultado['start_time'])}")
    print("No se ejecuta ninguna orden real.")
    print()

    for item in resultado["resultados"]:
        config = item["config"]
        fila = item["ultima_fila"]
        abierta = next(
            (
                datos["trades_abiertos"]
                for datos in status["configs"]
                if datos["config"]["CONFIG_ID"] == config["CONFIG_ID"]
            ),
            0,
        )
        metricas = next(
            (
                datos
                for datos in status["configs"]
                if datos["config"]["CONFIG_ID"] == config["CONFIG_ID"]
            ),
            None,
        )

        print(config["ACTIVO"])
        print()
        print(f"CONFIG_ID: {config['CONFIG_ID']}")
        print(f"Configuracion: {config['MODELO']} + {config['SISTEMA']}")
        print(f"Threshold: {formatear_numero(config['THRESHOLD'])}")
        print(f"Inicio forward test: {formatear_fecha(item['inicio_forward'])}")
        print(f"Ultima vela disponible: {formatear_fecha(item['ultima_vela_disponible'])}")
        print(f"Ultima vela cerrada: {formatear_fecha(item['ultima_vela'])}")
        print(f"Ultima vela procesada: {formatear_fecha(item['ultima_vela_procesada'])}")
        print(f"Hay vela nueva: {'SI' if item['hay_vela_nueva'] else 'NO'}")
        print(f"Warm-up disponible: {item['warmup_disponible']} velas")
        print(
            "Barras elegibles totales desde forward start: "
            f"{item.get('barras_elegibles_totales', 0)}"
        )
        print(f"Barras ya procesadas: {item.get('barras_ya_procesadas', 0)}")
        print(f"Velas pendientes encontradas: {item.get('velas_pendientes', 0)}")
        print(
            "Velas procesadas en catch-up: "
            f"{item.get('velas_procesadas_catchup', 0)}"
        )
        timestamps_catchup = item.get("timestamps_procesados_catchup", [])
        if timestamps_catchup:
            print("Timestamps procesados:")
            for timestamp in timestamps_catchup:
                print(timestamp)
        print(
            "Trades abiertos durante catch-up: "
            f"{item.get('trades_abiertos_catchup', 0)}"
        )
        print(
            "Trades cerrados durante catch-up: "
            f"{item.get('trades_cerrados_catchup', 0)}"
        )
        print(
            "Ultima vela procesada catch-up: "
            f"{formatear_fecha(item.get('ultima_vela_procesada_catchup'))}"
        )
        print(f"Barras pendientes finales: {item.get('barras_pendientes_finales', 0)}")
        print(f"Modelo cargado: {'SI' if item['modelo_cargado'] else 'NO'}")
        if item["modelo_error"]:
            print(f"Causa modelo: {item['modelo_error']}")
        if item["estado_datos"]:
            print(f"Estado mercado/datos: {item['estado_datos']}")
        if item["proxima_accion"]:
            print(f"Proxima accion: {item['proxima_accion']}")

        contexto = item.get("contexto_fila")
        sufijo_contexto = "" if item["hay_vela_nueva"] else " contexto"
        if contexto is not None:
            print(f"Precio{sufijo_contexto}: {formatear_numero(float(contexto['Close']))}")
            print(f"RSI{sufijo_contexto}: {formatear_numero(float(contexto['RSI']))}")
            print(f"ATR{sufijo_contexto}: {formatear_numero(float(contexto['ATR']))}")
            print(f"EMA20{sufijo_contexto}: {formatear_numero(float(contexto['EMA20']))}")
            print(f"EMA50{sufijo_contexto}: {formatear_numero(float(contexto['EMA50']))}")
            print(f"EMA200{sufijo_contexto}: {formatear_numero(float(contexto['EMA200']))}")
        else:
            print("Precio contexto: N/A")
            print("RSI contexto: N/A")
            print("ATR contexto: N/A")
            print("EMA20 contexto: N/A")
            print("EMA50 contexto: N/A")
            print("EMA200 contexto: N/A")

        print(f"Probabilidad LONG (COMPRAR): {formatear_metrica(item['probabilidad'])}")
        print(f"Condicion tecnica: {str(bool(item['condicion_tecnica'])).upper()}")
        print(f"SENAL PAPER: {formatear_texto_direcciones(item['signal'])}")
        print(f"Posicion abierta: {'SI' if abierta else 'NO'}")
        if metricas:
            print(f"Trades paper acumulados: {metricas['trades_cerrados']}")
            print(f"Capital ficticio: {formatear_numero(metricas['capital'])}")
        if item["error"]:
            print(f"Estado: {item['error']}")
        print("-" * 48)

    print("No usar para trading real.")
    print("=" * 48)


def ejecutar_paper_status():
    from src.paper_trader import paper_status

    status = paper_status(RAIZ_PROYECTO)

    print("=" * 80)
    print("FOREX ML - PAPER STATUS")
    print("=" * 80)

    if not status["iniciado"]:
        print("Paper trading aun no iniciado. Ejecuta: python main.py PAPER")
        return

    print(f"Fecha inicio: {formatear_fecha(status['fecha_inicio'])}")
    print(f"Dias de forward test: {formatear_numero(status['dias_forward'])}")
    print()

    for datos in status["configs"]:
        config = datos["config"]
        print(f"CONFIG_ID: {config['CONFIG_ID']}")
        print(f"Direccion: {formatear_direccion(config['DIRECCION'])}")
        print(f"Trades cerrados: {datos['trades_cerrados']}")
        print(f"Trades abiertos: {datos['trades_abiertos']}")
        print(f"Wins: {datos['wins']}")
        print(f"Losses: {datos['losses']}")
        print(f"Win rate: {formatear_numero(datos['win_rate'])}%")
        print(f"Profit factor: {formatear_metrica(datos['profit_factor'])}")
        print(f"Expectancy: {formatear_metrica(datos['expectancy'])}")
        print(f"R acumulado: {formatear_metrica(datos['r_acumulado'])}")
        print(f"Capital ficticio: {formatear_numero(datos['capital'])}")
        print(f"Max drawdown: {formatear_numero(datos['max_drawdown'])}%")
        print("-" * 80)

    print("No usar para trading real.")


def _fecha_a_menor_b(left, right):
    import pandas as pd

    if not left or not right:
        return False
    return pd.to_datetime(left, utc=True) < pd.to_datetime(right, utc=True)


def ejecutar_paper_eurusd():
    from src.eurusd_forward import eurusd_forward_status, process_eurusd_forward

    resultado = process_eurusd_forward(RAIZ_PROYECTO)
    status = eurusd_forward_status(RAIZ_PROYECTO)

    print("FOREX ML - PAPER EURUSD FORWARD")
    print()
    print(f"Modelo congelado: {status['model_path']}")
    print(f"Hash: {status['model_hash']}")
    print(f"Training end: {status['training_end']}")
    print(f"Forward start: {status['forward_start']}")
    print(
        "Training end < forward start: "
        f"{'SI' if _fecha_a_menor_b(status['training_end'], status['forward_start']) else 'NO'}"
    )
    print()
    print(f"DATA SOURCE: {status['data_source']}")
    print(f"DATA STATUS: {status['data_status']}")
    print(f"Hora UTC actual: {status['utc_now']}")
    print(f"Forward start original: {status['forward_start_original']}")
    print(f"First full forward bar: {status['first_full_forward_bar']}")
    print(f"Ultima vela MT5 disponible: {status['last_mt5_bar'] or 'N/A'}")
    print(f"Ultima vela cerrada: {status['last_closed_bar'] or 'N/A'}")
    print(f"Offset servidor MT5 vs UTC: {status.get('server_utc_offset_hours', 'N/A')}")
    if status.get("data_error"):
        print(f"Data error: {status['data_error']}")
    print()
    print(f"Velas procesadas: {resultado['processed']}")
    print(f"Barras elegibles totales desde forward start: {resultado.get('eligible_bars_total', status.get('eligible_bars_total', 0))}")
    print(f"Barras ya procesadas: {resultado.get('processed_bars_before', status.get('processed_bars_total', 0))}")
    print(f"Barras pendientes encontradas: {resultado.get('pending_found', 0)}")
    if resultado.get("processed_bar_timestamps"):
        print("Timestamps procesados:")
        for timestamp in resultado["processed_bar_timestamps"]:
            print(timestamp)
    print(f"Barras pendientes: {status.get('pending_bars', 0)}")
    print(f"Operaciones abiertas: {resultado['opened']}")
    print(f"Operaciones cerradas: {resultado['closed']}")
    print(f"Skip riesgo: {resultado['skipped_risk']}")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_paper_eurusd_status():
    from src.eurusd_forward import eurusd_forward_status

    status = eurusd_forward_status(RAIZ_PROYECTO)
    metrics = status["metrics"]
    broker = status["broker"]

    print("FOREX ML - PAPER EURUSD STATUS")
    print()
    print(f"Activo: {status['asset']}")
    print(f"Direccion: {status['direction']}")
    print(f"Modelo: {status['model']}")
    print(f"Threshold: {status['threshold']}")
    print(f"Fecha inicio forward: {status['forward_start']}")
    print(f"Data source: {status['data_source']}")
    print(f"Data status: {status['data_status']}")
    print(f"Hora UTC actual: {status['utc_now']}")
    print(f"Forward start original: {status['forward_start_original']}")
    print(f"First full forward bar: {status['first_full_forward_bar']}")
    print(f"Ultima vela MT5 disponible: {status['last_mt5_bar'] or 'N/A'}")
    print(f"Ultima vela cerrada: {status['last_closed_bar'] or 'N/A'}")
    print(f"Offset servidor MT5 vs UTC: {status.get('server_utc_offset_hours', 'N/A')}")
    print(f"Ultima vela procesada: {status['last_processed_bar'] or 'N/A'}")
    print(f"Barras elegibles totales desde forward start: {status.get('eligible_bars_total', 0)}")
    print(f"Barras ya procesadas: {status.get('processed_bars_total', 0)}")
    print(f"Barras pendientes: {status['pending_bars']}")
    if status.get("pending_bar_timestamps"):
        print("Timestamps pendientes:")
        for timestamp in status["pending_bar_timestamps"]:
            print(timestamp)
    print(f"Probabilidad LONG: {formatear_numero(status['probability_long'])}")
    print(f"Senal: {status['signal']}")
    print(f"Estado ejecucion: {status['execution']}")
    print(f"Motivo: {status['reason']}")
    print()
    print(f"Trades: {metrics['trades']}")
    print(f"Wins: {metrics['wins']}")
    print(f"Losses: {metrics['losses']}")
    print(f"Win Rate: {formatear_numero(metrics['win_rate'])}%")
    print(f"Profit Factor: {formatear_numero(metrics['profit_factor'])}")
    print(f"Expectancy R: {formatear_numero(metrics['expectancy_r'])}")
    print(f"R total: {formatear_numero(metrics['r_total'])}")
    print(f"Max Drawdown: {formatear_numero(metrics['max_drawdown'])}%")
    print(f"Capital ficticio: {formatear_numero(metrics['capital'])}")
    if status["sample_insufficient"]:
        print("MUESTRA FORWARD INSUFICIENTE")
    print()
    print("Broker-aware USD 100:")
    print(f"Volume min: {formatear_numero(broker['volume_min'])}")
    print(f"Volume step: {formatear_numero(broker['volume_step'])}")
    print(f"Contract size: {formatear_numero(broker['contract_size'])}")
    print(f"Tick size: {formatear_numero(broker['tick_size'])}")
    print(f"Tick value: {formatear_numero(broker['tick_value'])}")
    print(f"Sample volume: {formatear_numero(broker['sample_volume'])}")
    print(f"Sample decision: {broker['sample_decision']}")
    print(f"Sample reason: {broker['sample_reason']}")
    if broker.get("metadata_error"):
        print(f"Broker metadata aviso: {broker['metadata_error']}")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_telegram_test():
    from src.notifier import enviar_telegram

    mensaje = "\n".join(
        [
            "FOREX ML",
            "",
            "Telegram conectado correctamente.",
            "",
            "Modo: PAPER TRADING",
            "Operaciones reales: NO",
        ]
    )
    enviado = enviar_telegram(mensaje)

    if enviado:
        print("Telegram conectado correctamente.")


def ejecutar_telegram_demo():
    from src.notifier import enviar_telegram

    mensajes = [
        "\n".join(
            [
                "FOREX ML - NUEVA SE\u00d1AL PAPER",
                "",
                "Activo: GOLD",
                "Direccion: LONG (COMPRAR)",
                "Modelo: LOGISTIC",
                "Sistema: BASE_PLUS_ML",
                "Threshold: 0.60",
                "Probabilidad: 64.3%",
                "",
                "Precio: 4325.20",
                "RSI: 57.40",
                "ATR: 16.25",
                "EMA20: 4321.00",
                "EMA50: 4315.00",
                "EMA200: 4280.00",
                "",
                "Modo: PAPER",
                "PRUEBA DE NOTIFICACION",
                "",
                "No se ejecuto ninguna orden real.",
            ]
        ),
        "\n".join(
            [
                "FOREX ML - PAPER TRADE ABIERTO",
                "",
                "Activo: GOLD",
                "Direccion: LONG (COMPRAR)",
                "",
                "Entry: 4325.20",
                "Stop Loss: 4308.95",
                "Take Profit: 4357.70",
                "",
                "Probabilidad: 64.3%",
                "ATR: 16.25",
                "Riesgo ficticio: 1%",
                "Capital ficticio: 1000.00",
                "",
                "PRUEBA DE NOTIFICACION",
                "",
                "No se envio ninguna orden real.",
            ]
        ),
        "\n".join(
            [
                "FOREX ML - PAPER TRADE CERRADO",
                "",
                "Activo: GOLD",
                "Resultado: WIN",
                "",
                "Entry: 4325.20",
                "Exit: 4357.70",
                "",
                "R bruto: +2.00R",
                "R neto: +1.95R",
                "",
                "Capital antes: 1000.00",
                "Capital despues: 1019.50",
                "",
                "PRUEBA DE NOTIFICACION.",
            ]
        ),
    ]

    resultados = []
    for indice, mensaje in enumerate(mensajes, start=1):
        resultados.append((indice, enviar_telegram(mensaje)))

    fallidos = [indice for indice, enviado in resultados if not enviado]
    if fallidos:
        print("Telegram DEMO no se envio completo.")
        for indice in fallidos:
            print(f"Mensaje {indice}: ERROR")
        print("No se modifico el estado PAPER.")
        return

    print("Telegram DEMO enviado correctamente.")
    print("3 mensajes de prueba enviados.")
    print("No se modifico el estado PAPER.")


def ejecutar_telegram_daily():
    from src.telegram_events import send_daily_report

    resultado = send_daily_report(RAIZ_PROYECTO)
    print(resultado["message"])
    print()
    if resultado["duplicate"]:
        print("Resumen diario ya notificado previamente. No se reenvio.")
    elif resultado["sent"]:
        print("Resumen diario enviado por Telegram.")
    else:
        print("Resumen diario generado, pero Telegram no se pudo enviar.")
        print("PAPER no se detiene por fallo de Telegram.")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_telegram_status():
    from src.telegram_events import telegram_status

    status = telegram_status(RAIZ_PROYECTO)
    print("FOREX ML - TELEGRAM STATUS")
    print()
    print(f"Telegram configurado: {'SI' if status['configured'] else 'NO'}")
    print(f"Ultimo evento enviado: {status['last_event_id'] or 'N/A'}")
    print(f"Tipo ultimo evento: {status['last_event_type'] or 'N/A'}")
    print(f"Hora ultimo evento: {status['last_event_at'] or 'N/A'}")
    print(f"Eventos enviados hoy: {status['events_today']}")
    print(f"Ultimo resumen diario: {status['last_daily'] or 'N/A'}")
    print(f"Registro eventos: {status['event_log']}")
    print()
    print("Secretos mostrados: NO")
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_health():
    from src.health import ejecutar_health_check

    resultado = ejecutar_health_check(RAIZ_PROYECTO)
    checks = {check["nombre"]: check for check in resultado["checks"]}

    def estado(nombre):
        return "OK" if checks[nombre]["ok"] else "ERROR"

    print("FOREX ML - HEALTH CHECK")
    print()
    print(f"Python: {estado('Python')}")
    print(f"Modelo PAPER: {estado('Modelo PAPER')}")
    print(f"Metadata modelo: {estado('Metadata modelo')}")
    print(f"Paper start: {estado('Paper start')}")
    print(f"Paper state: {estado('Paper state')}")
    print(f"Logs: {estado('Logs')}")
    print(f"Requirements: {estado('Requirements')}")
    print(f"Telegram: {estado('Telegram')}")
    print(f"Datos: {estado('Datos')}")
    print()
    print(f"SYSTEM READY: {'YES' if resultado['system_ready'] else 'NO'}")

    fallos = [check for check in resultado["checks"] if not check["ok"]]
    if fallos:
        print()
        print("Detalles:")
        for check in fallos:
            print(f"- {check['nombre']}: {check['detalle']}")


def ejecutar_gold_data_check():
    from src.gold_data_check import run_gold_data_check

    resultado = run_gold_data_check(RAIZ_PROYECTO)

    print("FOREX ML - GOLD DATA CHECK")
    print()
    print(f"Directorio trabajo: {resultado['cwd']}")
    print(f"Proyecto: {resultado['project_root']}")
    print(f"Timestamp UTC: {resultado['timestamp_utc']}")
    print(f"Ticker GOLD: {resultado['ticker']}")
    print(f"YFinance cache: {resultado['cache_dir']}")
    print()
    print("Directorios:")
    for nombre, check in resultado["checks"].items():
        estado = "OK" if check["exists"] and check["readable"] and check["writable"] else "ERROR"
        print(f"- {nombre}: {estado}")
        print(f"  ruta: {check['path']}")
        print(f"  existe: {_estado_bool(check['exists'])} lectura: {_estado_bool(check['readable'])} escritura: {_estado_bool(check['writable'])}")
        if check["error"]:
            print(f"  error: {check['error']}")
    print()
    print(f"Filas descargadas: {resultado['rows']}")
    print(f"Ultima vela disponible: {resultado['last_bar'] or 'N/A'}")
    print(f"Ultima vela cerrada: {resultado['last_closed_bar'] or 'N/A'}")
    print(f"DATA STATUS: {resultado['data_status']}")
    if resultado["error"]:
        print(f"ERROR: {resultado['error']}")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_mt5_check():
    from src.mt5_connector import verificar_mt5_demo_readonly

    resultado = verificar_mt5_demo_readonly()

    print("FOREX ML - MT5 CHECK")
    print()

    if resultado.get("cuenta_real_bloqueada"):
        print("CUENTA REAL DETECTADA - CONEXION BLOQUEADA")
        print()
        print("TRADING ENABLED: NO")
        print("ORDENES ENVIADAS: 0")
        print("MT5 DEMO READY: NO")
        return

    print(f"Conexion terminal: {'OK' if resultado['conexion_terminal'] else 'ERROR'}")
    print(f"Cuenta: {resultado['cuenta']}")
    print(f"Servidor: {'OK' if resultado['servidor_ok'] else 'ERROR'}")
    print(f"Saldo: {formatear_numero(resultado['saldo'])}")
    print(f"Equity: {formatear_numero(resultado['equity'])}")

    if resultado["error"]:
        print()
        print(f"Estado: {resultado['error']}")

    print()
    print("Instrumentos GOLD encontrados:")
    print()
    if resultado["instrumentos"]:
        for instrumento in resultado["instrumentos"]:
            print(instrumento["symbol"])
            if instrumento["description"]:
                print(f"Descripcion: {instrumento['description']}")
            print(f"Bid: {formatear_numero(instrumento['bid'])}")
            print(f"Ask: {formatear_numero(instrumento['ask'])}")
            print(f"Point: {formatear_numero(instrumento['point'])}")
            print(f"Digits: {instrumento['digits']}")
            print(
                "Contract size: "
                f"{formatear_numero(instrumento['trade_contract_size'])}"
            )
            print(f"Volume min: {formatear_numero(instrumento['volume_min'])}")
            print(f"Volume max: {formatear_numero(instrumento['volume_max'])}")
            print(f"Volume step: {formatear_numero(instrumento['volume_step'])}")
            print("-" * 48)
    else:
        print("No se encontraron coincidencias XAUUSD/GOLD.")

    print()
    print("TRADING ENABLED: NO")
    print(f"ORDENES ENVIADAS: {resultado['ordenes_enviadas']}")
    print(f"MT5 DEMO READY: {'YES' if resultado['mt5_demo_ready'] else 'NO'}")


def ejecutar_mt5_dryrun():
    from src.mt5_connector import ejecutar_mt5_dryrun_gold

    resultado = ejecutar_mt5_dryrun_gold()

    print("FOREX ML - MT5 DRY RUN")
    print()

    if resultado.get("cuenta_real_bloqueada"):
        print("CUENTA REAL DETECTADA - CONEXION BLOQUEADA")
        print()
        print("TRADING ENABLED: NO")
        print("ORDENES ENVIADAS: 0")
        return

    print(f"Cuenta: {resultado.get('cuenta', 'N/A')}")
    print(f"Broker: {resultado.get('broker', 'XM')}")
    print(f"Activo: {resultado.get('symbol', 'GOLD')}")
    print(f"Balance: {formatear_numero(resultado.get('balance'))}")
    print()

    if resultado.get("error") == "MERCADO CERRADO / SIN PRECIO DISPONIBLE":
        print("MERCADO CERRADO / SIN PRECIO DISPONIBLE")
        print()
        print("TRADING ENABLED: NO")
        print("ORDENES ENVIADAS: 0")
        return

    if "bid" in resultado:
        print(f"Bid: {formatear_numero(resultado.get('bid'))}")
    if "ask" in resultado:
        print(f"Ask: {formatear_numero(resultado.get('ask'))}")
    if "atr" in resultado:
        print(f"ATR: {formatear_numero(resultado.get('atr'))}")
        print(f"ATR fuente: {resultado.get('atr_fuente', 'N/A')}")
        if resultado.get("atr_fecha"):
            print(f"ATR fecha: {resultado['atr_fecha']}")

    if resultado.get("error") and "entry" not in resultado:
        print()
        print(f"Estado: {resultado['error']}")
        print()
        print("TRADING ENABLED: NO")
        print("ORDENES ENVIADAS: 0")
        return

    print()
    print(f"Direccion simulada: {formatear_direccion(resultado.get('direccion', 'LONG'))}")
    print()
    print(f"Entry: {formatear_numero(resultado.get('entry'))}")
    print(f"Stop Loss: {formatear_numero(resultado.get('stop_loss'))}")
    print(f"Take Profit: {formatear_numero(resultado.get('take_profit'))}")
    print()
    print(f"Riesgo objetivo: {formatear_numero(resultado.get('riesgo_objetivo_pct'))}%")
    print(f"Riesgo maximo USD: {formatear_numero(resultado.get('riesgo_maximo_usd'))}")
    print()
    print(f"Lote teorico: {formatear_numero(resultado.get('lote_teorico'))}")
    print(f"Lote minimo broker: {formatear_numero(resultado.get('lote_minimo'))}")
    print(f"Lote permitido: {formatear_numero(resultado.get('lote_permitido'))}")
    print()
    print(
        "Perdida estimada al SL: "
        f"{formatear_numero(resultado.get('perdida_estimada'))}"
    )
    print(
        "Riesgo porcentual real: "
        f"{formatear_numero(resultado.get('riesgo_pct_real'))}%"
    )

    if resultado.get("decision") == "SKIP - VOLUMEN MINIMO DEMASIADO GRANDE":
        print()
        print("OPERACION NO EJECUTABLE CON RIESGO DEL 1%")
        print(f"Lote requerido: {formatear_numero(resultado.get('lote_teorico'))}")
        print(f"Lote minimo broker: {formatear_numero(resultado.get('lote_minimo'))}")
        print(
            "Riesgo con lote minimo: $"
            f"{formatear_numero(resultado.get('perdida_minima'))}"
        )
        print(
            "Riesgo porcentual con lote minimo: "
            f"{formatear_numero(resultado.get('riesgo_pct_minimo'))}%"
        )

    print()
    print("Decision:")
    print(resultado.get("decision", "SKIP"))
    print()
    print("TRADING ENABLED: NO")
    print(f"ORDENES ENVIADAS: {resultado.get('ordenes_enviadas', 0)}")


def ejecutar_mt5_dryrun_all():
    from src.mt5_connector import ejecutar_mt5_dryrun_multi

    resultado = ejecutar_mt5_dryrun_multi(RAIZ_PROYECTO)

    print("FOREX ML - MT5 DRY RUN MULTI-ASSET")
    print()
    print(f"Cuenta: {resultado.get('cuenta', 'N/A')}")
    print(f"Broker: {resultado.get('broker', 'XM')}")
    print(f"Balance: {formatear_numero(resultado.get('balance'))}")
    print()

    if resultado.get("error") and not resultado.get("resultados"):
        print(f"Estado: {resultado['error']}")
        print()
        print("TRADING ENABLED: NO")
        print("ORDENES ENVIADAS: 0")
        return

    for item in resultado.get("resultados", []):
        print("-" * 48)
        print(f"Activo logico: {item.get('activo_logico')}")
        print(f"Simbolo MT5 resuelto: {item.get('simbolo_mt5') or 'N/A'}")
        print(f"Descripcion: {item.get('description') or 'N/A'}")
        print(f"Mercado: {'ABIERTO' if item.get('mercado_abierto') else 'CERRADO'}")
        print(f"Bid: {formatear_numero(item.get('bid'))}")
        print(f"Ask: {formatear_numero(item.get('ask'))}")
        print(f"Spread: {formatear_numero(item.get('spread'))}")
        print(f"ATR: {formatear_numero(item.get('ATR'))}")
        print(f"Modelo validado: {'SI' if item.get('modelo_validado') else 'NO'}")
        print(f"Senal: {item.get('signal', 'ESPERAR')}")
        print(f"Confianza: {formatear_numero(item.get('confidence'))}")
        print(f"SL: {formatear_numero(item.get('SL'))}")
        print(f"TP: {formatear_numero(item.get('TP'))}")
        print(f"Riesgo objetivo: {formatear_numero(item.get('risk_pct'))}%")
        print(f"Lote teorico: {formatear_numero(item.get('volume_theoretical'))}")
        print(f"Lote minimo broker: {formatear_numero(item.get('volume_min'))}")
        print(f"Lote valido: {formatear_numero(item.get('volume_allowed'))}")
        print(
            "Perdida estimada al SL: "
            f"{formatear_numero(item.get('estimated_loss'))}"
        )
        print(
            "Riesgo porcentual real: "
            f"{formatear_numero(item.get('risk_pct_real'))}%"
        )
        print(f"DECISION: {item.get('decision', 'ESPERAR')}")
        print(f"Motivo: {item.get('reason', 'N/A')}")

    if resultado.get("log_path"):
        print()
        print(f"Log V2: {resultado['log_path']}")

    print()
    print("TRADING ENABLED: NO")
    print(f"ORDENES ENVIADAS: {resultado.get('ordenes_enviadas', 0)}")


def _estado_bool(valor):
    return "SI" if valor else "NO"


def ejecutar_mt5_preview_all():
    from src.mt5_connector import ejecutar_mt5_preview_all as preview_all

    resultado = preview_all(RAIZ_PROYECTO)

    print("FOREX ML - MT5 PREVIEW ALL")
    print()
    print(f"Cuenta: {resultado.get('cuenta', 'N/A')}")
    print(f"Broker: {resultado.get('broker', 'XM')}")
    print(f"Balance: {formatear_numero(resultado.get('balance'))}")
    print()

    if resultado.get("error") and not resultado.get("resultados"):
        print(f"Estado: {resultado['error']}")
        print()
        print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
        print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
        print("ORDENES ENVIADAS: 0")
        return

    for item in resultado.get("resultados", []):
        print("-" * 56)
        print(f"Activo: {item.get('activo_logico')}")
        print(f"Simbolo MT5: {item.get('simbolo_mt5') or 'N/A'}")
        print(f"Senal: {item.get('signal', 'ESPERAR')}")
        print(f"Confianza: {formatear_numero(item.get('confidence'))}")
        print(f"Entry: {formatear_numero(item.get('entry'))}")
        print(f"SL: {formatear_numero(item.get('SL'))}")
        print(f"TP: {formatear_numero(item.get('TP'))}")
        print(f"Riesgo %: {formatear_numero(item.get('risk_pct'))}%")
        print(f"Riesgo USD: {formatear_numero(item.get('risk_amount'))}")
        print(f"Lote calculado: {formatear_numero(item.get('volume_allowed'))}")
        print(f"Lote minimo: {formatear_numero(item.get('volume_min'))}")
        print(f"Spread: {formatear_numero(item.get('spread'))}")
        print(f"Modelo LONG disponible: {_estado_bool(item.get('modelo_long_disponible'))}")
        print(f"Modelo SHORT disponible: {_estado_bool(item.get('modelo_short_disponible'))}")
        print(f"Modelo validado: {_estado_bool(item.get('modelo_validado'))}")
        print(f"Decision: {item.get('decision', 'ESPERAR')}")
        print(f"Motivo: {item.get('reason', 'N/A')}")
        faltante = item.get("validacion_faltante")
        if faltante:
            print(f"Validacion faltante: {faltante}")

    if resultado.get("log_path"):
        print()
        print(f"Log preview: {resultado['log_path']}")

    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print(f"ORDENES ENVIADAS: {resultado.get('ordenes_enviadas', 0)}")


def ejecutar_mt5_demo_test(modo="PREVIEW"):
    from src.mt5_demo_test import run_mt5_demo_test

    resultado = run_mt5_demo_test(RAIZ_PROYECTO, mode=modo)

    print("FOREX ML - MT5 DEMO TEST")
    print()
    print(f"Modo: {modo}")
    print(f"Cuenta: {resultado.account}")
    print(f"Servidor: {resultado.server}")
    print(f"Activo: {resultado.symbol}")
    print(f"Bid: {formatear_numero(resultado.bid)}")
    print(f"Ask: {formatear_numero(resultado.ask)}")
    print(f"Volume min: {formatear_numero(resultado.volume_min)}")
    print(f"Volume step: {formatear_numero(resultado.volume_step)}")
    print(f"Volumen de prueba: {formatear_numero(resultado.volume)}")
    print(f"SL: {formatear_numero(resultado.sl)}")
    print(f"TP: {formatear_numero(resultado.tp)}")
    print(f"Riesgo estimado: {formatear_numero(resultado.estimated_risk)}")
    print(f"order_check retcode: {resultado.order_check_retcode}")
    if resultado.order_check_comment:
        print(f"order_check comment: {resultado.order_check_comment}")
    print()
    print(f"Estado: {resultado.status}")
    print(f"Motivo: {resultado.reason}")
    if resultado.reason == "REAL ACCOUNT BLOCKED":
        print("REAL ACCOUNT BLOCKED")
    if resultado.order_retcode is not None:
        print(f"order_send retcode: {resultado.order_retcode}")
    if resultado.order_ticket is not None:
        print(f"Order ticket: {resultado.order_ticket}")
    if resultado.deal_ticket is not None:
        print(f"Deal ticket: {resultado.deal_ticket}")
    if resultado.requested_price is not None:
        print(f"Precio solicitado: {formatear_numero(resultado.requested_price)}")
    if resultado.executed_price is not None:
        print(f"Precio ejecutado: {formatear_numero(resultado.executed_price)}")
    if resultado.close_price is not None:
        print(f"Precio salida: {formatear_numero(resultado.close_price)}")
    if resultado.profit is not None:
        print(f"Profit/Loss: {formatear_numero(resultado.profit)}")
    if resultado.duration_seconds is not None:
        print(f"Duracion segundos: {resultado.duration_seconds}")
    print()
    print(f"TRADING ENABLED: {_estado_bool(resultado.trading_enabled)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(resultado.demo_execution_enabled)}")
    print(f"ORDENES ENVIADAS: {resultado.orders_sent}")


def ejecutar_mt5_demo_auto_preview():
    from src.mt5_demo_auto import AUTHORIZED_CONFIGS, run_preview

    resultados = run_preview(RAIZ_PROYECTO)

    print("FOREX ML - MT5 DEMO AUTO PREVIEW")
    print()
    print("Cuenta: DEMO")
    print("Estrategias autorizadas:")
    for config_id in AUTHORIZED_CONFIGS:
        print(f"- {config_id}")
    print()
    print(f"{'ACTIVO':<8}{'SIMBOLO':<10}{'CONFIGS':<62}{'EDAD':>8}  {'DECISION':<12}MOTIVO")
    ordenes = 0
    for item in resultados:
        configs = "|".join(item.config_ids)
        edad = formatear_numero(item.signal_age_minutes)
        decision = item.status
        print(f"{item.asset:<8}{(item.symbol or '-'):<10}{configs:<62}{edad:>8}  {decision:<12}{item.reason}")
        if item.price is not None:
            print(
                f"    entry {formatear_numero(item.price)} SL {formatear_numero(item.sl)} "
                f"TP {formatear_numero(item.tp)} lote {formatear_numero(item.volume)} "
                f"riesgo {formatear_numero(item.estimated_loss)}"
            )
        ordenes += item.orders_sent
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print(f"ORDENES ENVIADAS: {ordenes}")


def ejecutar_mt5_demo_auto_status():
    from src.mt5_demo_auto import run_status

    resultado = run_status(RAIZ_PROYECTO)

    print("FOREX ML - MT5 DEMO AUTO STATUS")
    print()
    print(f"DEMO execution enabled: {_estado_bool(resultado['demo_execution_enabled'])}")
    print(f"Cuenta: {resultado['account']}")
    print(f"Posiciones FOREX ML abiertas: {len(resultado['open_positions'])}")
    for posicion in resultado["open_positions"]:
        print(f"  #{getattr(posicion, 'ticket', 'N/A')} {getattr(posicion, 'symbol', 'N/A')} P/L {getattr(posicion, 'profit', 'N/A')}")
    print(f"Daily P/L demo: {formatear_numero(resultado['daily_pl'])}")
    print(f"Daily loss limit: {formatear_numero(resultado['daily_loss_limit'])}")
    last_signal = resultado.get("last_signal") or {}
    print(f"Ultima senal: {last_signal.get('asset', 'N/A')} {last_signal.get('config_id', 'N/A')} {last_signal.get('bar_timestamp', 'N/A')} {last_signal.get('signal', 'N/A')}/{last_signal.get('execution', 'N/A')}")
    print(f"Ultima orden enviada: {(resultado.get('last_order') or {}).get('timestamp', 'N/A')}")
    print(f"Ultimo cierre: {(resultado.get('last_close') or {}).get('timestamp', 'N/A')}")
    print(f"Ultimo error: {(resultado.get('last_error') or {}).get('reason', 'N/A')}")
    print(f"Ordenes enviadas hoy: {resultado['orders_sent_today']}")
    print()
    print(f"TRADING ENABLED: {_estado_bool(resultado['trading_enabled'])}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(resultado['demo_execution_enabled'])}")
    print("ORDENES ENVIADAS: 0")


def _avisar_fallo_runner(nombre, error):
    """Aviso de Telegram cuando un runner automatico falla (maximo uno por dia y tipo)."""
    from datetime import datetime, timezone

    from src.telegram_events import send_telegram_event

    dia = datetime.now(timezone.utc).strftime("%Y%m%d")
    send_telegram_event(
        RAIZ_PROYECTO,
        f"{nombre}_FAIL|{dia}|{type(error).__name__}",
        "MT5_DEMO",
        "\n".join([
            f"🚨 {nombre.replace('_', ' ')} - FALLO DEL RUNNER",
            f"Error: {str(error)[:300]}",
            "Revisa que MetaTrader 5 este abierto, logueado y con Algo Trading activo.",
            "DEMO / SIN DINERO REAL",
        ]),
        asset="ALL",
        config_id=nombre,
    )


def ejecutar_mt5_demo_auto_run():
    from src.mt5_demo_auto import run_auto

    try:
        resultados = run_auto(RAIZ_PROYECTO)
    except Exception as error:
        _avisar_fallo_runner("MT5_DEMO_AUTO", error)
        raise
    ordenes = sum(item.orders_sent for item in resultados)

    print("FOREX ML - MT5 DEMO AUTO RUN")
    print()
    for item in resultados:
        print(f"{item.asset} {item.symbol or '-'} {item.status}: {item.reason}")
        if item.ticket is not None:
            print(f"  ticket {item.ticket} deal {item.deal}")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print(f"ORDENES ENVIADAS: {ordenes}")


def ejecutar_mt5_demo_gold_preview():
    from src.mt5_demo_auto import gold_chain_preview

    resultado = gold_chain_preview(RAIZ_PROYECTO)
    signal = resultado.get("signal")
    broker = resultado.get("broker", {})

    print("FOREX ML - MT5 DEMO GOLD CHAIN PREVIEW")
    print()
    print(f"SIGNAL SOURCE: {resultado.get('signal_source', 'GC=F')}")
    print(f"EXECUTION SYMBOL: {resultado.get('execution_symbol', 'N/A')}")
    if signal is not None:
        print(f"signal timestamp: {signal.bar_timestamp}")
        print(f"configs: {'|'.join(resultado.get('config_ids', []))}")
        print(f"paper signal: {signal.signal}/{signal.execution}")
    print(f"broker Bid: {formatear_numero(resultado.get('bid'))}")
    print(f"broker Ask: {formatear_numero(resultado.get('ask'))}")
    print(f"ATR de estrategia: {formatear_numero(resultado.get('atr'))}")
    print(f"distancia SL: {formatear_numero(resultado.get('sl_distance'))}")
    print(f"distancia TP: {formatear_numero(resultado.get('tp_distance'))}")
    print(f"SL broker propuesto: {formatear_numero(resultado.get('sl'))}")
    print(f"TP broker propuesto: {formatear_numero(resultado.get('tp'))}")
    print(f"volumen teorico: {formatear_numero(resultado.get('theoretical_volume'))}")
    print(f"volumen final: {formatear_numero(resultado.get('final_volume'))}")
    print(f"riesgo USD: {formatear_numero(resultado.get('risk_usd'))}")
    print(f"decision: {resultado.get('decision', resultado.get('reason', 'N/A'))}")
    print()
    print("Broker metadata:")
    print(f"contract_size: {formatear_numero(broker.get('contract_size'))}")
    print(f"volume_min: {formatear_numero(broker.get('volume_min'))}")
    print(f"volume_step: {formatear_numero(broker.get('volume_step'))}")
    print(f"tick_size: {formatear_numero(broker.get('tick_size'))}")
    print(f"tick_value: {formatear_numero(broker.get('tick_value'))}")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print(f"ORDENES ENVIADAS: {resultado.get('orders_sent', 0)}")


def _imprimir_top_escenarios(titulo, datos, direccion=None, limite=6):
    print()
    print(titulo)
    if direccion is not None:
        datos = datos[datos["DIRECCION"] == direccion]
    if datos.empty:
        print("Sin resultados.")
        return

    columnas = [
        "DIRECCION",
        "MODELO",
        "SISTEMA",
        "THRESHOLD",
        "COST_BPS",
        "TRADES",
        "PROFIT_FACTOR",
        "EXPECTANCY_R",
        "R_TOTAL",
        "MAX_DRAWDOWN_PCT",
        "CLASIFICACION",
        "MOTIVO",
    ]
    disponibles = [columna for columna in columnas if columna in datos.columns]
    ordenado = datos.sort_values(
        ["COST_BPS", "EXPECTANCY_R", "PROFIT_FACTOR"],
        ascending=[True, False, False],
    )
    print(ordenado[disponibles].head(limite).to_string(index=False))


def ejecutar_research_eurusd():
    from src.eurusd_research import run_eurusd_research

    resultado = run_eurusd_research(RAIZ_PROYECTO)
    audit = resultado["audit"]
    walkforward = resultado["walkforward"]
    oof_summary = resultado["oof_summary"]
    robustness = resultado["robustness"]
    broker = resultado["broker_aware"]
    candidates = resultado["candidates"]
    discarded = resultado["discarded"]

    print("FOREX ML - RESEARCH EURUSD")
    print()
    print(f"Output: {resultado['output_dir']}")
    print("Ejecucion automatica: NO")
    print("Forward test iniciado: NO")
    print("Modelo validado declarado: NO")
    print()

    print("1. Modelos EURUSD encontrados")
    columnas_audit = [
        "DIRECCION",
        "EXISTE",
        "FECHA_ARCHIVO",
        "ALGORITMO",
        "N_FEATURES_MODELO",
        "COMPATIBLE_METODOLOGIA_ACTUAL",
        "NOTAS",
    ]
    print(audit[columnas_audit].to_string(index=False))

    print()
    print("2. LONG resultados")
    _imprimir_top_escenarios("LONG - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "LONG")

    print()
    print("3. SHORT resultados")
    _imprimir_top_escenarios("SHORT - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "SHORT")

    print()
    print("4. Walk-forward")
    wf_resumen = (
        walkforward.groupby(["DIRECCION", "MODELO"], dropna=False)
        .agg(
            FOLDS=("FOLD", "count"),
            AUC_MEDIO=("ROC_AUC", "mean"),
            F1_MEDIO=("F1", "mean"),
            PRECISION_MEDIA=("PRECISION", "mean"),
            RECALL_MEDIO=("RECALL", "mean"),
        )
        .reset_index()
    )
    print(wf_resumen.to_string(index=False))

    print()
    print("5. OOF")
    oof_2bps = oof_summary[oof_summary["COST_BPS"] == 2].copy()
    print(
        oof_2bps[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "PROFIT_FACTOR",
                "EXPECTANCY_R",
                "R_TOTAL",
                "MAX_DRAWDOWN_PCT",
            ]
        ]
        .sort_values(["DIRECCION", "EXPECTANCY_R"], ascending=[True, False])
        .head(12)
        .to_string(index=False)
    )

    print()
    print("6. Costes")
    costes = (
        oof_summary.groupby(["DIRECCION", "COST_BPS"])
        .agg(
            ESCENARIOS=("MODELO", "count"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            TRADES_TOTAL=("TRADES", "sum"),
        )
        .reset_index()
    )
    print(costes.to_string(index=False))

    print()
    print("7. Robustness")
    _imprimir_top_escenarios("Robustness principal 2 bps", robustness[robustness["COST_BPS"] == 2], None, limite=12)

    print()
    print("8. Broker-aware")
    if broker.empty:
        print("Sin filas broker-aware.")
    else:
        resumen_broker = (
            broker.groupby(["CAPITAL", "DECISION", "REASON"], dropna=False)
            .size()
            .reset_index(name="FILAS")
        )
        print(resumen_broker.to_string(index=False))

    print()
    print("9. Candidatos a forward test")
    if candidates.empty:
        print("Sin candidatos descriptivos a forward test.")
    else:
        print(
            candidates[
                [
                    "DIRECCION",
                    "MODELO",
                    "SISTEMA",
                    "THRESHOLD",
                    "TRADES",
                    "PROFIT_FACTOR",
                    "EXPECTANCY_R",
                    "FOLDS_POSITIVOS",
                    "PORCENTAJE_FOLDS_POSITIVOS",
                    "MOTIVO",
                ]
            ].to_string(index=False)
        )

    print()
    print("10. Descartados y motivos")
    print(
        discarded[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "CLASIFICACION",
                "MOTIVO",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()
    print("Archivos creados:")
    print("results/v2/eurusd/eurusd_walkforward_summary.csv")
    print("results/v2/eurusd/eurusd_oof_summary.csv")
    print("results/v2/eurusd/eurusd_robustness.csv")
    print("results/v2/eurusd/eurusd_trades.csv")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_research_gbpusd():
    from src.gbpusd_research import run_gbpusd_research

    resultado = run_gbpusd_research(RAIZ_PROYECTO)
    audit = resultado["audit"]
    walkforward = resultado["walkforward"]
    oof_summary = resultado["oof_summary"]
    robustness = resultado["robustness"]
    broker = resultado["broker_aware"]
    candidates = resultado["candidates"]
    discarded = resultado["discarded"]
    data_period = resultado["data_period"]

    print("FOREX ML - RESEARCH GBPUSD")
    print()
    print(f"Output: {resultado['output_dir']}")
    print("Ejecucion automatica: NO")
    print("Forward test iniciado: NO")
    print("Modelo validado declarado: NO")
    print()

    print("1. Modelos GBPUSD encontrados")
    columnas_audit = [
        "DIRECCION",
        "EXISTE",
        "FECHA_ARCHIVO",
        "ALGORITMO",
        "N_FEATURES_MODELO",
        "THRESHOLDS_DISPONIBLES",
        "COMPATIBLE_METODOLOGIA_ACTUAL",
        "LEAKAGE_LOOKAHEAD_CONOCIDO",
        "NOTAS",
    ]
    print(audit[columnas_audit].to_string(index=False))

    print()
    print("2. Periodo de datos")
    print(f"Inicio: {formatear_fecha(data_period['start'])}")
    print(f"Fin: {formatear_fecha(data_period['end'])}")
    print(f"Filas utiles: {data_period['rows']}")

    print()
    print("3. Resultados LONG")
    _imprimir_top_escenarios("LONG - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "LONG")

    print()
    print("4. Resultados SHORT")
    _imprimir_top_escenarios("SHORT - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "SHORT")

    print()
    print("5. Walk-forward")
    wf_2bps = walkforward[walkforward["COST_BPS"] == 2].copy()
    wf_resumen = (
        wf_2bps.groupby(["DIRECCION", "MODELO", "SISTEMA", "THRESHOLD"], dropna=False)
        .agg(
            FOLDS=("FOLD", "nunique"),
            TRADES=("TRADES", "sum"),
            AUC_MEDIO=("AUC", "mean"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            R_TOTAL=("R_TOTAL", "sum"),
            FOLDS_POSITIVOS=("FOLD_POSITIVO", "sum"),
        )
        .reset_index()
        .sort_values(["DIRECCION", "EXPECTANCY_MEDIA"], ascending=[True, False])
    )
    print(wf_resumen.head(18).to_string(index=False))

    print()
    print("6. OOF")
    oof_2bps = oof_summary[oof_summary["COST_BPS"] == 2].copy()
    print(
        oof_2bps[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "WIN_RATE",
                "PROFIT_FACTOR",
                "EXPECTANCY_R",
                "R_TOTAL",
                "MAX_DRAWDOWN_PCT",
            ]
        ]
        .sort_values(["DIRECCION", "EXPECTANCY_R"], ascending=[True, False])
        .head(18)
        .to_string(index=False)
    )

    print()
    print("7. Costes")
    costes = (
        oof_summary.groupby(["DIRECCION", "COST_BPS"])
        .agg(
            ESCENARIOS=("MODELO", "count"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            TRADES_TOTAL=("TRADES", "sum"),
        )
        .reset_index()
    )
    print(costes.to_string(index=False))

    print()
    print("8. Robustez")
    robust_2bps = robustness[robustness["COST_BPS"] == 2].copy()
    _imprimir_top_escenarios("Robustez principal 2 bps", robust_2bps, None, limite=18)
    cruza_cero = robust_2bps[
        (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_LOW"] <= 0)
        & (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_HIGH"] >= 0)
    ]
    if not cruza_cero.empty:
        print()
        print("Advertencia: en estas configuraciones el CI95 de expectancy cruza cero; ventaja NO confirmada.")

    print()
    print("9. Broker-aware XM")
    if broker.empty:
        print("Sin filas broker-aware.")
    else:
        resumen_broker = (
            broker.groupby(["CAPITAL", "DECISION", "REASON"], dropna=False)
            .size()
            .reset_index(name="FILAS")
        )
        totales = resumen_broker.groupby("CAPITAL")["FILAS"].transform("sum")
        resumen_broker["PORCENTAJE"] = (resumen_broker["FILAS"] / totales) * 100
        print(resumen_broker.to_string(index=False))

    print()
    print("10. Comportamiento con USD 100")
    if broker.empty:
        print("Sin metadata broker-aware.")
    else:
        broker_100 = broker[broker["CAPITAL"] == 100]
        total_100 = len(broker_100)
        ejecutables_100 = int((broker_100["DECISION"] == "EXECUTABLE").sum()) if total_100 else 0
        print(f"Operaciones potenciales evaluadas: {total_100}")
        print(f"Ejecutables: {ejecutables_100}")
        print(f"Skip: {total_100 - ejecutables_100}")
        print(f"Porcentaje ejecutable: {(ejecutables_100 / total_100 * 100) if total_100 else 0:.2f}%")

    print()
    print("11. Candidatos a forward test")
    if candidates.empty:
        print("Sin candidatos descriptivos a forward test.")
    else:
        print(
            candidates[
                [
                    "DIRECCION",
                    "MODELO",
                    "SISTEMA",
                    "THRESHOLD",
                    "TRADES",
                    "PROFIT_FACTOR",
                    "EXPECTANCY_R",
                    "FOLDS_POSITIVOS",
                    "PORCENTAJE_FOLDS_POSITIVOS",
                    "MOTIVO",
                ]
            ].to_string(index=False)
        )

    print()
    print("12. Configuraciones descartadas y motivo")
    print(
        discarded[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "CLASIFICACION",
                "MOTIVO",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("13. Limitaciones")
    print("No se congelo ningun modelo GBPUSD.")
    print("No se creo paper/gbpusd.")
    print("Los candidatos son descriptivos, no garantias de rentabilidad.")
    print("Forward test requiere autorizacion explicita posterior.")
    print()
    print("Archivos creados:")
    print("results/v2/gbpusd/gbpusd_existing_models_audit.csv")
    print("results/v2/gbpusd/gbpusd_walkforward_summary.csv")
    print("results/v2/gbpusd/gbpusd_oof_summary.csv")
    print("results/v2/gbpusd/gbpusd_oof_predictions.csv")
    print("results/v2/gbpusd/gbpusd_robustness.csv")
    print("results/v2/gbpusd/gbpusd_trades.csv")
    print("results/v2/gbpusd/gbpusd_broker_aware.csv")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_research_usdjpy():
    from src.usdjpy_research import run_usdjpy_research

    resultado = run_usdjpy_research(RAIZ_PROYECTO)
    audit = resultado["audit"]
    walkforward = resultado["walkforward"]
    oof_summary = resultado["oof_summary"]
    robustness = resultado["robustness"]
    broker = resultado["broker_aware"]
    candidates = resultado["candidates"]
    discarded = resultado["discarded"]
    data_period = resultado["data_period"]

    print("FOREX ML - RESEARCH USDJPY")
    print()
    print(f"Output: {resultado['output_dir']}")
    print("Ejecucion automatica: NO")
    print("Forward test iniciado: NO")
    print("Modelo validado declarado: NO")
    print()

    print("1. Modelos USDJPY encontrados")
    columnas_audit = [
        "DIRECCION",
        "EXISTE",
        "FECHA_ARCHIVO",
        "ALGORITMO",
        "N_FEATURES_MODELO",
        "THRESHOLDS_DISPONIBLES",
        "COMPATIBLE_METODOLOGIA_ACTUAL",
        "LEAKAGE_LOOKAHEAD_CONOCIDO",
        "NOTAS",
    ]
    print(audit[columnas_audit].to_string(index=False))

    print()
    print("2. Periodo de datos")
    print(f"Inicio: {formatear_fecha(data_period['start'])}")
    print(f"Fin: {formatear_fecha(data_period['end'])}")
    print(f"Filas utiles: {data_period['rows']}")

    print()
    print("3. Resultados LONG")
    _imprimir_top_escenarios("LONG - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "LONG")

    print()
    print("4. Resultados SHORT")
    _imprimir_top_escenarios("SHORT - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "SHORT")

    print()
    print("5. Walk-forward")
    wf_2bps = walkforward[walkforward["COST_BPS"] == 2].copy()
    wf_resumen = (
        wf_2bps.groupby(["DIRECCION", "MODELO", "SISTEMA", "THRESHOLD"], dropna=False)
        .agg(
            FOLDS=("FOLD", "nunique"),
            TRADES=("TRADES", "sum"),
            AUC_MEDIO=("AUC", "mean"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            R_TOTAL=("R_TOTAL", "sum"),
            FOLDS_POSITIVOS=("FOLD_POSITIVO", "sum"),
        )
        .reset_index()
        .sort_values(["DIRECCION", "EXPECTANCY_MEDIA"], ascending=[True, False])
    )
    print(wf_resumen.head(18).to_string(index=False))

    print()
    print("6. OOF")
    oof_2bps = oof_summary[oof_summary["COST_BPS"] == 2].copy()
    print(
        oof_2bps[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "WIN_RATE",
                "PROFIT_FACTOR",
                "EXPECTANCY_R",
                "R_TOTAL",
                "MAX_DRAWDOWN_PCT",
            ]
        ]
        .sort_values(["DIRECCION", "EXPECTANCY_R"], ascending=[True, False])
        .head(18)
        .to_string(index=False)
    )

    print()
    print("7. Costes")
    costes = (
        oof_summary.groupby(["DIRECCION", "COST_BPS"])
        .agg(
            ESCENARIOS=("MODELO", "count"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            TRADES_TOTAL=("TRADES", "sum"),
        )
        .reset_index()
    )
    print(costes.to_string(index=False))

    print()
    print("8. Robustez")
    robust_2bps = robustness[robustness["COST_BPS"] == 2].copy()
    _imprimir_top_escenarios("Robustez principal 2 bps", robust_2bps, None, limite=18)
    cruza_cero = robust_2bps[
        (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_LOW"] <= 0)
        & (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_HIGH"] >= 0)
    ]
    if not cruza_cero.empty:
        print()
        print("Advertencia: en estas configuraciones el CI95 de expectancy cruza cero; ventaja NO confirmada.")

    print()
    print("9. Broker-aware XM")
    if broker.empty:
        print("Sin filas broker-aware.")
    else:
        resumen_broker = (
            broker.groupby(["CAPITAL", "DECISION", "REASON"], dropna=False)
            .size()
            .reset_index(name="FILAS")
        )
        totales = resumen_broker.groupby("CAPITAL")["FILAS"].transform("sum")
        resumen_broker["PORCENTAJE"] = (resumen_broker["FILAS"] / totales) * 100
        print(resumen_broker.to_string(index=False))

    print()
    print("10. Comportamiento con USD 100")
    if broker.empty:
        print("Sin metadata broker-aware.")
    else:
        broker_100 = broker[broker["CAPITAL"] == 100]
        total_100 = len(broker_100)
        ejecutables_100 = int((broker_100["DECISION"] == "EXECUTABLE").sum()) if total_100 else 0
        print(f"Operaciones potenciales evaluadas: {total_100}")
        print(f"Ejecutables: {ejecutables_100}")
        print(f"Skip: {total_100 - ejecutables_100}")
        print(f"Porcentaje ejecutable: {(ejecutables_100 / total_100 * 100) if total_100 else 0:.2f}%")

    print()
    print("11. Candidatos a forward test")
    if candidates.empty:
        print("Sin candidatos descriptivos a forward test.")
    else:
        print(
            candidates[
                [
                    "DIRECCION",
                    "MODELO",
                    "SISTEMA",
                    "THRESHOLD",
                    "TRADES",
                    "PROFIT_FACTOR",
                    "EXPECTANCY_R",
                    "FOLDS_POSITIVOS",
                    "PORCENTAJE_FOLDS_POSITIVOS",
                    "MOTIVO",
                ]
            ].to_string(index=False)
        )

    print()
    print("12. Configuraciones descartadas y motivo")
    print(
        discarded[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "CLASIFICACION",
                "MOTIVO",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("13. Limitaciones")
    print("No se congelo ningun modelo USDJPY.")
    print("No se creo paper/usdjpy.")
    print("Los candidatos son descriptivos, no garantias de rentabilidad.")
    print("Forward test requiere autorizacion explicita posterior.")
    print()
    print("Archivos creados:")
    print("results/v2/usdjpy/usdjpy_existing_models_audit.csv")
    print("results/v2/usdjpy/usdjpy_walkforward_summary.csv")
    print("results/v2/usdjpy/usdjpy_oof_summary.csv")
    print("results/v2/usdjpy/usdjpy_oof_predictions.csv")
    print("results/v2/usdjpy/usdjpy_robustness.csv")
    print("results/v2/usdjpy/usdjpy_trades.csv")
    print("results/v2/usdjpy/usdjpy_broker_aware.csv")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_research_audusd():
    from src.audusd_research import run_audusd_research

    resultado = run_audusd_research(RAIZ_PROYECTO)
    audit = resultado["audit"]
    walkforward = resultado["walkforward"]
    oof_summary = resultado["oof_summary"]
    robustness = resultado["robustness"]
    broker = resultado["broker_aware"]
    sizing_validation = resultado["sizing_validation"]
    candidates = resultado["candidates"]
    discarded = resultado["discarded"]
    data_period = resultado["data_period"]

    print("FOREX ML - RESEARCH AUDUSD")
    print()
    print(f"Output: {resultado['output_dir']}")
    print("Ejecucion automatica: NO")
    print("Forward test iniciado: NO")
    print("Modelo validado declarado: NO")
    print()

    print("1. Periodo de datos")
    print(f"Inicio: {formatear_fecha(data_period['start'])}")
    print(f"Fin: {formatear_fecha(data_period['end'])}")
    print(f"Filas utiles: {data_period['rows']}")
    print(f"Fuente: {data_period['source']}")
    print(f"Intervalo: {data_period['interval']}")
    print(f"Periodo descargado: {data_period['downloaded_period']}")
    print(f"Dataset creado en esta ejecucion: {'SI' if data_period['created_now'] else 'NO'}")

    print()
    print("2. Modelos utilizados / existentes")
    columnas_audit = [
        "DIRECCION",
        "EXISTE",
        "FECHA_ARCHIVO",
        "ALGORITMO",
        "N_FEATURES_MODELO",
        "THRESHOLDS_DISPONIBLES",
        "COMPATIBLE_METODOLOGIA_ACTUAL",
        "LEAKAGE_LOOKAHEAD_CONOCIDO",
        "NOTAS",
    ]
    print(audit[columnas_audit].to_string(index=False))
    print("Modelos evaluados en investigacion: LOGISTIC, RF_SIMPLE, RF_ACTUAL")

    print()
    print("3. LONG resultados")
    _imprimir_top_escenarios("LONG - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "LONG")

    print()
    print("4. SHORT resultados")
    _imprimir_top_escenarios("SHORT - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "SHORT")

    print()
    print("5. Walk-forward")
    wf_2bps = walkforward[walkforward["COST_BPS"] == 2].copy()
    wf_resumen = (
        wf_2bps.groupby(["DIRECCION", "MODELO", "SISTEMA", "THRESHOLD"], dropna=False)
        .agg(
            FOLDS=("FOLD", "nunique"),
            TRADES=("TRADES", "sum"),
            AUC_MEDIO=("AUC", "mean"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            R_TOTAL=("R_TOTAL", "sum"),
            FOLDS_POSITIVOS=("FOLD_POSITIVO", "sum"),
        )
        .reset_index()
        .sort_values(["DIRECCION", "EXPECTANCY_MEDIA"], ascending=[True, False])
    )
    print(wf_resumen.head(18).to_string(index=False))

    print()
    print("6. OOF")
    oof_2bps = oof_summary[oof_summary["COST_BPS"] == 2].copy()
    print(
        oof_2bps[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "WIN_RATE",
                "PROFIT_FACTOR",
                "EXPECTANCY_R",
                "R_TOTAL",
                "MAX_DRAWDOWN_PCT",
            ]
        ]
        .sort_values(["DIRECCION", "EXPECTANCY_R"], ascending=[True, False])
        .head(18)
        .to_string(index=False)
    )

    print()
    print("7. Costes")
    costes = (
        oof_summary.groupby(["DIRECCION", "COST_BPS"])
        .agg(
            ESCENARIOS=("MODELO", "count"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            TRADES_TOTAL=("TRADES", "sum"),
        )
        .reset_index()
    )
    print(costes.to_string(index=False))

    print()
    print("8. Robustez")
    robust_2bps = robustness[robustness["COST_BPS"] == 2].copy()
    _imprimir_top_escenarios("Robustez principal 2 bps", robust_2bps, None, limite=18)
    cruza_cero = robust_2bps[
        (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_LOW"] <= 0)
        & (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_HIGH"] >= 0)
    ]
    if not cruza_cero.empty:
        print()
        print("Advertencia: en estas configuraciones el CI95 de expectancy cruza cero; VENTAJA NO CONFIRMADA.")

    print()
    print("9. Metadata XM")
    if broker.empty:
        print("Sin filas broker-aware.")
    else:
        metadata_cols = [
            "BROKER_METADATA_SOURCE",
            "SYMBOL",
            "CONTRACT_SIZE",
            "VOLUME_MIN",
            "VOLUME_STEP",
            "VOLUME_MAX",
            "TICK_SIZE",
            "TICK_VALUE",
            "TICK_VALUE_PROFIT",
            "TICK_VALUE_LOSS",
            "DIGITS",
            "POINT",
            "CURRENCY_BASE",
            "CURRENCY_PROFIT",
            "CURRENCY_MARGIN",
            "SPREAD",
        ]
        print(broker[metadata_cols].drop_duplicates().head(5).to_string(index=False))

    print()
    print("10. Risk Manager vs order_calc_profit")
    if sizing_validation.empty:
        print("Sin validacion sizing; MT5 no disponible o sin trades.")
    else:
        print(sizing_validation.to_string(index=False))

    for idx, capital in enumerate([100, 500, 1000, 5000], start=11):
        print()
        print(f"{idx}. Broker-aware USD {capital}")
        if broker.empty:
            print("Sin metadata broker-aware.")
            continue
        subset = broker[broker["CAPITAL"] == capital]
        total = len(subset)
        ejecutables = int((subset["DECISION"] == "EXECUTABLE").sum()) if total else 0
        skips_min = int((subset["REASON"] == "MINIMUM_VOLUME_EXCEEDS_RISK").sum()) if total else 0
        print(f"Total senales evaluadas: {total}")
        print(f"Ejecutables: {ejecutables}")
        print(f"SKIP por volumen minimo: {skips_min}")
        print(f"Skip total: {total - ejecutables}")
        print(f"Porcentaje ejecutable: {(ejecutables / total * 100) if total else 0:.2f}%")

    print()
    print("15. Candidatos a forward test")
    if candidates.empty:
        print("Sin candidatos descriptivos a forward test.")
    else:
        print(
            candidates[
                [
                    "DIRECCION",
                    "MODELO",
                    "SISTEMA",
                    "THRESHOLD",
                    "TRADES",
                    "PROFIT_FACTOR",
                    "EXPECTANCY_R",
                    "FOLDS_POSITIVOS",
                    "PORCENTAJE_FOLDS_POSITIVOS",
                    "MOTIVO",
                ]
            ].to_string(index=False)
        )

    print()
    print("16. Descartados y motivo")
    print(
        discarded[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "CLASIFICACION",
                "MOTIVO",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("17. Limitaciones")
    print("No se congelo ningun modelo AUDUSD.")
    print("No se creo paper/audusd.")
    print("Los candidatos son descriptivos, no garantias de rentabilidad.")
    print("Forward test requiere autorizacion explicita posterior.")
    print()
    print("Archivos creados:")
    print("results/v2/audusd/audusd_existing_models_audit.csv")
    print("results/v2/audusd/audusd_walkforward_summary.csv")
    print("results/v2/audusd/audusd_oof_summary.csv")
    print("results/v2/audusd/audusd_oof_predictions.csv")
    print("results/v2/audusd/audusd_robustness.csv")
    print("results/v2/audusd/audusd_trades.csv")
    print("results/v2/audusd/audusd_broker_aware.csv")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_research_usdcad():
    from src.usdcad_research import run_usdcad_research

    resultado = run_usdcad_research(RAIZ_PROYECTO)
    audit = resultado["audit"]
    walkforward = resultado["walkforward"]
    oof_summary = resultado["oof_summary"]
    robustness = resultado["robustness"]
    broker = resultado["broker_aware"]
    sizing_validation = resultado["sizing_validation"]
    candidates = resultado["candidates"]
    discarded = resultado["discarded"]
    data_period = resultado["data_period"]

    print("FOREX ML - RESEARCH USDCAD")
    print()
    print(f"Output: {resultado['output_dir']}")
    print("Ejecucion automatica: NO")
    print("Forward test iniciado: NO")
    print("Modelo validado declarado: NO")
    print()

    print("1. Periodo de datos")
    print(f"Inicio: {formatear_fecha(data_period['start'])}")
    print(f"Fin: {formatear_fecha(data_period['end'])}")
    print(f"Filas utiles: {data_period['rows']}")
    print(f"Fuente: {data_period['source']}")
    print(f"Intervalo: {data_period['interval']}")
    print(f"Periodo descargado: {data_period['downloaded_period']}")
    print(f"Dataset creado en esta ejecucion: {'SI' if data_period['created_now'] else 'NO'}")

    print()
    print("Modelos existentes / utilizados")
    columnas_audit = [
        "DIRECCION",
        "EXISTE",
        "FECHA_ARCHIVO",
        "ALGORITMO",
        "N_FEATURES_MODELO",
        "THRESHOLDS_DISPONIBLES",
        "COMPATIBLE_METODOLOGIA_ACTUAL",
        "LEAKAGE_LOOKAHEAD_CONOCIDO",
        "NOTAS",
    ]
    print(audit[columnas_audit].to_string(index=False))
    print("Modelos evaluados: LOGISTIC, RF_SIMPLE, RF_ACTUAL")

    print()
    print("2. LONG")
    _imprimir_top_escenarios("LONG - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "LONG")

    print()
    print("3. SHORT")
    _imprimir_top_escenarios("SHORT - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "SHORT")

    print()
    print("4. Walk-forward")
    wf_2bps = walkforward[walkforward["COST_BPS"] == 2].copy()
    wf_resumen = (
        wf_2bps.groupby(["DIRECCION", "MODELO", "SISTEMA", "THRESHOLD"], dropna=False)
        .agg(
            FOLDS=("FOLD", "nunique"),
            TRADES=("TRADES", "sum"),
            AUC_MEDIO=("AUC", "mean"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            R_TOTAL=("R_TOTAL", "sum"),
            FOLDS_POSITIVOS=("FOLD_POSITIVO", "sum"),
        )
        .reset_index()
        .sort_values(["DIRECCION", "EXPECTANCY_MEDIA"], ascending=[True, False])
    )
    print(wf_resumen.head(18).to_string(index=False))

    print()
    print("5. OOF")
    oof_2bps = oof_summary[oof_summary["COST_BPS"] == 2].copy()
    print(
        oof_2bps[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "WIN_RATE",
                "PROFIT_FACTOR",
                "EXPECTANCY_R",
                "R_TOTAL",
                "MAX_DRAWDOWN_PCT",
            ]
        ]
        .sort_values(["DIRECCION", "EXPECTANCY_R"], ascending=[True, False])
        .head(18)
        .to_string(index=False)
    )

    print()
    print("6. Costes")
    costes = (
        oof_summary.groupby(["DIRECCION", "COST_BPS"])
        .agg(
            ESCENARIOS=("MODELO", "count"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            TRADES_TOTAL=("TRADES", "sum"),
        )
        .reset_index()
    )
    print(costes.to_string(index=False))

    print()
    print("7. Robustez")
    robust_2bps = robustness[robustness["COST_BPS"] == 2].copy()
    _imprimir_top_escenarios("Robustez principal 2 bps", robust_2bps, None, limite=18)
    cruza_cero = robust_2bps[
        (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_LOW"] <= 0)
        & (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_HIGH"] >= 0)
    ]
    if not cruza_cero.empty:
        print()
        print("Advertencia: en estas configuraciones el CI95 de expectancy cruza cero; VENTAJA NO CONFIRMADA.")

    print()
    print("8. Metadata XM")
    if broker.empty:
        print("Sin filas broker-aware.")
    else:
        metadata_cols = [
            "BROKER_METADATA_SOURCE",
            "SYMBOL",
            "CONTRACT_SIZE",
            "VOLUME_MIN",
            "VOLUME_STEP",
            "VOLUME_MAX",
            "TICK_SIZE",
            "TICK_VALUE",
            "TICK_VALUE_PROFIT",
            "TICK_VALUE_LOSS",
            "DIGITS",
            "POINT",
            "CURRENCY_BASE",
            "CURRENCY_PROFIT",
            "CURRENCY_MARGIN",
            "SPREAD",
        ]
        print(broker[metadata_cols].drop_duplicates().head(5).to_string(index=False))

    print()
    print("9. Risk Manager vs order_calc_profit")
    if sizing_validation.empty:
        print("Sin validacion sizing; MT5 no disponible o sin trades.")
    else:
        print(sizing_validation.to_string(index=False))

    for idx, capital in enumerate([100, 500, 1000, 5000], start=10):
        print()
        print(f"{idx}. Broker-aware USD {capital}")
        if broker.empty:
            print("Sin metadata broker-aware.")
            continue
        subset = broker[broker["CAPITAL"] == capital]
        total = len(subset)
        ejecutables = int((subset["DECISION"] == "EXECUTABLE").sum()) if total else 0
        skips = total - ejecutables
        print(f"Total senales evaluadas: {total}")
        print(f"Ejecutables: {ejecutables}")
        print(f"SKIP: {skips}")
        print(f"Porcentaje ejecutable: {(ejecutables / total * 100) if total else 0:.2f}%")

    print()
    print("14. Candidatos")
    if candidates.empty:
        print("Sin candidatos descriptivos a forward test.")
    else:
        print(
            candidates[
                [
                    "DIRECCION",
                    "MODELO",
                    "SISTEMA",
                    "THRESHOLD",
                    "TRADES",
                    "PROFIT_FACTOR",
                    "EXPECTANCY_R",
                    "FOLDS_POSITIVOS",
                    "PORCENTAJE_FOLDS_POSITIVOS",
                    "MOTIVO",
                ]
            ].to_string(index=False)
        )

    print()
    print("15. Descartados")
    print(
        discarded[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "CLASIFICACION",
                "MOTIVO",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("16. Limitaciones")
    print("No se congelo ningun modelo USDCAD.")
    print("No se creo paper/usdcad.")
    print("Los candidatos son descriptivos, no garantias de rentabilidad.")
    print("Forward test requiere autorizacion explicita posterior.")
    print()
    print("Archivos creados:")
    print("results/v2/usdcad/usdcad_existing_models_audit.csv")
    print("results/v2/usdcad/usdcad_walkforward_summary.csv")
    print("results/v2/usdcad/usdcad_oof_summary.csv")
    print("results/v2/usdcad/usdcad_oof_predictions.csv")
    print("results/v2/usdcad/usdcad_robustness.csv")
    print("results/v2/usdcad/usdcad_trades.csv")
    print("results/v2/usdcad/usdcad_broker_aware.csv")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_research_usdchf():
    from src.usdchf_research import run_usdchf_research

    resultado = run_usdchf_research(RAIZ_PROYECTO)
    audit = resultado["audit"]
    walkforward = resultado["walkforward"]
    oof_summary = resultado["oof_summary"]
    robustness = resultado["robustness"]
    broker = resultado["broker_aware"]
    sizing_validation = resultado["sizing_validation"]
    candidates = resultado["candidates"]
    discarded = resultado["discarded"]
    data_period = resultado["data_period"]

    print("FOREX ML - RESEARCH USDCHF")
    print()
    print(f"Output: {resultado['output_dir']}")
    print("Ejecucion automatica: NO")
    print("Forward test iniciado: NO")
    print("Modelo validado declarado: NO")
    print()

    print("1. Periodo de datos")
    print(f"Fuente: {data_period['source']}")
    print(f"Inicio: {formatear_fecha(data_period['start'])}")
    print(f"Fin: {formatear_fecha(data_period['end'])}")
    print(f"Filas utiles: {data_period['rows']}")
    print(f"Intervalo: {data_period['interval']}")
    print(f"Periodo descargado: {data_period['downloaded_period']}")
    print(f"Dataset creado en esta ejecucion: {'SI' if data_period['created_now'] else 'NO'}")
    print()
    print("Modelos existentes / utilizados")
    columnas_audit = [
        "DIRECCION",
        "EXISTE",
        "FECHA_ARCHIVO",
        "ALGORITMO",
        "N_FEATURES_MODELO",
        "THRESHOLDS_DISPONIBLES",
        "COMPATIBLE_METODOLOGIA_ACTUAL",
        "LEAKAGE_LOOKAHEAD_CONOCIDO",
        "NOTAS",
    ]
    print(audit[columnas_audit].to_string(index=False))
    print("Modelos evaluados: LOGISTIC, RF_SIMPLE, RF_ACTUAL")

    print()
    print("2. LONG")
    _imprimir_top_escenarios("LONG - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "LONG")

    print()
    print("3. SHORT")
    _imprimir_top_escenarios("SHORT - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "SHORT")

    print()
    print("4. Walk-forward")
    wf_2bps = walkforward[walkforward["COST_BPS"] == 2].copy()
    wf_resumen = (
        wf_2bps.groupby(["DIRECCION", "MODELO", "SISTEMA", "THRESHOLD"], dropna=False)
        .agg(
            FOLDS=("FOLD", "nunique"),
            TRADES=("TRADES", "sum"),
            AUC_MEDIO=("AUC", "mean"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            R_TOTAL=("R_TOTAL", "sum"),
            FOLDS_POSITIVOS=("FOLD_POSITIVO", "sum"),
        )
        .reset_index()
        .sort_values(["DIRECCION", "EXPECTANCY_MEDIA"], ascending=[True, False])
    )
    print(wf_resumen.head(18).to_string(index=False))

    print()
    print("5. OOF")
    oof_2bps = oof_summary[oof_summary["COST_BPS"] == 2].copy()
    print(
        oof_2bps[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "WIN_RATE",
                "PROFIT_FACTOR",
                "EXPECTANCY_R",
                "R_TOTAL",
                "MAX_DRAWDOWN_PCT",
            ]
        ]
        .sort_values(["DIRECCION", "EXPECTANCY_R"], ascending=[True, False])
        .head(18)
        .to_string(index=False)
    )

    print()
    print("6. Costes")
    costes = (
        oof_summary.groupby(["DIRECCION", "COST_BPS"])
        .agg(
            ESCENARIOS=("MODELO", "count"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            TRADES_TOTAL=("TRADES", "sum"),
        )
        .reset_index()
    )
    print(costes.to_string(index=False))

    print()
    print("7. Robustez")
    robust_2bps = robustness[robustness["COST_BPS"] == 2].copy()
    _imprimir_top_escenarios("Robustez principal 2 bps", robust_2bps, None, limite=18)

    print()
    print("8. CI95")
    ci_cols = [
        "DIRECCION",
        "MODELO",
        "SISTEMA",
        "THRESHOLD",
        "TRADES",
        "EXPECTANCY_R",
        "BOOTSTRAP_EXPECTANCY_CI95_LOW",
        "BOOTSTRAP_EXPECTANCY_CI95_HIGH",
        "CLASIFICACION",
        "MOTIVO",
    ]
    print(
        robust_2bps.sort_values("EXPECTANCY_R", ascending=False)[ci_cols]
        .head(18)
        .to_string(index=False)
    )
    cruza_cero = robust_2bps[
        (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_LOW"] < 0)
        & (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_HIGH"] > 0)
    ]
    if not cruza_cero.empty:
        print("VENTAJA NO CONFIRMADA: existen configuraciones cuyo CI95 cruza cero.")

    print()
    print("9. Metadata XM")
    if broker.empty:
        print("Sin filas broker-aware.")
    else:
        metadata_cols = [
            "BROKER_METADATA_SOURCE",
            "SYMBOL",
            "CONTRACT_SIZE",
            "VOLUME_MIN",
            "VOLUME_STEP",
            "VOLUME_MAX",
            "TICK_SIZE",
            "TICK_VALUE",
            "TICK_VALUE_PROFIT",
            "TICK_VALUE_LOSS",
            "DIGITS",
            "POINT",
            "CURRENCY_BASE",
            "CURRENCY_PROFIT",
            "CURRENCY_MARGIN",
            "SPREAD",
        ]
        print(broker[metadata_cols].drop_duplicates().head(5).to_string(index=False))

    print()
    print("10. Risk Manager vs order_calc_profit")
    if sizing_validation.empty:
        print("Sin validacion sizing; MT5 no disponible o sin trades.")
    else:
        print(sizing_validation.to_string(index=False))

    for idx, capital in enumerate([100, 500, 1000, 5000], start=11):
        print()
        print(f"{idx}. USD {capital}")
        if broker.empty:
            print("Sin metadata broker-aware.")
            continue
        subset = broker[broker["CAPITAL"] == capital]
        total = len(subset)
        ejecutables = int((subset["DECISION"] == "EXECUTABLE").sum()) if total else 0
        skips = total - ejecutables
        print(f"Total evaluado: {total}")
        print(f"Ejecutables: {ejecutables}")
        print(f"SKIP: {skips}")
        print(f"Porcentaje ejecutable: {(ejecutables / total * 100) if total else 0:.2f}%")

    print()
    print("15. Candidatos")
    if candidates.empty:
        print("Sin candidatos descriptivos a forward test.")
    else:
        print(
            candidates[
                [
                    "DIRECCION",
                    "MODELO",
                    "SISTEMA",
                    "THRESHOLD",
                    "TRADES",
                    "PROFIT_FACTOR",
                    "EXPECTANCY_R",
                    "FOLDS_POSITIVOS",
                    "PORCENTAJE_FOLDS_POSITIVOS",
                    "MOTIVO",
                ]
            ].to_string(index=False)
        )

    print()
    print("16. Descartados")
    print(
        discarded[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "CLASIFICACION",
                "MOTIVO",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("17. Limitaciones")
    print("No se congelo ningun modelo USDCHF.")
    print("No se creo paper/usdchf.")
    print("Los candidatos son descriptivos, no garantias de rentabilidad.")
    print("Forward test requiere autorizacion explicita posterior.")
    print()
    print("Archivos creados:")
    print("results/v2/usdchf/usdchf_existing_models_audit.csv")
    print("results/v2/usdchf/usdchf_walkforward_summary.csv")
    print("results/v2/usdchf/usdchf_oof_summary.csv")
    print("results/v2/usdchf/usdchf_oof_predictions.csv")
    print("results/v2/usdchf/usdchf_robustness.csv")
    print("results/v2/usdchf/usdchf_trades.csv")
    print("results/v2/usdchf/usdchf_broker_aware.csv")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def _imprimir_research_fx_generico(asset, resultado):
    asset_lower = asset.lower()
    audit = resultado["audit"]
    walkforward = resultado["walkforward"]
    oof_summary = resultado["oof_summary"]
    robustness = resultado["robustness"]
    broker = resultado["broker_aware"]
    sizing_validation = resultado["sizing_validation"]
    candidates = resultado["candidates"]
    discarded = resultado["discarded"]
    data_period = resultado["data_period"]

    print(f"FOREX ML - RESEARCH {asset}")
    print()
    print(f"Output: {resultado['output_dir']}")
    print("Ejecucion automatica: NO")
    print("Forward test iniciado: NO")
    print("Modelo validado declarado: NO")
    print()

    print("1. Periodo de datos")
    print(f"Fuente: {data_period['source']}")
    print(f"Inicio: {formatear_fecha(data_period['start'])}")
    print(f"Fin: {formatear_fecha(data_period['end'])}")
    print(f"Filas utiles: {data_period['rows']}")
    print(f"Intervalo: {data_period['interval']}")
    print(f"Periodo descargado: {data_period['downloaded_period']}")
    print(f"Dataset creado en esta ejecucion: {'SI' if data_period['created_now'] else 'NO'}")
    print()
    print("Modelos existentes / utilizados")
    columnas_audit = [
        "DIRECCION",
        "EXISTE",
        "FECHA_ARCHIVO",
        "ALGORITMO",
        "N_FEATURES_MODELO",
        "THRESHOLDS_DISPONIBLES",
        "COMPATIBLE_METODOLOGIA_ACTUAL",
        "LEAKAGE_LOOKAHEAD_CONOCIDO",
        "NOTAS",
    ]
    print(audit[columnas_audit].to_string(index=False))
    print("Modelos evaluados: LOGISTIC, RF_SIMPLE, RF_ACTUAL")

    print()
    print("2. LONG")
    _imprimir_top_escenarios("LONG - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "LONG")

    print()
    print("3. SHORT")
    _imprimir_top_escenarios("SHORT - escenarios 2 bps", robustness[robustness["COST_BPS"] == 2], "SHORT")

    print()
    print("4. Walk-forward")
    wf_2bps = walkforward[walkforward["COST_BPS"] == 2].copy()
    wf_resumen = (
        wf_2bps.groupby(["DIRECCION", "MODELO", "SISTEMA", "THRESHOLD"], dropna=False)
        .agg(
            FOLDS=("FOLD", "nunique"),
            TRADES=("TRADES", "sum"),
            AUC_MEDIO=("AUC", "mean"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            R_TOTAL=("R_TOTAL", "sum"),
            FOLDS_POSITIVOS=("FOLD_POSITIVO", "sum"),
        )
        .reset_index()
        .sort_values(["DIRECCION", "EXPECTANCY_MEDIA"], ascending=[True, False])
    )
    print(wf_resumen.head(18).to_string(index=False))

    print()
    print("5. OOF")
    oof_2bps = oof_summary[oof_summary["COST_BPS"] == 2].copy()
    print(
        oof_2bps[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "WIN_RATE",
                "PROFIT_FACTOR",
                "EXPECTANCY_R",
                "R_TOTAL",
                "MAX_DRAWDOWN_PCT",
            ]
        ]
        .sort_values(["DIRECCION", "EXPECTANCY_R"], ascending=[True, False])
        .head(18)
        .to_string(index=False)
    )

    print()
    print("6. Costes")
    costes = (
        oof_summary.groupby(["DIRECCION", "COST_BPS"])
        .agg(
            ESCENARIOS=("MODELO", "count"),
            EXPECTANCY_MEDIA=("EXPECTANCY_R", "mean"),
            PF_MEDIO=("PROFIT_FACTOR", "mean"),
            TRADES_TOTAL=("TRADES", "sum"),
        )
        .reset_index()
    )
    print(costes.to_string(index=False))

    print()
    print("7. Robustez")
    robust_2bps = robustness[robustness["COST_BPS"] == 2].copy()
    _imprimir_top_escenarios("Robustez principal 2 bps", robust_2bps, None, limite=18)

    print()
    print("8. CI95")
    ci_cols = [
        "DIRECCION",
        "MODELO",
        "SISTEMA",
        "THRESHOLD",
        "TRADES",
        "EXPECTANCY_R",
        "BOOTSTRAP_EXPECTANCY_CI95_LOW",
        "BOOTSTRAP_EXPECTANCY_CI95_HIGH",
        "CLASIFICACION",
        "MOTIVO",
    ]
    print(
        robust_2bps.sort_values("EXPECTANCY_R", ascending=False)[ci_cols]
        .head(18)
        .to_string(index=False)
    )
    cruza_cero = robust_2bps[
        (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_LOW"] < 0)
        & (robust_2bps["BOOTSTRAP_EXPECTANCY_CI95_HIGH"] > 0)
    ]
    if not cruza_cero.empty:
        print("VENTAJA NO CONFIRMADA: existen configuraciones cuyo CI95 cruza cero.")

    print()
    print("9. Metadata XM")
    if broker.empty:
        print("Sin filas broker-aware.")
    else:
        metadata_cols = [
            "BROKER_METADATA_SOURCE",
            "SYMBOL",
            "CONTRACT_SIZE",
            "VOLUME_MIN",
            "VOLUME_STEP",
            "VOLUME_MAX",
            "TICK_SIZE",
            "TICK_VALUE",
            "TICK_VALUE_PROFIT",
            "TICK_VALUE_LOSS",
            "DIGITS",
            "POINT",
            "CURRENCY_BASE",
            "CURRENCY_PROFIT",
            "CURRENCY_MARGIN",
            "SPREAD",
        ]
        print(broker[metadata_cols].drop_duplicates().head(5).to_string(index=False))

    print()
    print("10. Risk Manager vs order_calc_profit")
    if sizing_validation.empty:
        print("Sin validacion sizing; MT5 no disponible o sin trades.")
    else:
        print(sizing_validation.to_string(index=False))

    for idx, capital in enumerate([100, 500, 1000, 5000], start=11):
        print()
        print(f"{idx}. USD {capital}")
        if broker.empty:
            print("Sin metadata broker-aware.")
            continue
        subset = broker[broker["CAPITAL"] == capital]
        total = len(subset)
        ejecutables = int((subset["DECISION"] == "EXECUTABLE").sum()) if total else 0
        skips = total - ejecutables
        print(f"Total evaluado: {total}")
        print(f"Ejecutables: {ejecutables}")
        print(f"SKIP: {skips}")
        print(f"Porcentaje ejecutable: {(ejecutables / total * 100) if total else 0:.2f}%")

    print()
    print("15. Candidatos")
    if candidates.empty:
        print("Sin candidatos descriptivos a forward test.")
    else:
        print(
            candidates[
                [
                    "DIRECCION",
                    "MODELO",
                    "SISTEMA",
                    "THRESHOLD",
                    "TRADES",
                    "PROFIT_FACTOR",
                    "EXPECTANCY_R",
                    "FOLDS_POSITIVOS",
                    "PORCENTAJE_FOLDS_POSITIVOS",
                    "MOTIVO",
                ]
            ].to_string(index=False)
        )

    print()
    print("16. Descartados")
    print(
        discarded[
            [
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
                "TRADES",
                "CLASIFICACION",
                "MOTIVO",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("17. Limitaciones")
    print(f"No se congelo ningun modelo {asset}.")
    print(f"No se creo paper/{asset_lower}.")
    print("Los candidatos son descriptivos, no garantias de rentabilidad.")
    print("Forward test requiere autorizacion explicita posterior.")
    print()
    print("Archivos creados:")
    print(f"results/v2/{asset_lower}/{asset_lower}_existing_models_audit.csv")
    print(f"results/v2/{asset_lower}/{asset_lower}_walkforward_summary.csv")
    print(f"results/v2/{asset_lower}/{asset_lower}_oof_summary.csv")
    print(f"results/v2/{asset_lower}/{asset_lower}_oof_predictions.csv")
    print(f"results/v2/{asset_lower}/{asset_lower}_robustness.csv")
    print(f"results/v2/{asset_lower}/{asset_lower}_trades.csv")
    print(f"results/v2/{asset_lower}/{asset_lower}_broker_aware.csv")
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_research_nzdusd():
    from src.nzdusd_research import run_nzdusd_research

    resultado = run_nzdusd_research(RAIZ_PROYECTO)
    _imprimir_research_fx_generico("NZDUSD", resultado)


def ejecutar_paper_freeze():
    from src.paper_trader import freeze_paper_model

    resultado = freeze_paper_model(RAIZ_PROYECTO)
    metadata = resultado["metadata"]

    print("=" * 80)
    print("FOREX ML - PAPER FREEZE")
    print("=" * 80)

    if not resultado["creado"]:
        print("MODELO PAPER YA CONGELADO")
        print("No se ha reentrenado.")
        print()

    print("Modelo congelado: SI")
    print(f"Ruta: {resultado['ruta']}")
    print(f"Fecha fin entrenamiento: {metadata.get('fecha_fin_entrenamiento', 'N/A')}")
    print(f"Paper start: {metadata.get('paper_start', 'N/A')}")
    print(f"Filas entrenamiento: {metadata.get('numero_filas_entrenamiento', 'N/A')}")
    print("Features:")
    for feature in metadata.get("features", []):
        print(f"- {feature}")
    print("Data posterior al paper utilizada: NO")
    print("No se ejecuta ninguna orden real.")


def ejecutar_backtest_v2_gold():
    from src.backtest_v2 import ejecutar_backtest_v2_gold

    resultado = ejecutar_backtest_v2_gold(RAIZ_PROYECTO)
    resumen = resultado["summary"]
    broker_metadata = resultado["broker_metadata"]

    print("=" * 110)
    print("FOREX ML - BACKTEST V2 GOLD")
    print("=" * 110)
    print("Fase: comparativa / exploratoria")
    print("Activo: GOLD")
    print("Modelos nuevos: NO")
    print("Trading automatico: NO")
    print(f"Broker metadata source: {broker_metadata.source}")
    print(f"Contract size: {formatear_numero(broker_metadata.contract_size)}")
    print(f"Volume min: {formatear_numero(broker_metadata.volume_min)}")
    print(f"Volume step: {formatear_numero(broker_metadata.volume_step)}")
    print()
    print(f"Entradas base: {len(resultado['entries'])}")
    print(f"Resumen guardado: {resultado['ruta_summary']}")
    print(f"Trades guardados: {resultado['ruta_trades']}")
    print(f"Entradas guardadas: {resultado['ruta_entries']}")
    print()
    print(
        f"{'CONFIG':<38} {'MODE':<24} {'COST':>4} {'CAP':>7} "
        f"{'BRK':>3} {'TR':>4} {'WIN%':>8} {'PF':>8} {'AVG_R':>8} "
        f"{'R':>8} {'RET%':>8} {'DD%':>8} {'FINAL':>10} {'NO_MIN':>6}"
    )
    print("-" * 140)

    for fila in resumen.to_dict("records"):
        broker = "SI" if fila["BROKER_AWARE"] else "NO"
        print(
            f"{fila['CONFIG_ID']:<38} "
            f"{fila['SL_TP_MODE']:<24} "
            f"{int(fila['COST_BPS']):>4} "
            f"{formatear_numero(float(fila['CAPITAL_INICIAL'])):>7} "
            f"{broker:>3} "
            f"{int(fila['TOTAL_TRADES']):>4} "
            f"{formatear_numero(float(fila['WIN_RATE'])):>8} "
            f"{formatear_numero(float(fila['PROFIT_FACTOR'])):>8} "
            f"{formatear_numero(float(fila['AVERAGE_R'])):>8} "
            f"{formatear_numero(float(fila['R_ACUMULADO'])):>8} "
            f"{formatear_numero(float(fila['RETURN_PCT'])):>8} "
            f"{formatear_numero(float(fila['MAX_DRAWDOWN'])):>8} "
            f"{formatear_numero(float(fila['CAPITAL_FINAL'])):>10} "
            f"{int(fila['NO_EJECUTABLES_BROKER_MIN']):>6}"
        )

    print()
    print("No se declara ganador automatico.")
    print("No se ejecuto walk-forward V2.")
    print("TRADING ENABLED: NO")
    print("ORDENES ENVIADAS: 0")


def ejecutar_walkforward_v2_gold():
    from src.walkforward_v2 import ejecutar_walkforward_v2_gold

    resultado = ejecutar_walkforward_v2_gold(RAIZ_PROYECTO)
    summary = resultado["summary"]
    folds = resultado["folds"]
    broker_metadata = resultado["broker_metadata"]
    principal = summary[
        (summary["COST_BPS"] == 2)
        & (summary["CAPITAL_INICIAL"] == 1000)
        & (~summary["BROKER_AWARE"])
    ]
    broker = summary[
        (summary["COST_BPS"] == 2)
        & (summary["BROKER_AWARE"])
    ]

    print("=" * 120)
    print("FOREX ML - WALK-FORWARD V2 GOLD")
    print("=" * 120)
    print("Fase: validacion temporal comparativa / exploratoria")
    print("Ventanas: EXPANDING")
    print("Shuffle: NO")
    print("Embargo minimo: 24 barras")
    print("Modelos nuevos: NO")
    print("Trading automatico: NO")
    print(f"Broker metadata source: {broker_metadata.source}")
    print(f"Summary: {resultado['ruta_summary']}")
    print(f"Folds: {resultado['ruta_folds']}")
    print(f"Trades: {resultado['ruta_trades']}")
    print()
    print("Folds generados:")
    for fold in (
        folds[
            [
                "CONFIG_ID",
                "FOLD",
                "TRAIN_END",
                "EMBARGO_BARS",
                "TEST_START",
                "TEST_END",
                "TEST_BARS",
            ]
        ]
        .drop_duplicates()
        .to_dict("records")
    ):
        print(
            f"{fold['CONFIG_ID']} | Fold {int(fold['FOLD'])} | "
            f"train_end={fold['TRAIN_END']} | embargo={int(fold['EMBARGO_BARS'])} | "
            f"test={fold['TEST_START']} -> {fold['TEST_END']} "
            f"({int(fold['TEST_BARS'])} barras)"
        )

    print()
    print("Escenario principal: 2 bps, $1000, teorico porcentual")
    print(
        f"{'CONFIG':<38} {'MODE':<24} {'TR':>4} {'MEAN_EXP':>9} "
        f"{'MED_EXP':>9} {'POS%':>7} {'CI_LOW':>9} {'CI_HIGH':>9} "
        f"{'DD95':>8} {'CLASIFICACION':<20}"
    )
    print("-" * 140)
    for fila in principal.to_dict("records"):
        print(
            f"{fila['CONFIG_ID']:<38} "
            f"{fila['SL_TP_MODE']:<24} "
            f"{int(fila['TOTAL_TRADES']):>4} "
            f"{formatear_numero(float(fila['MEAN_EXPECTANCY'])):>9} "
            f"{formatear_numero(float(fila['MEDIAN_EXPECTANCY'])):>9} "
            f"{formatear_numero(float(fila['POSITIVE_FOLDS_PCT'])):>7} "
            f"{formatear_numero(fila['EXPECTANCY_CI95_LOW']):>9} "
            f"{formatear_numero(fila['EXPECTANCY_CI95_HIGH']):>9} "
            f"{formatear_numero(fila['DRAWDOWN_95']):>8} "
            f"{fila['CLASSIFICATION']:<20}"
        )

    print()
    print("Broker-aware 2 bps: no ejecutables por minimo de broker")
    print(
        broker.groupby(["CONFIG_ID", "SL_TP_MODE", "CAPITAL_INICIAL"])[
            "NO_EJECUTABLES_BROKER_MIN"
        ]
        .sum()
        .to_string()
    )
    print()
    print("No se declara ganador automatico.")
    print("TRADING ENABLED: NO")
    print("ORDENES ENVIADAS: 0")


def _imprimir_resultado_ejecucion(resultado):
    print(f"  Estado: {resultado.status}")
    print(f"  Motivo: {resultado.reason}")
    print(f"  Simbolo: {resultado.symbol or resultado.asset}")
    print(f"  Direccion: {resultado.direction}")
    print(f"  Volumen: {resultado.volume}")
    print(f"  Precio: {resultado.price}  SL: {resultado.sl}  TP: {resultado.tp}")
    print(f"  Spread: {resultado.spread}")
    print(f"  Ticket: {resultado.ticket}")


def ejecutar_live(activo, forzar_dryrun=False):
    from src.live_executor import STATUS_SENT, kill_switch_active
    from src.live_runner import run_eurusd_live, run_gold_live

    runner = run_gold_live if activo == "GOLD" else run_eurusd_live
    try:
        resultado = runner(RAIZ_PROYECTO, dry_run=True if forzar_dryrun else None)
    except RuntimeError as error:
        # El PAPER ya quedo procesado; solo fallo la conexion MT5 del ejecutor.
        print(f"FOREX ML - LIVE {activo}: ERROR MT5: {error}")
        print("PAPER procesado. Ejecutor MT5 no disponible: ORDENES ENVIADAS: 0")
        sys.exit(1)
    paper = resultado["paper"] or {}

    print(f"FOREX ML - LIVE {activo} (MT5 DEMO)")
    print()
    print(f"CONFIG: {resultado['config_id']}")
    print(f"MODO: {'DRY RUN (no envia ordenes)' if resultado['dry_run'] else 'EJECUCION DEMO'}")
    print(f"KILL SWITCH: {'ACTIVO' if kill_switch_active(RAIZ_PROYECTO) else 'NO'}")
    if resultado["paper_error"]:
        print(f"ERROR PAPER: {resultado['paper_error']}")
    elif activo == "EURUSD":
        print(f"DATA STATUS PAPER: {paper.get('data_status')}")
        print(f"Velas PAPER procesadas: {paper.get('processed', 0)}")
    print()
    intent = resultado["intent"]
    if intent is None:
        print("Senal pendiente: ninguna")
    else:
        print(f"Senal pendiente: {intent.signal_id} ({'fresca' if resultado['fresh'] else 'VIEJA, no se ejecuta'})")
    if resultado["execution"] is not None:
        print("Ejecucion:")
        _imprimir_resultado_ejecucion(resultado["execution"])
    print(f"Cierres por tiempo: {len(resultado['closed'])}")
    for cierre in resultado["closed"]:
        _imprimir_resultado_ejecucion(cierre)
    enviadas = 1 if resultado["execution"] is not None and resultado["execution"].status == STATUS_SENT else 0
    print()
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print(f"ORDENES ENVIADAS: {enviadas}")
    if resultado["paper_error"]:
        sys.exit(1)


def ejecutar_live_portfolio(forzar_dryrun=False):
    from src.demo_portfolio import run_portfolio_live
    from src.live_executor import STATUS_SENT, kill_switch_active

    try:
        resultado = run_portfolio_live(RAIZ_PROYECTO, dry_run=True if forzar_dryrun else None)
    except Exception as error:
        print(f"FOREX ML - LIVE PORTFOLIO: ERROR: {error}")
        if not forzar_dryrun:
            _avisar_fallo_runner("LIVE_PORTFOLIO", error)
        sys.exit(1)

    print("FOREX ML - LIVE PORTFOLIO DEMO (exploracion, no validado para real)")
    print()
    print(f"MODO: {'DRY RUN (no envia ordenes)' if resultado['dry_run'] else 'EJECUCION DEMO'}")
    print(f"KILL SWITCH: {'ACTIVO' if kill_switch_active(RAIZ_PROYECTO) else 'NO'}")
    print()
    print(f"{'ACTIVO':<8}{'SIMBOLO':<10}{'P LONG':>8}{'P SHORT':>9}  {'DECISION':<10}EJECUCION")
    enviadas = 0
    for item in resultado["results"]:
        ejecucion = item["execution"]
        detalle = ""
        if ejecucion is not None:
            detalle = f"{ejecucion.status} {ejecucion.reason}"
            if ejecucion.volume:
                detalle += f" | {ejecucion.volume} lotes @ {ejecucion.price} SL {ejecucion.sl} TP {ejecucion.tp}"
            enviadas += ejecucion.status == STATUS_SENT
        p_long = f"{item['prob_long']:.3f}" if "prob_long" in item else "-"
        p_short = f"{item['prob_short']:.3f}" if "prob_short" in item else "-"
        print(f"{item['asset']:<8}{item['symbol'] or '-':<10}{p_long:>8}{p_short:>9}  {item['decision']:<10}{detalle}")
        for cierre in item["closed"]:
            print(f"    cierre por tiempo: ticket {cierre.ticket} {cierre.status}")
    print()
    print(f"Actualizaciones SL protectoras: {len(resultado.get('protective_updates', []))}")
    print(f"Cierres avisados por Telegram (SL/TP/manual): {len(resultado['closed_deals'])}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print(f"ORDENES ENVIADAS: {enviadas}")


def ejecutar_live_portfolio_audit():
    from src.demo_portfolio import audit_portfolio_models

    rows = audit_portfolio_models(RAIZ_PROYECTO)
    print("FOREX ML - AUDITORIA MODELOS PORTAFOLIO DEMO")
    print()
    print(f"{'ACTIVO':<8}{'DIR':<7}{'OK':<5}{'TIMEFRAME':<10}CONFIG / MOTIVO")
    for row in rows:
        ok = "SI" if row["compatible"] else "NO"
        detail = row["config_id"] if row["compatible"] else row["reason"]
        print(f"{row['asset']:<8}{row['direction']:<7}{ok:<5}{str(row['timeframe']):<10}{detail}")
    print()
    print(f"Modelos compatibles: {sum(1 for row in rows if row['compatible'])}/{len(rows)}")
    print(f"TRADING ENABLED: {_estado_bool(TRADING_ENABLED)}")
    print(f"DEMO EXECUTION ENABLED: {_estado_bool(DEMO_EXECUTION_ENABLED)}")
    print("ORDENES ENVIADAS: 0")


def ejecutar_live_portfolio_train():
    import pandas as pd

    from src.demo_portfolio import train_portfolio_models

    print("Entrenando modelos del portafolio DEMO (puede tardar unos minutos)...")
    reporte = pd.DataFrame(train_portfolio_models(RAIZ_PROYECTO))
    print(reporte[["activo", "direccion", "train_end", "holdout_signals_per_day", "holdout_win_rate"]].to_string(index=False))


def ejecutar_live_portfolio_research(descargar=True):
    import pandas as pd

    from config import DEMO_PORTFOLIO_ASSETS, NEW_PORTFOLIO_ASSETS
    from src.portfolio_research import CONFIRMATION_START, download_mt5_history, run_full_research

    activos = list(dict.fromkeys(DEMO_PORTFOLIO_ASSETS + NEW_PORTFOLIO_ASSETS))
    if descargar:
        print("Descargando historico H1 del broker (MT5)...")
        for activo, estado in download_mt5_history(RAIZ_PROYECTO, activos).items():
            print(f"  {activo}: {estado}")
    salida = run_full_research(RAIZ_PROYECTO, activos)
    tabla = salida["table"]
    pd.set_option("display.width", 220)
    print()
    print("FOREX ML - INVESTIGACION WALK-FORWARD PORTAFOLIO (R despues de costos)")
    print(tabla.to_string(index=False))
    print()
    print(f"Comparacion de portafolios desde {CONFIRMATION_START:%Y-%m-%d} (riesgo 1%, max 5 posiciones, 2 por moneda):")
    for nombre, sim in salida["portfolios"].items():
        print(f"  {nombre:<30} capital x{sim['final_equity']:.2f}  max DD {sim['max_drawdown']:.1%}  "
              f"operaciones {sim['trades']} (omitidas {sim['skipped']})")
    print()
    print(f"Direcciones habilitadas: {int(tabla['enabled'].sum())}/{len(tabla)}")
    print(f"Seleccion guardada en: {salida['selection_path']}")


def ejecutar_live_test(activo, direccion):
    from src.demo_portfolio import send_test_order

    direccion = {"COMPRA": "BUY", "VENTA": "SELL"}.get(direccion, direccion)
    resultado = send_test_order(RAIZ_PROYECTO, activo, direccion)
    print(f"FOREX ML - ORDEN DE PRUEBA {activo} {direccion}")
    print()
    print(f"MODO: {'DRY RUN (no envia ordenes)' if resultado.dry_run else 'EJECUCION DEMO'}")
    _imprimir_resultado_ejecucion(resultado)


def ejecutar_live_closeall():
    from src.live_executor import close_all_positions, mt5_session, telegram_notifier

    dry_run = not DEMO_EXECUTION_ENABLED
    notify = None if dry_run else telegram_notifier(RAIZ_PROYECTO, "ALL", "CLOSEALL")
    with mt5_session() as mt5:
        resultados = close_all_positions(mt5, RAIZ_PROYECTO, dry_run=dry_run, notify=notify)
    print("FOREX ML - CERRAR TODAS LAS POSICIONES DEL BOT")
    print(f"MODO: {'DRY RUN (no cierra)' if dry_run else 'EJECUCION DEMO'}")
    print(f"Posiciones procesadas: {len(resultados)}")
    for resultado in resultados:
        _imprimir_resultado_ejecucion(resultado)


def ejecutar_live_reconcile():
    from src.live_executor import mt5_session, reconcile

    with mt5_session() as mt5:
        resultado = reconcile(mt5, RAIZ_PROYECTO)

    print("FOREX ML - LIVE RECONCILE")
    print()
    print(f"Ordenes enviadas en journal: {resultado['journal_sent']}")
    print(f"Posiciones abiertas del bot en MT5: {len(resultado['open_positions'])}")
    for posicion in resultado["open_positions"]:
        print(
            f"  #{posicion.ticket} {posicion.symbol} vol {posicion.volume} "
            f"precio {posicion.price_open} SL {posicion.sl} TP {posicion.tp} P/L {posicion.profit}"
        )
    print(f"En journal pero no en MT5: {len(resultado['missing_in_mt5'])}")
    for fila in resultado["missing_in_mt5"]:
        print(f"  {fila['SIGNAL_ID']} ticket {fila['TICKET']}")
    print(f"En MT5 pero no en journal: {len(resultado['orphans_in_mt5'])}")
    for posicion in resultado["orphans_in_mt5"]:
        print(f"  #{posicion.ticket} {posicion.symbol} comentario {posicion.comment}")
    print()
    print(f"RESULTADO: {'OK' if resultado['ok'] else 'REVISAR'}")
    if not resultado["ok"]:
        sys.exit(1)


def ejecutar_live_stop():
    from src.live_executor import kill_switch_path

    ruta = kill_switch_path(RAIZ_PROYECTO)
    ruta.write_text("Aperturas bloqueadas manualmente.\n", encoding="utf-8")
    print(f"KILL SWITCH ACTIVADO: {ruta}")
    print("No se abriran nuevas posiciones. Los cierres por tiempo siguen funcionando.")


def ejecutar_live_resume():
    from src.live_executor import kill_switch_path

    ruta = kill_switch_path(RAIZ_PROYECTO)
    if ruta.exists():
        ruta.unlink()
    print("KILL SWITCH DESACTIVADO.")


def main():
    if len(sys.argv) < 2:
        mostrar_activos_disponibles()
        return

    activo = sys.argv[1].upper()

    if activo == "GOLD":
        modo_gold = sys.argv[2].upper() if len(sys.argv) >= 3 else ""
        submodo_gold = sys.argv[3].upper() if len(sys.argv) >= 4 else ""
        if modo_gold == "DATA" and submodo_gold == "CHECK":
            ejecutar_gold_data_check()
            return

    if activo == "PAPER":
        modo_paper = sys.argv[2].upper() if len(sys.argv) >= 3 else "RUN"
        submodo_paper = sys.argv[3].upper() if len(sys.argv) >= 4 else "RUN"
        if modo_paper == "EURUSD":
            if submodo_paper == "STATUS":
                ejecutar_paper_eurusd_status()
            else:
                ejecutar_paper_eurusd()
        elif modo_paper == "STATUS":
            ejecutar_paper_status()
        elif modo_paper == "FREEZE":
            ejecutar_paper_freeze()
        else:
            ejecutar_paper()
        return

    if activo == "TELEGRAM":
        modo_telegram = sys.argv[2].upper() if len(sys.argv) >= 3 else ""
        if modo_telegram == "TEST":
            ejecutar_telegram_test()
        elif modo_telegram == "DEMO":
            ejecutar_telegram_demo()
        elif modo_telegram == "DAILY":
            ejecutar_telegram_daily()
        elif modo_telegram == "STATUS":
            ejecutar_telegram_status()
        else:
            print("Usa: python main.py TELEGRAM TEST")
            print("O: python main.py TELEGRAM DEMO")
            print("O: python main.py TELEGRAM DAILY")
            print("O: python main.py TELEGRAM STATUS")
        return

    if activo == "HEALTH":
        ejecutar_health()
        return

    if activo == "LIVE":
        modo_live = sys.argv[2].upper() if len(sys.argv) >= 3 else ""
        submodo_live = sys.argv[3].upper() if len(sys.argv) >= 4 else ""
        if modo_live in ("EURUSD", "GOLD"):
            ejecutar_live(modo_live, forzar_dryrun=submodo_live == "DRYRUN")
        elif modo_live == "PORTFOLIO":
            if submodo_live == "AUDIT":
                ejecutar_live_portfolio_audit()
            elif submodo_live == "TRAIN":
                ejecutar_live_portfolio_train()
            elif submodo_live == "RESEARCH":
                ejecutar_live_portfolio_research(descargar="NODOWNLOAD" not in [x.upper() for x in sys.argv[4:]])
            else:
                ejecutar_live_portfolio(forzar_dryrun=submodo_live == "DRYRUN")
        elif modo_live == "TEST" and len(sys.argv) >= 5:
            ejecutar_live_test(sys.argv[3].upper(), sys.argv[4].upper())
        elif modo_live == "CLOSEALL":
            ejecutar_live_closeall()
        elif modo_live == "RECONCILE":
            ejecutar_live_reconcile()
        elif modo_live == "STOP":
            ejecutar_live_stop()
        elif modo_live == "RESUME":
            ejecutar_live_resume()
        else:
            print("Usa: python main.py LIVE EURUSD [DRYRUN]")
            print("O: python main.py LIVE GOLD [DRYRUN]")
            print("O: python main.py LIVE PORTFOLIO [DRYRUN|TRAIN|AUDIT|RESEARCH [NODOWNLOAD]]")
            print("O: python main.py LIVE TEST EURUSD BUY|SELL")
            print("O: python main.py LIVE CLOSEALL")
            print("O: python main.py LIVE RECONCILE")
            print("O: python main.py LIVE STOP")
            print("O: python main.py LIVE RESUME")
        return

    if activo == "RESEARCH":
        scope = sys.argv[2].upper() if len(sys.argv) >= 3 else ""
        if scope == "EURUSD":
            ejecutar_research_eurusd()
        elif scope == "GBPUSD":
            ejecutar_research_gbpusd()
        elif scope == "USDJPY":
            ejecutar_research_usdjpy()
        elif scope == "AUDUSD":
            ejecutar_research_audusd()
        elif scope == "USDCAD":
            ejecutar_research_usdcad()
        elif scope == "USDCHF":
            ejecutar_research_usdchf()
        elif scope == "NZDUSD":
            ejecutar_research_nzdusd()
        else:
            print("Usa: python main.py RESEARCH EURUSD")
            print("O: python main.py RESEARCH GBPUSD")
            print("O: python main.py RESEARCH USDJPY")
            print("O: python main.py RESEARCH AUDUSD")
            print("O: python main.py RESEARCH USDCAD")
            print("O: python main.py RESEARCH USDCHF")
            print("O: python main.py RESEARCH NZDUSD")
        return

    if activo == "BACKTEST":
        fase = sys.argv[2].upper() if len(sys.argv) >= 3 else ""
        scope = sys.argv[3].upper() if len(sys.argv) >= 4 else ""
        if fase == "V2" and scope == "GOLD":
            ejecutar_backtest_v2_gold()
        else:
            print("Usa: python main.py BACKTEST V2 GOLD")
        return

    if activo == "WALKFORWARD":
        fase = sys.argv[2].upper() if len(sys.argv) >= 3 else ""
        scope = sys.argv[3].upper() if len(sys.argv) >= 4 else ""
        if fase == "V2" and scope == "GOLD":
            ejecutar_walkforward_v2_gold()
        else:
            print("Usa: python main.py WALKFORWARD V2 GOLD")
        return

    if activo == "MT5":
        modo_mt5 = sys.argv[2].upper() if len(sys.argv) >= 3 else ""
        if modo_mt5 == "CHECK":
            ejecutar_mt5_check()
        elif modo_mt5 == "DEMO":
            scope_mt5 = sys.argv[3].upper() if len(sys.argv) >= 4 else ""
            accion_mt5 = sys.argv[4].upper() if len(sys.argv) >= 5 else ""
            if scope_mt5 == "TEST":
                if accion_mt5 == "CLOSE":
                    ejecutar_mt5_demo_test("CLOSE")
                elif accion_mt5 == "PREVIEW":
                    ejecutar_mt5_demo_test("PREVIEW")
                elif accion_mt5 == "":
                    ejecutar_mt5_demo_test("TEST")
                else:
                    print("Usa: python main.py MT5 DEMO TEST [PREVIEW|CLOSE]")
            elif scope_mt5 == "AUTO":
                if accion_mt5 == "PREVIEW":
                    ejecutar_mt5_demo_auto_preview()
                elif accion_mt5 == "STATUS":
                    ejecutar_mt5_demo_auto_status()
                elif accion_mt5 == "RUN":
                    ejecutar_mt5_demo_auto_run()
                else:
                    print("Usa: python main.py MT5 DEMO AUTO PREVIEW")
                    print("O: python main.py MT5 DEMO AUTO STATUS")
                    print("O: python main.py MT5 DEMO AUTO RUN")
            elif scope_mt5 == "GOLD":
                if accion_mt5 == "PREVIEW":
                    ejecutar_mt5_demo_gold_preview()
                else:
                    print("Usa: python main.py MT5 DEMO GOLD PREVIEW")
            else:
                print("Usa: python main.py MT5 DEMO TEST [PREVIEW|CLOSE]")
                print("O: python main.py MT5 DEMO AUTO PREVIEW")
                print("O: python main.py MT5 DEMO AUTO STATUS")
                print("O: python main.py MT5 DEMO AUTO RUN")
                print("O: python main.py MT5 DEMO GOLD PREVIEW")
        elif modo_mt5 == "PREVIEW":
            scope_mt5 = sys.argv[3].upper() if len(sys.argv) >= 4 else ""
            if scope_mt5 == "ALL":
                ejecutar_mt5_preview_all()
            else:
                print("Usa: python main.py MT5 PREVIEW ALL")
        elif modo_mt5 == "DRYRUN":
            scope_mt5 = sys.argv[3].upper() if len(sys.argv) >= 4 else ""
            if scope_mt5 == "ALL":
                ejecutar_mt5_dryrun_all()
            else:
                ejecutar_mt5_dryrun()
        else:
            print("Usa: python main.py MT5 CHECK")
            print("O: python main.py MT5 DEMO TEST PREVIEW")
            print("O: python main.py MT5 DEMO TEST")
            print("O: python main.py MT5 DEMO TEST CLOSE")
            print("O: python main.py MT5 DEMO AUTO PREVIEW")
            print("O: python main.py MT5 DEMO AUTO STATUS")
            print("O: python main.py MT5 DEMO AUTO RUN")
            print("O: python main.py MT5 DEMO GOLD PREVIEW")
            print("O: python main.py MT5 PREVIEW ALL")
            print("O: python main.py MT5 DRYRUN")
            print("O: python main.py MT5 DRYRUN ALL")
        return

    if activo == "ROBUST":
        ejecutar_robustez()
        return

    if activo == "STABILITY":
        ejecutar_estabilidad()
        return

    modo = sys.argv[2].upper() if len(sys.argv) >= 3 else "DESCARGA"

    if modo not in ("DESCARGA", "BACKTEST", "ML", "WF", "OOF_BACKTEST"):
        print(f"Modo no soportado: {modo}")
        print("Usa DESCARGA, BACKTEST, ML, WF o OOF_BACKTEST.")
        return

    if activo == "ALL":
        if modo == "OOF_BACKTEST":
            ejecutar_oof_backtest_todos()
        elif modo == "WF":
            ejecutar_wf_todos()
        elif modo == "ML":
            ejecutar_ml_todos()
        elif modo == "BACKTEST":
            ejecutar_backtest_todos()
        else:
            ejecutar_descarga_todos()
        return

    if activo not in ACTIVOS:
        print(f"Activo no soportado: {activo}")
        print()
        mostrar_activos_disponibles()
        return

    if modo == "OOF_BACKTEST":
        ejecutar_oof_backtest_individual(activo)
    elif modo == "WF":
        ejecutar_wf_individual(activo)
    elif modo == "ML":
        ejecutar_ml_individual(activo)
    elif modo == "BACKTEST":
        ejecutar_backtest_individual(activo)
    else:
        ejecutar_descarga_individual(activo)


if __name__ == "__main__":
    main()
