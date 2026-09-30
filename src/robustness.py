import pandas as pd


COSTOS_REQUERIDOS = [0, 1, 2, 5]
COLUMNAS_REQUERIDAS = [
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "COST_BPS",
    "TRADES",
    "PROFIT FACTOR",
    "EXPECTANCY R",
    "R TOTAL",
    "RENTABILIDAD %",
    "MAX DRAWDOWN %",
]


def _validar_summary(df, ruta):
    faltantes = [columna for columna in COLUMNAS_REQUERIDAS if columna not in df.columns]
    if faltantes:
        raise ValueError(
            f"{ruta} no tiene columnas requeridas: {', '.join(faltantes)}"
        )


def _clasificar_muestra(trades):
    if trades < 30:
        return "MUESTRA_BAJA"

    if trades <= 99:
        return "MUESTRA_MEDIA"

    return "MUESTRA_ALTA"


def _sobrevive(fila):
    return (
        fila["PROFIT FACTOR"] > 1
        and fila["EXPECTANCY R"] > 0
        and fila["R TOTAL"] > 0
    )


def _normalizar_numericos(df):
    datos = df.copy()

    for columna in [
        "THRESHOLD",
        "COST_BPS",
        "TRADES",
        "PROFIT FACTOR",
        "EXPECTANCY R",
        "R TOTAL",
        "RENTABILIDAD %",
        "MAX DRAWDOWN %",
    ]:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    return datos.dropna(subset=["COST_BPS", "TRADES"])


def analizar_robustez(raiz_proyecto, activos):
    filas = []
    total_configuraciones = 0
    archivos_leidos = []
    archivos_faltantes = []

    for activo in activos:
        ruta = (
            raiz_proyecto
            / "results"
            / activo.lower()
            / "oof_backtest_summary.csv"
        )

        if not ruta.exists():
            archivos_faltantes.append(str(ruta))
            continue

        summary = pd.read_csv(ruta)
        _validar_summary(summary, ruta)
        summary = _normalizar_numericos(summary)
        archivos_leidos.append(str(ruta))

        claves = ["ACTIVO", "DIRECCION", "MODELO", "SISTEMA", "THRESHOLD"]
        for _, grupo in summary.groupby(claves, dropna=False):
            costos = {
                int(fila["COST_BPS"]): fila
                for _, fila in grupo.iterrows()
                if int(fila["COST_BPS"]) in COSTOS_REQUERIDOS
            }

            if not all(costo in costos for costo in COSTOS_REQUERIDOS):
                continue

            total_configuraciones += 1
            fila_1 = costos[1]

            if not (fila_1["TRADES"] >= 25 and _sobrevive(fila_1)):
                continue

            filas.append(
                {
                    "ACTIVO": fila_1["ACTIVO"],
                    "DIRECCION": fila_1["DIRECCION"],
                    "MODELO": fila_1["MODELO"],
                    "SISTEMA": fila_1["SISTEMA"],
                    "THRESHOLD": fila_1["THRESHOLD"],
                    "TRADES": int(fila_1["TRADES"]),
                    "PF_0": costos[0]["PROFIT FACTOR"],
                    "PF_1": costos[1]["PROFIT FACTOR"],
                    "PF_2": costos[2]["PROFIT FACTOR"],
                    "PF_5": costos[5]["PROFIT FACTOR"],
                    "EXPECT_1": fila_1["EXPECTANCY R"],
                    "RETURN_1": fila_1["RENTABILIDAD %"],
                    "MAX_DD_1": fila_1["MAX DRAWDOWN %"],
                    "SURVIVES_1_BPS": _sobrevive(costos[1]),
                    "SURVIVES_2_BPS": _sobrevive(costos[2]),
                    "SURVIVES_5_BPS": _sobrevive(costos[5]),
                    "TAMANO_MUESTRA": _clasificar_muestra(int(fila_1["TRADES"])),
                }
            )

    columnas = [
        "ACTIVO",
        "DIRECCION",
        "MODELO",
        "SISTEMA",
        "THRESHOLD",
        "TRADES",
        "PF_0",
        "PF_1",
        "PF_2",
        "PF_5",
        "EXPECT_1",
        "RETURN_1",
        "MAX_DD_1",
        "SURVIVES_1_BPS",
        "SURVIVES_2_BPS",
        "SURVIVES_5_BPS",
        "TAMANO_MUESTRA",
    ]
    robust = pd.DataFrame(filas, columns=columnas)

    if not robust.empty:
        robust = robust.sort_values(
            by=[
                "SURVIVES_2_BPS",
                "SURVIVES_1_BPS",
                "ACTIVO",
                "DIRECCION",
                "MODELO",
                "SISTEMA",
                "THRESHOLD",
            ],
            ascending=[False, False, True, True, True, True, True],
        )

    ruta_salida = raiz_proyecto / "results" / "robust_summary.csv"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    robust.to_csv(ruta_salida, index=False)

    return {
        "summary": robust,
        "ruta_salida": ruta_salida,
        "total_configuraciones": total_configuraciones,
        "survives_1": int(robust["SURVIVES_1_BPS"].sum()) if not robust.empty else 0,
        "survives_2": int(robust["SURVIVES_2_BPS"].sum()) if not robust.empty else 0,
        "survives_5": int(robust["SURVIVES_5_BPS"].sum()) if not robust.empty else 0,
        "archivos_leidos": archivos_leidos,
        "archivos_faltantes": archivos_faltantes,
    }
