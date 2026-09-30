import zlib

import numpy as np
import pandas as pd


CAPITAL_INICIAL = 1000
RIESGO_POR_TRADE = 0.01
COSTOS_ANALISIS = [1, 2, 5]
COSTO_PRINCIPAL = 2
BOOTSTRAP_SIMULACIONES = 1000
COLUMNAS_ROBUST = [
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "TRADES",
    "SURVIVES_2_BPS",
]
COLUMNAS_TRADES = [
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "COST_BPS",
    "FOLD_SIGNAL",
    "FECHA_SIGNAL",
    "R",
]


def _validar_columnas(df, columnas, ruta):
    faltantes = [columna for columna in columnas if columna not in df.columns]
    if faltantes:
        raise ValueError(f"{ruta} no tiene columnas requeridas: {', '.join(faltantes)}")


def _normalizar_booleano(serie):
    if serie.dtype == bool:
        return serie

    return serie.astype(str).str.lower().isin(["true", "1", "yes"])


def _preparar_robust(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero python main.py ROBUST.")

    robust = pd.read_csv(ruta)
    _validar_columnas(robust, COLUMNAS_ROBUST, ruta)
    robust = robust.copy()
    robust["TRADES"] = pd.to_numeric(robust["TRADES"], errors="coerce")
    robust["THRESHOLD"] = pd.to_numeric(robust["THRESHOLD"], errors="coerce")
    robust["SURVIVES_2_BPS"] = _normalizar_booleano(robust["SURVIVES_2_BPS"])

    return robust[
        (robust["TRADES"] >= 25)
        & (robust["SURVIVES_2_BPS"])
    ].copy()


def _preparar_trades(ruta):
    trades = pd.read_csv(ruta)
    _validar_columnas(trades, COLUMNAS_TRADES, ruta)
    trades = trades.copy()
    trades["FECHA_SIGNAL"] = (
        pd.to_datetime(trades["FECHA_SIGNAL"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    for columna in ["THRESHOLD", "COST_BPS", "R"]:
        trades[columna] = pd.to_numeric(trades[columna], errors="coerce")

    return trades.dropna(subset=["FECHA_SIGNAL", "THRESHOLD", "COST_BPS", "R"])


def _filtrar_trades_config(trades, config, cost_bps):
    mascara = (
        (trades["ACTIVO"] == config["ACTIVO"])
        & (trades["DIRECCION"] == config["DIRECCION"])
        & (trades["MODELO"] == config["MODELO"])
        & (trades["SISTEMA"] == config["SISTEMA"])
        & (trades["COST_BPS"] == cost_bps)
        & np.isclose(trades["THRESHOLD"], config["THRESHOLD"])
    )
    return trades.loc[mascara].sort_values("FECHA_SIGNAL").copy()


def _max_drawdown_desde_r(r_values):
    capital = CAPITAL_INICIAL
    pico = capital
    max_drawdown = 0

    for r in r_values:
        capital += capital * RIESGO_POR_TRADE * r
        pico = max(pico, capital)
        if pico > 0:
            max_drawdown = max(max_drawdown, (pico - capital) / pico)

    return max_drawdown * 100


def _return_desde_r(r_values):
    capital = CAPITAL_INICIAL

    for r in r_values:
        capital += capital * RIESGO_POR_TRADE * r

    return ((capital - CAPITAL_INICIAL) / CAPITAL_INICIAL) * 100


def _metricas_r(trades):
    if trades.empty:
        return {
            "TRADES": 0,
            "WIN RATE": 0,
            "PROFIT FACTOR": 0,
            "EXPECTANCY R": 0,
            "R TOTAL": 0,
            "RETURN %": 0,
            "MAX DRAWDOWN %": 0,
        }

    r_values = trades["R"].astype(float)
    ganancias = r_values[r_values > 0]
    perdidas = r_values[r_values < 0]
    gross_profit = ganancias.sum()
    gross_loss = abs(perdidas.sum())

    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else 0
    else:
        profit_factor = gross_profit / gross_loss

    return {
        "TRADES": len(r_values),
        "WIN RATE": (len(ganancias) / len(r_values)) * 100,
        "PROFIT FACTOR": profit_factor,
        "EXPECTANCY R": r_values.mean(),
        "R TOTAL": r_values.sum(),
        "RETURN %": _return_desde_r(r_values),
        "MAX DRAWDOWN %": _max_drawdown_desde_r(r_values),
    }


def _periodos(trades, config, cost_bps):
    detalles = []
    datos = trades.copy()
    datos["YEAR"] = datos["FECHA_SIGNAL"].dt.year.astype(str)
    datos["QUARTER"] = datos["FECHA_SIGNAL"].dt.to_period("Q").astype(str)
    datos["FOLD"] = datos["FOLD_SIGNAL"].astype(str)

    for tipo, columna in [
        ("YEAR", "YEAR"),
        ("QUARTER", "QUARTER"),
        ("FOLD", "FOLD"),
    ]:
        for periodo, grupo in datos.groupby(columna, sort=True):
            metricas = _metricas_r(grupo)
            detalles.append(
                {
                    "ACTIVO": config["ACTIVO"],
                    "DIRECCION": config["DIRECCION"],
                    "MODELO": config["MODELO"],
                    "SISTEMA": config["SISTEMA"],
                    "THRESHOLD": config["THRESHOLD"],
                    "COST_BPS": cost_bps,
                    "PERIOD_TYPE": tipo,
                    "PERIOD": periodo,
                    **metricas,
                }
            )

    return detalles


def _streaks(r_values):
    max_win = 0
    max_loss = 0
    actual_win = 0
    actual_loss = 0

    for r in r_values:
        if r > 0:
            actual_win += 1
            actual_loss = 0
        elif r <= 0:
            actual_loss += 1
            actual_win = 0

        max_win = max(max_win, actual_win)
        max_loss = max(max_loss, actual_loss)

    return max_win, max_loss


def _bootstrap(r_values, config_key):
    if len(r_values) < 30:
        return {
            "EXPECTANCY_CI_LOW": np.nan,
            "EXPECTANCY_CI_HIGH": np.nan,
            "R_TOTAL_CI_LOW": np.nan,
            "R_TOTAL_CI_HIGH": np.nan,
            "BOOTSTRAP_DD_MEDIAN": np.nan,
            "BOOTSTRAP_DD_95": np.nan,
        }

    seed = 42 + zlib.crc32(config_key.encode("utf-8"))
    rng = np.random.default_rng(seed)
    r_values = np.asarray(r_values, dtype=float)
    expectancy = []
    r_total = []
    drawdowns = []

    for _ in range(BOOTSTRAP_SIMULACIONES):
        muestra = rng.choice(r_values, size=len(r_values), replace=True)
        expectancy.append(muestra.mean())
        r_total.append(muestra.sum())
        drawdowns.append(_max_drawdown_desde_r(muestra))

    return {
        "EXPECTANCY_CI_LOW": np.percentile(expectancy, 2.5),
        "EXPECTANCY_CI_HIGH": np.percentile(expectancy, 97.5),
        "R_TOTAL_CI_LOW": np.percentile(r_total, 2.5),
        "R_TOTAL_CI_HIGH": np.percentile(r_total, 97.5),
        "BOOTSTRAP_DD_MEDIAN": np.percentile(drawdowns, 50),
        "BOOTSTRAP_DD_95": np.percentile(drawdowns, 95),
    }


def _consistencia_trimestral(trades):
    datos = trades.copy()
    datos["QUARTER"] = datos["FECHA_SIGNAL"].dt.to_period("Q").astype(str)
    r_por_periodo = datos.groupby("QUARTER")["R"].sum()
    r_total = datos["R"].sum()
    positivos = r_por_periodo[r_por_periodo > 0]

    mejor = r_por_periodo.max() if not r_por_periodo.empty else 0
    peor = r_por_periodo.min() if not r_por_periodo.empty else 0
    profit_concentration = 0
    if positivos.sum() > 0 and not positivos.empty:
        profit_concentration = (positivos.max() / positivos.sum()) * 100

    porcentaje_mejor = 0
    if r_total != 0:
        porcentaje_mejor = (mejor / r_total) * 100

    return {
        "NUM_PERIODOS": len(r_por_periodo),
        "PERIODOS_POSITIVOS": int((r_por_periodo > 0).sum()),
        "PERIODOS_NEGATIVOS": int((r_por_periodo < 0).sum()),
        "PORCENTAJE_PERIODOS_POSITIVOS": (
            ((r_por_periodo > 0).sum() / len(r_por_periodo)) * 100
            if len(r_por_periodo)
            else 0
        ),
        "R_MEJOR_PERIODO": mejor,
        "R_PEOR_PERIODO": peor,
        "PORCENTAJE_R_MEJOR_PERIODO": porcentaje_mejor,
        "PROFIT_CONCENTRATION": profit_concentration,
    }


def analizar_estabilidad(raiz_proyecto):
    robust = _preparar_robust(raiz_proyecto / "results" / "robust_summary.csv")
    resumen = []
    detalles_periodo = []
    archivos_faltantes = []

    for _, config in robust.iterrows():
        ruta_trades = (
            raiz_proyecto
            / "results"
            / str(config["ACTIVO"]).lower()
            / "oof_backtest_trades.csv"
        )

        if not ruta_trades.exists():
            archivos_faltantes.append(str(ruta_trades))
            continue

        trades_activo = _preparar_trades(ruta_trades)

        for cost_bps in COSTOS_ANALISIS:
            trades_costo = _filtrar_trades_config(trades_activo, config, cost_bps)
            detalles_periodo.extend(_periodos(trades_costo, config, cost_bps))

        trades = _filtrar_trades_config(trades_activo, config, COSTO_PRINCIPAL)
        if trades.empty:
            continue

        metricas = _metricas_r(trades)
        consistencia = _consistencia_trimestral(trades)
        max_win, max_loss = _streaks(trades["R"].astype(float).tolist())
        clave = (
            f"{config['ACTIVO']}|{config['DIRECCION']}|{config['MODELO']}|"
            f"{config['SISTEMA']}|{config['THRESHOLD']}|{COSTO_PRINCIPAL}"
        )
        bootstrap = _bootstrap(trades["R"].astype(float).to_numpy(), clave)

        resumen.append(
            {
                "ACTIVO": config["ACTIVO"],
                "DIRECCION": config["DIRECCION"],
                "MODELO": config["MODELO"],
                "SISTEMA": config["SISTEMA"],
                "THRESHOLD": config["THRESHOLD"],
                "TRADES": metricas["TRADES"],
                "PF_2BPS": metricas["PROFIT FACTOR"],
                "EXPECT_2BPS": metricas["EXPECTANCY R"],
                "PERIODOS_POSITIVOS": consistencia["PERIODOS_POSITIVOS"],
                "NUM_PERIODOS": consistencia["NUM_PERIODOS"],
                "% PERIODOS POSITIVOS": consistencia["PORCENTAJE_PERIODOS_POSITIVOS"],
                "MEJOR PERIODO R": consistencia["R_MEJOR_PERIODO"],
                "PEOR PERIODO R": consistencia["R_PEOR_PERIODO"],
                "PORCENTAJE_R_MEJOR_PERIODO": consistencia[
                    "PORCENTAJE_R_MEJOR_PERIODO"
                ],
                "PROFIT_CONCENTRATION": consistencia["PROFIT_CONCENTRATION"],
                "MAX_WIN_STREAK": max_win,
                "MAX_LOSS_STREAK": max_loss,
                "MAX_CONSECUTIVE_LOSSES": max_loss,
                **bootstrap,
            }
        )

    summary = pd.DataFrame(resumen)
    by_period = pd.DataFrame(detalles_periodo)

    ruta_summary = raiz_proyecto / "results" / "stability_summary.csv"
    ruta_period = raiz_proyecto / "results" / "stability_by_period.csv"
    summary.to_csv(ruta_summary, index=False)
    by_period.to_csv(ruta_period, index=False)

    return {
        "summary": summary,
        "by_period": by_period,
        "ruta_summary": ruta_summary,
        "ruta_period": ruta_period,
        "archivos_faltantes": archivos_faltantes,
    }
