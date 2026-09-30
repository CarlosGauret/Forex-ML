from pathlib import Path

import pandas as pd

from config import COST_BPS
from config_paper import PAPER_CONFIGS
from src.backtest_v2 import (
    CAPITALS,
    MAX_HOLD_BARS,
    RISK_PER_TRADE,
    SL_TP_MODES,
    BrokerMetadata,
    DEFAULT_GOLD_BROKER_METADATA,
    _base_temporal,
    _ejecutar_escenario,
    cargar_gold_oof,
    construir_entradas_base,
    obtener_gold_broker_metadata,
)


EMBARGO_BARS = 24
N_FOLDS = 5
MIN_TRAIN_FRACTION = 0.35
BOOTSTRAP_ITERATIONS = 500
BOOTSTRAP_SEED = 42

FOLD_COLUMNS = [
    "FOLD",
    "TRAIN_START",
    "TRAIN_END",
    "EMBARGO_START",
    "EMBARGO_END",
    "TEST_START",
    "TEST_END",
    "TRAIN_BARS",
    "EMBARGO_BARS",
    "TEST_BARS",
    "ACTIVO",
    "CONFIG_ID",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "SL_TP_MODE",
    "COST_BPS",
    "CAPITAL_INICIAL",
    "BROKER_AWARE",
    "BROKER_METADATA_SOURCE",
    "TOTAL_ENTRADAS_BASE",
    "TOTAL_TRADES",
    "WINS",
    "LOSSES",
    "WIN_RATE",
    "PROFIT_FACTOR",
    "EXPECTANCY",
    "AVERAGE_R",
    "R_ACUMULADO",
    "RETURN_PCT",
    "MAX_DRAWDOWN",
    "MAX_LOSING_STREAK",
    "CAPITAL_FINAL",
    "NO_EJECUTABLES_BROKER_MIN",
    "SKIPPED_SLTP",
]

SUMMARY_COLUMNS = [
    "ACTIVO",
    "CONFIG_ID",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "SL_TP_MODE",
    "COST_BPS",
    "CAPITAL_INICIAL",
    "BROKER_AWARE",
    "BROKER_METADATA_SOURCE",
    "FOLDS",
    "TOTAL_TRADES",
    "TOTAL_WINS",
    "TOTAL_LOSSES",
    "MEAN_EXPECTANCY",
    "MEDIAN_EXPECTANCY",
    "STD_EXPECTANCY",
    "MIN_EXPECTANCY",
    "MAX_EXPECTANCY",
    "MEAN_R_TOTAL",
    "MEDIAN_R_TOTAL",
    "STD_R_TOTAL",
    "MIN_R_TOTAL",
    "MAX_R_TOTAL",
    "MEAN_RETURN_PCT",
    "MEDIAN_RETURN_PCT",
    "STD_RETURN_PCT",
    "MIN_RETURN_PCT",
    "MAX_RETURN_PCT",
    "MEAN_MAX_DRAWDOWN",
    "MEDIAN_MAX_DRAWDOWN",
    "STD_MAX_DRAWDOWN",
    "MIN_MAX_DRAWDOWN",
    "MAX_MAX_DRAWDOWN",
    "POSITIVE_FOLDS",
    "POSITIVE_FOLDS_PCT",
    "EXPECTANCY_CI95_LOW",
    "EXPECTANCY_CI95_HIGH",
    "DRAWDOWN_95",
    "CLASSIFICATION",
    "NO_EJECUTABLES_BROKER_MIN",
    "SKIPPED_SLTP",
]

TRADE_EXTRA_COLUMNS = [
    "FOLD",
    "TRAIN_END",
    "TEST_START",
    "TEST_END",
]


def _normalizar_oof(oof_predictions):
    datos = oof_predictions.copy()
    datos["FECHA"] = pd.to_datetime(datos["FECHA"], utc=True, errors="coerce").dt.tz_convert(None)
    datos = datos.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    return datos


def generar_folds_expanding(
    fechas,
    n_folds=N_FOLDS,
    min_train_fraction=MIN_TRAIN_FRACTION,
    embargo_bars=EMBARGO_BARS,
):
    fechas = pd.Series(pd.to_datetime(fechas)).drop_duplicates().sort_values().reset_index(drop=True)
    total = len(fechas)
    min_train = max(int(total * min_train_fraction), embargo_bars * 2)
    disponible = total - min_train - embargo_bars
    if disponible < n_folds:
        raise ValueError("No hay suficientes barras para crear folds walk-forward V2.")

    test_size = max(1, disponible // n_folds)
    folds = []
    for fold in range(1, n_folds + 1):
        train_end_idx = min_train + ((fold - 1) * test_size) - 1
        embargo_start_idx = train_end_idx + 1
        embargo_end_idx = min(train_end_idx + embargo_bars, total - 1)
        test_start_idx = embargo_end_idx + 1
        if test_start_idx >= total:
            break

        test_end_idx = (
            total - 1
            if fold == n_folds
            else min(test_start_idx + test_size - 1, total - 1)
        )
        if test_end_idx < test_start_idx:
            continue

        folds.append(
            {
                "FOLD": fold,
                "TRAIN_START_IDX": 0,
                "TRAIN_END_IDX": train_end_idx,
                "EMBARGO_START_IDX": embargo_start_idx,
                "EMBARGO_END_IDX": embargo_end_idx,
                "TEST_START_IDX": test_start_idx,
                "TEST_END_IDX": test_end_idx,
                "TRAIN_START": fechas.iloc[0],
                "TRAIN_END": fechas.iloc[train_end_idx],
                "EMBARGO_START": fechas.iloc[embargo_start_idx],
                "EMBARGO_END": fechas.iloc[embargo_end_idx],
                "TEST_START": fechas.iloc[test_start_idx],
                "TEST_END": fechas.iloc[test_end_idx],
                "TRAIN_BARS": train_end_idx + 1,
                "EMBARGO_BARS": embargo_end_idx - embargo_start_idx + 1,
                "TEST_BARS": test_end_idx - test_start_idx + 1,
            }
        )

    return folds


def _entradas_fold(entradas, fold):
    return [
        entrada
        for entrada in entradas
        if fold["TEST_START_IDX"] <= entrada["SIGNAL_INDEX"] <= fold["TEST_END_IDX"]
        and entrada["ENTRY_INDEX"] <= fold["TEST_END_IDX"]
    ]


def _escenario_cols(row):
    return [
        "ACTIVO",
        "CONFIG_ID",
        "DIRECCION",
        "MODELO",
        "SISTEMA",
        "THRESHOLD",
        "SL_TP_MODE",
        "COST_BPS",
        "CAPITAL_INICIAL",
        "BROKER_AWARE",
        "BROKER_METADATA_SOURCE",
    ]


def _std(series):
    valor = pd.to_numeric(series, errors="coerce").std(ddof=1)
    return 0.0 if pd.isna(valor) else float(valor)


def _bootstrap_metrics(r_values, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    valores = pd.Series(r_values, dtype="float64").dropna().to_numpy()
    if len(valores) < 2:
        return None, None, None

    import random

    rng = random.Random(seed)
    expectancies = []
    drawdowns = []
    for _ in range(iterations):
        muestra = [valores[rng.randrange(len(valores))] for _ in range(len(valores))]
        expectancies.append(sum(muestra) / len(muestra))
        capital = 1.0
        pico = capital
        max_dd = 0.0
        for r_value in muestra:
            capital += RISK_PER_TRADE * r_value
            pico = max(pico, capital)
            if pico > 0:
                max_dd = max(max_dd, ((pico - capital) / pico) * 100)
        drawdowns.append(max_dd)

    exp = pd.Series(expectancies)
    dd = pd.Series(drawdowns)
    return (
        float(exp.quantile(0.025)),
        float(exp.quantile(0.975)),
        float(dd.quantile(0.95)),
    )


def _clasificar(folds_df, trades_total):
    folds_con_trades = folds_df[folds_df["TOTAL_TRADES"] > 0]
    if len(folds_con_trades) < 3 or trades_total < 20:
        return "MUESTRA INSUFICIENTE"

    positive_pct = (folds_con_trades["R_ACUMULADO"] > 0).mean() * 100
    median_expectancy = folds_con_trades["EXPECTANCY"].median()
    if positive_pct >= 70 and median_expectancy > 0:
        return "ESTABLE"
    if positive_pct >= 40:
        return "MIXTO"
    return "INESTABLE"


def _agregar_summary(folds_df, trades_df):
    if folds_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    group_cols = _escenario_cols(folds_df.iloc[0])
    rows = []
    for keys, grupo in folds_df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        trades_grupo = trades_df
        for col, value in base.items():
            trades_grupo = trades_grupo[trades_grupo[col] == value]
        trades_ejecutados = trades_grupo[trades_grupo["RESULTADO"].isin(["WIN", "LOSS"])]
        r_values = pd.to_numeric(trades_ejecutados["R_NETO"], errors="coerce").dropna()
        ci_low, ci_high, dd95 = _bootstrap_metrics(r_values)
        positive_folds = int((grupo["R_ACUMULADO"] > 0).sum())
        folds = len(grupo)
        total_trades = int(grupo["TOTAL_TRADES"].sum())
        row = {
            **base,
            "FOLDS": folds,
            "TOTAL_TRADES": total_trades,
            "TOTAL_WINS": int(grupo["WINS"].sum()),
            "TOTAL_LOSSES": int(grupo["LOSSES"].sum()),
            "MEAN_EXPECTANCY": float(grupo["EXPECTANCY"].mean()),
            "MEDIAN_EXPECTANCY": float(grupo["EXPECTANCY"].median()),
            "STD_EXPECTANCY": _std(grupo["EXPECTANCY"]),
            "MIN_EXPECTANCY": float(grupo["EXPECTANCY"].min()),
            "MAX_EXPECTANCY": float(grupo["EXPECTANCY"].max()),
            "MEAN_R_TOTAL": float(grupo["R_ACUMULADO"].mean()),
            "MEDIAN_R_TOTAL": float(grupo["R_ACUMULADO"].median()),
            "STD_R_TOTAL": _std(grupo["R_ACUMULADO"]),
            "MIN_R_TOTAL": float(grupo["R_ACUMULADO"].min()),
            "MAX_R_TOTAL": float(grupo["R_ACUMULADO"].max()),
            "MEAN_RETURN_PCT": float(grupo["RETURN_PCT"].mean()),
            "MEDIAN_RETURN_PCT": float(grupo["RETURN_PCT"].median()),
            "STD_RETURN_PCT": _std(grupo["RETURN_PCT"]),
            "MIN_RETURN_PCT": float(grupo["RETURN_PCT"].min()),
            "MAX_RETURN_PCT": float(grupo["RETURN_PCT"].max()),
            "MEAN_MAX_DRAWDOWN": float(grupo["MAX_DRAWDOWN"].mean()),
            "MEDIAN_MAX_DRAWDOWN": float(grupo["MAX_DRAWDOWN"].median()),
            "STD_MAX_DRAWDOWN": _std(grupo["MAX_DRAWDOWN"]),
            "MIN_MAX_DRAWDOWN": float(grupo["MAX_DRAWDOWN"].min()),
            "MAX_MAX_DRAWDOWN": float(grupo["MAX_DRAWDOWN"].max()),
            "POSITIVE_FOLDS": positive_folds,
            "POSITIVE_FOLDS_PCT": (positive_folds / folds) * 100 if folds else 0.0,
            "EXPECTANCY_CI95_LOW": ci_low,
            "EXPECTANCY_CI95_HIGH": ci_high,
            "DRAWDOWN_95": dd95,
            "CLASSIFICATION": _clasificar(grupo, total_trades),
            "NO_EJECUTABLES_BROKER_MIN": int(grupo["NO_EJECUTABLES_BROKER_MIN"].sum()),
            "SKIPPED_SLTP": int(grupo["SKIPPED_SLTP"].sum()),
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _output_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"No se sobrescribe resultado existente: {path}")
    return path


def ejecutar_walkforward_v2_gold(
    raiz_proyecto,
    oof_predictions=None,
    broker_metadata=None,
    costs=None,
    capitals=None,
    n_folds=N_FOLDS,
    embargo_bars=EMBARGO_BARS,
):
    raiz = Path(raiz_proyecto)
    oof = cargar_gold_oof(raiz) if oof_predictions is None else _normalizar_oof(oof_predictions)
    broker_metadata = broker_metadata or obtener_gold_broker_metadata()
    costs = COST_BPS if costs is None else costs
    capitals = CAPITALS if capitals is None else capitals

    configs = [
        config
        for config in PAPER_CONFIGS
        if config["ACTIVO"] == "GOLD"
        and config["DIRECCION"] == "LONG"
        and config["MODELO"] == "LOGISTIC"
        and config["SISTEMA"] == "BASE_PLUS_ML"
    ]

    fold_rows = []
    trade_rows = []
    fold_defs_by_config = {}

    for config in configs:
        datos = _base_temporal(oof, config)
        folds = generar_folds_expanding(
            datos["FECHA"],
            n_folds=n_folds,
            embargo_bars=embargo_bars,
        )
        fold_defs_by_config[config["CONFIG_ID"]] = folds
        entradas = construir_entradas_base(
            datos,
            config,
            config.get("MAX_HOLD_BARS", MAX_HOLD_BARS),
        )

        for fold in folds:
            entradas_fold = _entradas_fold(entradas, fold)
            datos_fold = datos.iloc[: fold["TEST_END_IDX"] + 1].copy()
            for mode in SL_TP_MODES:
                for cost_bps in costs:
                    for capital in capitals:
                        for broker_aware in (False, True):
                            filas, metricas = _ejecutar_escenario(
                                datos=datos_fold,
                                entradas=entradas_fold,
                                config=config,
                                mode=mode,
                                cost_bps=cost_bps,
                                capital_inicial=capital,
                                broker_metadata=broker_metadata,
                                broker_aware=broker_aware,
                            )
                            fold_meta = {
                                "FOLD": fold["FOLD"],
                                "TRAIN_START": fold["TRAIN_START"],
                                "TRAIN_END": fold["TRAIN_END"],
                                "EMBARGO_START": fold["EMBARGO_START"],
                                "EMBARGO_END": fold["EMBARGO_END"],
                                "TEST_START": fold["TEST_START"],
                                "TEST_END": fold["TEST_END"],
                                "TRAIN_BARS": fold["TRAIN_BARS"],
                                "EMBARGO_BARS": fold["EMBARGO_BARS"],
                                "TEST_BARS": fold["TEST_BARS"],
                            }
                            fold_rows.append({**fold_meta, **metricas})
                            for fila in filas:
                                trade_rows.append(
                                    {
                                        **{
                                            "FOLD": fold["FOLD"],
                                            "TRAIN_END": fold["TRAIN_END"],
                                            "TEST_START": fold["TEST_START"],
                                            "TEST_END": fold["TEST_END"],
                                        },
                                        **fila,
                                    }
                                )

    folds_df = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    trades_df = pd.DataFrame(trade_rows)
    summary_df = _agregar_summary(folds_df, trades_df)

    carpeta = raiz / "results" / "v2"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_summary = _output_path(carpeta / "walkforward_gold_summary.csv")
    ruta_folds = _output_path(carpeta / "walkforward_gold_folds.csv")
    ruta_trades = _output_path(carpeta / "walkforward_gold_trades.csv")

    summary_df.to_csv(ruta_summary, index=False)
    folds_df.to_csv(ruta_folds, index=False)
    trades_df.to_csv(ruta_trades, index=False)

    return {
        "summary": summary_df,
        "folds": folds_df,
        "trades": trades_df,
        "fold_defs_by_config": fold_defs_by_config,
        "ruta_summary": ruta_summary,
        "ruta_folds": ruta_folds,
        "ruta_trades": ruta_trades,
        "broker_metadata": broker_metadata,
    }
