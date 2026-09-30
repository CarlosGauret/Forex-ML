from pathlib import Path

import pandas as pd

from config import PROB_THRESHOLDS
from src import usdcad_research as shared_research
from src.model import FEATURES_ML, TARGET_HORIZON, _agregar_base_signal


ASSET = "NZDUSD"
MAIN_COST_BPS = 2


def _root(raiz_proyecto=None):
    return Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]


def _research_dir(raiz):
    path = raiz / "results" / "v2" / "nzdusd"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _with_nzdusd_asset(function, *args, **kwargs):
    original_asset = shared_research.ASSET
    try:
        shared_research.ASSET = ASSET
        return function(*args, **kwargs)
    finally:
        shared_research.ASSET = original_asset


def _create_dataset(raiz):
    from config import ACTIVOS
    from src.data import _descargar_y_preparar
    from src.features import crear_dataset_ml
    from src.indicators import calcular_indicadores

    raw = _descargar_y_preparar(ACTIVOS[ASSET]["ticker"], ASSET)
    indicators = calcular_indicadores(raw)
    dataset = crear_dataset_ml(indicators)
    return (
        dataset,
        {
            "source": "Yahoo Finance NZDUSD=X",
            "interval": raw.attrs.get("intervalo", "1h"),
            "downloaded_period": raw.attrs.get("periodo_descargado", "N/A"),
            "created_now": True,
        },
    )


def _read_or_create_dataset(raiz):
    data, source = _create_dataset(raiz)

    required = [
        "fecha",
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "TARGET_LONG",
        "TARGET_SHORT",
    ] + FEATURES_ML
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset NZDUSD incompleto. Faltan: {', '.join(missing)}")

    data["fecha"] = pd.to_datetime(data["fecha"], utc=True).dt.tz_convert(None)
    data = data.sort_values("fecha").reset_index(drop=True)
    for column in ["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    data = data.dropna(subset=["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML)
    return _agregar_base_signal(data), source


def _audit_existing_models(raiz, data):
    rows = []
    models_dir = raiz / "models" / "nzdusd"
    train_end_idx = max(0, int(len(data) * 0.70) - TARGET_HORIZON - 1)
    train_start = data["fecha"].iloc[0] if not data.empty else ""
    train_end = data["fecha"].iloc[train_end_idx] if not data.empty else ""

    for direction in shared_research.DIRECTIONS:
        path = models_dir / f"random_forest_{direction.lower()}.pkl"
        row = {
            "ACTIVO": ASSET,
            "DIRECCION": direction,
            "RUTA": str(path),
            "EXISTE": path.exists(),
            "FECHA_ARCHIVO": "",
            "ALGORITMO": "",
            "FEATURES": ",".join(FEATURES_ML),
            "N_FEATURES_MODELO": "",
            "THRESHOLDS_DISPONIBLES": ",".join(str(item) for item in PROB_THRESHOLDS),
            "PERIODO_ENTRENAMIENTO_INFERIDO_INICIAL": train_start,
            "PERIODO_ENTRENAMIENTO_INFERIDO_FINAL": train_end,
            "PERIODO_DATASET_INICIAL": data["fecha"].min(),
            "PERIODO_DATASET_FINAL": data["fecha"].max(),
            "COMPATIBLE_METODOLOGIA_ACTUAL": False,
            "LEAKAGE_LOOKAHEAD_CONOCIDO": False,
            "NOTAS": "",
        }
        if path.exists():
            row["FECHA_ARCHIVO"] = pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()
            try:
                import joblib

                model = joblib.load(path)
                row["ALGORITMO"] = type(model).__name__
                n_features = getattr(model, "n_features_in_", "")
                row["N_FEATURES_MODELO"] = n_features
                feature_names = list(getattr(model, "feature_names_in_", []))
                feature_ok = (not feature_names and n_features == len(FEATURES_ML)) or feature_names == FEATURES_ML
                row["COMPATIBLE_METODOLOGIA_ACTUAL"] = bool(
                    row["ALGORITMO"] == "RandomForestClassifier" and feature_ok
                )
                row["NOTAS"] = (
                    "Modelo previo localizado; utilizable como referencia. "
                    "No se sobrescribe ni se declara validado."
                )
            except Exception as error:
                row["NOTAS"] = f"No se pudo cargar modelo: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_candidate_safety_rules(robustness):
    return _with_nzdusd_asset(shared_research._apply_candidate_safety_rules, robustness)


def run_nzdusd_research(raiz_proyecto=None):
    raiz = _root(raiz_proyecto)
    out = _research_dir(raiz)
    data, data_source = _read_or_create_dataset(raiz)
    audit = _audit_existing_models(raiz, data)
    walkforward_raw, oof, oof_summary, trades, robustness = _with_nzdusd_asset(
        shared_research._run_with_usdcad_asset,
        data,
    )
    robustness = _apply_candidate_safety_rules(robustness)
    walkforward = _with_nzdusd_asset(shared_research._walkforward_economics, walkforward_raw, oof)
    broker, sizing_validation = _with_nzdusd_asset(shared_research._broker_aware, trades)

    audit.to_csv(out / "nzdusd_existing_models_audit.csv", index=False)
    walkforward.to_csv(out / "nzdusd_walkforward_summary.csv", index=False)
    oof.to_csv(out / "nzdusd_oof_predictions.csv", index=False)
    oof_summary.to_csv(out / "nzdusd_oof_summary.csv", index=False)
    robustness.to_csv(out / "nzdusd_robustness.csv", index=False)
    trades.to_csv(out / "nzdusd_trades.csv", index=False)
    broker.to_csv(out / "nzdusd_broker_aware.csv", index=False)

    candidates = robustness[
        (robustness["COST_BPS"] == MAIN_COST_BPS)
        & (robustness["CLASIFICACION"] == "CANDIDATO A FORWARD TEST")
    ].copy()
    discarded = robustness[
        (robustness["COST_BPS"] == MAIN_COST_BPS)
        & (robustness["CLASIFICACION"] != "CANDIDATO A FORWARD TEST")
    ].copy()

    return {
        "output_dir": out,
        "data_period": {
            "start": data["fecha"].min(),
            "end": data["fecha"].max(),
            "rows": len(data),
            "source": data_source["source"],
            "interval": data_source["interval"],
            "downloaded_period": data_source["downloaded_period"],
            "created_now": data_source["created_now"],
        },
        "audit": audit,
        "walkforward": walkforward,
        "walkforward_raw": walkforward_raw,
        "oof_summary": oof_summary,
        "robustness": robustness,
        "broker_aware": broker,
        "sizing_validation": sizing_validation,
        "candidates": candidates,
        "discarded": discarded,
        "trades": trades,
    }
