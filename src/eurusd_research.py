from pathlib import Path

import numpy as np
import pandas as pd

from config import COST_BPS, PROB_THRESHOLDS
from src.model import FEATURES_ML, TARGET_HORIZON, _agregar_base_signal, _crear_modelo
from src.risk import RiskLimits, RiskManager


ASSET = "EURUSD"
MODELS = ("LOGISTIC", "RF_SIMPLE", "RF_ACTUAL")
DIRECTIONS = ("LONG", "SHORT")
SYSTEMS = ("ML_ONLY", "BASE_PLUS_ML")
CAPITALS = (100, 500, 1000, 5000)
MAIN_COST_BPS = 2
N_FOLDS = 5
RISK_PER_TRADE = 0.01


def _root(raiz_proyecto=None):
    return Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]


def _research_dir(raiz):
    path = raiz / "results" / "v2" / "eurusd"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_dataset(raiz):
    path = raiz / "data" / "eurusd" / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe dataset EURUSD: {path}")

    data = pd.read_csv(path)
    required = ["fecha", "Open", "High", "Low", "Close", "ATR", "TARGET_LONG", "TARGET_SHORT"] + FEATURES_ML
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset EURUSD incompleto. Faltan: {', '.join(missing)}")

    data["fecha"] = pd.to_datetime(data["fecha"])
    data = data.sort_values("fecha").reset_index(drop=True)
    for column in ["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    data = data.dropna(subset=["Open", "High", "Low", "Close", "ATR"] + FEATURES_ML)
    return _agregar_base_signal(data)


def _target_for_direction(direction):
    return "TARGET_LONG" if direction == "LONG" else "TARGET_SHORT"


def _prepare_direction_data(data, direction):
    target = _target_for_direction(direction)
    prepared = data.dropna(subset=[target]).copy()
    prepared[target] = prepared[target].astype(int)
    if prepared[target].nunique() < 2:
        raise ValueError(f"{ASSET} {direction} necesita dos clases.")
    return prepared.reset_index(drop=True)


def _folds(data):
    total = len(data)
    block = total // (N_FOLDS + 1)
    if block <= TARGET_HORIZON:
        raise ValueError("No hay datos suficientes para walk-forward EURUSD.")

    result = []
    for fold in range(1, N_FOLDS + 1):
        test_start = block * fold
        test_end = block * (fold + 1) if fold < N_FOLDS else total
        train_end = test_start - TARGET_HORIZON
        train = data.iloc[:train_end].copy()
        test = data.iloc[test_start:test_end].copy()
        result.append(
            {
                "fold": fold,
                "train": train,
                "test": test,
                "embargo_bars": TARGET_HORIZON,
            }
        )
    return result


def _positive_probability(model, x):
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return pd.Series([0.0] * len(x), index=x.index)
    return pd.Series(probabilities[:, classes.index(1)], index=x.index)


def _roc_auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score

    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _classification_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "ROC_AUC": _roc_auc(y_true, y_prob),
        "ACCURACY": float(accuracy_score(y_true, y_pred)),
        "PRECISION": float(precision_score(y_true, y_pred, zero_division=0)),
        "RECALL": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _run_walk_forward(data):
    rows = []
    oof_rows = []

    for direction in DIRECTIONS:
        direction_data = _prepare_direction_data(data, direction)
        target = _target_for_direction(direction)
        for fold in _folds(direction_data):
            train = fold["train"]
            test = fold["test"]
            for model_name in MODELS:
                row = {
                    "ACTIVO": ASSET,
                    "DIRECCION": direction,
                    "MODELO": model_name,
                    "FOLD": fold["fold"],
                    "N_TRAIN": len(train),
                    "N_TEST": len(test),
                    "TRAIN_FECHA_INICIAL": train["fecha"].iloc[0],
                    "TRAIN_FECHA_FINAL": train["fecha"].iloc[-1],
                    "TEST_FECHA_INICIAL": test["fecha"].iloc[0],
                    "TEST_FECHA_FINAL": test["fecha"].iloc[-1],
                    "EMBARGO_BARS": fold["embargo_bars"],
                    "ERROR": "",
                }
                if train[target].nunique() < 2:
                    row["ERROR"] = "TRAIN con una sola clase"
                    rows.append(row)
                    continue

                model = _crear_modelo(model_name)
                model.fit(train[FEATURES_ML], train[target])
                y_prob = _positive_probability(model, test[FEATURES_ML])
                y_pred = model.predict(test[FEATURES_ML])
                row.update(_classification_metrics(test[target], y_pred, y_prob))
                rows.append(row)

                for position, (_, item) in enumerate(test.iterrows()):
                    oof_rows.append(
                        {
                            "FECHA": item["fecha"],
                            "ACTIVO": ASSET,
                            "DIRECCION": direction,
                            "MODELO": model_name,
                            "FOLD": fold["fold"],
                            "CLOSE": item["Close"],
                            "OPEN": item["Open"],
                            "HIGH": item["High"],
                            "LOW": item["Low"],
                            "ATR": item["ATR"],
                            "PROBABILIDAD": float(y_prob.iloc[position]),
                            "TARGET_REAL": item[target],
                            "BASE_SIGNAL": item["BASE_SIGNAL"],
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(oof_rows)


def _has_signal(row, system, direction, threshold):
    probability_ok = float(row["PROBABILIDAD"]) >= float(threshold)
    base_required = "BUY" if direction == "LONG" else "SELL"
    base_ok = row["BASE_SIGNAL"] == base_required
    if system == "ML_ONLY":
        return probability_ok
    if system == "BASE_PLUS_ML":
        return probability_ok and base_ok
    raise ValueError(f"Sistema no soportado: {system}")


def _apply_costs(entry, exit_price, raw_r, atr, cost_bps):
    cost_per_unit = (abs(entry) + abs(exit_price)) * (cost_bps / 10000)
    return raw_r - (cost_per_unit / atr)


def _resolve_trade(data, signal_idx, direction, cost_bps):
    entry_idx = signal_idx + 1
    if entry_idx >= len(data):
        return None

    signal = data.iloc[signal_idx]
    entry_row = data.iloc[entry_idx]
    entry = float(entry_row["OPEN"])
    atr = float(signal["ATR"])
    if atr <= 0:
        return None

    if direction == "LONG":
        stop_loss = entry - atr
        take_profit = entry + (2 * atr)
    else:
        stop_loss = entry + atr
        take_profit = entry - (2 * atr)

    exit_idx = None
    exit_price = None
    exit_reason = None
    limit = min(entry_idx + TARGET_HORIZON - 1, len(data) - 1)

    for idx in range(entry_idx, limit + 1):
        row = data.iloc[idx]
        high = float(row["HIGH"])
        low = float(row["LOW"])
        if direction == "LONG":
            touches_stop = low <= stop_loss
            touches_tp = high >= take_profit
        else:
            touches_stop = high >= stop_loss
            touches_tp = low <= take_profit

        if touches_stop:
            exit_idx = idx
            exit_price = stop_loss
            exit_reason = "SL"
            break
        if touches_tp:
            exit_idx = idx
            exit_price = take_profit
            exit_reason = "TP"
            break

    if exit_idx is None:
        exit_idx = limit
        exit_price = float(data.iloc[exit_idx]["CLOSE"])
        exit_reason = "TIME"

    raw_r = (exit_price - entry) / atr if direction == "LONG" else (entry - exit_price) / atr
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "r": _apply_costs(entry, exit_price, raw_r, atr, cost_bps),
    }


def _max_drawdown_pct(values):
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd * 100


def _run_oof_backtest(oof):
    summary_rows = []
    trade_rows = []

    for direction in DIRECTIONS:
        for model_name in MODELS:
            base = oof[(oof["DIRECCION"] == direction) & (oof["MODELO"] == model_name)]
            base = base.sort_values("FECHA").reset_index(drop=True)
            for system in SYSTEMS:
                for threshold in PROB_THRESHOLDS:
                    for cost_bps in COST_BPS:
                        capital = 1000.0
                        capital_curve = [capital]
                        scenario_trades = []
                        idx = 0
                        while idx < len(base) - 1:
                            signal = base.iloc[idx]
                            if not _has_signal(signal, system, direction, threshold):
                                idx += 1
                                continue

                            trade = _resolve_trade(base, idx, direction, cost_bps)
                            if trade is None:
                                idx += 1
                                continue

                            pnl = capital * RISK_PER_TRADE * trade["r"]
                            capital += pnl
                            capital_curve.append(capital)
                            entry_row = base.iloc[trade["entry_idx"]]
                            exit_row = base.iloc[trade["exit_idx"]]
                            scenario_trades.append(
                                {
                                    "ACTIVO": ASSET,
                                    "DIRECCION": direction,
                                    "MODELO": model_name,
                                    "SISTEMA": system,
                                    "THRESHOLD": threshold,
                                    "COST_BPS": cost_bps,
                                    "FOLD_SIGNAL": signal["FOLD"],
                                    "FECHA_SIGNAL": signal["FECHA"],
                                    "FECHA_ENTRADA": entry_row["FECHA"],
                                    "FECHA_SALIDA": exit_row["FECHA"],
                                    "ENTRY": trade["entry"],
                                    "STOP_LOSS": trade["stop_loss"],
                                    "TAKE_PROFIT": trade["take_profit"],
                                    "EXIT": trade["exit_price"],
                                    "MOTIVO_SALIDA": trade["exit_reason"],
                                    "PROBABILIDAD": signal["PROBABILIDAD"],
                                    "TARGET_REAL": signal["TARGET_REAL"],
                                    "BASE_SIGNAL": signal["BASE_SIGNAL"],
                                    "RESULTADO": "WIN" if pnl > 0 else "LOSS",
                                    "R": trade["r"],
                                    "GANANCIA_PERDIDA": pnl,
                                    "CAPITAL_DESPUES": capital,
                                }
                            )
                            idx = trade["exit_idx"] + 1

                        r_values = [item["R"] for item in scenario_trades]
                        gains = [item["GANANCIA_PERDIDA"] for item in scenario_trades if item["GANANCIA_PERDIDA"] > 0]
                        losses = [item["GANANCIA_PERDIDA"] for item in scenario_trades if item["GANANCIA_PERDIDA"] < 0]
                        gross_profit = sum(gains)
                        gross_loss = abs(sum(losses))
                        pf = float("inf") if gross_loss == 0 and gross_profit > 0 else 0.0
                        if gross_loss > 0:
                            pf = gross_profit / gross_loss

                        summary_rows.append(
                            {
                                "ACTIVO": ASSET,
                                "DIRECCION": direction,
                                "MODELO": model_name,
                                "SISTEMA": system,
                                "THRESHOLD": threshold,
                                "COST_BPS": cost_bps,
                                "TRADES": len(scenario_trades),
                                "WIN_RATE": (len(gains) / len(scenario_trades) * 100) if scenario_trades else 0.0,
                                "PROFIT_FACTOR": pf,
                                "EXPECTANCY_R": float(np.mean(r_values)) if r_values else 0.0,
                                "AVERAGE_R": float(np.mean(r_values)) if r_values else 0.0,
                                "R_TOTAL": float(np.sum(r_values)) if r_values else 0.0,
                                "MAX_DRAWDOWN_PCT": _max_drawdown_pct(capital_curve),
                                "CAPITAL_FINAL": capital,
                            }
                        )
                        trade_rows.extend(scenario_trades)

    return pd.DataFrame(summary_rows), pd.DataFrame(trade_rows)


def _max_losing_streak(r_values):
    streak = 0
    max_streak = 0
    for value in r_values:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _bootstrap_metrics(r_values, iterations=500, seed=42):
    if len(r_values) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    expectancy = []
    drawdowns = []
    values = np.array(r_values, dtype=float)
    for _ in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        expectancy.append(float(np.mean(sample)))
        curve = [1000.0]
        capital = 1000.0
        for r_value in sample:
            capital += capital * RISK_PER_TRADE * r_value
            curve.append(capital)
        drawdowns.append(_max_drawdown_pct(curve))
    return (
        float(np.percentile(expectancy, 2.5)),
        float(np.percentile(expectancy, 97.5)),
        float(np.percentile(drawdowns, 95)),
    )


def _classification(row, robustness):
    if row["TRADES"] < 30:
        return "MUESTRA INSUFICIENTE", "Menos de 30 trades"
    if row["EXPECTANCY_R"] <= 0 or row["PROFIT_FACTOR"] <= 1:
        return "NO APTO", "Expectancy o PF no superan 2 bps"
    if robustness["FOLDS_POSITIVOS"] < 2:
        return "MIXTO", "Pocos folds positivos"
    if robustness["PORCENTAJE_FOLDS_POSITIVOS"] < 40:
        return "MIXTO", "Estabilidad entre folds debil"
    return "CANDIDATO A FORWARD TEST", "Cumple filtros descriptivos minimos; no validado"


def _robustness(summary, trades):
    rows = []
    for _, row in summary.iterrows():
        scenario = trades[
            (trades["DIRECCION"] == row["DIRECCION"])
            & (trades["MODELO"] == row["MODELO"])
            & (trades["SISTEMA"] == row["SISTEMA"])
            & (trades["THRESHOLD"] == row["THRESHOLD"])
            & (trades["COST_BPS"] == row["COST_BPS"])
        ].copy()
        r_values = scenario["R"].astype(float).tolist() if not scenario.empty else []
        fold_r = scenario.groupby("FOLD_SIGNAL")["R"].sum() if not scenario.empty else pd.Series(dtype=float)
        folds_positive = int((fold_r > 0).sum())
        total_folds = int(fold_r.count())
        ci_low, ci_high, dd95 = _bootstrap_metrics(r_values)
        robust_row = {
            "ACTIVO": ASSET,
            "DIRECCION": row["DIRECCION"],
            "MODELO": row["MODELO"],
            "SISTEMA": row["SISTEMA"],
            "THRESHOLD": row["THRESHOLD"],
            "COST_BPS": row["COST_BPS"],
            "TRADES": row["TRADES"],
            "PROFIT_FACTOR": row["PROFIT_FACTOR"],
            "EXPECTANCY_R": row["EXPECTANCY_R"],
            "AVERAGE_R": row["AVERAGE_R"],
            "R_TOTAL": row["R_TOTAL"],
            "MAX_DRAWDOWN_PCT": row["MAX_DRAWDOWN_PCT"],
            "MAX_LOSING_STREAK": _max_losing_streak(r_values),
            "FOLDS_POSITIVOS": folds_positive,
            "PORCENTAJE_FOLDS_POSITIVOS": (folds_positive / total_folds * 100) if total_folds else 0.0,
            "ESTABILIDAD_FOLDS_STD_R": float(fold_r.std(ddof=0)) if total_folds else 0.0,
            "BOOTSTRAP_EXPECTANCY_CI95_LOW": ci_low,
            "BOOTSTRAP_EXPECTANCY_CI95_HIGH": ci_high,
            "BOOTSTRAP_DD95": dd95,
        }
        classification, reason = _classification(row, robust_row)
        robust_row["CLASIFICACION"] = classification
        robust_row["MOTIVO"] = reason
        rows.append(robust_row)
    return pd.DataFrame(rows)


def _audit_existing_models(raiz, data):
    rows = []
    models_dir = raiz / "models" / "eurusd"
    train_end_idx = max(0, int(len(data) * 0.70) - TARGET_HORIZON - 1)
    train_start = data["fecha"].iloc[0] if not data.empty else ""
    train_end = data["fecha"].iloc[train_end_idx] if not data.empty else ""
    for direction in DIRECTIONS:
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
                row["NOTAS"] = "Modelo previo reutilizable solo como referencia de investigacion"
            except Exception as error:
                row["NOTAS"] = f"No se pudo cargar modelo: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


def _broker_metadata():
    try:
        from src.mt5_connector import _credenciales_mt5, _modo_cuenta, _valor
    except Exception as error:
        return None, f"MT5 no disponible: {error}"

    credentials, error = _credenciales_mt5()
    if error:
        return None, error

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None, "MetaTrader5 no esta instalado"

    initialized = False
    try:
        initialized = mt5.initialize()
        if not initialized:
            return None, f"No se pudo inicializar MT5: {mt5.last_error()}"
        if not mt5.login(credentials["login"], password=credentials["password"], server=credentials["server"]):
            return None, f"No se pudo conectar MT5: {mt5.last_error()}"
        account = mt5.account_info()
        if _modo_cuenta(mt5, account) != "DEMO":
            return None, "Cuenta MT5 no DEMO; metadata bloqueada"
        info = mt5.symbol_info("EURUSD") or mt5.symbol_info("EURUSDm")
        if info is None:
            return None, "No se encontro EURUSD/EURUSDm en MT5"
        return {
            "symbol": _valor(info, "name", "EURUSD"),
            "contract_size": _valor(info, "trade_contract_size", _valor(info, "contract_size")),
            "tick_value": _valor(info, "trade_tick_value"),
            "tick_size": _valor(info, "trade_tick_size"),
            "volume_min": _valor(info, "volume_min"),
            "volume_step": _valor(info, "volume_step"),
            "volume_max": _valor(info, "volume_max"),
        }, ""
    finally:
        if initialized:
            mt5.shutdown()


def _broker_aware(trades):
    metadata, error = _broker_metadata()
    rows = []
    representative = trades[trades["COST_BPS"] == MAIN_COST_BPS].copy()
    if not representative.empty:
        representative = representative.sort_values("FECHA_ENTRADA").head(100)

    for capital in CAPITALS:
        for _, trade in representative.iterrows():
            base = {
                "ACTIVO": ASSET,
                "CAPITAL": capital,
                "RISK_PCT": RISK_PER_TRADE * 100,
                "DIRECCION": trade["DIRECCION"],
                "MODELO": trade["MODELO"],
                "SISTEMA": trade["SISTEMA"],
                "THRESHOLD": trade["THRESHOLD"],
                "COST_BPS": trade["COST_BPS"],
                "ENTRY": trade["ENTRY"],
                "STOP_LOSS": trade["STOP_LOSS"],
                "BROKER_METADATA_SOURCE": "MT5_DEMO" if metadata else "UNAVAILABLE",
                "BROKER_METADATA_ERROR": error,
                "VOLUME_MIN": metadata.get("volume_min") if metadata else None,
                "VOLUME_STEP": metadata.get("volume_step") if metadata else None,
                "VOLUME_CALCULATED": None,
                "DECISION": "SKIP",
                "REASON": error or "OK",
            }
            if metadata:
                manager = RiskManager(RiskLimits(risk_per_trade=RISK_PER_TRADE))
                sizing = manager.calculate_position_size(
                    symbol=metadata["symbol"],
                    direction="BUY" if trade["DIRECCION"] == "LONG" else "SELL",
                    entry=trade["ENTRY"],
                    stop_loss=trade["STOP_LOSS"],
                    balance=capital,
                    symbol_info=metadata,
                )
                base["VOLUME_CALCULATED"] = sizing.volume
                base["DECISION"] = sizing.decision
                base["REASON"] = sizing.reason
            rows.append(base)
    return pd.DataFrame(rows)


def run_eurusd_research(raiz_proyecto=None):
    raiz = _root(raiz_proyecto)
    out = _research_dir(raiz)
    data = _read_dataset(raiz)
    audit = _audit_existing_models(raiz, data)
    walkforward, oof = _run_walk_forward(data)
    oof_summary, trades = _run_oof_backtest(oof)
    robustness = _robustness(oof_summary, trades)
    broker = _broker_aware(trades)

    audit.to_csv(out / "eurusd_existing_models_audit.csv", index=False)
    walkforward.to_csv(out / "eurusd_walkforward_summary.csv", index=False)
    oof.to_csv(out / "eurusd_oof_predictions.csv", index=False)
    oof_summary.to_csv(out / "eurusd_oof_summary.csv", index=False)
    robustness.to_csv(out / "eurusd_robustness.csv", index=False)
    trades.to_csv(out / "eurusd_trades.csv", index=False)
    broker.to_csv(out / "eurusd_broker_aware.csv", index=False)

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
        "audit": audit,
        "walkforward": walkforward,
        "oof_summary": oof_summary,
        "robustness": robustness,
        "broker_aware": broker,
        "candidates": candidates,
        "discarded": discarded,
        "trades": trades,
    }
