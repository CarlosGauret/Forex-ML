from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.display import formatear_direccion


TARGET_HORIZON = 24

FEATURES_ML = [
    "EMA20_EMA50",
    "PRECIO_EMA20",
    "PRECIO_EMA50",
    "PRECIO_EMA200",
    "RSI",
    "ATR_PCT",
    "RET_1H",
    "RET_4H",
    "RET_24H",
    "VOLATILIDAD",
    "RANGO_VELA",
    "CUERPO_VELA",
    "HORA",
    "DIA_SEMANA",
]

PARAMETROS_RANDOM_FOREST = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

PARAMETROS_RF_SIMPLE = {
    "n_estimators": 400,
    "max_depth": 4,
    "min_samples_leaf": 50,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def _validar_columnas(dataset, target):
    requeridas = ["fecha", target] + FEATURES_ML
    faltantes = [columna for columna in requeridas if columna not in dataset.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


def _preparar_dataset(dataset, target):
    _validar_columnas(dataset, target)

    datos = dataset.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = datos.sort_values("fecha")

    for columna in FEATURES_ML:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos = datos.replace([float("inf"), float("-inf")], pd.NA)
    datos = datos.dropna(subset=FEATURES_ML + [target])
    datos[target] = datos[target].astype(int)

    if datos[target].nunique() < 2:
        raise ValueError(f"El target {target} necesita al menos dos clases para entrenar.")

    return datos


def _agregar_base_signal(datos):
    datos = datos.copy()

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

    datos["BASE_SIGNAL"] = "WAIT"
    datos.loc[nueva_buy, "BASE_SIGNAL"] = "BUY"
    datos.loc[nueva_sell, "BASE_SIGNAL"] = "SELL"

    return datos


def _division_temporal(datos):
    total = len(datos)
    train_fin = int(total * 0.70)
    valid_fin = int(total * 0.85)

    train = datos.iloc[: max(0, train_fin - TARGET_HORIZON)].copy()
    validation = datos.iloc[train_fin: max(train_fin, valid_fin - TARGET_HORIZON)].copy()
    test = datos.iloc[valid_fin:].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "No hay datos suficientes para division temporal 70/15/15 con embargo."
        )

    return train, validation, test


def _probabilidad_clase_positiva(modelo, x):
    probabilidades = modelo.predict_proba(x)
    clases = list(modelo.classes_)

    if 1 not in clases:
        return pd.Series([0.0] * len(x), index=x.index)

    indice_positivo = clases.index(1)
    return pd.Series(probabilidades[:, indice_positivo], index=x.index)


def _roc_auc_seguro(y_true, y_prob):
    if len(set(y_true)) < 2:
        return None

    return roc_auc_score(y_true, y_prob)


def _calcular_metricas(modelo, datos, target):
    x = datos[FEATURES_ML]
    y = datos[target]
    y_pred = modelo.predict(x)
    y_prob = _probabilidad_clase_positiva(modelo, x)
    matriz = confusion_matrix(y, y_pred, labels=[0, 1])

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ROC AUC": _roc_auc_seguro(y, y_prob),
        "Confusion Matrix": matriz,
    }


def _crear_modelo(nombre_modelo):
    if nombre_modelo == "RF_ACTUAL":
        return RandomForestClassifier(**PARAMETROS_RANDOM_FOREST)

    if nombre_modelo == "RF_SIMPLE":
        return RandomForestClassifier(**PARAMETROS_RF_SIMPLE)

    if nombre_modelo == "LOGISTIC":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(f"Modelo no soportado: {nombre_modelo}")


def _calcular_metricas_predicciones(y_true, y_pred, y_prob):
    return {
        "ROC AUC": _roc_auc_seguro(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def _prevalencia_clase_positiva(y):
    if len(y) == 0:
        return None

    return float((y == 1).mean())


def _baseline_accuracy(y):
    if len(y) == 0:
        return None

    conteos = y.value_counts(normalize=True)
    return float(conteos.max())


def _crear_resultado_test(modelo, test, target, nombre_probabilidad):
    resultado = test[["fecha", target] + FEATURES_ML].copy()
    resultado["PREDICCION"] = modelo.predict(test[FEATURES_ML])
    resultado[nombre_probabilidad] = _probabilidad_clase_positiva(
        modelo, test[FEATURES_ML]
    )
    return resultado


def _entrenar_modelo(dataset, target, activo, lado, raiz_proyecto):
    datos = _preparar_dataset(dataset, target)
    train, validation, test = _division_temporal(datos)

    if train[target].nunique() < 2:
        raise ValueError(
            f"TRAIN de {activo} {formatear_direccion(lado)} necesita al menos dos clases."
        )

    modelo = RandomForestClassifier(**PARAMETROS_RANDOM_FOREST)
    modelo.fit(train[FEATURES_ML], train[target])

    metricas = {
        "TRAIN": _calcular_metricas(modelo, train, target),
        "VALIDATION": _calcular_metricas(modelo, validation, target),
        "TEST": _calcular_metricas(modelo, test, target),
    }

    carpeta_activo = activo.lower()
    carpeta_modelos = raiz_proyecto / "models" / carpeta_activo
    carpeta_resultados = raiz_proyecto / "results" / carpeta_activo
    carpeta_modelos.mkdir(parents=True, exist_ok=True)
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    lado_archivo = lado.lower()
    ruta_modelo = carpeta_modelos / f"random_forest_{lado_archivo}.pkl"
    ruta_test = carpeta_resultados / f"ml_{lado_archivo}_test.csv"
    ruta_importancias = carpeta_resultados / f"feature_importance_{lado_archivo}.csv"

    joblib.dump(modelo, ruta_modelo)

    nombre_probabilidad = f"PROB_{lado}"
    resultado_test = _crear_resultado_test(
        modelo, test, target, nombre_probabilidad
    )
    resultado_test.to_csv(ruta_test, index=False)

    importancias = pd.DataFrame(
        {
            "feature": FEATURES_ML,
            "importance": modelo.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importancias.to_csv(ruta_importancias, index=False)

    return {
        "activo": activo,
        "modelo": lado,
        "target": target,
        "metricas": metricas,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "ruta_modelo": ruta_modelo,
        "ruta_test": ruta_test,
        "ruta_importancias": ruta_importancias,
        "importancias": importancias,
    }


def entrenar_modelos_activo(dataset, activo, raiz_proyecto=None):
    raiz = Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]

    resultado_long = _entrenar_modelo(
        dataset=dataset,
        target="TARGET_LONG",
        activo=activo,
        lado="LONG",
        raiz_proyecto=raiz,
    )
    resultado_short = _entrenar_modelo(
        dataset=dataset,
        target="TARGET_SHORT",
        activo=activo,
        lado="SHORT",
        raiz_proyecto=raiz,
    )

    return {
        "LONG": resultado_long,
        "SHORT": resultado_short,
    }


def _generar_folds_walkforward(datos, n_folds=5):
    total = len(datos)
    bloque = total // (n_folds + 1)

    if bloque <= TARGET_HORIZON:
        raise ValueError(
            "No hay datos suficientes para walk-forward con embargo de 24 velas."
        )

    folds = []

    for fold in range(1, n_folds + 1):
        test_inicio = bloque * fold
        test_fin = bloque * (fold + 1) if fold < n_folds else total
        train_fin = test_inicio - TARGET_HORIZON

        train = datos.iloc[:train_fin].copy()
        test = datos.iloc[test_inicio:test_fin].copy()

        if train.empty or test.empty:
            raise ValueError(f"Fold {fold} quedo sin datos suficientes.")

        folds.append(
            {
                "fold": fold,
                "train": train,
                "test": test,
                "embargo_inicio": train.index[-1] + 1,
                "embargo_fin": test.index[0] - 1,
            }
        )

    return folds


def _evaluar_modelo_walkforward(nombre_modelo, train, test, target):
    y_train = train[target]
    y_test = test[target]

    fila_base = {
        "ROC AUC": None,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "error": "",
        "predicciones": None,
        "probabilidades": None,
    }

    if y_train.nunique() < 2:
        fila_base["error"] = "TRAIN con una sola clase"
        return fila_base

    modelo = _crear_modelo(nombre_modelo)
    modelo.fit(train[FEATURES_ML], y_train)
    y_pred = modelo.predict(test[FEATURES_ML])
    y_prob = _probabilidad_clase_positiva(modelo, test[FEATURES_ML])
    metricas = _calcular_metricas_predicciones(y_test, y_pred, y_prob)

    fila_base.update(metricas)
    fila_base["predicciones"] = y_pred
    fila_base["probabilidades"] = y_prob
    return fila_base


def _resumir_walkforward(resultados):
    datos = pd.DataFrame(resultados)
    resumen = []

    for modelo, grupo in datos.groupby("MODELO"):
        auc = pd.to_numeric(grupo["ROC AUC"], errors="coerce")
        f1 = pd.to_numeric(grupo["F1"], errors="coerce")
        precision = pd.to_numeric(grupo["Precision"], errors="coerce")
        recall = pd.to_numeric(grupo["Recall"], errors="coerce")

        resumen.append(
            {
                "MODELO": modelo,
                "AUC MEDIO": auc.mean(),
                "AUC STD": auc.std(ddof=0),
                "AUC MIN": auc.min(),
                "AUC MAX": auc.max(),
                "F1 MEDIO": f1.mean(),
                "PRECISION MEDIA": precision.mean(),
                "RECALL MEDIO": recall.mean(),
            }
        )

    return pd.DataFrame(resumen)


def _ejecutar_walkforward_modelo(dataset, activo, direccion, target, raiz_proyecto):
    datos = _agregar_base_signal(_preparar_dataset(dataset, target)).reset_index(drop=True)
    folds = _generar_folds_walkforward(datos)
    resultados = []
    predicciones_oof = []
    modelos = ["RF_ACTUAL", "RF_SIMPLE", "LOGISTIC"]

    for fold in folds:
        train = fold["train"]
        test = fold["test"]
        y_test = test[target]
        prevalencia = _prevalencia_clase_positiva(y_test)
        baseline = _baseline_accuracy(y_test)

        for nombre_modelo in modelos:
            metricas = _evaluar_modelo_walkforward(
                nombre_modelo=nombre_modelo,
                train=train,
                test=test,
                target=target,
            )

            resultados.append(
                {
                    "ACTIVO": activo,
                    "DIRECCION": direccion,
                    "MODELO": nombre_modelo,
                    "FOLD": fold["fold"],
                    "N_TRAIN": len(train),
                    "N_TEST": len(test),
                    "TRAIN_FECHA_INICIAL": train["fecha"].iloc[0],
                    "TRAIN_FECHA_FINAL": train["fecha"].iloc[-1],
                    "TEST_FECHA_INICIAL": test["fecha"].iloc[0],
                    "TEST_FECHA_FINAL": test["fecha"].iloc[-1],
                    "EMBARGO_VELAS": TARGET_HORIZON,
                    "PREVALENCIA_POSITIVA": prevalencia,
                    "BASELINE_ACCURACY": baseline,
                    "ROC AUC": metricas["ROC AUC"],
                    "Accuracy": metricas["Accuracy"],
                    "Precision": metricas["Precision"],
                    "Recall": metricas["Recall"],
                    "F1": metricas["F1"],
                    "ERROR": metricas["error"],
                }
            )

            if metricas["probabilidades"] is None:
                continue

            for posicion, (_, fila_test) in enumerate(test.iterrows()):
                predicciones_oof.append(
                    {
                        "FECHA": fila_test["fecha"],
                        "ACTIVO": activo,
                        "DIRECCION": direccion,
                        "MODELO": nombre_modelo,
                        "FOLD": fold["fold"],
                        "CLOSE": fila_test["Close"],
                        "OPEN": fila_test["Open"],
                        "HIGH": fila_test["High"],
                        "LOW": fila_test["Low"],
                        "ATR": fila_test["ATR"],
                        "PROBABILIDAD": float(metricas["probabilidades"].iloc[posicion]),
                        "TARGET_REAL": fila_test[target],
                        "BASE_SIGNAL": fila_test["BASE_SIGNAL"],
                    }
                )

    carpeta_activo = activo.lower()
    carpeta_resultados = raiz_proyecto / "results" / carpeta_activo
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    ruta = carpeta_resultados / f"walkforward_{direccion.lower()}.csv"

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(ruta, index=False)
    resumen = _resumir_walkforward(resultados)

    return {
        "direccion": direccion,
        "resultados": resultados_df,
        "resumen": resumen,
        "oof": pd.DataFrame(predicciones_oof),
        "ruta": ruta,
    }


def ejecutar_walkforward_activo(dataset, activo, raiz_proyecto=None):
    raiz = Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]

    long = _ejecutar_walkforward_modelo(
        dataset=dataset,
        activo=activo,
        direccion="LONG",
        target="TARGET_LONG",
        raiz_proyecto=raiz,
    )
    short = _ejecutar_walkforward_modelo(
        dataset=dataset,
        activo=activo,
        direccion="SHORT",
        target="TARGET_SHORT",
        raiz_proyecto=raiz,
    )

    carpeta_activo = activo.lower()
    carpeta_resultados = raiz / "results" / carpeta_activo
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    ruta_oof = carpeta_resultados / "oof_predictions.csv"
    oof = pd.concat([long["oof"], short["oof"]], ignore_index=True)
    oof = oof.sort_values(["FECHA", "DIRECCION", "MODELO", "FOLD"])
    oof.to_csv(ruta_oof, index=False)

    return {
        "LONG": long,
        "SHORT": short,
        "oof": oof,
        "ruta_oof": ruta_oof,
    }
