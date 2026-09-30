import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import ACTIVOS
from config_paper import CAPITAL_INICIAL, COST_BPS, PAPER_CONFIGS
from src.display import formatear_direccion


COLUMNAS_PRECIO = ["Open", "High", "Low", "Close", "Volume"]
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
SIGNALS_COLUMNS = [
    "CONFIG_ID",
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "FECHA_SIGNAL",
    "SIGNAL",
    "PROBABILIDAD",
    "CONDICION_TECNICA",
    "CLOSE",
    "RSI",
    "ATR",
    "EMA20",
    "EMA50",
    "EMA200",
]
OPEN_COLUMNS = [
    "CONFIG_ID",
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "FECHA_SIGNAL",
    "FECHA_ENTRADA",
    "ENTRY",
    "STOP",
    "TP",
    "ATR_SIGNAL",
    "PROBABILIDAD",
    "COST_BPS",
    "CAPITAL_ANTES",
]
TRADES_COLUMNS = [
    "CONFIG_ID",
    "ACTIVO",
    "DIRECCION",
    "MODELO",
    "SISTEMA",
    "THRESHOLD",
    "FECHA_SIGNAL",
    "FECHA_ENTRADA",
    "FECHA_SALIDA",
    "ENTRY",
    "STOP",
    "TP",
    "ATR_SIGNAL",
    "PROBABILIDAD",
    "RESULTADO",
    "R_BRUTO",
    "R_NETO",
    "COST_BPS",
    "CAPITAL_ANTES",
    "CAPITAL_DESPUES",
]
EQUITY_COLUMNS = ["FECHA", "CONFIG_ID", "CAPITAL"]


def _ahora_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _paper_dir(raiz):
    return Path(raiz) / "paper"


def _rutas(raiz):
    carpeta = _paper_dir(raiz)
    return {
        "carpeta": carpeta,
        "start": carpeta / "start.json",
        "state": carpeta / "state.json",
        "signals": carpeta / "signals.csv",
        "open_positions": carpeta / "open_positions.csv",
        "trades": carpeta / "trades.csv",
        "equity": carpeta / "equity.csv",
    }


def _paper_model_paths(raiz):
    carpeta = Path(raiz) / "models" / "paper"
    return {
        "carpeta": carpeta,
        "modelo": carpeta / "gold_logistic_long.pkl",
        "metadata": carpeta / "gold_logistic_long_metadata.json",
    }


def _asegurar_archivos(raiz, inicializar_start=False):
    rutas = _rutas(raiz)
    rutas["carpeta"].mkdir(parents=True, exist_ok=True)

    for ruta, columnas in [
        (rutas["signals"], SIGNALS_COLUMNS),
        (rutas["open_positions"], OPEN_COLUMNS),
        (rutas["trades"], TRADES_COLUMNS),
        (rutas["equity"], EQUITY_COLUMNS),
    ]:
        if not ruta.exists():
            pd.DataFrame(columns=columnas).to_csv(ruta, index=False)

    if not rutas["state"].exists():
        rutas["state"].write_text(
            json.dumps({"last_processed_bar": {}, "pending_signals": []}, indent=2),
            encoding="utf-8",
        )

    if inicializar_start and not rutas["start"].exists():
        start = _ahora_utc()
        rutas["start"].write_text(
            json.dumps({"start_time_utc": start.isoformat()}, indent=2),
            encoding="utf-8",
        )
        equity = pd.read_csv(rutas["equity"])
        if equity.empty:
            filas = [
                {
                    "FECHA": start.isoformat(),
                    "CONFIG_ID": config["CONFIG_ID"],
                    "CAPITAL": CAPITAL_INICIAL,
                }
                for config in PAPER_CONFIGS
            ]
            pd.DataFrame(filas).to_csv(rutas["equity"], mode="a", index=False, header=False)

    return rutas


def _leer_json(ruta, default):
    if not ruta.exists():
        return default

    return json.loads(ruta.read_text(encoding="utf-8"))


def _leer_start_time(ruta_start):
    if not ruta_start.exists():
        raise FileNotFoundError("No existe paper/start.json. Ejecute primero python main.py PAPER.")

    start_data = _leer_json(ruta_start, {})
    if "start_time_utc" not in start_data:
        raise ValueError("paper/start.json no contiene start_time_utc.")

    return pd.to_datetime(start_data["start_time_utc"], utc=True).tz_convert(None)


def _guardar_json(ruta, data):
    ruta.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _leer_csv(ruta, columnas):
    if not ruta.exists():
        return pd.DataFrame(columns=columnas)

    return pd.read_csv(ruta)


def _append_csv(ruta, filas, columnas):
    if not filas:
        return

    df = pd.DataFrame(filas, columns=columnas)
    df.to_csv(ruta, mode="a", index=False, header=False)


def _normalizar_indice(datos):
    datos = datos.copy()
    datos.index = pd.to_datetime(datos.index, utc=True).tz_convert(None)
    return datos


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


def _descargar_datos_recientes(activo):
    import yfinance as yf

    from src.indicators import calcular_indicadores

    ticker = ACTIVOS[activo]["ticker"]
    datos = yf.download(
        ticker,
        interval="1h",
        period="60d",
        progress=False,
        auto_adjust=False,
    )

    if datos.empty:
        raise ValueError(f"No se descargaron datos recientes para {activo}.")

    datos = _corregir_multiindex(datos)
    if "Volume" not in datos.columns:
        datos["Volume"] = 0

    faltantes = [columna for columna in COLUMNAS_PRECIO if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas de precio: {', '.join(faltantes)}")

    datos = datos[COLUMNAS_PRECIO].dropna().sort_index()
    datos = _normalizar_indice(datos)
    ahora = _ahora_utc().replace(tzinfo=None)
    datos = datos[datos.index + pd.Timedelta(hours=1) <= ahora]

    if datos.empty:
        raise ValueError("No hay velas H1 cerradas disponibles.")

    return calcular_indicadores(datos).dropna(subset=["EMA200", "RSI", "ATR"])


def _agregar_features_live(datos):
    datos = datos.copy()
    datos["EMA20_EMA50"] = (datos["EMA20"] / datos["EMA50"]) - 1
    datos["PRECIO_EMA20"] = (datos["Close"] / datos["EMA20"]) - 1
    datos["PRECIO_EMA50"] = (datos["Close"] / datos["EMA50"]) - 1
    datos["PRECIO_EMA200"] = (datos["Close"] / datos["EMA200"]) - 1
    datos["ATR_PCT"] = datos["ATR"] / datos["Close"]
    datos["RANGO_VELA"] = (datos["High"] - datos["Low"]) / datos["Close"]
    datos["CUERPO_VELA"] = (datos["Close"] - datos["Open"]).abs() / datos["Close"]
    datos["HORA"] = datos.index.hour
    datos["DIA_SEMANA"] = datos.index.dayofweek
    return datos.replace([float("inf"), float("-inf")], pd.NA)


def _cargar_datos_gold_local(raiz):
    ruta = Path(raiz) / "data" / "gold" / "gold_h1.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el historico local requerido: {ruta}")

    datos = pd.read_csv(ruta)
    if datos.empty:
        raise ValueError("El historico local de GOLD esta vacio.")

    posibles_fecha = ["fecha", "Datetime", "Date", "index", "Unnamed: 0"]
    columna_fecha = next(
        (columna for columna in posibles_fecha if columna in datos.columns),
        datos.columns[0],
    )
    datos[columna_fecha] = pd.to_datetime(datos[columna_fecha], utc=True, errors="coerce")
    datos = datos.dropna(subset=[columna_fecha])
    datos.index = datos[columna_fecha].dt.tz_convert(None)

    if "Volume" not in datos.columns:
        datos["Volume"] = 0

    faltantes = [columna for columna in COLUMNAS_PRECIO if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en gold_h1.csv: {', '.join(faltantes)}")

    datos = datos[COLUMNAS_PRECIO].copy()
    for columna in COLUMNAS_PRECIO:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos = datos.dropna(subset=COLUMNAS_PRECIO).sort_index()
    datos = datos[~datos.index.duplicated(keep="last")]
    return datos


def _crear_target_long_paper(datos, horizonte=TARGET_HORIZON):
    targets = []

    for i in range(len(datos)):
        fila = datos.iloc[i]
        close = fila["Close"]
        atr = fila["ATR"]

        if i + horizonte >= len(datos) or pd.isna(close) or pd.isna(atr) or atr <= 0:
            targets.append(pd.NA)
            continue

        take_profit = close + (2 * atr)
        stop_loss = close - atr
        objetivo = pd.NA

        for j in range(i + 1, i + horizonte + 1):
            futura = datos.iloc[j]

            if futura["Low"] <= stop_loss:
                objetivo = 0
                break

            if futura["High"] >= take_profit:
                objetivo = 1
                break

        targets.append(objetivo)

    return targets


def _preparar_dataset_freeze(raiz, paper_start):
    from src.indicators import calcular_indicadores

    datos = _cargar_datos_gold_local(raiz)
    datos = datos[datos.index < paper_start].copy()
    if len(datos) <= TARGET_HORIZON:
        raise ValueError("No hay suficiente historico anterior al inicio PAPER.")

    datos = calcular_indicadores(datos)
    datos = _agregar_features_live(datos)
    datos["TARGET_LONG"] = _crear_target_long_paper(datos, TARGET_HORIZON)
    datos = datos.replace([float("inf"), float("-inf")], pd.NA)
    datos = datos.dropna(subset=FEATURES_ML + ["TARGET_LONG"]).copy()
    datos["TARGET_LONG"] = datos["TARGET_LONG"].astype(int)

    if datos["TARGET_LONG"].nunique() < 2:
        raise ValueError(
            "Target LONG (COMPRAR) necesita al menos dos clases "
            "para congelar LOGISTIC."
        )

    dataset = datos.reset_index(names="fecha")
    if not dataset.empty and pd.to_datetime(dataset["fecha"].max()) >= paper_start:
        raise ValueError("Control de fuga fallido: el dataset contiene fechas posteriores al PAPER.")

    return dataset


def freeze_paper_model(raiz):
    from datetime import datetime, timezone

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rutas = _asegurar_archivos(raiz, inicializar_start=False)
    paper_start = _leer_start_time(rutas["start"])
    rutas_modelo = _paper_model_paths(raiz)

    if rutas_modelo["modelo"].exists():
        metadata = _leer_json(rutas_modelo["metadata"], {})
        return {
            "creado": False,
            "modelo_congelado": True,
            "ruta": rutas_modelo["modelo"],
            "metadata": metadata,
            "mensaje": "MODELO PAPER YA CONGELADO",
        }

    dataset = _preparar_dataset_freeze(raiz, paper_start)
    modelo = Pipeline(
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

    modelo.fit(dataset[FEATURES_ML], dataset["TARGET_LONG"])
    rutas_modelo["carpeta"].mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, rutas_modelo["modelo"])

    metadata = {
        "activo": "GOLD",
        "direccion": "LONG",
        "modelo": "LOGISTIC",
        "features": FEATURES_ML,
        "fecha_inicio_datos": pd.to_datetime(dataset["fecha"].min()).isoformat(),
        "fecha_fin_entrenamiento": pd.to_datetime(dataset["fecha"].max()).isoformat(),
        "paper_start": paper_start.isoformat(),
        "numero_filas_entrenamiento": int(len(dataset)),
        "target_horizon": TARGET_HORIZON,
        "fecha_creacion_modelo": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "thresholds_usados": [0.55, 0.60],
        "frozen": True,
        "data_posterior_al_paper_utilizada": False,
    }
    _guardar_json(rutas_modelo["metadata"], metadata)

    return {
        "creado": True,
        "modelo_congelado": True,
        "ruta": rutas_modelo["modelo"],
        "metadata": metadata,
        "mensaje": "MODELO PAPER CONGELADO",
    }


def _metadata_modelo_paper_valida(raiz, ruta_modelo, start_time):
    rutas_modelo = _paper_model_paths(raiz)
    if Path(ruta_modelo) != rutas_modelo["modelo"]:
        return False, "Ruta de modelo PAPER no reconocida."

    if not rutas_modelo["metadata"].exists():
        return False, "Falta metadata del modelo PAPER congelado."

    metadata = _leer_json(rutas_modelo["metadata"], {})
    if not metadata.get("frozen", False):
        return False, "Metadata PAPER no marca el modelo como congelado."

    if metadata.get("activo") != "GOLD" or metadata.get("direccion") != "LONG":
        return False, "Metadata PAPER no corresponde a GOLD LONG (COMPRAR)."

    if metadata.get("modelo") != "LOGISTIC":
        return False, "Metadata PAPER no corresponde a LOGISTIC."

    if metadata.get("features") != FEATURES_ML:
        return False, "Features del modelo PAPER no coinciden con la configuracion."

    if int(metadata.get("target_horizon", -1)) != TARGET_HORIZON:
        return False, "Target horizon del modelo PAPER no coincide."

    fecha_fin = pd.to_datetime(metadata.get("fecha_fin_entrenamiento"), utc=True).tz_convert(None)
    if fecha_fin >= start_time:
        return False, "Modelo PAPER entrenado con fecha final no anterior al inicio PAPER."

    metadata_start = pd.to_datetime(metadata.get("paper_start"), utc=True).tz_convert(None)
    if metadata_start != start_time:
        return False, "El paper_start de metadata no coincide con paper/start.json."

    if metadata.get("data_posterior_al_paper_utilizada", True):
        return False, "Metadata indica uso de datos posteriores al PAPER."

    return True, ""


def _modelo_path(raiz, config):
    activo = config["ACTIVO"].lower()
    direccion = config["DIRECCION"].lower()
    modelo = config["MODELO"].upper()
    carpeta = Path(raiz) / "models" / activo

    if activo == "gold" and direccion == "long" and modelo == "LOGISTIC":
        ruta_paper = _paper_model_paths(raiz)["modelo"]
        if ruta_paper.exists():
            return ruta_paper
        return None

    candidatos = {
        "LOGISTIC": [
            carpeta / f"logistic_{direccion}.pkl",
            carpeta / f"logistic_regression_{direccion}.pkl",
        ],
        "RF_ACTUAL": [carpeta / f"random_forest_{direccion}.pkl"],
        "RANDOM_FOREST": [carpeta / f"random_forest_{direccion}.pkl"],
    }.get(modelo, [])

    for ruta in candidatos:
        if ruta.exists():
            return ruta

    return None


def _modelo_estado(raiz, config, start_time):
    ruta_modelo = _modelo_path(raiz, config)
    if ruta_modelo is None:
        return {
            "cargado": False,
            "ruta": None,
            "causa": (
                "MODELO PAPER NO CONGELADO. "
                "Ejecute: python main.py PAPER FREEZE"
            ),
        }

    rutas_modelo = _paper_model_paths(raiz)
    if Path(ruta_modelo) == rutas_modelo["modelo"]:
        valido, causa = _metadata_modelo_paper_valida(raiz, ruta_modelo, start_time)
        return {
            "cargado": valido,
            "ruta": str(ruta_modelo),
            "causa": causa,
        }

    modelo_mtime = datetime.fromtimestamp(ruta_modelo.stat().st_mtime, tz=timezone.utc)
    modelo_mtime = pd.Timestamp(modelo_mtime).tz_convert(None)
    if modelo_mtime > start_time:
        return {
            "cargado": False,
            "ruta": str(ruta_modelo),
            "causa": (
                f"Modelo posterior al inicio PAPER para {config['CONFIG_ID']}; "
                "no se usa para forward test."
            ),
        }

    return {
        "cargado": True,
        "ruta": str(ruta_modelo),
        "causa": "",
    }


def _probabilidad_long(raiz, config, fila, start_time):
    import joblib

    estado = _modelo_estado(raiz, config, start_time)
    if not estado["cargado"]:
        return None, estado["causa"]

    modelo = joblib.load(estado["ruta"])
    x = pd.DataFrame([fila[FEATURES_ML].astype(float).to_dict()])
    probabilidades = modelo.predict_proba(x)
    clases = list(modelo.classes_)

    if 1 not in clases:
        return 0.0, ""

    return float(probabilidades[0][clases.index(1)]), ""


def _condicion_base_long(fila):
    return (
        fila["Close"] > fila["EMA200"]
        and fila["EMA20"] > fila["EMA50"]
        and fila["RSI"] > 50
        and fila["RSI"] < 70
    )


def _capital_actual(equity, config_id):
    datos = equity[equity["CONFIG_ID"] == config_id]
    if datos.empty:
        return CAPITAL_INICIAL
    return float(datos.iloc[-1]["CAPITAL"])


def _r_neto(entry, exit_price, r_bruto, atr, cost_bps):
    costo = (abs(entry) + abs(exit_price)) * (cost_bps / 10000)
    return r_bruto - (costo / atr)


def _fmt_notif(valor):
    if valor is None or pd.isna(valor):
        return "N/A"

    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return str(valor)

    if abs(valor) >= 1000:
        return f"{valor:.2f}"

    if abs(valor) >= 10:
        return f"{valor:.4f}"

    return f"{valor:.6f}"


def _enviar_telegram_unico(state, evento_id, mensaje):
    from src.telegram_events import send_telegram_event

    enviados = state.setdefault("telegram_sent", [])
    if evento_id in enviados:
        return

    root = state.get("_telegram_root") or Path(__file__).resolve().parents[1]
    event_type = state.get("_telegram_event_type", "PAPER")
    asset = state.get("_telegram_asset", "")
    config_id = state.get("_telegram_config_id", "")
    result = send_telegram_event(root, evento_id, event_type, mensaje, asset=asset, config_id=config_id)
    if result.get("sent") or result.get("duplicate"):
        enviados.append(evento_id)


def _mensaje_senal(signal_row):
    from src.telegram_events import format_new_signal_message

    return format_new_signal_message(signal_row)


def _mensaje_posicion_abierta(posicion):
    from src.telegram_events import format_open_message

    return format_open_message(posicion)


def _mensaje_posicion_cerrada(trade):
    from src.telegram_events import format_close_message

    return format_close_message(trade)


def _mensaje_error_critico(error):
    from src.telegram_events import format_critical_error_message

    return format_critical_error_message("GOLD", "ERROR_CRITICO", error)


def _cerrar_o_actualizar_posicion(posicion, datos, rutas):
    entrada = pd.to_datetime(posicion["FECHA_ENTRADA"])
    velas = datos[datos.index >= entrada]
    if velas.empty:
        return None, posicion

    entry = float(posicion["ENTRY"])
    stop = float(posicion["STOP"])
    tp = float(posicion["TP"])
    atr = float(posicion["ATR_SIGNAL"])
    max_hold = next(
        config["MAX_HOLD_BARS"]
        for config in PAPER_CONFIGS
        if config["CONFIG_ID"] == posicion["CONFIG_ID"]
    )

    salida_fecha = None
    exit_price = None
    resultado = None
    r_bruto = None
    resultado_notificacion = None

    for idx, (fecha, fila) in enumerate(velas.iterrows(), start=1):
        if fila["Low"] <= stop:
            salida_fecha = fecha
            exit_price = stop
            resultado = "LOSS"
            resultado_notificacion = "LOSS / STOP LOSS"
            r_bruto = (exit_price - entry) / atr
            break

        if fila["High"] >= tp:
            salida_fecha = fecha
            exit_price = tp
            resultado = "WIN"
            resultado_notificacion = "WIN / TAKE PROFIT"
            r_bruto = (exit_price - entry) / atr
            break

        if idx >= max_hold:
            salida_fecha = fecha
            exit_price = float(fila["Close"])
            r_bruto = (exit_price - entry) / atr
            resultado = "WIN" if r_bruto > 0 else "LOSS"
            resultado_notificacion = "TIME EXIT"
            break

    if salida_fecha is None:
        return None, posicion

    equity = _leer_csv(rutas["equity"], EQUITY_COLUMNS)
    capital_antes = float(posicion["CAPITAL_ANTES"])
    capital_actual = _capital_actual(equity, posicion["CONFIG_ID"])
    r_neto = _r_neto(entry, exit_price, r_bruto, atr, COST_BPS)
    capital_despues = capital_actual + (capital_actual * 0.01 * r_neto)

    trade = {
        "CONFIG_ID": posicion["CONFIG_ID"],
        "ACTIVO": posicion["ACTIVO"],
        "DIRECCION": posicion["DIRECCION"],
        "MODELO": posicion["MODELO"],
        "SISTEMA": posicion["SISTEMA"],
        "THRESHOLD": posicion["THRESHOLD"],
        "FECHA_SIGNAL": posicion["FECHA_SIGNAL"],
        "FECHA_ENTRADA": posicion["FECHA_ENTRADA"],
        "FECHA_SALIDA": salida_fecha.isoformat(),
        "ENTRY": entry,
        "STOP": stop,
        "TP": tp,
        "ATR_SIGNAL": atr,
        "PROBABILIDAD": posicion["PROBABILIDAD"],
        "RESULTADO": resultado,
        "R_BRUTO": r_bruto,
        "R_NETO": r_neto,
        "COST_BPS": COST_BPS,
        "CAPITAL_ANTES": capital_antes,
        "CAPITAL_DESPUES": capital_despues,
        "_EXIT": exit_price,
        "_RESULTADO_NOTIFICACION": resultado_notificacion,
    }
    equity_row = {
        "FECHA": salida_fecha.isoformat(),
        "CONFIG_ID": posicion["CONFIG_ID"],
        "CAPITAL": capital_despues,
    }
    return (trade, equity_row), None


def _abrir_desde_senal(pending, datos, rutas):
    fecha_signal = pd.to_datetime(pending["FECHA_SIGNAL"])
    futuras = datos[datos.index > fecha_signal]

    if futuras.empty:
        return None, pending

    fila_entrada = futuras.iloc[0]
    fecha_entrada = futuras.index[0]
    entry = float(fila_entrada["Open"])
    atr = float(pending["ATR_SIGNAL"])
    capital = _capital_actual(_leer_csv(rutas["equity"], EQUITY_COLUMNS), pending["CONFIG_ID"])

    posicion = {
        "CONFIG_ID": pending["CONFIG_ID"],
        "ACTIVO": pending["ACTIVO"],
        "DIRECCION": pending["DIRECCION"],
        "MODELO": pending["MODELO"],
        "SISTEMA": pending["SISTEMA"],
        "THRESHOLD": pending["THRESHOLD"],
        "FECHA_SIGNAL": pending["FECHA_SIGNAL"],
        "FECHA_ENTRADA": fecha_entrada.isoformat(),
        "ENTRY": entry,
        "STOP": entry - atr,
        "TP": entry + (2 * atr),
        "ATR_SIGNAL": atr,
        "PROBABILIDAD": pending["PROBABILIDAD"],
        "COST_BPS": COST_BPS,
        "CAPITAL_ANTES": capital,
    }
    return posicion, None


def _procesar_estado_abierto(config, datos, rutas, state):
    config_id = config["CONFIG_ID"]
    open_positions = _leer_csv(rutas["open_positions"], OPEN_COLUMNS)
    nuevas_open = []
    trades = []
    equity_rows = []
    posiciones_abiertas = []

    for _, posicion in open_positions.iterrows():
        if posicion["CONFIG_ID"] != config_id:
            nuevas_open.append(posicion.to_dict())
            continue

        cerrado, abierta = _cerrar_o_actualizar_posicion(posicion.to_dict(), datos, rutas)
        if cerrado is None:
            nuevas_open.append(abierta)
        else:
            trade, equity = cerrado
            trades.append(trade)
            equity_rows.append(equity)

    pendientes_restantes = []
    for pending in state.get("pending_signals", []):
        if pending["CONFIG_ID"] != config_id:
            pendientes_restantes.append(pending)
            continue

        posicion, pending_restante = _abrir_desde_senal(pending, datos, rutas)
        if pending_restante is not None:
            pendientes_restantes.append(pending_restante)
            continue

        posiciones_abiertas.append(posicion)
        cerrado, abierta = _cerrar_o_actualizar_posicion(posicion, datos, rutas)
        if cerrado is None:
            nuevas_open.append(abierta)
        else:
            trade, equity = cerrado
            trades.append(trade)
            equity_rows.append(equity)

    pd.DataFrame(nuevas_open, columns=OPEN_COLUMNS).to_csv(
        rutas["open_positions"], index=False
    )
    _append_csv(rutas["trades"], trades, TRADES_COLUMNS)
    _append_csv(rutas["equity"], equity_rows, EQUITY_COLUMNS)
    state["pending_signals"] = pendientes_restantes

    for posicion in posiciones_abiertas:
        evento_id = (
            f"OPEN|{posicion['CONFIG_ID']}|"
            f"{posicion['FECHA_SIGNAL']}|{posicion['FECHA_ENTRADA']}"
        )
        state["_telegram_root"] = str(rutas["carpeta"].parent)
        state["_telegram_event_type"] = "OPEN"
        state["_telegram_asset"] = posicion["ACTIVO"]
        state["_telegram_config_id"] = posicion["CONFIG_ID"]
        _enviar_telegram_unico(state, evento_id, _mensaje_posicion_abierta(posicion))

    for trade in trades:
        evento_id = (
            f"CLOSE|{trade['CONFIG_ID']}|"
            f"{trade['FECHA_ENTRADA']}|{trade['FECHA_SALIDA']}"
        )
        state["_telegram_root"] = str(rutas["carpeta"].parent)
        state["_telegram_event_type"] = "CLOSE"
        state["_telegram_asset"] = trade["ACTIVO"]
        state["_telegram_config_id"] = trade["CONFIG_ID"]
        _enviar_telegram_unico(state, evento_id, _mensaje_posicion_cerrada(trade))


def _config_tiene_abierta_o_pendiente(config_id, rutas, state):
    open_positions = _leer_csv(rutas["open_positions"], OPEN_COLUMNS)
    tiene_abierta = (
        not open_positions.empty
        and (open_positions["CONFIG_ID"] == config_id).any()
    )
    tiene_pendiente = any(
        pending["CONFIG_ID"] == config_id
        for pending in state.get("pending_signals", [])
    )
    return tiene_abierta or tiene_pendiente


def _separar_por_config(filas, config_id):
    propias = []
    otras = []

    for fila in filas:
        if fila.get("CONFIG_ID") == config_id:
            propias.append(fila)
        else:
            otras.append(fila)

    return propias, otras


def _bar_timestamp(value):
    return pd.to_datetime(value).isoformat()


def _normalize_timestamp_set(values):
    normalized = set()
    for value in values or []:
        if value is None or pd.isna(value):
            continue
        normalized.add(_bar_timestamp(value))
    return normalized


def _sorted_timestamps(values):
    return sorted(_normalize_timestamp_set(values), key=lambda item: pd.to_datetime(item))


def _processed_bar_timestamps(config_id, rutas, state):
    state_processed = state.get("processed_bar_timestamps", {})
    if isinstance(state_processed, dict):
        processed = _normalize_timestamp_set(state_processed.get(config_id, []))
    else:
        processed = _normalize_timestamp_set(state_processed)

    signals = _leer_csv(rutas["signals"], SIGNALS_COLUMNS)
    if not signals.empty and "FECHA_SIGNAL" in signals.columns:
        propias = signals[signals["CONFIG_ID"] == config_id]
        processed |= _normalize_timestamp_set(propias["FECHA_SIGNAL"].dropna().tolist())
    return processed


def _mark_bar_processed(config_id, rutas, state, timestamp):
    processed = _processed_bar_timestamps(config_id, rutas, state)
    processed.add(_bar_timestamp(timestamp))
    ordered = _sorted_timestamps(processed)
    state.setdefault("processed_bar_timestamps", {})[config_id] = ordered
    state.setdefault("last_processed_bar", {})[config_id] = ordered[-1] if ordered else None


def _crear_signal_row(config, fecha, fila, senal, probabilidad, condicion):
    return {
        "CONFIG_ID": config["CONFIG_ID"],
        "ACTIVO": config["ACTIVO"],
        "DIRECCION": config["DIRECCION"],
        "MODELO": config["MODELO"],
        "SISTEMA": config["SISTEMA"],
        "THRESHOLD": config["THRESHOLD"],
        "FECHA_SIGNAL": fecha.isoformat(),
        "SIGNAL": senal,
        "PROBABILIDAD": probabilidad,
        "CONDICION_TECNICA": condicion,
        "CLOSE": fila["Close"],
        "RSI": fila["RSI"],
        "ATR": fila["ATR"],
        "EMA20": fila["EMA20"],
        "EMA50": fila["EMA50"],
        "EMA200": fila["EMA200"],
    }


def _crear_pending_signal(config, fecha, fila, probabilidad):
    return {
        "CONFIG_ID": config["CONFIG_ID"],
        "ACTIVO": config["ACTIVO"],
        "DIRECCION": config["DIRECCION"],
        "MODELO": config["MODELO"],
        "SISTEMA": config["SISTEMA"],
        "THRESHOLD": config["THRESHOLD"],
        "FECHA_SIGNAL": fecha.isoformat(),
        "ATR_SIGNAL": float(fila["ATR"]),
        "PROBABILIDAD": probabilidad,
    }


def _status_base_catchup(config, datos, state, start_time):
    config_id = config["CONFIG_ID"]
    return {
        "config": config,
        "signal": "WAIT",
        "probabilidad": None,
        "condicion_tecnica": False,
        "error": "",
        "ultima_fila": None,
        "contexto_fila": None,
        "inicio_forward": start_time,
        "ultima_vela_disponible": datos.index[-1] if not datos.empty else None,
        "ultima_vela_procesada": state.get("last_processed_bar", {}).get(config_id),
        "hay_vela_nueva": False,
        "warmup_disponible": 0,
        "modelo_cargado": False,
        "modelo_error": "",
        "estado_datos": "",
        "proxima_accion": "",
        "velas_pendientes": 0,
        "barras_elegibles_totales": 0,
        "barras_ya_procesadas": 0,
        "timestamps_pendientes": [],
        "barras_pendientes_finales": 0,
        "timestamps_pendientes_finales": [],
        "timestamps_procesados": [],
        "timestamps_procesados_catchup": [],
        "velas_procesadas_catchup": 0,
        "trades_abiertos_catchup": 0,
        "trades_cerrados_catchup": 0,
        "ultima_vela_procesada_catchup": None,
    }


def _procesar_catchup_config(config, datos, rutas, state, start_time, raiz):
    config_id = config["CONFIG_ID"]
    status = _status_base_catchup(config, datos, state, start_time)

    modelo_estado = _modelo_estado(raiz, config, start_time)
    status["modelo_cargado"] = modelo_estado["cargado"]
    status["modelo_error"] = modelo_estado["causa"]

    datos_features = _agregar_features_live(datos)
    datos_features = datos_features.dropna(subset=FEATURES_ML)
    status["warmup_disponible"] = len(datos_features)
    if not datos_features.empty:
        status["contexto_fila"] = datos_features.iloc[-1]

    datos_elegibles = datos_features[datos_features.index >= start_time]
    processed_timestamps = _processed_bar_timestamps(config_id, rutas, state)
    ultimo_procesado = state.get("last_processed_bar", {}).get(config_id)
    if not processed_timestamps and ultimo_procesado:
        ultimo_procesado = pd.to_datetime(ultimo_procesado)
        processed_timestamps = _normalize_timestamp_set(
            datos_elegibles.loc[datos_elegibles.index <= ultimo_procesado].index.tolist()
        )
    datos_catchup = datos_elegibles[
        ~datos_elegibles.index.map(lambda value: _bar_timestamp(value)).isin(processed_timestamps)
    ]

    status["velas_pendientes"] = len(datos_catchup)
    status["barras_elegibles_totales"] = len(datos_elegibles)
    status["barras_ya_procesadas"] = len(processed_timestamps)
    status["timestamps_pendientes"] = [
        _bar_timestamp(value) for value in datos_catchup.index.tolist()
    ]
    status["timestamps_procesados"] = _sorted_timestamps(processed_timestamps)
    state.setdefault("processed_bar_timestamps", {})[config_id] = status["timestamps_procesados"]
    status["hay_vela_nueva"] = not datos_catchup.empty

    if datos_catchup.empty:
        status["signal"] = "WAIT - SIN NUEVA VELA"
        status["estado_datos"] = "SIN NUEVA VELA CERRADA DESDE INICIO PAPER"
        status["proxima_accion"] = "ESPERAR NUEVA VELA"
        return status

    if not modelo_estado["cargado"]:
        status["error"] = modelo_estado["causa"]
        status["estado_datos"] = "MODELO NO DISPONIBLE"
        status["proxima_accion"] = "NO CREAR TRADE"
        return status

    open_positions = _leer_csv(rutas["open_positions"], OPEN_COLUMNS)
    posiciones = open_positions.to_dict("records") if not open_positions.empty else []
    posiciones_config, posiciones_otras = _separar_por_config(posiciones, config_id)
    posicion_abierta = posiciones_config[0] if posiciones_config else None

    pendientes_config, pendientes_otras = _separar_por_config(
        state.get("pending_signals", []),
        config_id,
    )
    pending = pendientes_config[0] if pendientes_config else None

    signals = []
    trades = []
    equity_rows = []
    posiciones_abiertas_notificar = []

    for fecha, fila in datos_catchup.iterrows():
        ocupado_al_inicio = pending is not None or posicion_abierta is not None
        datos_hasta_actual = datos[datos.index <= fecha]

        if pending is not None and posicion_abierta is None:
            posicion, pending_restante = _abrir_desde_senal(
                pending,
                datos_hasta_actual,
                rutas,
            )
            if pending_restante is not None:
                pending = pending_restante
            else:
                pending = None
                posicion_abierta = posicion
                posiciones_abiertas_notificar.append(posicion)
                status["trades_abiertos_catchup"] += 1

        if posicion_abierta is not None:
            cerrado, abierta = _cerrar_o_actualizar_posicion(
                posicion_abierta,
                datos_hasta_actual,
                rutas,
            )
            if cerrado is None:
                posicion_abierta = abierta
            else:
                trade, equity = cerrado
                trades.append(trade)
                equity_rows.append(equity)
                posicion_abierta = None
                status["trades_cerrados_catchup"] += 1

        if (
            posicion_abierta is None
            and pending is None
            and not ocupado_al_inicio
        ):
            condicion = _condicion_base_long(fila)
            probabilidad, error = _probabilidad_long(raiz, config, fila, start_time)
            if error:
                status["error"] = error
                status["estado_datos"] = "ERROR DE MODELO"
                status["proxima_accion"] = "NO CREAR TRADE"
                break

            senal = (
                "LONG"
                if condicion and probabilidad is not None and probabilidad >= config["THRESHOLD"]
                else "WAIT"
            )
            signal_row = _crear_signal_row(
                config,
                fecha,
                fila,
                senal,
                probabilidad,
                condicion,
            )
            signals.append(signal_row)
            status.update(
                {
                    "signal": senal,
                    "probabilidad": probabilidad,
                    "condicion_tecnica": condicion,
                    "ultima_fila": fila,
                    "contexto_fila": fila,
                    "estado_datos": "VELA NUEVA PROCESADA",
                    "proxima_accion": "ESPERAR SIGUIENTE VELA",
                }
            )

            if senal == "LONG":
                pending = _crear_pending_signal(config, fecha, fila, probabilidad)
        else:
            status.update(
                {
                    "ultima_fila": fila,
                    "contexto_fila": fila,
                    "estado_datos": "VELA NUEVA PROCESADA",
                    "proxima_accion": (
                        "GESTIONAR POSICION"
                        if posicion_abierta is not None
                        else "ESPERAR ENTRADA PENDIENTE"
                    ),
                }
            )

        _mark_bar_processed(config_id, rutas, state, fecha)
        status["velas_procesadas_catchup"] += 1
        status["timestamps_procesados_catchup"].append(_bar_timestamp(fecha))
        status["ultima_vela_procesada_catchup"] = fecha
        status["ultima_vela_procesada"] = state["last_processed_bar"][config_id]

    processed_timestamps = _processed_bar_timestamps(config_id, rutas, state)
    restantes = datos_elegibles[
        ~datos_elegibles.index.map(lambda value: _bar_timestamp(value)).isin(processed_timestamps)
    ]
    status["barras_pendientes_finales"] = len(restantes)
    status["barras_ya_procesadas"] = len(processed_timestamps)
    status["timestamps_pendientes_finales"] = [
        _bar_timestamp(value) for value in restantes.index.tolist()
    ]
    status["timestamps_procesados"] = _sorted_timestamps(processed_timestamps)

    nuevas_open = posiciones_otras
    if posicion_abierta is not None:
        nuevas_open.append(posicion_abierta)
    pd.DataFrame(nuevas_open, columns=OPEN_COLUMNS).to_csv(
        rutas["open_positions"],
        index=False,
    )

    nuevos_pendientes = pendientes_otras
    if pending is not None:
        nuevos_pendientes.append(pending)
    state["pending_signals"] = nuevos_pendientes

    _append_csv(rutas["signals"], signals, SIGNALS_COLUMNS)
    _append_csv(rutas["trades"], trades, TRADES_COLUMNS)
    _append_csv(rutas["equity"], equity_rows, EQUITY_COLUMNS)

    for posicion in posiciones_abiertas_notificar:
        evento_id = (
            f"OPEN|{posicion['CONFIG_ID']}|"
            f"{posicion['FECHA_SIGNAL']}|{posicion['FECHA_ENTRADA']}"
        )
        state["_telegram_root"] = str(raiz)
        state["_telegram_event_type"] = "OPEN"
        state["_telegram_asset"] = posicion["ACTIVO"]
        state["_telegram_config_id"] = posicion["CONFIG_ID"]
        _enviar_telegram_unico(state, evento_id, _mensaje_posicion_abierta(posicion))

    for trade in trades:
        evento_id = (
            f"CLOSE|{trade['CONFIG_ID']}|"
            f"{trade['FECHA_ENTRADA']}|{trade['FECHA_SALIDA']}"
        )
        state["_telegram_root"] = str(raiz)
        state["_telegram_event_type"] = "CLOSE"
        state["_telegram_asset"] = trade["ACTIVO"]
        state["_telegram_config_id"] = trade["CONFIG_ID"]
        _enviar_telegram_unico(state, evento_id, _mensaje_posicion_cerrada(trade))

    for signal_row in signals:
        if signal_row["SIGNAL"] != "LONG":
            continue

        evento_id = (
            f"SIGNAL|{signal_row['CONFIG_ID']}|"
            f"{signal_row['FECHA_SIGNAL']}|{signal_row['THRESHOLD']}"
        )
        state["_telegram_root"] = str(raiz)
        state["_telegram_event_type"] = "SIGNAL"
        state["_telegram_asset"] = signal_row["ACTIVO"]
        state["_telegram_config_id"] = signal_row["CONFIG_ID"]
        _enviar_telegram_unico(state, evento_id, _mensaje_senal(signal_row))

    if status["velas_procesadas_catchup"] == 0 and not status["error"]:
        status["signal"] = "WAIT - SIN NUEVA VELA"
        status["estado_datos"] = "SIN NUEVA VELA CERRADA DESDE INICIO PAPER"
        status["proxima_accion"] = "ESPERAR NUEVA VELA"

    return status


def _procesar_senales(config, datos, rutas, state, start_time, raiz):
    config_id = config["CONFIG_ID"]
    signals = []
    status = {
        "config": config,
        "signal": "WAIT",
        "probabilidad": None,
        "condicion_tecnica": False,
        "error": "",
        "ultima_fila": None,
        "contexto_fila": None,
        "inicio_forward": start_time,
        "ultima_vela_disponible": datos.index[-1] if not datos.empty else None,
        "ultima_vela_procesada": state.get("last_processed_bar", {}).get(config_id),
        "hay_vela_nueva": False,
        "warmup_disponible": 0,
        "modelo_cargado": False,
        "modelo_error": "",
        "estado_datos": "",
        "proxima_accion": "",
    }

    modelo_estado = _modelo_estado(raiz, config, start_time)
    status["modelo_cargado"] = modelo_estado["cargado"]
    status["modelo_error"] = modelo_estado["causa"]

    datos_features = _agregar_features_live(datos)
    datos_features = datos_features.dropna(subset=FEATURES_ML)
    status["warmup_disponible"] = len(datos_features)
    if not datos_features.empty:
        status["contexto_fila"] = datos_features.iloc[-1]

    if _config_tiene_abierta_o_pendiente(config_id, rutas, state):
        status["estado_datos"] = "POSICION ABIERTA O SENAL PENDIENTE"
        status["proxima_accion"] = "GESTIONAR POSICION"
        return status

    datos = datos_features[datos_features.index >= start_time]
    ultimo_procesado = state.get("last_processed_bar", {}).get(config_id)
    if ultimo_procesado:
        datos = datos[datos.index > pd.to_datetime(ultimo_procesado)]

    status["hay_vela_nueva"] = not datos.empty

    if datos.empty:
        status["signal"] = "WAIT - SIN NUEVA VELA"
        status["estado_datos"] = "SIN NUEVA VELA CERRADA DESDE INICIO PAPER"
        status["proxima_accion"] = "ESPERAR NUEVA VELA"
        return status

    if not modelo_estado["cargado"]:
        status["error"] = modelo_estado["causa"]
        status["estado_datos"] = "MODELO NO DISPONIBLE"
        status["proxima_accion"] = "NO CREAR TRADE"
        return status

    for fecha, fila in datos.iterrows():
        condicion = _condicion_base_long(fila)
        probabilidad, error = _probabilidad_long(raiz, config, fila, start_time)
        if error:
            status["error"] = error
            status["estado_datos"] = "ERROR DE MODELO"
            status["proxima_accion"] = "NO CREAR TRADE"
            break

        senal = (
            "LONG"
            if condicion and probabilidad is not None and probabilidad >= config["THRESHOLD"]
            else "WAIT"
        )
        signal_row = {
            "CONFIG_ID": config_id,
            "ACTIVO": config["ACTIVO"],
            "DIRECCION": config["DIRECCION"],
            "MODELO": config["MODELO"],
            "SISTEMA": config["SISTEMA"],
            "THRESHOLD": config["THRESHOLD"],
            "FECHA_SIGNAL": fecha.isoformat(),
            "SIGNAL": senal,
            "PROBABILIDAD": probabilidad,
            "CONDICION_TECNICA": condicion,
            "CLOSE": fila["Close"],
            "RSI": fila["RSI"],
            "ATR": fila["ATR"],
            "EMA20": fila["EMA20"],
            "EMA50": fila["EMA50"],
            "EMA200": fila["EMA200"],
        }
        signals.append(signal_row)
        state.setdefault("last_processed_bar", {})[config_id] = fecha.isoformat()
        status.update(
            {
                "signal": senal,
                "probabilidad": probabilidad,
                "condicion_tecnica": condicion,
                "ultima_fila": fila,
                "contexto_fila": fila,
                "estado_datos": "VELA NUEVA PROCESADA",
                "proxima_accion": "ESPERAR SIGUIENTE VELA",
            }
        )

        if senal == "LONG":
            state.setdefault("pending_signals", []).append(
                {
                    "CONFIG_ID": config_id,
                    "ACTIVO": config["ACTIVO"],
                    "DIRECCION": config["DIRECCION"],
                    "MODELO": config["MODELO"],
                    "SISTEMA": config["SISTEMA"],
                    "THRESHOLD": config["THRESHOLD"],
                    "FECHA_SIGNAL": fecha.isoformat(),
                    "ATR_SIGNAL": float(fila["ATR"]),
                    "PROBABILIDAD": probabilidad,
                }
            )
            break

    _append_csv(rutas["signals"], signals, SIGNALS_COLUMNS)
    for signal_row in signals:
        if signal_row["SIGNAL"] != "LONG":
            continue

        evento_id = (
            f"SIGNAL|{signal_row['CONFIG_ID']}|"
            f"{signal_row['FECHA_SIGNAL']}|{signal_row['THRESHOLD']}"
        )
        state["_telegram_root"] = str(raiz)
        state["_telegram_event_type"] = "SIGNAL"
        state["_telegram_asset"] = signal_row["ACTIVO"]
        state["_telegram_config_id"] = signal_row["CONFIG_ID"]
        _enviar_telegram_unico(state, evento_id, _mensaje_senal(signal_row))

    return status


def ejecutar_paper(raiz):
    rutas = _asegurar_archivos(raiz, inicializar_start=True)
    start_data = _leer_json(rutas["start"], {})
    start_time = pd.to_datetime(start_data["start_time_utc"], utc=True).tz_convert(None)
    state = _leer_json(rutas["state"], {"last_processed_bar": {}, "pending_signals": []})
    resultados = []
    datos_por_activo = {}

    try:
        for config in PAPER_CONFIGS:
            activo = config["ACTIVO"]
            if activo not in datos_por_activo:
                datos_por_activo[activo] = _descargar_datos_recientes(activo)

            datos = datos_por_activo[activo]
            resultado = _procesar_catchup_config(config, datos, rutas, state, start_time, raiz)
            resultado["ultima_vela"] = datos.index[-1] if not datos.empty else None
            resultados.append(resultado)
    except Exception as error:
        from src.telegram_events import notify_critical_error

        notify_critical_error(raiz, "GOLD", "ERROR_CRITICO", str(error))
        raise

    for key in ["_telegram_root", "_telegram_event_type", "_telegram_asset", "_telegram_config_id"]:
        state.pop(key, None)
    _guardar_json(rutas["state"], state)
    return {
        "start_time": start_time,
        "resultados": resultados,
        "rutas": rutas,
    }


def _metricas_status(trades, equity, config_id):
    trades_config = trades[trades["CONFIG_ID"] == config_id].copy()
    equity_config = equity[equity["CONFIG_ID"] == config_id].copy()
    capital = _capital_actual(equity, config_id)

    if trades_config.empty:
        return {
            "trades_cerrados": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "r_acumulado": 0,
            "capital": capital,
            "max_drawdown": 0,
        }

    trades_config["R_NETO"] = pd.to_numeric(trades_config["R_NETO"], errors="coerce")
    ganancias = trades_config[trades_config["R_NETO"] > 0]["R_NETO"]
    perdidas = trades_config[trades_config["R_NETO"] < 0]["R_NETO"]
    gross_profit = ganancias.sum()
    gross_loss = abs(perdidas.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0
    )
    max_drawdown = 0
    if not equity_config.empty:
        capitales = pd.to_numeric(equity_config["CAPITAL"], errors="coerce").dropna()
        pico = capitales.iloc[0]
        for valor in capitales:
            pico = max(pico, valor)
            if pico > 0:
                max_drawdown = max(max_drawdown, ((pico - valor) / pico) * 100)

    return {
        "trades_cerrados": len(trades_config),
        "wins": int((trades_config["R_NETO"] > 0).sum()),
        "losses": int((trades_config["R_NETO"] <= 0).sum()),
        "win_rate": (trades_config["R_NETO"] > 0).mean() * 100,
        "profit_factor": profit_factor,
        "expectancy": trades_config["R_NETO"].mean(),
        "r_acumulado": trades_config["R_NETO"].sum(),
        "capital": capital,
        "max_drawdown": max_drawdown,
    }


def paper_status(raiz):
    rutas = _asegurar_archivos(raiz, inicializar_start=False)
    start = _leer_json(rutas["start"], None)
    trades = _leer_csv(rutas["trades"], TRADES_COLUMNS)
    open_positions = _leer_csv(rutas["open_positions"], OPEN_COLUMNS)
    equity = _leer_csv(rutas["equity"], EQUITY_COLUMNS)

    if start is None:
        return {
            "iniciado": False,
            "configs": [],
        }

    start_time = pd.to_datetime(start["start_time_utc"], utc=True)
    dias = (_ahora_utc() - start_time.to_pydatetime()).total_seconds() / 86400
    configs = []

    for config in PAPER_CONFIGS:
        config_id = config["CONFIG_ID"]
        metricas = _metricas_status(trades, equity, config_id)
        configs.append(
            {
                "config": config,
                "trades_abiertos": int(
                    (open_positions["CONFIG_ID"] == config_id).sum()
                    if not open_positions.empty
                    else 0
                ),
                **metricas,
            }
        )

    return {
        "iniciado": True,
        "fecha_inicio": start_time,
        "dias_forward": dias,
        "configs": configs,
    }
