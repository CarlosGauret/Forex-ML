import os
import math
from pathlib import Path


SIMBOLO_DRYRUN = "GOLD"
RIESGO_PORCENTAJE = 0.01


def _cargar_env():
    raiz = Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(raiz / ".env")


def _credenciales_mt5():
    _cargar_env()

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    faltantes = []
    if not login:
        faltantes.append("MT5_LOGIN")
    if not password:
        faltantes.append("MT5_PASSWORD")
    if not server:
        faltantes.append("MT5_SERVER")

    if faltantes:
        return None, f"Faltan variables en .env: {', '.join(faltantes)}"

    try:
        login = int(login)
    except ValueError:
        return None, "MT5_LOGIN debe ser numerico."

    return {
        "login": login,
        "password": password,
        "server": server,
    }, ""


def _modo_cuenta(mt5, account_info):
    trade_mode = getattr(account_info, "trade_mode", None)
    demo = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", None)
    real = getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", None)
    contest = getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", None)

    if demo is not None and trade_mode == demo:
        return "DEMO"

    if real is not None and trade_mode == real:
        return "REAL"

    if contest is not None and trade_mode == contest:
        return "CONTEST"

    return "DESCONOCIDA"


def _valor(objeto, nombre, default=None):
    return getattr(objeto, nombre, default) if objeto is not None else default


def _resultado_error(error, conexion_terminal=False, cuenta="N/A", servidor_ok=False):
    return {
        "ok": False,
        "error": error,
        "conexion_terminal": conexion_terminal,
        "cuenta": cuenta,
        "servidor_ok": servidor_ok,
        "saldo": None,
        "equity": None,
        "instrumentos": [],
        "trading_enabled": False,
        "ordenes_enviadas": 0,
        "mt5_demo_ready": False,
        "cuenta_real_bloqueada": False,
    }


def _buscar_instrumentos_gold(mt5):
    simbolos = mt5.symbols_get()
    if simbolos is None:
        return []

    coincidencias = []
    for simbolo in simbolos:
        nombre = _valor(simbolo, "name", "")
        descripcion = _valor(simbolo, "description", "")
        texto = f"{nombre} {descripcion}".upper()
        if "XAUUSD" not in texto and "GOLD" not in texto:
            continue

        info = mt5.symbol_info(nombre)
        tick = mt5.symbol_info_tick(nombre)
        coincidencias.append(
            {
                "symbol": nombre,
                "description": _valor(info, "description", descripcion),
                "bid": _valor(tick, "bid", _valor(info, "bid")),
                "ask": _valor(tick, "ask", _valor(info, "ask")),
                "point": _valor(info, "point"),
                "digits": _valor(info, "digits"),
                "trade_contract_size": _valor(info, "trade_contract_size"),
                "volume_min": _valor(info, "volume_min"),
                "volume_max": _valor(info, "volume_max"),
                "volume_step": _valor(info, "volume_step"),
            }
        )

    return coincidencias


def _leer_atr_diagnostico():
    raiz = Path(__file__).resolve().parents[1]
    ruta_state = raiz / "paper" / "state.json"
    if ruta_state.exists():
        try:
            import json

            state = json.loads(ruta_state.read_text(encoding="utf-8"))
            for pending in state.get("pending_signals", []):
                if pending.get("ACTIVO") == "GOLD" and pending.get("DIRECCION") == "LONG":
                    return {
                        "atr": float(pending["ATR_SIGNAL"]),
                        "fuente": "SENAL PAPER PENDIENTE",
                        "fecha": pending.get("FECHA_SIGNAL"),
                    }
        except Exception:
            pass

    ruta_indicadores = raiz / "data" / "gold" / "gold_h1_indicators.csv"
    if not ruta_indicadores.exists():
        return {
            "atr": None,
            "fuente": "NO DISPONIBLE",
            "fecha": None,
        }

    try:
        import pandas as pd

        datos = pd.read_csv(ruta_indicadores)
        if "ATR" not in datos.columns:
            return {
                "atr": None,
                "fuente": "NO DISPONIBLE",
                "fecha": None,
            }

        datos["ATR"] = pd.to_numeric(datos["ATR"], errors="coerce")
        datos = datos.dropna(subset=["ATR"])
        if datos.empty:
            return {
                "atr": None,
                "fuente": "NO DISPONIBLE",
                "fecha": None,
            }

        ultima = datos.iloc[-1]
        columna_fecha = datos.columns[0]
        return {
            "atr": float(ultima["ATR"]),
            "fuente": "SIMULACION - ultimo ATR disponible",
            "fecha": str(ultima[columna_fecha]),
        }
    except Exception:
        return {
            "atr": None,
            "fuente": "NO DISPONIBLE",
            "fecha": None,
        }


def _precision_volumen(step):
    texto = f"{step:.10f}".rstrip("0").rstrip(".")
    if "." not in texto:
        return 0
    return len(texto.split(".")[1])


def _normalizar_volumen_abajo(volumen, step, volume_max):
    if step <= 0:
        return volumen

    precision = _precision_volumen(step)
    volumen = math.floor((volumen + 1e-12) / step) * step
    volumen = min(volumen, volume_max)
    return round(volumen, precision)


def _perdida_estimada(mt5, symbol, volumen, entry, stop_loss):
    profit = mt5.order_calc_profit(
        mt5.ORDER_TYPE_BUY,
        symbol,
        volumen,
        entry,
        stop_loss,
    )
    if profit is None:
        return None

    return abs(float(profit))


def _calcular_volumen_por_riesgo(mt5, symbol, entry, stop_loss, riesgo_maximo, info):
    volume_min = float(_valor(info, "volume_min", 0) or 0)
    volume_max = float(_valor(info, "volume_max", 0) or 0)
    volume_step = float(_valor(info, "volume_step", 0) or 0)

    if volume_min <= 0 or volume_max <= 0 or volume_step <= 0:
        return {
            "ok": False,
            "error": "Parametros de volumen invalidos.",
            "lote_teorico": None,
            "lote_permitido": None,
            "perdida_estimada": None,
            "perdida_minima": None,
            "riesgo_pct_minimo": None,
        }

    perdida_1_lote = _perdida_estimada(mt5, symbol, 1.0, entry, stop_loss)
    perdida_minima = _perdida_estimada(mt5, symbol, volume_min, entry, stop_loss)
    if not perdida_1_lote or perdida_1_lote <= 0 or perdida_minima is None:
        return {
            "ok": False,
            "error": "No se pudo calcular perdida estimada con order_calc_profit().",
            "lote_teorico": None,
            "lote_permitido": None,
            "perdida_estimada": None,
            "perdida_minima": perdida_minima,
            "riesgo_pct_minimo": None,
        }

    lote_teorico = riesgo_maximo / perdida_1_lote
    riesgo_pct_minimo = None

    if perdida_minima > riesgo_maximo:
        return {
            "ok": False,
            "error": "VOLUMEN_MINIMO_DEMASIADO_GRANDE",
            "lote_teorico": lote_teorico,
            "lote_permitido": None,
            "perdida_estimada": None,
            "perdida_minima": perdida_minima,
            "riesgo_pct_minimo": None,
        }

    candidato = _normalizar_volumen_abajo(lote_teorico, volume_step, volume_max)
    if candidato < volume_min:
        candidato = volume_min

    perdida = _perdida_estimada(mt5, symbol, candidato, entry, stop_loss)
    while candidato >= volume_min and (perdida is None or perdida > riesgo_maximo):
        candidato = round(candidato - volume_step, _precision_volumen(volume_step))
        if candidato < volume_min:
            break
        perdida = _perdida_estimada(mt5, symbol, candidato, entry, stop_loss)

    if candidato < volume_min or perdida is None:
        return {
            "ok": False,
            "error": "No se encontro volumen permitido sin superar el riesgo.",
            "lote_teorico": lote_teorico,
            "lote_permitido": None,
            "perdida_estimada": None,
            "perdida_minima": perdida_minima,
            "riesgo_pct_minimo": riesgo_pct_minimo,
        }

    while True:
        siguiente = round(candidato + volume_step, _precision_volumen(volume_step))
        if siguiente > volume_max:
            break
        perdida_siguiente = _perdida_estimada(mt5, symbol, siguiente, entry, stop_loss)
        if perdida_siguiente is None or perdida_siguiente > riesgo_maximo:
            break
        candidato = siguiente
        perdida = perdida_siguiente

    return {
        "ok": True,
        "error": "",
        "lote_teorico": lote_teorico,
        "lote_permitido": candidato,
        "perdida_estimada": perdida,
        "perdida_minima": perdida_minima,
        "riesgo_pct_minimo": None,
    }


def verificar_mt5_demo_readonly():
    credenciales, error = _credenciales_mt5()
    if error:
        return {
            "ok": False,
            "error": error,
            "conexion_terminal": False,
            "cuenta": "N/A",
            "servidor_ok": False,
            "saldo": None,
            "equity": None,
            "instrumentos": [],
            "trading_enabled": False,
            "ordenes_enviadas": 0,
            "mt5_demo_ready": False,
            "cuenta_real_bloqueada": False,
        }

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return {
            "ok": False,
            "error": "MetaTrader5 no esta instalado.",
            "conexion_terminal": False,
            "cuenta": "N/A",
            "servidor_ok": False,
            "saldo": None,
            "equity": None,
            "instrumentos": [],
            "trading_enabled": False,
            "ordenes_enviadas": 0,
            "mt5_demo_ready": False,
            "cuenta_real_bloqueada": False,
        }

    inicializado = False
    try:
        inicializado = mt5.initialize()
        if not inicializado:
            return {
                "ok": False,
                "error": f"No se pudo inicializar MT5: {mt5.last_error()}",
                "conexion_terminal": False,
                "cuenta": "N/A",
                "servidor_ok": False,
                "saldo": None,
                "equity": None,
                "instrumentos": [],
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "mt5_demo_ready": False,
                "cuenta_real_bloqueada": False,
            }

        conectado = mt5.login(
            credenciales["login"],
            password=credenciales["password"],
            server=credenciales["server"],
        )
        if not conectado:
            return {
                "ok": False,
                "error": f"No se pudo conectar a la cuenta MT5: {mt5.last_error()}",
                "conexion_terminal": True,
                "cuenta": "N/A",
                "servidor_ok": False,
                "saldo": None,
                "equity": None,
                "instrumentos": [],
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "mt5_demo_ready": False,
                "cuenta_real_bloqueada": False,
            }

        account_info = mt5.account_info()
        terminal_info = mt5.terminal_info()
        version = mt5.version()

        if account_info is None:
            return {
                "ok": False,
                "error": "No se pudo obtener account_info().",
                "conexion_terminal": terminal_info is not None,
                "cuenta": "N/A",
                "servidor_ok": False,
                "saldo": None,
                "equity": None,
                "instrumentos": [],
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "mt5_demo_ready": False,
                "cuenta_real_bloqueada": False,
            }

        cuenta = _modo_cuenta(mt5, account_info)
        if cuenta == "REAL":
            return {
                "ok": False,
                "error": "CUENTA REAL DETECTADA - CONEXION BLOQUEADA",
                "conexion_terminal": terminal_info is not None,
                "cuenta": cuenta,
                "servidor_ok": bool(_valor(account_info, "server")),
                "saldo": _valor(account_info, "balance"),
                "equity": _valor(account_info, "equity"),
                "instrumentos": [],
                "terminal_info": terminal_info,
                "version": version,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "mt5_demo_ready": False,
                "cuenta_real_bloqueada": True,
            }

        if cuenta != "DEMO":
            return {
                "ok": False,
                "error": f"La cuenta no es DEMO. Tipo detectado: {cuenta}",
                "conexion_terminal": terminal_info is not None,
                "cuenta": cuenta,
                "servidor_ok": bool(_valor(account_info, "server")),
                "saldo": _valor(account_info, "balance"),
                "equity": _valor(account_info, "equity"),
                "instrumentos": [],
                "terminal_info": terminal_info,
                "version": version,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "mt5_demo_ready": False,
                "cuenta_real_bloqueada": False,
            }

        instrumentos = _buscar_instrumentos_gold(mt5)
        return {
            "ok": True,
            "error": "",
            "conexion_terminal": terminal_info is not None,
            "cuenta": cuenta,
            "servidor_ok": bool(_valor(account_info, "server")),
            "saldo": _valor(account_info, "balance"),
            "equity": _valor(account_info, "equity"),
            "instrumentos": instrumentos,
            "terminal_info": terminal_info,
            "version": version,
            "trading_enabled": False,
            "ordenes_enviadas": 0,
            "mt5_demo_ready": True,
            "cuenta_real_bloqueada": False,
        }
    finally:
        if inicializado:
            mt5.shutdown()


def ejecutar_mt5_dryrun_gold():
    credenciales, error = _credenciales_mt5()
    if error:
        return _resultado_error(error)

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return _resultado_error("MetaTrader5 no esta instalado.")

    inicializado = False
    try:
        inicializado = mt5.initialize()
        if not inicializado:
            return _resultado_error(f"No se pudo inicializar MT5: {mt5.last_error()}")

        conectado = mt5.login(
            credenciales["login"],
            password=credenciales["password"],
            server=credenciales["server"],
        )
        if not conectado:
            return _resultado_error(
                f"No se pudo conectar a la cuenta MT5: {mt5.last_error()}",
                conexion_terminal=True,
            )

        account_info = mt5.account_info()
        if account_info is None:
            return _resultado_error(
                "No se pudo obtener account_info().",
                conexion_terminal=True,
            )

        cuenta = _modo_cuenta(mt5, account_info)
        servidor_ok = bool(_valor(account_info, "server"))
        saldo = float(_valor(account_info, "balance", 0) or 0)
        equity = _valor(account_info, "equity")

        if cuenta == "REAL":
            resultado = _resultado_error(
                "CUENTA REAL DETECTADA - CONEXION BLOQUEADA",
                conexion_terminal=True,
                cuenta=cuenta,
                servidor_ok=servidor_ok,
            )
            resultado.update(
                {
                    "saldo": saldo,
                    "equity": equity,
                    "cuenta_real_bloqueada": True,
                }
            )
            return resultado

        if cuenta != "DEMO":
            resultado = _resultado_error(
                f"La cuenta no es DEMO. Tipo detectado: {cuenta}",
                conexion_terminal=True,
                cuenta=cuenta,
                servidor_ok=servidor_ok,
            )
            resultado.update({"saldo": saldo, "equity": equity})
            return resultado

        symbol = SIMBOLO_DRYRUN
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None:
            return {
                "ok": False,
                "error": "No se encontro el simbolo exacto GOLD.",
                "cuenta": cuenta,
                "broker": "XM",
                "symbol": symbol,
                "balance": saldo,
                "equity": equity,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "decision": "SKIP",
            }

        bid = _valor(tick, "bid")
        ask = _valor(tick, "ask")
        if tick is None or bid is None or ask is None or bid <= 0 or ask <= 0:
            return {
                "ok": False,
                "error": "MERCADO CERRADO / SIN PRECIO DISPONIBLE",
                "cuenta": cuenta,
                "broker": "XM",
                "symbol": symbol,
                "balance": saldo,
                "equity": equity,
                "bid": bid,
                "ask": ask,
                "info": info,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "decision": "SKIP",
            }

        atr_data = _leer_atr_diagnostico()
        atr = atr_data["atr"]
        if atr is None or atr <= 0:
            return {
                "ok": False,
                "error": "ATR no disponible para simulacion.",
                "cuenta": cuenta,
                "broker": "XM",
                "symbol": symbol,
                "balance": saldo,
                "equity": equity,
                "bid": bid,
                "ask": ask,
                "info": info,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "decision": "SKIP",
            }

        entry = float(ask)
        stop_loss = entry - atr
        take_profit = entry + (2 * atr)
        riesgo_maximo = saldo * RIESGO_PORCENTAJE
        volumen = _calcular_volumen_por_riesgo(
            mt5,
            symbol,
            entry,
            stop_loss,
            riesgo_maximo,
            info,
        )

        perdida_minima = volumen.get("perdida_minima")
        riesgo_pct_minimo = (
            (perdida_minima / saldo) * 100
            if perdida_minima is not None and saldo > 0
            else None
        )
        perdida_estimada = volumen.get("perdida_estimada")
        riesgo_pct_real = (
            (perdida_estimada / saldo) * 100
            if perdida_estimada is not None and saldo > 0
            else None
        )

        if volumen["error"] == "VOLUMEN_MINIMO_DEMASIADO_GRANDE":
            decision = "SKIP - VOLUMEN MINIMO DEMASIADO GRANDE"
        elif volumen["ok"]:
            decision = "EXECUTABLE"
        else:
            decision = "SKIP"

        return {
            "ok": volumen["ok"],
            "error": volumen["error"],
            "cuenta": cuenta,
            "broker": "XM",
            "symbol": symbol,
            "balance": saldo,
            "equity": equity,
            "bid": bid,
            "ask": ask,
            "atr": atr,
            "atr_fuente": atr_data["fuente"],
            "atr_fecha": atr_data["fecha"],
            "direccion": "LONG",
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "riesgo_objetivo_pct": RIESGO_PORCENTAJE * 100,
            "riesgo_maximo_usd": riesgo_maximo,
            "lote_teorico": volumen["lote_teorico"],
            "lote_minimo": float(_valor(info, "volume_min", 0) or 0),
            "lote_maximo": float(_valor(info, "volume_max", 0) or 0),
            "lote_step": float(_valor(info, "volume_step", 0) or 0),
            "lote_permitido": volumen["lote_permitido"],
            "perdida_estimada": perdida_estimada,
            "perdida_minima": perdida_minima,
            "riesgo_pct_real": riesgo_pct_real,
            "riesgo_pct_minimo": riesgo_pct_minimo,
            "info": info,
            "decision": decision,
            "trading_enabled": False,
            "ordenes_enviadas": 0,
        }
    finally:
        if inicializado:
            mt5.shutdown()


def _leer_atr_local(activo):
    raiz = Path(__file__).resolve().parents[1]
    carpeta = activo.lower()
    ruta = raiz / "data" / carpeta / f"{carpeta}_h1_indicators.csv"
    if not ruta.exists():
        return {"atr": None, "fuente": "NO DISPONIBLE", "fecha": None}

    try:
        import pandas as pd

        datos = pd.read_csv(ruta)
        if "ATR" not in datos.columns:
            return {"atr": None, "fuente": "NO DISPONIBLE", "fecha": None}

        datos["ATR"] = pd.to_numeric(datos["ATR"], errors="coerce")
        datos = datos.dropna(subset=["ATR"])
        if datos.empty:
            return {"atr": None, "fuente": "NO DISPONIBLE", "fecha": None}

        ultima = datos.iloc[-1]
        columna_fecha = datos.columns[0]
        return {
            "atr": float(ultima["ATR"]),
            "fuente": "LOCAL_INDICATORS",
            "fecha": str(ultima[columna_fecha]),
        }
    except Exception:
        return {"atr": None, "fuente": "NO DISPONIBLE", "fecha": None}


def _pipeline_state_mt5_v2(activo):
    from src.multi_asset import inspect_asset_pipeline

    return inspect_asset_pipeline(activo)


def _modelo_validado_mt5_v2(activo):
    return _pipeline_state_mt5_v2(activo).validated


def evaluar_dryrun_activo_v2(
    activo,
    resolved_symbol,
    info,
    tick,
    balance,
    atr_data=None,
    timestamp=None,
):
    from src.analytics_logger import utc_timestamp
    from src.trade_filters import DECISION_SKIP_DATA, DECISION_WAIT, evaluate_trade_filters

    atr_data = atr_data or {"atr": None, "fuente": "NO DISPONIBLE", "fecha": None}
    timestamp = timestamp or utc_timestamp()
    pipeline_state = _pipeline_state_mt5_v2(activo)
    mt5_symbol = getattr(resolved_symbol, "mt5_symbol", "")
    found = bool(getattr(resolved_symbol, "found", False))

    base = {
        "timestamp": timestamp,
        "activo_logico": activo,
        "simbolo_mt5": mt5_symbol,
        "timeframe": "H1",
        "bid": None,
        "ask": None,
        "spread": None,
        "signal": "ESPERAR",
        "confidence": None,
        "modelo": "N/A",
        "model_version": "N/A",
        "ATR": atr_data.get("atr"),
        "swing_high": None,
        "swing_low": None,
        "entry": None,
        "SL": None,
        "TP": None,
        "SL_mode": "N/A",
        "TP_mode": "N/A",
        "RR": None,
        "risk_pct": RIESGO_PORCENTAJE * 100,
        "risk_amount": (float(balance) * RIESGO_PORCENTAJE) if balance else None,
        "volume_theoretical": None,
        "volume_allowed": None,
        "estimated_loss": None,
        "decision": DECISION_WAIT,
        "reason": "",
        "modelo_validado": False,
        "modelos_disponibles": pipeline_state.models_available,
        "modelo_long_disponible": pipeline_state.long_model.available,
        "modelo_short_disponible": pipeline_state.short_model.available,
        "validacion_faltante": ", ".join(pipeline_state.missing_validation),
        "mercado_abierto": False,
        "description": getattr(resolved_symbol, "description", ""),
        "digits": getattr(resolved_symbol, "digits", None),
        "point": getattr(resolved_symbol, "point", None),
        "contract_size": getattr(resolved_symbol, "contract_size", None),
        "volume_min": getattr(resolved_symbol, "volume_min", None),
        "volume_max": getattr(resolved_symbol, "volume_max", None),
        "volume_step": getattr(resolved_symbol, "volume_step", None),
        "ordenes_enviadas": 0,
        "trading_enabled": False,
    }

    if not found:
        base.update(
            {
                "decision": DECISION_SKIP_DATA,
                "reason": getattr(resolved_symbol, "reason", "SYMBOL_NOT_FOUND"),
            }
        )
        return base

    bid = _valor(tick, "bid")
    ask = _valor(tick, "ask")
    base["bid"] = bid
    base["ask"] = ask
    point = _valor(info, "point", getattr(resolved_symbol, "point", None))
    if bid is not None and ask is not None:
        base["spread"] = float(ask) - float(bid)
        base["mercado_abierto"] = float(bid) > 0 and float(ask) > 0

    market_filters = evaluate_trade_filters(
        symbol_info=info,
        tick=tick,
        model_validated=True,
    )
    if not market_filters.passed:
        base.update({"decision": market_filters.decision, "reason": market_filters.reason})
        return base

    modelo_validado = _modelo_validado_mt5_v2(activo)
    base["modelo_validado"] = modelo_validado
    if not modelo_validado:
        base.update({"decision": DECISION_WAIT, "reason": "MODELO_VALIDADO_NO"})
        return base

    base.update({"decision": DECISION_WAIT, "reason": "SIN_SENAL_VALIDADA"})
    return base


def ejecutar_mt5_dryrun_multi(raiz_proyecto=None):
    from src.analytics_logger import write_dryrun_log
    from src.execution_demo import evaluate_demo_account
    from src.symbols import TARGET_LOGICAL_SYMBOLS, discover_symbols

    raiz = Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[1]
    credenciales, error = _credenciales_mt5()
    if error:
        return {
            "ok": False,
            "error": error,
            "cuenta": "N/A",
            "broker": "XM",
            "balance": None,
            "resultados": [],
            "log_path": None,
            "trading_enabled": False,
            "ordenes_enviadas": 0,
        }

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return {
            "ok": False,
            "error": "MetaTrader5 no esta instalado.",
            "cuenta": "N/A",
            "broker": "XM",
            "balance": None,
            "resultados": [],
            "log_path": None,
            "trading_enabled": False,
            "ordenes_enviadas": 0,
        }

    inicializado = False
    try:
        inicializado = mt5.initialize()
        if not inicializado:
            return {
                "ok": False,
                "error": f"No se pudo inicializar MT5: {mt5.last_error()}",
                "cuenta": "N/A",
                "broker": "XM",
                "balance": None,
                "resultados": [],
                "log_path": None,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
            }

        conectado = mt5.login(
            credenciales["login"],
            password=credenciales["password"],
            server=credenciales["server"],
        )
        if not conectado:
            return {
                "ok": False,
                "error": f"No se pudo conectar a la cuenta MT5: {mt5.last_error()}",
                "cuenta": "N/A",
                "broker": "XM",
                "balance": None,
                "resultados": [],
                "log_path": None,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
            }

        account_info = mt5.account_info()
        if account_info is None:
            return {
                "ok": False,
                "error": "No se pudo obtener account_info().",
                "cuenta": "N/A",
                "broker": "XM",
                "balance": None,
                "resultados": [],
                "log_path": None,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
            }

        cuenta = _modo_cuenta(mt5, account_info)
        balance = float(_valor(account_info, "balance", 0) or 0)
        demo_gate = evaluate_demo_account(mt5, account_info)
        if not demo_gate.allowed and cuenta == "REAL":
            return {
                "ok": False,
                "error": "CUENTA REAL DETECTADA - CONEXION BLOQUEADA",
                "cuenta": cuenta,
                "broker": "XM",
                "balance": balance,
                "resultados": [],
                "log_path": None,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
                "cuenta_real_bloqueada": True,
            }

        if not demo_gate.allowed:
            return {
                "ok": False,
                "error": f"La cuenta no es DEMO. Tipo detectado: {cuenta}",
                "cuenta": cuenta,
                "broker": "XM",
                "balance": balance,
                "resultados": [],
                "log_path": None,
                "trading_enabled": False,
                "ordenes_enviadas": 0,
            }

        resolved = discover_symbols(mt5, TARGET_LOGICAL_SYMBOLS)
        resultados = []
        for activo in TARGET_LOGICAL_SYMBOLS:
            item = resolved[activo]
            info = mt5.symbol_info(item.mt5_symbol) if item.found else None
            tick = mt5.symbol_info_tick(item.mt5_symbol) if item.found else None
            atr_data = _leer_atr_local(activo)
            resultados.append(
                evaluar_dryrun_activo_v2(
                    activo=activo,
                    resolved_symbol=item,
                    info=info,
                    tick=tick,
                    balance=balance,
                    atr_data=atr_data,
                )
            )

        log_path = write_dryrun_log(resultados, raiz)
        return {
            "ok": True,
            "error": "",
            "cuenta": cuenta,
            "broker": "XM",
            "balance": balance,
            "resultados": resultados,
            "log_path": log_path,
            "trading_enabled": False,
            "demo_execution_enabled": False,
            "ordenes_enviadas": 0,
        }
    finally:
        if inicializado:
            mt5.shutdown()


def ejecutar_mt5_preview_all(raiz_proyecto=None):
    return ejecutar_mt5_dryrun_multi(raiz_proyecto)
