import importlib.util
import io
import json
import logging
import os
import platform
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


REQUIREMENTS_PRINCIPALES = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "plotly",
    "yfinance",
    "ta",
    "joblib",
    "requests",
    "dotenv",
]


def _ok(nombre, estado, detalle=""):
    return {
        "nombre": nombre,
        "ok": bool(estado),
        "detalle": detalle,
    }


def _import_disponible(nombre):
    return importlib.util.find_spec(nombre) is not None


def _cargar_env(raiz):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(Path(raiz) / ".env")


def _check_python():
    return _ok("Python", True, platform.python_version())


def _check_requirements():
    faltantes = [
        paquete
        for paquete in REQUIREMENTS_PRINCIPALES
        if not _import_disponible(paquete)
    ]
    if faltantes:
        return _ok("Requirements", False, "Faltan: " + ", ".join(faltantes))
    return _ok("Requirements", True, "OK")


def _check_archivo_json(ruta, nombre):
    if not ruta.exists():
        return _ok(nombre, False, f"No existe: {ruta}")

    try:
        json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return _ok(nombre, False, "JSON no legible")

    return _ok(nombre, True, "OK")


def _check_modelo_paper(raiz):
    ruta_modelo = Path(raiz) / "models" / "paper" / "gold_logistic_long.pkl"
    if not ruta_modelo.exists():
        return _ok("Modelo PAPER", False, "No existe gold_logistic_long.pkl")
    return _ok("Modelo PAPER", True, "OK")


def _check_metadata_modelo(raiz):
    ruta_metadata = (
        Path(raiz) / "models" / "paper" / "gold_logistic_long_metadata.json"
    )
    return _check_archivo_json(ruta_metadata, "Metadata modelo")


def _check_logs(raiz):
    ruta_logs = Path(raiz) / "logs"
    if not ruta_logs.exists() or not ruta_logs.is_dir():
        return _ok("Logs", False, "No existe carpeta logs")

    if not os.access(ruta_logs, os.R_OK | os.W_OK):
        return _ok("Logs", False, "Carpeta logs no accesible")

    return _ok("Logs", True, "OK")


def _check_telegram(raiz):
    _cargar_env(raiz)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return _ok("Telegram", False, "No configurado")

    try:
        import requests

        respuesta = requests.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=10,
        )
        if not respuesta.ok:
            return _ok("Telegram", False, f"HTTP {respuesta.status_code}")
    except Exception:
        return _ok("Telegram", False, "Sin conexion disponible")

    return _ok("Telegram", True, "OK")


def _check_datos():
    try:
        import yfinance as yf

        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            datos = yf.download(
                "GC=F",
                interval="1h",
                period="5d",
                progress=False,
                auto_adjust=False,
                timeout=10,
            )
        logging.disable(logging.NOTSET)
        if datos.empty:
            return _ok("Datos", False, "Yahoo Finance devolvio datos vacios")
    except Exception:
        logging.disable(logging.NOTSET)
        return _ok("Datos", False, "Fuente de datos no accesible")

    return _ok("Datos", True, "OK")


def ejecutar_health_check(raiz):
    raiz = Path(raiz)
    checks = [
        _check_python(),
        _check_requirements(),
        _check_modelo_paper(raiz),
        _check_metadata_modelo(raiz),
        _check_archivo_json(raiz / "paper" / "start.json", "Paper start"),
        _check_archivo_json(raiz / "paper" / "state.json", "Paper state"),
        _check_logs(raiz),
        _check_telegram(raiz),
        _check_datos(),
    ]

    return {
        "checks": checks,
        "system_ready": all(check["ok"] for check in checks),
    }
