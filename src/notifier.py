import os
import sys
from pathlib import Path


def _cargar_env():
    raiz = Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(raiz / ".env")


def testing_mode_enabled():
    value = os.getenv("FOREX_ML_TESTING", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    # No usar "unittest" in sys.modules: numpy.testing lo importa en produccion
    # y eso bloqueaba todos los mensajes reales. tests/__init__.py fija la variable.
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    main_name = getattr(main_spec, "name", "") or ""
    return "pytest" in sys.modules or main_name.startswith("unittest")


def enviar_telegram(mensaje):
    if testing_mode_enabled():
        return False

    _cargar_env()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram no configurado.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensaje,
        "disable_web_page_preview": True,
    }

    try:
        import requests

        respuesta = requests.post(url, data=payload, timeout=10)
        if not respuesta.ok:
            print(f"Error Telegram: HTTP {respuesta.status_code}")
            return False
    except ImportError:
        print("Error Telegram: falta instalar requests.")
        return False
    except Exception:
        print("Error Telegram: no se pudo enviar el mensaje.")
        return False

    return True
