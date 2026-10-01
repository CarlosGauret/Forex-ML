from datetime import datetime, timezone
from pathlib import Path
import os
import traceback

import pandas as pd

from config import ACTIVOS
from src.paper_trader import COLUMNAS_PRECIO, _corregir_multiindex, _normalizar_indice
from src.yfinance_cache import ensure_yfinance_cache


def _can_write(path):
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, ""
    except Exception as error:
        return False, str(error)


def run_gold_data_check(root):
    import yfinance as yf

    root = Path(root)
    cache_dir = ensure_yfinance_cache(root)
    data_dir = root / "data"
    gold_dir = data_dir / "gold"
    logs_dir = root / "logs"
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    ticker = ACTIVOS["GOLD"]["ticker"]

    checks = {}
    for name, path in {
        "project": root,
        "data": data_dir,
        "gold_data": gold_dir,
        "yfinance_cache": cache_dir,
        "logs": logs_dir,
    }.items():
        can_write, error = _can_write(path)
        checks[name] = {
            "path": str(path),
            "exists": path.exists(),
            "readable": os.access(path, os.R_OK),
            "writable": can_write and os.access(path, os.W_OK),
            "error": error,
        }

    result = {
        "cwd": os.getcwd(),
        "project_root": str(root),
        "timestamp_utc": now_utc.isoformat(),
        "ticker": ticker,
        "cache_dir": str(cache_dir),
        "checks": checks,
        "rows": 0,
        "last_bar": None,
        "last_closed_bar": None,
        "data_status": "ERROR",
        "error": "",
        "traceback": "",
    }

    try:
        data = yf.download(
            ticker,
            interval="1h",
            period="60d",
            progress=False,
            auto_adjust=False,
        )
        result["rows"] = int(len(data))
        if data.empty:
            result["error"] = "DESCARGA VACIA"
            return result

        data = _corregir_multiindex(data)
        if "Volume" not in data.columns:
            data["Volume"] = 0
        missing = [column for column in COLUMNAS_PRECIO if column not in data.columns]
        if missing:
            result["error"] = f"Faltan columnas: {', '.join(missing)}"
            return result

        data = data[COLUMNAS_PRECIO].dropna().sort_index()
        data = _normalizar_indice(data)
        last_bar = data.index[-1]
        closed = data[data.index + pd.Timedelta(hours=1) <= now_utc.replace(tzinfo=None)]
        result["last_bar"] = last_bar.isoformat()
        result["last_closed_bar"] = None if closed.empty else closed.index[-1].isoformat()
        if closed.empty:
            result["data_status"] = "STALE"
            result["error"] = "No hay velas H1 cerradas disponibles."
        else:
            age_minutes = (now_utc.replace(tzinfo=None) - closed.index[-1] - pd.Timedelta(hours=1)).total_seconds() / 60
            result["data_status"] = "LIVE" if age_minutes <= 180 else "STALE"
        return result
    except Exception as error:
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        return result
