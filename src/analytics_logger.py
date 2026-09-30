import csv
from datetime import datetime, timezone
from pathlib import Path


DRYRUN_LOG_COLUMNS = [
    "timestamp",
    "activo_logico",
    "simbolo_mt5",
    "timeframe",
    "bid",
    "ask",
    "spread",
    "signal",
    "confidence",
    "modelo",
    "model_version",
    "ATR",
    "swing_high",
    "swing_low",
    "entry",
    "SL",
    "TP",
    "SL_mode",
    "TP_mode",
    "RR",
    "risk_pct",
    "risk_amount",
    "volume_theoretical",
    "volume_allowed",
    "estimated_loss",
    "decision",
    "reason",
]


def utc_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dryrun_log_path(raiz_proyecto):
    return Path(raiz_proyecto) / "logs" / "mt5_dryrun_v2.csv"


def write_dryrun_log(rows, raiz_proyecto):
    path = dryrun_log_path(raiz_proyecto)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DRYRUN_LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in DRYRUN_LOG_COLUMNS})

    return path
