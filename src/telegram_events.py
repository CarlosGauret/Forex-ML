import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


EVENT_COLUMNS = [
    "EVENT_ID",
    "EVENT_TYPE",
    "ASSET",
    "CONFIG_ID",
    "SENT_AT_UTC",
]
LIMA_TZ = ZoneInfo("America/Lima")


def _root(root=None):
    return Path(root) if root is not None else Path(__file__).resolve().parents[1]


def event_log_path(root=None):
    return _root(root) / "logs" / "telegram_events.csv"


def error_log_path(root=None):
    return _root(root) / "logs" / "telegram_errors.log"


def _ensure_event_log(root=None):
    path = event_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=EVENT_COLUMNS).writeheader()
    return path


def _read_events(root=None):
    path = _ensure_event_log(root)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def _append_event(root, row):
    path = _ensure_event_log(root)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writerow({column: row.get(column, "") for column in EVENT_COLUMNS})


def _log_telegram_error(root, event_id, reason):
    path = error_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    safe_reason = str(reason).replace("\n", " ")[:300]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} EVENT_ID={event_id} ERROR={safe_reason}\n")


def _load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_root() / ".env")


def telegram_configured():
    _load_env()
    return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))


def was_event_sent(root, event_id):
    return any(row.get("EVENT_ID") == event_id for row in _read_events(root))


def send_telegram_event(
    root,
    event_id,
    event_type,
    message,
    asset="",
    config_id="",
    send_func=None,
    now=None,
):
    if was_event_sent(root, event_id):
        return {"sent": False, "duplicate": True, "error": ""}

    try:
        if send_func is None:
            from src.notifier import enviar_telegram

            send_func = enviar_telegram
        ok = bool(send_func(message))
    except Exception as error:
        _log_telegram_error(root, event_id, error)
        return {"sent": False, "duplicate": False, "error": str(error)}

    if not ok:
        _log_telegram_error(root, event_id, "send_func returned False")
        return {"sent": False, "duplicate": False, "error": "TELEGRAM_SEND_FAILED"}

    sent_at = now or datetime.now(timezone.utc)
    if not isinstance(sent_at, datetime):
        sent_at = pd.to_datetime(sent_at, utc=True).to_pydatetime()
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)
    _append_event(
        root,
        {
            "EVENT_ID": event_id,
            "EVENT_TYPE": event_type,
            "ASSET": asset,
            "CONFIG_ID": config_id,
            "SENT_AT_UTC": sent_at.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
        },
    )
    return {"sent": True, "duplicate": False, "error": ""}


def _fmt_price(value):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.5f}"


def _fmt_float(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def _fmt_money(value):
    if value is None or pd.isna(value):
        return "$0.00"
    value = float(value)
    if value > 0:
        return f"+${value:.2f}"
    if value < 0:
        return f"-${abs(value):.2f}"
    return "$0.00"


def _fmt_currency(value):
    if value is None or pd.isna(value):
        return "$0.00"
    return f"${float(value):.2f}"


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _to_utc_iso(value):
    return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d %H:%M UTC")


def config_label(config_id, asset=None):
    config_id = str(config_id or "")
    asset = str(asset or "").upper()
    if asset == "GOLD":
        if "T055" in config_id:
            return "GOLD T055"
        if "T060" in config_id:
            return "GOLD T060"
    if config_id:
        return config_id
    return asset or "N/A"


def _direction(row):
    return str(row.get("DIRECCION", row.get("direction", "LONG"))).upper()


def _event_asset(row):
    return str(row.get("ACTIVO", row.get("asset", ""))).upper()


def format_new_signal_message(row):
    asset = _event_asset(row)
    lines = [
        "🟡 NUEVA SEÑAL PAPER",
        "",
        f"Activo: {asset}",
        f"Direccion: {_direction(row)}",
    ]
    if asset == "GOLD":
        lines.extend(["", f"Configuracion: {config_label(row.get('CONFIG_ID'), asset)}"])
    lines.extend(
        [
            f"Probabilidad: {_fmt_float(row.get('PROBABILIDAD'), 4)}",
            f"Threshold: {_fmt_float(row.get('THRESHOLD'), 2)}",
            f"Hora vela: {_to_utc_iso(row.get('FECHA_SIGNAL'))}",
            "",
            "Estado:",
            "EVALUANDO RIESGO",
            "",
            "PAPER / SIMULACION",
        ]
    )
    return "\n".join(lines)


def format_open_message(row):
    asset = _event_asset(row)
    capital = float(row.get("CAPITAL_ANTES", 0) or 0)
    risk_usd = row.get("RISK_USD")
    if risk_usd is None or pd.isna(risk_usd):
        risk_usd = capital * 0.01 if capital else None
    lines = [
        "🟢 OPERACION PAPER ABIERTA",
        "",
        "PAPER / SIMULACION",
        "",
        f"Activo: {asset}",
        f"Direccion: {_direction(row)}",
    ]
    if asset == "GOLD":
        lines.extend(["", f"Configuracion: {config_label(row.get('CONFIG_ID'), asset)}"])
    lines.extend(
        [
            "",
            f"Entrada: {_fmt_price(row.get('ENTRY'))}",
            f"Stop Loss: {_fmt_price(row.get('STOP'))}",
            f"Take Profit: {_fmt_price(row.get('TP'))}",
            "",
            f"Volumen simulado: {_fmt_float(row.get('VOLUME'), 2)}",
            "Riesgo objetivo: 1.00%",
            f"Riesgo USD: {_fmt_currency(risk_usd)}",
            "",
            "Capital antes:",
            _fmt_currency(capital),
            "",
            "Hora:",
            _to_utc_iso(row.get("FECHA_ENTRADA")),
        ]
    )
    return "\n".join(lines)


def format_close_message(row):
    pnl = float(row.get("CAPITAL_DESPUES", 0) or 0) - float(row.get("CAPITAL_ANTES", 0) or 0)
    header = "✅ OPERACION PAPER CERRADA" if pnl >= 0 else "🔴 OPERACION PAPER CERRADA"
    reason = row.get("MOTIVO_SALIDA") or row.get("_RESULTADO_NOTIFICACION") or row.get("RESULTADO") or "N/A"
    return "\n".join(
        [
            header,
            "",
            "PAPER / SIMULACION",
            "",
            f"Activo: {_event_asset(row)}",
            f"Direccion: {_direction(row)}",
            "",
            f"Entrada: {_fmt_price(row.get('ENTRY'))}",
            f"Salida: {_fmt_price(row.get('EXIT', row.get('_EXIT')))}",
            "",
            "Resultado:",
            _fmt_money(pnl),
            "",
            "Resultado R:",
            f"{float(row.get('R_NETO', 0) or 0):+.2f}R",
            "",
            "Motivo:",
            str(reason).replace("_", " "),
            "",
            "Capital anterior:",
            _fmt_currency(row.get("CAPITAL_ANTES")),
            "",
            "Capital nuevo:",
            _fmt_currency(row.get("CAPITAL_DESPUES")),
            "",
            "Duracion:",
            _duration_hours(row.get("FECHA_ENTRADA"), row.get("FECHA_SALIDA")),
        ]
    )


def format_skip_risk_message(row):
    return "\n".join(
        [
            "⚠️ SEÑAL NO EJECUTADA",
            "",
            f"Activo: {_event_asset(row)}",
            f"Direccion: {_direction(row)}",
            "",
            "Motivo:",
            str(row.get("REASON", "VOLUMEN MINIMO DEL BROKER SUPERA EL RIESGO PERMITIDO")),
            "",
            "Riesgo objetivo:",
            "1.00%",
            "",
            "Lote requerido:",
            _fmt_float(row.get("VOLUME_THEORETICAL"), 3),
            "",
            "Lote minimo:",
            _fmt_float(row.get("VOLUME_MIN"), 2),
            "",
            "No se abre operacion PAPER.",
        ]
    )


def format_critical_error_message(asset, status, detail=""):
    lines = [
        "🚨 FOREX ML - ERROR",
        "",
        str(asset or "PAPER"),
        f"DATA STATUS: {status}",
    ]
    if detail:
        lines.extend(["", str(detail)])
    lines.extend(["", "No se generaron señales.", "Ordenes enviadas: 0"])
    return "\n".join(lines)


def _duration_hours(start, end):
    try:
        delta = pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)
        hours = int(round(delta.total_seconds() / 3600))
        return f"{hours} horas"
    except Exception:
        return "N/A"


def notify_signal(root, row, send_func=None):
    if str(row.get("SIGNAL", "")).upper() not in {"LONG", "SHORT"}:
        return {"sent": False, "duplicate": False, "error": "WAIT_IGNORED"}
    asset = _event_asset(row)
    event_id = f"{asset}_{pd.to_datetime(row.get('FECHA_SIGNAL')).strftime('%Y%m%dT%H%M%S')}_SIGNAL_{_direction(row)}_{row.get('CONFIG_ID', '')}"
    return send_telegram_event(
        root,
        event_id,
        "SIGNAL",
        format_new_signal_message(row),
        asset=asset,
        config_id=row.get("CONFIG_ID", ""),
        send_func=send_func,
    )


def notify_open(root, row, send_func=None):
    asset = _event_asset(row)
    event_id = f"{asset}_{pd.to_datetime(row.get('FECHA_ENTRADA')).strftime('%Y%m%dT%H%M%S')}_OPEN_{row.get('CONFIG_ID', '')}"
    return send_telegram_event(
        root,
        event_id,
        "OPEN",
        format_open_message(row),
        asset=asset,
        config_id=row.get("CONFIG_ID", ""),
        send_func=send_func,
    )


def notify_close(root, row, send_func=None):
    asset = _event_asset(row)
    trade_id = row.get("TRADE_ID") or f"{row.get('CONFIG_ID', '')}_{row.get('FECHA_ENTRADA', '')}_{row.get('FECHA_SALIDA', '')}"
    event_id = f"{asset}_{trade_id}_CLOSE"
    return send_telegram_event(
        root,
        event_id,
        "CLOSE",
        format_close_message(row),
        asset=asset,
        config_id=row.get("CONFIG_ID", ""),
        send_func=send_func,
    )


def notify_skip_risk(root, row, send_func=None):
    asset = _event_asset(row)
    signal_id = row.get("SIGNAL_ID") or f"{row.get('CONFIG_ID', '')}_{row.get('FECHA_SIGNAL', '')}"
    event_id = f"{asset}_{signal_id}_SKIP_RISK"
    return send_telegram_event(
        root,
        event_id,
        "SKIP_RISK",
        format_skip_risk_message(row),
        asset=asset,
        config_id=row.get("CONFIG_ID", ""),
        send_func=send_func,
    )


def notify_critical_error(root, asset, status, detail="", send_func=None):
    event_id = f"{asset or 'PAPER'}_{status}_{str(detail)[:80]}_ERROR"
    return send_telegram_event(
        root,
        event_id,
        "ERROR_CRITICAL",
        format_critical_error_message(asset, status, detail),
        asset=asset,
        send_func=send_func,
    )


def _read_csv(path, columns=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path)


def _local_dates(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(LIMA_TZ).dt.date


def _rows_for_lima_date(df, date_column, target_date):
    if df.empty or date_column not in df.columns:
        return df.iloc[0:0].copy()
    local_dates = _local_dates(df[date_column])
    return df[local_dates == target_date].copy()


def _trade_rows(root, target_date):
    root = _root(root)
    sources = [
        (root / "paper" / "trades.csv", "GOLD"),
        (root / "paper" / "eurusd" / "trades.csv", "EURUSD"),
    ]
    rows = []
    for path, asset in sources:
        df = _read_csv(path)
        day = _rows_for_lima_date(df, "FECHA_SALIDA", target_date)
        for _, item in day.iterrows():
            row = item.to_dict()
            row["ACTIVO"] = row.get("ACTIVO") or asset
            row["_LABEL"] = config_label(row.get("CONFIG_ID"), row["ACTIVO"])
            row["_PNL"] = float(row.get("CAPITAL_DESPUES", 0) or 0) - float(row.get("CAPITAL_ANTES", 0) or 0)
            rows.append(row)
    return rows


def _signal_count(root, target_date):
    root = _root(root)
    total = 0
    for path in [root / "paper" / "signals.csv", root / "paper" / "eurusd" / "signals.csv"]:
        df = _read_csv(path)
        day = _rows_for_lima_date(df, "FECHA_SIGNAL", target_date)
        if not day.empty and "SIGNAL" in day.columns:
            total += int(day["SIGNAL"].astype(str).str.upper().isin(["LONG", "SHORT"]).sum())
    return total


def _skip_count(root, target_date):
    root = _root(root)
    total = 0
    path = root / "paper" / "eurusd" / "signals.csv"
    df = _read_csv(path)
    day = _rows_for_lima_date(df, "FECHA_SIGNAL", target_date)
    if not day.empty and "EXECUTION" in day.columns:
        total += int((day["EXECUTION"].astype(str).str.upper() == "SKIP_RISK").sum())
    return total


def _error_count(root, target_date):
    rows = _read_events(root)
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    day = _rows_for_lima_date(df, "SENT_AT_UTC", target_date)
    if day.empty or "EVENT_TYPE" not in day.columns:
        return 0
    return int((day["EVENT_TYPE"] == "ERROR_CRITICAL").sum())


def _capital_lines(root):
    root = _root(root)
    items = []
    for path in [root / "paper" / "equity.csv", root / "paper" / "eurusd" / "equity.csv"]:
        df = _read_csv(path)
        if df.empty:
            continue
        for config_id, group in df.groupby("CONFIG_ID", dropna=False):
            row = group.iloc[-1]
            label = config_label(config_id, "GOLD" if "GOLD" in str(config_id) else "EURUSD")
            items.append((label, float(row["CAPITAL"])))
    return items


def build_daily_report(root, now=None):
    now_ts = pd.to_datetime(now or datetime.now(timezone.utc), utc=True).tz_convert(LIMA_TZ)
    target_date = now_ts.date()
    trades = _trade_rows(root, target_date)
    wins = sum(1 for trade in trades if trade["_PNL"] > 0)
    losses = sum(1 for trade in trades if trade["_PNL"] < 0)
    net = sum(trade["_PNL"] for trade in trades)
    signals = _signal_count(root, target_date)
    skips = _skip_count(root, target_date)
    errors = _error_count(root, target_date)

    lines = [
        "📊 FOREX ML - RESUMEN DIARIO",
        now_ts.strftime("%d/%m/%Y"),
        "",
        "PAPER / SIMULACION",
        "",
        f"Operaciones cerradas: {len(trades)}",
    ]
    if trades:
        lines.extend([f"Ganadas: {wins}", f"Perdidas: {losses}", ""])
        by_label = {}
        for trade in trades:
            by_label[trade["_LABEL"]] = by_label.get(trade["_LABEL"], 0.0) + trade["_PNL"]
        for label, pnl in by_label.items():
            lines.append(f"{label}:")
            lines.append(_fmt_money(pnl))
            lines.append("")
        best = max(trades, key=lambda item: item["_PNL"])
        worst = min(trades, key=lambda item: item["_PNL"])
        lines.extend(
            [
                "--------------------",
                "",
                "Resultado neto del dia:",
                _fmt_money(net),
                "",
                "Mejor operacion:",
                f"{best['_LABEL']} {_fmt_money(best['_PNL'])}",
                "",
                "Peor operacion:",
                f"{worst['_LABEL']} {_fmt_money(worst['_PNL'])}",
            ]
        )
    else:
        lines.extend(["Resultado neto: $0.00", "", "No inventar ganancias ni perdidas."])

    capital_items = _capital_lines(root)
    if capital_items:
        lines.extend(["", "--------------------", "", "Capital PAPER:", ""])
        for label, capital in capital_items:
            lines.append(f"{label}:")
            lines.append(_fmt_money(capital).replace("+", ""))
            lines.append("")

    lines.extend(
        [
            "--------------------",
            "",
            f"Senales generadas: {signals}",
            f"SKIP por riesgo: {skips}",
            f"Errores criticos: {errors}",
        ]
    )
    return "\n".join(lines)


def send_daily_report(root, now=None, send_func=None):
    now_ts = pd.to_datetime(now or datetime.now(timezone.utc), utc=True).tz_convert(LIMA_TZ)
    event_id = f"DAILY_{now_ts.strftime('%Y%m%d')}"
    message = build_daily_report(root, now=now_ts)
    result = send_telegram_event(
        root,
        event_id,
        "DAILY",
        message,
        asset="ALL",
        config_id="DAILY",
        send_func=send_func,
        now=now_ts.tz_convert(timezone.utc).to_pydatetime(),
    )
    result["message"] = message
    return result


def telegram_status(root, now=None):
    now_ts = pd.to_datetime(now or datetime.now(timezone.utc), utc=True).tz_convert(LIMA_TZ)
    today = now_ts.date()
    rows = _read_events(root)
    last = rows[-1] if rows else {}
    if rows:
        df = pd.DataFrame(rows)
        today_rows = _rows_for_lima_date(df, "SENT_AT_UTC", today)
        events_today = len(today_rows)
        daily_rows = df[df["EVENT_TYPE"] == "DAILY"]
        last_daily = daily_rows.iloc[-1]["SENT_AT_UTC"] if not daily_rows.empty else ""
    else:
        events_today = 0
        last_daily = ""
    return {
        "configured": telegram_configured(),
        "last_event_id": last.get("EVENT_ID", ""),
        "last_event_type": last.get("EVENT_TYPE", ""),
        "last_event_at": last.get("SENT_AT_UTC", ""),
        "events_today": events_today,
        "last_daily": last_daily,
        "event_log": str(event_log_path(root)),
    }
