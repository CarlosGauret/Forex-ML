"""Dispatcher de ejecucion automatica MT5 DEMO para forwards congelados.

PAPER es la fuente de senal. Este modulo solo lee sus salidas y, si el flag
demo lo permite, traduce senales frescas autorizadas a ordenes MT5 demo.
"""

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from src.execution_demo import DEMO_DISABLED_REASON, evaluate_demo_account
from src.live_executor import RETCODE_DONE, RETCODE_PLACED, _filling_mode, mt5_session
from src.risk import BUY, RiskLimits, RiskManager
from src.symbols import discover_symbols


AUTHORIZED_EURUSD_CONFIG = "EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065"
AUTHORIZED_GOLD_CONFIGS = (
    "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T055",
    "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060",
)
AUTHORIZED_CONFIGS = (AUTHORIZED_EURUSD_CONFIG, *AUTHORIZED_GOLD_CONFIGS)
EURUSD_EXPECTED_HASH = "b7d2136797e433bb3f2a59170b77569570c60c58f0cf7f75ba19f4ae17aa8a6f"
MAGIC_BY_ASSET = {"EURUSD": 561001, "GOLD": 561002}
COMMENT_BY_ASSET = {"EURUSD": "FOREX_ML_EURUSD_DEMO", "GOLD": "FOREX_ML_GOLD_DEMO"}
MAX_DEMO_POSITIONS = 2
PERU_TZ = timezone(timedelta(hours=-5))
LOG_COLUMNS = [
    "timestamp",
    "status",
    "reason",
    "execution_id",
    "config_ids",
    "asset",
    "symbol",
    "direction",
    "bar_timestamp",
    "volume",
    "price",
    "sl",
    "tp",
    "risk_usd",
    "ticket",
    "deal",
    "retcode",
    "comment",
    "event",
    "profit",
    "commission",
    "swap",
]


@dataclass(frozen=True)
class ForwardSignal:
    config_id: str
    asset: str
    direction: str
    bar_timestamp: str
    signal: str
    execution: str
    probability: float | None
    atr: float | None
    entry: float | None
    sl: float | None
    tp: float | None
    reason: str


@dataclass(frozen=True)
class DemoDecision:
    execution_id: str
    config_ids: tuple[str, ...]
    asset: str
    symbol: str = ""
    direction: str = BUY
    bar_timestamp: str = ""
    signal_age_minutes: float | None = None
    status: str = "WAIT"
    reason: str = "WAIT"
    price: float | None = None
    sl: float | None = None
    tp: float | None = None
    volume: float | None = None
    theoretical_volume: float | None = None
    risk_usd: float | None = None
    estimated_loss: float | None = None
    order_check_retcode: int | None = None
    order_check_comment: str = ""
    ticket: int | None = None
    deal: int | None = None
    retcode: int | None = None
    orders_sent: int = 0
    bid: float | None = None
    ask: float | None = None
    signal_source: str = ""
    sl_distance: float | None = None
    tp_distance: float | None = None
    contract_size: float | None = None
    volume_min: float | None = None
    volume_step: float | None = None
    tick_size: float | None = None
    tick_value: float | None = None


def _value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _log_path(root):
    return Path(root) / "live" / "forward_demo_orders.csv"


def _read_log(root):
    path = _log_path(root)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_log(root, row, now=None):
    path = _log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    record = {key: row.get(key, "") for key in LOG_COLUMNS}
    record["timestamp"] = (now or _now_utc()).isoformat()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)


def _parse_time(value):
    if not value:
        return None
    return pd.to_datetime(value, utc=True).to_pydatetime()


def execution_id(asset, symbol, bar_timestamp, direction):
    return f"FORWARD_DEMO|{asset}|{symbol}|{pd.to_datetime(bar_timestamp).isoformat()}|{direction}"


def _read_latest_signal_csv(path, allowed_configs):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("CONFIG_ID") in allowed_configs:
                rows.append(row)
    latest = {}
    for row in rows:
        key = row["CONFIG_ID"]
        if key not in latest or pd.to_datetime(row["FECHA_SIGNAL"]) > pd.to_datetime(latest[key]["FECHA_SIGNAL"]):
            latest[key] = row
    return [signal_from_row(row) for row in latest.values()]


def signal_from_row(row):
    atr_key = "ATR" if "ATR" in row else "ATR_SIGNAL"
    sl_key = "STOP" if "STOP" in row else "SL"
    direction = str(row.get("DIRECCION", "LONG")).upper()
    direction = "BUY" if direction in ("LONG", "BUY") else "SELL" if direction in ("SHORT", "SELL") else direction
    signal = str(row.get("SIGNAL", "WAIT"))
    execution = str(row.get("EXECUTION") or ("OPEN" if signal == "LONG" else "WAIT"))
    return ForwardSignal(
        config_id=row.get("CONFIG_ID", ""),
        asset=row.get("ACTIVO", ""),
        direction=direction,
        bar_timestamp=pd.to_datetime(row.get("FECHA_SIGNAL")).isoformat(),
        signal=signal,
        execution=execution,
        probability=_float_or_none(row.get("PROBABILIDAD")),
        atr=_float_or_none(row.get(atr_key)),
        entry=_float_or_none(row.get("ENTRY")),
        sl=_float_or_none(row.get(sl_key)),
        tp=_float_or_none(row.get("TP")),
        reason=str(row.get("REASON", "")),
    )


def _float_or_none(value):
    if value in (None, "", "nan"):
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _signal_source(asset):
    return "GC=F" if asset == "GOLD" else "MT5 XM"


def _broker_entry_price(direction, bid, ask):
    return ask if direction == "BUY" else bid


def _mt5_order_type(mt5, direction):
    return mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL


def _broker_levels(signal, price, digits):
    atr = float(signal.atr or 0)
    stop_distance = atr * 1.0
    take_profit_distance = atr * 2.0
    if signal.direction == "BUY":
        sl = price - stop_distance
        tp = price + take_profit_distance
    else:
        sl = price + stop_distance
        tp = price - take_profit_distance
    return round(sl, digits), round(tp, digits), stop_distance, take_profit_distance


def _broker_metadata(info):
    return {
        "contract_size": _value(info, "trade_contract_size", _value(info, "contract_size")),
        "volume_min": _value(info, "volume_min"),
        "volume_step": _value(info, "volume_step"),
        "tick_size": _value(info, "trade_tick_size", _value(info, "tick_size")),
        "tick_value": _value(info, "trade_tick_value", _value(info, "tick_value")),
    }


def load_forward_signals(root):
    root = Path(root)
    signals = []
    signals += _read_latest_signal_csv(root / "paper" / "eurusd" / "signals.csv", [AUTHORIZED_EURUSD_CONFIG])
    signals += _read_latest_signal_csv(root / "paper" / "signals.csv", AUTHORIZED_GOLD_CONFIGS)
    return signals


def dedupe_signals(signals):
    grouped = {}
    for signal in signals:
        key = (signal.asset, signal.bar_timestamp, signal.direction)
        if signal.asset == "GOLD":
            grouped.setdefault(key, []).append(signal)
        else:
            grouped[(signal.asset, signal.config_id, signal.bar_timestamp, signal.direction)] = [signal]
    deduped = []
    for items in grouped.values():
        items = sorted(items, key=lambda item: item.config_id)
        primary = next((item for item in items if item.signal == "LONG" and item.execution == "OPEN"), items[-1])
        deduped.append((primary, tuple(item.config_id for item in items)))
    return deduped


def validate_eurusd_hash(root):
    path = Path(root) / "models" / "paper" / "eurusd_long_rf_actual_t065_metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata.get("hash_modelo") == EURUSD_EXPECTED_HASH


def _demo_positions(mt5):
    allowed = set(MAGIC_BY_ASSET.values())
    comments = set(COMMENT_BY_ASSET.values())
    return [
        position for position in (mt5.positions_get() or [])
        if _value(position, "magic") in allowed and _value(position, "comment") in comments
    ]


def _positions_by_symbol(mt5):
    return {str(_value(position, "symbol", "")).upper(): position for position in _demo_positions(mt5)}


def _daily_loss(mt5, now=None):
    now = now or _now_utc()
    peru_now = now.astimezone(PERU_TZ)
    start_peru = peru_now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_peru.astimezone(timezone.utc)
    deals = mt5.history_deals_get(start_utc, now + timedelta(minutes=1)) or []
    realized = 0.0
    for deal in deals:
        if _value(deal, "magic") not in set(MAGIC_BY_ASSET.values()):
            continue
        for field in ("profit", "commission", "swap", "fee"):
            realized += float(_value(deal, field, 0) or 0)
    floating = sum(float(_value(p, "profit", 0) or 0) for p in _demo_positions(mt5))
    return max(0.0, -(realized + floating))


def _executed_ids(root):
    return {
        row.get("execution_id")
        for row in _read_log(root)
        if row.get("status") in {"SENT", "CONFIRMED", "CLOSED"} and row.get("execution_id")
    }


def reconcile(mt5, root):
    positions = _demo_positions(mt5)
    open_tickets = {str(_value(position, "ticket")) for position in positions}
    sent_rows = [row for row in _read_log(root) if row.get("status") in {"SENT", "CONFIRMED"}]
    missing = [
        row for row in sent_rows
        if row.get("ticket") and row.get("ticket") not in open_tickets
    ]
    return {
        "ok": not missing,
        "open_positions": positions,
        "missing_local_in_mt5": missing,
    }


def _account_gate(mt5, require_flag):
    account = mt5.account_info()
    if account is None:
        return False, "ACCOUNT_INFO_UNAVAILABLE", account
    demo = evaluate_demo_account(mt5, account)
    if not demo.allowed:
        return False, "REAL ACCOUNT BLOCKED", account
    if not str(_value(account, "server", "") or ""):
        return False, "MT5_SERVER_UNAVAILABLE", account
    if config.TRADING_ENABLED is True:
        return False, "TRADING_ENABLED_MUST_REMAIN_FALSE", account
    if require_flag and config.DEMO_EXECUTION_ENABLED is not True:
        return False, DEMO_DISABLED_REASON, account
    return True, "OK", account


def _age_minutes(signal, now):
    timestamp = _parse_time(signal.bar_timestamp)
    if timestamp is None:
        return None
    closed_at = timestamp + timedelta(hours=1)
    return max(0.0, (now - closed_at).total_seconds() / 60.0)


def _signal_is_fresh(signal, now):
    age = _age_minutes(signal, now)
    if age is None:
        return False, age
    last_closed = pd.Timestamp(now).floor("h") - pd.Timedelta(hours=1)
    same_bar = pd.to_datetime(signal.bar_timestamp, utc=True) == pd.to_datetime(last_closed, utc=True)
    return same_bar and age <= float(config.DEMO_MAX_SIGNAL_AGE_MINUTES), age


def _decision_wait(signal, config_ids, reason, now, symbol=""):
    asset = signal.asset or "N/A"
    return DemoDecision(
        execution_id=execution_id(asset, symbol or asset, signal.bar_timestamp, signal.direction),
        config_ids=tuple(config_ids),
        asset=asset,
        symbol=symbol,
        direction=signal.direction,
        bar_timestamp=signal.bar_timestamp,
        signal_age_minutes=_age_minutes(signal, now),
        status="WAIT" if signal.signal != "LONG" else "SKIP",
        reason=reason,
    )


def build_decisions(mt5, root, send=False, now=None, notify=None):
    now = now or _now_utc()
    decisions = []
    gate_ok, gate_reason, account = _account_gate(mt5, require_flag=send)
    if not gate_ok:
        for signal, config_ids in dedupe_signals(load_forward_signals(root)):
            decisions.append(_decision_wait(signal, config_ids, gate_reason, now))
        return decisions

    reconciliation = reconcile(mt5, root)
    if not reconciliation["ok"]:
        for signal, config_ids in dedupe_signals(load_forward_signals(root)):
            decisions.append(_decision_wait(signal, config_ids, "ERROR / RECONCILIATION_REQUIRED", now))
        return decisions

    resolved = discover_symbols(mt5, ["EURUSD", "GOLD"])
    open_by_symbol = _positions_by_symbol(mt5)
    executed = _executed_ids(root)
    daily_loss = _daily_loss(mt5, now=now)
    manager = RiskManager(RiskLimits(maximum_simultaneous_positions=MAX_DEMO_POSITIONS))
    balance = float(_value(account, "balance", 0) or 0)

    for signal, config_ids in dedupe_signals(load_forward_signals(root)):
        if signal.asset not in ("EURUSD", "GOLD"):
            decisions.append(_decision_wait(signal, config_ids, "UNAUTHORIZED_ASSET", now))
            continue
        if signal.asset == "EURUSD" and not validate_eurusd_hash(root):
            decisions.append(_decision_wait(signal, config_ids, "EURUSD_HASH_MISMATCH_ABORT", now))
            continue

        resolved_symbol = resolved.get(signal.asset)
        symbol = resolved_symbol.mt5_symbol if resolved_symbol and resolved_symbol.found else ""
        if not symbol:
            decisions.append(_decision_wait(signal, config_ids, "SYMBOL_NOT_FOUND", now))
            continue
        exec_id = execution_id(signal.asset, symbol, signal.bar_timestamp, signal.direction)
        if signal.signal != "LONG" or signal.execution != "OPEN":
            decisions.append(_decision_wait(signal, config_ids, "WAIT", now, symbol=symbol))
            continue
        fresh, age = _signal_is_fresh(signal, now)
        if not fresh:
            decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                          signal.bar_timestamp, age, "SKIP", "SKIP_STALE_SIGNAL"))
            continue
        if exec_id in executed:
            decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                          signal.bar_timestamp, age, "SKIP", "DUPLICATE_EXECUTION_ID"))
            continue
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        bid = float(_value(tick, "bid", 0) or 0)
        ask = float(_value(tick, "ask", 0) or 0)
        if info is None or bid <= 0 or ask <= 0:
            decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                          signal.bar_timestamp, age, "SKIP", "NO_LIVE_DATA"))
            continue
        price = _broker_entry_price(signal.direction, bid, ask)
        digits = int(_value(info, "digits", 5) or 5)
        sl, tp, sl_distance, tp_distance = _broker_levels(signal, price, digits)
        broker_metadata = _broker_metadata(info)
        order_type = _mt5_order_type(mt5, signal.direction)
        permission = manager.evaluate_trade_permission(
            balance=balance,
            daily_loss=daily_loss,
            open_positions=len(open_by_symbol),
            symbol=symbol,
            open_symbols=set(open_by_symbol),
            safety_gate=type("Gate", (), {"allowed": True, "reason": "OK"})(),
        )
        if not permission.allowed:
            decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                          signal.bar_timestamp, age, "SKIP", permission.reason,
                                          price=price, sl=sl, tp=tp, bid=bid, ask=ask,
                                          signal_source=_signal_source(signal.asset),
                                          sl_distance=sl_distance, tp_distance=tp_distance,
                                          **broker_metadata))
            continue
        sizing = manager.calculate_position_size(
            symbol=symbol,
            direction=signal.direction,
            entry=price,
            stop_loss=sl,
            balance=balance,
            symbol_info=info,
            order_calc_profit=getattr(mt5, "order_calc_profit", None),
            order_type=order_type,
        )
        if not sizing.ok or sizing.volume is None:
            decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                          signal.bar_timestamp, age, "SKIP", "SKIP_RISK",
                                          price=price, sl=sl, tp=tp, theoretical_volume=sizing.theoretical_volume,
                                          risk_usd=sizing.risk_amount, estimated_loss=sizing.min_volume_loss,
                                          bid=bid, ask=ask, signal_source=_signal_source(signal.asset),
                                          sl_distance=sl_distance, tp_distance=tp_distance,
                                          **broker_metadata))
            _notify_skip_risk(root, notify, signal, sizing)
            continue
        if sizing.volume > float(config.DEMO_MAX_VOLUME):
            max_loss = mt5.order_calc_profit(order_type, symbol, float(config.DEMO_MAX_VOLUME), price, sl)
            max_loss = abs(float(max_loss)) if max_loss is not None else None
            if max_loss is None or max_loss > sizing.risk_amount:
                decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                              signal.bar_timestamp, age, "SKIP", "SKIP_RISK",
                                              price=price, sl=sl, tp=tp, theoretical_volume=sizing.theoretical_volume,
                                              risk_usd=sizing.risk_amount, estimated_loss=max_loss,
                                              bid=bid, ask=ask, signal_source=_signal_source(signal.asset),
                                              sl_distance=sl_distance, tp_distance=tp_distance,
                                              **broker_metadata))
                continue
            volume = float(config.DEMO_MAX_VOLUME)
            estimated_loss = max_loss
        else:
            volume = sizing.volume
            estimated_loss = sizing.estimated_loss
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": MAGIC_BY_ASSET[signal.asset],
            "comment": COMMENT_BY_ASSET[signal.asset],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling_mode(mt5, info),
        }
        check = mt5.order_check(request)
        check_retcode = _value(check, "retcode")
        check_comment = str(_value(check, "comment", "") or "")
        if check_retcode not in (0, RETCODE_DONE, RETCODE_PLACED):
            decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                          signal.bar_timestamp, age, "SKIP", "ORDER_CHECK_FAILED",
                                          price=price, sl=sl, tp=tp, volume=volume,
                                          theoretical_volume=sizing.theoretical_volume,
                                          risk_usd=sizing.risk_amount, estimated_loss=estimated_loss,
                                          order_check_retcode=check_retcode,
                                          order_check_comment=check_comment,
                                          bid=bid, ask=ask, signal_source=_signal_source(signal.asset),
                                          sl_distance=sl_distance, tp_distance=tp_distance,
                                          **broker_metadata))
            continue
        if not send:
            decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                          signal.bar_timestamp, age, "PREVIEW", "READY",
                                          price=price, sl=sl, tp=tp, volume=volume,
                                          theoretical_volume=sizing.theoretical_volume,
                                          risk_usd=sizing.risk_amount, estimated_loss=estimated_loss,
                                          order_check_retcode=check_retcode,
                                          order_check_comment=check_comment,
                                          bid=bid, ask=ask, signal_source=_signal_source(signal.asset),
                                          sl_distance=sl_distance, tp_distance=tp_distance,
                                          **broker_metadata))
            continue
        sent = mt5.order_send(request)
        retcode = _value(sent, "retcode")
        ticket = _value(sent, "order")
        deal = _value(sent, "deal")
        ok = retcode in (RETCODE_DONE, RETCODE_PLACED)
        confirmed_position = _confirmed_position(mt5, ticket, signal.asset) if ok else None
        if ok and confirmed_position is None:
            ok = False
            reason = "ORDER_SENT_BUT_POSITION_NOT_CONFIRMED"
        else:
            reason = "OK" if ok else f"ORDER_SEND_FAILED: {_value(sent, 'comment', mt5.last_error())}"
        status = "SENT" if ok else "ERROR"
        _append_log(root, {
            "status": status,
            "reason": reason,
            "execution_id": exec_id,
            "config_ids": "|".join(config_ids),
            "asset": signal.asset,
            "symbol": symbol,
            "direction": signal.direction,
            "bar_timestamp": signal.bar_timestamp,
            "volume": volume,
            "price": _value(sent, "price", price),
            "sl": sl,
            "tp": tp,
            "risk_usd": estimated_loss,
            "ticket": ticket,
            "deal": deal,
            "retcode": retcode,
            "comment": COMMENT_BY_ASSET[signal.asset],
            "event": "OPEN",
        }, now=now)
        if ok:
            confirmed_ticket = _value(confirmed_position, "ticket", ticket)
            _notify_open(root, notify, signal, symbol, volume, price, sl, tp, estimated_loss, confirmed_ticket, deal)
        decisions.append(DemoDecision(exec_id, tuple(config_ids), signal.asset, symbol, signal.direction,
                                      signal.bar_timestamp, age, status, reason, price=price, sl=sl, tp=tp,
                                      volume=volume, theoretical_volume=sizing.theoretical_volume,
                                      risk_usd=sizing.risk_amount, estimated_loss=estimated_loss,
                                      order_check_retcode=check_retcode,
                                      order_check_comment=check_comment, ticket=ticket, deal=deal,
                                      retcode=retcode, orders_sent=1 if ok else 0,
                                      bid=bid, ask=ask, signal_source=_signal_source(signal.asset),
                                      sl_distance=sl_distance, tp_distance=tp_distance,
                                      **broker_metadata))
    return decisions


def _confirmed_position(mt5, ticket, asset):
    comment = COMMENT_BY_ASSET[asset]
    magic = MAGIC_BY_ASSET[asset]
    for position in (mt5.positions_get() or []):
        if str(_value(position, "ticket")) == str(ticket):
            return position
        if _value(position, "magic") == magic and _value(position, "comment") == comment:
            return position
    return None


def close_due_positions(mt5, root, notify=None, now=None, max_hold_bars=24):
    now = now or _now_utc()
    gate_ok, gate_reason, account = _account_gate(mt5, require_flag=True)
    if not gate_ok:
        return []
    results = []
    for position in _demo_positions(mt5):
        symbol = _value(position, "symbol")
        tick = mt5.symbol_info_tick(symbol)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, max_hold_bars + 100)
        opened = int(_value(position, "time", 0) or 0)
        held = 0 if rates is None else sum(1 for rate in rates if int(rate["time"]) > opened)
        if held < max_hold_bars:
            continue
        is_buy = _value(position, "type") == mt5.POSITION_TYPE_BUY
        price = float(_value(tick, "bid" if is_buy else "ask", 0) or 0)
        asset = "EURUSD" if _value(position, "magic") == MAGIC_BY_ASSET["EURUSD"] else "GOLD"
        info = mt5.symbol_info(symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": _value(position, "volume"),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": _value(position, "ticket"),
            "price": price,
            "deviation": 20,
            "magic": MAGIC_BY_ASSET[asset],
            "comment": COMMENT_BY_ASSET[asset],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling_mode(mt5, info),
        }
        check = mt5.order_check(request)
        check_retcode = _value(check, "retcode")
        if check_retcode not in (0, RETCODE_DONE, RETCODE_PLACED):
            continue
        sent = mt5.order_send(request)
        retcode = _value(sent, "retcode")
        ok = retcode in (RETCODE_DONE, RETCODE_PLACED)
        profit = float(_value(sent, "profit", _value(position, "profit", 0)) or 0)
        status = "CLOSED" if ok else "ERROR"
        reason = "MAX_HOLD" if ok else f"CLOSE_FAILED: {_value(sent, 'comment', mt5.last_error())}"
        row = {
            "status": status,
            "reason": reason,
            "execution_id": f"CLOSE|{_value(position, 'ticket')}",
            "asset": asset,
            "symbol": symbol,
            "direction": "SELL" if is_buy else "BUY",
            "volume": _value(position, "volume"),
            "price": price,
            "ticket": _value(position, "ticket"),
            "retcode": retcode,
            "comment": COMMENT_BY_ASSET[asset],
            "event": "CLOSE",
            "profit": profit,
            "commission": _value(sent, "commission", 0),
            "swap": _value(sent, "swap", 0),
        }
        _append_log(root, row, now=now)
        if ok:
            _notify_close(root, notify, asset, symbol, position, price, profit, held, account)
        results.append(row)
    return results


def _notify(root, notify, event_id, message, asset, config_id):
    try:
        if notify is not None:
            notify(event_id, message)
            return
        from src.telegram_events import send_telegram_event

        send_telegram_event(root, event_id, "MT5_DEMO", message, asset=asset, config_id=config_id)
    except Exception:
        return


def _notify_open(root, notify, signal, symbol, volume, price, sl, tp, risk, ticket, deal):
    message = "\n".join([
        "MT5 DEMO - OPERACION ABIERTA",
        f"Activo: {signal.asset}",
        f"Direccion: {signal.direction}",
        f"Entrada: {price}",
        f"SL: {sl}",
        f"TP: {tp}",
        f"Volumen: {volume}",
        f"Riesgo USD: {risk}",
        f"Ticket: {ticket}",
        f"Deal: {deal}",
        f"Estrategia: {signal.config_id}",
        "DEMO / SIN DINERO REAL",
    ])
    _notify(root, notify, f"OPEN|{signal.config_id}|{signal.bar_timestamp}", message, signal.asset, signal.config_id)


def _notify_skip_risk(root, notify, signal, sizing):
    if signal.signal != "LONG":
        return
    message = "\n".join([
        "MT5 DEMO - SENAL RECHAZADA POR RIESGO",
        f"Activo: {signal.asset}",
        f"Estrategia: {signal.config_id}",
        f"Riesgo USD: {sizing.risk_amount}",
        f"Lote minimo riesgo: {sizing.min_volume_loss}",
        "DEMO / SIN DINERO REAL",
    ])
    _notify(root, notify, f"SKIP_RISK|{signal.config_id}|{signal.bar_timestamp}", message, signal.asset, signal.config_id)


def _notify_close(root, notify, asset, symbol, position, price, profit, held, account):
    balance = _value(account, "balance", "N/A")
    equity = _value(account, "equity", "N/A")
    entry = _value(position, "price_open")
    result_pct = "" if not entry else f"{((price - float(entry)) / float(entry)) * 100:+.4f}%"
    message = "\n".join([
        "MT5 DEMO - OPERACION CERRADA",
        f"Activo: {asset}",
        f"Entrada: {entry}",
        f"Salida: {price}",
        f"Resultado USD: {profit:+.2f}",
        f"Resultado %: {result_pct}",
        "Motivo: MAX_HOLD",
        f"Duracion: {held} barras H1",
        f"Balance/equity actual: {balance}/{equity}",
        "DEMO / SIN DINERO REAL",
    ])
    _notify(root, notify, f"CLOSE|{asset}|{_value(position, 'ticket')}", message, asset, "MT5_DEMO_AUTO")


def status(mt5, root, now=None):
    now = now or _now_utc()
    gate_ok, gate_reason, account = _account_gate(mt5, require_flag=False)
    rows = _read_log(root)
    daily_loss = _daily_loss(mt5, now=now) if gate_ok else 0.0
    balance = float(_value(account, "balance", 0) or 0) if account is not None else 0.0
    return {
        "demo_execution_enabled": config.DEMO_EXECUTION_ENABLED is True,
        "trading_enabled": config.TRADING_ENABLED is True,
        "account": "DEMO" if gate_ok else gate_reason,
        "open_positions": _demo_positions(mt5) if gate_ok else [],
        "daily_pl": -daily_loss,
        "daily_loss_limit": balance * float(config.MAXIMUM_DAILY_LOSS),
        "last_signal": _latest_signal_summary(root),
        "last_order": next((row for row in reversed(rows) if row.get("event") == "OPEN"), None),
        "last_close": next((row for row in reversed(rows) if row.get("event") == "CLOSE"), None),
        "last_error": next((row for row in reversed(rows) if row.get("status") == "ERROR"), None),
        "orders_sent_today": _orders_sent_today(rows, now),
    }


def gold_chain_preview(root, mt5_factory=mt5_session, now=None):
    now = now or _now_utc()
    with mt5_factory() as mt5:
        gate_ok, gate_reason, account = _account_gate(mt5, require_flag=False)
        gold_items = [
            item for item in dedupe_signals(load_forward_signals(root))
            if item[0].asset == "GOLD"
        ]
        if not gold_items:
            return {"ok": False, "reason": "NO_GOLD_SIGNAL", "orders_sent": 0}

        signal, config_ids = max(gold_items, key=lambda item: pd.to_datetime(item[0].bar_timestamp))
        resolved = discover_symbols(mt5, ["GOLD"]).get("GOLD")
        symbol = resolved.mt5_symbol if resolved and resolved.found else ""
        if not gate_ok:
            return {
                "ok": False,
                "reason": gate_reason,
                "signal": signal,
                "config_ids": config_ids,
                "symbol": symbol,
                "orders_sent": 0,
            }
        if not symbol:
            return {
                "ok": False,
                "reason": "SYMBOL_NOT_FOUND",
                "signal": signal,
                "config_ids": config_ids,
                "orders_sent": 0,
            }

        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        bid = float(_value(tick, "bid", 0) or 0)
        ask = float(_value(tick, "ask", 0) or 0)
        direction = signal.direction if signal.direction in ("BUY", "SELL") else "BUY"
        price = _broker_entry_price(direction, bid, ask)
        digits = int(_value(info, "digits", 2) or 2)
        sl, tp, sl_distance, tp_distance = _broker_levels(signal, price, digits)
        manager = RiskManager(RiskLimits(maximum_simultaneous_positions=MAX_DEMO_POSITIONS))
        balance = float(_value(account, "balance", 0) or 0)
        order_type = _mt5_order_type(mt5, direction)
        sizing = manager.calculate_position_size(
            symbol=symbol,
            direction=direction,
            entry=price,
            stop_loss=sl,
            balance=balance,
            symbol_info=info,
            order_calc_profit=getattr(mt5, "order_calc_profit", None),
            order_type=order_type,
        )
        volume = sizing.volume
        estimated_loss = sizing.estimated_loss
        risk_preview = estimated_loss if estimated_loss is not None else sizing.min_volume_loss
        decision = "READY" if sizing.ok else "SKIP_RISK"
        if sizing.ok and volume is not None and volume > float(config.DEMO_MAX_VOLUME):
            cap_loss = mt5.order_calc_profit(order_type, symbol, float(config.DEMO_MAX_VOLUME), price, sl)
            cap_loss = abs(float(cap_loss)) if cap_loss is not None else None
            if cap_loss is not None and cap_loss <= sizing.risk_amount:
                volume = float(config.DEMO_MAX_VOLUME)
                estimated_loss = cap_loss
                risk_preview = cap_loss
            else:
                volume = None
                estimated_loss = cap_loss
                risk_preview = cap_loss
                decision = "SKIP_RISK"
        fresh, age = _signal_is_fresh(signal, now)
        if signal.signal != "LONG" or signal.execution != "OPEN":
            decision = "WAIT"
        elif not fresh:
            decision = "SKIP_STALE_SIGNAL"
        return {
            "ok": True,
            "reason": "OK",
            "signal_source": "GC=F",
            "execution_symbol": symbol,
            "signal": signal,
            "config_ids": config_ids,
            "signal_age_minutes": age,
            "bid": bid,
            "ask": ask,
            "atr": signal.atr,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "sl": sl,
            "tp": tp,
            "theoretical_volume": sizing.theoretical_volume,
            "final_volume": volume,
            "risk_usd": risk_preview,
            "decision": decision,
            "broker": _broker_metadata(info),
            "orders_sent": 0,
        }


def _latest_signal_summary(root):
    signals = load_forward_signals(root)
    if not signals:
        return None
    signal = max(signals, key=lambda item: pd.to_datetime(item.bar_timestamp))
    return {
        "asset": signal.asset,
        "config_id": signal.config_id,
        "bar_timestamp": signal.bar_timestamp,
        "signal": signal.signal,
        "execution": signal.execution,
        "reason": signal.reason,
    }


def _orders_sent_today(rows, now):
    start = now.astimezone(PERU_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    count = 0
    for row in rows:
        if row.get("status") != "SENT":
            continue
        timestamp = _parse_time(row.get("timestamp"))
        if timestamp and timestamp.astimezone(PERU_TZ) >= start:
            count += 1
    return count


def run_preview(root, mt5_factory=mt5_session, now=None):
    with mt5_factory() as mt5:
        return build_decisions(mt5, root, send=False, now=now)


def run_status(root, mt5_factory=mt5_session, now=None):
    with mt5_factory() as mt5:
        return status(mt5, root, now=now)


def run_auto(root, mt5_factory=mt5_session, now=None, notify=None):
    with mt5_factory() as mt5:
        close_due_positions(mt5, root, notify=notify, now=now)
        return build_decisions(mt5, root, send=True, now=now, notify=notify)
