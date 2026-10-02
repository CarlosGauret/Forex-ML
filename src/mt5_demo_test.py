"""Prueba controlada de ejecucion MT5 exclusivamente en cuenta demo.

Este modulo no usa modelos, thresholds ni PAPER. Su unica responsabilidad es
probar la ruta Python -> MT5 -> cuenta DEMO con una orden EURUSD pequena.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config
from src.execution_demo import DEMO_DISABLED_REASON, DEMO_REQUIRED_REASON, evaluate_demo_account
from src.live_executor import RETCODE_DONE, RETCODE_PLACED, _filling_mode, mt5_session
from src.symbols import discover_symbols


DEMO_TEST_MAGIC = 560001
DEMO_TEST_COMMENT = "FOREX_ML_DEMO_TEST"
DEMO_TEST_SYMBOL = "EURUSD"
DEMO_TEST_VOLUME = 0.01
DEMO_TEST_DEVIATION = 20
DEMO_TEST_SL_POINTS = 100
DEMO_TEST_TP_POINTS = 200
LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "volume",
    "entry",
    "sl",
    "tp",
    "ticket",
    "retcode",
    "event",
    "profit",
]


@dataclass(frozen=True)
class DemoTestResult:
    status: str
    reason: str
    account: str = "N/A"
    server: str = "N/A"
    symbol: str = DEMO_TEST_SYMBOL
    bid: float | None = None
    ask: float | None = None
    volume_min: float | None = None
    volume_step: float | None = None
    volume: float | None = None
    sl: float | None = None
    tp: float | None = None
    estimated_risk: float | None = None
    order_check_retcode: int | None = None
    order_check_comment: str = ""
    order_retcode: int | None = None
    order_ticket: int | None = None
    deal_ticket: int | None = None
    requested_price: float | None = None
    executed_price: float | None = None
    close_price: float | None = None
    profit: float | None = None
    duration_seconds: int | None = None
    orders_sent: int = 0
    trading_enabled: bool = False
    demo_execution_enabled: bool = False


def _value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _log_path(root):
    return Path(root) / "logs" / "mt5_demo_execution.csv"


def _append_log(root, event, symbol=DEMO_TEST_SYMBOL, side="", volume=None, entry=None,
                sl=None, tp=None, ticket=None, retcode=None, profit=None, now=None):
    path = _log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": (now or _now_utc()).isoformat(),
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "ticket": ticket,
            "retcode": retcode,
            "event": event,
            "profit": profit,
        })


def _account_label(mt5, account):
    gate = evaluate_demo_account(mt5, account)
    if gate.allowed:
        return "DEMO"
    if _value(account, "trade_mode") == getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", object()):
        return "REAL"
    return "UNKNOWN"


def _base_result(status, reason, mt5=None, account=None, **fields):
    return DemoTestResult(
        status=status,
        reason=reason,
        account=_account_label(mt5, account) if mt5 is not None else "N/A",
        server=str(_value(account, "server", "N/A") or "N/A"),
        trading_enabled=config.TRADING_ENABLED is True,
        demo_execution_enabled=config.DEMO_EXECUTION_ENABLED is True,
        **fields,
    )


def _order_gate(mt5, allow_when_flag_disabled=False):
    account = mt5.account_info()
    if account is None:
        return _base_result("ABORT", "ACCOUNT_INFO_UNAVAILABLE", mt5, account)

    gate = evaluate_demo_account(mt5, account)
    if not gate.allowed:
        return _base_result("ABORT", "REAL ACCOUNT BLOCKED", mt5, account)

    if config.TRADING_ENABLED is True:
        return _base_result("ABORT", "TRADING_ENABLED_MUST_REMAIN_FALSE", mt5, account)

    if not allow_when_flag_disabled and config.DEMO_EXECUTION_ENABLED is not True:
        return _base_result("ABORT", DEMO_DISABLED_REASON, mt5, account)

    return None


def _test_positions(mt5):
    return [
        position for position in (mt5.positions_get() or [])
        if _value(position, "magic") == DEMO_TEST_MAGIC
        and _value(position, "comment") == DEMO_TEST_COMMENT
    ]


def _build_request(mt5):
    resolved = discover_symbols(mt5, [DEMO_TEST_SYMBOL])[DEMO_TEST_SYMBOL]
    if not resolved.found:
        return None, _base_result("ABORT", resolved.reason, mt5, mt5.account_info())

    symbol = resolved.mt5_symbol
    mt5.symbol_select(symbol, True)
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    bid = float(_value(tick, "bid", 0) or 0)
    ask = float(_value(tick, "ask", 0) or 0)
    if info is None or bid <= 0 or ask <= 0:
        return None, _base_result("ABORT", "NO_PRICE", mt5, mt5.account_info(), symbol=symbol)

    volume_min = float(_value(info, "volume_min", 0) or 0)
    volume_step = float(_value(info, "volume_step", 0) or 0)
    if volume_min > DEMO_TEST_VOLUME:
        return None, _base_result(
            "ABORT",
            "BROKER_MINIMUM_VOLUME_ABOVE_TEST_VOLUME",
            mt5,
            mt5.account_info(),
            symbol=symbol,
            bid=bid,
            ask=ask,
            volume_min=volume_min,
            volume_step=volume_step,
            volume=DEMO_TEST_VOLUME,
        )

    digits = int(_value(info, "digits", 5) or 5)
    point = float(_value(info, "point", 0.00001) or 0.00001)
    min_stop_points = int(_value(info, "trade_stops_level", 0) or 0)
    stop_points = max(DEMO_TEST_SL_POINTS, min_stop_points + 10)
    tp_points = max(DEMO_TEST_TP_POINTS, stop_points * 2)
    price = ask
    sl = round(price - stop_points * point, digits)
    tp = round(price + tp_points * point, digits)
    risk = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, DEMO_TEST_VOLUME, price, sl)
    estimated_risk = abs(float(risk)) if risk is not None else None
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": DEMO_TEST_VOLUME,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEMO_TEST_DEVIATION,
        "magic": DEMO_TEST_MAGIC,
        "comment": DEMO_TEST_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(mt5, info),
    }
    context = {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "volume_min": volume_min,
        "volume_step": volume_step,
        "volume": DEMO_TEST_VOLUME,
        "sl": sl,
        "tp": tp,
        "estimated_risk": estimated_risk,
        "requested_price": price,
    }
    return (request, context), None


def _check_order(mt5, request):
    checker = getattr(mt5, "order_check", None)
    if checker is None:
        return None
    return checker(request)


def preview_demo_test(mt5, root):
    gate = _order_gate(mt5, allow_when_flag_disabled=True)
    if gate is not None:
        return gate

    built, error = _build_request(mt5)
    if error is not None:
        return error
    request, context = built
    check = _check_order(mt5, request)
    return _base_result(
        "PREVIEW",
        "OK",
        mt5,
        mt5.account_info(),
        order_check_retcode=_value(check, "retcode"),
        order_check_comment=str(_value(check, "comment", "") or ""),
        **context,
    )


def open_demo_test(mt5, root, notify=None, now=None):
    gate = _order_gate(mt5)
    if gate is not None:
        return gate

    existing = _test_positions(mt5)
    if existing:
        return _base_result("ABORT", "DEMO_TEST_POSITION_ALREADY_OPEN", mt5, mt5.account_info())

    built, error = _build_request(mt5)
    if error is not None:
        return error
    request, context = built
    check = _check_order(mt5, request)
    check_retcode = _value(check, "retcode")
    check_comment = str(_value(check, "comment", "") or "")
    ok_check = check is None or check_retcode in (RETCODE_DONE, RETCODE_PLACED, 0)
    if not ok_check:
        _append_log(
            root,
            "ORDER_CHECK_REJECTED",
            symbol=context["symbol"],
            side="BUY",
            volume=context["volume"],
            entry=context["requested_price"],
            sl=context["sl"],
            tp=context["tp"],
            retcode=check_retcode,
            now=now,
        )
        return _base_result(
            "ABORT",
            "ORDER_CHECK_FAILED",
            mt5,
            mt5.account_info(),
            order_check_retcode=check_retcode,
            order_check_comment=check_comment,
            **context,
        )

    sent = mt5.order_send(request)
    retcode = _value(sent, "retcode")
    ok = retcode in (RETCODE_DONE, RETCODE_PLACED)
    order_ticket = _value(sent, "order")
    deal_ticket = _value(sent, "deal")
    executed_price = float(_value(sent, "price", context["requested_price"]) or context["requested_price"])
    status = "OPENED" if ok else "ERROR"
    reason = "OK" if ok else f"ORDER_SEND_FAILED: {_value(sent, 'comment', mt5.last_error())}"
    _append_log(
        root,
        "OPEN" if ok else "OPEN_ERROR",
        symbol=context["symbol"],
        side="BUY",
        volume=context["volume"],
        ticket=order_ticket,
        retcode=retcode,
        entry=executed_price,
        sl=context["sl"],
        tp=context["tp"],
        now=now,
    )
    if ok and _confirmed_test_position(mt5, order_ticket) is not None:
        _notify_open(root, notify, context["symbol"], DEMO_TEST_VOLUME, executed_price)
    return _base_result(
        status,
        reason,
        mt5,
        mt5.account_info(),
        order_check_retcode=check_retcode,
        order_check_comment=check_comment,
        order_retcode=retcode,
        order_ticket=order_ticket,
        deal_ticket=deal_ticket,
        executed_price=executed_price,
        orders_sent=1 if ok else 0,
        **context,
    )


def close_demo_test(mt5, root, notify=None, now=None):
    gate = _order_gate(mt5)
    if gate is not None:
        return gate

    positions = _test_positions(mt5)
    if not positions:
        return _base_result("SKIP", "NO_DEMO_TEST_POSITION", mt5, mt5.account_info())
    position = sorted(positions, key=lambda item: int(_value(item, "time", 0) or 0))[0]
    symbol = _value(position, "symbol")
    tick = mt5.symbol_info_tick(symbol)
    is_buy = _value(position, "type") == mt5.POSITION_TYPE_BUY
    close_price = float(_value(tick, "bid" if is_buy else "ask", 0) or 0)
    info = mt5.symbol_info(symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": _value(position, "volume"),
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "position": _value(position, "ticket"),
        "price": close_price,
        "deviation": DEMO_TEST_DEVIATION,
        "magic": DEMO_TEST_MAGIC,
        "comment": DEMO_TEST_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(mt5, info),
    }
    check = _check_order(mt5, request)
    check_retcode = _value(check, "retcode")
    check_comment = str(_value(check, "comment", "") or "")
    ok_check = check is None or check_retcode in (RETCODE_DONE, RETCODE_PLACED, 0)
    if not ok_check:
        return _base_result(
            "ABORT",
            "ORDER_CHECK_FAILED",
            mt5,
            mt5.account_info(),
            order_check_retcode=check_retcode,
            order_check_comment=check_comment,
            symbol=symbol,
            volume=_value(position, "volume"),
            requested_price=close_price,
        )

    sent = mt5.order_send(request)
    retcode = _value(sent, "retcode")
    ok = retcode in (RETCODE_DONE, RETCODE_PLACED)
    profit = float(_value(sent, "profit", _value(position, "profit", 0.0)) or 0.0)
    opened_at = int(_value(position, "time", 0) or 0)
    server_time = int(_value(tick, "time", opened_at) or opened_at)
    duration = max(0, server_time - opened_at) if opened_at else None
    ticket = _value(position, "ticket")
    _append_log(
        root,
        "CLOSE" if ok else "CLOSE_ERROR",
        symbol=symbol,
        side="SELL" if is_buy else "BUY",
        volume=_value(position, "volume"),
        entry=_value(position, "price_open"),
        ticket=ticket,
        retcode=retcode,
        profit=profit,
        now=now,
    )
    if ok:
        _notify_close(root, notify, symbol, profit)
    return _base_result(
        "CLOSED" if ok else "ERROR",
        "OK" if ok else f"CLOSE_FAILED: {_value(sent, 'comment', mt5.last_error())}",
        mt5,
        mt5.account_info(),
        symbol=symbol,
        volume=_value(position, "volume"),
        order_check_retcode=check_retcode,
        order_check_comment=check_comment,
        order_retcode=retcode,
        order_ticket=ticket,
        requested_price=close_price,
        close_price=close_price,
        executed_price=float(_value(sent, "price", close_price) or close_price),
        profit=profit,
        duration_seconds=duration,
        orders_sent=1 if ok else 0,
    )


def _notify_open(root, notify, symbol, volume, price):
    message = "\n".join([
        "MT5 DEMO - OPERACION ABIERTA",
        "DEMO / SIN DINERO REAL",
        f"Simbolo: {symbol}",
        f"Volumen: {volume}",
        f"Precio: {price}",
    ])
    _notify(root, notify, "MT5_DEMO_TEST_OPEN", message)


def _confirmed_test_position(mt5, ticket):
    for position in (mt5.positions_get() or []):
        if str(_value(position, "ticket")) == str(ticket):
            return position
        if _value(position, "magic") == DEMO_TEST_MAGIC and _value(position, "comment") == DEMO_TEST_COMMENT:
            return position
    return None


def _notify_close(root, notify, symbol, profit):
    message = "\n".join([
        "MT5 DEMO - OPERACION CERRADA",
        "DEMO / SIN DINERO REAL",
        f"Simbolo: {symbol}",
        f"Profit/Loss: {profit:+.2f}",
    ])
    _notify(root, notify, "MT5_DEMO_TEST_CLOSE", message)


def _notify(root, notify, event_id, message):
    try:
        if notify is not None:
            notify(event_id, message)
            return
        from src.telegram_events import send_telegram_event

        send_telegram_event(root, event_id, "MT5_DEMO_TEST", message, asset=DEMO_TEST_SYMBOL, config_id="MT5_DEMO_TEST")
    except Exception:
        return


def run_mt5_demo_test(root, mode="PREVIEW", mt5_factory=mt5_session, notify=None):
    mode = (mode or "PREVIEW").upper()
    with mt5_factory() as mt5:
        if mode == "PREVIEW":
            return preview_demo_test(mt5, root)
        if mode == "CLOSE":
            return close_demo_test(mt5, root, notify=notify)
        if mode in ("TEST", "OPEN"):
            return open_demo_test(mt5, root, notify=notify)
    return DemoTestResult(status="ABORT", reason="INVALID_MODE")
