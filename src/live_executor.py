"""Ejecucion automatica de ordenes en MT5 (solo cuenta DEMO).

Reglas de seguridad:
- Cuenta REAL siempre bloqueada (evaluate_demo_account).
- Sin DEMO_EXECUTION_ENABLED solo se permite DRY RUN (calcula todo, no envia).
- SL y TP viajan con la orden: quedan en el servidor del broker.
- Una senal nunca se envia dos veces (journal + comentario de la orden en MT5).
- Archivo STOP_TRADING en la raiz bloquea aperturas nuevas (no bloquea cierres).
"""

import csv
import hashlib
import time as _time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import (
    BREAK_EVEN_OFFSET_POINTS,
    BREAK_EVEN_TRIGGER_R,
    EXIT_MODE,
    MAX_SPREAD_ATR_RATIO_BY_ASSET,
    TRAILING_DISTANCE_R,
    TRAILING_START_R,
)
from src.execution_demo import evaluate_demo_account, evaluate_demo_order_gate
from src.risk import BUY, SELL, RiskManager
from src.risk_state import (
    ProtectiveExitState,
    load_exit_states,
    load_or_roll_daily_risk_state,
    save_exit_states,
)
from src.symbols import discover_symbols


MAGIC_NUMBER = 20260930
COMMENT_PREFIX = "FXML-"
KILL_SWITCH_FILENAME = "STOP_TRADING"
MAX_SPREAD_ATR_RATIO = 0.25
DEVIATION_POINTS = 20
MAX_SEND_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 0.5
HISTORY_LOOKBACK_DAYS = 30

RETCODE_PLACED = 10008
RETCODE_DONE = 10009
RETRY_RETCODES = {10004, 10020, 10021}  # REQUOTE, PRICE_CHANGED, PRICE_OFF
RETCODE_ALGO_TRADING_DISABLED = 10027

STATUS_SENT = "SENT"
STATUS_DRY_RUN = "DRY_RUN"
STATUS_SKIP = "SKIP"
STATUS_ERROR = "ERROR"
STATUS_CLOSED_TIME = "CLOSED_TIME"
STATUS_DRY_RUN_CLOSE = "DRY_RUN_CLOSE"
STATUS_SL_UPDATED = "SL_UPDATED"
STATUS_DRY_RUN_SL_UPDATE = "DRY_RUN_SL_UPDATE"

JOURNAL_COLUMNS = [
    "TIMESTAMP_UTC",
    "STATUS",
    "REASON",
    "SIGNAL_ID",
    "CONFIG_ID",
    "ASSET",
    "SYMBOL",
    "DIRECTION",
    "VOLUME",
    "PRICE",
    "SL",
    "TP",
    "SPREAD",
    "TICKET",
    "RETCODE",
    "COMMENT",
    "DRY_RUN",
    "EXIT_MODE",
    "RISK_PERCENT",
    "RISK_USD",
    "RISK_PCT_REAL",
    "EQUITY_BEFORE_TRADE",
    "BALANCE_BEFORE_TRADE",
    "SPREAD_AT_ENTRY",
    "R_CURRENT",
    "BREAK_EVEN_ACTIVATED",
    "TRAILING_ACTIVATED",
    "HIGHEST_FAVORABLE_PRICE",
    "HIGHEST_FAVORABLE_R",
    "SIGNAL_EXECUTED",
    "SIGNAL_SKIPPED_REASON",
    "SHADOW_RESULT",
]


@dataclass(frozen=True)
class OrderIntent:
    signal_id: str
    config_id: str
    asset: str
    direction: str
    atr: float
    stop_atr: float = 1.0
    take_profit_atr: float = 2.0


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    reason: str
    signal_id: str = ""
    config_id: str = ""
    asset: str = ""
    symbol: str = ""
    direction: str = ""
    volume: float | None = None
    price: float | None = None
    sl: float | None = None
    tp: float | None = None
    spread: float | None = None
    ticket: int | None = None
    retcode: int | None = None
    comment: str = ""
    dry_run: bool = True
    exit_mode: str = EXIT_MODE
    risk_percent: float | None = None
    risk_usd: float | None = None
    risk_pct_real: float | None = None
    equity_before_trade: float | None = None
    balance_before_trade: float | None = None
    spread_at_entry: float | None = None
    r_current: float | None = None
    break_even_activated: bool = False
    trailing_activated: bool = False
    highest_favorable_price: float | None = None
    highest_favorable_r: float | None = None
    signal_executed: bool = False
    signal_skipped_reason: str = ""
    shadow_result: str = ""


def _value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def order_comment(signal_id):
    # MT5 limita el comentario a 31 caracteres: se usa un hash corto y estable.
    digest = hashlib.sha1(str(signal_id).encode("utf-8")).hexdigest()[:12]
    return f"{COMMENT_PREFIX}{digest}"


def kill_switch_path(root):
    return Path(root) / KILL_SWITCH_FILENAME


def kill_switch_active(root):
    return kill_switch_path(root).exists()


def journal_path(root):
    return Path(root) / "live" / "orders.csv"


def read_journal(root):
    path = journal_path(root)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_journal(root, result, now=None):
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    row = {key.upper(): value for key, value in asdict(result).items()}
    row["TIMESTAMP_UTC"] = (now or _now_utc()).isoformat()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOURNAL_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def bot_positions(mt5, magic=MAGIC_NUMBER):
    return [p for p in (mt5.positions_get() or []) if _value(p, "magic") == magic]


def bot_deals(mt5, since_server_time=None, magic=MAGIC_NUMBER, now=None):
    now = now or _now_utc()
    deals = mt5.history_deals_get(
        now - timedelta(days=HISTORY_LOOKBACK_DAYS),
        now + timedelta(days=1),
    ) or []
    deals = [d for d in deals if _value(d, "magic") == magic]
    if since_server_time is not None:
        deals = [d for d in deals if int(_value(d, "time", 0) or 0) >= since_server_time]
    return deals


def signal_already_executed(mt5, root, signal_id, dry_run=False, now=None):
    statuses = {STATUS_SENT, STATUS_DRY_RUN} if dry_run else {STATUS_SENT}
    for row in read_journal(root):
        if row.get("SIGNAL_ID") == signal_id and row.get("STATUS") in statuses:
            return True

    # Respaldo por si el journal no se escribio despues de un envio real.
    comment = order_comment(signal_id)
    if any(_value(p, "comment") == comment for p in bot_positions(mt5)):
        return True
    return any(_value(d, "comment") == comment for d in bot_deals(mt5, now=now))


def daily_loss(mt5, server_now, now=None):
    """Perdida del bot en las ultimas 24h (realizada + flotante), en moneda de cuenta."""
    realized = 0.0
    for deal in bot_deals(mt5, since_server_time=server_now - 86400, now=now):
        for field in ("profit", "commission", "swap", "fee"):
            realized += float(_value(deal, field, 0) or 0)
    floating = sum(float(_value(p, "profit", 0) or 0) for p in bot_positions(mt5))
    return max(0.0, -(realized + floating))


def _filling_mode(mt5, info):
    mode = int(_value(info, "filling_mode", 0) or 0)
    if mode & 1:
        return getattr(mt5, "ORDER_FILLING_FOK", 0)
    if mode & 2:
        return getattr(mt5, "ORDER_FILLING_IOC", 1)
    return getattr(mt5, "ORDER_FILLING_RETURN", 2)


def build_levels(direction, price, atr, stop_atr, take_profit_atr, digits):
    if direction == BUY:
        sl = price - stop_atr * atr
        tp = price + take_profit_atr * atr
    else:
        sl = price + stop_atr * atr
        tp = price - take_profit_atr * atr
    return round(sl, digits), round(tp, digits)


def _account_balance_equity(account):
    balance = float(_value(account, "balance", 0) or 0)
    equity = float(_value(account, "equity", balance) or balance)
    return balance, equity


def _risk_amount_for_position(mt5, position):
    symbol = _value(position, "symbol")
    sl = float(_value(position, "sl", 0) or 0)
    entry = float(_value(position, "price_open", 0) or 0)
    volume = float(_value(position, "volume", 0) or 0)
    if not symbol or sl <= 0 or entry <= 0 or volume <= 0:
        return 0.0
    is_buy = _value(position, "type") == getattr(mt5, "POSITION_TYPE_BUY", 0)
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    calc = getattr(mt5, "order_calc_profit", None)
    if calc is None:
        return 0.0
    loss = calc(order_type, symbol, volume, entry, sl)
    return abs(float(loss or 0))


def open_risk_amount(mt5, positions):
    return sum(_risk_amount_for_position(mt5, position) for position in positions)


def spread_atr_ratio_limit(asset, symbol):
    asset_key = str(asset or "").upper()
    symbol_key = str(symbol or "").upper()
    return MAX_SPREAD_ATR_RATIO_BY_ASSET.get(
        asset_key,
        MAX_SPREAD_ATR_RATIO_BY_ASSET.get(symbol_key, MAX_SPREAD_ATR_RATIO),
    )


def _base_result(intent, dry_run, **fields):
    return ExecutionResult(
        signal_id=intent.signal_id,
        config_id=intent.config_id,
        asset=intent.asset,
        direction=intent.direction,
        comment=order_comment(intent.signal_id),
        dry_run=dry_run,
        **fields,
    )


def _notify(notify, event_id, message):
    if notify is None:
        return
    try:
        notify(event_id, message)
    except Exception:
        return


def execute_signal(
    mt5,
    intent,
    root,
    dry_run=True,
    risk_manager=None,
    notify=None,
    sleep=_time.sleep,
    now=None,
):
    """Valida una senal contra todas las puertas y, si pasa, envia la orden."""
    now = now or _now_utc()

    def finish(result, journal=True):
        if journal:
            append_journal(root, result, now)
        if result.status in (STATUS_SENT, STATUS_ERROR):
            _notify(
                notify,
                f"LIVE_{result.status}|{result.signal_id}",
                format_execution_message(result),
            )
        return result

    def skip(reason, journal=True, **fields):
        return finish(
            _base_result(
                intent,
                dry_run,
                status=STATUS_SKIP,
                reason=reason,
                signal_skipped_reason=reason,
                **fields,
            ),
            journal,
        )

    if intent.direction not in (BUY, SELL):
        return skip("INVALID_DIRECTION")
    if kill_switch_active(root):
        return skip("KILL_SWITCH_ACTIVE")

    account = mt5.account_info()
    gate = evaluate_demo_account(mt5, account) if dry_run else evaluate_demo_order_gate(mt5, account)
    if not gate.allowed:
        return skip(gate.reason)

    resolved = discover_symbols(mt5, [intent.asset])[intent.asset.upper()]
    if not resolved.found:
        return skip(resolved.reason)
    symbol = resolved.mt5_symbol
    mt5.symbol_select(symbol, True)

    if signal_already_executed(mt5, root, intent.signal_id, dry_run=dry_run, now=now):
        return skip("DUPLICATE_SIGNAL", journal=False, symbol=symbol)

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    bid = float(_value(tick, "bid", 0) or 0)
    ask = float(_value(tick, "ask", 0) or 0)
    if info is None or bid <= 0 or ask <= 0:
        return skip("NO_PRICE", symbol=symbol)

    manager = risk_manager or RiskManager()
    balance, equity = _account_balance_equity(account)
    positions = bot_positions(mt5)
    daily_state = load_or_roll_daily_risk_state(root, equity, now)
    expected_trade_risk = equity * manager.limits.risk_per_trade
    permission = manager.evaluate_trade_permission(
        balance=balance,
        equity=equity,
        starting_day_equity=daily_state.starting_equity,
        daily_loss=daily_loss(mt5, int(_value(tick, "time", 0) or 0), now=now),
        open_positions=len(positions),
        open_risk_amount=open_risk_amount(mt5, positions),
        new_trade_risk_amount=expected_trade_risk,
        symbol=symbol,
        open_symbols={_value(p, "symbol") for p in positions},
        safety_gate=gate,
    )
    if not permission.allowed:
        return skip(permission.reason, symbol=symbol)

    spread = ask - bid
    atr = float(intent.atr)
    if atr <= 0:
        return skip("INVALID_ATR", symbol=symbol, spread=spread)
    if spread > spread_atr_ratio_limit(intent.asset, symbol) * atr:
        return skip("SPREAD_TOO_WIDE", symbol=symbol, spread=spread)

    digits = int(_value(info, "digits", 5) or 5)
    point = float(_value(info, "point", 0) or 0)
    price = ask if intent.direction == BUY else bid
    sl, tp = build_levels(intent.direction, price, atr, intent.stop_atr, intent.take_profit_atr, digits)

    min_distance = float(_value(info, "trade_stops_level", 0) or 0) * point
    if abs(price - sl) < min_distance or abs(tp - price) < min_distance:
        return skip("STOPS_TOO_CLOSE", symbol=symbol, price=price, sl=sl, tp=tp, spread=spread)

    order_type = mt5.ORDER_TYPE_BUY if intent.direction == BUY else mt5.ORDER_TYPE_SELL
    sizing = manager.calculate_position_size(
        symbol=symbol,
        direction=intent.direction,
        entry=price,
        stop_loss=sl,
        balance=balance,
        equity=equity,
        symbol_info=info,
        order_calc_profit=getattr(mt5, "order_calc_profit", None),
        order_type=order_type,
    )
    if not sizing.ok:
        return skip(sizing.reason, symbol=symbol, price=price, sl=sl, tp=tp, spread=spread)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": sizing.volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC_NUMBER,
        "comment": order_comment(intent.signal_id),
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(mt5, info),
    }
    fields = {
        "symbol": symbol,
        "volume": sizing.volume,
        "spread": spread,
        "spread_at_entry": spread,
        "risk_percent": manager.limits.risk_per_trade * 100,
        "risk_usd": sizing.risk_amount,
        "risk_pct_real": sizing.risk_pct_real,
        "equity_before_trade": equity,
        "balance_before_trade": balance,
    }

    if dry_run:
        return finish(_base_result(intent, dry_run, status=STATUS_DRY_RUN, reason="NOT_SENT_DRY_RUN",
                                   price=price, sl=sl, tp=tp, signal_executed=False, **fields))

    result = None
    for attempt in range(MAX_SEND_ATTEMPTS):
        if attempt > 0:
            sleep(RETRY_SLEEP_SECONDS)
            tick = mt5.symbol_info_tick(symbol)
            price = float(_value(tick, "ask" if intent.direction == BUY else "bid", price))
            sl, tp = build_levels(intent.direction, price, atr, intent.stop_atr, intent.take_profit_atr, digits)
            request.update({"price": price, "sl": sl, "tp": tp})
        result = mt5.order_send(request)
        retcode = _value(result, "retcode")
        if retcode in (RETCODE_DONE, RETCODE_PLACED):
            return finish(_base_result(
                intent, dry_run, status=STATUS_SENT, reason="OK",
                price=float(_value(result, "price", 0) or price), sl=sl, tp=tp,
                ticket=_value(result, "order"), retcode=retcode, signal_executed=True, **fields,
            ))
        if retcode not in RETRY_RETCODES:
            break

    reason = _value(result, "comment") if result is not None else str(mt5.last_error())
    if _value(result, "retcode") == RETCODE_ALGO_TRADING_DISABLED:
        reason = "ALGO TRADING DESACTIVADO EN MT5: activa el boton 'Algo Trading' de la terminal"
    return finish(_base_result(
        intent, dry_run, status=STATUS_ERROR, reason=f"ORDER_SEND_FAILED: {reason}",
        price=price, sl=sl, tp=tp, retcode=_value(result, "retcode"), **fields,
    ))


def bars_held(mt5, position, server_now, timeframe="TIMEFRAME_H1", bar_seconds=3600, max_bars=24):
    """Velas cerradas desde la entrada, igual que BARS_HELD del PAPER.

    Cuenta velas reales de MT5 (no horas): el fin de semana no suma velas.
    Devuelve None si MT5 no entrega rates.
    """
    rates = mt5.copy_rates_from_pos(
        _value(position, "symbol"), getattr(mt5, timeframe), 0, max_bars + 100
    )
    if rates is None or len(rates) == 0:
        return None
    opened = int(_value(position, "time", 0) or 0)
    return sum(
        1 for rate in rates
        if int(rate["time"]) > opened and int(rate["time"]) + bar_seconds <= server_now
    )


STATUS_CLOSED_MANUAL = "CLOSED_MANUAL"


def _closed_event_id(position_ticket):
    # Mismo EVENT_ID que notify_closed_deals: Telegram avisa una sola vez por posicion.
    return f"LIVE_CLOSED|{position_ticket}"


def close_position(mt5, root, position, reason, closed_status, dry_run=True, notify=None, now=None):
    symbol = _value(position, "symbol")
    tick = mt5.symbol_info_tick(symbol)
    is_buy = _value(position, "type") == mt5.POSITION_TYPE_BUY
    price = float(_value(tick, "bid" if is_buy else "ask", 0) or 0)
    ticket = _value(position, "ticket")
    base = {
        "signal_id": f"{reason}|{ticket}",
        "symbol": symbol,
        "direction": SELL if is_buy else BUY,
        "volume": _value(position, "volume"),
        "price": price,
        "ticket": ticket,
        "comment": _value(position, "comment", ""),
        "dry_run": dry_run,
    }
    if dry_run:
        result = ExecutionResult(status=STATUS_DRY_RUN_CLOSE, reason=reason, **base)
    else:
        info = mt5.symbol_info(symbol)
        sent = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": _value(position, "volume"),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": DEVIATION_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": _value(position, "comment", ""),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling_mode(mt5, info),
        })
        retcode = _value(sent, "retcode")
        ok = retcode in (RETCODE_DONE, RETCODE_PLACED)
        result = ExecutionResult(
            status=closed_status if ok else STATUS_ERROR,
            reason=reason if ok else f"CLOSE_FAILED: {_value(sent, 'comment', mt5.last_error())}",
            retcode=retcode,
            **base,
        )
        event_id = _closed_event_id(ticket) if ok else f"LIVE_{STATUS_ERROR}|CLOSE|{ticket}"
        _notify(notify, event_id, format_execution_message(result))
    append_journal(root, result, now)
    return result


def _position_side(mt5, position):
    is_buy = _value(position, "type") == getattr(mt5, "POSITION_TYPE_BUY", 0)
    return BUY if is_buy else SELL


def _current_exit_price(mt5, position):
    symbol = _value(position, "symbol")
    tick = mt5.symbol_info_tick(symbol)
    return float(_value(tick, "bid" if _position_side(mt5, position) == BUY else "ask", 0) or 0)


def _favorable_r(direction, entry, current_price, initial_risk):
    if initial_risk <= 0:
        return 0.0
    if direction == BUY:
        return (current_price - entry) / initial_risk
    return (entry - current_price) / initial_risk


def _better_stop(direction, candidate, current_sl):
    if current_sl is None or current_sl <= 0:
        return True
    return candidate > current_sl if direction == BUY else candidate < current_sl


def manage_protective_exits(
    mt5,
    root,
    exit_mode=EXIT_MODE,
    dry_run=True,
    notify=None,
    now=None,
):
    """Move SL to break-even/trailing for bot positions when an experiment asks for it.

    FIXED_TP intentionally returns no actions, preserving current behavior.
    """
    if exit_mode == "FIXED_TP":
        return []
    if not _close_gate_allowed(mt5, dry_run):
        return []

    states = load_exit_states(root)
    results = []
    active_tickets = set()
    for position in bot_positions(mt5):
        ticket = str(_value(position, "ticket"))
        active_tickets.add(ticket)
        symbol = _value(position, "symbol")
        info = mt5.symbol_info(symbol)
        direction = _position_side(mt5, position)
        entry = float(_value(position, "price_open", 0) or 0)
        current_sl = float(_value(position, "sl", 0) or 0)
        tp = float(_value(position, "tp", 0) or 0)
        if entry <= 0 or current_sl <= 0:
            continue

        state = states.get(ticket) or ProtectiveExitState(ticket=ticket, symbol=symbol)
        state.symbol = symbol
        state.entry_price = state.entry_price or entry
        state.initial_sl = state.initial_sl or current_sl
        state.initial_risk = state.initial_risk or abs(entry - state.initial_sl)
        state.current_sl = current_sl
        if not state.initial_risk or state.initial_risk <= 0:
            states[ticket] = state
            continue

        current_price = _current_exit_price(mt5, position)
        if current_price <= 0:
            states[ticket] = state
            continue
        current_r = _favorable_r(direction, entry, current_price, state.initial_risk)
        state.highest_favorable_r = max(state.highest_favorable_r, current_r)
        if (
            state.highest_favorable_price is None
            or (direction == BUY and current_price > state.highest_favorable_price)
            or (direction == SELL and current_price < state.highest_favorable_price)
        ):
            state.highest_favorable_price = current_price

        digits = int(_value(info, "digits", 5) or 5)
        point = float(_value(info, "point", 0) or 0)
        offset = BREAK_EVEN_OFFSET_POINTS * point
        candidate_sl = None
        reason = ""
        if current_r >= BREAK_EVEN_TRIGGER_R and not state.break_even_activated:
            candidate_sl = entry + offset if direction == BUY else entry - offset
            reason = "BREAK_EVEN"
        if exit_mode == "TRAILING" and current_r >= TRAILING_START_R:
            trail_distance = state.initial_risk * TRAILING_DISTANCE_R
            trailing_sl = current_price - trail_distance if direction == BUY else current_price + trail_distance
            if candidate_sl is None or _better_stop(direction, trailing_sl, candidate_sl):
                candidate_sl = trailing_sl
                reason = "TRAILING_STOP"

        if candidate_sl is None:
            states[ticket] = state
            continue
        candidate_sl = round(candidate_sl, digits)
        if not _better_stop(direction, candidate_sl, current_sl):
            states[ticket] = state
            continue

        state.current_sl = candidate_sl
        state.break_even_activated = state.break_even_activated or reason in ("BREAK_EVEN", "TRAILING_STOP")
        state.trailing_activated = state.trailing_activated or reason == "TRAILING_STOP"
        base = {
            "signal_id": f"{reason}|{ticket}",
            "symbol": symbol,
            "direction": direction,
            "price": current_price,
            "sl": candidate_sl,
            "tp": tp,
            "ticket": _value(position, "ticket"),
            "comment": _value(position, "comment", ""),
            "dry_run": dry_run,
            "exit_mode": exit_mode,
            "r_current": current_r,
            "break_even_activated": state.break_even_activated,
            "trailing_activated": state.trailing_activated,
            "highest_favorable_price": state.highest_favorable_price,
            "highest_favorable_r": state.highest_favorable_r,
        }
        if dry_run:
            result = ExecutionResult(status=STATUS_DRY_RUN_SL_UPDATE, reason=reason, **base)
        else:
            sent = mt5.order_send({
                "action": getattr(mt5, "TRADE_ACTION_SLTP", 6),
                "position": _value(position, "ticket"),
                "symbol": symbol,
                "sl": candidate_sl,
                "tp": tp,
                "magic": MAGIC_NUMBER,
                "comment": _value(position, "comment", ""),
            })
            retcode = _value(sent, "retcode")
            ok = retcode in (RETCODE_DONE, RETCODE_PLACED)
            result = ExecutionResult(
                status=STATUS_SL_UPDATED if ok else STATUS_ERROR,
                reason=reason if ok else f"SL_UPDATE_FAILED: {_value(sent, 'comment', mt5.last_error())}",
                retcode=retcode,
                **base,
            )
            if ok:
                _notify(notify, f"LIVE_SL_UPDATE|{ticket}|{reason}", format_execution_message(result))
        append_journal(root, result, now)
        results.append(result)
        states[ticket] = state

    for ticket in list(states):
        if ticket not in active_tickets:
            del states[ticket]
    save_exit_states(root, states)
    return results


def _close_gate_allowed(mt5, dry_run):
    account = mt5.account_info()
    gate = evaluate_demo_account(mt5, account) if dry_run else evaluate_demo_order_gate(mt5, account)
    return gate.allowed


def close_expired_positions(mt5, root, symbols, max_hold_bars, timeframe="TIMEFRAME_H1",
                            bar_seconds=3600, dry_run=True, notify=None, now=None):
    """Cierra posiciones del bot que superaron max_hold_bars (salida por tiempo).

    El kill switch no bloquea cierres: cerrar siempre reduce riesgo.
    """
    if not _close_gate_allowed(mt5, dry_run):
        return []

    results = []
    for position in bot_positions(mt5):
        symbol = _value(position, "symbol")
        if symbol not in symbols:
            continue
        tick = mt5.symbol_info_tick(symbol)
        held = bars_held(mt5, position, int(_value(tick, "time", 0) or 0),
                         timeframe=timeframe, bar_seconds=bar_seconds, max_bars=max_hold_bars)
        if held is None or held < max_hold_bars:
            continue
        results.append(close_position(mt5, root, position, "MAX_HOLD_BARS", STATUS_CLOSED_TIME,
                                      dry_run=dry_run, notify=notify, now=now))
    return results


def close_all_positions(mt5, root, dry_run=True, notify=None, now=None):
    """Cierra TODAS las posiciones del bot (boton de emergencia / fin de prueba)."""
    if not _close_gate_allowed(mt5, dry_run):
        return []
    return [
        close_position(mt5, root, position, "CLOSE_ALL", STATUS_CLOSED_MANUAL,
                       dry_run=dry_run, notify=notify, now=now)
        for position in bot_positions(mt5)
    ]


DEAL_ENTRY_OUT = 1
DEAL_REASONS = {0: "MANUAL", 1: "MANUAL", 2: "MANUAL", 3: "BOT", 4: "SL", 5: "TP", 6: "STOP_OUT"}


def notify_closed_deals(mt5, notify, now=None):
    """Avisa cada cierre del bot (SL/TP del broker, tiempo o manual) una sola vez.

    La deduplicacion la hace telegram_events por EVENT_ID (ticket del deal).
    """
    if notify is None:
        return []
    closed = [
        deal for deal in bot_deals(mt5, now=now)
        if _value(deal, "entry") == getattr(mt5, "DEAL_ENTRY_OUT", DEAL_ENTRY_OUT)
    ]
    for deal in closed:
        profit = sum(float(_value(deal, field, 0) or 0) for field in ("profit", "commission", "swap", "fee"))
        header = "✅ POSICION DEMO CERRADA" if profit >= 0 else "🔴 POSICION DEMO CERRADA"
        _notify(notify, _closed_event_id(_value(deal, "position_id")), "\n".join([
            header,
            "",
            f"Simbolo: {_value(deal, 'symbol')}",
            f"Volumen: {_value(deal, 'volume')}",
            f"Precio cierre: {_value(deal, 'price')}",
            f"Motivo: {DEAL_REASONS.get(_value(deal, 'reason'), _value(deal, 'reason'))}",
            f"Resultado: {profit:+.2f}",
            f"Posicion: {_value(deal, 'position_id')}",
        ]))
    return closed


def reconcile(mt5, root, now=None):
    """Compara el journal con lo que realmente existe en MT5."""
    sent = [row for row in read_journal(root) if row.get("STATUS") == STATUS_SENT]
    positions = bot_positions(mt5)
    deals = bot_deals(mt5, now=now)

    open_tickets = {str(_value(p, "ticket")) for p in positions}
    dealt_positions = {str(_value(d, "position_id")) for d in deals}
    journal_comments = {row.get("COMMENT") for row in sent}

    missing = [
        row for row in sent
        if row.get("TICKET") not in open_tickets and row.get("TICKET") not in dealt_positions
    ]
    orphans = [p for p in positions if _value(p, "comment") not in journal_comments]
    return {
        "journal_sent": len(sent),
        "open_positions": positions,
        "missing_in_mt5": missing,
        "orphans_in_mt5": orphans,
        "ok": not missing and not orphans,
    }


def format_execution_message(result):
    header = {
        STATUS_SENT: "🟢 ORDEN DEMO ENVIADA",
        STATUS_ERROR: "🚨 ERROR DE EJECUCION DEMO",
        STATUS_CLOSED_TIME: "⏱️ POSICION DEMO CERRADA POR TIEMPO",
        STATUS_CLOSED_MANUAL: "⏹️ POSICION DEMO CERRADA (CLOSEALL)",
    }.get(result.status, f"FOREX ML - {result.status}")
    return "\n".join([
        header,
        "",
        f"Simbolo: {result.symbol or result.asset}",
        f"Direccion: {result.direction}",
        f"Volumen: {result.volume}",
        f"Precio: {result.price}",
        f"SL: {result.sl}",
        f"TP: {result.tp}",
        f"Ticket: {result.ticket}",
        f"Motivo: {result.reason}",
    ])


def telegram_notifier(root, asset="", config_id=""):
    def notify(event_id, message):
        from src.telegram_events import send_telegram_event

        send_telegram_event(root, event_id, "LIVE", message, asset=asset, config_id=config_id)

    return notify


@contextmanager
def mt5_session():
    from src.mt5_connector import _credenciales_mt5

    credentials, error = _credenciales_mt5()
    if error:
        raise RuntimeError(error)
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 no esta instalado (solo Windows)") from exc

    if not mt5.initialize():
        raise RuntimeError(f"No se pudo inicializar MT5: {mt5.last_error()}")
    try:
        if not mt5.login(credentials["login"], password=credentials["password"], server=credentials["server"]):
            raise RuntimeError(f"No se pudo conectar MT5: {mt5.last_error()}")
        yield mt5
    finally:
        mt5.shutdown()
