"""Conecta los forward PAPER congelados con el ejecutor MT5.

El PAPER sigue siendo la fuente de verdad de la senal (no se modifica). Tras
procesarlo, si queda una senal pendiente de la ultima vela cerrada, se envia
la orden ahora: equivale a la entrada del PAPER en la apertura de la vela
siguiente. Una senal pendiente mas vieja (datos atrasados) no se ejecuta y se
registra como STALE_SIGNAL.
"""

import pandas as pd

from config import DEMO_EXECUTION_ENABLED
from src.live_executor import (
    STATUS_SKIP,
    ExecutionResult,
    OrderIntent,
    append_journal,
    close_expired_positions,
    execute_signal,
    mt5_session,
    notify_closed_deals,
    order_comment,
    read_journal,
    telegram_notifier,
)
from src.risk import BUY, SELL
from src.symbols import discover_symbols


STALE_REASON = "STALE_SIGNAL"
GOLD_LIVE_CONFIG_ID = "GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060"


def _same_bar(left, right):
    if not left or not right:
        return False
    return pd.to_datetime(left) == pd.to_datetime(right)


def last_closed_h1_bar(now=None):
    now_ts = pd.to_datetime(now or pd.Timestamp.now(tz="UTC"), utc=True)
    return (now_ts.floor("h") - pd.Timedelta(hours=1)).tz_convert(None).isoformat()


def fresh_pending_signal(state):
    """Devuelve la senal pendiente EURUSD solo si es de la ultima vela cerrada."""
    pending = state.get("pending_signal")
    if not pending:
        return None
    if not _same_bar(pending.get("FECHA_SIGNAL"), state.get("last_closed_bar")):
        return None
    return pending


def _record_stale_once(root, intent, dry_run):
    for row in read_journal(root):
        if row.get("SIGNAL_ID") == intent.signal_id and row.get("REASON") == STALE_REASON:
            return None
    result = ExecutionResult(
        status=STATUS_SKIP,
        reason=STALE_REASON,
        signal_id=intent.signal_id,
        config_id=intent.config_id,
        asset=intent.asset,
        direction=intent.direction,
        comment=order_comment(intent.signal_id),
        dry_run=dry_run,
    )
    append_journal(root, result)
    return result


def _run_live(root, asset, config_id, intent, fresh, max_hold_bars, paper, paper_error,
              dry_run, mt5_factory):
    notify = None if dry_run else telegram_notifier(root, asset, config_id)
    with mt5_factory() as mt5:
        notify_closed_deals(mt5, notify)
        symbol = discover_symbols(mt5, [asset])[asset].mt5_symbol
        closed = close_expired_positions(
            mt5,
            root,
            symbols={symbol},
            max_hold_bars=max_hold_bars,
            dry_run=dry_run,
            notify=notify,
        )
        execution = None
        if intent is not None:
            if fresh:
                execution = execute_signal(mt5, intent, root, dry_run=dry_run, notify=notify)
            else:
                execution = _record_stale_once(root, intent, dry_run)

    return {
        "asset": asset,
        "config_id": config_id,
        "paper": paper,
        "paper_error": paper_error,
        "dry_run": dry_run,
        "intent": intent,
        "fresh": fresh,
        "execution": execution,
        "closed": closed,
    }


def _resolve_dry_run(dry_run):
    return (not DEMO_EXECUTION_ENABLED) if dry_run is None else dry_run


def run_eurusd_live(root, dry_run=None, mt5_factory=mt5_session, now=None):
    from src import eurusd_forward as forward

    dry_run = _resolve_dry_run(dry_run)
    paper, paper_error = None, ""
    try:
        paper = forward.process_eurusd_forward(root, now=now)
    except Exception as error:
        # Sin PAPER no hay senal nueva, pero los cierres por tiempo deben seguir.
        paper_error = str(error)

    intent, fresh = None, False
    if not paper_error:
        state = forward._read_json(forward._paths(root)["state"], {})
        pending = state.get("pending_signal")
        if pending:
            intent = OrderIntent(
                signal_id=pending["SIGNAL_ID"],
                config_id=forward.CONFIG_ID,
                asset=forward.ASSET,
                direction=BUY,
                atr=float(pending["ATR"]),
            )
            fresh = fresh_pending_signal(state) is not None

    return _run_live(root, forward.ASSET, forward.CONFIG_ID, intent, fresh,
                     forward.MAX_HOLD_BARS, paper, paper_error, dry_run, mt5_factory)


def run_gold_live(root, dry_run=None, mt5_factory=mt5_session, now=None):
    """GOLD: senal del PAPER congelado (datos GC=F), orden en el oro de MT5 (XAUUSD).

    Solo T060 (principal) ejecuta: T055 sigue en PAPER porque ambas operan el
    mismo simbolo y el bot mantiene una posicion por simbolo.
    """
    from config_paper import PAPER_CONFIGS
    from src import paper_trader

    config = next(c for c in PAPER_CONFIGS if c["CONFIG_ID"] == GOLD_LIVE_CONFIG_ID)
    dry_run = _resolve_dry_run(dry_run)
    paper, paper_error = None, ""
    try:
        paper = paper_trader.ejecutar_paper(root)
    except Exception as error:
        paper_error = str(error)

    intent, fresh = None, False
    if not paper_error:
        state = paper_trader._leer_json(paper_trader._rutas(root)["state"], {})
        pending = next(
            (p for p in state.get("pending_signals", []) if p.get("CONFIG_ID") == GOLD_LIVE_CONFIG_ID),
            None,
        )
        if pending:
            intent = OrderIntent(
                signal_id=f"{GOLD_LIVE_CONFIG_ID}|{pd.to_datetime(pending['FECHA_SIGNAL']).isoformat()}",
                config_id=GOLD_LIVE_CONFIG_ID,
                asset=config["ACTIVO"],
                direction=BUY if config["DIRECCION"] == "LONG" else SELL,
                atr=float(pending["ATR_SIGNAL"]),
                stop_atr=float(config["STOP_ATR"]),
                take_profit_atr=float(config["TAKE_PROFIT_ATR"]),
            )
            fresh = _same_bar(pending["FECHA_SIGNAL"], last_closed_h1_bar(now))

    return _run_live(root, config["ACTIVO"], GOLD_LIVE_CONFIG_ID, intent, fresh,
                     int(config["MAX_HOLD_BARS"]), paper, paper_error, dry_run, mt5_factory)
