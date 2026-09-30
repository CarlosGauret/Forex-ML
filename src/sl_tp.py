from dataclasses import dataclass
from typing import Optional


ATR_ONLY = "ATR_ONLY"
SWING_ATR = "SWING_ATR"
SWING_ATR_RR = "SWING_ATR_RR"
SWING_STRUCTURAL_TARGET = "SWING_STRUCTURAL_TARGET"

LONG = "LONG"
SHORT = "SHORT"


@dataclass(frozen=True)
class SLTPResult:
    ok: bool
    direction: str
    mode: str
    entry: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_distance: Optional[float]
    rr: Optional[float]
    reason: str


def _validar_base(entry, atr, direction, mode):
    if direction not in (LONG, SHORT):
        raise ValueError(f"Direccion no soportada: {direction}")

    if mode not in (ATR_ONLY, SWING_ATR, SWING_ATR_RR, SWING_STRUCTURAL_TARGET):
        raise ValueError(f"Modo SL/TP no soportado: {mode}")

    entry = float(entry)
    atr = float(atr)
    if entry <= 0:
        raise ValueError("entry debe ser mayor a 0.")
    if atr <= 0:
        raise ValueError("ATR debe ser mayor a 0.")

    return entry, atr


def _risk_distance(direction, entry, stop_loss):
    if direction == LONG:
        return entry - stop_loss
    return stop_loss - entry


def _validar_stop(direction, entry, stop_loss):
    if stop_loss is None:
        return False
    if direction == LONG:
        return stop_loss < entry
    return stop_loss > entry


def _tp_por_rr(direction, entry, risk_distance, rr):
    if direction == LONG:
        return entry + (risk_distance * rr)
    return entry - (risk_distance * rr)


def _sl_atr_only(direction, entry, atr, atr_multiplier):
    if direction == LONG:
        return entry - (atr * atr_multiplier)
    return entry + (atr * atr_multiplier)


def _sl_swing_atr(direction, entry, atr, atr_multiplier, swing_low, swing_high):
    buffer = atr * atr_multiplier
    if direction == LONG:
        if swing_low is None or float(swing_low) >= entry:
            return None
        return float(swing_low) - buffer

    if swing_high is None or float(swing_high) <= entry:
        return None
    return float(swing_high) + buffer


def calcular_sl_tp(
    entry,
    direction,
    atr,
    mode=ATR_ONLY,
    atr_multiplier=1.0,
    rr=2.0,
    take_profit_atr_multiplier=2.0,
    swing_low=None,
    swing_high=None,
    structural_target=None,
):
    entry, atr = _validar_base(entry, atr, direction, mode)
    atr_multiplier = float(atr_multiplier)
    rr = float(rr)

    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier debe ser mayor a 0.")
    if rr <= 0:
        raise ValueError("rr debe ser mayor a 0.")

    if mode == ATR_ONLY:
        stop_loss = _sl_atr_only(direction, entry, atr, atr_multiplier)
        risk_distance = _risk_distance(direction, entry, stop_loss)
        take_profit = _tp_por_rr(direction, entry, atr * take_profit_atr_multiplier, 1.0)
        return SLTPResult(
            ok=True,
            direction=direction,
            mode=mode,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_distance=risk_distance,
            rr=abs(take_profit - entry) / risk_distance,
            reason="OK",
        )

    stop_loss = _sl_swing_atr(
        direction=direction,
        entry=entry,
        atr=atr,
        atr_multiplier=atr_multiplier,
        swing_low=swing_low,
        swing_high=swing_high,
    )
    if not _validar_stop(direction, entry, stop_loss):
        return SLTPResult(
            ok=False,
            direction=direction,
            mode=mode,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=None,
            risk_distance=None,
            rr=None,
            reason="STRUCTURAL_STOP_NOT_AVAILABLE",
        )

    risk_distance = _risk_distance(direction, entry, stop_loss)
    if risk_distance <= 0:
        return SLTPResult(
            ok=False,
            direction=direction,
            mode=mode,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=None,
            risk_distance=None,
            rr=None,
            reason="INVALID_RISK_DISTANCE",
        )

    if mode == SWING_ATR:
        return SLTPResult(
            ok=True,
            direction=direction,
            mode=mode,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=None,
            risk_distance=risk_distance,
            rr=None,
            reason="TAKE_PROFIT_NOT_SET",
        )

    if mode == SWING_ATR_RR:
        take_profit = _tp_por_rr(direction, entry, risk_distance, rr)
    else:
        if structural_target is None:
            return SLTPResult(
                ok=False,
                direction=direction,
                mode=mode,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=None,
                risk_distance=risk_distance,
                rr=None,
                reason="STRUCTURAL_TARGET_NOT_AVAILABLE",
            )
        take_profit = float(structural_target)

    if direction == LONG and take_profit <= entry:
        return SLTPResult(
            ok=False,
            direction=direction,
            mode=mode,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_distance=risk_distance,
            rr=None,
            reason="INVALID_TAKE_PROFIT",
        )
    if direction == SHORT and take_profit >= entry:
        return SLTPResult(
            ok=False,
            direction=direction,
            mode=mode,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_distance=risk_distance,
            rr=None,
            reason="INVALID_TAKE_PROFIT",
        )

    return SLTPResult(
        ok=True,
        direction=direction,
        mode=mode,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_distance=risk_distance,
        rr=abs(take_profit - entry) / risk_distance,
        reason="OK",
    )
