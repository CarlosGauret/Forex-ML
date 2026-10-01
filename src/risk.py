import math
from dataclasses import dataclass
from typing import Optional

from config import (
    MAX_POSITIONS_PER_SYMBOL,
    MAX_TOTAL_OPEN_RISK_PERCENT,
    MAXIMUM_DAILY_LOSS,
    MAXIMUM_DRAWDOWN,
    MAXIMUM_SIMULTANEOUS_POSITIONS,
    RISK_PER_TRADE,
)
from src.safety import evaluate_trade_gate


BUY = "BUY"
SELL = "SELL"
DECISION_EXECUTABLE = "EXECUTABLE"
DECISION_SKIP = "SKIP"


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = RISK_PER_TRADE
    maximum_daily_loss: float = MAXIMUM_DAILY_LOSS
    maximum_simultaneous_positions: int = MAXIMUM_SIMULTANEOUS_POSITIONS
    maximum_positions_per_symbol: int = MAX_POSITIONS_PER_SYMBOL
    maximum_total_open_risk: float = MAX_TOTAL_OPEN_RISK_PERCENT / 100
    maximum_drawdown: Optional[float] = MAXIMUM_DRAWDOWN
    kill_switch: bool = False


@dataclass(frozen=True)
class RiskGateResult:
    allowed: bool
    decision: str
    reason: str


@dataclass(frozen=True)
class PositionSizingResult:
    ok: bool
    decision: str
    reason: str
    symbol: str
    direction: str
    risk_amount: float
    theoretical_volume: Optional[float]
    volume: Optional[float]
    estimated_loss: Optional[float]
    min_volume_loss: Optional[float]
    risk_pct_real: Optional[float]
    risk_pct_minimum: Optional[float]
    volume_min: float
    volume_max: float
    volume_step: float
    loss_source: str


def _value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _precision(step):
    text = f"{float(step):.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".")[1])


def normalize_volume_down(volume, volume_step, volume_max):
    if volume is None or volume_step <= 0:
        return volume

    precision = _precision(volume_step)
    normalized = math.floor((float(volume) + 1e-12) / volume_step) * volume_step
    normalized = min(normalized, volume_max)
    return round(normalized, precision)


def _loss_from_order_calc(order_calc_profit, order_type, symbol, volume, entry, stop_loss):
    if order_calc_profit is None:
        return None

    profit = order_calc_profit(order_type, symbol, volume, entry, stop_loss)
    if profit is None:
        return None
    return abs(float(profit))


def _loss_from_symbol_info(symbol_info, volume, entry, stop_loss):
    distance = abs(float(entry) - float(stop_loss))
    tick_size = float(
        _value(symbol_info, "trade_tick_size", _value(symbol_info, "tick_size", 0))
        or 0
    )
    tick_value = float(
        _value(symbol_info, "trade_tick_value", _value(symbol_info, "tick_value", 0))
        or 0
    )
    contract_size = float(
        _value(
            symbol_info,
            "trade_contract_size",
            _value(symbol_info, "contract_size", 0),
        )
        or 0
    )

    if tick_size > 0 and tick_value > 0:
        return (distance / tick_size) * tick_value * volume, "SYMBOL_TICK_VALUE"

    if contract_size > 0:
        return distance * contract_size * volume, "CONTRACT_SIZE_FALLBACK"

    return None, "NO_LOSS_MODEL"


def _estimate_loss(
    symbol,
    direction,
    volume,
    entry,
    stop_loss,
    symbol_info,
    order_calc_profit=None,
    order_type=None,
):
    mt5_loss = _loss_from_order_calc(
        order_calc_profit=order_calc_profit,
        order_type=order_type if order_type is not None else direction,
        symbol=symbol,
        volume=volume,
        entry=entry,
        stop_loss=stop_loss,
    )
    if mt5_loss is not None:
        return mt5_loss, "MT5_ORDER_CALC_PROFIT"

    loss, source = _loss_from_symbol_info(symbol_info, volume, entry, stop_loss)
    return loss, source


class RiskManager:
    def __init__(self, limits=None):
        self.limits = limits or RiskLimits()

    def evaluate_trade_permission(
        self,
        balance,
        equity=None,
        starting_day_equity=None,
        daily_loss=0.0,
        open_positions=0,
        open_risk_amount=0.0,
        new_trade_risk_amount=0.0,
        current_drawdown=0.0,
        symbol=None,
        open_symbols=None,
        safety_gate=None,
    ):
        # safety_gate permite que la ejecucion DEMO pase su propia puerta
        # (cuenta demo + DEMO_EXECUTION_ENABLED). Por defecto: TRADING_ENABLED.
        safety = safety_gate if safety_gate is not None else evaluate_trade_gate("ORDER_SEND")
        if not safety.allowed:
            return RiskGateResult(False, DECISION_SKIP, safety.reason)

        if self.limits.kill_switch:
            return RiskGateResult(False, DECISION_SKIP, "KILL_SWITCH_ACTIVE")

        balance = float(balance)
        equity = balance if equity is None else float(equity)
        risk_base = equity if equity > 0 else balance
        daily_loss = abs(float(daily_loss))
        if starting_day_equity is not None:
            day_base = float(starting_day_equity)
            if day_base > 0 and equity <= day_base * (1 - self.limits.maximum_daily_loss):
                return RiskGateResult(False, DECISION_SKIP, "DAILY_LOSS_LIMIT")
        if risk_base > 0 and daily_loss >= risk_base * self.limits.maximum_daily_loss:
            return RiskGateResult(False, DECISION_SKIP, "MAXIMUM_DAILY_LOSS_REACHED")

        if open_positions >= self.limits.maximum_simultaneous_positions:
            return RiskGateResult(False, DECISION_SKIP, "MAXIMUM_POSITIONS_REACHED")

        if symbol is not None and open_symbols is not None:
            normalized_symbol = str(symbol).upper()
            normalized_open_symbols = {str(item).upper() for item in open_symbols}
            symbol_count = sum(1 for item in normalized_open_symbols if item == normalized_symbol)
            if symbol_count >= self.limits.maximum_positions_per_symbol:
                return RiskGateResult(False, DECISION_SKIP, "SYMBOL_POSITION_ALREADY_OPEN")

        total_open_risk = float(open_risk_amount or 0) + float(new_trade_risk_amount or 0)
        if risk_base > 0 and total_open_risk > risk_base * self.limits.maximum_total_open_risk:
            return RiskGateResult(False, DECISION_SKIP, "MAX_TOTAL_OPEN_RISK")

        if (
            self.limits.maximum_drawdown is not None
            and current_drawdown >= self.limits.maximum_drawdown
        ):
            return RiskGateResult(False, DECISION_SKIP, "MAXIMUM_DRAWDOWN_REACHED")

        return RiskGateResult(True, DECISION_EXECUTABLE, "OK")

    def calculate_position_size(
        self,
        symbol,
        direction,
        entry,
        stop_loss,
        balance,
        symbol_info,
        equity=None,
        order_calc_profit=None,
        order_type=None,
    ):
        entry = float(entry)
        stop_loss = float(stop_loss)
        balance = float(balance)
        equity = balance if equity is None else float(equity)
        risk_base = equity if equity > 0 else balance
        risk_amount = risk_base * self.limits.risk_per_trade
        volume_min = float(_value(symbol_info, "volume_min", 0) or 0)
        volume_max = float(_value(symbol_info, "volume_max", 0) or 0)
        volume_step = float(_value(symbol_info, "volume_step", 0) or 0)

        if direction not in (BUY, SELL):
            raise ValueError(f"Direccion no soportada: {direction}")
        if risk_base <= 0 or risk_amount <= 0:
            raise ValueError("equity/balance y risk_amount deben ser mayores a 0.")
        if entry <= 0 or stop_loss <= 0 or entry == stop_loss:
            raise ValueError("entry y stop_loss deben ser validos y diferentes.")
        if volume_min <= 0 or volume_max <= 0 or volume_step <= 0:
            return PositionSizingResult(
                ok=False,
                decision=DECISION_SKIP,
                reason="INVALID_VOLUME_CONSTRAINTS",
                symbol=symbol,
                direction=direction,
                risk_amount=risk_amount,
                theoretical_volume=None,
                volume=None,
                estimated_loss=None,
                min_volume_loss=None,
                risk_pct_real=None,
                risk_pct_minimum=None,
                volume_min=volume_min,
                volume_max=volume_max,
                volume_step=volume_step,
                loss_source="NONE",
            )

        loss_one_lot, source = _estimate_loss(
            symbol,
            direction,
            1.0,
            entry,
            stop_loss,
            symbol_info,
            order_calc_profit,
            order_type,
        )
        min_volume_loss, source_min = _estimate_loss(
            symbol,
            direction,
            volume_min,
            entry,
            stop_loss,
            symbol_info,
            order_calc_profit,
            order_type,
        )
        source = source if source != "NO_LOSS_MODEL" else source_min

        if loss_one_lot is None or loss_one_lot <= 0 or min_volume_loss is None:
            return PositionSizingResult(
                ok=False,
                decision=DECISION_SKIP,
                reason="LOSS_ESTIMATION_UNAVAILABLE",
                symbol=symbol,
                direction=direction,
                risk_amount=risk_amount,
                theoretical_volume=None,
                volume=None,
                estimated_loss=None,
                min_volume_loss=min_volume_loss,
                risk_pct_real=None,
                risk_pct_minimum=None,
                volume_min=volume_min,
                volume_max=volume_max,
                volume_step=volume_step,
                loss_source=source,
            )

        risk_pct_minimum = (min_volume_loss / risk_base) * 100
        if min_volume_loss > risk_amount:
            return PositionSizingResult(
                ok=False,
                decision=DECISION_SKIP,
                reason="MINIMUM_VOLUME_EXCEEDS_RISK",
                symbol=symbol,
                direction=direction,
                risk_amount=risk_amount,
                theoretical_volume=risk_amount / loss_one_lot,
                volume=None,
                estimated_loss=None,
                min_volume_loss=min_volume_loss,
                risk_pct_real=None,
                risk_pct_minimum=risk_pct_minimum,
                volume_min=volume_min,
                volume_max=volume_max,
                volume_step=volume_step,
                loss_source=source,
            )

        theoretical_volume = risk_amount / loss_one_lot
        volume = normalize_volume_down(theoretical_volume, volume_step, volume_max)
        if volume is None or volume < volume_min:
            return PositionSizingResult(
                ok=False,
                decision=DECISION_SKIP,
                reason="NORMALIZED_VOLUME_BELOW_MINIMUM",
                symbol=symbol,
                direction=direction,
                risk_amount=risk_amount,
                theoretical_volume=theoretical_volume,
                volume=None,
                estimated_loss=None,
                min_volume_loss=min_volume_loss,
                risk_pct_real=None,
                risk_pct_minimum=risk_pct_minimum,
                volume_min=volume_min,
                volume_max=volume_max,
                volume_step=volume_step,
                loss_source=source,
            )

        estimated_loss, source = _estimate_loss(
            symbol,
            direction,
            volume,
            entry,
            stop_loss,
            symbol_info,
            order_calc_profit,
            order_type,
        )

        while volume >= volume_min and (
            estimated_loss is None or estimated_loss > risk_amount
        ):
            volume = round(volume - volume_step, _precision(volume_step))
            if volume < volume_min:
                break
            estimated_loss, source = _estimate_loss(
                symbol,
                direction,
                volume,
                entry,
                stop_loss,
                symbol_info,
                order_calc_profit,
                order_type,
            )

        if volume < volume_min or estimated_loss is None:
            return PositionSizingResult(
                ok=False,
                decision=DECISION_SKIP,
                reason="NO_VALID_VOLUME_WITHIN_RISK",
                symbol=symbol,
                direction=direction,
                risk_amount=risk_amount,
                theoretical_volume=theoretical_volume,
                volume=None,
                estimated_loss=None,
                min_volume_loss=min_volume_loss,
                risk_pct_real=None,
                risk_pct_minimum=risk_pct_minimum,
                volume_min=volume_min,
                volume_max=volume_max,
                volume_step=volume_step,
                loss_source=source,
            )

        return PositionSizingResult(
            ok=True,
            decision=DECISION_EXECUTABLE,
            reason="OK",
            symbol=symbol,
            direction=direction,
            risk_amount=risk_amount,
            theoretical_volume=theoretical_volume,
            volume=volume,
            estimated_loss=estimated_loss,
            min_volume_loss=min_volume_loss,
            risk_pct_real=(estimated_loss / risk_base) * 100,
            risk_pct_minimum=risk_pct_minimum,
            volume_min=volume_min,
            volume_max=volume_max,
            volume_step=volume_step,
            loss_source=source,
        )
