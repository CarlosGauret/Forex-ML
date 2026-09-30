from dataclasses import dataclass, field


DECISION_WAIT = "ESPERAR"
DECISION_BUY = "COMPRAR (LONG)"
DECISION_SELL = "VENDER (SHORT)"
DECISION_SKIP_RISK = "SKIP - RIESGO EXCESIVO"
DECISION_SKIP_BROKER = "SKIP - BROKER RESTRICTION"
DECISION_SKIP_DATA = "SKIP - DATOS INSUFICIENTES"


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    decision: str
    reasons: list[str] = field(default_factory=list)

    @property
    def reason(self):
        return "; ".join(self.reasons)


def _value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _valid_number(value, positive=False):
    if value is None:
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if positive:
        return value > 0
    return value == value


def _distance_ok(price_a, price_b, minimum):
    if price_a is None or price_b is None:
        return True
    return abs(float(price_a) - float(price_b)) >= minimum


def evaluate_trade_filters(
    symbol_info,
    tick,
    volume=None,
    entry=None,
    stop_loss=None,
    take_profit=None,
    max_spread_points=None,
    daily_loss_ok=True,
    drawdown_ok=True,
    max_positions_ok=True,
    model_validated=True,
):
    reasons = []

    if not model_validated:
        return FilterResult(False, DECISION_WAIT, ["MODELO_VALIDADO_NO"])

    bid = _value(tick, "bid")
    ask = _value(tick, "ask")
    if tick is None or not _valid_number(bid, positive=True) or not _valid_number(ask, positive=True):
        return FilterResult(False, DECISION_WAIT, ["MERCADO_CERRADO_O_PRECIO_INVALIDO"])

    if float(ask) < float(bid):
        return FilterResult(False, DECISION_WAIT, ["BID_ASK_INVALIDO"])

    point = _value(symbol_info, "point")
    digits = _value(symbol_info, "digits")
    volume_min = _value(symbol_info, "volume_min")
    volume_max = _value(symbol_info, "volume_max")
    volume_step = _value(symbol_info, "volume_step")
    stops_level = _value(symbol_info, "trade_stops_level", _value(symbol_info, "stops_level", 0)) or 0
    freeze_level = _value(symbol_info, "trade_freeze_level", _value(symbol_info, "freeze_level", 0)) or 0

    if not _valid_number(point, positive=True):
        reasons.append("POINT_INVALIDO")
    if digits is None:
        reasons.append("DIGITS_INVALIDO")
    if not _valid_number(volume_min, positive=True):
        reasons.append("VOLUME_MIN_INVALIDO")
    if not _valid_number(volume_max, positive=True):
        reasons.append("VOLUME_MAX_INVALIDO")
    if not _valid_number(volume_step, positive=True):
        reasons.append("VOLUME_STEP_INVALIDO")

    if reasons:
        return FilterResult(False, DECISION_SKIP_BROKER, reasons)

    point = float(point)
    volume_min = float(volume_min)
    volume_max = float(volume_max)
    volume_step = float(volume_step)
    spread_points = (float(ask) - float(bid)) / point

    if max_spread_points is not None and spread_points > float(max_spread_points):
        return FilterResult(False, DECISION_WAIT, ["SPREAD_EXCESIVO"])

    if volume is not None:
        volume = float(volume)
        if volume < volume_min:
            reasons.append("VOLUME_BELOW_MIN")
        if volume > volume_max:
            reasons.append("VOLUME_ABOVE_MAX")
        step_units = round((volume - volume_min) / volume_step)
        normalized = volume_min + (step_units * volume_step)
        if abs(normalized - volume) > 1e-9:
            reasons.append("VOLUME_STEP_INVALID")

    if reasons:
        return FilterResult(False, DECISION_SKIP_BROKER, reasons)

    minimum_stop_distance = float(stops_level) * point
    if minimum_stop_distance > 0:
        if not _distance_ok(entry, stop_loss, minimum_stop_distance):
            return FilterResult(False, DECISION_SKIP_BROKER, ["STOPS_LEVEL_VIOLATION"])
        if not _distance_ok(entry, take_profit, minimum_stop_distance):
            return FilterResult(False, DECISION_SKIP_BROKER, ["STOPS_LEVEL_VIOLATION"])

    minimum_freeze_distance = float(freeze_level) * point
    if minimum_freeze_distance > 0:
        reference = float(ask)
        if not _distance_ok(reference, stop_loss, minimum_freeze_distance):
            return FilterResult(False, DECISION_SKIP_BROKER, ["FREEZE_LEVEL_VIOLATION"])
        if not _distance_ok(reference, take_profit, minimum_freeze_distance):
            return FilterResult(False, DECISION_SKIP_BROKER, ["FREEZE_LEVEL_VIOLATION"])

    if not daily_loss_ok:
        return FilterResult(False, DECISION_SKIP_RISK, ["MAXIMUM_DAILY_LOSS_REACHED"])
    if not drawdown_ok:
        return FilterResult(False, DECISION_SKIP_RISK, ["MAXIMUM_DRAWDOWN_REACHED"])
    if not max_positions_ok:
        return FilterResult(False, DECISION_SKIP_RISK, ["MAXIMUM_POSITIONS_REACHED"])

    return FilterResult(True, "OK", ["OK"])


def final_decision_from_signal(signal, filters):
    if not filters.passed:
        return filters.decision
    if signal == "LONG":
        return DECISION_BUY
    if signal == "SHORT":
        return DECISION_SELL
    return DECISION_WAIT
