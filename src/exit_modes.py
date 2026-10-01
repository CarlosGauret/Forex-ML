import math

import pandas as pd

from config import BREAK_EVEN_TRIGGER_R, TRAILING_DISTANCE_R, TRAILING_START_R
from src.risk import BUY, SELL


EXIT_FIXED_TP = "FIXED_TP"
EXIT_BREAK_EVEN_TP = "BREAK_EVEN_TP"
EXIT_TRAILING = "TRAILING"
EXIT_MODES = (EXIT_FIXED_TP, EXIT_BREAK_EVEN_TP, EXIT_TRAILING)


def _trade_r(direction, entry, exit_price, initial_risk):
    if initial_risk <= 0:
        return 0.0
    if direction == BUY:
        return (exit_price - entry) / initial_risk
    return (entry - exit_price) / initial_risk


def _advance_stop(direction, current_sl, candidate):
    if direction == BUY:
        return max(current_sl, candidate)
    return min(current_sl, candidate)


def simulate_exit_mode(entry, future_bars, mode):
    """Simulate one trade with conservative same-bar ordering.

    `future_bars` must be chronological and include High/Low. Same-bar SL/TP
    resolves to SL first, matching the existing conservative project convention.
    """
    direction = entry["direction"]
    if direction not in (BUY, SELL):
        raise ValueError(f"Unsupported direction: {direction}")
    if mode not in EXIT_MODES:
        raise ValueError(f"Unsupported exit mode: {mode}")

    entry_price = float(entry["entry"])
    initial_sl = float(entry["sl"])
    initial_tp = None if pd.isna(entry.get("tp")) else float(entry.get("tp"))
    initial_risk = abs(entry_price - initial_sl)
    if entry_price <= 0 or initial_sl <= 0 or initial_risk <= 0:
        return {"exit_reason": "INVALID_SL", "profit_r": 0.0, "bars_held": 0}

    current_sl = initial_sl
    break_even_activated = False
    trailing_activated = False
    best_r = 0.0

    for bars_held, (_, bar) in enumerate(future_bars.iterrows(), start=1):
        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar.get("Close", high if direction == BUY else low))
        favorable_price = high if direction == BUY else low
        best_r = max(best_r, _trade_r(direction, entry_price, favorable_price, initial_risk))

        if mode in (EXIT_BREAK_EVEN_TP, EXIT_TRAILING) and best_r >= BREAK_EVEN_TRIGGER_R:
            current_sl = _advance_stop(direction, current_sl, entry_price)
            break_even_activated = True

        if mode == EXIT_TRAILING and best_r >= TRAILING_START_R:
            trail = (
                favorable_price - initial_risk * TRAILING_DISTANCE_R
                if direction == BUY
                else favorable_price + initial_risk * TRAILING_DISTANCE_R
            )
            current_sl = _advance_stop(direction, current_sl, trail)
            trailing_activated = True

        sl_hit = low <= current_sl if direction == BUY else high >= current_sl
        tp_hit = False
        if mode != EXIT_TRAILING and initial_tp is not None:
            tp_hit = high >= initial_tp if direction == BUY else low <= initial_tp

        if sl_hit:
            reason = "BREAK_EVEN" if break_even_activated and math.isclose(current_sl, entry_price) else "STOP_LOSS"
            if trailing_activated and not math.isclose(current_sl, entry_price):
                reason = "TRAILING_STOP"
            return {
                "exit_reason": reason,
                "exit_price": current_sl,
                "profit_r": _trade_r(direction, entry_price, current_sl, initial_risk),
                "bars_held": bars_held,
                "break_even_activated": break_even_activated,
                "trailing_activated": trailing_activated,
            }
        if tp_hit:
            return {
                "exit_reason": "TAKE_PROFIT",
                "exit_price": initial_tp,
                "profit_r": _trade_r(direction, entry_price, initial_tp, initial_risk),
                "bars_held": bars_held,
                "break_even_activated": break_even_activated,
                "trailing_activated": trailing_activated,
            }

    exit_price = close if len(future_bars) else entry_price
    return {
        "exit_reason": "UNKNOWN" if len(future_bars) == 0 else "END_OF_DATA",
        "exit_price": exit_price,
        "profit_r": _trade_r(direction, entry_price, exit_price, initial_risk),
        "bars_held": len(future_bars),
        "break_even_activated": break_even_activated,
        "trailing_activated": trailing_activated,
    }


def summarize_exit_mode_results(results):
    rows = pd.DataFrame(results)
    if rows.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_r": 0.0,
            "net_r": 0.0,
            "average_bars": 0.0,
        }
    wins = rows[rows["profit_r"] > 0]["profit_r"]
    losses = rows[rows["profit_r"] < 0]["profit_r"]
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(rows)),
        "win_rate": float((rows["profit_r"] > 0).mean()),
        "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
        "expectancy": float(rows["profit_r"].mean()),
        "average_r": float(rows["profit_r"].mean()),
        "median_r": float(rows["profit_r"].median()),
        "net_r": float(rows["profit_r"].sum()),
        "average_bars": float(rows["bars_held"].mean()),
        "best_trade_r": float(rows["profit_r"].max()),
        "worst_trade_r": float(rows["profit_r"].min()),
    }
