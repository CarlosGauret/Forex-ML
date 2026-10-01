import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass
class DailyRiskState:
    trading_day: str
    starting_equity: float


@dataclass
class ProtectiveExitState:
    ticket: str
    symbol: str
    entry_price: float | None = None
    initial_sl: float | None = None
    initial_risk: float | None = None
    highest_favorable_price: float | None = None
    highest_favorable_r: float = 0.0
    break_even_activated: bool = False
    trailing_activated: bool = False
    current_sl: float | None = None


def _state_dir(root):
    path = Path(root) / "live" / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def daily_risk_state_path(root):
    return _state_dir(root) / "daily_risk.json"


def exit_state_path(root):
    return _state_dir(root) / "protective_exits.json"


def _read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_or_roll_daily_risk_state(root, equity, now=None):
    """Persist the reference equity for the current trading day.

    The state survives process/Windows/MT5 restarts and rolls automatically when
    the date changes. Dates use UTC because MT5 ticks and journals are UTC-based.
    """
    today = (now.date() if hasattr(now, "date") else date.today()).isoformat()
    path = daily_risk_state_path(root)
    raw = _read_json(path, {})
    if raw.get("trading_day") != today or float(raw.get("starting_equity", 0) or 0) <= 0:
        state = DailyRiskState(trading_day=today, starting_equity=float(equity))
        _write_json(path, asdict(state))
        return state
    return DailyRiskState(
        trading_day=raw["trading_day"],
        starting_equity=float(raw["starting_equity"]),
    )


def load_exit_states(root):
    raw = _read_json(exit_state_path(root), {})
    return {
        str(ticket): ProtectiveExitState(ticket=str(ticket), **{
            key: value for key, value in data.items() if key != "ticket"
        })
        for ticket, data in raw.items()
    }


def save_exit_states(root, states):
    _write_json(exit_state_path(root), {str(k): asdict(v) for k, v in states.items()})
