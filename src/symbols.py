from dataclasses import dataclass


TARGET_LOGICAL_SYMBOLS = [
    "GOLD",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
]

LOGICAL_BASES = {
    "GOLD": ["XAUUSD", "GOLD"],
    "EURUSD": ["EURUSD"],
    "GBPUSD": ["GBPUSD"],
    "USDJPY": ["USDJPY"],
    "AUDUSD": ["AUDUSD"],
    "USDCAD": ["USDCAD"],
    "USDCHF": ["USDCHF"],
    "NZDUSD": ["NZDUSD"],
}

COMMON_SUFFIXES = ["", ".", "m", "#", ".m", "_m", "-m"]


@dataclass(frozen=True)
class ResolvedSymbol:
    logical_symbol: str
    mt5_symbol: str
    description: str
    digits: int | None
    point: float | None
    contract_size: float | None
    volume_min: float | None
    volume_max: float | None
    volume_step: float | None
    found: bool
    reason: str


def _value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text(obj):
    name = str(_value(obj, "name", "") or "")
    description = str(_value(obj, "description", "") or "")
    return name, description, f"{name} {description}".upper()


def _score_symbol(symbol_obj, logical_symbol):
    logical_symbol = logical_symbol.upper()
    bases = LOGICAL_BASES.get(logical_symbol, [logical_symbol])
    name, description, haystack = _text(symbol_obj)
    name_upper = name.upper()

    best = None
    for base in bases:
        base = base.upper()
        exact_variants = {f"{base}{suffix}".upper() for suffix in COMMON_SUFFIXES}
        if name_upper == base:
            best = 0 if best is None else min(best, 0)
        elif name_upper in exact_variants:
            best = 1 if best is None else min(best, 1)
        elif name_upper.startswith(base):
            best = 2 if best is None else min(best, 2)
        elif base in haystack:
            best = 3 if best is None else min(best, 3)

    if logical_symbol == "GOLD":
        gold_words = ("XAUUSD", "GOLD")
        if any(word == name_upper for word in gold_words):
            best = 0 if best is None else min(best, 0)
        elif any(name_upper.startswith(word) for word in gold_words):
            best = 1 if best is None else min(best, 1)
        elif any(word in haystack for word in gold_words):
            best = 2 if best is None else min(best, 2)

    if best is None:
        return None

    visible_penalty = 0 if bool(_value(symbol_obj, "visible", True)) else 10
    return best + visible_penalty


def _empty_result(logical_symbol, reason="SYMBOL_NOT_FOUND"):
    return ResolvedSymbol(
        logical_symbol=logical_symbol,
        mt5_symbol="",
        description="",
        digits=None,
        point=None,
        contract_size=None,
        volume_min=None,
        volume_max=None,
        volume_step=None,
        found=False,
        reason=reason,
    )


def _result_from_info(logical_symbol, symbol_obj, info=None):
    info = info or symbol_obj
    name = _value(info, "name", _value(symbol_obj, "name", ""))
    return ResolvedSymbol(
        logical_symbol=logical_symbol,
        mt5_symbol=str(name or ""),
        description=str(_value(info, "description", _value(symbol_obj, "description", "")) or ""),
        digits=_value(info, "digits"),
        point=_value(info, "point"),
        contract_size=_value(
            info,
            "trade_contract_size",
            _value(info, "contract_size"),
        ),
        volume_min=_value(info, "volume_min"),
        volume_max=_value(info, "volume_max"),
        volume_step=_value(info, "volume_step"),
        found=True,
        reason="OK",
    )


def resolve_symbol(logical_symbol, symbols, symbol_info_get=None):
    logical_symbol = logical_symbol.upper()
    candidates = []

    for symbol_obj in symbols or []:
        score = _score_symbol(symbol_obj, logical_symbol)
        if score is None:
            continue
        name = str(_value(symbol_obj, "name", "") or "")
        candidates.append((score, len(name), name, symbol_obj))

    if not candidates:
        return _empty_result(logical_symbol)

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = candidates[0][3]
    selected_name = _value(selected, "name", "")
    info = symbol_info_get(selected_name) if symbol_info_get is not None else selected
    return _result_from_info(logical_symbol, selected, info)


def discover_symbols(mt5, logical_symbols=None):
    logical_symbols = logical_symbols or TARGET_LOGICAL_SYMBOLS
    symbols = mt5.symbols_get()
    if symbols is None:
        return {
            logical.upper(): _empty_result(logical.upper(), "MT5_SYMBOLS_UNAVAILABLE")
            for logical in logical_symbols
        }

    return {
        logical.upper(): resolve_symbol(
            logical_symbol=logical,
            symbols=symbols,
            symbol_info_get=mt5.symbol_info,
        )
        for logical in logical_symbols
    }


def resolved_to_dict(resolved):
    return {
        "logical_symbol": resolved.logical_symbol,
        "mt5_symbol": resolved.mt5_symbol,
        "description": resolved.description,
        "digits": resolved.digits,
        "point": resolved.point,
        "contract_size": resolved.contract_size,
        "volume_min": resolved.volume_min,
        "volume_max": resolved.volume_max,
        "volume_step": resolved.volume_step,
        "found": resolved.found,
        "reason": resolved.reason,
    }
