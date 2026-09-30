from dataclasses import dataclass

from config import TRADING_ENABLED


@dataclass(frozen=True)
class TradeGateResult:
    allowed: bool
    reason: str
    trading_enabled: bool
    requested_action: str


def trading_enabled():
    return TRADING_ENABLED is True


def evaluate_trade_gate(requested_action="TRADE"):
    enabled = trading_enabled()
    if not enabled:
        return TradeGateResult(
            allowed=False,
            reason="TRADING_DISABLED_FAIL_CLOSED",
            trading_enabled=False,
            requested_action=requested_action,
        )

    return TradeGateResult(
        allowed=True,
        reason="TRADING_ENABLED",
        trading_enabled=True,
        requested_action=requested_action,
    )


def assert_trade_allowed(requested_action="TRADE"):
    result = evaluate_trade_gate(requested_action)
    if not result.allowed:
        raise PermissionError(result.reason)
    return result
