from dataclasses import dataclass

from config import DEMO_EXECUTION_ENABLED


DEMO_REQUIRED_REASON = "MT5_ACCOUNT_NOT_DEMO_BLOCKED"
DEMO_DISABLED_REASON = "DEMO_EXECUTION_DISABLED"


@dataclass(frozen=True)
class DemoAccountGate:
    allowed: bool
    reason: str
    trade_mode: object
    expected_demo_mode: object


def demo_execution_enabled():
    return DEMO_EXECUTION_ENABLED is True


def evaluate_demo_account(mt5, account_info=None):
    account_info = account_info if account_info is not None else mt5.account_info()
    trade_mode = getattr(account_info, "trade_mode", None)
    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", None)

    if account_info is None or demo_mode is None or trade_mode != demo_mode:
        return DemoAccountGate(
            allowed=False,
            reason=DEMO_REQUIRED_REASON,
            trade_mode=trade_mode,
            expected_demo_mode=demo_mode,
        )

    return DemoAccountGate(
        allowed=True,
        reason="OK",
        trade_mode=trade_mode,
        expected_demo_mode=demo_mode,
    )


def evaluate_demo_order_gate(mt5, account_info=None):
    demo_gate = evaluate_demo_account(mt5, account_info)
    if not demo_gate.allowed:
        return demo_gate
    if not demo_execution_enabled():
        return DemoAccountGate(
            allowed=False,
            reason=DEMO_DISABLED_REASON,
            trade_mode=demo_gate.trade_mode,
            expected_demo_mode=demo_gate.expected_demo_mode,
        )
    return demo_gate


def block_demo_order_send(mt5, account_info=None):
    gate = evaluate_demo_order_gate(mt5, account_info)
    return {
        "ok": False,
        "sent": False,
        "reason": gate.reason if not gate.allowed else "ORDER_SEND_NOT_IMPLEMENTED",
        "ordenes_enviadas": 0,
    }
