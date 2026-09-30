COSTO_POR_OPERACION = 0
PROB_THRESHOLD = 0.60
PROB_THRESHOLDS = [0.55, 0.60, 0.65]
COST_BPS = [0, 1, 2, 5]

TRADING_ENABLED = False
DEMO_EXECUTION_ENABLED = False
RISK_PER_TRADE = 0.01
MAXIMUM_DAILY_LOSS = 0.03
MAXIMUM_DRAWDOWN = None
MAXIMUM_SIMULTANEOUS_POSITIONS = 3


ACTIVOS = {
    "GOLD": {
        "ticker": "GC=F",
        "nombre": "Oro",
        "tipo": "commodity",
    },
    "EURUSD": {
        "ticker": "EURUSD=X",
        "nombre": "Euro / Dolar",
        "tipo": "forex",
    },
    "GBPUSD": {
        "ticker": "GBPUSD=X",
        "nombre": "Libra / Dolar",
        "tipo": "forex",
    },
    "USDJPY": {
        "ticker": "JPY=X",
        "nombre": "Dolar / Yen",
        "tipo": "forex",
    },
    "AUDUSD": {
        "ticker": "AUDUSD=X",
        "nombre": "Dolar australiano / Dolar",
        "tipo": "forex",
    },
    "USDCAD": {
        "ticker": "CAD=X",
        "nombre": "Dolar / Dolar canadiense",
        "tipo": "forex",
    },
    "USDCHF": {
        "ticker": "CHF=X",
        "nombre": "Dolar / Franco suizo",
        "tipo": "forex",
    },
    "NZDUSD": {
        "ticker": "NZDUSD=X",
        "nombre": "Dolar neozelandes / Dolar",
        "tipo": "forex",
    },
}
