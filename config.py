COSTO_POR_OPERACION = 0
PROB_THRESHOLD = 0.60
PROB_THRESHOLDS = [0.55, 0.60, 0.65]
COST_BPS = [0, 1, 2, 5]

TRADING_ENABLED = False
DEMO_EXECUTION_ENABLED = True

# Risk is configured in percent for readability and converted to ratios for
# existing modules/tests that already import the older names.
RISK_PER_TRADE_PERCENT = 1.0
MAX_DAILY_LOSS_PERCENT = 3.0
MAX_TOTAL_OPEN_RISK_PERCENT = 3.0
MAX_OPEN_POSITIONS = 3
MAX_POSITIONS_PER_SYMBOL = 1
# Maximo de posiciones netas en el mismo sentido por moneda (ej. 2 ventas de JPY).
MAX_POSITIONS_PER_CURRENCY = 2
# Maximo de operaciones abiertas por dia (hora Peru), sumando todas las estrategias demo.
MAX_TRADES_PER_DAY = 10

RISK_PER_TRADE = RISK_PER_TRADE_PERCENT / 100
MAXIMUM_DAILY_LOSS = MAX_DAILY_LOSS_PERCENT / 100
MAXIMUM_DRAWDOWN = None
MAXIMUM_SIMULTANEOUS_POSITIONS = MAX_OPEN_POSITIONS
DEMO_MAX_SIGNAL_AGE_MINUTES = 10
DEMO_MAX_VOLUME = 0.01

# Exit management modes for experiments. FIXED_TP preserves the current
# server-side SL/TP behavior; the other modes add protective SL updates.
EXIT_MODE = "FIXED_TP"
EXIT_MODES = ("FIXED_TP", "BREAK_EVEN_TP", "TRAILING")
BREAK_EVEN_TRIGGER_R = 1.0
BREAK_EVEN_OFFSET_POINTS = 0
TRAILING_START_R = 2.0
TRAILING_DISTANCE_R = 1.0
CLOSE_ON_STRONG_OPPOSITE_SIGNAL = False
STRONG_OPPOSITE_SIGNAL_THRESHOLD = 0.70

SHADOW_PAPER_TRADING = True
MAX_CURRENCY_EXPOSURE_PERCENT = 2.0

MAX_SPREAD_ATR_RATIO_BY_ASSET = {
    "EURUSD": 0.25,
    "GBPUSD": 0.25,
    "USDJPY": 0.25,
    "AUDUSD": 0.25,
    "USDCAD": 0.25,
    "USDCHF": 0.25,
    "NZDUSD": 0.25,
    "GOLD": 0.20,
    "XAUUSD": 0.20,
}

# Portafolio DEMO multi-activo (exploracion, solo cuenta demo; NO es un forward validado)
DEMO_PORTFOLIO_ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "GOLD",
    "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "EURAUD", "EURCAD", "AUDNZD", "AUDCAD", "GBPAUD",
    "SILVER", "BTCUSD",
]
DEMO_PORTFOLIO_THRESHOLD = 0.60
DEMO_PORTFOLIO_MAX_POSITIONS = 5
# Desfase horario valido del servidor MT5 vs UTC. XM: GMT+2 (invierno) / GMT+3 (verano).
MT5_SERVER_UTC_OFFSETS = (2, 3)


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
    # Activos del portafolio DEMO ampliado (exploracion, no validados para real)
    "EURJPY": {"ticker": "EURJPY=X", "nombre": "Euro / Yen", "tipo": "forex"},
    "GBPJPY": {"ticker": "GBPJPY=X", "nombre": "Libra / Yen", "tipo": "forex"},
    "AUDJPY": {"ticker": "AUDJPY=X", "nombre": "Dolar australiano / Yen", "tipo": "forex"},
    "CADJPY": {"ticker": "CADJPY=X", "nombre": "Dolar canadiense / Yen", "tipo": "forex"},
    "EURGBP": {"ticker": "EURGBP=X", "nombre": "Euro / Libra", "tipo": "forex"},
    "EURCHF": {"ticker": "EURCHF=X", "nombre": "Euro / Franco suizo", "tipo": "forex"},
    "EURAUD": {"ticker": "EURAUD=X", "nombre": "Euro / Dolar australiano", "tipo": "forex"},
    "EURCAD": {"ticker": "EURCAD=X", "nombre": "Euro / Dolar canadiense", "tipo": "forex"},
    "GBPCHF": {"ticker": "GBPCHF=X", "nombre": "Libra / Franco suizo", "tipo": "forex"},
    "AUDNZD": {"ticker": "AUDNZD=X", "nombre": "Dolar australiano / Dolar neozelandes", "tipo": "forex"},
    "AUDCAD": {"ticker": "AUDCAD=X", "nombre": "Dolar australiano / Dolar canadiense", "tipo": "forex"},
    "GBPAUD": {"ticker": "GBPAUD=X", "nombre": "Libra / Dolar australiano", "tipo": "forex"},
    "SILVER": {"ticker": "SI=F", "nombre": "Plata", "tipo": "commodity"},
    "BTCUSD": {"ticker": "BTC-USD", "nombre": "Bitcoin / Dolar", "tipo": "crypto"},
}

NEW_PORTFOLIO_ASSETS = [
    "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
    "EURGBP", "EURCHF", "EURAUD", "EURCAD",
    "GBPCHF", "AUDNZD", "AUDCAD", "GBPAUD",
    "SILVER", "BTCUSD",
]

# Regla de activacion para los activos nuevos (holdout 20% final, threshold 0.60, SL 1 ATR / TP 2 ATR):
# una direccion se desactiva si tuvo >= 10 senales en holdout y win rate < 40%
# (con TP = 2R hace falta ~40% para cubrir costos). EURGBP, EURCHF y GBPCHF quedan
# fuera porque ambas direcciones fallan la regla. Los 8 activos originales no cambian.
DEMO_PORTFOLIO_DISABLED_DIRECTIONS = {
    "EURJPY": ("LONG",),   # 66 senales, 39.4%
    "GBPJPY": ("LONG",),   # 382 senales, 36.4%
    "CADJPY": ("LONG",),   # 370 senales, 33.2%
    "EURCAD": ("SHORT",),  # 25 senales, 36.0%
    "AUDNZD": ("LONG",),   # 232 senales, 20.3%
}
