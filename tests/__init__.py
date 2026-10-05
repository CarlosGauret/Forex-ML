import os

os.environ.setdefault("FOREX_ML_TESTING", "True")

# Los tests parten siempre de DEMO_EXECUTION_ENABLED = False, aunque config.py
# lo tenga activo para operar la demo en automatico. Se fija antes de que los
# modulos de src hagan "from config import DEMO_EXECUTION_ENABLED".
# TRADING_ENABLED (dinero real) NO se toca: los tests deben leer el valor real.
import config  # noqa: E402

config.DEMO_EXECUTION_ENABLED = False
