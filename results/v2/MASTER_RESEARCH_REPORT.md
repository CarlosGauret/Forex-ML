# MASTER RESEARCH REPORT - FOREX ML

Fecha de consolidacion: 2026-09-30

Alcance: cierre de la primera ronda de investigacion multi-asset usando exclusivamente artefactos existentes en `results/`, `results/v2/`, `models/paper/` y `paper/`.

No se entrenaron modelos, no se recalcularon estrategias, no se cambiaron thresholds, features, targets, PAPER ni forward tests.

## 1. Tabla Maestra

| Activo | Estado | LONG | SHORT | Mejor configuracion observada | Threshold | Trades | PF | Expectancy R | Folds + | CI95 | Coste | Clasificacion final | Forward | Modelo congelado | Paper activo | Broker-aware USD 100 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|---|---|---|---|
| GOLD | Investigacion V2 cerrada; PAPER activo | SI | SI | GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060 ATR_ONLY | 0.60 | 44 comparison / 20 WF | 1.147289 | 0.106817 comparison / 0.477890 WF | 3 | [-0.043476, 1.154022] | 2 bps | PAPER ACTIVO / MIXTO | SI | SI | SI | 0/44 T060; 0/72 T055 |
| EURUSD | Investigacion cerrada; PAPER forward activo | SI | SI | LONG RF_ACTUAL ML_ONLY | 0.65 | 40 | 1.097721 | 0.084322 | 3 | [-0.309373, 0.548387] | 2 bps | CANDIDATO / FORWARD ACTIVO | SI | SI | SI | 24/100 (24.00%) |
| GBPUSD | Investigacion cerrada | SI | SI | SHORT RF_SIMPLE BASE_PLUS_ML | 0.65 | 7 | 5.174399 | 0.971006 | 1 | [0.106174, 1.418095] | 2 bps | SIN CANDIDATO: MUESTRA INSUFICIENTE | NO | NO | NO | 699/3926 (17.80%) |
| USDJPY | Investigacion cerrada; broker-aware auditado | SI | SI | SHORT LOGISTIC BASE_PLUS_ML | 0.60 | 15 | 2.227552 | 0.588685 | 1 | [-0.239098, 1.210505] | 2 bps | SIN CANDIDATO: MUESTRA INSUFICIENTE | NO | NO | NO | 409/3684 (11.10%) |
| AUDUSD | Investigacion cerrada | SI | SI | LONG LOGISTIC BASE_PLUS_ML | 0.65 | 3 | 3.395758 | 0.908774 | 1 | [-1.135067, 1.935261] | 2 bps | SIN CANDIDATO: MUESTRA INSUFICIENTE | NO | NO | NO | 2204/4116 (53.55%) |
| USDCAD | Investigacion cerrada | SI | SI | SHORT RF_SIMPLE BASE_PLUS_ML | 0.60 | 1 | inf | 1.854225 | 1 | [1.854225, 1.854225] | 2 bps | SIN CANDIDATO: MUESTRA INSUFICIENTE | NO | NO | NO | 708/3690 (19.19%) |
| USDCHF | Investigacion cerrada | SI | SI | SHORT LOGISTIC ML_ONLY | 0.65 | 75 | 1.156314 | 0.118863 | 1 | [-0.209170, 0.454296] | 2 bps | SIN CANDIDATO: MIXTO | NO | NO | NO | 102/4032 (2.53%) |
| NZDUSD | Investigacion cerrada | SI | SI | LONG LOGISTIC BASE_PLUS_ML | 0.65 | 11 | 7.434963 | 1.340168 | 1 | [0.520685, 1.893333] | 2 bps | SIN CANDIDATO: MUESTRA INSUFICIENTE | NO | NO | NO | 2871/4336 (66.21%) |

Notas de lectura:

- `CALIDAD ESTADISTICA` y `EJECUTABILIDAD CON USD 100` son dimensiones separadas.
- Que una configuracion sea ejecutable con USD 100 no la convierte en candidata.
- Que una configuracion tenga PF o expectancy positivos no la convierte en candidata si falla por muestra, folds o CI95.
- En GOLD, PF/trades provienen de `gold_sl_tp_comparison.csv`; folds/CI95 provienen de `walkforward_gold_summary.csv`.

## 2. Forward Tests Activos

### GOLD

Configuraciones activas:

- `GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T055`
- `GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060`

Estado PAPER existente:

| Campo | Valor |
|---|---|
| Fecha inicio | 2026-09-27T17:16:28+00:00 |
| Trades cerrados | 0 |
| Wins | 0 |
| Losses | 0 |
| Capital paper T055 | 1000 |
| Capital paper T060 | 1000 |
| Ultima barra T055 | 2026-09-29T23:00:00 |
| Ultima barra T060 | 2026-09-29T23:00:00 |

### EURUSD

Configuracion activa:

- Direccion: LONG
- Modelo: RF_ACTUAL
- Sistema: ML_ONLY
- Threshold: 0.65
- Config ID: `EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065`

Estado PAPER existente:

| Campo | Valor |
|---|---|
| Fecha inicio | 2026-09-29T12:54:32+00:00 |
| Hash modelo | b7d2136797e433bb3f2a59170b77569570c60c58f0cf7f75ba19f4ae17aa8a6f |
| Trades cerrados | 0 |
| Wins | 0 |
| Losses | 0 |
| Capital paper | 100.0 |
| Ultima barra procesada | 2026-09-30T00:00:00 |
| Ultima senal | WAIT |
| Ultima razon | PROBABILITY_BELOW_THRESHOLD |

## 3. Activos Sin Candidato

Sin candidato a forward test nuevo en esta consolidacion:

- GBPUSD
- USDJPY
- AUDUSD
- USDCAD
- USDCHF
- NZDUSD

Motivos dominantes: muestra insuficiente, pocos folds positivos, CI95 cruzando cero, PF/expectancy insuficientes al coste principal de 2 bps, o ejecutabilidad USD 100 insuficiente. La ejecutabilidad se documenta separada de la calidad estadistica.

## 4. PAPER GOLD Status

PAPER GOLD esta activo con T055 y T060.

- trades cerrados: 0
- signals: 96
- capital por configuracion: 1000
- ultima barra procesada T055/T060: 2026-09-29T23:00:00
- pending signals: ninguno

## 5. PAPER EURUSD Status

PAPER EURUSD esta activo en LONG RF_ACTUAL ML_ONLY threshold 0.65.

- trades cerrados: 0
- signals: 12
- capital paper: 100.0
- ultima barra procesada: 2026-09-30T00:00:00
- pending signal: null
- data source: MT5 XM
- data status: LIVE
- ultima ejecucion: WAIT

## 6. Risk Manager

- Se detecto y corrigio bug broker-aware de USDJPY.
- `ORDER_TYPE_BUY = 0` esta soportado.
- `tick_size` y `tick_value` estan soportados.
- Con MT5 disponible se prefiere `mt5.order_calc_profit`.
- GOLD, EURUSD, GBPUSD y USDJPY fueron auditados.
- La regresion multi-asset del Risk Manager paso.
- No se modifico `src/risk.py` durante esta consolidacion.

## 7. Broker-Aware USD 100

| Activo | Ejecutables USD 100 | Lectura |
|---|---:|---|
| GOLD | 0/44 T060; 0/72 T055 | Paper activo, pero broker-aware V2 historico con USD 100 no ejecuta esos backtests |
| EURUSD | 24/100 (24.00%) | Ejecutabilidad parcial; calidad estadistica y forward ya estaban autorizados |
| GBPUSD | 699/3926 (17.80%) | Ejecutabilidad parcial; sin candidato por estadistica |
| USDJPY | 409/3684 (11.10%) | Ejecutabilidad parcial con USD 100 tras auditoria broker-aware corregida; no cambia la clasificacion estadistica |
| AUDUSD | 2204/4116 (53.55%) | Ejecutable parcial; sin candidato por estadistica |
| USDCAD | 708/3690 (19.19%) | Ejecutabilidad parcial; sin candidato por estadistica |
| USDCHF | 102/4032 (2.53%) | Ejecutabilidad muy baja; sin candidato por estadistica |
| NZDUSD | 2871/4336 (66.21%) | Ejecutable parcial; sin candidato por estadistica |

## 8. Seguridad

- `TRADING_ENABLED = False`
- `DEMO_EXECUTION_ENABLED = False`
- ordenes enviadas durante esta consolidacion: 0
- no se crearon nuevos forward tests
- no se entrenaron modelos
- no se modificaron thresholds, features, targets, SL/TP, max hold ni costes principales
- no se modifico PAPER

## 9. Archivos Maestros

- `results/v2/master_asset_summary.csv`
- `results/v2/MASTER_RESEARCH_REPORT.md`
- `results/v2/EXPERIMENT_LOCK.md`
