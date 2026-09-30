# EXPERIMENT LOCK

Fecha de consolidacion: 2026-09-30

Este archivo bloquea la primera ronda de investigacion multi-asset de FOREX ML a partir de la consolidacion actual.

## Alcance Bloqueado

Mientras los forward tests de GOLD y EURUSD esten activos, no cambiar:

- modelos congelados;
- thresholds;
- features;
- targets;
- SL/TP;
- max hold;
- costes principales;
- reglas de clasificacion.

Los resultados futuros de estos mismos forward tests no deben utilizarse para reoptimizar los experimentos activos.

## GOLD PAPER Activo

Activo: GOLD

Direccion: LONG

Modelo congelado: LOGISTIC

Sistema: BASE_PLUS_ML

Thresholds activos:

- T060 principal
- T055 secundario

SL/TP:

- SL: 1 ATR
- TP: 2 ATR
- Max hold: 24 barras

Forward test iniciado: 2026-09-27T17:16:28+00:00

Estado observado al consolidar:

- trades cerrados: 0
- capital paper T055: 1000
- capital paper T060: 1000
- ultima barra procesada T055: 2026-09-29T23:00:00
- ultima barra procesada T060: 2026-09-29T23:00:00

Decision estadistica existente:

- T060 ATR_ONLY, coste 2 bps: MIXTO
- trades WF: 20
- mean expectancy WF: 0.477890
- positive folds: 3/5
- CI95: [-0.043476, 1.154022]

## EURUSD PAPER Forward Activo

Activo: EURUSD

Direccion: LONG

Modelo congelado: RF_ACTUAL

Sistema: ML_ONLY

Threshold activo: 0.65

Config ID: `EURUSD_LONG_RF_ACTUAL_ML_ONLY_T065`

Hash modelo:

`b7d2136797e433bb3f2a59170b77569570c60c58f0cf7f75ba19f4ae17aa8a6f`

SL/TP:

- SL: 1 ATR
- TP: 2 ATR
- Max hold: 24 barras

Forward test iniciado: 2026-09-29T12:54:32+00:00

Estado observado al consolidar:

- trades cerrados: 0
- capital paper: 100.0
- ultima barra procesada: 2026-09-30T00:00:00
- ultima senal: WAIT
- ultima razon: PROBABILITY_BELOW_THRESHOLD

Decision estadistica existente:

- coste principal: 2 bps
- trades OOF: 40
- Profit Factor: 1.097721
- expectancy R: 0.084322
- folds positivos: 3
- CI95: [-0.309373, 0.548387]
- clasificacion en CSV existente: CANDIDATO A FORWARD TEST

## Activos Sin Nuevo Forward

No crear forward tests para:

- GBPUSD
- USDJPY
- AUDUSD
- USDCAD
- USDCHF
- NZDUSD

Sin autorizacion posterior explicita, estos activos permanecen sin candidato operativo.

## Reglas De Seguridad

- `TRADING_ENABLED = False`
- `DEMO_EXECUTION_ENABLED = False`
- no enviar ordenes;
- no usar dinero real;
- no modificar PAPER activo;
- no modificar modelos congelados;
- no recalcular ni reoptimizar la primera ronda para justificar cambios retrospectivos.
