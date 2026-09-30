# Automatizacion local de forward test PAPER

Este proyecto mantiene desactivada la ejecucion real:

```text
TRADING_ENABLED = False
DEMO_EXECUTION_ENABLED = False
```

El runner automatico reutiliza:

```text
D:\PROYECTOS\FOREX ML\run_paper.bat
```

## Orden de ejecucion

Cada corrida programada ejecuta, en este orden:

```bat
D:\PROYECTOS\FOREX ML\.venv\Scripts\python.exe D:\PROYECTOS\FOREX ML\main.py PAPER
D:\PROYECTOS\FOREX ML\.venv\Scripts\python.exe D:\PROYECTOS\FOREX ML\main.py PAPER EURUSD
```

`PAPER` corresponde al forward paper de GOLD con sus configs actuales T060 y T055.
`PAPER EURUSD` corresponde al forward paper EURUSD con datos MT5 XM.

## Fallos independientes

El runner intenta siempre ambos bloques.

- Si GOLD falla, igualmente intenta EURUSD.
- Si EURUSD falla porque MT5 no esta abierto o no hay datos validos, GOLD no se invalida.
- El exit code individual queda registrado como `GOLD EXIT CODE` y `EURUSD EXIT CODE`.
- El exit code global del runner es `1` si alguno de los dos bloques falla.

EURUSD falla cerrado: no usa fuente alternativa si MT5/XM no esta disponible.

## Logs

Resumen general:

```text
D:\PROYECTOS\FOREX ML\logs\paper_runner.log
```

GOLD:

```text
D:\PROYECTOS\FOREX ML\logs\paper_gold_runner.log
```

EURUSD:

```text
D:\PROYECTOS\FOREX ML\logs\paper_eurusd_runner.log
```

Cada log incluye hora de inicio, comando, salida stdout/stderr, hora de fin y exit code.

## Lock

El runner usa:

```text
D:\PROYECTOS\FOREX ML\paper\paper.lock
```

Si una corrida previa sigue activa, la siguiente no inicia otra instancia concurrente.

## Programador de tareas de Windows

Tarea detectada:

```text
FOREX ML - PAPER
```

Accion:

```text
C:\Windows\System32\cmd.exe /c ""D:\PROYECTOS\FOREX ML\run_paper.bat""
```

Directorio de inicio:

```text
D:\PROYECTOS\FOREX ML
```

Configuracion esperada:

- Repeticion: cada 1 hora.
- Minuto recomendado: `05`, unos minutos despues del cierre H1.
- Instancias multiples: no iniciar una nueva si ya hay una corriendo.
- Ejecutar lo antes posible si se omitio un inicio programado.
- Wake timers: no habilitados.

La automatizacion solo corre mientras Windows esta encendido. Si la laptop estuvo apagada, GOLD y EURUSD recuperan pendientes con el catch-up al volver a ejecutar el runner.

## Comandos utiles

Probar manualmente:

```bat
D:\PROYECTOS\FOREX ML\run_paper.bat
```

Ver estado GOLD:

```bat
D:\PROYECTOS\FOREX ML\.venv\Scripts\python.exe D:\PROYECTOS\FOREX ML\main.py PAPER STATUS
```

Ver estado EURUSD:

```bat
D:\PROYECTOS\FOREX ML\.venv\Scripts\python.exe D:\PROYECTOS\FOREX ML\main.py PAPER EURUSD STATUS
```

## Ejecucion automatica en MT5 DEMO (LIVE)

`run_paper.bat` ahora ejecuta `main.py LIVE GOLD` y `main.py LIVE EURUSD` en lugar de
`main.py PAPER` y `main.py PAPER EURUSD`. Cada `LIVE` primero procesa su PAPER exactamente
igual que antes (no cambia modelo, threshold, SL/TP ni max hold) y despues pasa la senal
al ejecutor MT5.

GOLD:

- Solo `GOLD_LONG_LOGISTIC_BASE_PLUS_ML_T060` envia ordenes; T055 sigue solo en PAPER
  (ambas operan el mismo simbolo y el bot mantiene una posicion por simbolo).
- La senal sale del PAPER congelado con datos GC=F (yfinance); la orden se ejecuta en el
  oro de MT5 (XAUUSD) con precio real del broker y el ATR de la senal.
- Si MT5 no esta abierto, el PAPER GOLD igual se procesa; solo falla el ejecutor.

Modo segun `config.py`:

| DEMO_EXECUTION_ENABLED | Cuenta MT5 | Resultado |
|---|---|---|
| False (defecto) | DEMO | DRY RUN: calcula volumen/SL/TP y lo registra, NO envia |
| True | DEMO | Envia la orden a la cuenta demo con SL/TP en el servidor |
| cualquiera | REAL | Bloqueado siempre |

Protecciones del ejecutor (`src/live_executor.py`):

- SL y TP viajan con la orden: protegen aunque el VPS se apague.
- Magic number `20260930`: el bot ignora posiciones abiertas a mano.
- Una senal nunca se envia dos veces (journal `live/orders.csv` + comentario en MT5).
- Solo se ejecuta la senal de la ultima vela cerrada. Si los datos llegan atrasados, la
  senal no se ejecuta y queda en el journal como `SKIP STALE_SIGNAL` (una vez).
- Limite de perdida diaria (3%, realizada + flotante, ultimas 24h) y maximo de posiciones.
- No entra si el spread supera el 25% del ATR.
- Reintenta ante requote / cambio de precio (3 intentos).
- Cierra por tiempo las posiciones que superan `MAX_HOLD_BARS`, contando velas H1 reales
  de MT5 (el fin de semana no suma velas, igual que el PAPER).

Comandos:

```bat
.venv\Scripts\python.exe main.py LIVE EURUSD          :: PAPER + ejecutor (modo segun config)
.venv\Scripts\python.exe main.py LIVE EURUSD DRYRUN   :: fuerza DRY RUN
.venv\Scripts\python.exe main.py LIVE GOLD             :: igual para GOLD T060
.venv\Scripts\python.exe main.py LIVE RECONCILE       :: compara journal vs MT5
.venv\Scripts\python.exe main.py LIVE STOP            :: kill switch: bloquea aperturas nuevas
.venv\Scripts\python.exe main.py LIVE RESUME          :: quita el kill switch
```

## Plan de prueba

1. **Fase 1 - tests** (cualquier PC): `python -m pytest tests`.
2. **Fase 2 - DRY RUN en VPS** (1-3 dias, `DEMO_EXECUTION_ENABLED = False`):
   comparar `live/orders.csv` (filas `DRY_RUN`) con `paper/eurusd/signals.csv` y
   `paper/signals.csv` (GOLD T060). Cada senal LONG del PAPER debe tener su fila con
   volumen, SL y TP coherentes. Contar las filas `STALE_SIGNAL`: si GOLD pierde muchas
   senales por atraso de yfinance, hay que resolverlo antes de la Fase 3.
3. **Fase 3 - DEMO real** (4-8 semanas, `DEMO_EXECUTION_ENABLED = True`):
   correr `LIVE RECONCILE` a diario; debe dar `OK`. Comparar precio de entrada
   MT5 vs PAPER (diferencia esperada: spread + 1-2 pips).
4. **Fase 4 - cuenta real**: solo con 30+ trades demo, expectancy neta positiva y
   cero errores de ejecucion. Requiere un cambio de codigo explicito: hoy REAL esta bloqueado.

## Portafolio DEMO multi-activo (compra y venta)

`run_paper.bat` tambien ejecuta `main.py LIVE PORTFOLIO` cada hora: 8 activos
(EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, GOLD), LONG y SHORT, modelo
LOGISTIC threshold 0.60, SL 1 ATR / TP 2 ATR / max 24 velas, maximo 5 posiciones.

Es EXPLORACION en demo, separada de los forward congelados. Modelos en `models/demo/`
(con metadata y hash); se reentrenan con `main.py LIVE PORTFOLIO TRAIN`.

`run_paper.bat` y `backup_paper.bat` ya no dependen de `D:\PROYECTOS\FOREX ML`: usan la
carpeta donde esten.

Guia de la primera prueba: `PRUEBA_DEMO.md`.
