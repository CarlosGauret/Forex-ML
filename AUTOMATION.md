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
