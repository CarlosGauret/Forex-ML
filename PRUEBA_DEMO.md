# Prueba en cuenta DEMO - paso a paso (Windows)

Objetivo: ver al bot abrir y cerrar compras y ventas solo, en la cuenta demo de XM.
Todo corre en cuenta DEMO. Una cuenta REAL esta bloqueada por codigo.

Tiempo estimado: 45-60 minutos la primera vez.

---

## 0. Antes de empezar

- MT5 abierto y conectado a tu **cuenta DEMO** de XM.
- En MT5, el boton **Algo Trading** de la barra superior debe estar en verde.
  Si esta en rojo, MT5 rechaza todas las ordenes del bot.
- Git instalado (https://git-scm.com/download/win), o descarga el ZIP desde GitHub.

---

## 1. Pausar la tarea automatica actual

Programador de tareas -> `FOREX ML - PAPER` -> clic derecho -> **Deshabilitar**.

Asi nada corre mientras cambiamos de version.

---

## 2. Guardar la carpeta actual

Renombra la carpeta actual:

```text
D:\PROYECTOS\FOREX ML   ->   D:\PROYECTOS\FOREX ML VIEJO
```

No la borres: ahi esta el estado del PAPER mas nuevo (el de GitHub es una copia vieja).

---

## 3. Descargar la version nueva

Abre `cmd` y ejecuta:

```bat
cd /d D:\PROYECTOS
git clone https://github.com/CarlosGauret/Forex-ML "FOREX ML"
```

(Si usas ZIP: descomprimelo y renombra la carpeta a `D:\PROYECTOS\FOREX ML`.)

---

## 4. Traer tus datos desde la carpeta vieja

Copia desde `FOREX ML VIEJO` hacia `FOREX ML`, **reemplazando** si pregunta:

| Copiar | Por que |
|---|---|
| carpeta `paper\` completa | estado real de los forward GOLD y EURUSD |
| carpeta `logs\` completa | historial y avisos de Telegram ya enviados |
| archivo `.env` | tus claves de Telegram y MT5 |

Si el `.env` viejo no tiene `MT5_LOGIN`, `MT5_PASSWORD` y `MT5_SERVER`, agregalos
(ver `.env.example`).

---

## 5. Crear el entorno de Python

```bat
cd /d "D:\PROYECTOS\FOREX ML"
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verificar que todo esta bien (deben pasar todos los tests):

```bat
.venv\Scripts\python.exe -m pytest tests -q
```

---

## 6. Prueba sin enviar ordenes (DRY RUN)

```bat
.venv\Scripts\python.exe main.py MT5 CHECK
.venv\Scripts\python.exe main.py LIVE PORTFOLIO DRYRUN
.venv\Scripts\python.exe main.py LIVE TEST EURUSD BUY
```

Que esperar:

- `MT5 CHECK`: conexion OK y cuenta **DEMO**.
- `LIVE PORTFOLIO DRYRUN`: una tabla con los 8 activos, su probabilidad de compra
  (P LONG) y de venta (P SHORT), y la decision (LONG / SHORT / WAIT).
  Si todos dicen `NO_DATA`, el mercado esta cerrado (fin de semana) o MT5 no esta
  conectado.
- `LIVE TEST EURUSD BUY`: `Estado: DRY_RUN` con volumen, precio, SL y TP.

---

## 7. Activar la ejecucion DEMO

Abre `config.py` y cambia:

```python
DEMO_EXECUTION_ENABLED = True
```

No toques `TRADING_ENABLED`: la cuenta real sigue bloqueada igual.

---

## 8. Prueba real en demo: abrir y cerrar

```bat
.venv\Scripts\python.exe main.py LIVE TEST EURUSD BUY
.venv\Scripts\python.exe main.py LIVE TEST GBPUSD SELL
```

Debes ver:

- `Estado: SENT` y un numero de ticket.
- En MT5 (pestana **Operaciones**): una COMPRA de EURUSD y una VENTA de GBPUSD,
  cada una con su SL y TP.
- En Telegram: "ORDEN DEMO ENVIADA" por cada una.

Revisar que el registro coincide con MT5:

```bat
.venv\Scripts\python.exe main.py LIVE RECONCILE
```

Debe decir `RESULTADO: OK`.

Cerrar las dos:

```bat
.venv\Scripts\python.exe main.py LIVE CLOSEALL
```

Debes ver las posiciones cerradas en MT5 y el aviso en Telegram.

---

## 9. Dejarlo operando solo

1. Programador de tareas -> `FOREX ML - PAPER` -> **Habilitar**.
   La ruta no cambio (`D:\PROYECTOS\FOREX ML\run_paper.bat`), no hay que editar nada.
2. Cada hora, en el minuto 05, el bot:
   - actualiza los PAPER de GOLD y EURUSD;
   - revisa los 8 activos del portafolio y abre COMPRAS o VENTAS si hay senal;
   - cierra posiciones que cumplieron 24 velas;
   - avisa por Telegram cada apertura y cada cierre (incluidos SL y TP del broker).
3. MT5 debe quedar abierto con Algo Trading en verde y la PC encendida.

Con el threshold 0.60 el portafolio da pocas senales (en la prueba historica ~3-4 por
dia entre todos los activos, sobre todo USDJPY y GOLD). Puede pasar varias horas sin
abrir nada: es normal.

---

## Comandos utiles

| Comando | Que hace |
|---|---|
| `main.py LIVE STOP` | Boton de panico: no abre nada nuevo (sigue cerrando) |
| `main.py LIVE RESUME` | Quita el boton de panico |
| `main.py LIVE CLOSEALL` | Cierra todas las posiciones del bot |
| `main.py LIVE RECONCILE` | Compara el registro del bot con MT5 |
| `main.py LIVE PORTFOLIO DRYRUN` | Ver probabilidades actuales sin operar |

Registros:

- `live\orders.csv`: cada orden enviada, simulada, bloqueada o cerrada, con el motivo.
- `live\portfolio_signals.csv`: probabilidades de cada activo, cada hora.
- `logs\live_portfolio_runner.log`: salida de cada corrida automatica.

---

## Si algo falla

| Mensaje | Solucion |
|---|---|
| `ALGO TRADING DESACTIVADO EN MT5` | Activa el boton Algo Trading en MT5 |
| `MT5_ACCOUNT_NOT_DEMO_BLOCKED` | Estas en una cuenta real: cambia a la demo |
| `DEMO_EXECUTION_DISABLED` | Falta el paso 7 |
| `MINIMUM_VOLUME_EXCEEDS_RISK` | Saldo demo muy bajo para arriesgar 1%: usa una demo de USD 10,000 |
| `SPREAD_TOO_WIDE` | Spread alto (noticias, rollover): el bot espera, es normal |
| Todos `NO_DATA` | Mercado cerrado o MT5 desconectado |

---

## Importante

El portafolio es de **exploracion en demo**. Sus modelos aciertan entre ~31% y 41% en
el historico, y hace falta ~37% solo para no perder con TP 2 / SL 1 y costos. No hay
evidencia de que gane dinero. Esta prueba valida que el bot **opera solo y de forma
segura**, no que sea rentable.
