# Deployment en Windows VPS 24/7

Esta guia prepara FOREX ML para ejecutarse en un Windows VPS. El proyecto sigue en modo educativo y PAPER: no conecta MetaTrader, no envia ordenes reales y no usa dinero real.

## 1. Instalar Python

Instala Python 3.10 o superior desde:

```text
https://www.python.org/downloads/windows/
```

Durante la instalacion marca:

```text
Add python.exe to PATH
```

Verifica:

```bat
python --version
```

## 2. Crear carpeta del proyecto

En el VPS crea:

```text
D:\PROYECTOS\FOREX ML
```

Copia el proyecto dentro de esa carpeta.

## 3. No copiar .venv

La carpeta:

```text
.venv
```

NO debe copiarse al VPS. Debe crearse nuevamente en el servidor para evitar problemas de rutas, librerias compiladas y version de Python.

## 4. Archivos que si deben migrar

Conserva estos archivos y carpetas:

```text
models/paper/
paper/start.json
paper/state.json
paper/trades.csv
paper/signals.csv
paper/equity.csv
config.py
config_paper.py
main.py
src/
requirements.txt
run_paper.bat
backup_paper.bat
AUTOMATION.md
DEPLOYMENT.md
```

Tambien migra cualquier otro codigo del proyecto que sea necesario para ejecutar `main.py`.

Muy importante:

```text
paper/start.json
```

debe conservar EXACTAMENTE la fecha original del forward test. No crees un nuevo forward test al mover al VPS.

## 5. Crear .venv en el VPS

Desde:

```bat
cd /d "D:\PROYECTOS\FOREX ML"
```

crea el entorno:

```bat
python -m venv .venv
```

## 6. Activar .venv

```bat
.venv\Scripts\activate
```

## 7. Instalar dependencias

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 8. Crear .env manualmente

No subas `.env` a GitHub y no lo copies en respaldos publicos.

Crea manualmente en el VPS:

```text
D:\PROYECTOS\FOREX ML\.env
```

con:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

Nunca imprimas estas variables en consola o logs.

## 9. Configurar Telegram

1. En Telegram abre `@BotFather`.
2. Crea un bot con `/newbot`.
3. Copia el token en `TELEGRAM_BOT_TOKEN`.
4. Escribe `/start` a tu bot.
5. Consulta:

```text
https://api.telegram.org/botTU_TOKEN/getUpdates
```

6. Copia `chat.id` en `TELEGRAM_CHAT_ID`.

## 10. Ejecutar PAPER STATUS

```bat
.venv\Scripts\python.exe main.py PAPER STATUS
```

Debe mostrar el forward test existente. No debe crear un nuevo `paper/start.json`.

## 11. Ejecutar PAPER

```bat
.venv\Scripts\python.exe main.py PAPER
```

Esto sigue siendo PAPER TRADING: no envia ordenes reales.

## 12. Ejecutar TELEGRAM TEST

```bat
.venv\Scripts\python.exe main.py TELEGRAM TEST
```

Si Telegram esta configurado, recibiras el mensaje de prueba.

## 13. Ejecutar HEALTH

```bat
.venv\Scripts\python.exe main.py HEALTH
```

Debe revisar Python, librerias, modelo paper, metadata, archivos paper, logs, Telegram y fuente de datos sin modificar estado.

## 14. Configurar ejecucion automatica cada hora

Usa el Programador de tareas de Windows:

1. Abrir **Task Scheduler** / **Programador de tareas**.
2. Crear una tarea llamada `FOREX ML Paper Trading`.
3. Trigger: repetir cada 1 hora.
4. Recomendado: ejecutar en `XX:05`, unos minutos despues del cierre de vela H1.
5. Action: `Start a program`.
6. Program/script:

```text
D:\PROYECTOS\FOREX ML\run_paper.bat
```

7. Start in:

```text
D:\PROYECTOS\FOREX ML
```

`run_paper.bat` usa `.venv\Scripts\python.exe` directamente, asi que no depende de activar el entorno previamente.

## 15. Logs y rotacion

El runner escribe en:

```text
logs/paper_runner.log
```

Si el archivo supera 20 MB, `run_paper.bat` lo renombra con timestamp y empieza un log nuevo. No borra logs historicos automaticamente.

## 16. Backups

Antes de cambios importantes o de mover el servidor, ejecuta:

```bat
D:\PROYECTOS\FOREX ML\backup_paper.bat
```

Esto crea una carpeta timestamped dentro de:

```text
backups/
```

No copia `.env`.
