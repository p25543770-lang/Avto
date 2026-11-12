@echo off
REM Запуск AI-режиссёра на Windows (активируйте venv заранее)
if not exist .venv (
    echo Создаю виртуальное окружение...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
start "" http://127.0.0.1:8000
python run.py --open
