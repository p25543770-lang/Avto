#!/usr/bin/env bash
# Запуск AI-режиссёра на Linux / macOS
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Создаю виртуальное окружение..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

exec python run.py --open "$@"
