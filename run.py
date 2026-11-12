#!/usr/bin/env python3
"""Точка входа: запускает веб-приложение «AI-режиссёр».

    python run.py            # http://127.0.0.1:8000
    python run.py --port 9000
"""

from __future__ import annotations

import argparse
import webbrowser
from threading import Timer


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-режиссёр: авто-монтаж видео")
    parser.add_argument("--host", default="127.0.0.1", help="адрес прослушивания")
    parser.add_argument("--port", type=int, default=8000, help="порт")
    parser.add_argument("--open", action="store_true", help="открыть браузер")
    args = parser.parse_args()

    import uvicorn

    if args.open:
        Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    print(f"AI-режиссёр: http://{args.host}:{args.port}")
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
