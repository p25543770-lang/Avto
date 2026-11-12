"""Заглушка локального LLM для тестов (OpenAI-совместимый /v1/chat/completions).

Используется в автотестах, чтобы проверить весь пайплайн без настоящей модели.
Запуск:  python tests/stub_llm.py [port]
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TIME_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+(?:\.\d+)?)\]")

RESPONSE = {
    "keep_segments": [{"start": 0.0, "end": 600.0}],
    "chapters": [{"time": "00:00", "title": "Вступление"}],
    "moments": [
        {
            "start": 2.0,
            "end": 40.0,
            "title": "Главная мысль",
            "hook": "Самое интересное из ролика",
            "score": 9,
        }
    ],
}


def _build_response(user_message: str) -> dict:
    starts, ends = [], []
    for match in TIME_RE.finditer(user_message or ""):
        start = int(match.group(1)) * 60 + float(match.group(2))
        end = int(match.group(3)) * 60 + float(match.group(4))
        starts.append(start)
        ends.append(end)
    payload = json.loads(json.dumps(RESPONSE))
    if starts and ends:
        payload["keep_segments"] = [{"start": min(starts), "end": max(ends)}]
        payload["chapters"] = [{"time": "00:00", "title": "Вступление"}]
    if "shorts" in (user_message or ""):
        payload = {
            "shorts": [
                {
                    "start": min(starts) if starts else 0.0,
                    "end": (min(starts) if starts else 0.0) + 35.0,
                    "title": "Яркий фрагмент",
                    "hook": "Коротко о главном",
                    "score": 9,
                }
            ]
        }
    return {
        "id": "stub",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```",
                },
                "finish_reason": "stop",
            }
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # noqa: D102
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": "stub-model", "object": "model"}]})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            request = {}
        messages = request.get("messages") or []
        user_message = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
        self._send(_build_response(user_message))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"stub LLM on http://127.0.0.1:{port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
