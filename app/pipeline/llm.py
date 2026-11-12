"""Клиент локального OpenAI-совместимого сервера (LM Studio, Ollama, llama.cpp).

Сервер ожидается на ``http://127.0.0.1:1234/v1`` (LM Studio) или
``http://127.0.0.1:11434/v1`` (Ollama). Модель полностью локальная — данные
наружу не уходят.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Ошибка общения с локальным LLM."""


@dataclass
class LLMClient:
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = ""
    model: str = "local-model"
    timeout: float = 240.0
    max_tokens: int = 4096
    retries: int = 2

    # ------------------------------------------------------------------
    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1") and "/v1" not in base:
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------------
    def ping(self) -> tuple[bool, str]:
        """Проверяет доступность сервера. Возвращает (ok, описание)."""
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1") and "/v1" not in base:
            base = f"{base}/v1"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base}/models", headers=self._headers())
            if resp.status_code >= 400:
                return False, f"HTTP {resp.status_code}"
            data = resp.json()
            models = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
            if models:
                return True, "доступны модели: " + ", ".join(models[:5])
            return True, "сервер отвечает"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    # ------------------------------------------------------------------
    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        """Отправляет запрос и возвращает разобранный JSON-объект."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.15,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        # response_format поддерживают не все серверы — пробуем, при ошибке идём без него
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self._post(payload)
                text = self._extract_text(data)
                return parse_json_object(text)
            except LLMError as exc:
                last_error = exc
                if "json_object" in str(exc) and "response_format" in payload:
                    payload.pop("response_format", None)
                    continue
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        raise LLMError(f"LLM недоступен: {last_error}")

    # ------------------------------------------------------------------
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.endpoint, headers=self._headers(), json=payload
                )
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise LLMError("Ответ не является JSON") from exc

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Пустой ответ модели")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text")
        if isinstance(content, list):  # некоторые серверы отдают массив частей
            content = "".join(part.get("text", "") for part in content)
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Модель вернула пустой ответ")
        return content


# ---------------------------------------------------------------------------
# Разбор JSON
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_text(raw: str) -> str:
    """Достаёт JSON-объект из ответа модели (в т.ч. из markdown-блока)."""
    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def parse_json_object(raw: str) -> dict[str, Any]:
    """Разбирает JSON-объект, tolerating markdown-обёртки и лишний текст."""
    candidate = extract_json_text(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # попытка починить хвостовые запятые
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Не удалось разобрать JSON: {candidate[:200]}") from exc
    if isinstance(data, list):
        data = {"items": data}
    if not isinstance(data, dict):
        raise LLMError("Ожидался JSON-объект")
    return data
