from __future__ import annotations

import json
import re
from typing import Any

import requests


class OpenRouterClient:
    def __init__(
            self,
            api_key: str,
            model: str,
            base_url: str = "https://openrouter.ai/api/v1",
            app_name: str = "Music Agent API",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name

    def chat_text(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.2,
            max_tokens: int = 700,
            model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenRouter returned empty content: {data}")

        return content.strip()

    def chat_json(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.2,
            max_tokens: int = 700,
            model: str | None = None,
    ) -> dict[str, Any]:
        text = self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        return self._extract_json_object(text)

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        raw_obj = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if raw_obj:
            try:
                parsed = json.loads(raw_obj.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        raise RuntimeError(f"Failed to parse JSON from LLM output:\n{text}")