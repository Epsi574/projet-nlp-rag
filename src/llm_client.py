from __future__ import annotations
import requests


class LLMClient:
    """
    Client minimal compatible avec l'API /api/generate (type ollama proxy/ngrok)
    qui renvoie généralement un JSON {"response": "..."}.
    """

    def __init__(self, api_url: str, model: str):
        self.api_url = (api_url or "").strip()
        self.model = (model or "").strip()

        if not self.api_url:
            raise ValueError("LLM_API_URL est vide. Renseigne-le dans .env")

        if not self.model:
            raise ValueError("LLM_MODEL est vide. Renseigne-le dans .env")

    def generate(self, prompt: str, stream: bool = False, timeout: int = 120) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        r = requests.post(self.api_url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        if "response" in data:
            return data["response"]
        # fallback si le proxy change
        return str(data)
