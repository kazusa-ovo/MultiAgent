from __future__ import annotations

import os

import httpx


class Embedder:
    """OpenAI-compatible embeddings client."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._api_key = os.getenv("OPENAI_API_KEY", "")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding API error: {resp.status_code} {resp.text}")
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
