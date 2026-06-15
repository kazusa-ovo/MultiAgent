from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.tools.base import BaseTool


@dataclass
class RAGSearchTool(BaseTool):
    name: str = "rag_search"
    description: str = "Search uploaded documents and knowledge base for relevant information"

    def __post_init__(self):
        super().__post_init__()
        self._embedder = Embedder()
        self._store = VectorStore()

    async def run(self, query: str = "", top_k: int = 3, **kwargs: Any) -> str:
        try:
            embedding = await self._embedder.embed_query(query)
        except Exception as e:
            return f"Error: embedding failed - {e}"

        results = self._store.search(embedding, top_k=top_k)

        if not results:
            return "No relevant documents found in the knowledge base."

        lines = [f"Found {len(results)} relevant document(s):\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"---Source: {r['source']} (relevance: {r['score']:.2f})---\n"
                f"{r['content'][:1200]}"
            )
        return "\n".join(lines)
