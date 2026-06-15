from __future__ import annotations

import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB-backed vector store for document embeddings."""

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name="documents")

    def add(self, chunks: list[str], embeddings: list[list[float]], source: str = "") -> None:
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source} for _ in chunks]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        if not results["documents"] or not results["documents"][0]:
            return []

        output: list[dict] = []
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0] * len(docs)
        for doc, meta, dist in zip(docs, metas, distances):
            output.append({
                "content": doc,
                "source": meta.get("source", "") if meta else "",
                "score": round(1 - dist, 4) if dist else 1.0,
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(name="documents")
