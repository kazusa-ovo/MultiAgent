from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SUPPORTED = {".txt", ".md", ".py", ".yaml", ".json", ".pdf", ".docx"}


class DocumentLoader:
    """Load and chunk documents of various formats."""

    @staticmethod
    def load(file_path: str) -> list[str]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported file type: {ext}")

        if ext == ".pdf":
            text = DocumentLoader._load_pdf(file_path)
        elif ext == ".docx":
            text = DocumentLoader._load_docx(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        return DocumentLoader._chunk(text, file_name=os.path.basename(file_path))

    @staticmethod
    def _chunk(text: str, file_name: str = "") -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if file_name:
                chunk = f"[Source: {file_name}]\n{chunk}"
            chunks.append(chunk)
            start = end - CHUNK_OVERLAP
        return chunks

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        import fitz
        doc = fitz.open(file_path)
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts)

    @staticmethod
    def _load_docx(file_path: str) -> str:
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
