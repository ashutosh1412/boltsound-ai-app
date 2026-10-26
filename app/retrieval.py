"""Retrieval (RAG): find the help-centre text most relevant to a question."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from google import genai

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
EMBED_MODEL = "gemini-embedding-001"


@dataclass
class Chunk:
    source: str  # file name, shown to the customer as the source
    text: str


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """One chunk per document (the docs are short)."""
    return [Chunk(path.name, path.read_text()) for path in sorted(docs_dir.glob("*.md"))]


class Retriever:
    def __init__(self, api_key: str, chunks: list[Chunk], top_k: int = 1) -> None:
        self._client = genai.Client(api_key=api_key)
        self.chunks = chunks
        self.top_k = top_k
        self._chunk_vectors = self._embed([chunk.text for chunk in chunks])

    def search(self, question: str) -> list[Chunk]:
        question_vector = self._embed([question])[0]
        scores = self._chunk_vectors @ question_vector
        best_first = np.argsort(scores)[::-1][: self.top_k]
        return [self.chunks[i] for i in best_first]

    def _embed(self, texts: list[str]) -> np.ndarray:
        result = self._client.models.embed_content(model=EMBED_MODEL, contents=texts)
        vectors = np.array([item.values for item in result.embeddings])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
