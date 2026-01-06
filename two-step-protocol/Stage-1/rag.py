"""
rag.py

Lightweight chapter-only RAG system used as supporting context
for Stage 1 (Scenario 1).

This is NOT a document ingestion or search engine.
"""

from typing import List, Dict
import math
import re


class SimpleRAGSystem:
    """
    Extremely lightweight RAG:
    - Splits text into fixed-size chunks
    - Scores chunks using simple token overlap
    - Returns top-k chunks with metadata
    """

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.chunks: List[Dict] = []

    # --------------------------------------------------------
    # Index building
    # --------------------------------------------------------

    def build_index_from_blocks(self, blocks: List[Dict]) -> None:
        """
        Each block must contain:
          - text
          - source_title
          - source_url
        """
        self.chunks = []

        for block in blocks:
            text = block.get("text", "")
            source_title = block.get("source_title", "")
            source_url = block.get("source_url", "")

            for i in range(0, len(text), self.chunk_size):
                chunk_text = text[i:i + self.chunk_size].strip()
                if not chunk_text:
                    continue

                self.chunks.append({
                    "text": chunk_text,
                    "source_title": source_title,
                    "source_url": source_url,
                })

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    def retrieve_relevant_chunks_with_meta(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Returns top-k chunks ranked by token overlap.
        Output is JSON-serializable.
        """
        if not query or not self.chunks:
            return []

        query_tokens = self._tokenize(query)

        scored = []
        for chunk in self.chunks:
            chunk_tokens = self._tokenize(chunk["text"])
            score = self._overlap_score(query_tokens, chunk_tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored[:top_k]:
            results.append({
                "text": chunk["text"],
                "source_title": chunk["source_title"],
                "source_url": chunk["source_url"],
                "score": round(score, 4),
            })

        return results

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _tokenize(self, text: str) -> set:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return set(text.split())

    def _overlap_score(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / math.sqrt(len(a) * len(b))
