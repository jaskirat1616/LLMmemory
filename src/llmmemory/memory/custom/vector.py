from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

from ..base import ChatMessage, Memory


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _embed(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VectorMemory(Memory):
    """
    Lightweight vector-ish memory using hashed bag-of-words embeddings.
    Swap in a real embedder/FAISS later when needed.
    """

    def __init__(self, dim: int = 512, max_messages: int = 200):
        self.dim = dim
        self.max_messages = max_messages
        self._store: List[Tuple[ChatMessage, np.ndarray]] = []

    def append(self, message: ChatMessage) -> None:
        embedding = _embed(message.content, self.dim)
        self._store.append((message, embedding))
        if len(self._store) > self.max_messages:
            # drop oldest
            self._store = self._store[-self.max_messages :]

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        if not self._store:
            return []

        if query is None:
            query = self._store[-1][0].content
        query_vec = _embed(query, self.dim)

        ranked = sorted(
            self._store,
            key=lambda item: _cosine(item[1], query_vec),
            reverse=True,
        )
        top = [msg for msg, _ in ranked[: max(k, 1)]]
        # keep chronological ordering within the selected set for better coherence
        seen = set(id(m) for m in top)
        ordered = [msg for msg, _ in self._store if id(msg) in seen]
        return ordered

    def clear(self) -> None:
        self._store.clear()

