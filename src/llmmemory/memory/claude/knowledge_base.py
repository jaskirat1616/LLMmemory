"""
Claude-style knowledge base for long-form knowledge storage.
Manages documents and knowledge that can be attached to conversations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..base import ChatMessage, Memory


@dataclass
class KnowledgeDocument:
    """A document in the knowledge base."""

    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """Simple embedding for semantic search."""
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    for token in tokens:
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class KnowledgeBase(Memory):
    """
    Claude-style knowledge base memory.
    Stores long-form documents and retrieves relevant knowledge based on queries.
    """

    def __init__(self, max_messages: int = 50, max_documents: int = 100):
        """
        Args:
            max_messages: Maximum messages in conversation buffer
            max_documents: Maximum documents in knowledge base
        """
        self.max_messages = max_messages
        self.max_documents = max_documents

        # Conversation buffer
        self._messages: List[ChatMessage] = []

        # Knowledge base
        self._documents: List[KnowledgeDocument] = []
        self._document_embeddings: List[np.ndarray] = []

    def append(self, message: ChatMessage) -> None:
        """Add a message to the conversation buffer."""
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        """
        Get context combining:
        1. Relevant knowledge base documents (if query provided)
        2. Recent conversation messages
        """
        context: list[ChatMessage] = []

        # Retrieve relevant documents if query provided
        if query and self._documents:
            relevant_docs = self._search_documents(query, top_k=3)
            if relevant_docs:
                docs_text = "\n\n".join(
                    [f"[{doc.title}]\n{doc.content}" for doc in relevant_docs]
                )
                context.append(
                    ChatMessage(
                        role="system",
                        content=f"Relevant knowledge:\n{docs_text}",
                        metadata={"source": "knowledge_base"},
                    )
                )

        # Add recent conversation
        context.extend(self._messages[-k:])

        return context

    def clear(self) -> None:
        """Clear conversation buffer (keeps knowledge base)."""
        self._messages.clear()

    def add_document(self, title: str, content: str, tags: List[str] | None = None) -> None:
        """Add a document to the knowledge base."""
        doc = KnowledgeDocument(title=title, content=content, tags=tags or [])
        self._documents.append(doc)
        self._document_embeddings.append(_simple_embed(f"{title} {content}"))

        # Limit documents
        if len(self._documents) > self.max_documents:
            self._documents = self._documents[-self.max_documents :]
            self._document_embeddings = self._document_embeddings[-self.max_documents :]

    def _search_documents(self, query: str, top_k: int = 3) -> List[KnowledgeDocument]:
        """Search documents by semantic similarity."""
        if not self._documents:
            return []

        query_embedding = _simple_embed(query)
        scored_docs = []

        for doc, doc_embedding in zip(self._documents, self._document_embeddings):
            score = _cosine_similarity(query_embedding, doc_embedding)
            # Boost score if query keywords match document tags
            if doc.tags:
                query_lower = query.lower()
                tag_matches = sum(1 for tag in doc.tags if tag.lower() in query_lower)
                score += 0.1 * tag_matches
            scored_docs.append((doc, score))

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def get_documents(self) -> List[KnowledgeDocument]:
        """Get all documents (for inspection)."""
        return self._documents.copy()

    def clear_knowledge_base(self) -> None:
        """Clear the entire knowledge base."""
        self._documents.clear()
        self._document_embeddings.clear()
