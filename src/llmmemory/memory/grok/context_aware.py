"""
Grok-style context-aware memory system.
Dynamically builds context based on conversation flow and real-time needs.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, List

import numpy as np

from ..base import ChatMessage, Memory


def _simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """Simple embedding for semantic analysis."""
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


def _extract_intent(text: str) -> List[str]:
    """Extract intent/keywords from text."""
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
    }
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [w for w in words if w not in stopwords and len(w) > 2]


class ContextAwareMemory(Memory):
    """
    Grok-style context-aware memory.
    Dynamically builds context based on:
    - Current conversation flow
    - User intent analysis
    - Semantic relevance to recent messages
    - Real-time context adaptation
    """

    def __init__(self, max_messages: int = 200, context_window: int = 20):
        """
        Args:
            max_messages: Maximum messages to store
            context_window: Number of recent messages to consider for context building
        """
        self.max_messages = max_messages
        self.context_window = context_window

        # Message storage with embeddings
        self._messages: List[ChatMessage] = []
        self._message_embeddings: List[np.ndarray] = []
        self._message_intents: List[List[str]] = []  # Keywords/intents per message

        # Conversation flow tracking
        self._conversation_topics: deque = deque(maxlen=10)  # Recent topics
        self._current_intent: List[str] = []

    def append(self, message: ChatMessage) -> None:
        """Add a message and update context awareness."""
        self._messages.append(message)
        embedding = _simple_embed(message.content)
        self._message_embeddings.append(embedding)
        intent = _extract_intent(message.content)
        self._message_intents.append(intent)

        # Update conversation topics
        if message.role == "user":
            self._current_intent = intent
            # Extract main topic (most frequent keyword)
            if intent:
                from collections import Counter
                top_keyword = Counter(intent).most_common(1)[0][0]
                self._conversation_topics.append(top_keyword)

        # Limit storage
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
            self._message_embeddings = self._message_embeddings[-self.max_messages :]
            self._message_intents = self._message_intents[-self.max_messages :]

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        """
        Dynamically build context based on:
        1. Recent messages (always included)
        2. Semantically relevant past messages (if query provided)
        3. Messages related to current conversation topics
        """
        context: list[ChatMessage] = []

        # Always include recent messages
        recent_messages = self._messages[-self.context_window :]
        context.extend(recent_messages)

        # If we have a query, find semantically relevant older messages
        if query and len(self._messages) > self.context_window:
            query_embedding = _simple_embed(query)
            query_intent = set(_extract_intent(query))

            # Score older messages (not in recent window)
            older_messages = self._messages[: -self.context_window]
            scored_messages = []

            for i, (msg, msg_embedding, msg_intent) in enumerate(
                zip(older_messages, self._message_embeddings[: -self.context_window], self._message_intents[: -self.context_window])
            ):
                # Semantic similarity
                semantic_score = _cosine_similarity(query_embedding, msg_embedding)

                # Intent/keyword overlap
                msg_intent_set = set(msg_intent)
                intent_overlap = len(query_intent & msg_intent_set) / max(
                    len(query_intent), 1
                )

                # Topic relevance (if message topic matches recent topics)
                topic_relevance = 0.0
                if msg_intent:
                    for topic in self._conversation_topics:
                        if topic in msg_intent:
                            topic_relevance += 0.2
                            break

                # Combined score
                combined_score = 0.5 * semantic_score + 0.3 * intent_overlap + 0.2 * topic_relevance
                scored_messages.append((msg, combined_score))

            # Get top relevant messages
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            relevant_messages = [msg for msg, _ in scored_messages[:k]]

            # Insert relevant messages before recent ones (chronological order)
            # But avoid duplicates
            seen_ids = {id(m) for m in context}
            for msg in relevant_messages:
                if id(msg) not in seen_ids:
                    context.insert(-len(recent_messages), msg)
                    seen_ids.add(id(msg))

        # Limit total context size
        return context[:k * 2]  # Allow up to 2x k for dynamic context

    def clear(self) -> None:
        """Clear all memory."""
        self._messages.clear()
        self._message_embeddings.clear()
        self._message_intents.clear()
        self._conversation_topics.clear()
        self._current_intent.clear()

    def get_conversation_topics(self) -> List[str]:
        """Get recent conversation topics (for inspection)."""
        return list(self._conversation_topics)

    def get_current_intent(self) -> List[str]:
        """Get current user intent keywords (for inspection)."""
        return self._current_intent.copy()
