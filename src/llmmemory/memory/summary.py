from __future__ import annotations

from collections import deque
from typing import Callable, Deque, List

from .base import ChatMessage, Memory


def _default_summarizer(messages: List[ChatMessage], prior_summary: str) -> str:
    """Lightweight heuristic summary to avoid model calls in the skeleton."""
    recent_text = " ".join(msg.content for msg in messages[-4:])
    if prior_summary:
        return f"{prior_summary}\nRecent: {recent_text}"
    return f"Recent: {recent_text}"


class SummaryMemory(Memory):
    """
    Maintains a running summary plus a short buffer.
    Summarization uses a pluggable callable to allow swapping in the model later.
    """

    def __init__(
        self,
        buffer_size: int = 6,
        summarize_every: int = 6,
        summarizer: Callable[[list[ChatMessage], str], str] | None = None,
    ):
        self.buffer_size = buffer_size
        self.summarize_every = summarize_every
        self._buffer: Deque[ChatMessage] = deque(maxlen=buffer_size)
        self._summary: str = ""
        self._summarizer = summarizer or _default_summarizer
        self._seen_since_summary = 0

    def append(self, message: ChatMessage) -> None:
        self._buffer.append(message)
        self._seen_since_summary += 1

        if self._seen_since_summary >= self.summarize_every:
            self._summary = self._summarizer(list(self._buffer), self._summary)
            self._seen_since_summary = 0

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        del query, k  # unused
        context: list[ChatMessage] = []
        if self._summary:
            context.append(ChatMessage(role="system", content=f"Summary: {self._summary}"))
        context.extend(list(self._buffer))
        return context

    def clear(self) -> None:
        self._buffer.clear()
        self._summary = ""
        self._seen_since_summary = 0

