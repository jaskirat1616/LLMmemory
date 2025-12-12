from collections import deque
from typing import Deque

from .base import ChatMessage, Memory


class BufferMemory(Memory):
    """Plain sliding window buffer."""

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self._messages: Deque[ChatMessage] = deque(maxlen=max_messages)

    def append(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        del query, k  # unused but kept for interface compatibility
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

