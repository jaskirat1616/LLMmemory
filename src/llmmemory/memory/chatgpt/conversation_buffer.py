"""
ChatGPT-style conversation buffer - sliding window of recent messages.
This is the first layer of ChatGPT's memory system.
"""

from collections import deque
from typing import Deque

from ..base import ChatMessage, Memory


class ConversationBuffer(Memory):
    """
    Sliding window conversation buffer - maintains the last N messages.
    This mimics ChatGPT's immediate context window for ongoing conversations.
    """

    def __init__(self, max_messages: int = 50):
        """
        Args:
            max_messages: Maximum number of messages to keep in the buffer
        """
        self.max_messages = max_messages
        self._messages: Deque[ChatMessage] = deque(maxlen=max_messages)

    def append(self, message: ChatMessage) -> None:
        """Add a message to the buffer."""
        self._messages.append(message)

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        """
        Get all messages in the buffer (query and k are ignored for simplicity).
        In a full implementation, you might want to filter based on query relevance.
        """
        del query, k  # unused but kept for interface compatibility
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self._messages.clear()

    @property
    def message_count(self) -> int:
        """Get the current number of messages in the buffer."""
        return len(self._messages)
