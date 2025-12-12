from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ChatMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Memory(Protocol):
    def append(self, message: ChatMessage) -> None:
        ...

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        ...

    def clear(self) -> None:
        ...

