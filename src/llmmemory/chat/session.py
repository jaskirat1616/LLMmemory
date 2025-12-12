from __future__ import annotations

from typing import Any

from llmmemory.config import ModelConfig, SYSTEM_PROMPT
from llmmemory.memory.base import ChatMessage, Memory
from llmmemory.models.loader import generate_reply


class ChatSession:
    """Orchestrates a model chat loop with a pluggable memory backend."""

    def __init__(
        self,
        memory: Memory,
        model: Any,
        tokenizer: Any,
        model_kind: str = "auto",
        system_prompt: str = SYSTEM_PROMPT,
        model_config: ModelConfig | None = None,
    ):
        self.memory = memory
        self.model = model
        self.tokenizer = tokenizer
        self.model_kind = model_kind
        self.model_config = model_config or ModelConfig()

        if system_prompt:
            self.memory.append(ChatMessage(role="system", content=system_prompt))

    @property
    def messages(self) -> list[ChatMessage]:
        return self.memory.get_context()

    def add_user_message(self, content: str) -> None:
        self.memory.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.memory.append(ChatMessage(role="assistant", content=content))

    def generate_assistant_reply(self) -> str:
        # use the last user message as a query hint for vector-style memory
        query = None
        for msg in reversed(self.memory.get_context()):
            if msg.role == "user":
                query = msg.content
                break
        context = self.memory.get_context(query=query)
        reply = generate_reply(
            self.model,
            self.tokenizer,
            context,
            self.model_config,
            model_kind=self.model_kind,
        )
        self.add_assistant_message(reply)
        return reply

    def reset(self) -> None:
        self.memory.clear()
        if SYSTEM_PROMPT:
            self.memory.append(ChatMessage(role="system", content=SYSTEM_PROMPT))

