"""
Adapter for mem0 memory system.
Wraps mem0 to implement the Memory protocol.
"""

from __future__ import annotations

from typing import Any, List

from ..base import ChatMessage, Memory


class Mem0Adapter(Memory):
    """
    Adapter that wraps mem0's memory system to implement the Memory protocol.
    
    mem0 is a graph-based memory system that stores memories as entities
    and relationships. This adapter allows using mem0 within LLMmemory.
    """

    def __init__(
        self,
        agent_id: str = "default",
        config: dict[str, Any] | None = None,
    ):
        """
        Args:
            agent_id: Unique identifier for this agent/memory instance
            config: Optional mem0 configuration dictionary
        """
        try:
            from mem0 import Memory as Mem0Memory
        except ImportError:
            raise ImportError(
                "mem0ai is required for mem0 integration. Install with `pip install mem0ai`."
            )

        self.agent_id = agent_id
        self.config = config or {}
        
        # Initialize mem0
        self._mem0 = Mem0Memory(config=self.config)
        
        # Local buffer for recent messages (mem0 handles long-term)
        self._recent_messages: List[ChatMessage] = []
        self._max_recent = 20

    def append(self, message: ChatMessage) -> None:
        """
        Add a message to memory.
        Stores in mem0 for long-term and keeps in local buffer for immediate context.
        """
        # Add to recent buffer
        self._recent_messages.append(message)
        if len(self._recent_messages) > self._max_recent:
            self._recent_messages = self._recent_messages[-self._max_recent :]

        # Store in mem0 if it's a user or assistant message (not system)
        if message.role in ["user", "assistant"]:
            try:
                # mem0 automatically extracts and stores memories
                self._mem0.add(
                    messages=[
                        {"role": message.role, "content": message.content}
                    ],
                    agent_id=self.agent_id,
                )
            except Exception as e:
                # If mem0 fails, just log and continue with buffer
                print(f"Warning: mem0 storage failed: {e}")

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        """
        Get context combining:
        1. Recent messages from buffer
        2. Relevant memories from mem0 (if query provided)
        """
        context: list[ChatMessage] = []

        # Get relevant memories from mem0
        if query:
            try:
                memories = self._mem0.search(
                    query=query,
                    agent_id=self.agent_id,
                    limit=k,
                )
                
                if memories:
                    # Format memories as system messages
                    memories_text = "\n".join(
                        [f"- {mem.get('memory', mem)}" for mem in memories]
                    )
                    context.append(
                        ChatMessage(
                            role="system",
                            content=f"Relevant memories:\n{memories_text}",
                            metadata={"source": "mem0"},
                        )
                    )
            except Exception as e:
                print(f"Warning: mem0 search failed: {e}")

        # Add recent messages
        context.extend(self._recent_messages[-k:])

        return context

    def clear(self) -> None:
        """Clear recent buffer. Note: mem0 memories persist unless explicitly deleted."""
        self._recent_messages.clear()
        # Optionally clear mem0 memories for this agent
        # self._mem0.delete_all(agent_id=self.agent_id)

    def get_all_memories(self) -> List[dict]:
        """Get all memories from mem0 (for inspection)."""
        try:
            return self._mem0.get_all(agent_id=self.agent_id)
        except Exception as e:
            print(f"Warning: mem0 get_all failed: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory from mem0."""
        try:
            self._mem0.delete(memory_id=memory_id, agent_id=self.agent_id)
            return True
        except Exception as e:
            print(f"Warning: mem0 delete failed: {e}")
            return False

    def clear_mem0_memories(self) -> None:
        """Clear all mem0 memories for this agent."""
        try:
            self._mem0.delete_all(agent_id=self.agent_id)
        except Exception as e:
            print(f"Warning: mem0 delete_all failed: {e}")
