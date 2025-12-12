"""
Factory for creating memory instances from system and implementation names.
"""

from typing import Any

from .base import Memory
from .chatgpt import ConversationBuffer, EntityMemory
from .claude import KnowledgeBase, ProjectMemory
from .custom import BufferMemory, SummaryMemory, VectorMemory
from .grok import ContextAwareMemory
from .mem0 import Mem0Adapter


def create_memory(
    system: str,
    implementation: str,
    model: Any = None,
    tokenizer: Any = None,
    model_config: Any = None,
    **kwargs,
) -> Memory:
    """
    Create a memory instance from system and implementation names.
    
    Args:
        system: Memory system name (chatgpt, claude, grok, mem0, custom)
        implementation: Implementation name within the system
        model: Optional model for implementations that need it
        tokenizer: Optional tokenizer for implementations that need it
        model_config: Optional model config for implementations that need it
        **kwargs: Additional arguments passed to the memory constructor
    
    Returns:
        Memory instance
    """
    if system == "chatgpt":
        if implementation == "conversation_buffer":
            return ConversationBuffer(max_messages=kwargs.get("max_messages", 50))
        elif implementation == "entity_memory":
            return EntityMemory(
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                buffer_size=kwargs.get("buffer_size", 20),
                extract_facts_every=kwargs.get("extract_facts_every", 5),
                summarize_every=kwargs.get("summarize_every", 15),
                max_facts=kwargs.get("max_facts", 100),
            )
        else:
            raise ValueError(f"Unknown ChatGPT implementation: {implementation}")

    elif system == "claude":
        if implementation == "project_memory":
            return ProjectMemory(
                current_project=kwargs.get("current_project", "default"),
                max_messages_per_project=kwargs.get("max_messages_per_project", 100),
            )
        elif implementation == "knowledge_base":
            return KnowledgeBase(
                max_messages=kwargs.get("max_messages", 50),
                max_documents=kwargs.get("max_documents", 100),
            )
        else:
            raise ValueError(f"Unknown Claude implementation: {implementation}")

    elif system == "grok":
        if implementation == "context_aware":
            return ContextAwareMemory(
                max_messages=kwargs.get("max_messages", 200),
                context_window=kwargs.get("context_window", 20),
            )
        else:
            raise ValueError(f"Unknown Grok implementation: {implementation}")

    elif system == "mem0":
        if implementation == "mem0_adapter":
            return Mem0Adapter(
                agent_id=kwargs.get("agent_id", "default"),
                config=kwargs.get("config"),
            )
        else:
            raise ValueError(f"Unknown mem0 implementation: {implementation}")

    elif system == "custom":
        if implementation == "buffer":
            return BufferMemory(max_messages=kwargs.get("max_messages", 50))
        elif implementation == "summary":
            return SummaryMemory(
                buffer_size=kwargs.get("buffer_size", 8),
                summarize_every=kwargs.get("summarize_every", 8),
            )
        elif implementation == "vector":
            return VectorMemory(
                dim=kwargs.get("dim", 512),
                max_messages=kwargs.get("max_messages", 200),
            )
        else:
            raise ValueError(f"Unknown custom implementation: {implementation}")

    else:
        raise ValueError(f"Unknown memory system: {system}")
