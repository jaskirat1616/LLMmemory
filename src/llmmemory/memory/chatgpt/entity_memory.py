"""
ChatGPT-style entity memory with fact extraction, storage, and retrieval.
This implements the full ChatGPT memory system with:
- Conversation buffer (sliding window)
- Fact extraction and storage
- Hybrid search (semantic + keyword)
- Conversation summaries
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List

import numpy as np

from ..base import ChatMessage, Memory
from ...models.loader import format_prompt, generate_reply


@dataclass
class MemoryFact:
    """A structured fact stored in long-term memory."""

    fact: str  # The actual fact text
    category: str  # e.g., "user_preference", "user_info", "project_detail"
    source_message: str  # The original message that led to this fact
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    relevance_keywords: List[str] = field(default_factory=list)  # Keywords for search


@dataclass
class ConversationSummary:
    """Summary of a past conversation."""

    summary: str
    key_topics: List[str]
    timestamp: float = field(default_factory=lambda: __import__("time").time())


def _simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """
    Simple embedding using hashed bag-of-words.
    In production, you'd use a proper embedding model.
    """
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
    """Calculate cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text for keyword-based search."""
    # Simple keyword extraction - remove stopwords and get important terms
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
    }
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    # Return most frequent keywords
    from collections import Counter
    return [word for word, _ in Counter(keywords).most_common(max_keywords)]


class EntityMemory(Memory):
    """
    Full ChatGPT-style memory system with:
    1. Conversation buffer (sliding window)
    2. Long-term fact storage
    3. Conversation summaries
    4. Hybrid search (semantic + keyword)
    """

    def __init__(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        model_config: Any | None = None,
        buffer_size: int = 20,
        extract_facts_every: int = 5,
        summarize_every: int = 15,
        max_facts: int = 100,
        fact_extractor: Callable[[List[ChatMessage], Any, Any, Any], List[MemoryFact]] | None = None,
    ):
        """
        Args:
            model: The MLX model for fact extraction and summarization
            tokenizer: The tokenizer for the model
            model_config: Model configuration
            buffer_size: Size of the conversation buffer
            extract_facts_every: Extract facts every N messages
            summarize_every: Create conversation summary every N messages
            max_facts: Maximum number of facts to store
            fact_extractor: Custom fact extraction function
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.extract_facts_every = extract_facts_every
        self.summarize_every = summarize_every
        self.max_facts = max_facts

        # Storage
        self._buffer: Deque[ChatMessage] = deque(maxlen=buffer_size)
        self._facts: List[MemoryFact] = []
        self._summaries: List[ConversationSummary] = []

        # Counters
        self._messages_since_extraction = 0
        self._messages_since_summary = 0

        # Fact extraction function
        self._fact_extractor = fact_extractor or self._default_fact_extractor

    def append(self, message: ChatMessage) -> None:
        """Add a message and potentially extract facts or create summaries."""
        self._buffer.append(message)
        self._messages_since_extraction += 1
        self._messages_since_summary += 1

        # Extract facts periodically
        if (
            self._messages_since_extraction >= self.extract_facts_every
            and self.model is not None
            and self.tokenizer is not None
        ):
            self._extract_and_store_facts()
            self._messages_since_extraction = 0

        # Create summaries periodically
        if (
            self._messages_since_summary >= self.summarize_every
            and self.model is not None
            and self.tokenizer is not None
        ):
            self._create_summary()
            self._messages_since_summary = 0

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        """
        Get context combining:
        1. Recent conversation buffer
        2. Relevant facts from long-term memory
        3. Recent conversation summaries
        """
        context: list[ChatMessage] = []

        # Add relevant facts if we have a query
        if query and self._facts:
            relevant_facts = self._search_facts(query, top_k=min(5, len(self._facts)))
            if relevant_facts:
                facts_text = "\n".join([f"- {fact.fact}" for fact in relevant_facts])
                context.append(
                    ChatMessage(
                        role="system",
                        content=f"Relevant memories:\n{facts_text}",
                        metadata={"source": "long_term_memory"},
                    )
                )

        # Add recent summaries
        if self._summaries:
            recent_summaries = self._summaries[-2:]  # Last 2 summaries
            summaries_text = "\n\n".join([s.summary for s in recent_summaries])
            context.append(
                ChatMessage(
                    role="system",
                    content=f"Previous conversation summaries:\n{summaries_text}",
                    metadata={"source": "conversation_summaries"},
                )
            )

        # Add recent conversation buffer
        context.extend(list(self._buffer))

        return context

    def _extract_and_store_facts(self) -> None:
        """Extract facts from recent conversation and store them."""
        if not self.model or not self.tokenizer:
            return

        recent_messages = list(self._buffer)[-self.extract_facts_every:]
        facts = self._fact_extractor(recent_messages, self.model, self.tokenizer, self.model_config)

        # Add facts to storage
        for fact in facts:
            self._facts.append(fact)
            # Limit total facts
            if len(self._facts) > self.max_facts:
                self._facts = self._facts[-self.max_facts:]

    def _create_summary(self) -> None:
        """Create a summary of recent conversation."""
        if not self.model or not self.tokenizer:
            return

        recent_messages = list(self._buffer)[-self.summarize_every:]
        summary_text = self._summarize_conversation(recent_messages)
        key_topics = _extract_keywords(" ".join([msg.content for msg in recent_messages]))

        summary = ConversationSummary(summary=summary_text, key_topics=key_topics)
        self._summaries.append(summary)

        # Keep only last 5 summaries
        if len(self._summaries) > 5:
            self._summaries = self._summaries[-5:]

    def _search_facts(self, query: str, top_k: int = 5) -> List[MemoryFact]:
        """
        Hybrid search: combines semantic similarity and keyword matching.
        Returns top_k most relevant facts.
        """
        if not self._facts:
            return []

        query_embedding = _simple_embed(query)
        query_keywords = set(_extract_keywords(query))

        scored_facts = []

        for fact in self._facts:
            # Semantic similarity score
            fact_embedding = _simple_embed(fact.fact)
            semantic_score = _cosine_similarity(query_embedding, fact_embedding)

            # Keyword match score
            fact_keywords = set(fact.relevance_keywords)
            keyword_matches = len(query_keywords & fact_keywords)
            keyword_score = keyword_matches / max(len(query_keywords), 1)

            # Combined score (weighted average)
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score
            scored_facts.append((fact, combined_score))

        # Sort by score and return top_k
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in scored_facts[:top_k]]

    def _default_fact_extractor(
        self, messages: List[ChatMessage], model: Any, tokenizer: Any, config: Any
    ) -> List[MemoryFact]:
        """
        Default fact extraction using the model.
        Prompts the model to extract key facts about the user and conversation.
        """
        # Build prompt for fact extraction
        conversation_text = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in messages if msg.role != "system"]
        )

        extraction_prompt = f"""Analyze the following conversation and extract key facts that should be remembered for future conversations. Focus on:
- User preferences and interests
- Personal information (name, location, profession, etc.)
- Goals and plans mentioned
- Important context or background

Conversation:
{conversation_text}

Extract facts in the format:
FACT: [the fact]
CATEGORY: [user_info|user_preference|goal|project_detail|other]

If no relevant facts are found, respond with "NO_FACTS"."""

        try:
            # Use model to extract facts
            extraction_messages = [
                ChatMessage(role="system", content="You are a fact extraction assistant."),
                ChatMessage(role="user", content=extraction_prompt),
            ]
            
            # Create a config for fact extraction (lower temperature, fewer tokens)
            from llmmemory.config import ModelConfig as MC
            extract_config = MC(
                max_tokens=256,
                temperature=0.3,  # Lower temperature for more deterministic extraction
            )
            
            # Use generate_reply which handles both LM and VLM
            response = generate_reply(
                model, tokenizer, extraction_messages, extract_config, model_kind="auto"
            )

            # Parse response
            facts = []
            if "NO_FACTS" not in response.upper():
                # Parse extracted facts
                fact_blocks = re.split(r"FACT:\s*", response, flags=re.IGNORECASE)
                for block in fact_blocks[1:]:  # Skip first empty split
                    lines = block.strip().split("\n")
                    if lines:
                        fact_text = lines[0].strip()
                        category = "other"
                        if len(lines) > 1:
                            category_match = re.search(r"CATEGORY:\s*(\w+)", lines[1], re.IGNORECASE)
                            if category_match:
                                category = category_match.group(1).lower()

                        if fact_text:
                            source = messages[-1].content if messages else ""
                            keywords = _extract_keywords(fact_text)
                            facts.append(
                                MemoryFact(
                                    fact=fact_text,
                                    category=category,
                                    source_message=source,
                                    relevance_keywords=keywords,
                                )
                            )

        except Exception as e:
            # Fallback: simple heuristic extraction
            print(f"Fact extraction error: {e}, using fallback")
            facts = self._heuristic_fact_extraction(messages)

        return facts

    def _heuristic_fact_extraction(self, messages: List[ChatMessage]) -> List[MemoryFact]:
        """
        Fallback heuristic fact extraction when model extraction fails.
        Looks for common patterns like "I am", "I like", "My name is", etc.
        """
        facts = []
        patterns = [
            (r"my name is (\w+)", "user_info", "name"),
            (r"i am (\w+)", "user_info", "identity"),
            (r"i like ([^.!?]+)", "user_preference", "preference"),
            (r"i work (?:as|at|in) ([^.!?]+)", "user_info", "profession"),
            (r"i live (?:in|at) ([^.!?]+)", "user_info", "location"),
        ]

        for msg in messages:
            if msg.role == "user":
                content = msg.content.lower()
                for pattern, category, keyword in patterns:
                    match = re.search(pattern, content)
                    if match:
                        fact_text = match.group(0)
                        facts.append(
                            MemoryFact(
                                fact=fact_text,
                                category=category,
                                source_message=msg.content,
                                relevance_keywords=[keyword],
                            )
                        )
                        break  # One fact per message to avoid duplicates

        return facts

    def _summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """Create a summary of the conversation using the model."""
        conversation_text = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in messages if msg.role != "system"]
        )

        summary_prompt = f"""Summarize the following conversation in 2-3 sentences, focusing on:
- Main topics discussed
- Key decisions or conclusions
- Important context for future conversations

Conversation:
{conversation_text}

Summary:"""

        try:
            summary_messages = [
                ChatMessage(role="system", content="You are a conversation summarizer."),
                ChatMessage(role="user", content=summary_prompt),
            ]
            
            # Create a config for summarization
            from llmmemory.config import ModelConfig as MC
            summary_config = MC(
                max_tokens=150,
                temperature=0.5,
            )
            
            summary = generate_reply(
                self.model, self.tokenizer, summary_messages, summary_config, model_kind="auto"
            )

            return summary.strip()
        except Exception as e:
            # Fallback: simple summary
            print(f"Summary error: {e}, using fallback")
            return f"Conversation about: {', '.join(_extract_keywords(conversation_text)[:5])}"

    def clear(self) -> None:
        """Clear all memory (buffer, facts, summaries)."""
        self._buffer.clear()
        self._facts.clear()
        self._summaries.clear()
        self._messages_since_extraction = 0
        self._messages_since_summary = 0

    def get_facts(self) -> List[MemoryFact]:
        """Get all stored facts (for inspection/debugging)."""
        return self._facts.copy()

    def get_summaries(self) -> List[ConversationSummary]:
        """Get all conversation summaries (for inspection/debugging)."""
        return self._summaries.copy()

    def add_fact(self, fact: str, category: str = "other", source: str = "") -> None:
        """Manually add a fact to memory."""
        keywords = _extract_keywords(fact)
        self._facts.append(
            MemoryFact(
                fact=fact,
                category=category,
                source_message=source,
                relevance_keywords=keywords,
            )
        )
        if len(self._facts) > self.max_facts:
            self._facts = self._facts[-self.max_facts:]

    def delete_fact(self, fact_text: str) -> bool:
        """Delete a fact by matching its text. Returns True if found and deleted."""
        original_count = len(self._facts)
        self._facts = [f for f in self._facts if f.fact.lower() != fact_text.lower()]
        return len(self._facts) < original_count
