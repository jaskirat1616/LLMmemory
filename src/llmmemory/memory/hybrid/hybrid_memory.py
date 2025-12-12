"""
Hybrid memory system combining semantic, episodic, and summary stores.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import ChatMessage, Memory
from .backends.base import MemoryBackend
from .backends.in_memory import InMemoryBackend
from .types import (
    DecayStrategy,
    EpisodicEvent,
    MemoryPolicy,
    MemoryRationale,
    MemorySummary,
    MemoryType,
    RetrievalOptions,
    SemanticFact,
)


class HybridMemory(Memory):
    """
    Hybrid memory system with:
    - Semantic memory (facts, entities, relationships)
    - Episodic memory (events with timestamps)
    - Summary memory (compressed summaries)
    - Safety & governance (PII detection, poisoning defense)
    - Retrieval rationale (explain why memories were retrieved)
    """

    def __init__(
        self,
        backend: Optional[MemoryBackend] = None,
        policy: Optional[MemoryPolicy] = None,
        working_buffer_size: int = 20,
        enable_pii_detection: bool = True,
        enable_poisoning_defense: bool = True,
    ):
        """
        Args:
            backend: Memory backend (defaults to InMemoryBackend)
            policy: Memory policy configuration
            working_buffer_size: Size of working memory buffer
            enable_pii_detection: Enable PII detection and scrubbing
            enable_poisoning_defense: Enable poisoning attack detection
        """
        self.backend = backend or InMemoryBackend()
        self.policy = policy or MemoryPolicy()
        self.working_buffer_size = working_buffer_size
        self.enable_pii_detection = enable_pii_detection
        self.enable_poisoning_defense = enable_poisoning_defense

        # Working memory buffer (current conversation)
        self._working_buffer: deque[ChatMessage] = deque(maxlen=working_buffer_size)

        # Event tracking
        self._events_since_summary = 0
        self._last_summary_time = datetime.now()

        # Retrieval cache
        self._retrieval_cache: Dict[str, tuple[List[Any], datetime]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

        # Simple embedding for semantic search (replace with proper embedding model if needed)
        self._embedding_dim = 128

    # ========== Core Memory Interface ==========

    def append(self, message: ChatMessage) -> None:
        """Add a message to memory and process it."""
        # PII scrubbing
        if self.enable_pii_detection:
            message = self._scrub_pii(message)

        # Poisoning defense check
        if self.enable_poisoning_defense:
            if self._detect_poisoning(message):
                # Log suspicious content but don't store dangerous patterns
                print(f"Warning: Potential poisoning detected in message, sanitizing...")
                message = self._sanitize_message(message)

        # Add to working buffer
        self._working_buffer.append(message)

        # Eventization: Convert message to episodic event
        event = self._message_to_event(message)
        self.backend.save_episodic_event(event)
        self._events_since_summary += 1

        # Extract semantic facts periodically
        if len(self._working_buffer) % 5 == 0:  # Every 5 messages
            self._extract_semantic_facts()

        # Create summaries periodically
        if (
            self._events_since_summary >= self.policy.summarize_every_n_events
            or (datetime.now() - self._last_summary_time).total_seconds() > 3600
        ):  # Or every hour
            self._create_summary()
            self._events_since_summary = 0
            self._last_summary_time = datetime.now()

    def get_context(self, query: str | None = None, k: int = 6) -> List[ChatMessage]:
        """Get context for a query using hybrid retrieval."""
        options = RetrievalOptions(
            query=query or "",
            max_results=k,
            include_rationale=False,  # Simplified for base interface
        )
        result = self.retrieve(options)
        rationale = result["rationale"]

        # Convert retrieved memories back to ChatMessages
        messages = []
        
        # Add working buffer first (most recent)
        messages.extend(list(self._working_buffer)[-k:])

        # Add retrieved memories
        for memory in rationale.retrieved_memories[:k]:
            if isinstance(memory, EpisodicEvent):
                messages.append(
                    ChatMessage(
                        role=memory.actor or "user",
                        content=memory.content,
                        metadata={"memory_type": "episodic", "event_id": memory.event_id},
                    )
                )
            elif isinstance(memory, SemanticFact):
                messages.append(
                    ChatMessage(
                        role="system",
                        content=f"Fact: {memory.fact}",
                        metadata={"memory_type": "semantic", "category": memory.category},
                    )
                )

        return messages[:k]

    def clear(self) -> None:
        """Clear working memory buffer."""
        self._working_buffer.clear()
        self._retrieval_cache.clear()

    # ========== Retrieval ==========

    def retrieve(self, options: RetrievalOptions) -> Dict[str, Any]:
        """
        Retrieve memories with full rationale.
        
        Returns:
            Dict with 'memories' and 'rationale' keys
        """
        # Check cache
        cache_key = self._get_cache_key(options)
        if options.use_cache and cache_key in self._retrieval_cache:
            cached_memories, cached_time = self._retrieval_cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return {
                    "memories": cached_memories,
                    "rationale": MemoryRationale(
                        retrieved_memories=cached_memories,
                        scores=[0.0] * len(cached_memories),
                        filters_applied=options.filters,
                        retrieval_method="cached",
                        candidate_count=len(cached_memories),
                        reranked=False,
                    ),
                }

        # Retrieve from different stores
        all_memories = []
        all_scores = []

        # Semantic retrieval
        if MemoryType.SEMANTIC in options.memory_types:
            facts = self.backend.get_semantic_facts(
                query=options.query,
                filters=options.filters,
                limit=options.max_results * 2,  # Get more for reranking
            )
            # Score facts by relevance (simple keyword matching + importance)
            for fact in facts:
                score = self._score_semantic_fact(fact, options.query)
                if score >= options.min_score:
                    all_memories.append(fact)
                    all_scores.append(score)

        # Episodic retrieval
        if MemoryType.EPISODIC in options.memory_types:
            start_time, end_time = options.time_range or (None, None)
            events = self.backend.get_episodic_events(
                start_time=start_time,
                end_time=end_time,
                topic=options.filters.get("topic") if options.filters else None,
                limit=options.max_results * 2,
            )
            # Score events by recency and relevance
            for event in events:
                score = self._score_episodic_event(event, options.query)
                if score >= options.min_score:
                    all_memories.append(event)
                    all_scores.append(score)

        # Summary retrieval
        if MemoryType.SUMMARY in options.memory_types:
            start_time, end_time = options.time_range or (None, None)
            summaries = self.backend.get_summaries(
                start_time=start_time,
                end_time=end_time,
                limit=5,
            )
            for summary in summaries:
                score = self._score_summary(summary, options.query)
                if score >= options.min_score:
                    all_memories.append(summary)
                    all_scores.append(score)

        # Rerank if enabled
        if options.rerank and len(all_memories) > 1:
            all_memories, all_scores = self._rerank_memories(
                all_memories, all_scores, options.query
            )

        # Take top-k
        top_indices = np.argsort(all_scores)[::-1][: options.max_results]
        retrieved = [all_memories[i] for i in top_indices]
        scores = [all_scores[i] for i in top_indices]

        # Update access times for facts
        for memory in retrieved:
            if isinstance(memory, SemanticFact):
                # Note: We'd need fact_id in SemanticFact for this
                # For now, skip
                pass

        # Create rationale
        rationale = MemoryRationale(
            retrieved_memories=retrieved,
            scores=scores,
            filters_applied=options.filters,
            retrieval_method="hybrid",
            candidate_count=len(all_memories),
            reranked=options.rerank,
        )

        # Cache result
        if options.use_cache:
            self._retrieval_cache[cache_key] = (retrieved, datetime.now())

        return {"memories": retrieved, "rationale": rationale}

    # ========== Memory Management ==========

    def remember(
        self,
        fact: str,
        category: str = "general",
        entities: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> str:
        """Explicitly remember a fact."""
        semantic_fact = SemanticFact(
            fact=fact,
            category=category,
            entities=entities or [],
            importance_score=importance,
        )
        return self.backend.save_semantic_fact(semantic_fact)

    def forget(self, memory_id: str, memory_type: str) -> bool:
        """Delete a specific memory."""
        return self.backend.delete_memory(memory_id, memory_type)

    def summarize(self, time_range: Optional[tuple[datetime, datetime]] = None) -> str:
        """Create a summary of memories in time range."""
        return self._create_summary(time_range)

    # ========== Internal Methods ==========

    def _message_to_event(self, message: ChatMessage) -> EpisodicEvent:
        """Convert a ChatMessage to an EpisodicEvent."""
        # Extract topic from content (simple keyword extraction)
        topic = self._extract_topic(message.content)
        
        return EpisodicEvent(
            event_id=str(uuid.uuid4()),
            content=message.content,
            actor=message.role,
            timestamp=datetime.now(),
            topic=topic,
            metadata=message.metadata.copy(),
        )

    def _extract_semantic_facts(self) -> None:
        """Extract semantic facts from recent messages."""
        # Simple extraction: look for factual statements
        recent_messages = list(self._working_buffer)[-10:]
        
        for msg in recent_messages:
            if msg.role == "user":
                # Simple heuristic: facts often contain "is", "has", "likes", etc.
                fact_patterns = [
                    r"I (?:am|like|love|hate|prefer|work|live)",
                    r"My (?:name|email|favorite|preferred)",
                    r"I'm (?:a|an|working|studying)",
                ]
                
                for pattern in fact_patterns:
                    matches = re.finditer(pattern, msg.content, re.IGNORECASE)
                    for match in matches:
                        # Extract sentence containing the match
                        sentence = self._extract_sentence(msg.content, match.start())
                        if sentence:
                            fact = SemanticFact(
                                fact=sentence,
                                category="user_info",
                                importance_score=0.6,
                            )
                            self.backend.save_semantic_fact(fact)

    def _extract_sentence(self, text: str, pos: int) -> Optional[str]:
        """Extract sentence containing position."""
        # Find sentence boundaries
        start = max(0, text.rfind(".", 0, pos))
        end = text.find(".", pos)
        if end == -1:
            end = len(text)
        
        sentence = text[start:end].strip()
        return sentence if len(sentence) > 10 else None

    def _create_summary(self, time_range: Optional[tuple[datetime, datetime]] = None) -> str:
        """Create a summary of events in time range."""
        if time_range is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
        else:
            start_time, end_time = time_range

        events = self.backend.get_episodic_events(
            start_time=start_time,
            end_time=end_time,
            limit=100,
        )

        if not events:
            return "No events to summarize."

        # Simple summarization: concatenate key events
        summary_text = f"Summary of {len(events)} events from {start_time} to {end_time}:\n"
        topics = {}
        for event in events[:10]:  # Top 10 events
            if event.topic:
                topics[event.topic] = topics.get(event.topic, 0) + 1
            summary_text += f"- {event.content[:100]}...\n"

        key_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = MemorySummary(
            summary_id=str(uuid.uuid4()),
            summary_text=summary_text,
            start_time=start_time,
            end_time=end_time,
            key_topics=[topic for topic, _ in key_topics],
            event_count=len(events),
            compression_ratio=len(summary_text) / sum(len(e.content) for e in events) if events else 0.0,
        )

        self.backend.save_summary(summary)
        return summary_text

    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract topic from text (simple keyword-based)."""
        # Simple topic extraction: use first noun phrase or key term
        words = re.findall(r"\b[A-Z][a-z]+\b|\b[a-z]{4,}\b", text.lower())
        if words:
            return words[0]
        return None

    def _score_semantic_fact(self, fact: SemanticFact, query: str) -> float:
        """Score a semantic fact for relevance to query."""
        query_lower = query.lower()
        fact_lower = fact.fact.lower()

        # Keyword match
        keyword_score = sum(1 for word in query_lower.split() if word in fact_lower) / max(len(query_lower.split()), 1)

        # Importance score
        importance_score = fact.importance_score

        # Recency (last accessed)
        recency_days = (datetime.now() - fact.last_accessed).days
        recency_score = max(0, 1.0 - recency_days / 30.0)  # Decay over 30 days

        # Combined score
        return (keyword_score * 0.5 + importance_score * 0.3 + recency_score * 0.2)

    def _score_episodic_event(self, event: EpisodicEvent, query: str) -> float:
        """Score an episodic event for relevance to query."""
        query_lower = query.lower()
        content_lower = event.content.lower()

        # Keyword match
        keyword_score = sum(1 for word in query_lower.split() if word in content_lower) / max(len(query_lower.split()), 1)

        # Recency
        age_days = (datetime.now() - event.timestamp).days
        recency_score = max(0, 1.0 - age_days / 7.0)  # Decay over 7 days

        return (keyword_score * 0.7 + recency_score * 0.3)

    def _score_summary(self, summary: MemorySummary, query: str) -> float:
        """Score a summary for relevance to query."""
        query_lower = query.lower()
        summary_lower = summary.summary_text.lower()

        # Check if query matches topics
        topic_match = any(topic.lower() in query_lower for topic in summary.key_topics)

        # Keyword match in summary text
        keyword_score = sum(1 for word in query_lower.split() if word in summary_lower) / max(len(query_lower.split()), 1)

        return (1.0 if topic_match else 0.0) * 0.5 + keyword_score * 0.5

    def _rerank_memories(
        self, memories: List[Any], scores: List[float], query: str
    ) -> tuple[List[Any], List[float]]:
        """Rerank memories using cross-encoder style scoring."""
        # Simple reranking: boost scores based on exact phrase matches
        reranked_memories = memories.copy()
        reranked_scores = scores.copy()

        query_phrases = query.lower().split()
        
        for i, memory in enumerate(reranked_memories):
            content = ""
            if isinstance(memory, SemanticFact):
                content = memory.fact.lower()
            elif isinstance(memory, EpisodicEvent):
                content = memory.content.lower()
            elif isinstance(memory, MemorySummary):
                content = memory.summary_text.lower()

            # Boost for phrase matches
            phrase_bonus = 0.0
            for phrase in query_phrases:
                if phrase in content:
                    phrase_bonus += 0.1

            reranked_scores[i] = min(1.0, reranked_scores[i] + phrase_bonus)

        # Sort by score
        sorted_indices = sorted(range(len(reranked_scores)), key=lambda i: reranked_scores[i], reverse=True)
        return (
            [reranked_memories[i] for i in sorted_indices],
            [reranked_scores[i] for i in sorted_indices],
        )

    def _get_cache_key(self, options: RetrievalOptions) -> str:
        """Generate cache key for retrieval options."""
        key_parts = [
            options.query,
            ",".join(sorted(mt.value for mt in options.memory_types)),
            str(options.max_results),
            str(options.filters),
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    # ========== Safety & Governance ==========

    def _scrub_pii(self, message: ChatMessage) -> ChatMessage:
        """Detect and scrub PII from message."""
        if not self.enable_pii_detection:
            return message

        content = message.content

        # Email pattern
        content = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", content)

        # Phone pattern (US format)
        content = re.sub(r"\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b", "[PHONE]", content)

        # SSN pattern
        content = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", content)

        # Credit card pattern (simplified)
        content = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]", content)

        if content != message.content:
            # Mark as having PII
            message.metadata["pii_detected"] = True
            message.metadata["original_length"] = len(message.content)

        return ChatMessage(
            role=message.role,
            content=content,
            metadata=message.metadata,
        )

    def _detect_poisoning(self, message: ChatMessage) -> bool:
        """Detect potential memory poisoning attacks."""
        if not self.enable_poisoning_defense:
            return False

        content = message.content.lower()

        # Check for suspicious patterns
        poisoning_patterns = [
            r"ignore previous instructions",
            r"forget everything",
            r"system prompt",
            r"assistant mode",
            r"new instructions:",
            r"important:",
        ]

        for pattern in poisoning_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _sanitize_message(self, message: ChatMessage) -> ChatMessage:
        """Sanitize message by removing suspicious patterns."""
        content = message.content

        # Remove suspicious patterns
        content = re.sub(r"(?i)(ignore previous instructions|forget everything)", "", content)
        content = re.sub(r"(?i)(system prompt|assistant mode)", "", content)

        return ChatMessage(
            role=message.role,
            content=content.strip(),
            metadata={**message.metadata, "sanitized": True},
        )
