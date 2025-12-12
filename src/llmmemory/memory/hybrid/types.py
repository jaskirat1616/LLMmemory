"""
Type definitions for production memory system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..base import ChatMessage


class MemoryType(str, Enum):
    """Types of memory stores."""

    SEMANTIC = "semantic"  # Facts, entities, relationships
    EPISODIC = "episodic"  # Events with timestamps
    SUMMARY = "summary"  # Compressed summaries
    WORKING = "working"  # Current conversation buffer


class DecayStrategy(str, Enum):
    """Memory decay strategies."""

    TIME_BASED = "time_based"  # Decay based on age
    RELEVANCE_BASED = "relevance_based"  # Decay based on access frequency
    HYBRID = "hybrid"  # Combination of both


@dataclass
class SemanticFact:
    """A semantic fact stored in long-term memory."""

    fact: str
    entities: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    category: str = "general"
    confidence: float = 1.0
    source_message_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)  # e.g., ["pii", "preference"]
    importance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodicEvent:
    """An episodic event with temporal context."""

    event_id: str
    content: str
    actor: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)


@dataclass
class MemorySummary:
    """A condensed summary of past interactions."""

    summary_id: str
    summary_text: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    key_topics: List[str] = field(default_factory=list)
    event_count: int = 0
    compression_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPolicy:
    """Policy configuration for memory management."""

    # TTL settings
    ttl_days: Optional[int] = None  # None = no expiration
    ttl_by_category: Dict[str, int] = field(default_factory=dict)

    # Decay settings
    decay_strategy: DecayStrategy = DecayStrategy.HYBRID
    decay_rate: float = 0.1  # Per time unit
    min_importance: float = 0.1  # Below this, memory is pruned

    # Privacy settings
    detect_pii: bool = True
    auto_anonymize: bool = False
    require_consent: bool = False

    # Scope settings
    namespace: str = "default"
    role_filter: Optional[List[str]] = None  # Restrict to certain roles
    tenant_id: Optional[str] = None

    # Compression settings
    max_facts: int = 1000
    max_events: int = 10000
    summarize_every_n_events: int = 100
    target_compression_ratio: float = 0.2  # 80% compression


@dataclass
class RetrievalOptions:
    """Options for memory retrieval."""

    query: str
    memory_types: List[MemoryType] = field(
        default_factory=lambda: [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.SUMMARY]
    )
    max_results: int = 10
    min_score: float = 0.0
    time_range: Optional[tuple[Optional[datetime], Optional[datetime]]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    include_rationale: bool = True
    rerank: bool = True
    use_cache: bool = True


@dataclass
class MemoryRationale:
    """Explanation of why certain memories were retrieved."""

    retrieved_memories: List[Any]
    scores: List[float]
    filters_applied: Dict[str, Any]
    retrieval_method: str
    candidate_count: int
    reranked: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
