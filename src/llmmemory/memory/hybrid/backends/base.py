"""
Base interface for memory backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..types import EpisodicEvent, MemorySummary, SemanticFact


class MemoryBackend(ABC):
    """Base class for memory storage backends."""

    @abstractmethod
    def save_semantic_fact(self, fact: SemanticFact) -> str:
        """Save a semantic fact and return its ID."""
        pass

    @abstractmethod
    def get_semantic_facts(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[SemanticFact]:
        """Retrieve semantic facts matching query and filters."""
        pass

    @abstractmethod
    def save_episodic_event(self, event: EpisodicEvent) -> str:
        """Save an episodic event and return its ID."""
        pass

    @abstractmethod
    def get_episodic_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[EpisodicEvent]:
        """Retrieve episodic events in time range."""
        pass

    @abstractmethod
    def save_summary(self, summary: MemorySummary) -> str:
        """Save a memory summary and return its ID."""
        pass

    @abstractmethod
    def get_summaries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[MemorySummary]:
        """Retrieve summaries in time range."""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str, memory_type: str) -> bool:
        """Delete a memory by ID and type."""
        pass

    @abstractmethod
    def update_fact_access(self, fact_id: str, access_time: datetime) -> None:
        """Update last access time and count for a fact."""
        pass

    @abstractmethod
    def apply_ttl(self, policy_ttl_days: Optional[int]) -> int:
        """Apply TTL policy and return number of deleted items."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close backend connections."""
        pass
