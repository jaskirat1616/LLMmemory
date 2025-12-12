"""
In-memory backend for development and testing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..types import EpisodicEvent, MemorySummary, SemanticFact
from .base import MemoryBackend


class InMemoryBackend(MemoryBackend):
    """Simple in-memory backend using Python dicts/lists."""

    def __init__(self):
        self._semantic_facts: Dict[str, SemanticFact] = {}
        self._episodic_events: Dict[str, EpisodicEvent] = {}
        self._summaries: Dict[str, MemorySummary] = {}
        self._event_time_index: List[tuple[datetime, str]] = []  # For time-range queries

    def save_semantic_fact(self, fact: SemanticFact) -> str:
        fact_id = str(uuid.uuid4())
        self._semantic_facts[fact_id] = fact
        return fact_id

    def get_semantic_facts(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[SemanticFact]:
        # Simple keyword matching for in-memory backend
        query_lower = query.lower()
        results = []
        
        for fact in self._semantic_facts.values():
            # Apply filters
            if filters:
                if "category" in filters and fact.category != filters["category"]:
                    continue
                if "tags" in filters:
                    required_tags = filters["tags"]
                    if not all(tag in fact.tags for tag in required_tags):
                        continue
            
            # Simple text matching
            if query_lower in fact.fact.lower():
                results.append(fact)
        
        return results[:limit]

    def save_episodic_event(self, event: EpisodicEvent) -> str:
        event_id = event.event_id or str(uuid.uuid4())
        self._episodic_events[event_id] = event
        self._event_time_index.append((event.timestamp, event_id))
        self._event_time_index.sort(key=lambda x: x[0])  # Keep sorted by time
        return event_id

    def get_episodic_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[EpisodicEvent]:
        results = []
        
        for timestamp, event_id in self._event_time_index:
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            
            event = self._episodic_events[event_id]
            if topic and event.topic != topic:
                continue
            
            results.append(event)
            if len(results) >= limit:
                break
        
        return results

    def save_summary(self, summary: MemorySummary) -> str:
        summary_id = summary.summary_id or str(uuid.uuid4())
        self._summaries[summary_id] = summary
        return summary_id

    def get_summaries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[MemorySummary]:
        results = []
        for summary in self._summaries.values():
            if start_time and summary.start_time < start_time:
                continue
            if end_time and summary.end_time > end_time:
                continue
            results.append(summary)
        
        # Sort by end_time descending
        results.sort(key=lambda s: s.end_time, reverse=True)
        return results[:limit]

    def delete_memory(self, memory_id: str, memory_type: str) -> bool:
        if memory_type == "semantic":
            return self._semantic_facts.pop(memory_id, None) is not None
        elif memory_type == "episodic":
            # Also remove from time index
            self._event_time_index = [
                (t, eid) for t, eid in self._event_time_index if eid != memory_id
            ]
            return self._episodic_events.pop(memory_id, None) is not None
        elif memory_type == "summary":
            return self._summaries.pop(memory_id, None) is not None
        return False

    def update_fact_access(self, fact_id: str, access_time: datetime) -> None:
        if fact_id in self._semantic_facts:
            fact = self._semantic_facts[fact_id]
            fact.last_accessed = access_time
            fact.access_count += 1

    def apply_ttl(self, policy_ttl_days: Optional[int]) -> int:
        if policy_ttl_days is None:
            return 0
        
        cutoff = datetime.now() - timedelta(days=policy_ttl_days)
        deleted = 0
        
        # Delete old facts
        to_delete = [
            fid for fid, fact in self._semantic_facts.items()
            if fact.timestamp < cutoff
        ]
        for fid in to_delete:
            del self._semantic_facts[fid]
            deleted += len(to_delete)
        
        # Delete old events
        to_delete_events = [
            eid for eid, event in self._episodic_events.items()
            if event.timestamp < cutoff
        ]
        for eid in to_delete_events:
            del self._episodic_events[eid]
            # Remove from time index
            self._event_time_index = [
                (t, eid) for t, eid in self._event_time_index if eid != eid
            ]
            deleted += 1
        
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        return {
            "semantic_facts": len(self._semantic_facts),
            "episodic_events": len(self._episodic_events),
            "summaries": len(self._summaries),
        }

    def close(self) -> None:
        # Nothing to close for in-memory backend
        pass
