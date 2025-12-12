"""
Observability and debugging tools for memory system.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import MemoryRationale, SemanticFact


@dataclass
class MemoryInspector:
    """Inspector for viewing and debugging memory state."""

    def __init__(self, memory_system: Any):
        """
        Args:
            memory_system: The HybridMemory instance to inspect
        """
        self.memory = memory_system

    def inspect_semantic_facts(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Inspect stored semantic facts."""
        if query or category:
            filters = {}
            if category:
                filters["category"] = category
            facts = self.memory.backend.get_semantic_facts(
                query=query or "",
                filters=filters,
                limit=limit,
            )
        else:
            # Get all facts (limited)
            facts = self.memory.backend.get_semantic_facts(query="", limit=limit)

        return {
            "count": len(facts),
            "facts": [
                {
                    "fact": f.fact,
                    "category": f.category,
                    "importance": f.importance_score,
                    "confidence": f.confidence,
                    "entities": f.entities,
                    "tags": f.tags,
                    "timestamp": f.timestamp.isoformat(),
                    "last_accessed": f.last_accessed.isoformat(),
                    "access_count": f.access_count,
                }
                for f in facts
            ],
        }

    def inspect_episodic_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Inspect episodic events."""
        events = self.memory.backend.get_episodic_events(
            start_time=start_time,
            end_time=end_time,
            topic=topic,
            limit=limit,
        )

        return {
            "count": len(events),
            "events": [
                {
                    "event_id": e.event_id,
                    "content": e.content[:100] + "..." if len(e.content) > 100 else e.content,
                    "actor": e.actor,
                    "topic": e.topic,
                    "timestamp": e.timestamp.isoformat(),
                    "related_events": e.related_events,
                }
                for e in events
            ],
        }

    def inspect_summaries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Inspect memory summaries."""
        summaries = self.memory.backend.get_summaries(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        return {
            "count": len(summaries),
            "summaries": [
                {
                    "summary_id": s.summary_id,
                    "summary_text": s.summary_text[:200] + "..." if len(s.summary_text) > 200 else s.summary_text,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat(),
                    "key_topics": s.key_topics,
                    "event_count": s.event_count,
                    "compression_ratio": s.compression_ratio,
                }
                for s in summaries
            ],
        }

    def explain_retrieval(self, rationale: MemoryRationale) -> Dict[str, Any]:
        """Explain why certain memories were retrieved."""
        return {
            "retrieved_count": len(rationale.retrieved_memories),
            "candidate_count": rationale.candidate_count,
            "retrieval_method": rationale.retrieval_method,
            "reranked": rationale.reranked,
            "filters_applied": rationale.filters_applied,
            "scores": rationale.scores,
            "memories": [
                {
                    "type": type(m).__name__,
                    "content": self._get_memory_content(m)[:100],
                    "score": rationale.scores[i] if i < len(rationale.scores) else 0.0,
                }
                for i, m in enumerate(rationale.retrieved_memories)
            ],
            "timestamp": rationale.timestamp.isoformat(),
        }

    def _get_memory_content(self, memory: Any) -> str:
        """Extract content string from memory object."""
        if isinstance(memory, SemanticFact):
            return memory.fact
        elif hasattr(memory, "content"):
            return memory.content
        elif hasattr(memory, "summary_text"):
            return memory.summary_text
        return str(memory)

    def get_stats(self) -> Dict[str, Any]:
        """Get overall memory system statistics."""
        backend_stats = self.memory.backend.get_stats()

        return {
            "backend": backend_stats,
            "working_buffer_size": len(self.memory._working_buffer),
            "cache_size": len(self.memory._retrieval_cache),
            "policy": {
                "namespace": self.memory.policy.namespace,
                "decay_strategy": self.memory.policy.decay_strategy.value,
                "max_facts": self.memory.policy.max_facts,
                "max_events": self.memory.policy.max_events,
            },
            "features": {
                "pii_detection": self.memory.enable_pii_detection,
                "poisoning_defense": self.memory.enable_poisoning_defense,
            },
        }

    def export_memory(self, format: str = "json") -> str:
        """Export all memory in a structured format."""
        facts = self.memory.backend.get_semantic_facts(query="", limit=1000)
        events = self.memory.backend.get_episodic_events(limit=1000)
        summaries = self.memory.backend.get_summaries(limit=100)

        export_data = {
            "semantic_facts": [
                {
                    "fact": f.fact,
                    "category": f.category,
                    "entities": f.entities,
                    "timestamp": f.timestamp.isoformat(),
                }
                for f in facts
            ],
            "episodic_events": [
                {
                    "event_id": e.event_id,
                    "content": e.content,
                    "actor": e.actor,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in events
            ],
            "summaries": [
                {
                    "summary_id": s.summary_id,
                    "summary_text": s.summary_text,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat(),
                }
                for s in summaries
            ],
            "export_timestamp": datetime.now().isoformat(),
        }

        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
