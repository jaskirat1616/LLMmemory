"""
Event extraction and temporal structuring for episodic memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import ChatMessage
from .types import EpisodicEvent


@dataclass
class ExtractedEvent:
    """An extracted event with metadata."""

    event_type: str  # e.g., "user_action", "system_response", "task_completion"
    content: str
    actor: Optional[str] = None
    timestamp: datetime = None
    entities: List[str] = None
    sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    metadata: Dict[str, Any] = None


class EventExtractor:
    """Extract structured events from messages."""

    def __init__(self):
        # Event type patterns
        self.event_patterns = {
            "user_action": [
                r"I (?:want|need|would like|prefer)",
                r"Can you (?:help|do|show)",
                r"Please (?:create|make|add|remove)",
            ],
            "preference": [
                r"I (?:like|love|hate|prefer|don't like)",
                r"My favorite",
                r"I always",
            ],
            "fact_statement": [
                r"I am|I'm",
                r"My name is",
                r"I work (?:at|for)",
            ],
            "task_completion": [
                r"done|completed|finished",
                r"thank you|thanks",
            ],
        }

    def extract_from_message(self, message: ChatMessage) -> List[ExtractedEvent]:
        """Extract events from a single message."""
        events = []
        content = message.content

        # Determine event type
        event_type = self._classify_event_type(content)
        
        # Extract entities
        entities = self._extract_entities(content)

        # Determine sentiment
        sentiment = self._detect_sentiment(content)

        event = ExtractedEvent(
            event_type=event_type,
            content=content,
            actor=message.role,
            timestamp=datetime.now(),
            entities=entities,
            sentiment=sentiment,
            metadata=message.metadata.copy(),
        )

        events.append(event)
        return events

    def _classify_event_type(self, content: str) -> str:
        """Classify the type of event based on content patterns."""
        content_lower = content.lower()

        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return event_type

        return "general"

    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content (simple heuristic)."""
        entities = []

        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", content)
        entities.extend(capitalized[:5])  # Limit to top 5

        # Email addresses
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", content)
        entities.extend(emails)

        return list(set(entities))  # Deduplicate

    def _detect_sentiment(self, content: str) -> Optional[str]:
        """Simple sentiment detection."""
        content_lower = content.lower()

        positive_words = ["good", "great", "excellent", "love", "like", "thank", "thanks", "perfect"]
        negative_words = ["bad", "terrible", "hate", "wrong", "error", "fail", "broken"]

        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def link_events(self, events: List[EpisodicEvent]) -> List[EpisodicEvent]:
        """Link related events together."""
        # Simple linking: events with same topic or actor within time window
        linked_events = []

        for i, event in enumerate(events):
            related_ids = []

            # Find related events (same topic within 1 hour)
            for j, other_event in enumerate(events):
                if i == j:
                    continue

                time_diff = abs((event.timestamp - other_event.timestamp).total_seconds())
                if (
                    time_diff < 3600  # Within 1 hour
                    and event.topic
                    and event.topic == other_event.topic
                ):
                    related_ids.append(other_event.event_id)

            event.related_events = related_ids
            linked_events.append(event)

        return linked_events
