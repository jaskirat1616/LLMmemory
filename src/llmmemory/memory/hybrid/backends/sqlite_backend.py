"""
SQLite backend for local storage.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types import EpisodicEvent, MemorySummary, SemanticFact
from .base import MemoryBackend


class SQLiteBackend(MemoryBackend):
    """SQLite-based backend for persistent local storage."""

    def __init__(self, db_path: str | Path = "memory.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Semantic facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_facts (
                id TEXT PRIMARY KEY,
                fact TEXT NOT NULL,
                entities TEXT,  -- JSON array
                relationships TEXT,  -- JSON object
                category TEXT,
                confidence REAL,
                source_message_id TEXT,
                timestamp TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                tags TEXT,  -- JSON array
                importance_score REAL,
                metadata TEXT  -- JSON object
            )
        """)

        # Episodic events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_events (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                actor TEXT,
                timestamp TEXT,
                topic TEXT,
                metadata TEXT,  -- JSON object
                related_events TEXT  -- JSON array
            )
        """)

        # Summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                summary_text TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                key_topics TEXT,  -- JSON array
                event_count INTEGER,
                compression_ratio REAL,
                metadata TEXT  -- JSON object
            )
        """)

        # Indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_timestamp ON semantic_facts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_category ON semantic_facts(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_topic ON episodic_events(topic)")

        conn.commit()

    def _serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()

    def _deserialize_datetime(self, s: str) -> datetime:
        """Deserialize ISO format string to datetime."""
        return datetime.fromisoformat(s)

    def save_semantic_fact(self, fact: SemanticFact) -> str:
        fact_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO semantic_facts 
            (id, fact, entities, relationships, category, confidence, source_message_id,
             timestamp, last_accessed, access_count, tags, importance_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact_id,
            fact.fact,
            json.dumps(fact.entities),
            json.dumps(fact.relationships),
            fact.category,
            fact.confidence,
            fact.source_message_id,
            self._serialize_datetime(fact.timestamp),
            self._serialize_datetime(fact.last_accessed),
            fact.access_count,
            json.dumps(fact.tags),
            fact.importance_score,
            json.dumps(fact.metadata),
        ))

        conn.commit()
        return fact_id

    def get_semantic_facts(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[SemanticFact]:
        conn = self._get_conn()
        cursor = conn.cursor()

        sql = "SELECT * FROM semantic_facts WHERE fact LIKE ?"
        params = [f"%{query}%"]

        if filters:
            if "category" in filters:
                sql += " AND category = ?"
                params.append(filters["category"])

        sql += " ORDER BY importance_score DESC, last_accessed DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        facts = []
        for row in rows:
            facts.append(SemanticFact(
                fact=row["fact"],
                entities=json.loads(row["entities"] or "[]"),
                relationships=json.loads(row["relationships"] or "{}"),
                category=row["category"],
                confidence=row["confidence"],
                source_message_id=row["source_message_id"],
                timestamp=self._deserialize_datetime(row["timestamp"]),
                last_accessed=self._deserialize_datetime(row["last_accessed"]),
                access_count=row["access_count"],
                tags=json.loads(row["tags"] or "[]"),
                importance_score=row["importance_score"],
                metadata=json.loads(row["metadata"] or "{}"),
            ))

        return facts

    def save_episodic_event(self, event: EpisodicEvent) -> str:
        event_id = event.event_id or str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO episodic_events
            (id, content, actor, timestamp, topic, metadata, related_events)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id,
            event.content,
            event.actor,
            self._serialize_datetime(event.timestamp),
            event.topic,
            json.dumps(event.metadata),
            json.dumps(event.related_events),
        ))

        conn.commit()
        return event_id

    def get_episodic_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[EpisodicEvent]:
        conn = self._get_conn()
        cursor = conn.cursor()

        sql = "SELECT * FROM episodic_events WHERE 1=1"
        params = []

        if start_time:
            sql += " AND timestamp >= ?"
            params.append(self._serialize_datetime(start_time))
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(self._serialize_datetime(end_time))
        if topic:
            sql += " AND topic = ?"
            params.append(topic)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            events.append(EpisodicEvent(
                event_id=row["id"],
                content=row["content"],
                actor=row["actor"],
                timestamp=self._deserialize_datetime(row["timestamp"]),
                topic=row["topic"],
                metadata=json.loads(row["metadata"] or "{}"),
                related_events=json.loads(row["related_events"] or "[]"),
            ))

        return events

    def save_summary(self, summary: MemorySummary) -> str:
        summary_id = summary.summary_id or str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO summaries
            (id, summary_text, start_time, end_time, key_topics, event_count, compression_ratio, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary_id,
            summary.summary_text,
            self._serialize_datetime(summary.start_time),
            self._serialize_datetime(summary.end_time),
            json.dumps(summary.key_topics),
            summary.event_count,
            summary.compression_ratio,
            json.dumps(summary.metadata),
        ))

        conn.commit()
        return summary_id

    def get_summaries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[MemorySummary]:
        conn = self._get_conn()
        cursor = conn.cursor()

        sql = "SELECT * FROM summaries WHERE 1=1"
        params = []

        if start_time:
            sql += " AND start_time >= ?"
            params.append(self._serialize_datetime(start_time))
        if end_time:
            sql += " AND end_time <= ?"
            params.append(self._serialize_datetime(end_time))

        sql += " ORDER BY end_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        summaries = []
        for row in rows:
            summaries.append(MemorySummary(
                summary_id=row["id"],
                summary_text=row["summary_text"],
                start_time=self._deserialize_datetime(row["start_time"]),
                end_time=self._deserialize_datetime(row["end_time"]),
                key_topics=json.loads(row["key_topics"] or "[]"),
                event_count=row["event_count"],
                compression_ratio=row["compression_ratio"],
                metadata=json.loads(row["metadata"] or "{}"),
            ))

        return summaries

    def delete_memory(self, memory_id: str, memory_type: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()

        table_map = {
            "semantic": "semantic_facts",
            "episodic": "episodic_events",
            "summary": "summaries",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        cursor.execute(f"DELETE FROM {table} WHERE id = ?", (memory_id,))
        conn.commit()
        return cursor.rowcount > 0

    def update_fact_access(self, fact_id: str, access_time: datetime) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE semantic_facts
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (self._serialize_datetime(access_time), fact_id))

        conn.commit()

    def apply_ttl(self, policy_ttl_days: Optional[int]) -> int:
        if policy_ttl_days is None:
            return 0

        cutoff = datetime.now() - timedelta(days=policy_ttl_days)
        cutoff_str = self._serialize_datetime(cutoff)
        conn = self._get_conn()
        cursor = conn.cursor()

        deleted = 0

        # Delete old facts
        cursor.execute("DELETE FROM semantic_facts WHERE timestamp < ?", (cutoff_str,))
        deleted += cursor.rowcount

        # Delete old events
        cursor.execute("DELETE FROM episodic_events WHERE timestamp < ?", (cutoff_str,))
        deleted += cursor.rowcount

        conn.commit()
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM semantic_facts")
        facts_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM episodic_events")
        events_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM summaries")
        summaries_count = cursor.fetchone()[0]

        return {
            "semantic_facts": facts_count,
            "episodic_events": events_count,
            "summaries": summaries_count,
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
