"""
Pluggable backends for memory storage.

Backends progress from local-first to cloud:
- InMemoryBackend: Development/testing
- SQLiteBackend: Local storage
- DuckDBBackend: Analytics workloads
- PostgreSQLBackend: Multi-user storage
- RedisBackend: Fast ephemeral cache
- VectorBackend: Semantic search (pgvector, Chroma, etc.)
"""

from .base import MemoryBackend
from .in_memory import InMemoryBackend
from .sqlite_backend import SQLiteBackend

__all__ = [
    "MemoryBackend",
    "InMemoryBackend",
    "SQLiteBackend",
]
