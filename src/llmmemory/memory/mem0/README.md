# mem0 Integration

This folder contains integration with [mem0](https://github.com/mem0ai/mem0), a local-first memory system for LLMs.

## What is mem0?

mem0 is a graph-based memory system that:
- Stores memories as entities and relationships in a graph structure
- Uses hybrid storage (vector + graph) for efficient retrieval
- Provides local-first architecture for privacy
- Supports automatic memory extraction and organization

## Implementation

### mem0_adapter.py
An adapter that wraps mem0's memory system to implement the `Memory` protocol. This allows you to use mem0's sophisticated memory capabilities within the LLMmemory framework.

## Setup

To use mem0, you'll need to install it:
```bash
pip install mem0ai
```

Then configure mem0 with your preferred storage backend (SQLite, PostgreSQL, etc.) as per mem0's documentation.
