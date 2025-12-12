
<p align="center">
  <img src="./logo.png" alt="LLMmemory logo" width="220" />
</p>

# LLMmemory

Local playground for experimenting with chat-style memory systems (buffer, summary, vector-style recall, etc.) powered by an MLX-hosted model. Current default: `mlx-community/Qwen3-VL-8B-Instruct-4bit` (MLX VLM build) [HF card](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit).

## Prerequisites
- macOS with Apple Silicon
- Python 3.11+ (recommended)
- Enough disk/RAM for the chosen model (8B class is several GB on disk)

## Quickstart
```bash
cd /Users/jaskiratsingh/Desktop/LLMmemory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export MODEL_ID="mlx-community/Qwen3-VL-8B-Instruct-4bit"
export MODEL_KIND="vlm"  # "auto", "vlm", or "lm"
# make sure the package is importable
export PYTHONPATH="src:${PYTHONPATH}"
```

Download the model weights ahead of time if you prefer:
```bash
huggingface-cli download "$MODEL_ID"
```

## Running a chat UI
- Streamlit: `streamlit run src/llmmemory/apps/streamlit_app.py`
- Gradio: `python -m llmmemory.apps.gradio_app`

Both UIs let you switch memory backends at runtime.

## Project layout
- `src/llmmemory/` — Python package
  - `memory/` — pluggable memory strategies (buffer, summary, vector-ish)
  - `models/` — MLX model loader + chat helper
  - `chat/` — session orchestration for model + memory
  - `apps/` — Streamlit and Gradio entrypoints
- `tests/` — slots for unit/integration tests
- `configs/` — place to store preset configs or prompt templates

## Notes on the model
- Default `MODEL_ID` targets the 4-bit Qwen3-VL Instruct MLX build. If you switch to a text-only MLX model, set `MODEL_KIND=lm`.
- The loader uses `trust_remote_code=True` because some MLX chat models ship custom tokenizers/processors; only point it at repos you trust.

## Hybrid Memory System

Hybrid memory system with multiple storage types.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Memory System                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │      Message Input (ChatMessage)     │
        └──────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────┐                        ┌───────────────┐
│ PII Detection │                        │Poisoning      │
│ & Scrubbing   │                        │Defense        │
└───────────────┘                        └───────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ▼
        ┌──────────────────────────────────────┐
        │      Event Extraction Pipeline        │
        └──────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────┐
│  Working Buffer  │                    │ Episodic Events  │
│  (Current Chat)  │                    │  (What/When)     │
└──────────────────┘                    └──────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │      Fact Extraction                 │
        └──────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Semantic Facts   │
                    │ (Facts/Entities) │
                    └──────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │      Summarization (Periodic)        │
        └──────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Memory Summaries │
                    │ (Compressed)     │
                    └──────────────────┘
```

### Memory Stores

```
┌──────────────────────────────────────────────────────────────┐
│                    Hybrid Memory Stores                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Semantic    │  │  Episodic    │  │   Summary    │      │
│  │  Memory      │  │  Memory      │  │   Memory     │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ • Facts      │  │ • Events     │  │ • Summaries  │      │
│  │ • Entities   │  │ • Timestamps │  │ • Topics     │      │
│  │ • Relations  │  │ • Topics     │  │ • Compressed │      │
│  │ • Categories │  │ • Actors     │  │ • Time-based │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Retrieval Pipeline

```
Query
  │
  ▼
┌─────────────────┐
│ Retrieve from   │
│ Multiple Stores │
└────────┬────────┘
         │
    ┌────┴────┬──────────────┬──────────────┐
    │         │              │              │
    ▼         ▼              ▼              ▼
┌────────┐ ┌────────┐  ┌────────┐  ┌────────┐
│Semantic│ │Episodic│  │Summary │  │Working │
│ Facts  │ │ Events │  │Summaries│ │ Buffer │
└───┬────┘ └───┬────┘  └───┬────┘  └───┬────┘
    │          │            │            │
    └──────────┴────────────┴────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  Score & Rerank  │
         └─────────┬────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   Top-K Results  │
         │   + Rationale    │
         └──────────────────┘
```

### Backend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridMemory                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  MemoryBackend   │
                  │   (Interface)    │
                  └────────┬─────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  InMemory    │  │   SQLite     │  │  PostgreSQL  │
│  Backend     │  │   Backend    │  │  Backend     │
│              │  │              │  │  (future)    │
│ • Dict-based │  │ • File-based │  │              │
│ • Fast       │  │ • Persistent │  │              │
│ • Testing    │  │ • Local      │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Quick Start

```python
from llmmemory.memory.hybrid import HybridMemory
from llmmemory.memory.hybrid.backends import SQLiteBackend
from llmmemory.memory.base import ChatMessage

# Create memory system
memory = HybridMemory(backend=SQLiteBackend("memory.db"))

# Add messages
memory.append(ChatMessage(role="user", content="My name is Alice"))
memory.append(ChatMessage(role="user", content="I love Python"))

# Retrieve memories
from llmmemory.memory.hybrid.types import RetrievalOptions
result = memory.retrieve(RetrievalOptions(query="What does the user like?", max_results=5))
```

### Data Flow

```
User Message
     │
     ├─► PII Detection & Scrubbing
     ├─► Poisoning Defense
     │
     ├─► Working Buffer (current conversation)
     ├─► Event Extraction → Episodic Memory
     ├─► Fact Extraction → Semantic Memory
     │
     └─► Periodic Summarization → Summary Memory
```

### Features

✅ **Hybrid Memory Architecture**
- Semantic memory (facts, entities, relationships)
- Episodic memory (events with temporal context)
- Summary memory (compressed summaries)

✅ **Pluggable Backends**
- InMemoryBackend (for testing)
- SQLiteBackend (local storage)
- Easy to extend with PostgreSQL, Redis, vector DBs

✅ **Safety Features**
- PII detection and scrubbing
- Protection against prompt injection attacks
- Audit logging

✅ **Inspection Tools**
- CLI tool to view memory
- See why memories were retrieved
- Statistics and debugging

✅ **Testing Tools**
- Accuracy tests
- Performance benchmarks
- Safety tests

### CLI Tools

```bash
# Inspect memory state
python3 -m llmmemory.cli.inspect --command stats

# View semantic facts
python3 -m llmmemory.cli.inspect --command facts --query "preference"

# Export memory
python3 -m llmmemory.cli.inspect --command export --output json
```


