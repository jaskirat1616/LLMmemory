# Custom Memory Implementations

This folder contains custom memory implementations that were part of the original LLMmemory project.

## Implementations

### BufferMemory
A simple sliding window buffer that maintains the last N messages. This is the most basic memory approach - it keeps recent conversation history but has no long-term memory capabilities.

**Use case:** Short conversations where recent context is sufficient.

### SummaryMemory
Maintains a running summary of the conversation plus a short buffer of recent messages. Periodically summarizes older messages to compress history while preserving key information.

**Use case:** Longer conversations where you want to preserve context without maintaining full history.

### VectorMemory
Uses vector similarity search to retrieve relevant messages based on the current query. Uses lightweight hashed bag-of-words embeddings (can be upgraded to real embeddings later).

**Use case:** When you need to retrieve relevant past context based on semantic similarity to the current question.
