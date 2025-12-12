# ChatGPT-Style Memory

This folder contains implementations inspired by ChatGPT's memory system (GPT-4 with memory feature).

## How ChatGPT Memory Works

ChatGPT's memory system combines two key components:

1. **Conversation Buffer**: Maintains a sliding window of recent messages for immediate context
2. **Entity/Fact Extraction**: Extracts and stores key facts about the user and conversation topics in a long-term knowledge base

When generating responses, ChatGPT:
- Uses the recent conversation buffer for immediate context
- Retrieves relevant long-term facts/entities based on the current conversation
- Combines both to provide contextually aware responses

## Implementations

### conversation_buffer.py
A sliding window conversation buffer similar to ChatGPT's short-term memory. Maintains the last N messages for immediate context.

### entity_memory.py
Extracts entities and facts from conversations and stores them long-term. When retrieving context, it combines:
- Recent conversation buffer
- Relevant extracted entities/facts based on the current query

This mimics ChatGPT's ability to remember user preferences, facts, and context across multiple conversations.
