# Claude-Style Memory

This folder contains implementations inspired by Claude's memory system (Anthropic's approach with Projects and Artifacts).

## How Claude Memory Works

Claude's memory system is organized around:

1. **Project-Based Context**: Conversations are organized into projects, each with attached documents, knowledge bases, and context
2. **Knowledge Base**: Long-form knowledge storage that can be attached to conversations
3. **Artifacts**: Tracks and versions generated content, allowing Claude to reference and build upon previous outputs

When generating responses, Claude:
- Uses project-scoped context to understand the conversation domain
- References attached knowledge bases for factual information
- Can recall and build upon previously generated artifacts

## Implementations

### project_memory.py
Organizes conversations into projects with associated context. Each project maintains its own conversation history and can have documents/knowledge attached.

### knowledge_base.py
Manages long-form knowledge storage that can be attached to conversations. Supports document storage, retrieval, and contextual recall based on project scope.
