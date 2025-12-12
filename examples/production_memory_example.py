#!/usr/bin/env python3
"""
Example usage of the hybrid memory system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmmemory.memory.base import ChatMessage
from llmmemory.memory.hybrid import HybridMemory, MemoryInspector, MemoryPolicy
from llmmemory.memory.hybrid.backends import InMemoryBackend, SQLiteBackend
from llmmemory.memory.hybrid.types import MemoryType, RetrievalOptions


def example_basic_usage():
    """Basic usage example."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Create memory system with in-memory backend
    memory = HybridMemory(backend=InMemoryBackend())

    # Add some messages
    memory.append(ChatMessage(role="user", content="My name is Alice"))
    memory.append(ChatMessage(role="user", content="I like Python programming"))
    memory.append(ChatMessage(role="user", content="I work at Tech Corp"))
    memory.append(ChatMessage(role="assistant", content="Nice to meet you, Alice!"))

    # Retrieve context
    context = memory.get_context(query="What do I like?")
    print(f"\nRetrieved {len(context)} messages:")
    for msg in context:
        print(f"  [{msg.role}]: {msg.content[:80]}")

    print()


def example_sqlite_backend():
    """Example with SQLite backend."""
    print("=" * 80)
    print("Example 2: SQLite Backend")
    print("=" * 80)

    # Create memory with SQLite backend
    backend = SQLiteBackend(db_path="example_memory.db")
    memory = HybridMemory(backend=backend)

    # Add messages
    memory.append(ChatMessage(role="user", content="I'm learning TypeScript"))
    memory.append(ChatMessage(role="user", content="My favorite editor is VS Code"))
    memory.append(ChatMessage(role="assistant", content="TypeScript is great!"))

    # Explicitly remember a fact
    fact_id = memory.remember(
        fact="User prefers VS Code for TypeScript development",
        category="preference",
        importance=0.8,
    )
    print(f"\nStored fact with ID: {fact_id}")

    # Retrieve with options
    options = RetrievalOptions(
        query="What editor does the user prefer?",
        memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        max_results=5,
        include_rationale=True,
    )
    result = memory.retrieve(options)
    rationale = result["rationale"]

    print(f"\nRetrieved {len(result['memories'])} memories:")
    print(f"  Method: {rationale.retrieval_method}")
    print(f"  Candidates: {rationale.candidate_count}")
    print(f"  Reranked: {rationale.reranked}")

    for i, mem in enumerate(result["memories"]):
        score = rationale.scores[i] if i < len(rationale.scores) else 0.0
        if hasattr(mem, "fact"):
            print(f"  [{score:.2f}] Fact: {mem.fact}")
        elif hasattr(mem, "content"):
            print(f"  [{score:.2f}] Event: {mem.content[:60]}")

    # Cleanup
    backend.close()
    print()


def example_memory_inspector():
    """Example using the memory inspector."""
    print("=" * 80)
    print("Example 3: Memory Inspector")
    print("=" * 80)

    memory = HybridMemory(backend=InMemoryBackend())

    # Add various messages
    memory.append(ChatMessage(role="user", content="I love hiking"))
    memory.append(ChatMessage(role="user", content="I work as a data scientist"))
    memory.append(ChatMessage(role="user", content="Python is my favorite language"))

    # Use inspector
    inspector = MemoryInspector(memory)

    # Inspect semantic facts
    facts_data = inspector.inspect_semantic_facts(limit=10)
    print(f"\nSemantic Facts ({facts_data['count']}):")
    for fact in facts_data["facts"]:
        print(f"  - {fact['fact']} (Category: {fact['category']}, Importance: {fact['importance']})")

    # Get stats
    stats = inspector.get_stats()
    print(f"\nSystem Statistics:")
    print(f"  Backend: {stats['backend']}")
    print(f"  Working Buffer: {stats['working_buffer_size']} messages")
    print()

    # Export memory
    exported = inspector.export_memory(format="json")
    print(f"Exported memory (first 200 chars): {exported[:200]}...")
    print()


def example_pii_detection():
    """Example demonstrating PII detection and scrubbing."""
    print("=" * 80)
    print("Example 4: PII Detection")
    print("=" * 80)

    memory = HybridMemory(backend=InMemoryBackend(), enable_pii_detection=True)

    # Add message with PII
    message_with_pii = ChatMessage(
        role="user",
        content="My email is alice@example.com and my phone is 555-123-4567",
    )
    
    print(f"Original message: {message_with_pii.content}")
    
    # Append will automatically scrub PII
    memory.append(message_with_pii)
    
    # Check what was stored
    context = memory.get_context(query="email")
    print(f"\nAfter PII scrubbing, retrieved:")
    for msg in context:
        print(f"  {msg.content}")
        if msg.metadata.get("pii_detected"):
            print(f"    [PII detected: {msg.metadata.get('pii_types', [])}]")
    print()


def example_policy_configuration():
    """Example with custom memory policy."""
    print("=" * 80)
    print("Example 5: Custom Policy")
    print("=" * 80)

    # Create custom policy
    policy = MemoryPolicy(
        ttl_days=30,  # Delete memories older than 30 days
        decay_strategy="hybrid",
        max_facts=500,
        max_events=5000,
        namespace="default",
        detect_pii=True,
    )

    memory = HybridMemory(backend=InMemoryBackend(), policy=policy)

    print(f"Memory policy configured:")
    print(f"  TTL: {policy.ttl_days} days")
    print(f"  Max facts: {policy.max_facts}")
    print(f"  Namespace: {policy.namespace}")
    print(f"  PII detection: {policy.detect_pii}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Hybrid Memory System Examples")
    print("=" * 80 + "\n")

    example_basic_usage()
    example_sqlite_backend()
    example_memory_inspector()
    example_pii_detection()
    example_policy_configuration()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
