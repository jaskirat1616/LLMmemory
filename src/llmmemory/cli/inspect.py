#!/usr/bin/env python3
"""
Memory inspector CLI tool.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llmmemory.memory.hybrid import HybridMemory, MemoryInspector
from llmmemory.memory.hybrid.backends import InMemoryBackend, SQLiteBackend


def main():
    parser = argparse.ArgumentParser(description="Inspect memory system state")
    parser.add_argument(
        "--backend",
        choices=["memory", "sqlite"],
        default="memory",
        help="Backend type",
    )
    parser.add_argument("--db-path", help="SQLite database path (for sqlite backend)")
    parser.add_argument("--command", choices=["facts", "events", "summaries", "stats", "export"], required=True)
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--limit", type=int, default=50, help="Result limit")
    parser.add_argument("--output", choices=["json", "table"], default="table", help="Output format")

    args = parser.parse_args()

    # Initialize backend
    if args.backend == "sqlite":
        backend = SQLiteBackend(args.db_path or "memory.db")
    else:
        backend = InMemoryBackend()

    # Initialize memory system
    memory = HybridMemory(backend=backend)
    inspector = MemoryInspector(memory)

    # Execute command
    if args.command == "facts":
        result = inspector.inspect_semantic_facts(
            query=args.query,
            category=args.category,
            limit=args.limit,
        )
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"\nSemantic Facts ({result['count']}):")
            print("-" * 80)
            for fact in result["facts"]:
                print(f"  Fact: {fact['fact']}")
                print(f"  Category: {fact['category']}, Importance: {fact['importance']}")
                print(f"  Tags: {', '.join(fact['tags'])}")
                print()

    elif args.command == "events":
        result = inspector.inspect_episodic_events(limit=args.limit)
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"\nEpisodic Events ({result['count']}):")
            print("-" * 80)
            for event in result["events"]:
                print(f"  [{event['timestamp']}] {event['actor']}: {event['content']}")
                print()

    elif args.command == "summaries":
        result = inspector.inspect_summaries(limit=args.limit)
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"\nSummaries ({result['count']}):")
            print("-" * 80)
            for summary in result["summaries"]:
                print(f"  [{summary['start_time']} - {summary['end_time']}]")
                print(f"  Topics: {', '.join(summary['key_topics'])}")
                print(f"  Events: {summary['event_count']}, Compression: {summary['compression_ratio']:.2%}")
                print(f"  {summary['summary_text'][:200]}...")
                print()

    elif args.command == "stats":
        stats = inspector.get_stats()
        if args.output == "json":
            print(json.dumps(stats, indent=2, default=str))
        else:
            print("\nMemory System Statistics:")
            print("-" * 80)
            print(f"Backend: {stats['backend']}")
            print(f"Working Buffer: {stats['working_buffer_size']} messages")
            print(f"Cache Size: {stats['cache_size']}")
            print(f"Policy Namespace: {stats['policy']['namespace']}")
            print(f"Features: PII Detection={stats['features']['pii_detection']}, Poisoning Defense={stats['features']['poisoning_defense']}")

    elif args.command == "export":
        exported = inspector.export_memory(format="json")
        print(exported)

    # Cleanup
    if hasattr(backend, "close"):
        backend.close()


if __name__ == "__main__":
    main()
