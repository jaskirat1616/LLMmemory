"""
Evaluation harness for memory system performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..base import ChatMessage
from .hybrid_memory import HybridMemory
from .types import RetrievalOptions


@dataclass
class EvaluationResult:
    """Result of a memory system evaluation."""

    test_name: str
    accuracy: float
    latency_ms: float
    token_count: int
    cost_estimate: float
    details: Dict[str, Any]


class MemoryEvaluator:
    """Evaluator for memory system performance."""

    def __init__(self, memory: HybridMemory):
        self.memory = memory

    def evaluate_retrieval_accuracy(
        self,
        test_cases: List[Dict[str, Any]],
        expected_facts: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate retrieval accuracy on test cases.
        
        Args:
            test_cases: List of dicts with 'query' and expected 'fact_ids' or 'facts'
            expected_facts: Optional list of fact strings that should be retrieved
        """
        correct = 0
        total = len(test_cases)
        latencies = []
        total_tokens = 0

        for test_case in test_cases:
            query = test_case["query"]
            expected = test_case.get("expected_facts", []) or expected_facts or []

            start_time = time.time()
            options = RetrievalOptions(
                query=query,
                max_results=10,
                include_rationale=True,
            )
            result = self.memory.retrieve(options)
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)

            # Check if expected facts were retrieved
            retrieved_facts = [
                m.fact if hasattr(m, "fact") else str(m)
                for m in result["memories"]
                if hasattr(m, "fact")
            ]

            # Simple matching: check if any expected fact appears in retrieved
            if expected:
                matches = sum(
                    1
                    for exp in expected
                    if any(exp.lower() in ret.lower() for ret in retrieved_facts)
                )
                if matches > 0:
                    correct += 1
            else:
                # If no expected facts, just check that something was retrieved
                if retrieved_facts:
                    correct += 1

            # Estimate tokens (rough)
            total_tokens += len(query.split()) + sum(len(f.split()) for f in retrieved_facts)

        accuracy = correct / total if total > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Rough cost estimate (assuming $0.01 per 1K tokens)
        cost_estimate = (total_tokens / 1000) * 0.01

        return EvaluationResult(
            test_name="retrieval_accuracy",
            accuracy=accuracy,
            latency_ms=avg_latency,
            token_count=total_tokens,
            cost_estimate=cost_estimate,
            details={
                "correct": correct,
                "total": total,
                "latencies": latencies,
            },
        )

    def evaluate_temporal_reasoning(
        self,
        events: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
    ) -> EvaluationResult:
        """
        Evaluate temporal reasoning capabilities.
        
        Args:
            events: List of events with 'content', 'timestamp', 'actor'
            queries: List of queries with 'query', 'expected_event_time' or 'expected_order'
        """
        # Load events into memory
        for event_data in events:
            message = ChatMessage(
                role=event_data.get("actor", "user"),
                content=event_data["content"],
                metadata={"timestamp": event_data["timestamp"]},
            )
            self.memory.append(message)

        correct = 0
        total = len(queries)
        latencies = []

        for query_data in queries:
            query = query_data["query"]
            expected_time = query_data.get("expected_event_time")

            start_time = time.time()
            options = RetrievalOptions(
                query=query,
                max_results=5,
                memory_types=["episodic"],
            )
            result = self.memory.retrieve(options)
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)

            # Check if retrieved events match expected time
            if expected_time and result["memories"]:
                # Simple check: see if any retrieved event is close to expected time
                retrieved_times = [
                    getattr(m, "timestamp", None) for m in result["memories"] if hasattr(m, "timestamp")
                ]
                if retrieved_times:
                    # Check if any time is within reasonable range
                    time_match = any(
                        abs((t - expected_time).total_seconds()) < 3600
                        for t in retrieved_times
                        if t
                    )
                    if time_match:
                        correct += 1

        accuracy = correct / total if total > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return EvaluationResult(
            test_name="temporal_reasoning",
            accuracy=accuracy,
            latency_ms=avg_latency,
            token_count=0,
            cost_estimate=0.0,
            details={
                "correct": correct,
                "total": total,
            },
        )

    def evaluate_poisoning_defense(
        self,
        poisoning_attempts: List[str],
        normal_messages: List[str],
    ) -> EvaluationResult:
        """
        Evaluate poisoning attack detection.
        
        Args:
            poisoning_attempts: List of messages containing poisoning attempts
            normal_messages: List of normal messages (should not be flagged)
        """
        from .safety import PoisoningDetector

        detector = PoisoningDetector()
        
        true_positives = 0  # Correctly detected poisoning
        false_positives = 0  # Normal message flagged as poisoning
        false_negatives = 0  # Poisoning not detected

        # Test poisoning attempts
        for attempt in poisoning_attempts:
            message = ChatMessage(role="user", content=attempt)
            is_poisoning, _ = detector.detect(message)
            if is_poisoning:
                true_positives += 1
            else:
                false_negatives += 1

        # Test normal messages
        for normal in normal_messages:
            message = ChatMessage(role="user", content=normal)
            is_poisoning, _ = detector.detect(message)
            if is_poisoning:
                false_positives += 1

        total_poisoning = len(poisoning_attempts)
        total_normal = len(normal_messages)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvaluationResult(
            test_name="poisoning_defense",
            accuracy=f1_score,
            latency_ms=0.0,
            token_count=0,
            cost_estimate=0.0,
            details={
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            },
        )

    def benchmark_retrieval(self, queries: List[str], iterations: int = 10) -> EvaluationResult:
        """Benchmark retrieval performance."""
        latencies = []

        for _ in range(iterations):
            for query in queries:
                start_time = time.time()
                options = RetrievalOptions(query=query, max_results=10)
                self.memory.retrieve(options)
                elapsed_ms = (time.time() - start_time) * 1000
                latencies.append(elapsed_ms)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

        return EvaluationResult(
            test_name="retrieval_benchmark",
            accuracy=0.0,
            latency_ms=avg_latency,
            token_count=0,
            cost_estimate=0.0,
            details={
                "p95_latency_ms": p95_latency,
                "iterations": iterations,
                "all_latencies": latencies,
            },
        )
