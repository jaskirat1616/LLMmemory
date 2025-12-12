"""
Safety and governance features: PII detection, poisoning defense, audit logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..base import ChatMessage


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"


@dataclass
class PIIMatch:
    """A detected PII match."""

    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0


@dataclass
class PoisoningDetector:
    """Detect memory poisoning attacks."""

    def __init__(self):
        # Common poisoning patterns
        self.poisoning_patterns = [
            r"(?i)ignore (?:all )?(?:previous|prior) instructions?",
            r"(?i)forget everything",
            r"(?i)system prompt",
            r"(?i)assistant mode",
            r"(?i)(?:new|override|override) instructions?:",
            r"(?i)important:.*(?:ignore|forget|override)",
            r"(?i)you are now",
            r"(?i)pretend you are",
            r"(?i)act as if",
        ]

        # Suspicious instruction patterns
        self.suspicious_instructions = [
            r"(?i)always (?:respond|say|tell)",
            r"(?i)never (?:mention|say|tell)",
            r"(?i)you must (?:always|never)",
            r"(?i)from now on",
        ]

    def detect(self, message: ChatMessage) -> tuple[bool, List[str]]:
        """
        Detect potential poisoning.
        
        Returns:
            (is_poisoning, reasons)
        """
        content = message.content.lower()
        reasons = []

        # Check for explicit poisoning patterns
        for pattern in self.poisoning_patterns:
            if re.search(pattern, content):
                reasons.append(f"Matches poisoning pattern: {pattern}")

        # Check for suspicious instructions
        for pattern in self.suspicious_instructions:
            if re.search(pattern, content):
                reasons.append(f"Contains suspicious instruction: {pattern}")

        is_poisoning = len(reasons) > 0
        return is_poisoning, reasons

    def sanitize(self, message: ChatMessage, remove_patterns: bool = True) -> ChatMessage:
        """Sanitize message by removing suspicious patterns."""
        content = message.content
        original_content = content

        if remove_patterns:
            # Remove poisoning patterns
            for pattern in self.poisoning_patterns:
                content = re.sub(pattern, "", content, flags=re.IGNORECASE)

            # Weaken suspicious instructions (replace with neutral text)
            for pattern in self.suspicious_instructions:
                # Extract the instruction part but neutralize it
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in reversed(list(matches)):  # Reverse to preserve indices
                    # Just remove the suspicious part
                    content = content[: match.start()] + content[match.end() :]

        # Clean up extra whitespace
        content = re.sub(r"\s+", " ", content).strip()

        metadata = message.metadata.copy()
        if content != original_content:
            metadata["sanitized"] = True
            metadata["original_length"] = len(original_content)

        return ChatMessage(
            role=message.role,
            content=content,
            metadata=metadata,
        )


@dataclass
class PIIDetector:
    """Detect Personally Identifiable Information."""

    def __init__(self):
        # Patterns for different PII types
        self.patterns = {
            PIIType.EMAIL: [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            ],
            PIIType.PHONE: [
                r"\b\d{3}-\d{3}-\d{4}\b",  # US format: 123-456-7890
                r"\(?\d{3}\)?\s*\d{3}-\d{4}",  # (123) 456-7890
                r"\b\d{10}\b",  # 10 digits
            ],
            PIIType.SSN: [
                r"\b\d{3}-\d{2}-\d{4}\b",  # 123-45-6789
                r"\b\d{9}\b",  # 9 digits
            ],
            PIIType.CREDIT_CARD: [
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # 16 digits
            ],
            PIIType.IP_ADDRESS: [
                r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            ],
            PIIType.DATE_OF_BIRTH: [
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY
                r"\b(?:born|birth date|DOB).*?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            ],
        }

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect all PII in text."""
        matches = []

        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append(
                        PIIMatch(
                            pii_type=pii_type,
                            value=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                        )
                    )

        return matches

    def redact(self, text: str, replacement: str = "[REDACTED]") -> tuple[str, List[PIIMatch]]:
        """Redact PII from text."""
        matches = self.detect(text)
        
        # Sort by position (reverse order to preserve indices)
        matches.sort(key=lambda m: m.start_pos, reverse=True)

        redacted = text
        for match in matches:
            redacted = redacted[: match.start_pos] + replacement + redacted[match.end_pos :]

        return redacted, matches

    def scrub(self, message: ChatMessage, replacement: str = "[REDACTED]") -> ChatMessage:
        """Scrub PII from a message."""
        matches = self.detect(message.content)

        if not matches:
            return message

        redacted_content, _ = self.redact(message.content, replacement)

        metadata = message.metadata.copy()
        metadata["pii_detected"] = True
        metadata["pii_types"] = [m.pii_type.value for m in matches]
        metadata["pii_count"] = len(matches)

        return ChatMessage(
            role=message.role,
            content=redacted_content,
            metadata=metadata,
        )


@dataclass
class AuditLog:
    """Audit log entry for memory operations."""

    operation: str  # "create", "read", "update", "delete"
    memory_type: str
    memory_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Logger for memory operations audit trail."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logs: List[AuditLog] = []

    def log(self, operation: str, memory_type: str, memory_id: Optional[str] = None, **metadata) -> None:
        """Log a memory operation."""
        if not self.enabled:
            return

        log_entry = AuditLog(
            operation=operation,
            memory_type=memory_type,
            memory_id=memory_id,
            metadata=metadata,
        )
        self.logs.append(log_entry)

    def get_logs(
        self,
        memory_type: Optional[str] = None,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditLog]:
        """Retrieve filtered audit logs."""
        filtered = self.logs

        if memory_type:
            filtered = [log for log in filtered if log.memory_type == memory_type]
        if operation:
            filtered = [log for log in filtered if log.operation == operation]
        if start_time:
            filtered = [log for log in filtered if log.timestamp >= start_time]
        if end_time:
            filtered = [log for log in filtered if log.timestamp <= end_time]

        return filtered

    def export(self, format: str = "json") -> str:
        """Export audit logs."""
        import json

        log_data = [
            {
                "operation": log.operation,
                "memory_type": log.memory_type,
                "memory_id": log.memory_id,
                "timestamp": log.timestamp.isoformat(),
                "user_id": log.user_id,
                "metadata": log.metadata,
            }
            for log in self.logs
        ]

        if format == "json":
            return json.dumps(log_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
