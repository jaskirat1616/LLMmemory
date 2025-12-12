"""
Claude-style project-based memory system.
Organizes conversations into projects with associated context and documents.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..base import ChatMessage, Memory


@dataclass
class Project:
    """A project with associated context and documents."""

    name: str
    description: str = ""
    documents: List[str] = field(default_factory=list)  # Document contents
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """Generated content artifact that can be referenced."""

    content: str
    artifact_type: str  # e.g., "code", "document", "plan"
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectMemory(Memory):
    """
    Claude-style project-based memory.
    Organizes conversations into projects, each with its own context and documents.
    """

    def __init__(self, current_project: str = "default", max_messages_per_project: int = 100):
        """
        Args:
            current_project: Name of the active project
            max_messages_per_project: Maximum messages to keep per project
        """
        self.current_project = current_project
        self.max_messages_per_project = max_messages_per_project

        # Project storage: project_name -> list of messages
        self._projects: Dict[str, List[ChatMessage]] = defaultdict(list)
        self._project_metadata: Dict[str, Project] = {}
        self._artifacts: Dict[str, List[Artifact]] = defaultdict(list)  # project -> artifacts

        # Initialize default project
        if current_project not in self._project_metadata:
            self._project_metadata[current_project] = Project(
                name=current_project, description="Default project"
            )

    def append(self, message: ChatMessage) -> None:
        """Add a message to the current project."""
        project_messages = self._projects[self.current_project]
        project_messages.append(message)

        # Limit messages per project
        if len(project_messages) > self.max_messages_per_project:
            self._projects[self.current_project] = project_messages[
                -self.max_messages_per_project :
            ]

    def get_context(self, query: str | None = None, k: int = 6) -> list[ChatMessage]:
        """
        Get context from the current project, including:
        - Recent messages from the project
        - Project documents (as system messages)
        - Relevant artifacts
        """
        context: list[ChatMessage] = []

        # Add project documents as context
        project = self._project_metadata.get(self.current_project)
        if project and project.documents:
            docs_text = "\n\n".join(project.documents)
            context.append(
                ChatMessage(
                    role="system",
                    content=f"Project '{project.name}' documents:\n{docs_text}",
                    metadata={"source": "project_documents"},
                )
            )

        # Add relevant artifacts
        artifacts = self._artifacts.get(self.current_project, [])
        if artifacts:
            artifacts_text = "\n\n".join(
                [f"[{a.artifact_type}] {a.content}" for a in artifacts[-3:]]  # Last 3 artifacts
            )
            context.append(
                ChatMessage(
                    role="system",
                    content=f"Project artifacts:\n{artifacts_text}",
                    metadata={"source": "artifacts"},
                )
            )

        # Add recent project messages
        project_messages = self._projects.get(self.current_project, [])
        context.extend(project_messages[-k:])

        return context

    def clear(self) -> None:
        """Clear the current project's messages."""
        self._projects[self.current_project].clear()

    def switch_project(self, project_name: str) -> None:
        """Switch to a different project."""
        self.current_project = project_name
        if project_name not in self._project_metadata:
            self._project_metadata[project_name] = Project(name=project_name)

    def add_document(self, document: str, project: str | None = None) -> None:
        """Add a document to a project's knowledge base."""
        project_name = project or self.current_project
        if project_name not in self._project_metadata:
            self._project_metadata[project_name] = Project(name=project_name)
        self._project_metadata[project_name].documents.append(document)

    def add_artifact(self, content: str, artifact_type: str, project: str | None = None) -> None:
        """Add an artifact to a project."""
        project_name = project or self.current_project
        artifacts = self._artifacts[project_name]
        # Check if similar artifact exists to version it
        version = 1
        for artifact in artifacts:
            if artifact.artifact_type == artifact_type:
                version = max(version, artifact.version + 1)
        artifacts.append(Artifact(content=content, artifact_type=artifact_type, version=version))

    def get_projects(self) -> List[str]:
        """Get list of all project names."""
        return list(self._project_metadata.keys())

    def get_project_info(self, project: str | None = None) -> Project | None:
        """Get metadata for a project."""
        project_name = project or self.current_project
        return self._project_metadata.get(project_name)
