"""Protocol definitions for storage backends."""

from __future__ import annotations

from typing import Protocol, Optional, List, Dict, Any, Set, runtime_checkable

from .models import RunResult, RegistryEntry


@runtime_checkable
class ResultStore(Protocol):
    """Protocol for storing simulation results."""

    def save_artifact(
        self,
        experiment_id: str,
        run_id: str,
        name: str,
        content: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Save an artifact, return its reference (path or key)."""
        ...

    def save_run(self, experiment_id: str, result: RunResult) -> None:
        """Save a complete run result."""
        ...

    def load_run(self, experiment_id: str, run_id: str) -> Optional[RunResult]:
        """Load a run result by ID."""
        ...

    def load_artifact(self, reference: str) -> bytes:
        """Load artifact content by reference."""
        ...


@runtime_checkable
class RegistryStore(Protocol):
    """Protocol for experiment registry storage."""

    def upsert(self, entry: RegistryEntry) -> None:
        """Insert or update a registry entry."""
        ...

    def get(self, experiment_id: str, run_id: str) -> Optional[RegistryEntry]:
        """Get a specific registry entry."""
        ...

    def list_runs(self, experiment_id: str) -> List[str]:
        """List all run IDs for an experiment."""
        ...

    def get_completed_keys(
        self,
        experiment_id: str,
        key_fields: Optional[List[str]] = None
    ) -> Set[Any]:
        """Get set of completed parameter keys for resumption."""
        ...

    def query(
        self,
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RegistryEntry]:
        """Query registry entries with optional filters."""
        ...
