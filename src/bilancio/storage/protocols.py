"""Protocol definitions for storage backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import RegistryEntry, RunResult


@runtime_checkable
class ResultStore(Protocol):
    """Protocol for storing simulation results."""

    def save_artifact(
        self,
        experiment_id: str,
        run_id: str,
        name: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Save an artifact, return its reference (path or key)."""
        ...

    def save_run(self, experiment_id: str, result: RunResult) -> None:
        """Save a complete run result."""
        ...

    def load_run(self, experiment_id: str, run_id: str) -> RunResult | None:
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

    def get(self, experiment_id: str, run_id: str) -> RegistryEntry | None:
        """Get a specific registry entry."""
        ...

    def list_runs(self, experiment_id: str) -> list[str]:
        """List all run IDs for an experiment."""
        ...

    def get_completed_keys(
        self, experiment_id: str, key_fields: list[str] | None = None
    ) -> set[Any]:
        """Get set of completed parameter keys for resumption."""
        ...

    def query(
        self, experiment_id: str, filters: dict[str, Any] | None = None
    ) -> list[RegistryEntry]:
        """Query registry entries with optional filters."""
        ...
