"""Data models for storage abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(Enum):
    """Status of a simulation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunArtifacts:
    """References to simulation output artifacts."""

    scenario_yaml: str | None = None
    events_jsonl: str | None = None
    balances_csv: str | None = None
    metrics_csv: str | None = None
    metrics_json: str | None = None
    run_html: str | None = None
    dealer_metrics_json: str | None = None
    trades_csv: str | None = None
    repayment_events_csv: str | None = None


@dataclass
class RunResult:
    """Complete result of a single simulation run."""

    run_id: str
    status: RunStatus
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: RunArtifacts = field(default_factory=RunArtifacts)
    error: str | None = None
    execution_time_ms: int | None = None


@dataclass
class RegistryEntry:
    """Metadata for registry storage."""

    run_id: str
    experiment_id: str
    status: RunStatus
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    error: str | None = None
