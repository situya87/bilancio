"""Data models for the runner abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bilancio.storage.models import RunStatus


@dataclass
class RunOptions:
    """Options passed to run_scenario().

    These options configure how the simulation is executed and what
    outputs are generated. The executor passes these through to the
    underlying simulation engine.

    Attributes:
        mode: Execution mode - "until_stable", "fixed_days", or "continuous".
        max_days: Maximum number of simulation days to run.
        quiet_days: Number of consecutive quiet days before stopping
            (for "until_stable" mode).
        check_invariants: When to check balance sheet invariants -
            "daily", "end", or "never".
        default_handling: How to handle defaults - "fail-fast" or "continue".
        show_events: Event display level for HTML report -
            "detailed", "summary", or "none".
        show_balances: List of agent names to show balances for in HTML report.
            If None, shows all agents.
        t_account: Whether to use T-account format in balance display.
        detailed_dealer_logging: Enable detailed logging for dealer simulations.
        run_id: Optional run identifier. If not provided, one will be generated.
        regime: Optional regime identifier for parameter sweeps.
    """

    mode: str = "until_stable"
    max_days: int = 90
    quiet_days: int = 2
    check_invariants: str = "daily"
    default_handling: str = "fail-fast"

    # Display options (for HTML report)
    show_events: str = "detailed"
    show_balances: Optional[List[str]] = None
    t_account: bool = False

    # Dealer options
    detailed_dealer_logging: bool = False
    run_id: Optional[str] = None
    regime: Optional[str] = None

    # Run parameters (for Supabase tracking in cloud execution)
    kappa: Optional[float] = None
    concentration: Optional[float] = None
    mu: Optional[float] = None
    outside_mid_ratio: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class ExecutionResult:
    """Result of simulation execution.

    Returned by the executor after a simulation run completes (or fails).
    Contains references to where artifacts are stored rather than the
    artifacts themselves.

    Attributes:
        run_id: Unique identifier for this run.
        status: Final status of the run (COMPLETED, FAILED, etc.).
        storage_type: Type of storage backend - "local", "s3", "gcs".
        storage_base: Base path/URI for artifacts - e.g., "/path/to/run"
            or "s3://bucket/prefix".
        artifacts: Mapping of artifact type to relative path within storage_base.
            Keys are artifact names like "events_jsonl", "balances_csv", etc.
        error: Error message if the run failed.
        execution_time_ms: Execution time in milliseconds.
        modal_call_id: Modal function call ID (for cloud execution debugging).
    """

    run_id: str
    status: RunStatus

    # Storage location
    storage_type: str  # "local", "s3", "gcs"
    storage_base: str  # "/path/to/run" or "s3://bucket/prefix"

    # Artifact references (relative to storage_base)
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Error info if failed
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None

    # Cloud execution info
    modal_call_id: Optional[str] = None

    # Computed metrics (from cloud execution)
    metrics: Optional[Dict[str, Any]] = None
