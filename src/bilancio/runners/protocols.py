"""Protocol definitions for simulation executors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from bilancio.runners.models import ExecutionResult, RunOptions
from bilancio.storage.models import RunResult, RunStatus


@runtime_checkable
class SimulationExecutor(Protocol):
    """Protocol for synchronous simulation execution.

    The executor only runs the simulation and returns where artifacts
    are stored. It does NOT compute metrics - that's MetricsComputer's job.
    """

    def execute(
        self,
        scenario_config: dict[str, Any],
        run_id: str,
        output_dir: Path,
        options: RunOptions,
    ) -> ExecutionResult:
        """Execute a simulation, return result with artifact paths.

        Args:
            scenario_config: Complete scenario configuration dict
            run_id: Unique identifier for this run
            output_dir: Directory for output files
            options: RunOptions with simulation parameters

        Returns:
            ExecutionResult with storage location and relative artifact paths
        """
        ...


@runtime_checkable
class JobExecutor(Protocol):
    """Protocol for async/distributed job execution (future use)."""

    def submit(self, scenario_config: dict[str, Any], run_id: str) -> str:
        """Submit job, return job_id."""
        ...

    def status(self, job_id: str) -> RunStatus:
        """Check job status."""
        ...

    def result(self, job_id: str) -> RunResult | None:
        """Get result if completed."""
        ...

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending/running job."""
        ...
