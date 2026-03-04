"""Local synchronous simulation executor."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from bilancio.runners.models import ExecutionResult, RunOptions
from bilancio.storage.models import RunStatus

EXECUTION_ERRORS = (
    FileNotFoundError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    AssertionError,
)


class LocalExecutor:
    """Execute simulations locally and synchronously.

    This executor only runs the simulation and returns artifact locations.
    It does NOT compute metrics - that's MetricsComputer's job.

    The executor:
    1. Writes the scenario YAML to the run directory
    2. Calls run_scenario() with appropriate parameters
    3. Returns an ExecutionResult with artifact paths (relative to storage_base)
    """

    def execute(
        self,
        scenario_config: dict[str, Any],
        run_id: str,
        output_dir: Path,
        options: RunOptions,
    ) -> ExecutionResult:
        """Execute simulation, return result with artifact paths.

        Args:
            scenario_config: Complete scenario configuration dict
            run_id: Unique identifier for this run
            output_dir: Directory for output files (must be provided)
            options: RunOptions with simulation parameters

        Returns:
            ExecutionResult with storage location and relative artifact paths
        """
        # Import here to avoid circular imports at module load time
        from bilancio.ui.run import run_scenario

        start_time = time.time()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write scenario YAML
        scenario_path = output_dir / "scenario.yaml"
        scenario_path.write_text(yaml.dump(scenario_config, default_flow_style=False))

        # Set up export paths
        exports_dir = output_dir / "out"
        exports_dir.mkdir(exist_ok=True)

        balances_path = exports_dir / "balances.csv"
        events_path = exports_dir / "events.jsonl"
        run_html_path = output_dir / "run.html"

        try:
            # Run simulation
            run_scenario(
                path=scenario_path,
                mode=options.mode,
                max_days=options.max_days,
                quiet_days=options.quiet_days,
                show=options.show_events,
                agent_ids=options.show_balances,
                check_invariants=options.check_invariants,
                export={
                    "balances_csv": str(balances_path),
                    "events_jsonl": str(events_path),
                },
                html_output=run_html_path,
                t_account=options.t_account,
                default_handling=options.default_handling,
                detailed_dealer_logging=options.detailed_dealer_logging,
                run_id=options.run_id or run_id,
                regime=options.regime or "",
                performance=options.performance,
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Build artifact paths (relative to output_dir)
            artifacts: dict[str, str] = {
                "scenario_yaml": "scenario.yaml",
            }
            if events_path.exists():
                artifacts["events_jsonl"] = "out/events.jsonl"
            if balances_path.exists():
                artifacts["balances_csv"] = "out/balances.csv"
            if run_html_path.exists():
                artifacts["run_html"] = "run.html"

            return ExecutionResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                storage_type="local",
                storage_base=str(output_dir.resolve()),
                artifacts=artifacts,
                execution_time_ms=execution_time_ms,
            )

        except EXECUTION_ERRORS as e:
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Still record what artifacts exist
            fail_artifacts: dict[str, str] = {}
            if scenario_path.exists():
                fail_artifacts["scenario_yaml"] = "scenario.yaml"

            return ExecutionResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                storage_type="local",
                storage_base=str(output_dir.resolve()),
                artifacts=fail_artifacts,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )
