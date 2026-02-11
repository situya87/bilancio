"""Cloud executor using Modal for remote simulation execution."""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bilancio.runners.models import ExecutionResult, RunOptions
from bilancio.storage.models import RunStatus


class CloudExecutor:
    """Execute simulations on Modal cloud infrastructure.

    Results are stored on a Modal Volume and optionally downloaded to local disk
    after execution completes.

    This executor implements the SimulationExecutor protocol from
    bilancio.runners.protocols.

    Example:
        executor = CloudExecutor(experiment_id="my_sweep", job_id="castle-river-forest-mountain")
        result = executor.execute(
            scenario_config=scenario,
            run_id="run_001",
            output_dir=Path("./output"),
            options=RunOptions(max_days=30),
        )
    """

    def __init__(
        self,
        experiment_id: str,
        download_artifacts: bool = True,
        local_output_dir: Optional[Path] = None,
        volume_name: str = "bilancio-results",
        job_id: str = "",
    ):
        """Initialize cloud executor.

        Args:
            experiment_id: Identifier for this experiment (groups runs together).
            download_artifacts: Whether to download results to local disk after execution.
            local_output_dir: Where to download artifacts. Defaults to
                out/experiments/{experiment_id}.
            volume_name: Name of the Modal Volume for result storage. Note: This must
                match the volume name hardcoded in modal_app.py ("bilancio-results").
            job_id: Bilancio job ID for tracking (displayed in Modal logs).
        """
        self.experiment_id = experiment_id
        self.download_artifacts = download_artifacts
        self.local_output_dir = local_output_dir or Path(
            f"out/experiments/{experiment_id}"
        )
        self.volume_name = volume_name
        self.app_name = "bilancio-simulations"
        self.job_id = job_id

        # Lazy reference to deployed function
        self._run_simulation: Any = None

    def _get_run_simulation(self) -> Any:
        """Get reference to the deployed Modal function.

        Uses Function.from_name() to reference the deployed function,
        which allows calling it from outside the Modal app context.
        """
        if self._run_simulation is None:
            # Apply proxy patch for environments with HTTP CONNECT proxy (e.g., Claude Code web)
            import bilancio.cloud.proxy_patch  # noqa: F401
            import modal

            self._run_simulation = modal.Function.from_name(
                self.app_name, "run_simulation"
            )
        return self._run_simulation

    def execute(
        self,
        scenario_config: Dict[str, Any],
        run_id: str,
        output_dir: Path,
        options: RunOptions,
    ) -> ExecutionResult:
        """Execute simulation on Modal cloud.

        Args:
            scenario_config: Full scenario configuration dict.
            run_id: Unique run identifier.
            output_dir: Local directory for results (used if download_artifacts=True).
            options: Simulation run options.

        Returns:
            ExecutionResult with status and artifact references.
        """
        run_simulation = self._get_run_simulation()

        # Convert options to dict for serialization
        options_dict = self._options_to_dict(options)

        # Execute remotely (blocks until complete)
        result = run_simulation.remote(
            scenario_config=scenario_config,
            run_id=run_id,
            experiment_id=self.experiment_id,
            options=options_dict,
            job_id=self.job_id,
        )

        # Download artifacts if requested and determine storage location
        if self.download_artifacts and result["status"] == "completed":
            self._download_run_artifacts(run_id, output_dir, result["artifacts"])
            # When downloading, the storage_base should be the local path
            storage_type = "local"
            storage_base = str(output_dir.resolve())
        else:
            # When not downloading, keep the modal_volume reference
            storage_type = result["storage_type"]
            storage_base = result["storage_base"]

        return ExecutionResult(
            run_id=result["run_id"],
            status=(
                RunStatus.COMPLETED
                if result["status"] == "completed"
                else RunStatus.FAILED
            ),
            storage_type=storage_type,
            storage_base=storage_base,
            artifacts=result["artifacts"],
            error=result.get("error"),
            execution_time_ms=result.get("execution_time_ms"),
            modal_call_id=result.get("modal_call_id"),
            metrics=result.get("metrics"),
        )

    def execute_batch(
        self,
        runs: List[Tuple[Dict[str, Any], str, RunOptions]],
        max_parallel: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ExecutionResult]:
        """Execute multiple simulations in parallel on Modal.

        Modal handles parallelization automatically. This method provides
        a convenient interface for batch execution.

        Args:
            runs: List of (scenario_config, run_id, options) tuples.
            max_parallel: Maximum concurrent Modal function calls (unused,
                Modal handles this internally).
            progress_callback: Called with (completed, total) after each completion.

        Returns:
            List of ExecutionResult in same order as input.
        """
        run_simulation = self._get_run_simulation()

        total = len(runs)
        results: List[Optional[ExecutionResult]] = [None] * total

        run_id_to_index = {run_id: idx for idx, (_, run_id, _) in enumerate(runs)}
        configs = [config for config, _, _ in runs]
        run_ids = [run_id for _, run_id, _ in runs]
        options_dicts = [self._options_to_dict(options) for _, _, options in runs]

        # Collect results as they complete (unordered) so progress doesn't stall
        completed = 0
        for result in run_simulation.map(
            configs,
            run_ids,
            itertools.repeat(self.experiment_id),
            options_dicts,
            itertools.repeat(self.job_id),
            order_outputs=False,
        ):
            run_id = result["run_id"]
            idx = run_id_to_index[run_id]

            # Download artifacts if requested and determine storage location
            if self.download_artifacts and result["status"] == "completed":
                output_dir = self.local_output_dir / "runs" / run_id
                self._download_run_artifacts(run_id, output_dir, result["artifacts"])
                # When downloading, the storage_base should be the local path
                storage_type = "local"
                storage_base = str(output_dir.resolve())
            else:
                # When not downloading, keep the modal_volume reference
                storage_type = result["storage_type"]
                storage_base = result["storage_base"]

            results[idx] = ExecutionResult(
                run_id=result["run_id"],
                status=(
                    RunStatus.COMPLETED
                    if result["status"] == "completed"
                    else RunStatus.FAILED
                ),
                storage_type=storage_type,
                storage_base=storage_base,
                artifacts=result["artifacts"],
                error=result.get("error"),
                execution_time_ms=result.get("execution_time_ms"),
                modal_call_id=result.get("modal_call_id"),
                metrics=result.get("metrics"),
            )

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

        return results  # type: ignore

    def _options_to_dict(self, options: RunOptions) -> Dict[str, Any]:
        """Convert RunOptions to serializable dict."""
        result = {
            "mode": options.mode,
            "max_days": options.max_days,
            "quiet_days": options.quiet_days,
            "check_invariants": options.check_invariants,
            "default_handling": options.default_handling,
            "show_events": options.show_events,
            "t_account": options.t_account,
            "detailed_dealer_logging": options.detailed_dealer_logging,
            "regime": options.regime or "",
        }
        # Add run parameters for Supabase tracking
        if options.kappa is not None:
            result["kappa"] = options.kappa
        if options.concentration is not None:
            result["concentration"] = options.concentration
        if options.mu is not None:
            result["mu"] = options.mu
        if options.outside_mid_ratio is not None:
            result["outside_mid_ratio"] = options.outside_mid_ratio
        if options.seed is not None:
            result["seed"] = options.seed
        return result

    def _download_run_artifacts(
        self,
        run_id: str,
        output_dir: Path,
        artifacts: Dict[str, str],
    ) -> None:
        """Download artifacts from Modal Volume to local disk.

        Args:
            run_id: The run identifier.
            output_dir: Local directory to download to.
            artifacts: Dict mapping artifact names to relative paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_subdir = output_dir / "out"
        out_subdir.mkdir(exist_ok=True)

        remote_base = f"{self.experiment_id}/runs/{run_id}"

        for artifact_name, artifact_path in artifacts.items():
            remote_path = f"{remote_base}/{artifact_path}"
            local_path = output_dir / artifact_path

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Use Modal CLI to download
            try:
                subprocess.run(
                    [
                        "modal",
                        "volume",
                        "get",
                        self.volume_name,
                        remote_path,
                        str(local_path),
                        "--force",  # Overwrite existing files
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                # Log but don't fail - artifact might be optional
                print(
                    f"Warning: Failed to download {artifact_name}: {e.stderr.decode()}"
                )

    def compute_aggregate_metrics(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compute aggregate metrics for completed runs on Modal.

        Calls the compute_aggregate_metrics Modal function to calculate
        trading effects and summary statistics, and updates the job in Supabase.

        Args:
            run_ids: List of run IDs to include in aggregation.

        Returns:
            Dict with aggregate metrics and status.
        """
        # Apply proxy patch for environments with HTTP CONNECT proxy
        import bilancio.cloud.proxy_patch  # noqa: F401
        import modal

        # Get reference to deployed function (same pattern as run_simulation)
        modal_aggregate = modal.Function.from_name(self.app_name, "compute_aggregate_metrics")

        print("Computing aggregate metrics on Modal...", flush=True)
        result = modal_aggregate.remote(
            job_id=self.job_id,
            run_ids=run_ids,
        )

        if result.get("status") == "completed":
            summary = result.get("summary", {})
            print(f"Aggregate metrics computed:", flush=True)
            print(f"  Comparisons: {summary.get('n_comparisons', 0)}", flush=True)
            print(f"  Mean trading effect: {summary.get('mean_trading_effect', 'N/A'):.4f}"
                  if summary.get('mean_trading_effect') is not None else "  Mean trading effect: N/A", flush=True)
        else:
            print(f"Warning: Aggregate metrics computation failed: {result.get('error')}", flush=True)

        return dict(result)
