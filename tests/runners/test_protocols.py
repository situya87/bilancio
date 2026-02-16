"""Tests for simulation executor protocol definitions.

This module tests:
- Protocol structure validation
- Protocol conformance checking
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bilancio.runners.local_executor import LocalExecutor
from bilancio.runners.models import ExecutionResult, RunOptions
from bilancio.runners.protocols import JobExecutor, SimulationExecutor
from bilancio.storage.models import RunResult, RunStatus


class TestSimulationExecutorProtocol:
    """Tests for SimulationExecutor protocol definition."""

    def test_local_executor_is_instance_of_protocol(self):
        """LocalExecutor should satisfy SimulationExecutor protocol."""
        executor = LocalExecutor()
        assert isinstance(executor, SimulationExecutor)

    def test_protocol_has_execute_method(self):
        """SimulationExecutor protocol requires execute method."""
        # Check that the protocol has the execute method defined
        assert hasattr(SimulationExecutor, "execute")

    def test_custom_class_satisfying_protocol(self):
        """A custom class with execute method should satisfy the protocol."""

        class CustomExecutor:
            def execute(
                self,
                scenario_config: dict[str, Any],
                run_id: str,
                output_dir: Path,
                options: RunOptions,
            ) -> ExecutionResult:
                return ExecutionResult(
                    run_id=run_id,
                    status=RunStatus.COMPLETED,
                    storage_type="local",
                    storage_base=str(output_dir),
                    artifacts={},
                )

        executor = CustomExecutor()
        assert isinstance(executor, SimulationExecutor)

    def test_class_missing_execute_does_not_satisfy_protocol(self):
        """A class without execute method should not satisfy the protocol."""

        class IncompleteExecutor:
            pass

        executor = IncompleteExecutor()
        assert not isinstance(executor, SimulationExecutor)

    def test_class_with_wrong_signature_still_satisfies_protocol(self):
        """Protocol checking is structural, not signature-exact at runtime."""
        # Note: Python's Protocol runtime checking only checks method existence,
        # not exact signatures. This is a known limitation.

        class WrongSignatureExecutor:
            def execute(self):
                pass

        executor = WrongSignatureExecutor()
        # This will still pass isinstance check because Python only checks
        # method existence at runtime, not the full signature
        assert isinstance(executor, SimulationExecutor)


class TestJobExecutorProtocol:
    """Tests for JobExecutor protocol definition."""

    def test_protocol_has_required_methods(self):
        """JobExecutor protocol requires submit, status, result, and cancel methods."""
        assert hasattr(JobExecutor, "submit")
        assert hasattr(JobExecutor, "status")
        assert hasattr(JobExecutor, "result")
        assert hasattr(JobExecutor, "cancel")

    def test_custom_class_satisfying_job_executor_protocol(self):
        """A custom class with all required methods should satisfy the protocol."""

        class CustomJobExecutor:
            def submit(self, scenario_config: dict[str, Any], run_id: str) -> str:
                return "job_123"

            def status(self, job_id: str) -> RunStatus:
                return RunStatus.RUNNING

            def result(self, job_id: str) -> RunResult | None:
                return None

            def cancel(self, job_id: str) -> bool:
                return True

        executor = CustomJobExecutor()
        assert isinstance(executor, JobExecutor)

    def test_class_missing_method_does_not_satisfy_protocol(self):
        """A class missing any required method should not satisfy the protocol."""

        class IncompleteJobExecutor:
            def submit(self, scenario_config: dict[str, Any], run_id: str) -> str:
                return "job_123"

            # Missing: status, result, cancel

        executor = IncompleteJobExecutor()
        assert not isinstance(executor, JobExecutor)

    def test_local_executor_does_not_satisfy_job_executor(self):
        """LocalExecutor should not satisfy JobExecutor protocol."""
        executor = LocalExecutor()
        assert not isinstance(executor, JobExecutor)
