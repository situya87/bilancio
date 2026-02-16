"""Tests for LocalExecutor simulation execution.

This module tests:
- Basic execution and result handling
- Output file creation
- Error handling
- Directory management

Note: Metrics computation is NOT tested here because LocalExecutor
no longer computes metrics. That's MetricsComputer's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bilancio.runners.local_executor import LocalExecutor
from bilancio.runners.models import ExecutionResult, RunOptions
from bilancio.storage.models import RunStatus

# Minimal scenario configuration for testing
MINIMAL_SCENARIO: dict[str, Any] = {
    "version": 1,
    "name": "Test Scenario",
    "description": "Minimal test scenario",
    "agents": [
        {"id": "bank", "kind": "bank", "name": "Test Bank"},
        {"id": "firm1", "kind": "firm", "name": "Test Firm 1"},
        {"id": "firm2", "kind": "firm", "name": "Test Firm 2"},
    ],
    "initial_actions": [],
    "run": {"max_days": 5},
}

# Scenario with activity that produces events
SCENARIO_WITH_ACTIVITY: dict[str, Any] = {
    "version": 1,
    "name": "Activity Test Scenario",
    "description": "Scenario with payment activity to generate events",
    "agents": [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
        {"id": "bank", "kind": "bank", "name": "Test Bank"},
        {"id": "firm1", "kind": "firm", "name": "Test Firm 1"},
        {"id": "firm2", "kind": "firm", "name": "Test Firm 2"},
    ],
    "initial_actions": [
        {"mint_reserves": {"to": "bank", "amount": 10000}},
        {"mint_cash": {"to": "firm1", "amount": 2000}},
        {"mint_cash": {"to": "firm2", "amount": 1500}},
        {"deposit_cash": {"customer": "firm1", "bank": "bank", "amount": 1800}},
        {"deposit_cash": {"customer": "firm2", "bank": "bank", "amount": 1000}},
        {"create_payable": {"from": "firm1", "to": "firm2", "amount": 500, "due_day": 1}},
    ],
    "run": {"max_days": 10},
}


class TestLocalExecutorBasicExecution:
    """Tests for basic LocalExecutor.execute functionality."""

    @pytest.mark.slow
    def test_execute_returns_execution_result(self, tmp_path: Path):
        """execute() returns an ExecutionResult object."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_basic_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert isinstance(result, ExecutionResult)
        assert result.run_id == "test_basic_001"

    @pytest.mark.slow
    def test_execute_simple_scenario_returns_completed_status(self, tmp_path: Path):
        """execute() with valid scenario returns COMPLETED status."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_completed_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.status == RunStatus.COMPLETED
        assert result.error is None

    @pytest.mark.slow
    def test_execute_populates_execution_time(self, tmp_path: Path):
        """execute() populates execution_time_ms."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_time_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0

    @pytest.mark.slow
    def test_execute_returns_local_storage_type(self, tmp_path: Path):
        """execute() returns storage_type='local'."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_storage_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.storage_type == "local"

    @pytest.mark.slow
    def test_execute_returns_absolute_storage_base(self, tmp_path: Path):
        """execute() returns absolute path as storage_base."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_storage_base_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.storage_base == str(tmp_path.resolve())


class TestLocalExecutorOutputFiles:
    """Tests for output file creation."""

    @pytest.mark.slow
    def test_execute_creates_scenario_yaml(self, tmp_path: Path):
        """execute() creates scenario.yaml file."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_yaml_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        scenario_path = tmp_path / "scenario.yaml"
        assert scenario_path.exists()
        assert result.artifacts.get("scenario_yaml") == "scenario.yaml"

    @pytest.mark.slow
    def test_execute_creates_exports_directory(self, tmp_path: Path):
        """execute() creates out/ exports directory."""
        executor = LocalExecutor()
        executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_exports_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        exports_dir = tmp_path / "out"
        assert exports_dir.exists()
        assert exports_dir.is_dir()

    @pytest.mark.slow
    def test_execute_creates_balances_csv(self, tmp_path: Path):
        """execute() creates balances.csv file."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_balances_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        balances_path = tmp_path / "out" / "balances.csv"
        assert balances_path.exists()
        assert result.artifacts.get("balances_csv") == "out/balances.csv"

    @pytest.mark.slow
    def test_execute_creates_events_jsonl(self, tmp_path: Path):
        """execute() creates events.jsonl file."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_events_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        events_path = tmp_path / "out" / "events.jsonl"
        assert events_path.exists()
        assert result.artifacts.get("events_jsonl") == "out/events.jsonl"

    @pytest.mark.slow
    def test_execute_creates_run_html(self, tmp_path: Path):
        """execute() creates run.html file."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_html_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        html_path = tmp_path / "run.html"
        assert html_path.exists()
        assert result.artifacts.get("run_html") == "run.html"

    @pytest.mark.slow
    def test_artifacts_are_relative_paths(self, tmp_path: Path):
        """execute() returns artifacts as relative paths."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_relative_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        # All artifact paths should be relative (not starting with /)
        for key, path in result.artifacts.items():
            assert not path.startswith("/"), f"Artifact {key} should be relative: {path}"


class TestLocalExecutorErrorHandling:
    """Tests for error handling in LocalExecutor."""

    @pytest.mark.slow
    def test_execute_returns_failed_status_on_invalid_scenario(self, tmp_path: Path):
        """execute() returns FAILED status when scenario is invalid."""
        executor = LocalExecutor()

        # Invalid scenario - missing required fields
        invalid_scenario: dict[str, Any] = {
            "agents": "not_a_list",  # Should be a list
            "setup": [],
        }

        result = executor.execute(
            scenario_config=invalid_scenario,
            run_id="test_failed_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.status == RunStatus.FAILED

    @pytest.mark.slow
    def test_execute_returns_error_message_on_failure(self, tmp_path: Path):
        """execute() includes error message when simulation fails."""
        executor = LocalExecutor()

        # Invalid scenario configuration
        invalid_scenario: dict[str, Any] = {
            "agents": [{"id": "bank"}],  # Missing 'kind'
            "setup": [],
        }

        result = executor.execute(
            scenario_config=invalid_scenario,
            run_id="test_error_msg_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.status == RunStatus.FAILED
        assert result.error is not None
        assert len(result.error) > 0

    @pytest.mark.slow
    def test_execute_records_execution_time_on_failure(self, tmp_path: Path):
        """execute() records execution_time_ms even on failure."""
        executor = LocalExecutor()

        invalid_scenario: dict[str, Any] = {
            "agents": "invalid",
            "setup": [],
        }

        result = executor.execute(
            scenario_config=invalid_scenario,
            run_id="test_time_fail_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.slow
    def test_failed_execution_still_includes_scenario_yaml(self, tmp_path: Path):
        """execute() includes scenario_yaml artifact even on failure."""
        executor = LocalExecutor()

        invalid_scenario: dict[str, Any] = {
            "agents": "invalid",
            "setup": [],
        }

        result = executor.execute(
            scenario_config=invalid_scenario,
            run_id="test_yaml_fail_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        # scenario.yaml should still be written and referenced
        assert "scenario_yaml" in result.artifacts
        scenario_path = tmp_path / result.artifacts["scenario_yaml"]
        assert scenario_path.exists()


class TestLocalExecutorDirectoryManagement:
    """Tests for output directory management."""

    @pytest.mark.slow
    def test_output_dir_parameter_is_respected(self, tmp_path: Path):
        """execute() uses provided output_dir."""
        executor = LocalExecutor()
        custom_dir = tmp_path / "custom_output"
        custom_dir.mkdir()

        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_custom_dir_001",
            output_dir=custom_dir,
            options=RunOptions(max_days=5),
        )

        # storage_base should be the custom directory
        assert str(custom_dir.resolve()) in result.storage_base

    @pytest.mark.slow
    def test_output_dir_created_if_not_exists(self, tmp_path: Path):
        """execute() creates output_dir if it doesn't exist."""
        executor = LocalExecutor()
        new_dir = tmp_path / "new_output_dir"

        # Directory should not exist yet
        assert not new_dir.exists()

        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_create_dir_001",
            output_dir=new_dir,
            options=RunOptions(max_days=5),
        )

        # Directory should now exist
        assert new_dir.exists()
        assert result.status == RunStatus.COMPLETED

    @pytest.mark.slow
    def test_nested_output_dir_created(self, tmp_path: Path):
        """execute() creates nested output directories."""
        executor = LocalExecutor()
        nested_dir = tmp_path / "level1" / "level2" / "level3"

        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_nested_dir_001",
            output_dir=nested_dir,
            options=RunOptions(max_days=5),
        )

        assert nested_dir.exists()
        assert result.status == RunStatus.COMPLETED


class TestLocalExecutorRunIdHandling:
    """Tests for run_id handling in results."""

    @pytest.mark.slow
    def test_run_id_preserved_in_result(self, tmp_path: Path):
        """execute() preserves run_id in result."""
        executor = LocalExecutor()
        custom_run_id = "unique_run_id_12345"

        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id=custom_run_id,
            output_dir=tmp_path,
            options=RunOptions(max_days=5),
        )

        assert result.run_id == custom_run_id


class TestLocalExecutorRunOptions:
    """Tests for RunOptions parameter handling."""

    @pytest.mark.slow
    def test_options_max_days_is_used(self, tmp_path: Path):
        """execute() uses max_days from RunOptions."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_options_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=3),
        )

        assert result.status == RunStatus.COMPLETED

    @pytest.mark.slow
    def test_options_mode_is_used(self, tmp_path: Path):
        """execute() uses mode from RunOptions."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_mode_001",
            output_dir=tmp_path,
            options=RunOptions(mode="until_stable", max_days=5),
        )

        assert result.status == RunStatus.COMPLETED

    @pytest.mark.slow
    def test_default_options_work(self, tmp_path: Path):
        """execute() works with default RunOptions."""
        executor = LocalExecutor()
        result = executor.execute(
            scenario_config=MINIMAL_SCENARIO,
            run_id="test_defaults_001",
            output_dir=tmp_path,
            options=RunOptions(),
        )

        assert result.status == RunStatus.COMPLETED


class TestLocalExecutorScenarioWithActivity:
    """Tests with scenarios that have actual activity."""

    @pytest.mark.slow
    def test_scenario_with_obligations_produces_events(self, tmp_path: Path):
        """Scenario with setup events produces events.jsonl."""
        executor = LocalExecutor()

        result = executor.execute(
            scenario_config=SCENARIO_WITH_ACTIVITY,
            run_id="test_activity_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=10),
        )

        assert result.status == RunStatus.COMPLETED
        # Events file should be created
        events_path = tmp_path / "out" / "events.jsonl"
        assert events_path.exists()
        # And it should have content
        assert events_path.stat().st_size > 0

    @pytest.mark.slow
    def test_result_artifacts_are_relative(self, tmp_path: Path):
        """execute() returns artifacts as relative paths for active scenario."""
        executor = LocalExecutor()

        result = executor.execute(
            scenario_config=SCENARIO_WITH_ACTIVITY,
            run_id="test_artifacts_001",
            output_dir=tmp_path,
            options=RunOptions(max_days=10),
        )

        assert isinstance(result, ExecutionResult)

        # All artifact paths should be relative
        for key, path in result.artifacts.items():
            assert not path.startswith("/"), f"{key} should be relative"
            # And should resolve to existing files
            full_path = tmp_path / path
            assert full_path.exists(), f"{key} at {path} should exist"
