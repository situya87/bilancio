"""Tests for runner data models."""

import pytest

from bilancio.runners.models import RunOptions, ExecutionResult
from bilancio.storage.models import RunStatus


class TestRunOptionsDefaults:
    """Tests for RunOptions default values."""

    @pytest.mark.parametrize("attr,expected", [
        ("mode", "until_stable"),
        ("max_days", 90),
        ("quiet_days", 2),
        ("check_invariants", "daily"),
        ("default_handling", "fail-fast"),
        ("show_events", "detailed"),
        ("show_balances", None),
        ("t_account", False),
        ("detailed_dealer_logging", False),
        ("run_id", None),
        ("regime", None),
    ])
    def test_default_value(self, attr, expected):
        """RunOptions defaults are correct for each attribute."""
        options = RunOptions()
        assert getattr(options, attr) == expected


class TestRunOptionsCustomization:
    """Tests for customizing RunOptions."""

    @pytest.mark.parametrize("attr,value", [
        ("mode", "fixed_days"),
        ("max_days", 30),
        ("quiet_days", 5),
        ("check_invariants", "end"),
        ("default_handling", "continue"),
        ("show_events", "summary"),
        ("show_balances", ["Bank1", "Firm1"]),
        ("t_account", True),
        ("detailed_dealer_logging", True),
        ("run_id", "custom_run_001"),
        ("regime", "baseline"),
    ])
    def test_single_option_can_be_customized(self, attr, value):
        """Each RunOptions attribute can be set to a custom value."""
        options = RunOptions(**{attr: value})
        assert getattr(options, attr) == value

    def test_multiple_options_can_be_customized(self):
        """Multiple options can be customized at once."""
        options = RunOptions(
            mode="continuous",
            max_days=100,
            quiet_days=3,
            check_invariants="never",
            default_handling="continue",
            show_events="none",
            show_balances=["Bank1"],
            t_account=True,
            detailed_dealer_logging=True,
            run_id="test_run",
            regime="treatment",
        )

        assert options.mode == "continuous"
        assert options.max_days == 100
        assert options.quiet_days == 3
        assert options.check_invariants == "never"
        assert options.default_handling == "continue"
        assert options.show_events == "none"
        assert options.show_balances == ["Bank1"]
        assert options.t_account is True
        assert options.detailed_dealer_logging is True
        assert options.run_id == "test_run"
        assert options.regime == "treatment"


class TestExecutionResultCreation:
    """Tests for ExecutionResult creation."""

    def test_creation_with_all_required_fields(self):
        """ExecutionResult can be created with all required fields."""
        result = ExecutionResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            storage_type="local",
            storage_base="/path/to/run",
        )

        assert result.run_id == "run_001"
        assert result.status == RunStatus.COMPLETED
        assert result.storage_type == "local"
        assert result.storage_base == "/path/to/run"

    def test_creation_with_failed_status(self):
        """ExecutionResult can be created with FAILED status."""
        result = ExecutionResult(
            run_id="run_fail",
            status=RunStatus.FAILED,
            storage_type="local",
            storage_base="/path/to/run",
            error="Simulation diverged",
        )

        assert result.status == RunStatus.FAILED
        assert result.error == "Simulation diverged"

    def test_creation_with_all_fields(self):
        """ExecutionResult can be created with all fields."""
        result = ExecutionResult(
            run_id="run_002",
            status=RunStatus.COMPLETED,
            storage_type="s3",
            storage_base="s3://bucket/prefix",
            artifacts={
                "events_jsonl": "out/events.jsonl",
                "balances_csv": "out/balances.csv",
            },
            error=None,
            execution_time_ms=5000,
        )

        assert result.run_id == "run_002"
        assert result.status == RunStatus.COMPLETED
        assert result.storage_type == "s3"
        assert result.storage_base == "s3://bucket/prefix"
        assert result.artifacts["events_jsonl"] == "out/events.jsonl"
        assert result.artifacts["balances_csv"] == "out/balances.csv"
        assert result.error is None
        assert result.execution_time_ms == 5000


class TestExecutionResultDefaults:
    """Tests for ExecutionResult default values."""

    @pytest.mark.parametrize("attr,expected", [
        ("artifacts", {}),
        ("error", None),
        ("execution_time_ms", None),
    ])
    def test_default_value(self, attr, expected):
        """ExecutionResult defaults are correct for each optional attribute."""
        result = ExecutionResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            storage_type="local",
            storage_base="/path",
        )
        assert getattr(result, attr) == expected


class TestExecutionResultArtifacts:
    """Tests for ExecutionResult artifacts handling."""

    def test_artifacts_can_store_multiple_paths(self):
        """artifacts can store multiple artifact paths."""
        artifacts = {
            "scenario_yaml": "scenario.yaml",
            "events_jsonl": "out/events.jsonl",
            "balances_csv": "out/balances.csv",
            "run_html": "run.html",
            "metrics_csv": "out/metrics.csv",
        }

        result = ExecutionResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            storage_type="local",
            storage_base="/path",
            artifacts=artifacts,
        )

        assert len(result.artifacts) == 5
        assert result.artifacts["scenario_yaml"] == "scenario.yaml"
        assert result.artifacts["events_jsonl"] == "out/events.jsonl"

    def test_artifacts_can_be_modified(self):
        """artifacts dict can be modified after creation."""
        result = ExecutionResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            storage_type="local",
            storage_base="/path",
        )

        result.artifacts["new_artifact"] = "path/to/artifact"
        assert result.artifacts["new_artifact"] == "path/to/artifact"


class TestExecutionResultStorageTypes:
    """Tests for ExecutionResult storage type variants."""

    @pytest.mark.parametrize("storage_type,storage_base", [
        ("local", "/Users/test/experiments/run_001"),
        ("s3", "s3://my-bucket/experiments/run_001"),
        ("gcs", "gs://my-bucket/experiments/run_001"),
    ])
    def test_storage_type_supported(self, storage_type, storage_base):
        """ExecutionResult supports various storage types."""
        result = ExecutionResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            storage_type=storage_type,
            storage_base=storage_base,
        )
        assert result.storage_type == storage_type
