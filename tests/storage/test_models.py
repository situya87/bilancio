"""Tests for storage data models."""

from bilancio.storage.models import (
    RegistryEntry,
    RunArtifacts,
    RunResult,
    RunStatus,
)


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_pending_value(self):
        """Test PENDING enum has correct value."""
        assert RunStatus.PENDING.value == "pending"

    def test_running_value(self):
        """Test RUNNING enum has correct value."""
        assert RunStatus.RUNNING.value == "running"

    def test_completed_value(self):
        """Test COMPLETED enum has correct value."""
        assert RunStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        """Test FAILED enum has correct value."""
        assert RunStatus.FAILED.value == "failed"

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        statuses = {s.value for s in RunStatus}
        assert statuses == {"pending", "running", "completed", "failed"}

    def test_status_from_string(self):
        """Test creating status from string value."""
        assert RunStatus("completed") == RunStatus.COMPLETED
        assert RunStatus("failed") == RunStatus.FAILED


class TestRunArtifacts:
    """Tests for RunArtifacts dataclass."""

    def test_defaults_all_none(self):
        """Test all fields default to None."""
        artifacts = RunArtifacts()
        assert artifacts.scenario_yaml is None
        assert artifacts.events_jsonl is None
        assert artifacts.balances_csv is None
        assert artifacts.metrics_csv is None
        assert artifacts.metrics_json is None
        assert artifacts.run_html is None
        assert artifacts.dealer_metrics_json is None
        assert artifacts.trades_csv is None
        assert artifacts.repayment_events_csv is None

    def test_with_some_values(self):
        """Test creating with specific values."""
        artifacts = RunArtifacts(
            scenario_yaml="path/to/scenario.yaml",
            events_jsonl="path/to/events.jsonl",
        )
        assert artifacts.scenario_yaml == "path/to/scenario.yaml"
        assert artifacts.events_jsonl == "path/to/events.jsonl"
        assert artifacts.balances_csv is None

    def test_with_all_values(self):
        """Test creating with all values set."""
        artifacts = RunArtifacts(
            scenario_yaml="scenario.yaml",
            events_jsonl="events.jsonl",
            balances_csv="balances.csv",
            metrics_csv="metrics.csv",
            metrics_json="metrics.json",
            run_html="run.html",
            dealer_metrics_json="dealer_metrics.json",
            trades_csv="trades.csv",
            repayment_events_csv="repayment_events.csv",
        )
        assert artifacts.scenario_yaml == "scenario.yaml"
        assert artifacts.events_jsonl == "events.jsonl"
        assert artifacts.balances_csv == "balances.csv"
        assert artifacts.metrics_csv == "metrics.csv"
        assert artifacts.metrics_json == "metrics.json"
        assert artifacts.run_html == "run.html"
        assert artifacts.dealer_metrics_json == "dealer_metrics.json"
        assert artifacts.trades_csv == "trades.csv"
        assert artifacts.repayment_events_csv == "repayment_events.csv"


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_minimal_creation(self):
        """Test creating with only required fields."""
        result = RunResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
        )
        assert result.run_id == "run_001"
        assert result.status == RunStatus.COMPLETED
        assert result.parameters == {}
        assert result.metrics == {}
        assert isinstance(result.artifacts, RunArtifacts)
        assert result.error is None
        assert result.execution_time_ms is None

    def test_full_creation(self):
        """Test creating with all fields."""
        artifacts = RunArtifacts(scenario_yaml="scenario.yaml")
        result = RunResult(
            run_id="run_002",
            status=RunStatus.FAILED,
            parameters={"seed": 42, "kappa": 0.1},
            metrics={"phi_total": 100.5},
            artifacts=artifacts,
            error="Simulation diverged",
            execution_time_ms=5000,
        )
        assert result.run_id == "run_002"
        assert result.status == RunStatus.FAILED
        assert result.parameters == {"seed": 42, "kappa": 0.1}
        assert result.metrics == {"phi_total": 100.5}
        assert result.artifacts.scenario_yaml == "scenario.yaml"
        assert result.error == "Simulation diverged"
        assert result.execution_time_ms == 5000

    def test_field_access(self):
        """Test accessing various field types."""
        result = RunResult(
            run_id="test_run",
            status=RunStatus.RUNNING,
            parameters={"n_agents": 10, "mu": 0.5},
            metrics={"delta_total": 42.0},
        )
        # Access parameters
        assert result.parameters["n_agents"] == 10
        assert result.parameters["mu"] == 0.5
        # Access metrics
        assert result.metrics["delta_total"] == 42.0
        # Status comparison
        assert result.status == RunStatus.RUNNING
        assert result.status.value == "running"


class TestRegistryEntry:
    """Tests for RegistryEntry dataclass."""

    def test_minimal_creation(self):
        """Test creating with only required fields."""
        entry = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
        )
        assert entry.run_id == "run_001"
        assert entry.experiment_id == "exp_001"
        assert entry.status == RunStatus.COMPLETED
        assert entry.parameters == {}
        assert entry.metrics == {}
        assert entry.artifact_paths == {}
        assert entry.error is None

    def test_full_creation(self):
        """Test creating with all fields."""
        entry = RegistryEntry(
            run_id="run_002",
            experiment_id="exp_001",
            status=RunStatus.FAILED,
            parameters={"seed": 123, "kappa": 0.2},
            metrics={"phi_total": 50.0, "delta_total": 25.0},
            artifact_paths={
                "scenario_yaml": "exp_001/runs/run_002/scenario.yaml",
                "events_jsonl": "exp_001/runs/run_002/out/events.jsonl",
            },
            error="OOM error",
        )
        assert entry.run_id == "run_002"
        assert entry.experiment_id == "exp_001"
        assert entry.status == RunStatus.FAILED
        assert entry.parameters == {"seed": 123, "kappa": 0.2}
        assert entry.metrics == {"phi_total": 50.0, "delta_total": 25.0}
        assert "scenario_yaml" in entry.artifact_paths
        assert entry.error == "OOM error"

    def test_field_access(self):
        """Test accessing entry fields."""
        entry = RegistryEntry(
            run_id="test_run",
            experiment_id="test_exp",
            status=RunStatus.PENDING,
            parameters={"concentration": 0.8},
            artifact_paths={"run_html": "path/to/run.html"},
        )
        # Access by key
        assert entry.parameters["concentration"] == 0.8
        assert entry.artifact_paths["run_html"] == "path/to/run.html"
        # Status value
        assert entry.status.value == "pending"
