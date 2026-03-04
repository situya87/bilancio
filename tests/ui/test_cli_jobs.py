"""Tests for the jobs CLI commands (bilancio/ui/cli/jobs.py).

Covers list_jobs, get_job, list_runs, show_metrics, and visualize_job
with mocked backends to avoid Supabase/filesystem dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bilancio.jobs.models import Job, JobConfig, JobStatus
from bilancio.storage.models import RegistryEntry, RunStatus
from bilancio.ui.cli import cli
from bilancio.ui.cli.jobs import format_datetime, format_duration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(
    job_id: str = "test-castle-river-mountain",
    status: JobStatus = JobStatus.COMPLETED,
    run_ids: list[str] | None = None,
    error: str | None = None,
) -> Job:
    """Create a minimal Job instance for testing."""
    now = datetime(2025, 6, 15, 10, 30, 0)
    completed = now + timedelta(minutes=12) if status == JobStatus.COMPLETED else None
    return Job(
        job_id=job_id,
        created_at=now,
        status=status,
        description="Test balanced comparison sweep",
        config=JobConfig(
            sweep_type="balanced",
            n_agents=50,
            kappas=[Decimal("0.3"), Decimal("0.5")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
            cloud=True,
            maturity_days=10,
        ),
        run_ids=run_ids or ["run-passive-001", "run-active-001"],
        completed_at=completed,
        error=error,
    )


def _make_registry_entry(
    run_id: str,
    status: RunStatus = RunStatus.COMPLETED,
    kappa: str = "0.5",
    concentration: str = "1",
    delta_total: float | None = 0.15,
) -> RegistryEntry:
    """Create a minimal RegistryEntry for testing."""
    metrics = {}
    if delta_total is not None:
        metrics["delta_total"] = delta_total
        metrics["phi_total"] = 1.0 - delta_total
    return RegistryEntry(
        run_id=run_id,
        experiment_id="test-job",
        status=status,
        parameters={"kappa": kappa, "concentration": concentration},
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestFormatHelpers:
    """Tests for format_datetime and format_duration."""

    def test_format_datetime_none(self):
        assert format_datetime(None) == "-"

    def test_format_datetime_value(self):
        dt = datetime(2025, 6, 15, 10, 30, 0)
        assert format_datetime(dt) == "2025-06-15 10:30"

    def test_format_duration_running(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        assert format_duration(start, None) == "running"

    def test_format_duration_seconds(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        end = start + timedelta(seconds=45)
        assert format_duration(start, end) == "45s"

    def test_format_duration_minutes(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        end = start + timedelta(minutes=12, seconds=30)
        result = format_duration(start, end)
        assert "12.5m" == result

    def test_format_duration_hours(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        end = start + timedelta(hours=2, minutes=15)
        result = format_duration(start, end)
        assert "h" in result


# ---------------------------------------------------------------------------
# Jobs group and help
# ---------------------------------------------------------------------------

class TestJobsHelp:
    """Tests for jobs --help and subcommand help."""

    def test_jobs_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "--help"])
        assert result.exit_code == 0
        assert "jobs" in result.output.lower()
        assert "ls" in result.output
        assert "get" in result.output

    def test_jobs_ls_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "ls", "--help"])
        assert result.exit_code == 0
        assert "--cloud" in result.output
        assert "--local" in result.output
        assert "--status" in result.output
        assert "--limit" in result.output

    def test_jobs_get_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "get", "--help"])
        assert result.exit_code == 0
        assert "JOB_ID" in result.output
        assert "--cloud" in result.output

    def test_jobs_runs_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "runs", "--help"])
        assert result.exit_code == 0
        assert "JOB_ID" in result.output
        assert "--cloud" in result.output

    def test_jobs_metrics_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "metrics", "--help"])
        assert result.exit_code == 0
        assert "JOB_ID" in result.output
        assert "--cloud" in result.output

    def test_jobs_visualize_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "visualize", "--help"])
        assert result.exit_code == 0
        assert "JOB_ID" in result.output
        assert "--output" in result.output or "-o" in result.output


# ---------------------------------------------------------------------------
# jobs ls
# ---------------------------------------------------------------------------

class TestJobsLs:
    """Tests for the 'jobs ls' command."""

    def test_ls_no_source_and_supabase_not_configured(self):
        """When no --cloud, no --local, and Supabase not configured, show hint."""
        with patch(
            "bilancio.storage.supabase_client.is_supabase_configured", return_value=False
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "ls"])
            assert result.exit_code == 0
            assert "No source specified" in result.output

    def test_ls_cloud_no_supabase(self):
        """--cloud but Supabase not configured raises error."""
        mock_store = MagicMock()
        mock_store.client = None

        with patch(
            "bilancio.jobs.supabase_store.SupabaseJobStore", return_value=mock_store
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "ls", "--cloud"])
            assert result.exit_code != 0
            assert "Supabase not configured" in result.output

    def test_ls_cloud_with_jobs(self):
        """--cloud with jobs returns a formatted table."""
        jobs_list = [_make_job("job-alpha"), _make_job("job-beta", status=JobStatus.RUNNING)]
        mock_store = MagicMock()
        mock_store.client = "not-none"
        mock_store.list_jobs.return_value = jobs_list
        mock_store.get_run_counts.return_value = {"job-alpha": 4, "job-beta": 2}

        with patch("bilancio.jobs.supabase_store.SupabaseJobStore", return_value=mock_store):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "ls", "--cloud"])
            assert result.exit_code == 0
            assert "job-alpha" in result.output
            assert "job-beta" in result.output
            assert "Total: 2 jobs" in result.output

    def test_ls_cloud_empty(self):
        """--cloud with no jobs shows 'No jobs found.'."""
        mock_store = MagicMock()
        mock_store.client = "not-none"
        mock_store.list_jobs.return_value = []

        with patch("bilancio.jobs.supabase_store.SupabaseJobStore", return_value=mock_store):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "ls", "--cloud"])
            assert result.exit_code == 0
            assert "No jobs found" in result.output

    def test_ls_local_with_jobs(self, tmp_path):
        """--local with a populated job directory lists jobs."""
        mock_manager = MagicMock()
        mock_manager.list_jobs.return_value = [
            _make_job("local-job-1"),
            _make_job("local-job-2", status=JobStatus.FAILED),
        ]

        with patch("bilancio.jobs.JobManager", return_value=mock_manager):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "ls", "--local", str(tmp_path)])
            assert result.exit_code == 0
            assert "local-job-1" in result.output
            assert "local-job-2" in result.output
            assert "Total: 2 jobs" in result.output

    def test_ls_local_with_status_filter(self, tmp_path):
        """--local --status completed filters correctly."""
        completed_job = _make_job("done-job", status=JobStatus.COMPLETED)
        running_job = _make_job("running-job", status=JobStatus.RUNNING)
        mock_manager = MagicMock()
        mock_manager.list_jobs.return_value = [completed_job, running_job]

        with patch("bilancio.jobs.JobManager", return_value=mock_manager):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["jobs", "ls", "--local", str(tmp_path), "--status", "completed"]
            )
            assert result.exit_code == 0
            assert "done-job" in result.output
            assert "running-job" not in result.output

    def test_ls_local_empty(self, tmp_path):
        """--local with no jobs shows 'No jobs found.'."""
        mock_manager = MagicMock()
        mock_manager.list_jobs.return_value = []

        with patch("bilancio.jobs.JobManager", return_value=mock_manager):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "ls", "--local", str(tmp_path)])
            assert result.exit_code == 0
            assert "No jobs found" in result.output

    def test_ls_cloud_import_error(self):
        """--cloud with import error raises ClickException."""
        import sys
        # Temporarily make the module unimportable
        with patch.dict(sys.modules, {"bilancio.jobs.supabase_store": None}):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "ls", "--cloud"])
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# jobs get
# ---------------------------------------------------------------------------

class TestJobsGet:
    """Tests for the 'jobs get' command."""

    def test_get_cloud_found(self):
        """--cloud finds and displays a job."""
        job = _make_job("found-job")
        mock_store = MagicMock()
        mock_store.client = "not-none"
        mock_store.get_job.return_value = job

        with patch("bilancio.jobs.supabase_store.SupabaseJobStore", return_value=mock_store):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "get", "found-job", "--cloud"])
            assert result.exit_code == 0
            assert "found-job" in result.output
            assert "balanced" in result.output
            assert "50" in result.output  # n_agents
            assert "0.3" in result.output  # kappa

    def test_get_cloud_not_found(self):
        """--cloud with nonexistent job raises error."""
        mock_store = MagicMock()
        mock_store.client = "not-none"
        mock_store.get_job.return_value = None

        with patch("bilancio.jobs.supabase_store.SupabaseJobStore", return_value=mock_store):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "get", "nonexistent-job", "--cloud"])
            assert result.exit_code != 0
            assert "Job not found" in result.output

    def test_get_local_found(self, tmp_path):
        """--local finds and displays a job."""
        job = _make_job("local-found")
        mock_manager = MagicMock()
        mock_manager.get_job.return_value = job

        with patch("bilancio.jobs.JobManager", return_value=mock_manager):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "get", "local-found", "--local", str(tmp_path)])
            assert result.exit_code == 0
            assert "local-found" in result.output
            assert "completed" in result.output

    def test_get_job_with_error(self):
        """Display job that has an error field."""
        job = _make_job("error-job", status=JobStatus.FAILED, error="Timeout on Modal")
        mock_store = MagicMock()
        mock_store.client = "not-none"
        mock_store.get_job.return_value = job

        with patch("bilancio.jobs.supabase_store.SupabaseJobStore", return_value=mock_store):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "get", "error-job", "--cloud"])
            assert result.exit_code == 0
            assert "Timeout on Modal" in result.output

    def test_get_job_with_many_runs(self):
        """Job with >10 runs shows truncation message."""
        run_ids = [f"run-{i:03d}" for i in range(15)]
        job = _make_job("many-runs-job", run_ids=run_ids)
        mock_store = MagicMock()
        mock_store.client = "not-none"
        mock_store.get_job.return_value = job

        with patch("bilancio.jobs.supabase_store.SupabaseJobStore", return_value=mock_store):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "get", "many-runs-job", "--cloud"])
            assert result.exit_code == 0
            assert "run-000" in result.output
            assert "and 5 more" in result.output

    def test_get_no_flags_fallback_local(self):
        """No --cloud/--local: tries Supabase then falls back to local paths."""
        job = _make_job("fallback-job")
        mock_manager = MagicMock()
        mock_manager.get_job.return_value = job

        with (
            patch(
                "bilancio.storage.supabase_client.is_supabase_configured", return_value=False
            ),
            patch("bilancio.jobs.JobManager", return_value=mock_manager),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "get", "fallback-job"])
            assert result.exit_code == 0
            assert "fallback-job" in result.output

    def test_get_not_found_anywhere(self):
        """Job not found via cloud or local raises error."""
        mock_manager = MagicMock()
        mock_manager.get_job.return_value = None

        with (
            patch(
                "bilancio.storage.supabase_client.is_supabase_configured", return_value=False
            ),
            patch("bilancio.jobs.JobManager", return_value=mock_manager),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "get", "phantom-job"])
            assert result.exit_code != 0
            assert "Job not found" in result.output


# ---------------------------------------------------------------------------
# jobs runs
# ---------------------------------------------------------------------------

class TestJobsRuns:
    """Tests for the 'jobs runs' command."""

    def test_runs_no_cloud_flag(self):
        """Without --cloud shows a note about Supabase."""
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "runs", "some-job"])
        assert result.exit_code == 0
        assert "Use --cloud" in result.output or "Note:" in result.output

    def test_runs_cloud_with_entries(self):
        """--cloud returns formatted table of runs."""
        entries = [
            _make_registry_entry("run-passive-001", delta_total=0.25),
            _make_registry_entry("run-active-001", delta_total=0.10),
        ]
        mock_store = MagicMock()
        mock_store.query.return_value = entries

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "runs", "test-job", "--cloud"])
            assert result.exit_code == 0
            assert "run-passive-001" in result.output
            assert "run-active-001" in result.output
            assert "0.2500" in result.output
            assert "Total: 2 runs" in result.output

    def test_runs_cloud_empty(self):
        """--cloud with no runs shows 'No runs found'."""
        mock_store = MagicMock()
        mock_store.query.return_value = []

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "runs", "empty-job", "--cloud"])
            assert result.exit_code == 0
            assert "No runs found" in result.output

    def test_runs_cloud_with_status_filter(self):
        """--cloud --status completed passes filter to store."""
        entries = [_make_registry_entry("run-001")]
        mock_store = MagicMock()
        mock_store.query.return_value = entries

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["jobs", "runs", "test-job", "--cloud", "--status", "completed"]
            )
            assert result.exit_code == 0
            mock_store.query.assert_called_once_with("test-job", {"status": "completed"})

    def test_runs_cloud_error(self):
        """--cloud with store error raises ClickException."""
        mock_store = MagicMock()
        mock_store.query.side_effect = RuntimeError("Connection failed")

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "runs", "test-job", "--cloud"])
            assert result.exit_code != 0
            assert "Failed to query runs" in result.output

    def test_runs_entry_without_delta(self):
        """Runs with missing delta_total display '-'."""
        entry = _make_registry_entry("run-no-delta", delta_total=None)
        entry.metrics = {}  # No metrics at all
        entry.parameters = {}  # No parameters either
        mock_store = MagicMock()
        mock_store.query.return_value = [entry]

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "runs", "test-job", "--cloud"])
            assert result.exit_code == 0
            assert "run-no-delta" in result.output


# ---------------------------------------------------------------------------
# jobs metrics
# ---------------------------------------------------------------------------

class TestJobsMetrics:
    """Tests for the 'jobs metrics' command."""

    def test_metrics_no_cloud_flag(self):
        """Without --cloud shows a hint."""
        runner = CliRunner()
        result = runner.invoke(cli, ["jobs", "metrics", "some-job"])
        assert result.exit_code == 0
        assert "Use --cloud" in result.output

    def test_metrics_cloud_with_data(self):
        """--cloud shows aggregate metrics from completed runs."""
        entries = [
            _make_registry_entry("run-1", delta_total=0.10),
            _make_registry_entry("run-2", delta_total=0.20),
            _make_registry_entry("run-3", delta_total=0.30),
        ]
        mock_store = MagicMock()
        mock_store.query.return_value = entries

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "metrics", "test-job", "--cloud"])
            assert result.exit_code == 0
            assert "Delta" in result.output
            assert "Mean" in result.output
            assert "Min" in result.output
            assert "Max" in result.output
            assert "Phi" in result.output
            assert "Completed runs: 3" in result.output

    def test_metrics_cloud_empty(self):
        """--cloud with no runs shows 'No runs found'."""
        mock_store = MagicMock()
        mock_store.query.return_value = []

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "metrics", "empty-job", "--cloud"])
            assert result.exit_code == 0
            assert "No runs found" in result.output

    def test_metrics_cloud_no_completed(self):
        """--cloud with no completed runs shows appropriate message."""
        entries = [_make_registry_entry("run-1", status=RunStatus.FAILED, delta_total=None)]
        entries[0].metrics = {}
        mock_store = MagicMock()
        mock_store.query.return_value = entries

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "metrics", "test-job", "--cloud"])
            assert result.exit_code == 0
            assert "No completed runs" in result.output

    def test_metrics_cloud_error(self):
        """--cloud with store error raises ClickException."""
        mock_store = MagicMock()
        mock_store.query.side_effect = ConnectionError("Network error")

        with patch(
            "bilancio.storage.supabase_registry.SupabaseRegistryStore",
            return_value=mock_store,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "metrics", "test-job", "--cloud"])
            assert result.exit_code != 0
            assert "Failed to query metrics" in result.output


# ---------------------------------------------------------------------------
# jobs visualize
# ---------------------------------------------------------------------------

class TestJobsVisualize:
    """Tests for the 'jobs visualize' command."""

    def test_visualize_success(self, tmp_path):
        """Visualize generates HTML and reports the path."""
        html_path = tmp_path / "viz.html"

        with patch(
            "bilancio.analysis.visualization.run_comparison.generate_comparison_html",
            return_value=html_path,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["jobs", "visualize", "test-job", "-o", str(html_path)],
            )
            assert result.exit_code == 0
            assert "Visualization saved" in result.output

    def test_visualize_value_error(self):
        """Visualize raises ClickException on ValueError."""
        with patch(
            "bilancio.analysis.visualization.run_comparison.generate_comparison_html",
            side_effect=ValueError("No comparison data found"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "visualize", "bad-job"])
            assert result.exit_code != 0
            assert "No comparison data found" in result.output

    def test_visualize_runtime_error(self):
        """Visualize raises ClickException on RuntimeError."""
        with patch(
            "bilancio.analysis.visualization.run_comparison.generate_comparison_html",
            side_effect=RuntimeError("plotly not found"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["jobs", "visualize", "bad-job"])
            assert result.exit_code != 0
            assert "Failed to generate visualization" in result.output

    def test_visualize_with_open_browser(self, tmp_path):
        """Visualize with --open-browser calls webbrowser.open."""
        html_path = tmp_path / "viz.html"

        with (
            patch(
                "bilancio.analysis.visualization.run_comparison.generate_comparison_html",
                return_value=html_path,
            ),
            patch("webbrowser.open") as mock_browser,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["jobs", "visualize", "test-job", "-o", str(html_path), "--open-browser"],
            )
            assert result.exit_code == 0
            assert "Opened in browser" in result.output
            mock_browser.assert_called_once()
