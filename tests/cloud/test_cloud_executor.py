"""Unit tests for CloudExecutor (mocked Modal)."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

from bilancio.runners.cloud_executor import CloudExecutor
from bilancio.runners.models import RunOptions
from bilancio.storage.models import RunStatus


class TestCloudExecutor:
    """Unit tests for CloudExecutor with mocked Modal functions."""

    def test_execute_success(self, tmp_path):
        """Test successful cloud execution."""
        # Setup mock function
        mock_func = MagicMock()
        mock_func.remote.return_value = {
            "run_id": "test_001",
            "status": "completed",
            "storage_type": "modal_volume",
            "storage_base": "exp/runs/test_001",
            "artifacts": {
                "events_jsonl": "out/events.jsonl",
                "balances_csv": "out/balances.csv",
            },
            "execution_time_ms": 5000,
            "error": None,
        }

        # Mock modal module
        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_func

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(
                experiment_id="exp",
                download_artifacts=False,  # Don't try to download
            )

            result = executor.execute(
                scenario_config={"agents": []},
                run_id="test_001",
                output_dir=tmp_path / "test_001",
                options=RunOptions(),
            )

        assert result.status == RunStatus.COMPLETED
        assert result.run_id == "test_001"
        assert "events_jsonl" in result.artifacts

    def test_execute_failure(self, tmp_path):
        """Test failed cloud execution."""
        mock_func = MagicMock()
        mock_func.remote.return_value = {
            "run_id": "test_001",
            "status": "failed",
            "storage_type": "modal_volume",
            "storage_base": "exp/runs/test_001",
            "artifacts": {},
            "execution_time_ms": 1000,
            "error": "Simulation diverged",
        }

        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_func

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(
                experiment_id="exp",
                download_artifacts=False,
            )

            result = executor.execute(
                scenario_config={"agents": []},
                run_id="test_001",
                output_dir=tmp_path / "test_001",
                options=RunOptions(),
            )

        assert result.status == RunStatus.FAILED
        assert result.error == "Simulation diverged"

    def test_options_serialization(self, tmp_path):
        """Test that RunOptions are properly serialized."""
        mock_func = MagicMock()
        mock_func.remote.return_value = {
            "run_id": "test_001",
            "status": "completed",
            "storage_type": "modal_volume",
            "storage_base": "exp/runs/test_001",
            "artifacts": {},
            "execution_time_ms": 1000,
            "error": None,
        }

        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_func

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(
                experiment_id="exp",
                download_artifacts=False,
            )

            options = RunOptions(
                mode="fixed_days",
                max_days=10,
                quiet_days=3,
                check_invariants="end",
            )

            executor.execute(
                scenario_config={"agents": []},
                run_id="test_001",
                output_dir=tmp_path / "test_001",
                options=options,
            )

        # Verify the options were serialized correctly
        call_kwargs = mock_func.remote.call_args[1]
        assert call_kwargs["options"]["mode"] == "fixed_days"
        assert call_kwargs["options"]["max_days"] == 10
        assert call_kwargs["options"]["quiet_days"] == 3
        assert call_kwargs["options"]["check_invariants"] == "end"


class TestCloudConfig:
    """Tests for CloudConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bilancio.cloud.config import CloudConfig

        config = CloudConfig()
        assert config.volume_name == "bilancio-results"
        assert config.timeout_seconds == 600
        assert config.memory_mb == 2048
        assert config.max_parallel == 7
        assert config.gpu is None

    def test_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        from bilancio.cloud.config import CloudConfig

        monkeypatch.setenv("BILANCIO_MODAL_VOLUME", "custom-volume")
        monkeypatch.setenv("BILANCIO_CLOUD_TIMEOUT", "1200")
        monkeypatch.setenv("BILANCIO_CLOUD_MEMORY", "4096")
        monkeypatch.setenv("BILANCIO_CLOUD_MAX_PARALLEL", "24")

        config = CloudConfig()
        assert config.volume_name == "custom-volume"
        assert config.timeout_seconds == 1200
        assert config.memory_mb == 4096
        assert config.max_parallel == 24


# ── New tests below ──────────────────────────────────────────────────────────


class TestOptionsToDict:
    """Tests for CloudExecutor._options_to_dict() serialization."""

    def _make_executor(self):
        return CloudExecutor(experiment_id="exp", download_artifacts=False)

    def test_basic_fields(self):
        """All mandatory fields are serialized."""
        executor = self._make_executor()
        options = RunOptions(
            mode="fixed_days",
            max_days=20,
            quiet_days=5,
            check_invariants="end",
            default_handling="expel-agent",
            show_events="summary",
            t_account=True,
            detailed_dealer_logging=True,
            regime="passive",
        )
        d = executor._options_to_dict(options)
        assert d["mode"] == "fixed_days"
        assert d["max_days"] == 20
        assert d["quiet_days"] == 5
        assert d["check_invariants"] == "end"
        assert d["default_handling"] == "expel-agent"
        assert d["show_events"] == "summary"
        assert d["t_account"] is True
        assert d["detailed_dealer_logging"] is True
        assert d["regime"] == "passive"

    def test_optional_sweep_params_included(self):
        """kappa, concentration, mu, outside_mid_ratio, seed are included when set."""
        executor = self._make_executor()
        options = RunOptions(
            kappa=0.5,
            concentration=1.0,
            mu=0.25,
            outside_mid_ratio=0.9,
            seed=42,
        )
        d = executor._options_to_dict(options)
        assert d["kappa"] == 0.5
        assert d["concentration"] == 1.0
        assert d["mu"] == 0.25
        assert d["outside_mid_ratio"] == 0.9
        assert d["seed"] == 42

    def test_optional_sweep_params_omitted_when_none(self):
        """Optional sweep params are absent when not set."""
        executor = self._make_executor()
        options = RunOptions()
        d = executor._options_to_dict(options)
        assert "kappa" not in d
        assert "concentration" not in d
        assert "mu" not in d
        assert "outside_mid_ratio" not in d
        assert "seed" not in d

    def test_regime_empty_string_when_none(self):
        """Regime is serialized as empty string when None."""
        executor = self._make_executor()
        options = RunOptions(regime=None)
        d = executor._options_to_dict(options)
        assert d["regime"] == ""

    def test_performance_config_serialized(self):
        """PerformanceConfig is serialized via to_dict()."""
        executor = self._make_executor()
        mock_perf = MagicMock()
        mock_perf.to_dict.return_value = {"skip_html": True, "skip_balances": False}
        options = RunOptions(performance=mock_perf)
        d = executor._options_to_dict(options)
        assert d["performance"] == {"skip_html": True, "skip_balances": False}

    def test_show_balances_serialized(self):
        """show_balances list is included when set."""
        executor = self._make_executor()
        options = RunOptions(show_balances=["agent_0", "agent_1"])
        d = executor._options_to_dict(options)
        assert d["show_balances"] == ["agent_0", "agent_1"]


class TestExecuteBatch:
    """Tests for CloudExecutor.execute_batch() with mocked Modal .map()."""

    def _make_result_dict(self, run_id, status="completed"):
        return {
            "run_id": run_id,
            "status": status,
            "storage_type": "modal_volume",
            "storage_base": f"exp/runs/{run_id}",
            "artifacts": {"events_jsonl": "out/events.jsonl"},
            "execution_time_ms": 3000,
            "error": None if status == "completed" else "boom",
        }

    def test_batch_maps_all_runs(self, tmp_path):
        """execute_batch sends all runs via .map() and returns results in order."""
        mock_func = MagicMock()
        # .map() returns results in arbitrary order
        mock_func.map.return_value = iter([
            self._make_result_dict("run_b"),
            self._make_result_dict("run_a"),
            self._make_result_dict("run_c"),
        ])

        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_func

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(
                experiment_id="exp",
                download_artifacts=False,
            )

            runs = [
                ({"agents": []}, "run_a", RunOptions()),
                ({"agents": []}, "run_b", RunOptions()),
                ({"agents": []}, "run_c", RunOptions()),
            ]

            results = executor.execute_batch(runs)

        # Results should be re-ordered to match input order
        assert results[0].run_id == "run_a"
        assert results[1].run_id == "run_b"
        assert results[2].run_id == "run_c"

    def test_batch_progress_callback_count(self, tmp_path):
        """Progress callback is called once per completed run."""
        mock_func = MagicMock()
        mock_func.map.return_value = iter([
            self._make_result_dict("run_a"),
            self._make_result_dict("run_b"),
        ])

        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_func

        progress_calls = []

        def track_progress(completed, total):
            progress_calls.append((completed, total))

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(experiment_id="exp", download_artifacts=False)

            runs = [
                ({"agents": []}, "run_a", RunOptions()),
                ({"agents": []}, "run_b", RunOptions()),
            ]

            executor.execute_batch(runs, progress_callback=track_progress)

        assert len(progress_calls) == 2
        # Each call has (completed_count, total)
        assert progress_calls[-1] == (2, 2)

    def test_batch_with_failed_run(self, tmp_path):
        """Batch handles mix of completed and failed runs."""
        mock_func = MagicMock()
        mock_func.map.return_value = iter([
            self._make_result_dict("run_a", status="completed"),
            self._make_result_dict("run_b", status="failed"),
        ])

        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_func

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(experiment_id="exp", download_artifacts=False)

            runs = [
                ({"agents": []}, "run_a", RunOptions()),
                ({"agents": []}, "run_b", RunOptions()),
            ]

            results = executor.execute_batch(runs)

        assert results[0].status == RunStatus.COMPLETED
        assert results[1].status == RunStatus.FAILED
        assert results[1].error == "boom"


class TestDownloadRunArtifacts:
    """Tests for CloudExecutor._download_run_artifacts()."""

    def test_successful_download(self, tmp_path):
        """Download calls subprocess.run for each artifact."""
        executor = CloudExecutor(
            experiment_id="exp",
            download_artifacts=True,
            volume_name="bilancio-results",
        )

        artifacts = {
            "events_jsonl": "out/events.jsonl",
            "balances_csv": "out/balances.csv",
        }

        with patch("bilancio.runners.cloud_executor.subprocess.run") as mock_run:
            executor._download_run_artifacts("run_001", tmp_path, artifacts)

        assert mock_run.call_count == 2
        # Verify each call used the correct remote path
        all_calls = mock_run.call_args_list
        remote_paths = [c[0][0][4] for c in all_calls]  # 5th arg in the command list
        assert "exp/runs/run_001/out/events.jsonl" in remote_paths
        assert "exp/runs/run_001/out/balances.csv" in remote_paths

    def test_download_creates_dirs(self, tmp_path):
        """Download creates the output directory and subdirectories."""
        output_dir = tmp_path / "new_dir"
        executor = CloudExecutor(experiment_id="exp", download_artifacts=True)

        artifacts = {"events_jsonl": "out/events.jsonl"}

        with patch("bilancio.runners.cloud_executor.subprocess.run"):
            executor._download_run_artifacts("run_001", output_dir, artifacts)

        assert output_dir.exists()
        assert (output_dir / "out").exists()

    def test_failed_download_prints_warning(self, tmp_path, capsys):
        """Failed download prints a warning but does not raise."""
        executor = CloudExecutor(experiment_id="exp", download_artifacts=True)

        artifacts = {"events_jsonl": "out/events.jsonl"}

        with patch(
            "bilancio.runners.cloud_executor.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "modal", stderr=b"Volume not found"
            ),
        ):
            # Should not raise
            executor._download_run_artifacts("run_001", tmp_path, artifacts)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "events_jsonl" in captured.out


class TestComputeAggregateMetrics:
    """Tests for CloudExecutor.compute_aggregate_metrics()."""

    def test_successful_aggregation(self, capsys):
        """Returns result dict when aggregation succeeds."""
        mock_aggregate_func = MagicMock()
        mock_aggregate_func.remote.return_value = {
            "status": "completed",
            "summary": {
                "n_comparisons": 10,
                "mean_trading_effect": 0.0325,
            },
        }

        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_aggregate_func

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(
                experiment_id="exp",
                download_artifacts=False,
                job_id="test-job-id",
            )
            result = executor.compute_aggregate_metrics(["run_a", "run_b"])

        assert result["status"] == "completed"
        assert result["summary"]["n_comparisons"] == 10
        captured = capsys.readouterr()
        assert "Aggregate metrics computed" in captured.out

    def test_failed_aggregation(self, capsys):
        """Prints warning when aggregation fails."""
        mock_aggregate_func = MagicMock()
        mock_aggregate_func.remote.return_value = {
            "status": "failed",
            "error": "No data found",
        }

        mock_modal = MagicMock()
        mock_modal.Function.from_name.return_value = mock_aggregate_func

        with patch.dict(sys.modules, {"modal": mock_modal}):
            executor = CloudExecutor(
                experiment_id="exp",
                download_artifacts=False,
                job_id="test-job-id",
            )
            result = executor.compute_aggregate_metrics(["run_x"])

        assert result["status"] == "failed"
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "No data found" in captured.out
