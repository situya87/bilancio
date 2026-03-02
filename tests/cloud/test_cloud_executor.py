"""Unit tests for CloudExecutor (mocked Modal)."""

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
        assert config.max_parallel == 10
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
