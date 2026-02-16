"""Tests for ModalVolumeArtifactLoader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from bilancio.storage.modal_artifact_loader import ModalVolumeArtifactLoader


class TestModalVolumeArtifactLoader:
    """Tests for ModalVolumeArtifactLoader."""

    def test_init_defaults(self):
        """Test default initialization."""
        loader = ModalVolumeArtifactLoader()
        assert loader.volume_name == "bilancio-results"
        assert loader.base_path == ""

    def test_init_custom(self, tmp_path):
        """Test custom initialization."""
        loader = ModalVolumeArtifactLoader(
            volume_name="custom-volume",
            base_path="exp/runs/run_001",
            cache_dir=tmp_path / "cache",
        )
        assert loader.volume_name == "custom-volume"
        assert loader.base_path == "exp/runs/run_001"

    @patch("bilancio.storage.modal_artifact_loader.subprocess.run")
    def test_load_text(self, mock_run, tmp_path):
        """Test loading text file."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create cached file
        test_file = cache_dir / "out" / "events.jsonl"
        test_file.parent.mkdir(parents=True)
        test_file.write_text('{"event": "test"}\n')

        loader = ModalVolumeArtifactLoader(
            volume_name="test-volume",
            base_path="exp/runs/run_001",
            cache_dir=cache_dir,
        )

        # File exists in cache, so no download should happen
        content = loader.load_text("out/events.jsonl")
        assert content == '{"event": "test"}\n'
        mock_run.assert_not_called()

    @patch("bilancio.storage.modal_artifact_loader.subprocess.run")
    def test_load_text_downloads_if_not_cached(self, mock_run, tmp_path):
        """Test that file is downloaded if not in cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Mock successful download
        def side_effect(cmd, check, capture_output):
            # Create the file when modal volume get is called
            target = Path(cmd[5])  # The local path argument
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("downloaded content")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        loader = ModalVolumeArtifactLoader(
            volume_name="test-volume",
            base_path="exp/runs/run_001",
            cache_dir=cache_dir,
        )

        content = loader.load_text("test.txt")
        assert content == "downloaded content"

        # Verify modal CLI was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "modal"
        assert call_args[1] == "volume"
        assert call_args[2] == "get"
        assert call_args[3] == "test-volume"
        assert "exp/runs/run_001/test.txt" in call_args[4]

    @patch("bilancio.storage.modal_artifact_loader.subprocess.run")
    def test_exists_true(self, mock_run):
        """Test exists returns True when file exists."""
        mock_run.return_value = MagicMock(returncode=0)

        loader = ModalVolumeArtifactLoader(
            volume_name="test-volume",
            base_path="exp/runs/run_001",
        )

        assert loader.exists("test.txt") is True

    @patch("bilancio.storage.modal_artifact_loader.subprocess.run")
    def test_exists_false(self, mock_run):
        """Test exists returns False when file doesn't exist."""
        mock_run.return_value = MagicMock(returncode=1)

        loader = ModalVolumeArtifactLoader(
            volume_name="test-volume",
            base_path="exp/runs/run_001",
        )

        assert loader.exists("nonexistent.txt") is False

    def test_clear_cache(self, tmp_path):
        """Test cache clearing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "test.txt").write_text("cached")

        loader = ModalVolumeArtifactLoader(cache_dir=cache_dir)
        loader.clear_cache()

        # Directory should still exist but be empty
        assert cache_dir.exists()
        assert list(cache_dir.iterdir()) == []
