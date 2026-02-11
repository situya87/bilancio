"""Artifact loader for Modal Volume storage."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class ModalVolumeArtifactLoader:
    """Load artifacts from a Modal Volume.

    Uses the Modal CLI to download files on demand.
    Implements caching to avoid repeated downloads.

    Example:
        loader = ModalVolumeArtifactLoader(
            volume_name="bilancio-results",
            base_path="my_experiment/runs/run_001",
        )
        events_text = loader.load_text("out/events.jsonl")
        html_bytes = loader.load_bytes("run.html")
    """

    def __init__(
        self,
        volume_name: str = "bilancio-results",
        base_path: str = "",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize loader.

        Args:
            volume_name: Name of the Modal Volume.
            base_path: Base path within the volume (e.g., "experiment_001/runs/run_001").
            cache_dir: Local directory for caching downloads. If None, uses a
                temporary directory.
        """
        self.volume_name = volume_name
        self.base_path = base_path
        self._cache_dir = cache_dir
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory, creating it if necessary."""
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            return self._cache_dir

        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="modal_cache_")
        return Path(self._temp_dir.name)

    def load_bytes(self, reference: str) -> bytes:
        """Load artifact as bytes.

        Args:
            reference: Relative path to artifact within base_path.

        Returns:
            File contents as bytes.

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
            subprocess.CalledProcessError: If download fails.
        """
        local_path = self._ensure_downloaded(reference)
        return local_path.read_bytes()

    def load_text(self, reference: str, encoding: str = "utf-8") -> str:
        """Load artifact as text.

        Args:
            reference: Relative path to artifact within base_path.
            encoding: Text encoding (default: utf-8).

        Returns:
            File contents as string.

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
            subprocess.CalledProcessError: If download fails.
        """
        local_path = self._ensure_downloaded(reference)
        return local_path.read_text(encoding=encoding)

    def exists(self, reference: str) -> bool:
        """Check if artifact exists on volume.

        Args:
            reference: Relative path to artifact within base_path.

        Returns:
            True if the file exists, False otherwise.
        """
        remote_path = f"{self.base_path}/{reference}" if self.base_path else reference

        result = subprocess.run(
            ["modal", "volume", "ls", self.volume_name, remote_path],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def get_local_path(self, reference: str) -> Path:
        """Get local path to cached artifact, downloading if necessary.

        Args:
            reference: Relative path to artifact within base_path.

        Returns:
            Path to the locally cached file.
        """
        return self._ensure_downloaded(reference)

    def _ensure_downloaded(self, reference: str) -> Path:
        """Download artifact if not already cached.

        Args:
            reference: Relative path to artifact within base_path.

        Returns:
            Path to the locally cached file.

        Raises:
            subprocess.CalledProcessError: If download fails.
        """
        # Create cache path maintaining directory structure
        cache_path = self.cache_dir / reference

        if cache_path.exists():
            return cache_path

        # Ensure parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Download from Modal Volume
        remote_path = f"{self.base_path}/{reference}" if self.base_path else reference

        subprocess.run(
            ["modal", "volume", "get", self.volume_name, remote_path, str(cache_path)],
            check=True,
            capture_output=True,
        )

        return cache_path

    def clear_cache(self) -> None:
        """Clear all cached files."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        elif self._cache_dir is not None and self._cache_dir.exists():
            import shutil

            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self) -> None:
        """Cleanup temporary directory on deletion."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
