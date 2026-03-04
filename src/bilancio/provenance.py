"""Run provenance collection for reproducibility tracking.

Captures environment metadata (git state, Python version, platform,
dependency fingerprint) to embed in job manifests.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from typing import Any


def _get_git_sha() -> str | None:
    """Return the current git commit SHA (40 hex chars), or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if len(sha) == 40:
                return sha
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _is_git_dirty() -> bool | None:
    """Return True if git working tree is dirty, None if git unavailable."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _compute_dep_fingerprint() -> str:
    """Compute SHA-256 of sorted installed package list.

    Uses importlib.metadata (works in uv, venv, conda, etc.)
    instead of ``pip freeze`` which may not be available.
    """
    try:
        from importlib.metadata import distributions

        pkgs = sorted(
            f"{d.metadata['Name']}=={d.metadata['Version']}"
            for d in distributions()
        )
        content = "\n".join(pkgs)
        return hashlib.sha256(content.encode()).hexdigest()
    except Exception:
        pass
    return ""


def collect_provenance() -> dict[str, Any]:
    """Collect environment provenance for reproducibility.

    Returns:
        Dict with git_sha, git_dirty, python_version, platform,
        cpu_count, bilancio_version, dep_fingerprint, timestamp_utc.
    """
    import os

    try:
        from bilancio import __version__ as bilancio_version
    except (ImportError, AttributeError):
        bilancio_version = "unknown"

    return {
        "git_sha": _get_git_sha(),
        "git_dirty": _is_git_dirty(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "bilancio_version": bilancio_version,
        "dep_fingerprint": _compute_dep_fingerprint(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
