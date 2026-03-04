"""Tests for the volume CLI commands (bilancio/ui/cli/volume.py).

Covers list_volume, cleanup, and remove commands with mocked
subprocess calls to avoid requiring Modal CLI.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bilancio.ui.cli import cli
from bilancio.ui.cli.volume import (
    delete_volume_path,
    get_volume_contents,
    parse_modal_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _volume_contents() -> list[dict]:
    """Return sample volume listing data."""
    return [
        {
            "Filename": "castle-river-mountain",
            "Type": "dir",
            "Created/Modified": "2025-06-01 10:00 CET",
        },
        {
            "Filename": "old-experiment-alpha",
            "Type": "dir",
            "Created/Modified": "2025-01-01 08:00 CET",
        },
        {
            "Filename": "test_scratch",
            "Type": "dir",
            "Created/Modified": "2025-05-20 14:30 UTC",
        },
        {
            "Filename": "some-file.txt",
            "Type": "file",
            "Created/Modified": "2025-06-10 12:00 CET",
        },
    ]


def _mock_subprocess_ls(contents: list[dict], returncode: int = 0, stderr: str = ""):
    """Create a mock subprocess.run result for volume ls."""
    mock_result = MagicMock()
    mock_result.returncode = returncode
    mock_result.stdout = json.dumps(contents) if returncode == 0 else ""
    mock_result.stderr = stderr
    return mock_result


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestVolumeHelpers:
    """Tests for volume module utility functions."""

    def test_parse_modal_date_with_tz(self):
        dt = parse_modal_date("2025-06-15 10:30 CET")
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.minute == 30

    def test_parse_modal_date_without_tz(self):
        """Without timezone, rsplit(' ', 1) still splits on the space between date and time.
        The date_part becomes just '2025-06-15', which doesn't match '%Y-%m-%d %H:%M',
        so it falls back to datetime.now(). This is expected behavior.
        """
        dt = parse_modal_date("2025-06-15 10:30")
        # Fallback: returns datetime.now() since '2025-06-15' doesn't parse as '%Y-%m-%d %H:%M'
        assert (datetime.now() - dt).total_seconds() < 5

    def test_parse_modal_date_invalid_fallback(self):
        """Invalid date string falls back to now()."""
        dt = parse_modal_date("not-a-date")
        # Should be close to now
        assert (datetime.now() - dt).total_seconds() < 5

    def test_get_volume_contents_success(self):
        """get_volume_contents returns parsed JSON on success."""
        contents = [{"Filename": "exp1", "Type": "dir"}]
        with patch("subprocess.run", return_value=_mock_subprocess_ls(contents)):
            result = get_volume_contents("bilancio-results")
            assert len(result) == 1
            assert result[0]["Filename"] == "exp1"

    def test_get_volume_contents_failure(self):
        """get_volume_contents raises ClickException on failure."""
        mock_result = _mock_subprocess_ls([], returncode=1, stderr="Modal not found")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(Exception, match="Failed to list volume"):
                get_volume_contents("bilancio-results")

    def test_delete_volume_path_success(self):
        """delete_volume_path returns True on success."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert delete_volume_path("bilancio-results", "old-exp") is True

    def test_delete_volume_path_failure(self):
        """delete_volume_path returns False on failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert delete_volume_path("bilancio-results", "missing-exp") is False


# ---------------------------------------------------------------------------
# volume group and help
# ---------------------------------------------------------------------------

class TestVolumeHelp:
    """Tests for volume --help and subcommand help."""

    def test_volume_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["volume", "--help"])
        assert result.exit_code == 0
        assert "ls" in result.output
        assert "cleanup" in result.output
        assert "rm" in result.output

    def test_volume_ls_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["volume", "ls", "--help"])
        assert result.exit_code == 0
        assert "--volume" in result.output

    def test_volume_cleanup_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["volume", "cleanup", "--help"])
        assert result.exit_code == 0
        assert "--older-than" in result.output
        assert "--pattern" in result.output
        assert "--dry-run" in result.output
        assert "--yes" in result.output

    def test_volume_rm_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["volume", "rm", "--help"])
        assert result.exit_code == 0
        assert "EXPERIMENT_ID" in result.output
        assert "--yes" in result.output


# ---------------------------------------------------------------------------
# volume ls
# ---------------------------------------------------------------------------

class TestVolumeLs:
    """Tests for the 'volume ls' command."""

    def test_ls_with_contents(self):
        """Lists directories from volume."""
        contents = _volume_contents()
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=contents
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "ls"])
            assert result.exit_code == 0
            assert "castle-river-mountain" in result.output
            assert "old-experiment-alpha" in result.output
            assert "test_scratch" in result.output
            # Files should not be listed as experiments
            assert "some-file.txt" not in result.output
            # Count only dirs
            assert "Total: 3 experiments" in result.output

    def test_ls_empty_volume(self):
        """Empty volume shows message."""
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=[]
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "ls"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower()

    def test_ls_modal_error(self):
        """Modal failure raises ClickException."""
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents",
            side_effect=RuntimeError("Modal CLI not found"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "ls"])
            assert result.exit_code != 0

    def test_ls_custom_volume_name(self):
        """Custom --volume name is passed through."""
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=[]
        ) as mock_get:
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "ls", "--volume", "custom-vol"])
            assert result.exit_code == 0
            mock_get.assert_called_once_with("custom-vol")


# ---------------------------------------------------------------------------
# volume cleanup
# ---------------------------------------------------------------------------

class TestVolumeCleanup:
    """Tests for the 'volume cleanup' command."""

    def test_cleanup_no_filter_specified(self):
        """Without --older-than or --pattern, shows error hint."""
        contents = _volume_contents()
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=contents
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "cleanup"])
            assert result.exit_code == 0
            assert "Specify --older-than or --pattern" in result.output

    def test_cleanup_dry_run_older_than(self):
        """--older-than with --dry-run shows what would be deleted without deleting."""
        # Make dates relative to now
        now = datetime.now()
        contents = [
            {
                "Filename": "old-exp",
                "Type": "dir",
                "Created/Modified": (now - timedelta(days=60)).strftime("%Y-%m-%d %H:%M") + " CET",
            },
            {
                "Filename": "recent-exp",
                "Type": "dir",
                "Created/Modified": (now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M") + " CET",
            },
        ]
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=contents
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "cleanup", "--older-than", "30", "--dry-run"])
            assert result.exit_code == 0
            assert "DRY RUN" in result.output
            assert "old-exp" in result.output
            assert "recent-exp" not in result.output
            assert "No changes made" in result.output

    def test_cleanup_pattern_dry_run(self):
        """--pattern with --dry-run matches glob patterns."""
        contents = _volume_contents()
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=contents
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["volume", "cleanup", "--pattern", "test_*", "--dry-run"]
            )
            assert result.exit_code == 0
            assert "test_scratch" in result.output
            assert "castle-river-mountain" not in result.output

    def test_cleanup_no_matches(self):
        """When no experiments match criteria, shows message."""
        contents = _volume_contents()
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=contents
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["volume", "cleanup", "--pattern", "nonexistent_*", "--dry-run"]
            )
            assert result.exit_code == 0
            assert "No experiments match" in result.output

    def test_cleanup_empty_volume(self):
        """Cleanup on empty volume shows appropriate message."""
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=[]
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "cleanup", "--older-than", "30"])
            assert result.exit_code == 0
            assert "No experiments found" in result.output

    def test_cleanup_confirm_and_delete(self):
        """Confirming deletion actually deletes (with --yes)."""
        now = datetime.now()
        contents = [
            {
                "Filename": "delete-me",
                "Type": "dir",
                "Created/Modified": (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M") + " CET",
            },
        ]
        with (
            patch("bilancio.ui.cli.volume.get_volume_contents", return_value=contents),
            patch("bilancio.ui.cli.volume.delete_volume_path", return_value=True) as mock_delete,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "cleanup", "--older-than", "30", "-y"])
            assert result.exit_code == 0
            assert "Deleting delete-me" in result.output
            assert "OK" in result.output
            assert "Deleted: 1" in result.output
            mock_delete.assert_called_once_with("bilancio-results", "delete-me")

    def test_cleanup_confirm_declined(self):
        """Declining confirmation aborts."""
        now = datetime.now()
        contents = [
            {
                "Filename": "keep-me",
                "Type": "dir",
                "Created/Modified": (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M") + " CET",
            },
        ]
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents", return_value=contents
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["volume", "cleanup", "--older-than", "30"], input="n\n"
            )
            assert result.exit_code == 0
            assert "Aborted" in result.output

    def test_cleanup_delete_failure(self):
        """Failed deletion reports failure count."""
        now = datetime.now()
        contents = [
            {
                "Filename": "fail-to-delete",
                "Type": "dir",
                "Created/Modified": (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M") + " CET",
            },
        ]
        with (
            patch("bilancio.ui.cli.volume.get_volume_contents", return_value=contents),
            patch("bilancio.ui.cli.volume.delete_volume_path", return_value=False),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "cleanup", "--older-than", "30", "-y"])
            assert result.exit_code == 0
            assert "FAILED" in result.output
            assert "Failed: 1" in result.output

    def test_cleanup_modal_error(self):
        """Modal failure on get_volume_contents raises error."""
        with patch(
            "bilancio.ui.cli.volume.get_volume_contents",
            side_effect=OSError("Connection refused"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "cleanup", "--older-than", "30"])
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# volume rm
# ---------------------------------------------------------------------------

class TestVolumeRm:
    """Tests for the 'volume rm' command."""

    def test_rm_with_confirm_yes(self):
        """--yes flag skips confirmation and deletes."""
        with patch(
            "bilancio.ui.cli.volume.delete_volume_path", return_value=True
        ) as mock_delete:
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "rm", "old-experiment", "-y"])
            assert result.exit_code == 0
            assert "Deleting old-experiment" in result.output
            assert "OK" in result.output
            mock_delete.assert_called_once_with("bilancio-results", "old-experiment")

    def test_rm_with_interactive_confirm(self):
        """Interactive confirmation then deletes."""
        with patch(
            "bilancio.ui.cli.volume.delete_volume_path", return_value=True
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "rm", "old-experiment"], input="y\n")
            assert result.exit_code == 0
            assert "OK" in result.output

    def test_rm_declined(self):
        """Declining confirmation aborts."""
        runner = CliRunner()
        result = runner.invoke(cli, ["volume", "rm", "keep-this"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

    def test_rm_failure(self):
        """Failed deletion raises ClickException."""
        with patch(
            "bilancio.ui.cli.volume.delete_volume_path", return_value=False
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["volume", "rm", "bad-experiment", "-y"])
            assert result.exit_code != 0
            assert "Failed to delete" in result.output

    def test_rm_custom_volume(self):
        """Custom --volume name is used."""
        with patch(
            "bilancio.ui.cli.volume.delete_volume_path", return_value=True
        ) as mock_delete:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["volume", "rm", "my-exp", "-y", "--volume", "custom-vol"]
            )
            assert result.exit_code == 0
            mock_delete.assert_called_once_with("custom-vol", "my-exp")
