"""Tests for bilancio.ui.cli.treynor CLI command."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

treynor_mod = pytest.importorskip(
    "bilancio.ui.cli.treynor",
    reason="treynor CLI module not available",
)
treynor = treynor_mod.treynor


class TestTreynorCommand:
    """Test the treynor CLI command directly (not registered in main CLI)."""

    @patch("webbrowser.open")
    @patch("bilancio.analysis.treynor_viz.build_treynor_dashboard")
    def test_treynor_default_output(self, mock_build, mock_wb_open, tmp_path):
        """treynor --run-dir writes to <run-dir>/treynor_dashboard.html by default."""
        run_dir = tmp_path / "run_output"
        run_dir.mkdir()
        mock_build.return_value = "<html>dashboard</html>"

        runner = CliRunner()
        result = runner.invoke(treynor, ["--run-dir", str(run_dir)])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_build.assert_called_once_with(run_dir)
        output_file = run_dir / "treynor_dashboard.html"
        assert output_file.exists()
        assert output_file.read_text() == "<html>dashboard</html>"
        assert "OK" in result.output
        assert "treynor_dashboard.html" in result.output

    @patch("webbrowser.open")
    @patch("bilancio.analysis.treynor_viz.build_treynor_dashboard")
    def test_treynor_custom_output(self, mock_build, mock_wb_open, tmp_path):
        """treynor --run-dir --output writes to specified path."""
        run_dir = tmp_path / "run_output"
        run_dir.mkdir()
        out_file = tmp_path / "custom" / "dash.html"
        mock_build.return_value = "<html>custom</html>"

        runner = CliRunner()
        result = runner.invoke(
            treynor,
            ["--run-dir", str(run_dir), "-o", str(out_file)],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert out_file.exists()
        assert out_file.read_text() == "<html>custom</html>"

    @patch("webbrowser.open", side_effect=OSError("no browser"))
    @patch("bilancio.analysis.treynor_viz.build_treynor_dashboard")
    def test_treynor_browser_open_failure(self, mock_build, mock_wb_open, tmp_path):
        """treynor gracefully handles browser open failure."""
        run_dir = tmp_path / "run_output"
        run_dir.mkdir()
        mock_build.return_value = "<html></html>"

        runner = CliRunner()
        result = runner.invoke(treynor, ["--run-dir", str(run_dir)])

        # Should succeed even if browser fails to open
        assert result.exit_code == 0, f"Failed with output:\n{result.output}"

    def test_treynor_missing_run_dir(self):
        """treynor without --run-dir fails."""
        runner = CliRunner()
        result = runner.invoke(treynor, [])
        assert result.exit_code != 0

    def test_treynor_nonexistent_run_dir(self, tmp_path):
        """treynor with non-existent --run-dir fails."""
        runner = CliRunner()
        result = runner.invoke(treynor, ["--run-dir", str(tmp_path / "nope")])
        assert result.exit_code != 0

    @patch("webbrowser.open")
    @patch("bilancio.analysis.treynor_viz.build_treynor_dashboard")
    def test_treynor_creates_parent_dirs(self, mock_build, mock_wb_open, tmp_path):
        """treynor creates parent directories for output file."""
        run_dir = tmp_path / "run_output"
        run_dir.mkdir()
        out_file = tmp_path / "deep" / "nested" / "dir" / "dash.html"
        mock_build.return_value = "<html></html>"

        runner = CliRunner()
        result = runner.invoke(
            treynor,
            ["--run-dir", str(run_dir), "-o", str(out_file)],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert out_file.exists()
