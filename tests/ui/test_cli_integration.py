"""Integration tests for CLI module.

Tests the CLI commands work correctly with real scenario files.
Uses Click's CliRunner for testing.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from bilancio.ui.cli import cli

# Path to example scenarios
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "scenarios"


class TestCLIHelp:
    """Test CLI help output for all commands."""

    def test_main_help_shows_expected_commands(self):
        """Test that main --help shows all expected commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "run" in result.output
        assert "validate" in result.output
        assert "new" in result.output
        assert "analyze" in result.output
        assert "sweep" in result.output

    def test_validate_help(self):
        """Test that validate command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate" in result.output
        assert "scenario" in result.output.lower()

    def test_analyze_help(self):
        """Test that analyze command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "--events" in result.output
        assert "--balances" in result.output
        assert "--out-csv" in result.output

    def test_sweep_help_shows_subcommands(self):
        """Test that sweep --help shows sweep subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "--help"])

        assert result.exit_code == 0
        assert "ring" in result.output
        assert "comparison" in result.output
        assert "balanced" in result.output

    def test_sweep_ring_help_shows_options(self):
        """Test that sweep ring --help shows ring options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "ring", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--out-dir" in result.output
        assert "--kappas" in result.output
        assert "--n-agents" in result.output
        assert "--grid" in result.output

    def test_sweep_comparison_help(self):
        """Test that sweep comparison --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "comparison", "--help"])

        assert result.exit_code == 0
        assert "--out-dir" in result.output
        assert "--n-agents" in result.output
        assert "--dealer-share" in result.output

    def test_sweep_balanced_help(self):
        """Test that sweep balanced --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "balanced", "--help"])

        assert result.exit_code == 0
        assert "--out-dir" in result.output
        assert "--face-value" in result.output
        assert "--big-entity-share" in result.output


class TestValidateCommand:
    """Test the validate command with real scenarios."""

    def test_validate_simple_bank_scenario(self):
        """Test validating the simple_bank.yaml scenario."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(scenario_path)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_two_banks_scenario(self):
        """Test validating the two_banks_interbank.yaml scenario."""
        scenario_path = EXAMPLES_DIR / "two_banks_interbank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(scenario_path)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_nonexistent_file(self):
        """Test validate with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "nonexistent_file.yaml"])

        assert result.exit_code != 0


class TestRunCommand:
    """Test the run command with real scenarios."""

    def test_run_simple_bank_with_max_days(self):
        """Test running simple_bank.yaml with --max-days limit."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "3"])

        assert result.exit_code == 0
        # Check that the scenario name appears
        assert "Simple Banking System" in result.output

    def test_run_with_show_summary(self):
        """Test running with --show summary option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "2", "--show", "summary"])

        assert result.exit_code == 0

    def test_run_with_agent_filter(self):
        """Test running with --agents filter option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "2", "--agents", "CB,B1"])

        assert result.exit_code == 0

    def test_run_with_check_invariants_none(self):
        """Test running with --check-invariants none."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "2", "--check-invariants", "none"])

        assert result.exit_code == 0

    def test_run_t_account_option(self):
        """Test running with --t-account option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "2", "--t-account"])

        assert result.exit_code == 0


class TestRunWithExport:
    """Test run command with export options."""

    def test_run_with_export_balances(self, tmp_path):
        """Test running with --export-balances option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        balances_file = tmp_path / "balances.csv"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", str(scenario_path), "--max-days", "2", "--export-balances", str(balances_file)],
        )

        assert result.exit_code == 0
        assert balances_file.exists()
        assert balances_file.stat().st_size > 0

    def test_run_with_export_events(self, tmp_path):
        """Test running with --export-events option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        events_file = tmp_path / "events.jsonl"

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "2", "--export-events", str(events_file)])

        assert result.exit_code == 0
        assert events_file.exists()
        assert events_file.stat().st_size > 0

    def test_run_with_html_export(self, tmp_path):
        """Test running with --html export option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        html_file = tmp_path / "output.html"

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "2", "--html", str(html_file)])

        assert result.exit_code == 0
        assert html_file.exists()
        content = html_file.read_text()
        assert "html" in content.lower()


class TestErrorHandling:
    """Test CLI error handling."""

    def test_run_invalid_mode(self):
        """Test running with invalid --mode option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--mode", "invalid-mode"])

        # Click should reject invalid choice
        assert result.exit_code != 0

    def test_run_invalid_show(self):
        """Test running with invalid --show option."""
        scenario_path = EXAMPLES_DIR / "simple_bank.yaml"
        if not scenario_path.exists():
            pytest.skip(f"Scenario file not found: {scenario_path}")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(scenario_path), "--show", "invalid-show"])

        # Click should reject invalid choice
        assert result.exit_code != 0

    def test_unknown_command(self):
        """Test invoking unknown command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["unknown-command"])

        assert result.exit_code != 0

    def test_sweep_unknown_subcommand(self):
        """Test invoking unknown sweep subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "unknown-subcommand"])

        assert result.exit_code != 0
