"""Tests for CLI functionality."""

import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner

from bilancio.ui.cli import cli


class TestCLI:
    """Test CLI commands."""

    def test_cli_help(self):
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Bilancio" in result.output
        assert "simulation" in result.output.lower()

    def test_run_help(self):
        """Test that run command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "scenario" in result.output.lower()
        assert "--max-days" in result.output
        assert "--mode" in result.output

    def test_new_help(self):
        """Test that new command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["new", "--help"])
        assert result.exit_code == 0
        assert "scenario" in result.output.lower()
        assert "--output" in result.output or "-o" in result.output

    def test_run_nonexistent_file(self):
        """Test running with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_run_simple_scenario(self):
        """Test running a simple scenario."""
        # Create a minimal scenario file
        scenario = {
            "version": 1,
            "name": "Test Scenario",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "B1", "kind": "bank", "name": "Bank"},
            ],
            "initial_actions": [{"mint_reserves": {"to": "B1", "amount": 1000}}],
            "run": {"mode": "until_stable", "max_days": 5, "quiet_days": 1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(scenario, f)
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["run", str(temp_path), "--mode", "until-stable", "--max-days", "5"]
            )

            # Check that it ran without crashing
            assert result.exit_code == 0
            assert "Test Scenario" in result.output
            assert "Day" in result.output

        finally:
            temp_path.unlink()

    def test_new_scenario_creation(self):
        """Test creating a new scenario file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "new_scenario.yaml"

            runner = CliRunner()
            # Use the template option to avoid interactive prompts
            runner.invoke(cli, ["new", "-o", str(output_path), "--from", "simple"])

            # Check file was created
            assert output_path.exists()

            # Load and validate the created file
            with open(output_path) as f:
                config = yaml.safe_load(f)

            # When using template, it uses default name "My Scenario"
            assert config["name"] == "My Scenario"
            assert config["version"] == 1
            assert len(config["agents"]) > 0

    def test_run_with_export(self):
        """Test running scenario with export options."""
        scenario = {
            "version": 1,
            "name": "Export Test",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank"},
            ],
            "initial_actions": [{"mint_reserves": {"to": "B1", "amount": 1000}}],
            "run": {"mode": "until_stable", "max_days": 2},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create scenario file
            scenario_path = Path(tmpdir) / "scenario.yaml"
            with open(scenario_path, "w") as f:
                yaml.dump(scenario, f)

            # Set export paths
            balances_path = Path(tmpdir) / "balances.csv"
            events_path = Path(tmpdir) / "events.jsonl"

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "run",
                    str(scenario_path),
                    "--export-balances",
                    str(balances_path),
                    "--export-events",
                    str(events_path),
                    "--max-days",
                    "2",
                ],
            )

            assert result.exit_code == 0

            # Check export files were created
            assert balances_path.exists()
            assert events_path.exists()

            # Check files have content
            assert balances_path.stat().st_size > 0
            assert events_path.stat().st_size > 0

    def test_run_with_html_export(self, tmp_path):
        """Test running scenario with --html export option."""
        scenario = {
            "version": 1,
            "name": "HTML Export Test",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "H1", "kind": "household", "name": "Alice"},
            ],
            "initial_actions": [{"mint_cash": {"to": "H1", "amount": 1000}}],
            "run": {
                "mode": "until_stable",
                "max_days": 1,
                "quiet_days": 1,
                "show": {"balances": ["CB", "H1"]},
            },
        }

        scenario_path = tmp_path / "scenario.yaml"
        html_path = tmp_path / "out.html"

        import yaml

        with open(scenario_path, "w") as f:
            yaml.dump(scenario, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(scenario_path),
                "--max-days",
                "1",
                "--quiet-days",
                "1",
                "--html",
                str(html_path),
            ],
        )

        assert result.exit_code == 0
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "Bilancio Simulation" in content
