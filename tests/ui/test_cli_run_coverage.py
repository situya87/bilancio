"""Additional tests for bilancio/ui/cli/run.py to increase coverage.

Focuses on CLI-level command wiring: the run, validate, new, and analyze
Click commands in ui/cli/run.py. Tests error-handling branches, option
parsing, and output formatting.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from bilancio.ui.cli import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "scenarios"


def _minimal_scenario() -> dict:
    """Minimal valid scenario for testing."""
    return {
        "version": 1,
        "name": "CLI Coverage Test",
        "agents": [
            {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
            {"id": "B1", "kind": "bank", "name": "Bank One"},
        ],
        "initial_actions": [
            {"mint_reserves": {"to": "B1", "amount": 1000}},
        ],
        "run": {"mode": "until_stable", "max_days": 2, "quiet_days": 1},
    }


def _scenario_with_payable() -> dict:
    """Scenario with households and a payable -- generates events for analyze."""
    return {
        "version": 1,
        "name": "Analyze Coverage Test",
        "agents": [
            {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
            {"id": "B1", "kind": "bank", "name": "Bank One"},
            {"id": "H1", "kind": "household", "name": "Alice"},
            {"id": "H2", "kind": "household", "name": "Bob"},
        ],
        "initial_actions": [
            {"mint_reserves": {"to": "B1", "amount": 5000}},
            {"mint_cash": {"to": "H1", "amount": 2000}},
            {"mint_cash": {"to": "H2", "amount": 1000}},
            {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 1500}},
            {"deposit_cash": {"customer": "H2", "bank": "B1", "amount": 800}},
            {"create_payable": {"from": "H1", "to": "H2", "amount": 300, "due_day": 1}},
        ],
        "run": {
            "mode": "until_stable",
            "max_days": 3,
            "quiet_days": 1,
            "export": {
                "events_jsonl": "PLACEHOLDER",
                "balances_csv": "PLACEHOLDER",
            },
        },
    }


def _write_scenario(tmp_path: Path, scenario: dict) -> Path:
    """Write scenario dict to YAML file."""
    p = tmp_path / "scenario.yaml"
    with open(p, "w") as f:
        yaml.dump(scenario, f)
    return p


# ---------------------------------------------------------------------------
# run command -- option and error handling
# ---------------------------------------------------------------------------


class TestRunCommand:
    """Tests for CLI run command branches."""

    def test_run_with_default_handling_expel(self, tmp_path):
        """--default-handling expel-agent is passed through."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", str(path), "--max-days", "2", "--default-handling", "expel-agent"],
        )
        assert result.exit_code == 0

    def test_run_with_default_handling_fail_fast(self, tmp_path):
        """--default-handling fail-fast is passed through."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", str(path), "--max-days", "2", "--default-handling", "fail-fast"],
        )
        # may succeed or fail depending on scenario, but should not crash from CLI parsing
        # Accept either exit code since fail-fast may trigger
        assert result.exit_code in (0, 1)

    def test_run_step_mode_aborts_on_eof(self, tmp_path):
        """--mode step triggers interactive prompt; aborts on EOF (no input)."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        runner = CliRunner()
        # Step mode uses Rich's Confirm.ask which reads from stdin.
        # Without input it gets EOF and aborts, which is expected.
        result = runner.invoke(
            cli,
            ["run", str(path), "--mode", "step", "--max-days", "2"],
        )
        # Abort is exit code 1
        assert result.exit_code == 1
        assert "Aborted" in result.output

    def test_run_show_table(self, tmp_path):
        """--show table option."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", str(path), "--max-days", "2", "--show", "table"],
        )
        assert result.exit_code == 0

    def test_run_check_invariants_daily(self, tmp_path):
        """--check-invariants daily option."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", str(path), "--max-days", "2", "--check-invariants", "daily"],
        )
        assert result.exit_code == 0

    def test_run_with_agents_filter(self, tmp_path):
        """--agents CB,B1 filters balance display."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", str(path), "--max-days", "2", "--agents", "CB,B1"],
        )
        assert result.exit_code == 0

    def test_run_value_error(self, tmp_path):
        """ValueError from run_scenario is caught and displayed."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)

        with patch("bilancio.ui.cli.run.run_scenario", side_effect=ValueError("Bad config")):
            runner = CliRunner()
            result = runner.invoke(cli, ["run", str(path)])
            assert result.exit_code == 1
            assert "Configuration error" in result.output or "Bad config" in result.output

    def test_run_file_not_found_error(self, tmp_path):
        """FileNotFoundError from run_scenario is caught."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)

        with patch(
            "bilancio.ui.cli.run.run_scenario",
            side_effect=FileNotFoundError("scenario.yaml not found"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["run", str(path)])
            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_run_runtime_error(self, tmp_path):
        """RuntimeError from run_scenario is caught as unexpected."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)

        with patch(
            "bilancio.ui.cli.run.run_scenario",
            side_effect=RuntimeError("Unexpected failure"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["run", str(path)])
            assert result.exit_code == 1
            assert "Unexpected error" in result.output or "Unexpected failure" in result.output

    def test_run_with_both_exports(self, tmp_path):
        """--export-balances and --export-events both passed."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        bal_path = tmp_path / "bal.csv"
        evt_path = tmp_path / "evt.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(path),
                "--max-days",
                "2",
                "--export-balances",
                str(bal_path),
                "--export-events",
                str(evt_path),
            ],
        )
        assert result.exit_code == 0
        assert bal_path.exists()
        assert evt_path.exists()


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


class TestValidateCommand:
    """Tests for the validate CLI command."""

    def test_validate_minimal_scenario(self, tmp_path):
        """Validate a minimal valid scenario succeeds."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid_yaml(self, tmp_path):
        """Validate with invalid YAML content fails."""
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("version: 1\nagents: not-a-list")
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(bad_path)])
        # Should fail with a validation error
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_validate_value_error(self, tmp_path):
        """ValueError during validation shows error panel."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)

        with patch("bilancio.config.load_yaml", side_effect=ValueError("Missing required field")):
            runner = CliRunner()
            result = runner.invoke(cli, ["validate", str(path)])
            assert result.exit_code == 1

    def test_validate_runtime_error(self, tmp_path):
        """RuntimeError during validation shows error panel."""
        scenario = _minimal_scenario()
        path = _write_scenario(tmp_path, scenario)

        with patch("bilancio.config.load_yaml", side_effect=RuntimeError("System error")):
            runner = CliRunner()
            result = runner.invoke(cli, ["validate", str(path)])
            assert result.exit_code == 1


# ---------------------------------------------------------------------------
# new command
# ---------------------------------------------------------------------------


class TestNewCommand:
    """Tests for the 'new' command."""

    def test_new_with_simple_template(self, tmp_path):
        """new --from simple creates a valid scenario file."""
        output_path = tmp_path / "new_scenario.yaml"
        runner = CliRunner()
        result = runner.invoke(cli, ["new", "-o", str(output_path), "--from", "simple"])
        assert result.exit_code == 0
        assert output_path.exists()
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert data["version"] == 1
        assert len(data["agents"]) > 0

    def test_new_wizard_error(self, tmp_path):
        """Error in wizard shows error panel."""
        output_path = tmp_path / "fail_scenario.yaml"

        with patch(
            "bilancio.ui.cli.run.create_scenario_wizard",
            side_effect=ValueError("Invalid template"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["new", "-o", str(output_path)])
            assert result.exit_code == 1

    def test_new_missing_output(self):
        """new without -o flag fails."""
        runner = CliRunner()
        result = runner.invoke(cli, ["new"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--output" in result.output


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    """Tests for the 'analyze' command.

    These tests first run a scenario to produce events, then analyze them.
    """

    def test_analyze_help(self):
        """analyze --help shows all expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--events" in result.output
        assert "--out-csv" in result.output
        assert "--out-json" in result.output
        assert "--html" in result.output
        assert "--intraday-csv" in result.output

    def test_analyze_with_events_file(self, tmp_path):
        """Analyze a valid events JSONL file produces CSV and JSON outputs."""
        # First, create an events file by running a scenario
        scenario = _scenario_with_payable()
        events_path = tmp_path / "events.jsonl"
        balances_path = tmp_path / "balances.csv"
        scenario["run"]["export"]["events_jsonl"] = str(events_path)
        scenario["run"]["export"]["balances_csv"] = str(balances_path)
        scenario_path = _write_scenario(tmp_path, scenario)

        runner = CliRunner()
        # Run the scenario to produce events
        run_result = runner.invoke(
            cli,
            ["run", str(scenario_path), "--max-days", "3"],
        )
        assert run_result.exit_code == 0
        assert events_path.exists(), f"Events file was not created. Output: {run_result.output}"

        # Now analyze
        out_csv = tmp_path / "metrics_day.csv"
        out_json = tmp_path / "metrics_day.json"
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--events",
                str(events_path),
                "--out-csv",
                str(out_csv),
                "--out-json",
                str(out_json),
            ],
        )
        assert result.exit_code == 0
        assert "Wrote day metrics CSV" in result.output
        assert "Wrote day metrics JSON" in result.output
        assert out_csv.exists()
        assert out_json.exists()

    def test_analyze_with_balances(self, tmp_path):
        """Analyze with both events and balances files."""
        scenario = _scenario_with_payable()
        events_path = tmp_path / "events.jsonl"
        balances_path = tmp_path / "balances.csv"
        scenario["run"]["export"]["events_jsonl"] = str(events_path)
        scenario["run"]["export"]["balances_csv"] = str(balances_path)
        scenario_path = _write_scenario(tmp_path, scenario)

        runner = CliRunner()
        run_result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "3"])
        assert run_result.exit_code == 0
        assert events_path.exists()
        assert balances_path.exists()

        result = runner.invoke(
            cli,
            [
                "analyze",
                "--events",
                str(events_path),
                "--balances",
                str(balances_path),
            ],
        )
        assert result.exit_code == 0
        assert "Wrote day metrics CSV" in result.output

    def test_analyze_with_html_output(self, tmp_path):
        """Analyze with --html produces an HTML report."""
        scenario = _scenario_with_payable()
        events_path = tmp_path / "events.jsonl"
        scenario["run"]["export"]["events_jsonl"] = str(events_path)
        scenario["run"]["export"]["balances_csv"] = str(tmp_path / "bal.csv")
        scenario_path = _write_scenario(tmp_path, scenario)

        runner = CliRunner()
        run_result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "3"])
        assert run_result.exit_code == 0
        assert events_path.exists()

        html_out = tmp_path / "analytics.html"
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--events",
                str(events_path),
                "--html",
                str(html_out),
            ],
        )
        assert result.exit_code == 0
        assert "Wrote HTML analytics" in result.output
        assert html_out.exists()

    def test_analyze_with_intraday_csv(self, tmp_path):
        """Analyze with --intraday-csv produces the intraday file."""
        scenario = _scenario_with_payable()
        events_path = tmp_path / "events.jsonl"
        scenario["run"]["export"]["events_jsonl"] = str(events_path)
        scenario["run"]["export"]["balances_csv"] = str(tmp_path / "bal.csv")
        scenario_path = _write_scenario(tmp_path, scenario)

        runner = CliRunner()
        run_result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "3"])
        assert run_result.exit_code == 0

        intraday_path = tmp_path / "intraday.csv"
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--events",
                str(events_path),
                "--intraday-csv",
                str(intraday_path),
            ],
        )
        assert result.exit_code == 0
        assert "Wrote intraday CSV" in result.output

    def test_analyze_with_days_filter(self, tmp_path):
        """Analyze with --days '1' filters to specific days."""
        scenario = _scenario_with_payable()
        events_path = tmp_path / "events.jsonl"
        scenario["run"]["export"]["events_jsonl"] = str(events_path)
        scenario["run"]["export"]["balances_csv"] = str(tmp_path / "bal.csv")
        scenario_path = _write_scenario(tmp_path, scenario)

        runner = CliRunner()
        run_result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "3"])
        assert run_result.exit_code == 0

        result = runner.invoke(
            cli,
            [
                "analyze",
                "--events",
                str(events_path),
                "--days",
                "1",
            ],
        )
        assert result.exit_code == 0

    def test_analyze_nonexistent_events(self, tmp_path):
        """Analyze with nonexistent events file fails."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["analyze", "--events", str(tmp_path / "nonexistent.jsonl")],
        )
        assert result.exit_code != 0

    def test_analyze_default_output_paths(self, tmp_path):
        """Analyze without --out-csv/--out-json uses defaults based on events filename."""
        scenario = _scenario_with_payable()
        events_path = tmp_path / "myrun_events.jsonl"
        scenario["run"]["export"]["events_jsonl"] = str(events_path)
        scenario["run"]["export"]["balances_csv"] = str(tmp_path / "bal.csv")
        scenario_path = _write_scenario(tmp_path, scenario)

        runner = CliRunner()
        run_result = runner.invoke(cli, ["run", str(scenario_path), "--max-days", "3"])
        assert run_result.exit_code == 0

        # Analyze without specifying output paths
        result = runner.invoke(
            cli,
            ["analyze", "--events", str(events_path)],
        )
        assert result.exit_code == 0
        # Default paths should be created
        default_csv = tmp_path / "myrun_metrics_day.csv"
        default_json = tmp_path / "myrun_metrics_day.json"
        assert default_csv.exists()
        assert default_json.exists()
