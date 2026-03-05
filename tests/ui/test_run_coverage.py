"""Tests for bilancio.ui.run module to increase coverage.

Targets uncovered lines in run_scenario, run_until_stable_mode, run_step_mode,
_filter_active_agent_ids, and related helpers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bilancio.core.errors import DefaultError, SimulationHalt
from bilancio.ui.run import (
    _filter_active_agent_ids,
    run_scenario,
    run_step_mode,
    run_until_stable_mode,
)


# ---------------------------------------------------------------------------
# Helpers: minimal YAML scenario builders
# ---------------------------------------------------------------------------

def _simple_bank_yaml() -> dict:
    """Minimal scenario: CB + bank + 2 households with a payment due day 1."""
    return {
        "version": 1,
        "name": "Test Scenario",
        "description": "Coverage test",
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
            "max_days": 5,
            "quiet_days": 1,
            "show": {"balances": ["CB", "B1", "H1", "H2"]},
        },
    }


def _write_scenario(tmp_path: Path, scenario: dict, name: str = "scenario.yaml") -> Path:
    """Write a scenario dict to a YAML file and return the path."""
    p = tmp_path / name
    with open(p, "w") as f:
        yaml.dump(scenario, f)
    return p


# ---------------------------------------------------------------------------
# 1. _filter_active_agent_ids
# ---------------------------------------------------------------------------

class TestFilterActiveAgentIds:

    def test_none_returns_none(self):
        """Passing None returns None."""
        system = MagicMock()
        assert _filter_active_agent_ids(system, None) is None

    def test_missing_agent_skipped(self):
        """Agent IDs not in system are skipped."""
        system = MagicMock()
        system.state.agents.get.return_value = None
        result = _filter_active_agent_ids(system, ["X1"])
        assert result == []

    def test_defaulted_agent_skipped(self):
        """Defaulted agents are filtered out."""
        agent = MagicMock()
        agent.defaulted = True
        system = MagicMock()
        system.state.agents.get.return_value = agent
        result = _filter_active_agent_ids(system, ["H1"])
        assert result == []

    def test_active_agent_included(self):
        """Non-defaulted agents are included."""
        agent = MagicMock()
        agent.defaulted = False
        system = MagicMock()
        system.state.agents.get.return_value = agent
        result = _filter_active_agent_ids(system, ["H1", "H2"])
        assert result == ["H1", "H2"]

    def test_mixed_agents(self):
        """Mix of active, defaulted, and missing."""
        active = MagicMock(); active.defaulted = False
        defaulted = MagicMock(); defaulted.defaulted = True

        def get_agent(aid):
            return {"H1": active, "H2": defaulted}.get(aid)

        system = MagicMock()
        system.state.agents.get.side_effect = get_agent
        result = _filter_active_agent_ids(system, ["H1", "H2", "MISSING"])
        assert result == ["H1"]


# ---------------------------------------------------------------------------
# 2. run_scenario - until_stable mode (default)
# ---------------------------------------------------------------------------

class TestRunScenarioUntilStable:

    def test_basic_until_stable(self, tmp_path):
        """Run a minimal scenario in until_stable mode to completion."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        # Should not raise
        run_scenario(path, mode="until_stable", max_days=5, quiet_days=1)

    def test_quiet_mode_show_none(self, tmp_path):
        """show='none' suppresses verbose output without error."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        run_scenario(path, mode="until_stable", max_days=5, quiet_days=1, show="none")

    def test_show_summary(self, tmp_path):
        """show='summary' mode runs without error."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        run_scenario(path, mode="until_stable", max_days=5, quiet_days=1, show="summary")

    def test_check_invariants_daily(self, tmp_path):
        """check_invariants='daily' exercises the daily invariant check path."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            check_invariants="daily",
        )

    def test_check_invariants_none(self, tmp_path):
        """check_invariants='none' skips setup invariant check."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            check_invariants="none",
        )

    def test_progress_callback(self, tmp_path):
        """progress_callback is invoked during the run."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        calls = []

        def cb(current: int, total: int) -> None:
            calls.append((current, total))

        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none", progress_callback=cb,
        )
        assert len(calls) > 0
        # Each call should have (day, max_days)
        for day, mx in calls:
            assert 1 <= day <= 5
            assert mx == 5

    def test_default_handling_override(self, tmp_path):
        """CLI default_handling override replaces config value."""
        scenario = _simple_bank_yaml()
        # scenario default is fail-fast; override to expel-agent
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            default_handling="expel-agent", show="none",
        )


# ---------------------------------------------------------------------------
# 3. run_scenario - export paths
# ---------------------------------------------------------------------------

class TestRunScenarioExport:

    def test_export_balances_csv(self, tmp_path):
        """Exporting balances CSV creates the file."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        csv_path = str(tmp_path / "out" / "balances.csv")
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            export={"balances_csv": csv_path}, show="none",
        )
        assert Path(csv_path).exists()
        assert Path(csv_path).stat().st_size > 0

    def test_export_events_jsonl(self, tmp_path):
        """Exporting events JSONL creates the file."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        events_path = str(tmp_path / "out" / "events.jsonl")
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            export={"events_jsonl": events_path}, show="none",
        )
        assert Path(events_path).exists()
        assert Path(events_path).stat().st_size > 0

    def test_export_both(self, tmp_path):
        """Exporting both balances and events works."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        csv_path = str(tmp_path / "out" / "balances.csv")
        events_path = str(tmp_path / "out" / "events.jsonl")
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            export={"balances_csv": csv_path, "events_jsonl": events_path},
            show="none",
        )
        assert Path(csv_path).exists()
        assert Path(events_path).exists()

    def test_html_output(self, tmp_path):
        """HTML export produces a file."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        html_path = tmp_path / "report.html"
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            html_output=html_path, show="none",
        )
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "html" in content.lower()

    def test_export_from_config(self, tmp_path):
        """Export paths taken from config when not passed explicitly."""
        scenario = _simple_bank_yaml()
        csv_path = str(tmp_path / "cfg_balances.csv")
        events_path = str(tmp_path / "cfg_events.jsonl")
        scenario["run"]["export"] = {
            "balances_csv": csv_path,
            "events_jsonl": events_path,
        }
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            show="none",
        )
        assert Path(csv_path).exists()
        assert Path(events_path).exists()


# ---------------------------------------------------------------------------
# 4. run_scenario - agent_ids parameter
# ---------------------------------------------------------------------------

class TestRunScenarioAgentIds:

    def test_explicit_agent_ids(self, tmp_path):
        """Passing explicit agent_ids filters display."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            agent_ids=["H1", "H2"],
        )

    def test_agent_ids_from_config(self, tmp_path):
        """Agent IDs taken from config when None passed."""
        scenario = _simple_bank_yaml()
        scenario["run"]["show"]["balances"] = ["CB", "H1"]
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
        )


# ---------------------------------------------------------------------------
# 5. run_scenario - rollover mode
# ---------------------------------------------------------------------------

class TestRunScenarioRollover:

    def test_rollover_enabled(self, tmp_path):
        """Rollover mode exercises the consecutive_no_defaults stability path."""
        scenario = _simple_bank_yaml()
        scenario["run"]["rollover_enabled"] = True
        # Need enough days for the stability check to trigger via the
        # rollover path (consecutive_no_defaults >= quiet_days)
        scenario["run"]["max_days"] = 10
        scenario["run"]["quiet_days"] = 2
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=10, quiet_days=2,
            show="none",
        )


# ---------------------------------------------------------------------------
# 6. run_scenario - max days exhaustion
# ---------------------------------------------------------------------------

class TestRunScenarioMaxDays:

    def test_max_days_reached(self, tmp_path):
        """When max_days=1, the loop ends without stable state."""
        scenario = _simple_bank_yaml()
        # Create multiple obligations spanning days to prevent stability
        scenario["initial_actions"].append(
            {"create_payable": {"from": "H2", "to": "H1", "amount": 100, "due_day": 2}}
        )
        path = _write_scenario(tmp_path, scenario)
        # max_days=1 means we only run 1 day iteration - cannot reach stability
        run_scenario(
            path, mode="until_stable", max_days=1, quiet_days=2,
            show="none",
        )


# ---------------------------------------------------------------------------
# 7. run_step_mode (via mode="step")
# ---------------------------------------------------------------------------

class TestRunStepMode:

    def test_step_mode_auto_confirm(self, tmp_path):
        """Step mode with mocked Confirm runs the loop."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)

        # Mock Confirm.ask to return True for first call, then False to stop
        call_count = 0

        def mock_confirm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Run 2 days then stop
            return call_count <= 2

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = mock_confirm
            run_scenario(
                path, mode="step", max_days=5, quiet_days=1,
            )
        assert call_count >= 1

    def test_step_mode_immediate_stop(self, tmp_path):
        """Step mode stops immediately when user declines."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = lambda *a, **kw: False
            run_scenario(
                path, mode="step", max_days=5, quiet_days=1,
            )

    def test_step_mode_with_agent_ids(self, tmp_path):
        """Step mode with explicit agent_ids captures balances."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)

        call_count = 0

        def mock_confirm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count <= 3

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = mock_confirm
            run_scenario(
                path, mode="step", max_days=5, quiet_days=1,
                agent_ids=["H1", "H2"],
            )

    def test_step_mode_check_invariants_daily(self, tmp_path):
        """Step mode with daily invariant checking."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)

        call_count = 0

        def mock_confirm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count <= 2

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = mock_confirm
            run_scenario(
                path, mode="step", max_days=5, quiet_days=1,
                check_invariants="daily",
            )


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------

class TestRunScenarioErrors:

    def test_nonexistent_path(self, tmp_path):
        """Non-existent YAML raises an error."""
        bad_path = tmp_path / "nonexistent.yaml"
        with pytest.raises((FileNotFoundError, SystemExit)):
            run_scenario(bad_path, max_days=1)

    def test_invalid_yaml_content(self, tmp_path):
        """Invalid YAML content raises an error."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("not: a: valid: scenario: {{{}}")
        with pytest.raises((SystemExit, yaml.YAMLError)):
            run_scenario(bad_file, max_days=1)

    def test_empty_agents_returns_none(self, tmp_path):
        """Scenario with no agents runs to max_days and returns None."""
        scenario = {
            "version": 1,
            "name": "Empty",
            "agents": [],
            "initial_actions": [],
            "run": {"max_days": 1},
        }
        path = _write_scenario(tmp_path, scenario)
        # Empty agent list hits max_days without stable state → returns None
        result = run_scenario(path, max_days=1, show="none")
        assert result is None


# ---------------------------------------------------------------------------
# 9. Dealer subsystem export paths (active & passive)
# ---------------------------------------------------------------------------

class TestDealerMetricsExport:

    def test_dealer_metrics_export_path_from_events(self, tmp_path):
        """When dealer subsystem exists and events_jsonl is set, dealer_metrics.json is created."""
        # Use the existing kalecki_with_dealer scenario which has dealer config
        scenario_path = (
            Path(__file__).parent.parent.parent / "examples" / "scenarios" / "kalecki_with_dealer.yaml"
        )
        if not scenario_path.exists():
            pytest.skip("kalecki_with_dealer.yaml not found")

        events_path = str(tmp_path / "out" / "events.jsonl")
        run_scenario(
            scenario_path, mode="until_stable", max_days=5, quiet_days=1,
            export={"events_jsonl": events_path}, show="none",
        )
        # If dealer subsystem was active, dealer_metrics.json should exist
        dealer_metrics = tmp_path / "out" / "dealer_metrics.json"
        if dealer_metrics.exists():
            import json
            data = json.loads(dealer_metrics.read_text())
            assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# 10. Config export fallback paths
# ---------------------------------------------------------------------------

class TestConfigExportFallback:

    def test_export_from_config_balances_only(self, tmp_path):
        """Config provides balances_csv but not events_jsonl."""
        scenario = _simple_bank_yaml()
        csv_path = str(tmp_path / "from_config.csv")
        scenario["run"]["export"] = {"balances_csv": csv_path}
        path = _write_scenario(tmp_path, scenario)
        run_scenario(path, mode="until_stable", max_days=3, quiet_days=1, show="none")
        assert Path(csv_path).exists()

    def test_explicit_export_overrides_config(self, tmp_path):
        """Explicit export dict overrides config export paths."""
        scenario = _simple_bank_yaml()
        scenario["run"]["export"] = {
            "balances_csv": str(tmp_path / "config_balances.csv"),
            "events_jsonl": str(tmp_path / "config_events.jsonl"),
        }
        path = _write_scenario(tmp_path, scenario)
        explicit_csv = str(tmp_path / "explicit_balances.csv")
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            export={"balances_csv": explicit_csv}, show="none",
        )
        # Explicit path used
        assert Path(explicit_csv).exists()
        # Config path for events also used as fallback
        assert Path(str(tmp_path / "config_events.jsonl")).exists()


# ---------------------------------------------------------------------------
# 11. Scheduled actions staging
# ---------------------------------------------------------------------------

class TestScheduledActions:

    def test_scenario_without_scheduled_actions(self, tmp_path):
        """Scenario without scheduled_actions does not error."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        # Should run fine - the try/except around scheduled_actions handles it
        run_scenario(path, mode="until_stable", max_days=3, quiet_days=1, show="none")


# ---------------------------------------------------------------------------
# 12. HTML output with agent_ids
# ---------------------------------------------------------------------------

class TestHtmlWithAgentIds:

    def test_html_with_explicit_agent_ids(self, tmp_path):
        """HTML export with explicit agent_ids captures per-agent balances."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        html_path = tmp_path / "with_agents.html"
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            html_output=html_path, agent_ids=["H1", "H2"],
        )
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_html_without_agent_ids(self, tmp_path):
        """HTML export without agent_ids captures all agents."""
        scenario = _simple_bank_yaml()
        # Remove show.balances so agent_ids stays None
        scenario["run"]["show"] = {"events": "detailed"}
        path = _write_scenario(tmp_path, scenario)
        html_path = tmp_path / "no_agents.html"
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            html_output=html_path,
        )
        assert html_path.exists()


# ---------------------------------------------------------------------------
# 13. t_account option
# ---------------------------------------------------------------------------

class TestTAccountOption:

    def test_t_account_display(self, tmp_path):
        """t_account=True exercises the T-account display path."""
        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=3, quiet_days=1,
            t_account=True,
        )


# ---------------------------------------------------------------------------
# 14. Default handling with expel-agent from config
# ---------------------------------------------------------------------------

class TestDefaultHandlingFromConfig:

    def test_expel_agent_from_config(self, tmp_path):
        """Config with default_handling=expel-agent runs without error."""
        scenario = _simple_bank_yaml()
        scenario["run"]["default_handling"] = "expel-agent"
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )

    def test_cli_override_matches_config(self, tmp_path):
        """When CLI override matches config, no model_copy is needed."""
        scenario = _simple_bank_yaml()
        scenario["run"]["default_handling"] = "expel-agent"
        path = _write_scenario(tmp_path, scenario)
        # CLI override same as config - the if-branch at line 100 is False
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            default_handling="expel-agent", show="none",
        )


# ---------------------------------------------------------------------------
# 15. Banking subsystem init from _balanced_config in YAML (lines 242-336)
# ---------------------------------------------------------------------------

class TestBankingSubsystemInit:

    def test_banking_init_from_balanced_config(self, tmp_path):
        """Scenario with _balanced_config.n_banks > 0 triggers banking init."""
        scenario = _simple_bank_yaml()
        # Add _balanced_config to trigger the banking init path
        scenario["_balanced_config"] = {
            "n_banks": 1,
            "kappa": "1.0",
            "maturity_days": 5,
            "credit_risk_loading": 0,
            "max_borrower_risk": "1.0",
            "min_coverage_ratio": 0,
            "Q_total": 300,
            "enable_bank_lending": False,
        }
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )

    def test_banking_init_with_credit_risk_loading(self, tmp_path):
        """Banking init with credit_risk_loading > 0 wires risk assessor."""
        scenario = _simple_bank_yaml()
        scenario["_balanced_config"] = {
            "n_banks": 1,
            "kappa": "0.5",
            "maturity_days": 5,
            "credit_risk_loading": "0.1",
            "max_borrower_risk": "0.8",
            "min_coverage_ratio": 0,
            "Q_total": 300,
        }
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )

    def test_banking_init_with_enable_banking_flag(self, tmp_path):
        """enable_banking flag in run config triggers banking init."""
        scenario = _simple_bank_yaml()
        scenario["_balanced_config"] = {
            "n_banks": 1,
            "kappa": "1.0",
            "maturity_days": 5,
            "Q_total": 300,
        }
        scenario["run"]["enable_banking"] = True
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )

    def test_banking_init_kappa_from_balanced_dealer(self, tmp_path):
        """Kappa fallback from balanced_dealer config."""
        scenario = _simple_bank_yaml()
        scenario["_balanced_config"] = {
            "n_banks": 1,
            "maturity_days": 5,
            "Q_total": 300,
        }
        scenario["balanced_dealer"] = {"kappa": "2.0"}
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )

    def test_banking_init_maturity_from_params(self, tmp_path):
        """Maturity days fallback from params.maturity.days."""
        scenario = _simple_bank_yaml()
        scenario["_balanced_config"] = {
            "n_banks": 1,
            "kappa": "1.0",
            "Q_total": 300,
        }
        scenario["params"] = {"maturity": {"days": 7}}
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )

    def test_banking_init_cb_escalation_params(self, tmp_path):
        """CB escalation params are wired when explicitly set."""
        scenario = _simple_bank_yaml()
        scenario["_balanced_config"] = {
            "n_banks": 1,
            "kappa": "1.0",
            "maturity_days": 5,
            "Q_total": 300,
            "cb_rate_escalation_slope": "0.1",
            "cb_max_outstanding_ratio": "3.0",
        }
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )

    def test_banking_init_with_bank_lending(self, tmp_path):
        """enable_bank_lending=True activates the bank lending flag."""
        scenario = _simple_bank_yaml()
        scenario["_balanced_config"] = {
            "n_banks": 1,
            "kappa": "1.0",
            "maturity_days": 5,
            "Q_total": 300,
            "enable_bank_lending": True,
        }
        path = _write_scenario(tmp_path, scenario)
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            show="none",
        )


# ---------------------------------------------------------------------------
# 16. Dealer metrics export via balances_csv fallback (lines 390-410)
# ---------------------------------------------------------------------------

class TestDealerMetricsExportBalancesFallback:

    def test_dealer_metrics_path_from_balances_csv(self, tmp_path):
        """Dealer metrics path falls back to balances_csv parent when no events_jsonl."""
        scenario_path = (
            Path(__file__).parent.parent.parent / "examples" / "scenarios" / "kalecki_with_dealer.yaml"
        )
        if not scenario_path.exists():
            pytest.skip("kalecki_with_dealer.yaml not found")

        csv_path = str(tmp_path / "out" / "balances.csv")
        run_scenario(
            scenario_path, mode="until_stable", max_days=5, quiet_days=1,
            export={"balances_csv": csv_path}, show="none",
        )
        dealer_metrics = tmp_path / "out" / "dealer_metrics.json"
        if dealer_metrics.exists():
            import json
            data = json.loads(dealer_metrics.read_text())
            assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# 17. Error handlers in run_until_stable_mode (lines 977-1004)
# ---------------------------------------------------------------------------

class TestUntilStableErrorHandlers:

    def _make_system(self, tmp_path):
        """Create a configured System from a minimal scenario."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)
        return system

    def test_default_error_caught(self, tmp_path):
        """DefaultError during simulation is caught and displayed."""
        system = self._make_system(tmp_path)

        with patch("bilancio.ui.run.run_day", side_effect=DefaultError("test default")):
            result = run_until_stable_mode(
                system=system,
                max_days=5,
                quiet_days=1,
                show="none",
                agent_ids=None,
                check_invariants="none",
                scenario_name="Test",
            )
        assert isinstance(result, list)

    def test_simulation_halt_caught(self, tmp_path):
        """SimulationHalt during simulation is caught and displayed."""
        system = self._make_system(tmp_path)

        with patch("bilancio.ui.run.run_day", side_effect=SimulationHalt("collapse")):
            result = run_until_stable_mode(
                system=system,
                max_days=5,
                quiet_days=1,
                show="none",
                agent_ids=None,
                check_invariants="none",
                scenario_name="Test",
            )
        assert isinstance(result, list)

    def test_runtime_error_caught(self, tmp_path):
        """RuntimeError (recoverable) during simulation is caught."""
        system = self._make_system(tmp_path)

        with patch("bilancio.ui.run.run_day", side_effect=RuntimeError("oops")):
            result = run_until_stable_mode(
                system=system,
                max_days=5,
                quiet_days=1,
                show="none",
                agent_ids=None,
                check_invariants="none",
                scenario_name="Test",
            )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 18. Error handlers in run_step_mode (lines 663-693)
# ---------------------------------------------------------------------------

class TestStepModeErrorHandlers:

    def _make_system(self, tmp_path):
        """Create a configured System from a minimal scenario."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)
        return system

    def test_default_error_in_step_mode(self, tmp_path):
        """DefaultError in step mode is caught."""
        system = self._make_system(tmp_path)

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = lambda *a, **kw: True
            with patch("bilancio.ui.run.run_day", side_effect=DefaultError("test")):
                result = run_step_mode(
                    system=system,
                    max_days=5,
                    show="detailed",
                    agent_ids=None,
                    check_invariants="none",
                    scenario_name="Test",
                )
        assert isinstance(result, list)

    def test_simulation_halt_in_step_mode(self, tmp_path):
        """SimulationHalt in step mode is caught."""
        system = self._make_system(tmp_path)

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = lambda *a, **kw: True
            with patch("bilancio.ui.run.run_day", side_effect=SimulationHalt("halt")):
                result = run_step_mode(
                    system=system,
                    max_days=5,
                    show="detailed",
                    agent_ids=None,
                    check_invariants="none",
                    scenario_name="Test",
                )
        assert isinstance(result, list)

    def test_recoverable_error_in_step_mode(self, tmp_path):
        """Recoverable error (RuntimeError) in step mode is caught."""
        system = self._make_system(tmp_path)

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = lambda *a, **kw: True
            with patch("bilancio.ui.run.run_day", side_effect=RuntimeError("runtime fail")):
                result = run_step_mode(
                    system=system,
                    max_days=5,
                    show="detailed",
                    agent_ids=None,
                    check_invariants="none",
                    scenario_name="Test",
                )
        assert isinstance(result, list)

    def test_validation_error_in_step_mode(self, tmp_path):
        """ValidationError in step mode is caught."""
        from bilancio.core.errors import ValidationError

        system = self._make_system(tmp_path)

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = lambda *a, **kw: True
            with patch("bilancio.ui.run.run_day", side_effect=ValidationError("bad state")):
                result = run_step_mode(
                    system=system,
                    max_days=5,
                    show="detailed",
                    agent_ids=None,
                    check_invariants="none",
                    scenario_name="Test",
                )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 19. CB lending cutoff (lines 772-773)
# ---------------------------------------------------------------------------

class TestCBLendingCutoff:

    def test_cb_lending_cutoff_in_until_stable(self, tmp_path):
        """CB lending cutoff activates when day >= cutoff_day."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)

        result = run_until_stable_mode(
            system=system,
            max_days=5,
            quiet_days=1,
            show="none",
            agent_ids=None,
            check_invariants="none",
            scenario_name="Test",
            cb_lending_cutoff_day=1,
        )
        # After running, cb_lending_frozen should be True (set at cutoff day)
        assert system.state.cb_lending_frozen is True


# ---------------------------------------------------------------------------
# 20. Daily invariant failure handling (lines 827-829)
# ---------------------------------------------------------------------------

class TestDailyInvariantFailure:

    def test_invariant_failure_logged(self, tmp_path):
        """Daily invariant check failure is caught and logged."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)

        # Run one day to get past day 0
        from bilancio.engines.simulation import run_day
        run_day(system)

        # Mock assert_invariants to raise on daily check
        original_assert = system.assert_invariants

        def failing_invariants():
            raise AssertionError("Invariant broken")

        system.assert_invariants = failing_invariants

        # Continue the simulation - the invariant failure should be caught
        result = run_until_stable_mode(
            system=system,
            max_days=5,
            quiet_days=1,
            show="detailed",  # not quiet so the console print branch executes
            agent_ids=None,
            check_invariants="daily",
            scenario_name="Test",
        )
        assert isinstance(result, list)

        # Restore
        system.assert_invariants = original_assert


# ---------------------------------------------------------------------------
# 21. Step mode - reaches stable state (lines 655-661)
# ---------------------------------------------------------------------------

class TestStepModeStableState:

    def test_step_mode_reaches_stability(self, tmp_path):
        """Step mode detects stable state and breaks."""
        scenario = _simple_bank_yaml()
        # Only one payment due day 1, should stabilize quickly
        path = _write_scenario(tmp_path, scenario)

        call_count = 0

        def mock_confirm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return True  # Always continue - stability check breaks the loop

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = mock_confirm
            run_scenario(
                path, mode="step", max_days=10, quiet_days=1,
            )
        # Should have broken before max_days due to stability
        assert call_count < 10


# ---------------------------------------------------------------------------
# 22. Detailed dealer logging (lines 412-470)
# ---------------------------------------------------------------------------

class TestDetailedDealerLogging:

    def test_detailed_dealer_logging_with_dealer(self, tmp_path):
        """Detailed dealer logging exports CSV files when dealer is active."""
        scenario_path = (
            Path(__file__).parent.parent.parent / "examples" / "scenarios" / "kalecki_with_dealer.yaml"
        )
        if not scenario_path.exists():
            pytest.skip("kalecki_with_dealer.yaml not found")

        events_path = str(tmp_path / "out" / "events.jsonl")
        run_scenario(
            scenario_path, mode="until_stable", max_days=5, quiet_days=1,
            export={"events_jsonl": events_path}, show="none",
            detailed_dealer_logging=True,
            run_id="test-run",
            regime="active",
        )
        out_dir = tmp_path / "out"
        # Check if dealer CSV files were created (only if dealer was active)
        dealer_metrics = out_dir / "dealer_metrics.json"
        if dealer_metrics.exists():
            # These should exist when detailed logging is on with active dealer
            for name in ["trades.csv", "inventory_timeseries.csv",
                        "system_state_timeseries.csv", "repayment_events.csv"]:
                expected = out_dir / name
                # Files exist only if there was actual trading activity
                if expected.exists():
                    assert expected.stat().st_size > 0


# ---------------------------------------------------------------------------
# 23. Validation error during setup (lines 127-129)
# ---------------------------------------------------------------------------

class TestSetupValidationError:

    def test_invalid_action_causes_setup_error(self, tmp_path):
        """Invalid initial action triggers ValidationError during setup."""
        scenario = {
            "version": 1,
            "name": "Bad Setup",
            "description": "Invalid initial action",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "H1", "kind": "household", "name": "Alice"},
            ],
            "initial_actions": [
                # Try to transfer more cash than exists - should cause error
                {"mint_cash": {"to": "H1", "amount": 100}},
                # Create payable from non-existent agent
                {"create_payable": {"from": "NONEXISTENT", "to": "H1", "amount": 50, "due_day": 1}},
            ],
            "run": {"max_days": 1},
        }
        path = _write_scenario(tmp_path, scenario)
        with pytest.raises(SystemExit):
            run_scenario(path, max_days=1, show="none")


# ---------------------------------------------------------------------------
# 24. Step mode CB lending cutoff (line 540-546)
# ---------------------------------------------------------------------------

class TestStepModeCBLendingCutoff:

    def test_cb_lending_cutoff_in_step_mode(self, tmp_path):
        """CB lending cutoff activates in step mode."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)

        call_count = 0

        def mock_confirm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count <= 3

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = mock_confirm
            result = run_step_mode(
                system=system,
                max_days=5,
                show="none",
                agent_ids=None,
                check_invariants="none",
                scenario_name="Test",
                cb_lending_cutoff_day=1,
            )
        assert system.state.cb_lending_frozen is True


# ---------------------------------------------------------------------------
# 25. ValidationError handler in until_stable mode (lines 988-997)
# ---------------------------------------------------------------------------

class TestUntilStableValidationError:

    def test_validation_error_caught(self, tmp_path):
        """ValidationError during until_stable simulation is caught."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.core.errors import ValidationError
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)

        with patch("bilancio.ui.run.run_day", side_effect=ValidationError("invalid state")):
            result = run_until_stable_mode(
                system=system,
                max_days=5,
                quiet_days=1,
                show="none",
                agent_ids=None,
                check_invariants="none",
                scenario_name="Test",
            )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 26. Defaults counter reset (line 930) - need a default event
# ---------------------------------------------------------------------------

class TestDefaultsCounterReset:

    def test_defaults_reset_consecutive_counter(self, tmp_path):
        """When defaults occur, consecutive_no_defaults resets to 0."""
        scenario = _simple_bank_yaml()
        # Create a scenario where a default will occur: H1 owes more than it has
        scenario["initial_actions"] = [
            {"mint_reserves": {"to": "B1", "amount": 5000}},
            {"mint_cash": {"to": "H1", "amount": 10}},  # very little cash
            {"mint_cash": {"to": "H2", "amount": 1000}},
            # H1 owes more than it has - should cause default with expel-agent
            {"create_payable": {"from": "H1", "to": "H2", "amount": 5000, "due_day": 1}},
        ]
        scenario["run"]["default_handling"] = "expel-agent"
        path = _write_scenario(tmp_path, scenario)
        # Run with expel-agent to allow continuation after default
        run_scenario(
            path, mode="until_stable", max_days=5, quiet_days=1,
            default_handling="expel-agent", show="none",
        )


# ---------------------------------------------------------------------------
# 27. Dealer metrics export with mock (lines 390-410)
# ---------------------------------------------------------------------------

class TestDealerMetricsExportMocked:

    def _make_system_with_dealer(self, tmp_path, *, enabled=True):
        """Create a configured System and inject a mock dealer subsystem."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)

        mock_subsystem = MagicMock()
        mock_subsystem.enabled = enabled
        mock_subsystem.metrics.summary.return_value = {"trades": 0, "pnl": 0}
        mock_subsystem.metrics.trades = []
        mock_subsystem.metrics.dealer_snapshots = []
        mock_subsystem.metrics.system_state_snapshots = []
        mock_subsystem.metrics.repayment_events = []
        system.state.dealer_subsystem = mock_subsystem
        return system, mock_subsystem, path

    def test_dealer_metrics_enabled_subsystem(self, tmp_path):
        """Dealer metrics export with enabled (active) subsystem via run_scenario + patch."""
        system, mock_sub, path = self._make_system_with_dealer(tmp_path, enabled=True)

        events_path = str(tmp_path / "out" / "events.jsonl")

        # Patch apply_to_system to inject our pre-configured system
        original_apply = None

        def patched_apply(config, sys_obj):
            """Copy our mock dealer subsystem onto the new system."""
            from bilancio.config import apply_to_system as real_apply
            real_apply(config, sys_obj)
            sys_obj.state.dealer_subsystem = mock_sub

        with patch("bilancio.ui.run.apply_to_system", side_effect=patched_apply):
            run_scenario(
                path, mode="until_stable", max_days=3, quiet_days=1,
                export={"events_jsonl": events_path}, show="none",
            )
        import json
        dealer_metrics_path = tmp_path / "out" / "dealer_metrics.json"
        assert dealer_metrics_path.exists()
        data = json.loads(dealer_metrics_path.read_text())
        assert data == {"trades": 0, "pnl": 0}

    def test_dealer_metrics_disabled_subsystem(self, tmp_path):
        """Dealer metrics export with disabled (passive) subsystem."""
        _, mock_sub, path = self._make_system_with_dealer(tmp_path, enabled=False)

        csv_path = str(tmp_path / "out" / "balances.csv")

        def patched_apply(config, sys_obj):
            from bilancio.config import apply_to_system as real_apply
            real_apply(config, sys_obj)
            sys_obj.state.dealer_subsystem = mock_sub

        with patch("bilancio.ui.run.apply_to_system", side_effect=patched_apply):
            with patch("bilancio.engines.dealer_integration.compute_passive_pnl", return_value={"passive_pnl": -5}):
                run_scenario(
                    path, mode="until_stable", max_days=3, quiet_days=1,
                    export={"balances_csv": csv_path}, show="none",
                )
        import json
        dealer_metrics_path = tmp_path / "out" / "dealer_metrics.json"
        assert dealer_metrics_path.exists()
        data = json.loads(dealer_metrics_path.read_text())
        assert data == {"passive_pnl": -5}

    def test_detailed_dealer_logging_mocked(self, tmp_path):
        """Detailed dealer logging with mock subsystem exports CSV files."""
        _, mock_sub, path = self._make_system_with_dealer(tmp_path, enabled=True)
        mock_metrics = mock_sub.metrics

        events_path = str(tmp_path / "out" / "events.jsonl")

        def patched_apply(config, sys_obj):
            from bilancio.config import apply_to_system as real_apply
            real_apply(config, sys_obj)
            sys_obj.state.dealer_subsystem = mock_sub

        with patch("bilancio.ui.run.apply_to_system", side_effect=patched_apply):
            with patch("bilancio.dealer.metrics.build_repayment_events", return_value=[]):
                run_scenario(
                    path, mode="until_stable", max_days=3, quiet_days=1,
                    export={"events_jsonl": events_path}, show="none",
                    detailed_dealer_logging=True,
                    run_id="test-run-123",
                    regime="active",
                )

        # Verify the CSV export methods were called
        mock_metrics.to_trade_log_csv.assert_called_once()
        mock_metrics.to_inventory_timeseries_csv.assert_called_once()
        mock_metrics.to_system_state_csv.assert_called_once()
        mock_metrics.to_repayment_events_csv.assert_called_once()

        # Verify run context was set
        assert mock_metrics.run_id == "test-run-123"
        assert mock_metrics.regime == "active"


# ---------------------------------------------------------------------------
# 28. Step mode with banking stability freeze (lines 657-659)
# ---------------------------------------------------------------------------

class TestStepModeBankingStabilityFreeze:

    def test_step_mode_freezes_cb_lending_on_stability(self, tmp_path):
        """Step mode freezes CB lending when stable state is reached with banking."""
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        scenario = _simple_bank_yaml()
        path = _write_scenario(tmp_path, scenario)
        config = load_yaml(path)
        system = System(default_mode="fail-fast")
        apply_to_system(config, system)

        call_count = 0

        def mock_confirm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return True  # Keep going until stability breaks the loop

        with patch("bilancio.ui.run.Confirm") as MockConfirm:
            MockConfirm.ask = mock_confirm
            result = run_step_mode(
                system=system,
                max_days=10,
                show="none",
                agent_ids=["H1", "H2"],
                check_invariants="none",
                scenario_name="Test",
                enable_banking=True,  # Banking enabled
            )
        # CB lending should have been frozen when stability was reached
        assert system.state.cb_lending_frozen is True
