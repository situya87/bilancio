"""Additional coverage tests for display.py.

Targets uncovered branches: show_scenario_header_renderable with list/dict agents,
show_day_summary_renderable with different event_mode settings, trial balance,
error panel, and simulation summary.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bilancio.config.apply import create_agent
from bilancio.core.errors import DefaultError, ValidationError
from bilancio.engines.system import System
from bilancio.ui.display import (
    show_day_summary_renderable,
    show_error_panel,
    show_scenario_header_renderable,
    show_simulation_summary_renderable,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_system_with_events():
    """Create a system with some agents and events."""
    sys = System()
    cb = create_agent(type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"}))
    h1 = create_agent(type("S", (), {"id": "H1", "kind": "household", "name": "Alice"}))
    sys.add_agent(cb)
    sys.add_agent(h1)
    sys.mint_cash("H1", 1000)
    return sys


# ============================================================================
# show_scenario_header_renderable
# ============================================================================


class TestShowScenarioHeaderRenderable:
    """Cover different agent formats in scenario header."""

    def test_no_agents_no_description(self):
        renderables = show_scenario_header_renderable("Test Scenario")
        assert len(renderables) == 1
        assert isinstance(renderables[0], Panel)

    def test_with_description(self):
        renderables = show_scenario_header_renderable("Test", description="A description")
        assert isinstance(renderables[0], Panel)

    def test_agents_as_list_of_objects(self):
        """Agents as list with .id .kind .name attributes."""
        agent_specs = [
            type("Spec", (), {"id": "CB", "kind": "central_bank", "name": "CB"})(),
            type("Spec", (), {"id": "H1", "kind": "household", "name": "Alice"})(),
        ]
        renderables = show_scenario_header_renderable("Test", agents=agent_specs)
        assert len(renderables) == 3  # panel + empty text + table

    def test_agents_as_list_of_dicts(self):
        """Agents as list of dicts."""
        agents = [
            {"id": "CB", "kind": "central_bank", "name": "CB"},
            {"id": "H1", "kind": "household", "name": "Alice"},
        ]
        renderables = show_scenario_header_renderable("Test", agents=agents)
        assert len(renderables) == 3

    def test_agents_as_dict(self):
        """Agents as dict mapping id -> config."""
        agents = {
            "CB": type("Spec", (), {"kind": "central_bank", "name": "CB"})(),
            "H1": type("Spec", (), {"kind": "household", "name": "Alice"})(),
        }
        renderables = show_scenario_header_renderable("Test", agents=agents)
        assert len(renderables) == 3

    def test_agents_as_dict_of_dicts(self):
        agents = {
            "CB": {"kind": "central_bank", "name": "CB"},
            "H1": {"kind": "household", "name": "Alice"},
        }
        renderables = show_scenario_header_renderable("Test", agents=agents)
        assert len(renderables) == 3


# ============================================================================
# show_day_summary_renderable
# ============================================================================


class TestShowDaySummaryRenderable:
    """Cover different event_mode and agent_id branches."""

    def test_event_mode_none(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, event_mode="none")
        assert result == []

    def test_event_mode_summary_with_day(self):
        sys = _make_system_with_events()
        # Events have phase="setup", day=0 by default
        result = show_day_summary_renderable(sys, event_mode="summary", day=0, agent_ids=["H1"])
        assert len(result) > 0

    def test_event_mode_summary_no_day(self):
        """Summary mode without specific day shows all events."""
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, event_mode="summary", agent_ids=["H1"])
        assert len(result) > 0

    def test_event_mode_table_with_day(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, event_mode="table", day=0, agent_ids=["H1"])
        assert len(result) > 0

    def test_event_mode_table_no_day(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, event_mode="table", agent_ids=["H1"])
        assert len(result) > 0

    def test_event_mode_detailed_with_day(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, event_mode="detailed", day=0, agent_ids=["H1"])
        assert len(result) > 0

    def test_event_mode_detailed_no_day(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, event_mode="detailed", agent_ids=["H1"])
        assert len(result) > 0

    def test_empty_agent_ids(self):
        """Empty agent list should show 'No active agents'."""
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, agent_ids=[], event_mode="none")
        # event_mode=none so no events, but agent_ids=[] should add balance info
        # Actually none mode returns [] early. Let's use detailed.
        result = show_day_summary_renderable(sys, agent_ids=[], event_mode="detailed", day=0)
        texts = [str(r) for r in result]
        assert any("No active agents" in t for t in texts)

    def test_single_agent_id(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, agent_ids=["H1"], event_mode="detailed", day=0)
        assert len(result) > 0

    def test_single_agent_t_account(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(
            sys, agent_ids=["H1"], event_mode="detailed", day=0, t_account=True
        )
        assert len(result) > 0

    def test_multiple_agents(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(
            sys, agent_ids=["CB", "H1"], event_mode="detailed", day=0
        )
        assert len(result) > 0

    def test_multiple_agents_t_account(self):
        sys = _make_system_with_events()
        result = show_day_summary_renderable(
            sys, agent_ids=["CB", "H1"], event_mode="detailed", day=0, t_account=True
        )
        assert len(result) > 0

    def test_no_agent_ids_shows_trial_balance(self):
        """agent_ids=None should show system trial balance."""
        sys = _make_system_with_events()
        result = show_day_summary_renderable(sys, agent_ids=None, event_mode="detailed", day=0)
        texts = [str(r) for r in result]
        assert any("Trial Balance" in t for t in texts)


# ============================================================================
# show_simulation_summary_renderable
# ============================================================================


class TestShowSimulationSummaryRenderable:
    """Cover show_simulation_summary_renderable."""

    def test_returns_panel(self):
        sys = _make_system_with_events()
        panel = show_simulation_summary_renderable(sys)
        assert isinstance(panel, Panel)


# ============================================================================
# show_error_panel
# ============================================================================


class TestShowErrorPanel:
    """Cover show_error_panel with different error types."""

    def test_default_error(self, capsys):
        err = DefaultError("Agent F1 cannot pay")
        show_error_panel(err, "day_1", context={"scenario": "test"})
        # Just verify it doesn't crash - output goes to console

    def test_validation_error(self, capsys):
        err = ValidationError("Negative balance detected")
        show_error_panel(err, "setup")

    def test_generic_error(self, capsys):
        err = RuntimeError("Something broke")
        show_error_panel(err, "simulation", context={"key": "value"})

    def test_no_context(self, capsys):
        err = RuntimeError("Oops")
        show_error_panel(err, "day_0")
