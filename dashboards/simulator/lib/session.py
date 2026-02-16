"""Session state management for the Streamlit balance sheet simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from bilancio.analysis.balances import AgentBalance
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.system import System


@dataclass
class SimState:
    """All simulator state, stored in Streamlit session."""

    system: System
    action_log: list[dict[str, Any]]
    snapshots: list[dict[str, AgentBalance]]  # per-day balance snapshots
    sim_config: dict[str, Any]
    events_by_day: dict[int, list[dict[str, Any]]]
    defaults_by_day: dict[int, int] = field(default_factory=dict)  # cumulative defaults per day

    @classmethod
    def fresh(cls) -> SimState:
        """Create a fresh SimState with default system and empty logs."""
        system = System(policy=PolicyEngine.default())
        return cls(
            system=system,
            action_log=[],
            snapshots=[],
            sim_config={
                "max_days": 30,
                "rollover": False,
                "default_handling": "expel-agent",
            },
            events_by_day={},
        )


def init_session() -> None:
    """Initialize Streamlit session state if not already present."""
    if "sim" not in st.session_state:
        st.session_state["sim"] = SimState.fresh()


def get_state() -> SimState:
    """Return the current SimState from session."""
    init_session()
    return st.session_state["sim"]


def get_system() -> System:
    """Return the current System instance."""
    return get_state().system


def reset_session() -> None:
    """Reset all session state to fresh defaults."""
    st.session_state["sim"] = SimState.fresh()


def rebuild_system(action_log: list[dict[str, Any]]) -> System:
    """Replay a list of actions to rebuild the System from scratch.

    Used for undo: remove the last action from the log, then rebuild.
    Import bridge functions lazily to avoid circular imports.
    """
    from dashboards.simulator.lib.bridge import replay_action

    system = System(policy=PolicyEngine.default())
    for action in action_log:
        replay_action(system, action)
    return system


def undo_last_action() -> bool:
    """Remove the last action and rebuild the system.

    Returns True if an action was undone, False if log was empty.
    """
    state = get_state()
    if not state.action_log:
        return False
    state.action_log.pop()
    state.system = rebuild_system(state.action_log)
    return True
