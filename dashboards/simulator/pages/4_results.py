"""Page 4: Run & Review.

Runs the simulation day-by-day, captures snapshots, and lets
users scrub through time to see balance sheet diffs.
"""

from __future__ import annotations

import streamlit as st

from bilancio.engines.simulation import run_day

from dashboards.simulator.lib.display import (
    render_balance_table,
    render_system_summary,
)
from dashboards.simulator.lib.policy_ui import AGENT_KIND_LABELS
from dashboards.simulator.lib.session import get_state, rebuild_system
from dashboards.simulator.lib.snapshot import capture_snapshot, compute_diff


def _build_export_yaml(state) -> dict:
    """Build a scenario YAML dict compatible with `bilancio run`."""
    agents_list = []
    initial_actions = []

    for action in state.action_log:
        t = action["type"]
        if t == "add_agent":
            agents_list.append({
                "id": action["agent_id"],
                "kind": action["kind"],
                "name": action["name"],
            })
        elif t == "mint_cash":
            initial_actions.append({
                "mint_cash": {
                    "to": action["to_agent_id"],
                    "amount": action["amount"],
                },
            })
        elif t == "mint_reserves":
            initial_actions.append({
                "mint_reserves": {
                    "to": action["to_bank_id"],
                    "amount": action["amount"],
                },
            })
        elif t == "create_payable":
            initial_actions.append({
                "create_payable": {
                    "from": action["debtor_id"],
                    "to": action["creditor_id"],
                    "amount": action["amount"],
                    "due_day": action["due_day"],
                },
            })
        elif t == "create_cb_loan":
            initial_actions.append({
                "create_cb_loan": {
                    "bank": action["bank_id"],
                    "amount": action["amount"],
                    "rate": action["rate"],
                },
            })

    return {
        "version": 1,
        "name": "Exported from Simulator",
        "agents": agents_list,
        "initial_actions": initial_actions,
        "run": {
            "max_days": state.sim_config.get("max_days", 30),
            "quiet_days": 2,
            "default_handling": state.sim_config.get("default_handling", "expel-agent"),
            "rollover": state.sim_config.get("rollover", False),
        },
    }


st.header("Step 4: Run & Review")

state = get_state()

if not state.system.state.agents:
    st.warning("No agents yet. Go to Step 1 first.")
    st.stop()

# ── Run Simulation ──────────────────────────────────────────────

if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    max_days = state.sim_config.get("max_days", 30)

    # Rebuild system from action log to ensure clean starting state
    system = rebuild_system(state.action_log)

    # [P1 fix] Apply simulation config to the rebuilt system
    default_handling = state.sim_config.get("default_handling", "expel-agent")
    system.default_mode = default_handling
    system.state.rollover_enabled = state.sim_config.get("rollover", False)

    state.system = system

    # Capture initial snapshot (setup state)
    snapshots = [capture_snapshot(system)]
    events_by_day: dict[int, list] = {}
    # [P1 fix] Track cumulative defaults per-day for accurate historical view
    defaults_by_day: dict[int, int] = {0: 0}
    already_defaulted: set[str] = set()

    # Run simulation day by day
    progress = st.progress(0, text="Running simulation...")

    for day in range(max_days):
        # system.state.day == day at this point (both start at 0)
        sim_day = system.state.day

        try:
            run_day(
                system,
                enable_dealer=False,
                enable_lender=False,
                enable_rating=False,
            )
        except Exception as e:
            st.error(f"Simulation failed on day {day + 1}: {e}")
            break

        # Capture snapshot after the day
        snapshots.append(capture_snapshot(system))

        # Collect events from the simulation's event log for this day
        day_events = []
        for evt in system.state.events_by_day.get(sim_day, []):
            kind = evt.get("kind", "")
            if kind == "PayableSettled":
                cid = evt.get("contract_id", "")
                day_events.append({"type": "settled", "instrument": cid})
            elif kind == "ObligationWrittenOff":
                cid = evt.get("contract_id", "")
                day_events.append({"type": "written_off", "instrument": cid})
            elif kind == "AgentDefaulted":
                aid = evt.get("agent", "")
                if aid and aid not in already_defaulted:
                    day_events.append({"type": "default", "agent": aid})
                    already_defaulted.add(aid)

        events_by_day[day + 1] = day_events
        defaults_by_day[day + 1] = len(already_defaulted)

        progress.progress(
            (day + 1) / max_days,
            text=f"Day {day + 1}/{max_days}",
        )

    progress.empty()

    # Store results in session
    state.snapshots = snapshots
    state.events_by_day = events_by_day
    state.defaults_by_day = defaults_by_day

    st.success(f"Simulation complete: {len(snapshots) - 1} days simulated.")

# ── Day Slider & Review ─────────────────────────────────────────

if not state.snapshots:
    st.info('Click "Run Simulation" to start.')
    st.stop()

n_days = len(state.snapshots) - 1  # snapshots[0] is setup state

day = st.slider("Day", min_value=0, max_value=n_days, value=0, help="Day 0 = initial setup state")

st.subheader(f"Day {day}" if day > 0 else "Initial State (Day 0)")

# Compute diff from previous day (if day > 0)
current_snapshot = state.snapshots[day]
diff_map = None
if day > 0:
    previous_snapshot = state.snapshots[day - 1]
    diff_map = compute_diff(previous_snapshot, current_snapshot)

# ── System Summary ──────────────────────────────────────────────

total_cash = 0
total_obligations = 0

for aid, bal in current_snapshot.items():
    total_cash += bal.assets_by_kind.get("cash", 0)
    total_obligations += bal.liabilities_by_kind.get("payable", 0)

# [P1 fix] Use per-day default counts instead of final system state
defaults_by_day = getattr(state, "defaults_by_day", {})
total_defaults = defaults_by_day.get(day, 0)

render_system_summary(
    agent_count=len(current_snapshot),
    total_cash=total_cash,
    total_obligations=total_obligations,
    total_defaults=total_defaults,
)

# ── Events for this day ─────────────────────────────────────────

if day > 0:
    events = state.events_by_day.get(day, [])
    if events:
        with st.expander(f"Events on Day {day} ({len(events)})", expanded=True):
            for evt in events:
                if evt["type"] == "settled":
                    st.markdown(f"- ✅ Settled: `{evt['instrument']}`")
                elif evt["type"] == "written_off":
                    st.markdown(f"- ⚠️ Written off: `{evt['instrument']}`")
                elif evt["type"] == "default":
                    st.markdown(f"- ❌ Default: **{evt['agent']}**")
                else:
                    st.markdown(f"- {evt}")
    else:
        st.caption("No events on this day.")

# ── Per-Agent Balance Sheets ────────────────────────────────────

st.divider()
st.subheader("Agent Balance Sheets")

agents = state.system.state.agents

# Display in a grid — 2 agents per row
agent_ids = list(current_snapshot.keys())
for i in range(0, len(agent_ids), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        idx = i + j
        if idx >= len(agent_ids):
            break
        aid = agent_ids[idx]
        agent = agents.get(aid)
        kind_val = getattr(agent.kind, "value", str(agent.kind)) if agent else "unknown"
        label = AGENT_KIND_LABELS.get(kind_val, kind_val)

        with col:
            with st.container(border=True):
                st.markdown(f"### {aid} ({label})")
                bal = current_snapshot[aid]
                diff = diff_map.get(aid) if diff_map else None
                render_balance_table(bal, diff)

# ── Export YAML ─────────────────────────────────────────────────

st.divider()
if st.button("📥 Export as YAML"):
    import yaml

    scenario = _build_export_yaml(state)
    yaml_str = yaml.dump(scenario, default_flow_style=False, sort_keys=False)
    st.download_button(
        label="Download scenario.yaml",
        data=yaml_str,
        file_name="scenario.yaml",
        mime="text/yaml",
    )
