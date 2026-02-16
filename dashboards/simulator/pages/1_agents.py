"""Page 1: Create Agents.

Users create agents by selecting a kind, entering an ID and name.
A central bank is auto-created on first load.
"""

from __future__ import annotations

import streamlit as st

from dashboards.simulator.lib import bridge
from dashboards.simulator.lib.policy_ui import AGENT_KIND_LABELS, AGENT_KINDS
from dashboards.simulator.lib.session import get_state


def _ensure_central_bank() -> None:
    """Auto-create a central bank if none exists."""
    state = get_state()
    agents = state.system.state.agents
    has_cb = any(
        getattr(a.kind, "value", a.kind) == "central_bank" for a in agents.values()
    )
    if not has_cb:
        action = bridge.add_agent(state.system, "central_bank", "CB", "Central Bank")
        state.action_log.append(action)


# ── Auto-create CB ──────────────────────────────────────────────

_ensure_central_bank()

# ── Page layout ─────────────────────────────────────────────────

st.header("Step 1: Create Agents")
st.markdown("Add agents to your financial system. A central bank is created automatically.")

state = get_state()

# ── Add Agent Form ──────────────────────────────────────────────

with st.form("add_agent_form", clear_on_submit=True):
    st.subheader("Add New Agent")

    col1, col2, col3 = st.columns(3)
    with col1:
        kind = st.selectbox(
            "Agent Kind",
            options=AGENT_KINDS,
            format_func=lambda k: AGENT_KIND_LABELS.get(k, k),
        )
    with col2:
        agent_id = st.text_input("Agent ID", placeholder="e.g., F1, B1, H1")
    with col3:
        name = st.text_input("Name", placeholder="e.g., Firm Alpha")

    submitted = st.form_submit_button("Add Agent", type="primary", use_container_width=True)

    if submitted:
        if not agent_id:
            st.error("Agent ID is required.")
        elif agent_id in state.system.state.agents:
            st.error(f"Agent '{agent_id}' already exists.")
        else:
            try:
                action = bridge.add_agent(state.system, kind, agent_id, name or agent_id)
                state.action_log.append(action)
                st.success(f"Added {AGENT_KIND_LABELS.get(kind, kind)}: {agent_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to add agent: {e}")

# ── Current Agents Table ────────────────────────────────────────

st.subheader("Current Agents")

agents = state.system.state.agents
if not agents:
    st.info("No agents yet. Add one above.")
else:
    rows = []
    for aid, agent in agents.items():
        kind_val = getattr(agent.kind, "value", str(agent.kind))
        n_assets = len(agent.asset_ids)
        n_liabs = len(agent.liability_ids)
        rows.append({
            "ID": aid,
            "Kind": AGENT_KIND_LABELS.get(kind_val, kind_val),
            "Name": agent.name,
            "Assets": n_assets,
            "Liabilities": n_liabs,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Remove Agent ────────────────────────────────────────────

    st.subheader("Remove Agent")
    removable = [
        aid
        for aid, a in agents.items()
        if len(a.asset_ids) == 0 and len(a.liability_ids) == 0
        and getattr(a.kind, "value", a.kind) != "central_bank"
    ]

    if removable:
        to_remove = st.selectbox("Select agent to remove", removable)
        if st.button("Remove Agent", type="secondary"):
            # Remove from system
            del state.system.state.agents[to_remove]
            # Remove from action log (filter out add_agent for this id)
            state.action_log = [
                a
                for a in state.action_log
                if not (a["type"] == "add_agent" and a["agent_id"] == to_remove)
            ]
            st.success(f"Removed agent: {to_remove}")
            st.rerun()
    else:
        st.caption("No removable agents (must have no instruments, and can't remove central bank).")
