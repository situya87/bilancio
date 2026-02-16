"""Page 3: Configure Simulation.

Users set simulation parameters and validate system state before running.
"""

from __future__ import annotations

import streamlit as st

from bilancio.analysis.balances import agent_balance

from dashboards.simulator.lib.session import get_state

st.header("Step 3: Configure Simulation")

state = get_state()
system = state.system
agents = system.state.agents
contracts = system.state.contracts

if not agents:
    st.warning("No agents yet. Go to Step 1 first.")
    st.stop()

# ── Simulation Parameters ───────────────────────────────────────

st.subheader("Simulation Parameters")

col1, col2 = st.columns(2)
with col1:
    state.sim_config["max_days"] = st.slider(
        "Max Days",
        min_value=1,
        max_value=100,
        value=state.sim_config.get("max_days", 30),
        help="Maximum number of days to simulate.",
    )

with col2:
    state.sim_config["default_handling"] = st.selectbox(
        "Default Handling",
        options=["expel-agent", "fail-fast"],
        index=0 if state.sim_config.get("default_handling") == "expel-agent" else 1,
        help="How to handle agent defaults.",
    )

state.sim_config["rollover"] = st.toggle(
    "Enable Rollover",
    value=state.sim_config.get("rollover", False),
    help="If enabled, settled payables are rolled over into new ones.",
)

# ── System Summary ──────────────────────────────────────────────

st.divider()
st.subheader("System Summary")

# Count instruments by kind
cash_total = 0
reserve_total = 0
payable_total = 0
cb_loan_total = 0
due_days: set[int] = set()

for c in contracts.values():
    kind = getattr(c.kind, "value", str(c.kind))
    if kind == "cash":
        cash_total += c.amount
    elif kind == "reserve_deposit":
        reserve_total += c.amount
    elif kind == "payable":
        payable_total += c.amount
        if c.due_day is not None:
            due_days.add(c.due_day)
    elif kind == "cb_loan":
        cb_loan_total += c.amount

c1, c2, c3, c4 = st.columns(4)
c1.metric("Agents", len(agents))
c2.metric("Total Cash", f"{cash_total:,}")
c3.metric("Total Obligations", f"{payable_total:,}")
c4.metric("Total Reserves", f"{reserve_total:,}")

if cb_loan_total > 0:
    st.metric("CB Loans", f"{cb_loan_total:,}")

if due_days:
    st.info(f"Payable due days: {sorted(due_days)}")
else:
    st.info("No payables configured. Settlement will have nothing to do.")

# ── Per-Agent Summary ───────────────────────────────────────────

st.subheader("Per-Agent Balances")

rows = []
for aid in agents:
    bal = agent_balance(system, aid)
    agent = agents[aid]
    kind_val = getattr(agent.kind, "value", str(agent.kind))
    rows.append({
        "Agent": aid,
        "Kind": kind_val.replace("_", " ").title(),
        "Total Assets": bal.total_financial_assets,
        "Total Liabilities": bal.total_financial_liabilities,
        "Net Position": bal.net_financial,
    })

st.dataframe(rows, use_container_width=True, hide_index=True)

# ── Validate ────────────────────────────────────────────────────

st.divider()
st.subheader("Validation")

if st.button("Validate System Invariants", type="primary", use_container_width=True):
    try:
        system.assert_invariants()
        st.success("All system invariants hold. Ready to run!")
    except AssertionError as e:
        st.error(f"Invariant violation: {e}")
    except Exception as e:
        st.error(f"Validation error: {e}")
