"""Page 2: Build Balance Sheets.

Users assign instruments to agents by selecting operations
appropriate to each agent kind.
"""

from __future__ import annotations

import streamlit as st

from bilancio.analysis.balances import agent_balance

from dashboards.simulator.lib import bridge
from dashboards.simulator.lib.display import render_balance_table
from dashboards.simulator.lib.policy_ui import (
    AGENT_KIND_LABELS,
    can_create_cb_loan,
    can_issue_payable,
    can_receive_reserves,
)
from dashboards.simulator.lib.session import get_state, undo_last_action

st.header("Step 2: Build Balance Sheets")

state = get_state()
agents = state.system.state.agents

if not agents:
    st.warning("No agents yet. Go to Step 1 to create agents first.")
    st.stop()

# ── Agent Selector ──────────────────────────────────────────────

agent_ids = list(agents.keys())
selected_id = st.selectbox(
    "Select Agent",
    agent_ids,
    format_func=lambda aid: f"{aid} ({agents[aid].name})",
)

selected_agent = agents[selected_id]
agent_kind = getattr(selected_agent.kind, "value", str(selected_agent.kind))

st.subheader(f"{selected_id} — {AGENT_KIND_LABELS.get(agent_kind, agent_kind)}")

# ── Current Balance Sheet ───────────────────────────────────────

balance = agent_balance(state.system, selected_id)
render_balance_table(balance)

st.divider()

# ── Operations ──────────────────────────────────────────────────

st.subheader("Add Instruments")

# All agents can receive cash (minted by CB)
# We show mint_cash and mint_reserves on CB page, and also as "receive" on other agents

if agent_kind == "central_bank":
    # CB operations: mint cash to an agent, mint reserves to a bank
    tab_cash, tab_reserves = st.tabs(["Mint Cash", "Mint Reserves"])

    with tab_cash:
        other_agents = [aid for aid in agent_ids]
        with st.form("mint_cash_form", clear_on_submit=True):
            to_agent = st.selectbox("To Agent", other_agents, key="cash_to")
            amount = st.number_input("Amount", min_value=1, value=100, step=10, key="cash_amt")
            if st.form_submit_button("Mint Cash", type="primary"):
                try:
                    action = bridge.mint_cash(state.system, to_agent, int(amount))
                    state.action_log.append(action)
                    st.success(f"Minted {amount} cash to {to_agent}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_reserves:
        banks = [
            aid
            for aid, a in agents.items()
            if can_receive_reserves(getattr(a.kind, "value", str(a.kind)))
            and getattr(a.kind, "value", str(a.kind)) != "central_bank"
        ]
        if not banks:
            st.info("No banks in the system to receive reserves.")
        else:
            with st.form("mint_reserves_form", clear_on_submit=True):
                to_bank = st.selectbox("To Bank", banks, key="res_to")
                amount = st.number_input("Amount", min_value=1, value=100, step=10, key="res_amt")
                if st.form_submit_button("Mint Reserves", type="primary"):
                    try:
                        action = bridge.mint_reserves(state.system, to_bank, int(amount))
                        state.action_log.append(action)
                        st.success(f"Minted {amount} reserves to {to_bank}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

else:
    # Non-CB agents
    tabs = []
    tab_names = []

    if can_issue_payable(agent_kind):
        tab_names.append("Create Payable")
    if can_create_cb_loan(agent_kind):
        tab_names.append("CB Loan")

    if not tab_names:
        st.info(
            "This agent kind has no direct operations. "
            "Use the Central Bank to mint cash/reserves to this agent."
        )
    else:
        active_tabs = st.tabs(tab_names)
        tab_idx = 0

        if can_issue_payable(agent_kind):
            with active_tabs[tab_idx]:
                # Create a payable where selected agent is debtor
                creditors = [aid for aid in agent_ids if aid != selected_id]
                if not creditors:
                    st.info("Need at least one other agent as creditor.")
                else:
                    with st.form("create_payable_form", clear_on_submit=True):
                        creditor = st.selectbox("Creditor (owed to)", creditors, key="pay_cred")
                        amount = st.number_input(
                            "Amount", min_value=1, value=50, step=10, key="pay_amt"
                        )
                        due_day = st.number_input(
                            "Due Day", min_value=1, value=2, step=1, key="pay_due"
                        )
                        if st.form_submit_button("Create Payable", type="primary"):
                            try:
                                action = bridge.create_payable(
                                    state.system,
                                    debtor_id=selected_id,
                                    creditor_id=creditor,
                                    amount=int(amount),
                                    due_day=int(due_day),
                                )
                                state.action_log.append(action)
                                st.success(
                                    f"Created payable: {selected_id} owes {creditor} "
                                    f"{amount} due day {due_day}"
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            tab_idx += 1

        if can_create_cb_loan(agent_kind):
            with active_tabs[tab_idx]:
                with st.form("create_cb_loan_form", clear_on_submit=True):
                    amount = st.number_input(
                        "Loan Amount", min_value=1, value=100, step=10, key="cbl_amt"
                    )
                    rate = st.text_input("Interest Rate", value="0.03", key="cbl_rate")
                    if st.form_submit_button("Create CB Loan", type="primary"):
                        try:
                            action = bridge.create_cb_loan(
                                state.system,
                                bank_id=selected_id,
                                amount=int(amount),
                                rate=rate,
                            )
                            state.action_log.append(action)
                            st.success(f"Created CB loan: {amount} to {selected_id} at {rate}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            tab_idx += 1

# ── Undo ────────────────────────────────────────────────────────

st.divider()
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("↩️ Undo Last Action", use_container_width=True):
        if undo_last_action():
            st.success("Undid last action.")
            st.rerun()
        else:
            st.warning("Nothing to undo.")

with col2:
    if state.action_log:
        last = state.action_log[-1]
        st.caption(f"Last action: {last['type']} — {last}")
