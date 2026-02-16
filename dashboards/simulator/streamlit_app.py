"""Bilancio Balance Sheet Simulator — Streamlit entry point.

Run with: uv run streamlit run dashboards/simulator/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `dashboards.*` imports work
# regardless of the working directory Streamlit uses.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from dashboards.simulator.lib.session import get_state, init_session, reset_session

# ── Page config (must be first Streamlit call) ──────────────────

st.set_page_config(
    page_title="Bilancio Simulator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session ──────────────────────────────────────────

init_session()

# ── Page definitions ────────────────────────────────────────────

agents_page = st.Page("pages/1_agents.py", title="Create Agents", icon="👤")
balance_page = st.Page("pages/2_balance_sheets.py", title="Balance Sheets", icon="📊")
config_page = st.Page("pages/3_configure.py", title="Configure", icon="⚙️")
results_page = st.Page("pages/4_results.py", title="Run & Review", icon="▶️")

pg = st.navigation([agents_page, balance_page, config_page, results_page])

# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("Bilancio Simulator")
    st.caption("Interactive balance sheet simulator")

    state = get_state()
    n_agents = len(state.system.state.agents)
    n_actions = len(state.action_log)
    n_contracts = len(state.system.state.contracts)

    st.metric("Agents", n_agents)
    st.metric("Instruments", n_contracts)
    st.metric("Actions", n_actions)

    st.divider()

    if st.button("🔄 Reset All", type="secondary", use_container_width=True):
        reset_session()
        st.rerun()

# ── Run selected page ───────────────────────────────────────────

pg.run()
