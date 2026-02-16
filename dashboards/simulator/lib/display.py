"""Streamlit rendering helpers for the balance sheet simulator."""

from __future__ import annotations

from typing import Any

import streamlit as st

from bilancio.analysis.balances import AgentBalance

from .snapshot import AgentDiff, InstrumentDelta


def _format_amount(amount: int) -> str:
    """Format an amount with thousands separator."""
    return f"{amount:,}"


def _delta_str(delta: int) -> str:
    """Format a delta with +/- prefix and color."""
    if delta > 0:
        return f":green[+{delta:,}]"
    elif delta < 0:
        return f":red[{delta:,}]"
    return ""


def _find_delta(deltas: list[InstrumentDelta], kind: str) -> InstrumentDelta | None:
    """Find delta for a specific instrument kind."""
    for d in deltas:
        if d.kind == kind:
            return d
    return None


def render_balance_table(
    balance: AgentBalance,
    diff: AgentDiff | None = None,
) -> None:
    """Render a T-account style balance table in Streamlit.

    Shows Assets on the left, Liabilities on the right.
    If diff is provided, shows +/- annotations in green/red.
    """
    col_a, col_l = st.columns(2)

    with col_a:
        st.markdown("**Assets**")
        if not balance.assets_by_kind:
            st.caption("(none)")
        else:
            for kind, amount in sorted(balance.assets_by_kind.items()):
                delta_text = ""
                if diff:
                    d = _find_delta(diff.asset_deltas, kind)
                    if d and d.delta != 0:
                        delta_text = f" {_delta_str(d.delta)}"
                st.markdown(f"- {kind}: **{_format_amount(amount)}**{delta_text}")
            # Show new kinds that weren't in the balance but appeared in diff
            if diff:
                for d in diff.asset_deltas:
                    if d.kind not in balance.assets_by_kind and d.delta != 0:
                        st.markdown(f"- {d.kind}: **0** {_delta_str(d.delta)}")

        st.markdown(f"**Total: {_format_amount(balance.total_financial_assets)}**")

    with col_l:
        st.markdown("**Liabilities**")
        if not balance.liabilities_by_kind:
            st.caption("(none)")
        else:
            for kind, amount in sorted(balance.liabilities_by_kind.items()):
                delta_text = ""
                if diff:
                    d = _find_delta(diff.liability_deltas, kind)
                    if d and d.delta != 0:
                        delta_text = f" {_delta_str(d.delta)}"
                st.markdown(f"- {kind}: **{_format_amount(amount)}**{delta_text}")
            if diff:
                for d in diff.liability_deltas:
                    if d.kind not in balance.liabilities_by_kind and d.delta != 0:
                        st.markdown(f"- {d.kind}: **0** {_delta_str(d.delta)}")

        st.markdown(f"**Total: {_format_amount(balance.total_financial_liabilities)}**")

    st.markdown(f"*Net financial position: {_format_amount(balance.net_financial)}*")


def render_events_table(events: list[dict[str, Any]]) -> None:
    """Render simulation events for a given day as a Streamlit table."""
    if not events:
        st.info("No events for this day.")
        return

    # Format events for display
    rows = []
    for evt in events:
        rows.append({
            "Type": evt.get("type", "unknown"),
            "From": evt.get("from", evt.get("debtor", "")),
            "To": evt.get("to", evt.get("creditor", "")),
            "Amount": evt.get("amount", ""),
            "Details": evt.get("detail", evt.get("reason", "")),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_agent_card(agent_id: str, kind: str, balance: AgentBalance) -> None:
    """Render a compact agent summary card."""
    label = kind.replace("_", " ").title()
    st.markdown(
        f"**{agent_id}** ({label})  \n"
        f"Assets: {_format_amount(balance.total_financial_assets)} | "
        f"Liabilities: {_format_amount(balance.total_financial_liabilities)}"
    )


def render_system_summary(
    agent_count: int,
    total_cash: int,
    total_obligations: int,
    total_defaults: int = 0,
) -> None:
    """Render system-level summary metrics."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents", agent_count)
    c2.metric("Total Cash", _format_amount(total_cash))
    c3.metric("Total Obligations", _format_amount(total_obligations))
    c4.metric("Defaults", total_defaults)
