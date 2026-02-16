"""Unit tests for semantic HTML export."""

from pathlib import Path

from bilancio.analysis.balances import agent_balance
from bilancio.config.apply import create_agent
from bilancio.engines.system import System
from bilancio.ui.html_export import export_pretty_html


def test_export_pretty_html_minimal(tmp_path: Path):
    sys = System()
    cb = create_agent(
        type("Spec", (), {"id": "CB", "kind": "central_bank", "name": "Central Bank"})
    )
    h1 = create_agent(type("Spec", (), {"id": "H1", "kind": "household", "name": "Alice"}))
    sys.add_agent(cb)
    sys.add_agent(h1)
    # seed a simple cash position
    sys.mint_cash("H1", 1000)

    out = tmp_path / "report.html"

    # Prepare initial data
    agent_ids = ["CB", "H1"]
    initial_balances = {aid: agent_balance(sys, aid) for aid in agent_ids}

    # No days_data yet; just ensure file writes and contains headers
    export_pretty_html(
        system=sys,
        out_path=out,
        scenario_name="Unit Test",
        description="Testing export",
        agent_ids=agent_ids,
        initial_balances=initial_balances,
        days_data=[],
        max_days=1,
        quiet_days=1,
    )

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # Basic sanity checks
    assert "Bilancio Simulation" in html
    assert "Agents" in html
    assert "Day 0 (Setup)" in html
    assert "Central Bank" in html and "Alice" in html


def test_export_pretty_html_handles_numeric_strings(tmp_path: Path):
    sys = System()
    f1 = create_agent(type("Spec", (), {"id": "F1", "kind": "firm", "name": "Firm One"}))
    sys.add_agent(f1)

    out = tmp_path / "report2.html"
    agent_ids = ["F1"]
    # Manually craft a balance-like object via agent_balance and then tweak days_data to include numbers-as-strings
    initial_balances = {"F1": agent_balance(sys, "F1")}

    # days_data with an event that includes formatted numeric strings
    days_data = [
        {
            "day": 1,
            "events": [
                {"kind": "PhaseB"},
                {"kind": "ClientPayment", "payer": "F1", "payee": "F1", "amount": "1,000"},
            ],
            "balances": {"F1": agent_balance(sys, "F1")},
        }
    ]

    export_pretty_html(
        system=sys,
        out_path=out,
        scenario_name="Numeric Test",
        description=None,
        agent_ids=agent_ids,
        initial_balances=initial_balances,
        days_data=days_data,
        max_days=2,
        quiet_days=1,
    )

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Numeric Test" in html
    # amount should be present in some formatted form
    assert "1,000" in html
