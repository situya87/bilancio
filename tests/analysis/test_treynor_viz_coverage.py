"""Additional coverage tests for treynor_viz.py.

Targets: build_treynor_dashboard, _add_dealer_sections, _add_bank_sections,
_fig_to_div, _dashboard_shell, and edge cases not covered by existing tests.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from bilancio.analysis.treynor_viz import (
    _add_bank_sections,
    _add_dealer_sections,
    _add_interbank_sections,
    _dashboard_shell,
    _fig_to_div,
    build_treynor_dashboard,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def dealer_csv_content():
    """Minimal dealer_state.csv content."""
    return (
        "day,bucket,vbt_mid,vbt_spread,inventory,ticket_size,X_star,"
        "lambda_,inside_width,midline,bid,ask\n"
        "0,short,0.855,0.040,3,20,660,0.0295,0.00118,0.855,0.834,0.856\n"
        "1,short,0.850,0.042,5,20,660,0.0295,0.00124,0.850,0.829,0.851\n"
    )


@pytest.fixture
def bank_csv_content():
    """Minimal bank_state.csv content."""
    return (
        "day,bank_id,reserve_remuneration_rate,cb_borrowing_rate,"
        "symmetric_capacity,ticket_size,inside_width,inventory,"
        "cash_tightness,risk_index,alpha,gamma,lambda_\n"
        "0,BK01,0.01,0.05,500,20,0.008,100,0.3,0.1,0.5,0.2,0.2\n"
        "1,BK01,0.01,0.05,500,20,0.008,120,0.35,0.12,0.5,0.2,0.2\n"
    )


@pytest.fixture
def interbank_events_content():
    """Minimal events.jsonl content with interbank auction state."""
    events = [
        {
            "kind": "InterbankAuction",
            "day": 0,
            "clearing_rate": "0.05",
            "total_volume": 100,
            "n_trades": 1,
            "n_unfilled": 0,
            "market_state": {
                "positions": [
                    {"bank_id": "BK01", "position": 100, "limit_rate": "0.03", "side": "lend"},
                    {"bank_id": "BK02", "position": -100, "limit_rate": "0.07", "side": "borrow"},
                ],
                "lender_asks": [
                    {"bank_id": "BK01", "quantity": 100, "limit_rate": "0.03"},
                ],
                "borrower_bids": [
                    {"bank_id": "BK02", "quantity": 100, "limit_rate": "0.07"},
                ],
            },
        },
        {
            "kind": "InterbankAuctionTrade",
            "day": 0,
            "lender": "BK01",
            "borrower": "BK02",
            "amount": 100,
            "rate": "0.05",
            "maturity_day": 1,
            "contract_id": "IBL_1",
        },
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


# ============================================================================
# _fig_to_div
# ============================================================================


class TestFigToDiv:
    """Tests for _fig_to_div."""

    def test_returns_string(self):
        fig = go.Figure()
        result = _fig_to_div(fig)
        assert isinstance(result, str)
        assert "<div" in result

    def test_with_div_id(self):
        fig = go.Figure()
        result = _fig_to_div(fig, "my-chart")
        assert "my-chart" in result


# ============================================================================
# _dashboard_shell
# ============================================================================


class TestDashboardShell:
    """Tests for _dashboard_shell."""

    def test_basic_structure(self):
        html = _dashboard_shell("Test Dashboard", [], "<p>content</p>")
        assert "<!DOCTYPE html>" in html
        assert "Test Dashboard" in html
        assert "<p>content</p>" in html
        assert "plotly" in html.lower()

    def test_nav_items(self):
        nav = [("sec1", "Section 1"), ("sec2", "Section 2")]
        html = _dashboard_shell("Test", nav, "body")
        assert 'href="#sec1"' in html
        assert "Section 1" in html
        assert 'href="#sec2"' in html
        assert "Section 2" in html


# ============================================================================
# build_treynor_dashboard
# ============================================================================


class TestBuildTreynorDashboard:
    """Tests for build_treynor_dashboard."""

    def test_no_data_returns_no_data_message(self, tmp_path):
        """When no CSV files exist, should return a page with 'No Data'."""
        html = build_treynor_dashboard(tmp_path)
        assert "No Data" in html
        assert "<!DOCTYPE html>" in html

    def test_with_dealer_data(self, tmp_path, dealer_csv_content):
        csv_path = tmp_path / "dealer_state.csv"
        csv_path.write_text(dealer_csv_content)
        html = build_treynor_dashboard(tmp_path)
        assert "Dealer Pricing" in html
        assert "Yield Curve" in html
        assert "<!DOCTYPE html>" in html

    def test_with_bank_data(self, tmp_path, bank_csv_content):
        csv_path = tmp_path / "bank_state.csv"
        csv_path.write_text(bank_csv_content)
        html = build_treynor_dashboard(tmp_path)
        assert "Bank Pricing" in html
        assert "<!DOCTYPE html>" in html

    def test_with_both_data(self, tmp_path, dealer_csv_content, bank_csv_content):
        (tmp_path / "dealer_state.csv").write_text(dealer_csv_content)
        (tmp_path / "bank_state.csv").write_text(bank_csv_content)
        html = build_treynor_dashboard(tmp_path)
        assert "Dealer Pricing" in html
        assert "Bank Pricing" in html
        assert "Yield Curve" in html

    def test_with_interbank_events(self, tmp_path, interbank_events_content):
        (tmp_path / "events.jsonl").write_text(interbank_events_content)
        html = build_treynor_dashboard(tmp_path)
        assert "Interbank Market" in html
        assert "Interbank Lending Flows" in html

    def test_from_out_subdir(self, tmp_path, dealer_csv_content):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "dealer_state.csv").write_text(dealer_csv_content)
        html = build_treynor_dashboard(tmp_path)
        assert "Dealer Pricing" in html


# ============================================================================
# _add_dealer_sections
# ============================================================================


class TestAddDealerSections:
    """Tests for _add_dealer_sections."""

    def test_adds_static_and_animation(self, dealer_csv_content, tmp_path):
        csv_path = tmp_path / "dealer_state.csv"
        csv_path.write_text(dealer_csv_content)
        df = pd.read_csv(csv_path)

        sections: list[str] = []
        nav_items: list[tuple[str, str]] = []
        _add_dealer_sections(df, sections, nav_items)

        assert len(sections) == 2  # static + animation
        assert len(nav_items) == 2
        assert nav_items[0][0] == "dealer-static"
        assert nav_items[1][0] == "dealer-anim"
        assert "Dealer Pricing Diagrams" in sections[0]
        assert "Dealer Pricing Animation" in sections[1]


# ============================================================================
# _add_bank_sections
# ============================================================================


class TestAddBankSections:
    """Tests for _add_bank_sections."""

    def test_adds_static_and_animation(self, bank_csv_content, tmp_path):
        csv_path = tmp_path / "bank_state.csv"
        csv_path.write_text(bank_csv_content)
        df = pd.read_csv(csv_path)

        sections: list[str] = []
        nav_items: list[tuple[str, str]] = []
        _add_bank_sections(df, sections, nav_items)

        assert len(sections) == 2  # static + animation
        assert len(nav_items) == 2
        assert nav_items[0][0] == "bank-static"
        assert nav_items[1][0] == "bank-anim"
        assert "Bank Pricing Diagrams" in sections[0]
        assert "Bank Pricing Animation" in sections[1]


# ============================================================================
# _add_interbank_sections
# ============================================================================

class TestAddInterbankSections:
    """Tests for _add_interbank_sections."""

    def test_adds_interbank_section(self, interbank_events_content, tmp_path):
        events_path = tmp_path / "events.jsonl"
        events_path.write_text(interbank_events_content)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]

        sections: list[str] = []
        nav_items: list[tuple[str, str]] = []
        _add_interbank_sections(events, sections, nav_items)

        assert len(sections) == 1
        assert nav_items == [("interbank-market", "Interbank Market")]
        assert "Interbank Market" in sections[0]
