"""Tests for treynor_viz: dealer and bank Treynor pricing diagrams.

Covers:
- dealer_pricing_plane(): static diagram structure, annotations, edge cases
- _build_dealer_frame_full(): frame builder for animation
- dealer_pricing_animation(): animated diagram across days
- bank_pricing_plane(): static bank diagram
- bank_pricing_animation(): animated bank diagram
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from bilancio.analysis.treynor_viz import (
    _build_dealer_frame_full,
    bank_pricing_animation,
    bank_pricing_plane,
    dealer_pricing_animation,
    dealer_pricing_plane,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dealer_params():
    """Typical dealer parameters for a short bucket."""
    return dict(
        vbt_mid=0.855,
        vbt_spread=0.040,
        inventory_x=200.0,
        X_star=660.0,
        lambda_=0.0295,
        inside_width=0.00118,
        ticket_size=20.0,
        bucket_name="short",
        day=3,
    )


@pytest.fixture
def dealer_snapshots_df():
    """Minimal dealer_state.csv DataFrame with 3 days."""
    rows = []
    for d in [0, 1, 2]:
        rows.append(dict(
            day=d,
            bucket="short",
            vbt_mid=0.855 - 0.005 * d,
            vbt_spread=0.040 + 0.002 * d,
            inventory=d * 3,
            ticket_size=20,
            X_star=660,
            lambda_=0.0295,
            inside_width=0.00118,
            midline=0.855,
            bid=0.834,
            ask=0.856,
        ))
    return pd.DataFrame(rows)


@pytest.fixture
def bank_params():
    """Typical bank pricing parameters."""
    return dict(
        i_R=0.01,
        i_B=0.05,
        symmetric_capacity=500,
        ticket_size=20,
        inventory=100,
        cash_tightness=0.3,
        risk_index=0.1,
        alpha=0.5,
        gamma=0.2,
        inside_width=0.008,
        lambda_=0.2,
        bank_id="BK01",
        day=2,
    )


@pytest.fixture
def bank_snapshots_df():
    """Minimal bank_state.csv DataFrame with 3 days."""
    rows = []
    for d in [0, 1, 2]:
        rows.append(dict(
            day=d,
            bank_id="BK01",
            reserve_remuneration_rate=0.01,
            cb_borrowing_rate=0.05,
            symmetric_capacity=500,
            ticket_size=20,
            inside_width=0.008,
            inventory=50 + d * 10,
            cash_tightness=0.3,
            risk_index=0.1,
            alpha=0.5,
            gamma=0.2,
            lambda_=0.2,
        ))
    return pd.DataFrame(rows)


# ============================================================================
# dealer_pricing_plane
# ============================================================================

class TestDealerPricingPlane:
    """Tests for dealer_pricing_plane()."""

    def test_returns_figure(self, dealer_params):
        fig = dealer_pricing_plane(**dealer_params)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, dealer_params):
        fig = dealer_pricing_plane(**dealer_params)
        # At minimum: spread fill, midline, ask, bid, purple dots = 5 traces
        assert len(fig.data) >= 5

    def test_trace_names(self, dealer_params):
        fig = dealer_pricing_plane(**dealer_params)
        names = [t.name for t in fig.data if t.name]
        assert "p(x) midline" in names
        assert "a(x) ask" in names
        assert "b(x) bid" in names

    def test_midline_at_balanced_inventory(self, dealer_params):
        """At x = X*/2, the midline should equal M (within grid discretization)."""
        fig = dealer_pricing_plane(**dealer_params)
        midline_trace = [t for t in fig.data if t.name == "p(x) midline"][0]
        xs = np.array(midline_trace.x)
        ys = np.array(midline_trace.y)
        # Find the point closest to X*/2 — tolerance reflects 200-point grid
        half_X = dealer_params["X_star"] / 2
        idx = np.argmin(np.abs(xs - half_X))
        assert abs(ys[idx] - dealer_params["vbt_mid"]) < 1e-3

    def test_vbt_bounds(self, dealer_params):
        """Ask curve should not exceed A and bid curve should not go below B."""
        M = dealer_params["vbt_mid"]
        O = dealer_params["vbt_spread"]
        A = M + O / 2
        B = M - O / 2
        fig = dealer_pricing_plane(**dealer_params)
        ask_trace = [t for t in fig.data if t.name == "a(x) ask"][0]
        bid_trace = [t for t in fig.data if t.name == "b(x) bid"][0]
        assert max(ask_trace.y) <= A + 1e-10
        assert min(bid_trace.y) >= B - 1e-10

    def test_custom_title(self, dealer_params):
        dealer_params["title"] = "Custom Title"
        fig = dealer_pricing_plane(**dealer_params)
        assert fig.layout.title.text == "Custom Title"

    def test_auto_title(self, dealer_params):
        fig = dealer_pricing_plane(**dealer_params)
        title = fig.layout.title.text
        assert "short" in title
        assert "Day 3" in title

    def test_layout_properties(self, dealer_params):
        fig = dealer_pricing_plane(**dealer_params)
        assert fig.layout.plot_bgcolor == "white"
        assert fig.layout.paper_bgcolor == "white"
        assert fig.layout.height == 600

    def test_y_axis_ticks(self, dealer_params):
        """Y-axis should have ticks at B, M, A."""
        M = dealer_params["vbt_mid"]
        O = dealer_params["vbt_spread"]
        A = M + O / 2
        B = M - O / 2
        fig = dealer_pricing_plane(**dealer_params)
        tick_vals = list(fig.layout.yaxis.tickvals)
        assert len(tick_vals) == 3
        assert abs(tick_vals[0] - B) < 1e-10
        assert abs(tick_vals[1] - M) < 1e-10
        assert abs(tick_vals[2] - A) < 1e-10

    def test_x_axis_ticks(self, dealer_params):
        """X-axis should have ticks at 0, X*/2, X*."""
        fig = dealer_pricing_plane(**dealer_params)
        tick_vals = list(fig.layout.xaxis.tickvals)
        assert len(tick_vals) == 3
        assert tick_vals[0] == 0
        assert tick_vals[1] == dealer_params["X_star"] / 2
        assert tick_vals[2] == dealer_params["X_star"]

    def test_annotations_present(self, dealer_params):
        """Should have info, equations, O/I labels, curve labels, axis labels."""
        fig = dealer_pricing_plane(**dealer_params)
        annots = fig.layout.annotations
        texts = [a.text for a in annots]
        # O bracket label
        assert any("O" in t and "0.0400" in t for t in texts)
        # I bracket label
        assert any("I" in t and "0.00118" in t for t in texts)
        # Curve labels
        assert any("p(x)" in t for t in texts)
        assert any("a(x)" in t for t in texts)
        assert any("b(x)" in t for t in texts)
        # Info line
        assert any("Bucket:" in t for t in texts)
        # Equations
        assert any("p(x) = M" in t for t in texts)

    def test_purple_intersection_annotations(self, dealer_params):
        """Current ask/bid values should be annotated in purple."""
        fig = dealer_pricing_plane(**dealer_params)
        annots = fig.layout.annotations
        purple_annots = [a for a in annots if a.font and a.font.color == "#7C3AED"]
        # Two purple annotations: a(x)=... and b(x)=...
        assert len(purple_annots) >= 2
        texts = [a.text for a in purple_annots]
        assert any(t.startswith("a(") for t in texts)
        assert any(t.startswith("b(") for t in texts)


class TestDealerPricingPlaneEdgeCases:
    """Edge cases for dealer_pricing_plane()."""

    def test_zero_x_star(self):
        """X_star=0 should not crash; axes still render."""
        fig = dealer_pricing_plane(
            vbt_mid=0.9, vbt_spread=0.04, inventory_x=0.0,
            X_star=0.0, lambda_=0.03, inside_width=0.001,
            ticket_size=20.0,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 5

    def test_zero_spread(self):
        """vbt_spread=0 means A=B=M; figure should still render."""
        fig = dealer_pricing_plane(
            vbt_mid=0.9, vbt_spread=0.0, inventory_x=0.0,
            X_star=100.0, lambda_=0.03, inside_width=0.0,
            ticket_size=20.0,
        )
        assert isinstance(fig, go.Figure)

    def test_inventory_at_capacity(self):
        """Inventory at X_star boundary."""
        fig = dealer_pricing_plane(
            vbt_mid=0.85, vbt_spread=0.04, inventory_x=660.0,
            X_star=660.0, lambda_=0.03, inside_width=0.001,
            ticket_size=20.0,
        )
        assert isinstance(fig, go.Figure)

    def test_large_spread(self):
        """Very wide VBT spread should still produce valid figure."""
        fig = dealer_pricing_plane(
            vbt_mid=0.5, vbt_spread=0.8, inventory_x=0.0,
            X_star=100.0, lambda_=0.5, inside_width=0.4,
            ticket_size=10.0,
        )
        assert isinstance(fig, go.Figure)

    def test_bid_never_exceeds_ask(self, dealer_params):
        """For all x, bid curve <= ask curve."""
        fig = dealer_pricing_plane(**dealer_params)
        ask_trace = [t for t in fig.data if t.name == "a(x) ask"][0]
        bid_trace = [t for t in fig.data if t.name == "b(x) bid"][0]
        asks = np.array(ask_trace.y)
        bids = np.array(bid_trace.y)
        assert np.all(asks >= bids - 1e-12)


# ============================================================================
# _build_dealer_frame_full
# ============================================================================

class TestBuildDealerFrameFull:
    """Tests for _build_dealer_frame_full()."""

    def test_returns_tuple_of_three(self, dealer_snapshots_df):
        row = dealer_snapshots_df.iloc[0]
        result = _build_dealer_frame_full(row, bucket_name="short")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_traces_list(self, dealer_snapshots_df):
        row = dealer_snapshots_df.iloc[0]
        traces, _, _ = _build_dealer_frame_full(row, bucket_name="short")
        assert isinstance(traces, list)
        # 6 traces: fill, midline, ask, bid, purple line, intersection dots
        assert len(traces) == 6
        for t in traces:
            assert isinstance(t, go.Scatter)

    def test_shapes_list(self, dealer_snapshots_df):
        row = dealer_snapshots_df.iloc[0]
        _, shapes, _ = _build_dealer_frame_full(row, bucket_name="short")
        assert isinstance(shapes, list)
        assert len(shapes) >= 10  # corridor, bounds, mid, verticals, brackets

    def test_annotations_list(self, dealer_snapshots_df):
        row = dealer_snapshots_df.iloc[0]
        _, _, annotations = _build_dealer_frame_full(row, bucket_name="short")
        assert isinstance(annotations, list)
        assert len(annotations) >= 8  # O, A, B labels, I, p/a/b, info, x-title, eqs

    def test_global_axis_range_respected(self, dealer_snapshots_df):
        """Purple line should extend to provided y_lo/y_hi."""
        row = dealer_snapshots_df.iloc[0]
        traces, _, _ = _build_dealer_frame_full(
            row, bucket_name="short",
            y_lo=0.5, y_hi=1.0,
        )
        # Purple line is trace index 4
        purple_line = traces[4]
        assert min(purple_line.y) == pytest.approx(0.5)
        assert max(purple_line.y) == pytest.approx(1.0)

    def test_intersection_dots_in_traces(self, dealer_snapshots_df):
        """Last trace should contain intersection dots with text labels."""
        row = dealer_snapshots_df.iloc[0]
        traces, _, _ = _build_dealer_frame_full(row, bucket_name="short")
        dots_trace = traces[-1]
        assert dots_trace.mode == "markers+text"
        assert len(dots_trace.x) == 2
        assert len(dots_trace.y) == 2


# ============================================================================
# dealer_pricing_animation
# ============================================================================

class TestDealerPricingAnimation:
    """Tests for dealer_pricing_animation()."""

    def test_returns_figure(self, dealer_snapshots_df):
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket="short")
        assert isinstance(fig, go.Figure)

    def test_frame_count(self, dealer_snapshots_df):
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket="short")
        assert len(fig.frames) == 3  # days 0, 1, 2

    def test_frame_names_match_days(self, dealer_snapshots_df):
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket="short")
        names = [f.name for f in fig.frames]
        assert names == ["0", "1", "2"]

    def test_slider_present(self, dealer_snapshots_df):
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket="short")
        assert fig.layout.sliders is not None
        assert len(fig.layout.sliders) == 1

    def test_play_pause_buttons(self, dealer_snapshots_df):
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket="short")
        menus = fig.layout.updatemenus
        assert menus is not None
        assert len(menus) == 1
        buttons = menus[0]["buttons"]
        labels = [b["label"] for b in buttons]
        assert "Play" in labels
        assert "Pause" in labels

    def test_none_on_empty_df(self):
        assert dealer_pricing_animation(pd.DataFrame()) is None

    def test_none_on_none_input(self):
        assert dealer_pricing_animation(None) is None

    def test_auto_selects_first_bucket(self, dealer_snapshots_df):
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket=None)
        assert isinstance(fig, go.Figure)
        assert "short" in fig.layout.title.text

    def test_nonexistent_bucket_returns_none(self, dealer_snapshots_df):
        assert dealer_pricing_animation(dealer_snapshots_df, bucket="xyz") is None

    def test_animation_layout(self, dealer_snapshots_df):
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket="short")
        assert fig.layout.height == 750
        assert fig.layout.plot_bgcolor == "white"

    def test_global_y_range_stable(self, dealer_snapshots_df):
        """All frames should share the same y-axis range."""
        fig = dealer_pricing_animation(dealer_snapshots_df, bucket="short")
        y_range = fig.layout.yaxis.range
        assert y_range is not None
        assert len(y_range) == 2
        assert y_range[0] < y_range[1]


# ============================================================================
# bank_pricing_plane
# ============================================================================

class TestBankPricingPlane:
    """Tests for bank_pricing_plane()."""

    def test_returns_figure(self, bank_params):
        fig = bank_pricing_plane(**bank_params)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, bank_params):
        fig = bank_pricing_plane(**bank_params)
        # At least: fill, tilted midline, loan rate, deposit rate, position = 5
        assert len(fig.data) >= 5

    def test_trace_names(self, bank_params):
        fig = bank_pricing_plane(**bank_params)
        names = [t.name for t in fig.data if t.name]
        assert any("midline" in n.lower() or "m(x)" in n.lower() for n in names)
        assert any("loan" in n.lower() or "r_L" in n.lower() for n in names)
        assert any("deposit" in n.lower() or "r_D" in n.lower() for n in names)

    def test_symmetric_x_range(self, bank_params):
        """Bank pricing plane uses symmetric x-axis around zero."""
        fig = bank_pricing_plane(**bank_params)
        midline_trace = [t for t in fig.data
                         if t.name and ("midline" in t.name.lower() or "m(x)" in t.name.lower())][0]
        xs = np.array(midline_trace.x)
        assert min(xs) < 0
        assert max(xs) > 0

    def test_custom_title(self, bank_params):
        bank_params["title"] = "Bank Custom Title"
        fig = bank_pricing_plane(**bank_params)
        assert fig.layout.title.text == "Bank Custom Title"

    def test_auto_title(self, bank_params):
        fig = bank_pricing_plane(**bank_params)
        assert "BK01" in fig.layout.title.text

    def test_tilt_annotation_when_nonzero(self, bank_params):
        """When tilt > 0, there should be a tilt annotation."""
        fig = bank_pricing_plane(**bank_params)
        annots = fig.layout.annotations
        tilt_annots = [a for a in annots if "tilt" in (a.text or "").lower()]
        # alpha=0.5, gamma=0.2, cash_tightness=0.3, risk_index=0.1
        # tilt = 0.5*0.3 + 0.2*0.1 = 0.17 > 0
        assert len(tilt_annots) >= 1

    def test_no_tilt_annotation_when_zero(self, bank_params):
        """When alpha=gamma=0, no tilt annotation."""
        bank_params["alpha"] = 0
        bank_params["gamma"] = 0
        fig = bank_pricing_plane(**bank_params)
        annots = fig.layout.annotations
        tilt_annots = [a for a in annots if "tilt" in (a.text or "").lower()]
        assert len(tilt_annots) == 0


# ============================================================================
# bank_pricing_animation
# ============================================================================

class TestBankPricingAnimation:
    """Tests for bank_pricing_animation()."""

    def test_returns_figure(self, bank_snapshots_df):
        fig = bank_pricing_animation(bank_snapshots_df, bank_id="BK01")
        assert isinstance(fig, go.Figure)

    def test_frame_count(self, bank_snapshots_df):
        fig = bank_pricing_animation(bank_snapshots_df, bank_id="BK01")
        assert len(fig.frames) == 3

    def test_frame_names(self, bank_snapshots_df):
        fig = bank_pricing_animation(bank_snapshots_df, bank_id="BK01")
        names = [f.name for f in fig.frames]
        assert names == ["0", "1", "2"]

    def test_none_on_empty_df(self):
        assert bank_pricing_animation(pd.DataFrame()) is None

    def test_none_on_none_input(self):
        assert bank_pricing_animation(None) is None

    def test_auto_selects_first_bank(self, bank_snapshots_df):
        fig = bank_pricing_animation(bank_snapshots_df, bank_id=None)
        assert isinstance(fig, go.Figure)
        assert "BK01" in fig.layout.title.text

    def test_nonexistent_bank_returns_none(self, bank_snapshots_df):
        assert bank_pricing_animation(bank_snapshots_df, bank_id="BK99") is None

    def test_slider_and_buttons(self, bank_snapshots_df):
        fig = bank_pricing_animation(bank_snapshots_df, bank_id="BK01")
        assert fig.layout.sliders is not None
        assert fig.layout.updatemenus is not None


# ============================================================================
# Loader functions
# ============================================================================

class TestLoadDealerSnapshots:
    """Tests for load_dealer_snapshots()."""

    def test_loads_from_root(self, tmp_path):
        from bilancio.analysis.loaders import load_dealer_snapshots
        csv = tmp_path / "dealer_state.csv"
        csv.write_text("day,bucket,vbt_mid\n0,short,0.855\n")
        df = load_dealer_snapshots(tmp_path)
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["bucket"] == "short"

    def test_loads_from_out_subdir(self, tmp_path):
        from bilancio.analysis.loaders import load_dealer_snapshots
        out = tmp_path / "out"
        out.mkdir()
        csv = out / "dealer_state.csv"
        csv.write_text("day,bucket,vbt_mid\n0,short,0.855\n")
        df = load_dealer_snapshots(tmp_path)
        assert df is not None
        assert len(df) == 1

    def test_returns_none_if_missing(self, tmp_path):
        from bilancio.analysis.loaders import load_dealer_snapshots
        assert load_dealer_snapshots(tmp_path) is None

    def test_accepts_string_path(self, tmp_path):
        from bilancio.analysis.loaders import load_dealer_snapshots
        csv = tmp_path / "dealer_state.csv"
        csv.write_text("day,bucket,vbt_mid\n0,short,0.855\n")
        df = load_dealer_snapshots(str(tmp_path))
        assert df is not None


class TestLoadBankSnapshots:
    """Tests for load_bank_snapshots()."""

    def test_loads_from_root(self, tmp_path):
        from bilancio.analysis.loaders import load_bank_snapshots
        csv = tmp_path / "bank_state.csv"
        csv.write_text("day,bank_id,i_R\n0,BK01,0.01\n")
        df = load_bank_snapshots(tmp_path)
        assert df is not None
        assert len(df) == 1

    def test_loads_from_out_subdir(self, tmp_path):
        from bilancio.analysis.loaders import load_bank_snapshots
        out = tmp_path / "out"
        out.mkdir()
        csv = out / "bank_state.csv"
        csv.write_text("day,bank_id,i_R\n0,BK01,0.01\n")
        df = load_bank_snapshots(tmp_path)
        assert df is not None

    def test_returns_none_if_missing(self, tmp_path):
        from bilancio.analysis.loaders import load_bank_snapshots
        assert load_bank_snapshots(tmp_path) is None

    def test_accepts_string_path(self, tmp_path):
        from bilancio.analysis.loaders import load_bank_snapshots
        csv = tmp_path / "bank_state.csv"
        csv.write_text("day,bank_id,i_R\n0,BK01,0.01\n")
        df = load_bank_snapshots(str(tmp_path))
        assert df is not None
