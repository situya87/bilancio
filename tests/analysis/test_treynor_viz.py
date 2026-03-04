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
    BUCKET_TAU,
    _add_yield_curve_sections,
    _build_dealer_frame_full,
    _build_yield_curve_frame,
    bank_pricing_animation,
    bank_pricing_plane,
    dealer_pricing_animation,
    dealer_pricing_plane,
    yield_curve_animation,
    yield_curve_static,
    yield_curve_timeseries,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dealer_params():
    """Typical dealer parameters for a short bucket."""
    return {
        "vbt_mid": 0.855,
        "vbt_spread": 0.040,
        "inventory_x": 200.0,
        "X_star": 660.0,
        "lambda_": 0.0295,
        "inside_width": 0.00118,
        "ticket_size": 20.0,
        "bucket_name": "short",
        "day": 3,
    }


@pytest.fixture
def dealer_snapshots_df():
    """Minimal dealer_state.csv DataFrame with 3 days."""
    rows = []
    for d in [0, 1, 2]:
        rows.append({
            "day": d,
            "bucket": "short",
            "vbt_mid": 0.855 - 0.005 * d,
            "vbt_spread": 0.040 + 0.002 * d,
            "inventory": d * 3,
            "ticket_size": 20,
            "X_star": 660,
            "lambda_": 0.0295,
            "inside_width": 0.00118,
            "midline": 0.855,
            "bid": 0.834,
            "ask": 0.856,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def bank_params():
    """Typical bank pricing parameters."""
    return {
        "i_R": 0.01,
        "i_B": 0.05,
        "symmetric_capacity": 500,
        "ticket_size": 20,
        "inventory": 100,
        "cash_tightness": 0.3,
        "risk_index": 0.1,
        "alpha": 0.5,
        "gamma": 0.2,
        "inside_width": 0.008,
        "lambda_": 0.2,
        "bank_id": "BK01",
        "day": 2,
    }


@pytest.fixture
def bank_snapshots_df():
    """Minimal bank_state.csv DataFrame with 3 days."""
    rows = []
    for d in [0, 1, 2]:
        rows.append({
            "day": d,
            "bank_id": "BK01",
            "reserve_remuneration_rate": 0.01,
            "cb_borrowing_rate": 0.05,
            "symmetric_capacity": 500,
            "ticket_size": 20,
            "inside_width": 0.008,
            "inventory": 50 + d * 10,
            "cash_tightness": 0.3,
            "risk_index": 0.1,
            "alpha": 0.5,
            "gamma": 0.2,
            "lambda_": 0.2,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def yield_curve_df():
    """DataFrame with 3 buckets x 3 days for yield curve tests."""
    rows = []
    for d in [0, 1, 2]:
        # Short bucket: highest price (lowest discount)
        rows.append({
            "day": d, "bucket": "short",
            "vbt_mid": 0.95 - 0.01 * d, "vbt_spread": 0.04,
            "midline": 0.94 - 0.01 * d, "bid": 0.92 - 0.01 * d, "ask": 0.96 - 0.01 * d,
            "inventory": 3, "ticket_size": 20, "X_star": 660, "lambda_": 0.03, "inside_width": 0.001,
        })
        # Mid bucket: medium price
        rows.append({
            "day": d, "bucket": "mid",
            "vbt_mid": 0.85 - 0.02 * d, "vbt_spread": 0.06,
            "midline": 0.84 - 0.02 * d, "bid": 0.81 - 0.02 * d, "ask": 0.87 - 0.02 * d,
            "inventory": 2, "ticket_size": 20, "X_star": 660, "lambda_": 0.03, "inside_width": 0.001,
        })
        # Long bucket: lowest price (highest discount)
        rows.append({
            "day": d, "bucket": "long",
            "vbt_mid": 0.75 - 0.03 * d, "vbt_spread": 0.08,
            "midline": 0.73 - 0.03 * d, "bid": 0.69 - 0.03 * d, "ask": 0.77 - 0.03 * d,
            "inventory": 1, "ticket_size": 20, "X_star": 660, "lambda_": 0.03, "inside_width": 0.001,
        })
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
        spread = dealer_params["vbt_spread"]
        A = M + spread / 2
        B = M - spread / 2
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
        spread = dealer_params["vbt_spread"]
        A = M + spread / 2
        B = M - spread / 2
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


# ============================================================================
# BUCKET_TAU
# ============================================================================

class TestBucketTau:
    """Tests for BUCKET_TAU constant."""

    def test_has_three_buckets(self):
        assert len(BUCKET_TAU) == 3

    def test_values(self):
        assert BUCKET_TAU["short"] == 2
        assert BUCKET_TAU["mid"] == 6
        assert BUCKET_TAU["long"] == 12

    def test_ascending_order(self):
        assert BUCKET_TAU["short"] < BUCKET_TAU["mid"] < BUCKET_TAU["long"]


# ============================================================================
# _build_yield_curve_frame
# ============================================================================

class TestBuildYieldCurveFrame:
    """Tests for _build_yield_curve_frame()."""

    def test_returns_four_traces(self, yield_curve_df):
        day_df = yield_curve_df[yield_curve_df["day"] == 0]
        traces = _build_yield_curve_frame(day_df, day=0)
        assert len(traces) == 4

    def test_all_traces_are_scatter(self, yield_curve_df):
        day_df = yield_curve_df[yield_curve_df["day"] == 0]
        traces = _build_yield_curve_frame(day_df, day=0)
        for t in traces:
            assert isinstance(t, go.Scatter)

    def test_vbt_mid_trace_values(self, yield_curve_df):
        day_df = yield_curve_df[yield_curve_df["day"] == 0]
        traces = _build_yield_curve_frame(day_df, day=0)
        vbt_trace = traces[1]  # VBT mid yield
        # Should have 3 points sorted by tau
        assert len(vbt_trace.x) == 3
        assert list(vbt_trace.x) == [2, 6, 12]  # short, mid, long tau
        # Yields should increase (short < mid < long)
        assert vbt_trace.y[0] < vbt_trace.y[1] < vbt_trace.y[2]

    def test_midline_trace_values(self, yield_curve_df):
        day_df = yield_curve_df[yield_curve_df["day"] == 0]
        traces = _build_yield_curve_frame(day_df, day=0)
        midline_trace = traces[2]  # Dealer midline yield
        assert len(midline_trace.x) == 3
        # Yield = 1/P - 1: short (1/0.94-1≈0.0638) < mid (1/0.84-1≈0.1905) < long (1/0.73-1≈0.3699)
        assert midline_trace.y[0] == pytest.approx(1.0 / 0.94 - 1.0, abs=1e-6)
        assert midline_trace.y[1] == pytest.approx(1.0 / 0.84 - 1.0, abs=1e-6)
        assert midline_trace.y[2] == pytest.approx(1.0 / 0.73 - 1.0, abs=1e-6)

    def test_empty_for_unknown_buckets(self):
        df = pd.DataFrame([{"day": 0, "bucket": "unknown", "vbt_mid": 0.9,
                            "vbt_spread": 0.04, "midline": 0.89, "bid": 0.87, "ask": 0.91}])
        traces = _build_yield_curve_frame(df, day=0)
        assert traces == []

    def test_single_bucket(self, yield_curve_df):
        day_df = yield_curve_df[(yield_curve_df["day"] == 0) & (yield_curve_df["bucket"] == "short")]
        traces = _build_yield_curve_frame(day_df, day=0)
        assert len(traces) == 4
        assert len(traces[1].x) == 1


# ============================================================================
# yield_curve_static
# ============================================================================

class TestYieldCurveStatic:
    """Tests for yield_curve_static()."""

    def test_returns_figure(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        assert isinstance(fig, go.Figure)

    def test_defaults_to_last_day(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        title = fig.layout.title.text
        assert "Day 2" in title

    def test_specific_day(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df, day=0)
        title = fig.layout.title.text
        assert "Day 0" in title

    def test_has_traces(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df, day=0)
        assert len(fig.data) == 4  # band + vbt + midline + placeholder

    def test_trace_names(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        names = [t.name for t in fig.data if t.name]
        assert "VBT mid yield" in names
        assert "Dealer midline yield" in names

    def test_x_axis_range(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        assert tuple(fig.layout.xaxis.range) == (0, 14)

    def test_y_axis_starts_at_zero(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        assert fig.layout.yaxis.rangemode == "tozero"

    def test_y_axis_percent_format(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        assert fig.layout.yaxis.tickformat == ".1%"

    def test_layout_style(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        assert fig.layout.plot_bgcolor == "white"
        assert fig.layout.paper_bgcolor == "white"
        assert fig.layout.height == 550

    def test_annotation_present(self, yield_curve_df):
        fig = yield_curve_static(yield_curve_df)
        annots = fig.layout.annotations
        texts = [a.text for a in annots]
        assert any("Day:" in t for t in texts)
        assert any("Buckets:" in t for t in texts)

    def test_none_on_empty_df(self):
        assert yield_curve_static(pd.DataFrame()) is None

    def test_none_on_none_input(self):
        assert yield_curve_static(None) is None

    def test_none_for_nonexistent_day(self, yield_curve_df):
        assert yield_curve_static(yield_curve_df, day=999) is None

    def test_upward_sloping(self, yield_curve_df):
        """Yield curve should slope upward (longer maturity = higher yield)."""
        fig = yield_curve_static(yield_curve_df, day=0)
        midline_trace = [t for t in fig.data if t.name == "Dealer midline yield"][0]
        # Points sorted by tau, yields should increase
        ys = list(midline_trace.y)
        assert ys == sorted(ys)


# ============================================================================
# yield_curve_animation
# ============================================================================

class TestYieldCurveAnimation:
    """Tests for yield_curve_animation()."""

    def test_returns_figure(self, yield_curve_df):
        fig = yield_curve_animation(yield_curve_df)
        assert isinstance(fig, go.Figure)

    def test_frame_count(self, yield_curve_df):
        fig = yield_curve_animation(yield_curve_df)
        assert len(fig.frames) == 3  # days 0, 1, 2

    def test_frame_names_match_days(self, yield_curve_df):
        fig = yield_curve_animation(yield_curve_df)
        names = [f.name for f in fig.frames]
        assert names == ["0", "1", "2"]

    def test_slider_present(self, yield_curve_df):
        fig = yield_curve_animation(yield_curve_df)
        assert fig.layout.sliders is not None
        assert len(fig.layout.sliders) == 1

    def test_play_pause_buttons(self, yield_curve_df):
        fig = yield_curve_animation(yield_curve_df)
        menus = fig.layout.updatemenus
        assert menus is not None
        assert len(menus) == 1
        buttons = menus[0]["buttons"]
        labels = [b["label"] for b in buttons]
        assert "Play" in labels
        assert "Pause" in labels

    def test_global_y_range(self, yield_curve_df):
        fig = yield_curve_animation(yield_curve_df)
        y_range = fig.layout.yaxis.range
        assert y_range is not None
        assert y_range[0] == 0
        assert y_range[1] > 0

    def test_layout_style(self, yield_curve_df):
        fig = yield_curve_animation(yield_curve_df)
        assert fig.layout.height == 550
        assert fig.layout.plot_bgcolor == "white"

    def test_none_on_empty_df(self):
        assert yield_curve_animation(pd.DataFrame()) is None

    def test_none_on_none_input(self):
        assert yield_curve_animation(None) is None


# ============================================================================
# yield_curve_timeseries
# ============================================================================

class TestYieldCurveTimeseries:
    """Tests for yield_curve_timeseries()."""

    def test_returns_figure(self, yield_curve_df):
        fig = yield_curve_timeseries(yield_curve_df)
        assert isinstance(fig, go.Figure)

    def test_three_traces(self, yield_curve_df):
        fig = yield_curve_timeseries(yield_curve_df)
        assert len(fig.data) == 3  # one per bucket

    def test_trace_names(self, yield_curve_df):
        fig = yield_curve_timeseries(yield_curve_df)
        names = [t.name for t in fig.data]
        assert any("short" in n for n in names)
        assert any("mid" in n for n in names)
        assert any("long" in n for n in names)

    def test_dash_styles(self, yield_curve_df):
        fig = yield_curve_timeseries(yield_curve_df)
        dashes = {t.name.split()[0]: t.line.dash for t in fig.data}
        assert dashes["short"] == "solid"
        assert dashes["mid"] == "dash"
        assert dashes["long"] == "dot"

    def test_x_values_are_days(self, yield_curve_df):
        fig = yield_curve_timeseries(yield_curve_df)
        for trace in fig.data:
            assert list(trace.x) == [0, 1, 2]

    def test_yields_increase_over_time(self, yield_curve_df):
        """Midline drops each day, so yield (1/P - 1) increases."""
        fig = yield_curve_timeseries(yield_curve_df)
        for trace in fig.data:
            ys = list(trace.y)
            assert ys == sorted(ys)

    def test_layout_style(self, yield_curve_df):
        fig = yield_curve_timeseries(yield_curve_df)
        assert fig.layout.height == 450
        assert fig.layout.plot_bgcolor == "white"
        assert fig.layout.yaxis.showgrid is True

    def test_title(self, yield_curve_df):
        fig = yield_curve_timeseries(yield_curve_df)
        assert "Term Structure" in fig.layout.title.text

    def test_none_on_empty_df(self):
        assert yield_curve_timeseries(pd.DataFrame()) is None

    def test_none_on_none_input(self):
        assert yield_curve_timeseries(None) is None

    def test_single_bucket(self):
        df = pd.DataFrame([
            {"day": 0, "bucket": "short", "vbt_mid": 0.95, "vbt_spread": 0.04,
                 "midline": 0.94, "bid": 0.92, "ask": 0.96},
            {"day": 1, "bucket": "short", "vbt_mid": 0.94, "vbt_spread": 0.04,
                 "midline": 0.93, "bid": 0.91, "ask": 0.95},
        ])
        fig = yield_curve_timeseries(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


# ============================================================================
# _add_yield_curve_sections
# ============================================================================

class TestAddYieldCurveSections:
    """Tests for _add_yield_curve_sections()."""

    def test_adds_two_sections(self, yield_curve_df):
        sections = []
        nav_items = []
        _add_yield_curve_sections(yield_curve_df, sections, nav_items)
        assert len(sections) == 2

    def test_adds_two_nav_items(self, yield_curve_df):
        sections = []
        nav_items = []
        _add_yield_curve_sections(yield_curve_df, sections, nav_items)
        assert len(nav_items) == 2
        ids = [item[0] for item in nav_items]
        assert "yield-curve" in ids
        assert "term-structure" in ids

    def test_sections_contain_html(self, yield_curve_df):
        sections = []
        nav_items = []
        _add_yield_curve_sections(yield_curve_df, sections, nav_items)
        # Yield curve section has static + animation charts
        assert 'id="yield-curve"' in sections[0]
        assert 'id="term-structure"' in sections[1]

    def test_sections_contain_chart_containers(self, yield_curve_df):
        sections = []
        nav_items = []
        _add_yield_curve_sections(yield_curve_df, sections, nav_items)
        assert "chart-container" in sections[0]
        assert "chart-container" in sections[1]
