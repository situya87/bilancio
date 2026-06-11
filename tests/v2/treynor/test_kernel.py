"""The kernel must reproduce the chapters' worked examples exactly.

Test vectors come from:
- Chapter 1, Part 4 (baseline: M=100, O=0.50, S=1, V=1000) and Part 5
  (trades at x=7 and x=6).
- Chapter 2, Part 8 (spot book X*=7 at x=5; speculative book with the
  asymmetric corridor X⁺=200, X⁻=500 and the +3.33bp tilt at x=0).

Plus property tests of the zero-profit identity itself: under the
dealer's internal 50/50 model, occupancy is uniform, layoff frequency
converges to λ, and net P&L per trade converges to zero.
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from bilancio_v2.treynor.kernel import (
    Orientation,
    TreynorCorridor,
    simulate_walk,
    snap_down,
)

D = Decimal


def close(a: Decimal, b: Decimal, tol: str = "0.0005") -> bool:
    return abs(a - b) <= D(tol)


class TestChapter1Baseline:
    """Ch1 Part 4: M=100, O=0.50, S=1, X*=10 (V=1000), dealer at x=7."""

    @pytest.fixture
    def corridor(self) -> TreynorCorridor:
        return TreynorCorridor.one_sided_price(ticket=D(1), capacity=D(10), mid=D(100), outside_spread=D("0.50"))

    def test_grid(self, corridor: TreynorCorridor) -> None:
        assert corridor.n_rungs == 11
        assert corridor.center == D(5)

    def test_layoff_probability(self, corridor: TreynorCorridor) -> None:
        assert corridor.layoff_probability == D(1) / D(11)

    def test_inside_spread(self, corridor: TreynorCorridor) -> None:
        # I = λ × O = 0.50/11 = 0.04545…
        assert close(corridor.inside_spread, D("0.04545"), "0.00001")

    def test_slope(self, corridor: TreynorCorridor) -> None:
        # k = O / (X* + 2S) = 0.50/12 = 0.04167
        assert close(corridor.slope, D("0.04167"), "0.00001")

    def test_midline_and_quotes_at_7(self, corridor: TreynorCorridor) -> None:
        assert close(corridor.midline(D(7)), D("99.917"))
        assert close(corridor.ask(D(7)), D("99.939"))
        assert close(corridor.bid(D(7)), D("99.894"))

    def test_quotes_at_6(self, corridor: TreynorCorridor) -> None:
        # Ch1 Part 5, Trade 2: midline(6) = 99.958, bid(6) = 99.936
        assert close(corridor.midline(D(6)), D("99.958"))
        assert close(corridor.bid(D(6)), D("99.936"))

    def test_balanced_position_quotes_at_mid(self, corridor: TreynorCorridor) -> None:
        assert corridor.midline(D(5)) == D(100)

    def test_ghost_points_hit_vbt_quotes(self, corridor: TreynorCorridor) -> None:
        assert corridor.ghost_value(upper=False) == D("100.25")  # A at x = −S
        assert corridor.ghost_value(upper=True) == D("99.75")  # B at x = X*+S

    def test_all_quotes_inside_corridor(self, corridor: TreynorCorridor) -> None:
        # Ch1 Part 4: every posted quote sits strictly inside [B, A].
        for rung in range(11):
            x = D(rung)
            assert D("99.75") < corridor.bid(x)
            assert corridor.ask(x) < D("100.25")
            assert close(corridor.ask(x) - corridor.bid(x), corridor.inside_spread, "1e-20")


class TestChapter2SpotBook:
    """Ch2 Part 8, Step 1: V=700 ⇒ X*=7, dealer at x_spot=5."""

    @pytest.fixture
    def corridor(self) -> TreynorCorridor:
        capacity = snap_down(D(700) / D(100), D(1))  # floor(V/M) = 7
        return TreynorCorridor.one_sided_price(ticket=D(1), capacity=capacity, mid=D(100), outside_spread=D("0.50"))

    def test_capacity_and_lambda(self, corridor: TreynorCorridor) -> None:
        assert corridor.upper_bound == D(7)
        assert corridor.layoff_probability == D("0.125")  # 1/8

    def test_inside_spread(self, corridor: TreynorCorridor) -> None:
        assert corridor.inside_spread == D("0.0625")

    def test_slope(self, corridor: TreynorCorridor) -> None:
        assert close(corridor.slope, D("0.0556"), "0.0001")  # 0.50/9

    def test_quotes_at_5(self, corridor: TreynorCorridor) -> None:
        assert close(corridor.midline(D(5)), D("99.917"))
        assert close(corridor.ask(D(5)), D("99.95"), "0.005")
        assert close(corridor.bid(D(5)), D("99.89"), "0.005")


class TestChapter2SpeculativeBook:
    """Ch2 Part 8, Step 2: the asymmetric speculative corridor.

    X⁺=200 (cash), X⁻=500 (securities), S=100, M_spec=4.00%, O_spec=0.20%.
    """

    @pytest.fixture
    def corridor(self) -> TreynorCorridor:
        return TreynorCorridor.two_sided_rate(
            ticket=D(100),
            upper_capacity=D(200),
            lower_capacity=D(500),
            mid_rate=D("0.0400"),
            outside_spread=D("0.0020"),
        )

    def test_corridor_width_is_v(self, corridor: TreynorCorridor) -> None:
        assert corridor.width == D(700)  # L_act = X⁺ + X⁻ = V

    def test_lambda(self, corridor: TreynorCorridor) -> None:
        assert corridor.layoff_probability == D("0.125")  # 100/800

    def test_inside_spread(self, corridor: TreynorCorridor) -> None:
        # I_spec = 0.125 × 20bp = 2.5bp
        assert corridor.inside_spread == D("0.00025")

    def test_corridor_center(self, corridor: TreynorCorridor) -> None:
        assert corridor.center == D(-150)  # μ = (X⁺ − X⁻)/2

    def test_midline_at_center_is_outside_mid(self, corridor: TreynorCorridor) -> None:
        assert corridor.midline(D(-150)) == D("0.0400")

    def test_tilt_at_balanced_position(self, corridor: TreynorCorridor) -> None:
        # r(0) = 4.00% + (0.20%/900) × 150 = 4.0333% — the +3.33bp tilt
        assert close(corridor.midline(D(0)), D("0.040333"), "0.000002")

    def test_posted_on_rates_at_balanced(self, corridor: TreynorCorridor) -> None:
        # Ch2: lender receives 4.021%, borrower pays 4.046%
        assert close(corridor.bid(D(0)), D("0.04021"), "0.00001")
        assert close(corridor.ask(D(0)), D("0.04046"), "0.00001")

    def test_rate_midline_slopes_upward(self, corridor: TreynorCorridor) -> None:
        assert corridor.midline(D(100)) > corridor.midline(D(0)) > corridor.midline(D(-100))

    def test_ghost_anchors(self, corridor: TreynorCorridor) -> None:
        # Wholesale ON band [3.90%, 4.10%] at the ghost points.
        assert corridor.ghost_value(upper=True) == D("0.0410")
        assert corridor.ghost_value(upper=False) == D("0.0390")


class TestSplitTolls:
    """Ch3's I₂ = λ₂(C₊ + C₋), with I = λO as the special case."""

    def test_symmetric_tolls_reduce_to_lambda_o(self) -> None:
        corridor = TreynorCorridor.one_sided_price(ticket=D(1), capacity=D(9), mid=D(100), outside_spread=D("0.40"))
        assert corridor.inside_spread == corridor.layoff_probability * D("0.40")

    def test_asymmetric_tolls(self) -> None:
        corridor = TreynorCorridor(
            ticket=D(50),
            lower_bound=D(-200),
            upper_bound=D(300),
            anchor_mid=D("0.05"),
            anchor_width=D("0.01"),
            upper_toll=D("0.008"),  # C₊: outside funding cost
            lower_toll=D("0.002"),  # C₋: deployment carry
            orientation=Orientation.RATE,
        )
        lam = D(50) / D(550)
        assert corridor.layoff_probability == lam
        assert corridor.inside_spread == lam * D("0.010")


class TestValidation:
    def test_rejects_non_grid_width(self) -> None:
        with pytest.raises(ValueError, match="not a multiple"):
            TreynorCorridor.one_sided_price(ticket=D(3), capacity=D(10), mid=D(100), outside_spread=D("0.5"))

    def test_rejects_inventory_off_corridor(self) -> None:
        corridor = TreynorCorridor.one_sided_price(ticket=D(1), capacity=D(5), mid=D(100), outside_spread=D("0.5"))
        with pytest.raises(ValueError, match="outside corridor"):
            corridor.midline(D(6))

    def test_degenerate_single_rung(self) -> None:
        corridor = TreynorCorridor.one_sided_price(ticket=D(1), capacity=D(0), mid=D(100), outside_spread=D("0.5"))
        assert corridor.n_rungs == 1
        assert corridor.layoff_probability == D(1)
        assert corridor.inside_spread == D("0.5")  # λ=1 ⇒ I = O


class TestZeroProfitIdentity:
    """The economic punchline, verified empirically under the 50/50 model."""

    @pytest.mark.parametrize(
        "corridor",
        [
            TreynorCorridor.one_sided_price(ticket=D(1), capacity=D(10), mid=D(100), outside_spread=D("0.50")),
            TreynorCorridor.two_sided_rate(
                ticket=D(100),
                upper_capacity=D(200),
                lower_capacity=D(500),
                mid_rate=D("0.0400"),
                outside_spread=D("0.0020"),
            ),
        ],
        ids=["ch1_spot", "ch2_speculative"],
    )
    def test_walk_converges_to_zero_profit(self, corridor: TreynorCorridor) -> None:
        stats = simulate_walk(corridor, 400_000, random.Random(7))

        # Layoff frequency → λ
        lam = corridor.layoff_probability
        assert abs(stats.realized_layoff_frequency - lam) < lam * D("0.05")

        # Occupancy → uniform: every rung visited within 15% of 1/N
        expected = stats.trades / corridor.n_rungs
        assert len(stats.occupancy) == corridor.n_rungs
        for count in stats.occupancy.values():
            assert abs(count - expected) < expected * 0.15

        # Net P&L per trade → 0 (relative to the per-trade half-spread)
        half_spread = corridor.inside_spread / 2
        assert abs(stats.pnl_per_trade) < half_spread * D("0.05")

    def test_one_sided_flow_breaks_the_dealer(self) -> None:
        """Crisis regime: 80% sellers — losses, not break-even (Ch1 Part 3)."""
        corridor = TreynorCorridor.one_sided_price(ticket=D(1), capacity=D(10), mid=D(100), outside_spread=D("0.50"))
        from bilancio_v2.treynor.kernel import WalkStats

        rng = random.Random(7)
        stats = WalkStats()
        x = corridor.center
        for _ in range(100_000):
            x = stats.record(corridor, x, toward_upper=rng.random() < 0.8)

        # Flow diagnostic detects the departure from the 50/50 prior...
        assert stats.realized_flow_ratio > D("0.75")
        # ...and the zero-profit arithmetic fails: sustained losses.
        assert stats.pnl_per_trade < -corridor.inside_spread
