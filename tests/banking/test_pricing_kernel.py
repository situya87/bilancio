"""
Unit tests for the banking pricing kernel (Treynor-style dealer pricing).

Tests verify:
- PricingParams derived quantities (outside_width, layoff_probability, inside_width)
- Midline computation (symmetric linear midline on 2-day funding plane)
- Tilted midline with risk loadings
- Quote computation with ceiling discipline
- CB rate sensitivity

References:
- "Banks-as-Dealers with deposits on demand" specification Section 6
"""

from decimal import Decimal

from bilancio.banking.pricing_kernel import (
    PricingParams,
    compute_inventory,
    compute_midline,
    compute_quotes,
    compute_tilted_midline,
    simple_risk_index,
)


class TestPricingParamsDerivedQuantities:
    """Tests for PricingParams computed properties."""

    def test_outside_width(self):
        """Ω^(2) = ceiling - floor."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        assert params.outside_width == Decimal("0.02")

    def test_outside_mid(self):
        """M^(2) = (floor + ceiling) / 2."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        assert params.outside_mid == Decimal("0.02")

    def test_layoff_probability(self):
        """λ = S / (2X* + S)."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # λ = 10000 / (2 × 50000 + 10000) = 10000 / 110000 ≈ 0.0909
        expected = Decimal("10000") / Decimal("110000")
        assert params.layoff_probability == expected

    def test_inside_width(self):
        """I^(2) = λ × Ω^(2)."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # I = λ × Ω = (10000/110000) × 0.02
        expected = params.layoff_probability * params.outside_width
        assert params.inside_width == expected

    def test_inside_width_smaller_than_outside(self):
        """Inside width is always smaller than outside width."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        assert params.inside_width < params.outside_width


class TestMidlineComputation:
    """Tests for symmetric linear midline."""

    def test_midline_at_zero_inventory(self):
        """m(0) = M^(2) (midpoint at balanced inventory)."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        midline = compute_midline(inventory=0, params=params)

        # At x=0, midline equals corridor midpoint
        assert midline == params.outside_mid

    def test_midline_decreases_with_inventory(self):
        """Midline decreases as inventory increases (more cash-rich)."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        midline_0 = compute_midline(inventory=0, params=params)
        midline_pos = compute_midline(inventory=10000, params=params)

        # Higher inventory means lower midline (willing to pay less for deposits)
        assert midline_pos < midline_0

    def test_midline_increases_when_short(self):
        """Midline increases when cash-short (negative inventory)."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        midline_0 = compute_midline(inventory=0, params=params)
        midline_neg = compute_midline(inventory=-10000, params=params)

        # Negative inventory means higher midline (willing to pay more for deposits)
        assert midline_neg > midline_0

    def test_midline_approaches_floor_at_long_limit(self):
        """m(X* + S) approaches floor rate."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # At long limit x = X* + S = 60000
        long_limit = params.symmetric_capacity + params.ticket_size
        midline = compute_midline(inventory=long_limit, params=params)

        # Should be close to floor rate
        assert midline == params.reserve_remuneration_rate

    def test_midline_approaches_ceiling_at_short_limit(self):
        """m(-X* - S) approaches ceiling rate."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # At short limit x = -(X* + S) = -60000
        short_limit = -(params.symmetric_capacity + params.ticket_size)
        midline = compute_midline(inventory=short_limit, params=params)

        # Should be close to ceiling rate
        assert midline == params.cb_borrowing_rate


class TestTiltedMidline:
    """Tests for risk-tilted midline."""

    def test_no_tilt_at_zero_risk(self):
        """Tilted midline equals base midline when risk is zero."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        base = compute_midline(inventory=0, params=params)
        tilted = compute_tilted_midline(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=params,
        )

        assert tilted == base

    def test_positive_tilt_with_cash_tightness(self):
        """Cash tightness pushes midline up."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
            alpha=Decimal("0.005"),  # 0.5% per unit of L*
        )

        base = compute_midline(inventory=0, params=params)
        tilted = compute_tilted_midline(
            inventory=0,
            cash_tightness=Decimal("1.0"),  # High cash tightness
            risk_index=Decimal("0"),
            params=params,
        )

        # Tilted should be higher by alpha × L* = 0.005 × 1.0 = 0.005
        expected = base + Decimal("0.005")
        assert tilted == expected

    def test_positive_tilt_with_risk_index(self):
        """Risk index pushes midline up."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
            gamma=Decimal("0.002"),  # 0.2% per unit of ρ
        )

        base = compute_midline(inventory=0, params=params)
        tilted = compute_tilted_midline(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("1.0"),  # High risk index
            params=params,
        )

        # Tilted should be higher by gamma × ρ = 0.002 × 1.0 = 0.002
        expected = base + Decimal("0.002")
        assert tilted == expected


class TestQuoteComputation:
    """Tests for deposit/loan rate quote computation."""

    def test_symmetric_quotes_around_midline(self):
        """Bid and ask are symmetric around tilted midline."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=params,
            day=1,
        )

        # At zero inventory and zero risk, midline = M^(2) = 0.02
        midline = quote.midline
        half_width = params.inside_width / 2

        # Deposit rate (bid) = midline - half_width
        # Loan rate (ask) = midline + half_width
        expected_deposit = midline - half_width
        expected_loan = midline + half_width

        assert quote.deposit_rate == max(Decimal("0"), expected_deposit)
        assert quote.loan_rate == expected_loan

    def test_spread_equals_inside_width(self):
        """Loan rate - deposit rate = inside width."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=params,
            day=1,
        )

        spread = quote.loan_rate - quote.deposit_rate
        # Allow small tolerance for Decimal precision
        assert abs(spread - params.inside_width) < Decimal("0.0000001")

    def test_ceiling_discipline_on_deposits(self):
        """Deposit rate capped at CB ceiling."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # Very negative inventory (short cash) would push midline very high
        quote = compute_quotes(
            inventory=-100000,  # Extremely short
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=params,
            day=1,
        )

        # Deposit rate should be capped at ceiling
        assert quote.deposit_rate <= params.cb_borrowing_rate

    def test_floor_discipline_on_deposits(self):
        """Deposit rate cannot go negative."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # Very positive inventory (long cash) would push midline very low
        quote = compute_quotes(
            inventory=100000,  # Extremely long
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=params,
            day=1,
        )

        # Deposit rate should not be negative
        assert quote.deposit_rate >= Decimal("0")

    def test_quote_includes_diagnostics(self):
        """Quote includes diagnostic information."""
        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        quote = compute_quotes(
            inventory=5000,
            cash_tightness=Decimal("0.3"),
            risk_index=Decimal("0.2"),
            params=params,
            day=5,
            ticket_number=3,
        )

        assert quote.inventory == 5000
        assert quote.cash_tightness == Decimal("0.3")
        assert quote.risk_index == Decimal("0.2")
        assert quote.day == 5
        assert quote.ticket_number == 3


class TestInventoryComputation:
    """Tests for inventory coordinate computation."""

    def test_inventory_at_target(self):
        """Inventory = 0 when projected reserves equal target."""
        inventory = compute_inventory(
            projected_reserves_t2=100000,
            reserve_target=100000,
        )

        assert inventory == 0

    def test_inventory_positive_when_above_target(self):
        """Positive inventory when above target (cash-rich)."""
        inventory = compute_inventory(
            projected_reserves_t2=120000,
            reserve_target=100000,
        )

        assert inventory == 20000

    def test_inventory_negative_when_below_target(self):
        """Negative inventory when below target (cash-short)."""
        inventory = compute_inventory(
            projected_reserves_t2=80000,
            reserve_target=100000,
        )

        assert inventory == -20000


class TestSimpleRiskIndex:
    """Tests for simple risk index computation."""

    def test_risk_index_from_cash_tightness(self):
        """Risk index based on cash tightness."""
        risk = simple_risk_index(
            cash_tightness=Decimal("0.5"),
        )

        # With default weights, ρ = cash_tightness
        assert risk == Decimal("0.5")

    def test_risk_index_with_custom_weights(self):
        """Risk index uses custom weights."""
        risk = simple_risk_index(
            cash_tightness=Decimal("0.5"),
            weight_reserve=Decimal("2.0"),
        )

        # ρ = 2.0 × 0.5 = 1.0
        assert risk == Decimal("1.0")

    def test_risk_index_with_loan_gap(self):
        """Risk index includes loan gap component."""
        risk = simple_risk_index(
            cash_tightness=Decimal("0.5"),
            loan_gap_ratio=Decimal("0.3"),
            weight_reserve=Decimal("1.0"),
            weight_loan=Decimal("0.5"),
        )

        # ρ = 1.0 × 0.5 + 0.5 × |0.3| = 0.5 + 0.15 = 0.65
        assert risk == Decimal("0.65")


class TestCBRateSensitivity:
    """Tests for how quotes respond to CB rate changes."""

    def test_quotes_shift_with_floor_increase(self):
        """Higher floor rate increases both deposit and loan rates."""
        base_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        raised_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.02"),  # Floor raised by 1%
            cb_borrowing_rate=Decimal("0.04"),  # Ceiling raised by 1%
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        base_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=base_params,
            day=1,
        )

        raised_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=raised_params,
            day=1,
        )

        # Both rates should increase by 1%
        assert raised_quote.deposit_rate > base_quote.deposit_rate
        assert raised_quote.loan_rate > base_quote.loan_rate

        # Midline should increase by 1%
        assert raised_quote.midline - base_quote.midline == Decimal("0.01")

    def test_wider_corridor_increases_spread(self):
        """Wider CB corridor increases bank's spread."""
        narrow_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.02"),  # Narrow: 1% corridor
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        wide_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.05"),  # Wide: 4% corridor
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        narrow_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=narrow_params,
            day=1,
        )

        wide_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=wide_params,
            day=1,
        )

        narrow_spread = narrow_quote.loan_rate - narrow_quote.deposit_rate
        wide_spread = wide_quote.loan_rate - wide_quote.deposit_rate

        # Wider corridor means wider bank spread
        assert wide_spread > narrow_spread

    def test_corridor_midpoint_shift(self):
        """Shifting corridor midpoint shifts quotes proportionally."""
        low_mid_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.00"),
            cb_borrowing_rate=Decimal("0.02"),  # Mid = 0.01
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        high_mid_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.04"),
            cb_borrowing_rate=Decimal("0.06"),  # Mid = 0.05
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        low_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=low_mid_params,
            day=1,
        )

        high_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=high_mid_params,
            day=1,
        )

        # Midline difference should equal corridor midpoint difference
        midline_diff = high_quote.midline - low_quote.midline
        corridor_mid_diff = high_mid_params.outside_mid - low_mid_params.outside_mid

        assert midline_diff == corridor_mid_diff


class TestCapacityEffects:
    """Tests for how capacity affects pricing."""

    def test_larger_capacity_narrows_spread(self):
        """Larger symmetric capacity reduces spread (lower λ)."""
        small_cap_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=10000,  # Small capacity
            ticket_size=10000,
            reserve_floor=10000,
        )

        large_cap_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=100000,  # Large capacity
            ticket_size=10000,
            reserve_floor=10000,
        )

        small_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=small_cap_params,
            day=1,
        )

        large_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=large_cap_params,
            day=1,
        )

        small_spread = small_quote.loan_rate - small_quote.deposit_rate
        large_spread = large_quote.loan_rate - large_quote.deposit_rate

        # Larger capacity means smaller spread
        assert large_spread < small_spread

    def test_larger_ticket_size_widens_spread(self):
        """Larger ticket size increases spread (higher λ)."""
        small_ticket_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=1000,  # Small tickets
            reserve_floor=10000,
        )

        large_ticket_params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=50000,  # Large tickets
            reserve_floor=10000,
        )

        small_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=small_ticket_params,
            day=1,
        )

        large_quote = compute_quotes(
            inventory=0,
            cash_tightness=Decimal("0"),
            risk_index=Decimal("0"),
            params=large_ticket_params,
            day=1,
        )

        small_spread = small_quote.loan_rate - small_quote.deposit_rate
        large_spread = large_quote.loan_rate - large_quote.deposit_rate

        # Larger tickets mean wider spread
        assert large_spread > small_spread
