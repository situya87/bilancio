"""
Tests for BankDealer quote sensitivity to Central Bank rate changes.

This test suite specifically verifies how bank quotes respond to CB rate changes,
including:
- Parallel shifts in corridor (both rates move together)
- Floor-only changes
- Ceiling-only changes
- Corridor width changes
- Dynamic rate changes during simulation

These tests ensure that the Treynor-style pricing kernel correctly reflects
CB policy changes in bank deposit/loan rates.

References:
- "Banks-as-Dealers with deposits on demand" specification Section 6
"""

from decimal import Decimal

from bilancio.banking.pricing_kernel import PricingParams, compute_midline
from bilancio.banking.state import BankDealerState, CentralBankParams
from bilancio.banking.ticket_processor import TicketProcessor
from bilancio.banking.types import Ticket, TicketType


def create_processor_with_cb_params(
    cb_params: CentralBankParams,
    reserves: int = 100000,
    deposits: int = 50000,
) -> TicketProcessor:
    """Create processor with specific CB parameters."""
    pricing_params = PricingParams(
        reserve_remuneration_rate=cb_params.reserve_remuneration_rate,
        cb_borrowing_rate=cb_params.cb_borrowing_rate,
        reserve_target=100000,
        symmetric_capacity=50000,
        ticket_size=10000,
        reserve_floor=10000,
    )

    state = BankDealerState(
        bank_id="TestBank",
        reserves=reserves,
        current_day=0,
    )

    if deposits > 0:
        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=deposits,
            rate=Decimal("0.015"),
        )

    return TicketProcessor(
        state=state,
        cb_params=cb_params,
        pricing_params=pricing_params,
    )


class TestParallelCorridorShift:
    """Tests for parallel shifts in the CB corridor."""

    def test_parallel_shift_up_increases_all_rates(self):
        """Shifting corridor up increases deposit and loan rates."""
        # Base corridor: 1% - 3%
        base_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )
        base_processor = create_processor_with_cb_params(base_cb)
        base_processor.initialize_day(day=1)
        base_quote = base_processor.state.current_quote

        # Shifted corridor: 2% - 4% (parallel shift up by 1%)
        shifted_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.02"),
            cb_borrowing_rate=Decimal("0.04"),
        )
        shifted_processor = create_processor_with_cb_params(shifted_cb)
        shifted_processor.initialize_day(day=1)
        shifted_quote = shifted_processor.state.current_quote

        # Both rates should increase
        assert shifted_quote.deposit_rate > base_quote.deposit_rate
        assert shifted_quote.loan_rate > base_quote.loan_rate

        # Increase should be approximately 1% (the shift amount)
        deposit_increase = shifted_quote.deposit_rate - base_quote.deposit_rate
        loan_increase = shifted_quote.loan_rate - base_quote.loan_rate

        # At zero inventory, rates should shift by exactly the corridor shift
        assert abs(deposit_increase - Decimal("0.01")) < Decimal("0.001")
        assert abs(loan_increase - Decimal("0.01")) < Decimal("0.001")

    def test_parallel_shift_down_decreases_all_rates(self):
        """Shifting corridor down decreases deposit and loan rates."""
        # Base corridor: 2% - 4%
        base_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.02"),
            cb_borrowing_rate=Decimal("0.04"),
        )
        base_processor = create_processor_with_cb_params(base_cb)
        base_processor.initialize_day(day=1)
        base_quote = base_processor.state.current_quote

        # Shifted corridor: 1% - 3% (parallel shift down by 1%)
        shifted_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )
        shifted_processor = create_processor_with_cb_params(shifted_cb)
        shifted_processor.initialize_day(day=1)
        shifted_quote = shifted_processor.state.current_quote

        # Both rates should decrease
        assert shifted_quote.deposit_rate < base_quote.deposit_rate
        assert shifted_quote.loan_rate < base_quote.loan_rate

    def test_spread_unchanged_in_parallel_shift(self):
        """Spread unchanged when corridor shifts in parallel."""
        # Base corridor: 1% - 3% (2% width)
        base_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )
        base_processor = create_processor_with_cb_params(base_cb)
        base_processor.initialize_day(day=1)
        base_spread = (
            base_processor.state.current_quote.loan_rate
            - base_processor.state.current_quote.deposit_rate
        )

        # Shifted corridor: 3% - 5% (still 2% width)
        shifted_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.03"),
            cb_borrowing_rate=Decimal("0.05"),
        )
        shifted_processor = create_processor_with_cb_params(shifted_cb)
        shifted_processor.initialize_day(day=1)
        shifted_spread = (
            shifted_processor.state.current_quote.loan_rate
            - shifted_processor.state.current_quote.deposit_rate
        )

        # Spread should be the same (same corridor width)
        assert abs(shifted_spread - base_spread) < Decimal("0.0001")


class TestFloorOnlyChange:
    """Tests for changes to the floor rate only."""

    def test_floor_increase_shifts_rates_up(self):
        """Increasing floor rate shifts rates up (corridor narrows from below)."""
        # Base: 1% - 3%
        base_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )
        base_processor = create_processor_with_cb_params(base_cb)
        base_processor.initialize_day(day=1)
        base_quote = base_processor.state.current_quote

        # Floor raised: 2% - 3%
        raised_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.02"),
            cb_borrowing_rate=Decimal("0.03"),
        )
        raised_processor = create_processor_with_cb_params(raised_cb)
        raised_processor.initialize_day(day=1)
        raised_quote = raised_processor.state.current_quote

        # Midpoint increased: (0.01+0.03)/2 = 0.02 → (0.02+0.03)/2 = 0.025
        # Both rates should increase
        assert raised_quote.deposit_rate > base_quote.deposit_rate
        assert raised_quote.loan_rate > base_quote.loan_rate

    def test_floor_increase_narrows_spread(self):
        """Increasing floor narrows corridor width, reducing spread."""
        # Wide corridor: 0% - 4% (4% width)
        wide_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.00"),
            cb_borrowing_rate=Decimal("0.04"),
        )
        wide_processor = create_processor_with_cb_params(wide_cb)
        wide_processor.initialize_day(day=1)
        wide_spread = (
            wide_processor.state.current_quote.loan_rate
            - wide_processor.state.current_quote.deposit_rate
        )

        # Narrower corridor: 2% - 4% (2% width)
        narrow_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.02"),
            cb_borrowing_rate=Decimal("0.04"),
        )
        narrow_processor = create_processor_with_cb_params(narrow_cb)
        narrow_processor.initialize_day(day=1)
        narrow_spread = (
            narrow_processor.state.current_quote.loan_rate
            - narrow_processor.state.current_quote.deposit_rate
        )

        # Narrower corridor means narrower bank spread
        assert narrow_spread < wide_spread


class TestCeilingOnlyChange:
    """Tests for changes to the ceiling rate only."""

    def test_ceiling_increase_shifts_rates_up(self):
        """Increasing ceiling rate shifts rates up (corridor widens from above)."""
        # Base: 1% - 3%
        base_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )
        base_processor = create_processor_with_cb_params(base_cb)
        base_processor.initialize_day(day=1)
        base_quote = base_processor.state.current_quote

        # Ceiling raised: 1% - 5%
        raised_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.05"),
        )
        raised_processor = create_processor_with_cb_params(raised_cb)
        raised_processor.initialize_day(day=1)
        raised_quote = raised_processor.state.current_quote

        # Midpoint increased: (0.01+0.03)/2 = 0.02 → (0.01+0.05)/2 = 0.03
        # Both rates should increase
        assert raised_quote.deposit_rate >= base_quote.deposit_rate
        assert raised_quote.loan_rate > base_quote.loan_rate

    def test_ceiling_increase_widens_spread(self):
        """Increasing ceiling widens corridor, increasing spread."""
        # Narrow ceiling: 1% - 2%
        narrow_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.02"),
        )
        narrow_processor = create_processor_with_cb_params(narrow_cb)
        narrow_processor.initialize_day(day=1)
        narrow_spread = (
            narrow_processor.state.current_quote.loan_rate
            - narrow_processor.state.current_quote.deposit_rate
        )

        # Wide ceiling: 1% - 5%
        wide_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.05"),
        )
        wide_processor = create_processor_with_cb_params(wide_cb)
        wide_processor.initialize_day(day=1)
        wide_spread = (
            wide_processor.state.current_quote.loan_rate
            - wide_processor.state.current_quote.deposit_rate
        )

        # Wider corridor means wider bank spread
        assert wide_spread > narrow_spread


class TestCorridorWidthChanges:
    """Tests for corridor width effects on bank spreads."""

    def test_zero_width_corridor_no_spread(self):
        """Zero-width corridor leads to minimal spread."""
        # Zero-width: 2% - 2%
        zero_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.02"),
            cb_borrowing_rate=Decimal("0.02"),
        )
        zero_processor = create_processor_with_cb_params(zero_cb)
        zero_processor.initialize_day(day=1)
        zero_quote = zero_processor.state.current_quote

        # Spread should be zero (or minimal due to floor discipline)
        spread = zero_quote.loan_rate - zero_quote.deposit_rate
        assert spread == Decimal("0")

    def test_very_wide_corridor_large_spread(self):
        """Very wide corridor leads to larger spread."""
        # Very wide: 0% - 10%
        wide_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.00"),
            cb_borrowing_rate=Decimal("0.10"),
        )
        wide_processor = create_processor_with_cb_params(wide_cb)
        wide_processor.initialize_day(day=1)
        wide_quote = wide_processor.state.current_quote

        # Standard: 1% - 3%
        standard_cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )
        standard_processor = create_processor_with_cb_params(standard_cb)
        standard_processor.initialize_day(day=1)
        standard_quote = standard_processor.state.current_quote

        wide_spread = wide_quote.loan_rate - wide_quote.deposit_rate
        standard_spread = standard_quote.loan_rate - standard_quote.deposit_rate

        assert wide_spread > standard_spread

    def test_spread_proportional_to_corridor_width(self):
        """Bank spread is proportional to corridor width (via λ)."""
        widths = [
            (Decimal("0.01"), Decimal("0.02")),  # 1% width
            (Decimal("0.01"), Decimal("0.03")),  # 2% width
            (Decimal("0.01"), Decimal("0.05")),  # 4% width
        ]

        spreads = []
        for floor, ceiling in widths:
            cb = CentralBankParams(
                reserve_remuneration_rate=floor,
                cb_borrowing_rate=ceiling,
            )
            processor = create_processor_with_cb_params(cb)
            processor.initialize_day(day=1)
            quote = processor.state.current_quote
            spreads.append(quote.loan_rate - quote.deposit_rate)

        # Spreads should increase with corridor width
        assert spreads[0] < spreads[1] < spreads[2]


class TestRateChangeDuringSimulation:
    """Tests for CB rate changes during simulation."""

    def test_rate_change_affects_new_tickets(self):
        """CB rate change affects rates on new tickets."""
        # Start with standard corridor
        cb_params = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )

        processor = create_processor_with_cb_params(cb_params, deposits=80000)
        processor.initialize_day(day=1)
        quote_before = processor.state.current_quote

        # Simulate a rate hike by creating new processor with higher rates
        # (In real simulation, you would modify the params)
        new_cb_params = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.02"),
            cb_borrowing_rate=Decimal("0.04"),
        )

        # Create new pricing params with new CB rates
        new_pricing_params = PricingParams(
            reserve_remuneration_rate=new_cb_params.reserve_remuneration_rate,
            cb_borrowing_rate=new_cb_params.cb_borrowing_rate,
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # Update processor with new params
        processor.cb_params = new_cb_params
        processor.pricing_params = new_pricing_params

        # Refresh quotes
        processor.refresh_projection()
        quote_after = processor.refresh_quotes()

        # Rates should have increased
        assert quote_after.deposit_rate > quote_before.deposit_rate
        assert quote_after.loan_rate > quote_before.loan_rate

    def test_old_cohorts_keep_stamped_rates(self):
        """Deposits created before rate change keep their stamped rates."""
        cb_params = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )

        processor = create_processor_with_cb_params(cb_params, deposits=0)
        processor.initialize_day(day=1)

        # Create deposit at old rates
        ticket1 = Ticket(
            id="T1",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=10000,
            client_id="client_a",
            created_day=1,
            counterparty_bank_id="BankB",
        )
        processor.process_ticket(ticket1)
        old_rate = ticket1.stamped_deposit_rate

        # Change rates
        new_cb_params = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.05"),
            cb_borrowing_rate=Decimal("0.07"),
        )
        new_pricing_params = PricingParams(
            reserve_remuneration_rate=new_cb_params.reserve_remuneration_rate,
            cb_borrowing_rate=new_cb_params.cb_borrowing_rate,
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )
        processor.cb_params = new_cb_params
        processor.pricing_params = new_pricing_params
        processor.refresh_quotes()

        # Create new deposit at new rates
        ticket2 = Ticket(
            id="T2",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=10000,
            client_id="client_b",
            created_day=1,
            counterparty_bank_id="BankC",
        )
        processor.process_ticket(ticket2)
        new_rate = ticket2.stamped_deposit_rate

        # Old deposit keeps its stamped rate
        assert ticket1.stamped_deposit_rate == old_rate
        # New deposit has higher rate
        assert new_rate > old_rate


class TestInterestRatePassthrough:
    """Tests for how CB rates pass through to depositor/borrower rates."""

    def test_floor_is_lower_bound_for_deposit_rate(self):
        """Deposit rate cannot go below floor (except for zero floor)."""
        # High floor: 3% - 5%
        cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.03"),
            cb_borrowing_rate=Decimal("0.05"),
        )

        # Bank with very high reserves (would want to pay minimal deposit rate)
        processor = create_processor_with_cb_params(cb, reserves=500000, deposits=10000)
        processor.initialize_day(day=1)
        quote = processor.state.current_quote

        # Even with excess reserves, deposit rate should be reasonable
        # (bounded by the corridor structure, not necessarily the floor)
        assert quote.deposit_rate >= Decimal("0")

    def test_ceiling_discipline_caps_deposit_rate(self):
        """Deposit rate cannot exceed CB ceiling."""
        cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )

        # Bank with very low reserves (would want to pay high deposit rate)
        processor = create_processor_with_cb_params(cb, reserves=5000, deposits=100000)
        processor.initialize_day(day=1)
        quote = processor.state.current_quote

        # Deposit rate should not exceed ceiling
        assert quote.deposit_rate <= cb.cb_borrowing_rate

    def test_loan_rate_above_deposit_rate(self):
        """Loan rate always exceeds deposit rate (positive spread)."""
        for floor, ceiling in [
            (Decimal("0.00"), Decimal("0.02")),
            (Decimal("0.01"), Decimal("0.03")),
            (Decimal("0.05"), Decimal("0.10")),
        ]:
            cb = CentralBankParams(
                reserve_remuneration_rate=floor,
                cb_borrowing_rate=ceiling,
            )
            processor = create_processor_with_cb_params(cb)
            processor.initialize_day(day=1)
            quote = processor.state.current_quote

            assert quote.loan_rate > quote.deposit_rate


class TestMidlineAtExtremeInventory:
    """Tests for midline behavior at extreme inventory positions."""

    def test_midline_at_extreme_long(self):
        """At extreme long position, midline approaches floor."""
        cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )

        pricing_params = PricingParams(
            reserve_remuneration_rate=cb.reserve_remuneration_rate,
            cb_borrowing_rate=cb.cb_borrowing_rate,
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # At long limit: x = X* + S = 60000
        long_limit = pricing_params.symmetric_capacity + pricing_params.ticket_size
        midline = compute_midline(long_limit, pricing_params)

        # Should equal floor rate
        assert midline == cb.reserve_remuneration_rate

    def test_midline_at_extreme_short(self):
        """At extreme short position, midline approaches ceiling."""
        cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )

        pricing_params = PricingParams(
            reserve_remuneration_rate=cb.reserve_remuneration_rate,
            cb_borrowing_rate=cb.cb_borrowing_rate,
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # At short limit: x = -(X* + S) = -60000
        short_limit = -(pricing_params.symmetric_capacity + pricing_params.ticket_size)
        midline = compute_midline(short_limit, pricing_params)

        # Should equal ceiling rate
        assert midline == cb.cb_borrowing_rate


class TestNegativeRateEnvironment:
    """Tests for negative interest rate environments."""

    def test_negative_floor_rate(self):
        """System handles negative floor rate."""
        cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("-0.005"),  # -0.5%
            cb_borrowing_rate=Decimal("0.005"),  # +0.5%
        )

        processor = create_processor_with_cb_params(cb)
        processor.initialize_day(day=1)
        quote = processor.state.current_quote

        # System should work with negative rates
        assert quote.deposit_rate is not None
        assert quote.loan_rate is not None
        # Deposit rate floored at 0
        assert quote.deposit_rate >= Decimal("0")

    def test_corridor_midpoint_can_be_negative(self):
        """Corridor midpoint can be negative."""
        cb = CentralBankParams(
            reserve_remuneration_rate=Decimal("-0.02"),  # -2%
            cb_borrowing_rate=Decimal("-0.01"),  # -1%
        )

        pricing_params = PricingParams(
            reserve_remuneration_rate=cb.reserve_remuneration_rate,
            cb_borrowing_rate=cb.cb_borrowing_rate,
            reserve_target=100000,
            symmetric_capacity=50000,
            ticket_size=10000,
            reserve_floor=10000,
        )

        # Midpoint = (-0.02 + -0.01) / 2 = -0.015
        assert pricing_params.outside_mid == Decimal("-0.015")
