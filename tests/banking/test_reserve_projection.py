"""
Unit tests for reserve projection and cash-tightness computation.

Tests verify:
- 10-day reserve path projection
- Cash-tightness (L*) calculation
- Long-side headroom computation
- Projection updates after tickets

References:
- "Banks-as-Dealers with deposits on demand" specification Section 6.1.3
"""

from decimal import Decimal

from bilancio.banking.reserve_projection import (
    ReserveProjection,
    compute_cash_tightness,
    compute_long_side_headroom,
    effective_reserves_after_withdrawals,
    project_reserves,
    update_projection_after_ticket,
)
from bilancio.banking.state import BankDealerState, CentralBankParams


def create_test_state(
    reserves: int = 100000,
    withdrawal_forecast: int = 0,
    day: int = 0,
) -> BankDealerState:
    """Create a test BankDealerState."""
    state = BankDealerState(
        bank_id="TestBank",
        reserves=reserves,
        current_day=day,
    )
    state.withdrawal_forecast = withdrawal_forecast
    return state


class TestReserveProjection:
    """Tests for 10-day reserve path projection."""

    def test_projection_length(self):
        """Projection covers 11 days (t through t+10)."""
        state = create_test_state(reserves=100000)
        cb_params = CentralBankParams()

        projection = project_reserves(state, cb_params, horizon=10)

        assert len(projection.path) == 11  # Day 0 through day 10
        assert projection.base_day == 0

    def test_projection_starting_point(self):
        """R̂(t) = R_t - W^rem_t."""
        state = create_test_state(
            reserves=100000,
            withdrawal_forecast=20000,
        )
        cb_params = CentralBankParams()

        projection = project_reserves(state, cb_params)

        # Starting point = 100000 - 20000 = 80000
        assert projection.path[0] == 80000

    def test_projection_includes_loan_repayment(self):
        """Projection includes scheduled loan repayments."""
        state = create_test_state(reserves=100000, day=0)
        cb_params = CentralBankParams()

        # Add loan that matures at day 10
        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        projection = project_reserves(state, cb_params)

        # Day 10 should include loan repayment (51500)
        day_9_reserves = projection.path[9]
        day_10_reserves = projection.path[10]

        # Loan repayment = (1 + 0.03) × 50000 = 51500
        assert day_10_reserves - day_9_reserves == 51500

    def test_projection_includes_cb_repayment(self):
        """Projection includes scheduled CB repayments (outflows)."""
        state = create_test_state(reserves=100000, day=0)
        cb_params = CentralBankParams()

        # Add CB borrowing that matures at day 2
        state.add_cb_borrowing(day=0, amount=30000, cb_rate=Decimal("0.03"))

        projection = project_reserves(state, cb_params)

        # Day 2 should include CB repayment (30900 outflow)
        day_1_reserves = projection.path[1]
        day_2_reserves = projection.path[2]

        # CB repayment = (1 + 0.03) × 30000 = 30900 (outflow)
        expected_outflow = -30900
        # Also includes CB remuneration at day 2
        cb_rem = int(Decimal("0.01") * 100000)  # 1% of initial reserves
        expected_delta = expected_outflow + cb_rem

        assert day_2_reserves - day_1_reserves == expected_delta

    def test_projection_min_reserves(self):
        """Projection tracks minimum reserves over horizon."""
        state = create_test_state(reserves=100000, day=0)
        cb_params = CentralBankParams()

        # Add CB borrowing creating large outflow at day 2
        state.add_cb_borrowing(day=0, amount=80000, cb_rate=Decimal("0.03"))

        projection = project_reserves(state, cb_params)

        # Minimum should occur around day 2 after CB repayment
        assert projection.min_reserves <= 100000

    def test_projection_at_horizon(self):
        """Can query projection at specific horizon."""
        state = create_test_state(reserves=100000, day=0)
        cb_params = CentralBankParams()

        projection = project_reserves(state, cb_params)

        # Can query any day in range
        assert projection.at_horizon(0) == projection.path[0]
        assert projection.at_horizon(5) == projection.path[5]
        assert projection.at_horizon(10) == projection.path[10]


class TestCashTightness:
    """Tests for cash-tightness (L*) computation."""

    def test_zero_tightness_above_floor(self):
        """L* = 0 when min reserves >= floor."""
        projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 5: 80000, 10: 90000},
            legs_by_day={},
        )

        # Floor = 10000, min = 80000 > floor
        tightness = compute_cash_tightness(projection, reserve_floor=10000)

        assert tightness == Decimal("0")

    def test_positive_tightness_below_floor(self):
        """L* > 0 when min reserves < floor."""
        projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 5: 5000, 10: 50000},
            legs_by_day={},
        )

        # Floor = 10000, min = 5000 < floor
        # Shortfall = 10000 - 5000 = 5000
        # L* = 5000 / 10000 = 0.5
        tightness = compute_cash_tightness(projection, reserve_floor=10000)

        assert tightness == Decimal("0.5")

    def test_tightness_normalized_by_floor(self):
        """L* is normalized by reserve floor."""
        projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 5: 0, 10: 50000},
            legs_by_day={},
        )

        # Floor = 20000, min = 0
        # Shortfall = 20000 - 0 = 20000
        # L* = 20000 / 20000 = 1.0
        tightness = compute_cash_tightness(projection, reserve_floor=20000)

        assert tightness == Decimal("1.0")

    def test_tightness_with_negative_reserves(self):
        """L* handles negative projected reserves."""
        projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 5: -10000, 10: 50000},
            legs_by_day={},
        )

        # Floor = 10000, min = -10000
        # Shortfall = 10000 - (-10000) = 20000
        # L* = 20000 / 10000 = 2.0
        tightness = compute_cash_tightness(projection, reserve_floor=10000)

        assert tightness == Decimal("2.0")


class TestLongSideHeadroom:
    """Tests for long-side headroom computation."""

    def test_positive_headroom(self):
        """Headroom > 0 when end reserves > floor."""
        projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 10: 80000},
            legs_by_day={},
        )

        # Floor = 20000, end = 80000
        # Headroom = 80000 - 20000 = 60000
        headroom = compute_long_side_headroom(projection, reserve_floor=20000)

        assert headroom == 60000

    def test_zero_headroom_at_floor(self):
        """Headroom = 0 when end reserves = floor."""
        projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 10: 20000},
            legs_by_day={},
        )

        headroom = compute_long_side_headroom(projection, reserve_floor=20000)

        assert headroom == 0

    def test_no_negative_headroom(self):
        """Headroom cannot be negative."""
        projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 10: 10000},
            legs_by_day={},
        )

        # Floor = 20000, end = 10000 < floor
        headroom = compute_long_side_headroom(projection, reserve_floor=20000)

        assert headroom == 0


class TestEffectiveReserves:
    """Tests for effective reserves computation."""

    def test_effective_reserves_basic(self):
        """R^eff = R - W^rem."""
        effective = effective_reserves_after_withdrawals(
            current_reserves=100000,
            remaining_withdrawals=20000,
        )

        assert effective == 80000

    def test_effective_reserves_can_be_negative(self):
        """Effective reserves can go negative."""
        effective = effective_reserves_after_withdrawals(
            current_reserves=10000,
            remaining_withdrawals=30000,
        )

        assert effective == -20000


class TestProjectionUpdate:
    """Tests for incremental projection updates."""

    def test_update_with_reserve_delta(self):
        """Update shifts entire path by reserve delta."""
        old_projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 5: 90000, 10: 80000},
            legs_by_day={},
        )

        # Reserve inflow of 10000
        new_projection = update_projection_after_ticket(
            old_projection,
            reserve_delta=10000,
            withdrawal_delta=0,
        )

        assert new_projection.path[0] == 110000
        assert new_projection.path[5] == 100000
        assert new_projection.path[10] == 90000

    def test_update_with_withdrawal_delta(self):
        """Realized withdrawals increase effective reserves."""
        old_projection = ReserveProjection(
            base_day=0,
            path={0: 80000, 5: 70000, 10: 60000},
            legs_by_day={},
        )

        # Withdrawal realized: withdrawal_delta = -5000 (forecast reduced)
        # This increases effective reserves by 5000
        new_projection = update_projection_after_ticket(
            old_projection,
            reserve_delta=0,
            withdrawal_delta=-5000,
        )

        assert new_projection.path[0] == 75000
        assert new_projection.path[5] == 65000
        assert new_projection.path[10] == 55000

    def test_update_with_new_scheduled_leg(self):
        """New scheduled leg affects path from that day onward."""
        old_projection = ReserveProjection(
            base_day=0,
            path={0: 100000, 5: 100000, 10: 100000},
            legs_by_day={},
        )

        # New loan creates repayment leg at day 10
        new_projection = update_projection_after_ticket(
            old_projection,
            reserve_delta=0,
            withdrawal_delta=0,
            new_scheduled_legs=[(10, 51500, "loan_repay:51500")],
        )

        # Only day 10 should be affected
        assert new_projection.path[0] == 100000
        assert new_projection.path[5] == 100000
        assert new_projection.path[10] == 151500


class TestProjectionDiagnostics:
    """Tests for projection diagnostic output."""

    def test_projection_str_format(self):
        """Projection has readable string format."""
        state = create_test_state(reserves=100000, day=0)
        cb_params = CentralBankParams()

        projection = project_reserves(state, cb_params)

        output = str(projection)

        assert "Reserve Projection" in output
        assert "Day 0" in output
        assert "Min:" in output

    def test_projection_legs_by_day(self):
        """Projection tracks which legs apply to each day."""
        state = create_test_state(reserves=100000, day=0)
        cb_params = CentralBankParams()

        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        projection = project_reserves(state, cb_params)

        # Day 10 should have loan repayment leg
        assert len(projection.legs_by_day.get(10, [])) > 0


class TestIntegrationWithState:
    """Integration tests for projection with full state."""

    def test_projection_respects_withdrawal_forecast(self):
        """Projection starts from reserves minus forecast."""
        state = BankDealerState(
            bank_id="TestBank",
            reserves=100000,
            current_day=0,
        )
        state.withdrawal_forecast = 30000
        state.withdrawals_realized = 10000  # 20000 remaining

        cb_params = CentralBankParams()

        projection = project_reserves(state, cb_params)

        # Starting point = 100000 - 20000 = 80000
        assert projection.path[0] == 80000

    def test_projection_with_multiple_cohorts(self):
        """Projection handles multiple loans and CB borrowing."""
        state = BankDealerState(
            bank_id="TestBank",
            reserves=200000,
            current_day=0,
        )
        cb_params = CentralBankParams()

        # Add various cohorts
        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))  # Matures day 10
        state.add_loan_cohort(
            day=3, amount=30000, rate=Decimal("0.025")
        )  # Matures day 13 (beyond horizon)
        state.add_cb_borrowing(day=0, amount=20000, cb_rate=Decimal("0.03"))  # Matures day 2

        projection = project_reserves(state, cb_params)

        # Should handle all cohorts correctly
        assert projection.base_day == 0
        assert len(projection.path) == 11

        # Check that CB repayment at day 2 is negative
        scheduled_legs = state.get_scheduled_legs(1, 15)
        cb_legs = [leg for leg in scheduled_legs if leg.leg_type == "cb_repay"]
        assert len(cb_legs) == 1
        assert cb_legs[0].amount < 0  # Outflow
