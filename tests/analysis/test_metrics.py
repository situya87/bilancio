"""Tests for bilancio.analysis.metrics module.

This file tests the Kalecki-style payment microstructure metrics and
financial placeholder functions (NPV/IRR).
"""

from decimal import Decimal
from typing import Dict, List
import pytest

from bilancio.analysis.metrics import (
    calculate_npv,
    calculate_irr,
    dues_for_day,
    net_vectors,
    raw_minimum_liquidity,
    size_and_bunching,
    phi_delta,
    replay_intraday_peak,
    velocity,
    creditor_hhi_plus,
    debtor_shortfall_shares,
    start_of_day_money,
    liquidity_gap,
    alpha,
    microstructure_gain_lower_bound,
)


class TestPlaceholderFunctions:
    """Test placeholder functions that are not yet implemented."""

    def test_calculate_npv_raises_not_implemented(self):
        """calculate_npv raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            calculate_npv([], 0.05)
        assert "NPV calculation not yet implemented" in str(exc_info.value)

    def test_calculate_irr_raises_not_implemented(self):
        """calculate_irr raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            calculate_irr([])
        assert "IRR calculation not yet implemented" in str(exc_info.value)


class TestDuesForDay:
    """Test dues_for_day function."""

    def test_empty_events_returns_empty_list(self):
        """Empty events list returns empty dues."""
        assert dues_for_day([], 1) == []

    def test_single_due_on_day_1(self):
        """Single PayableCreated event due on day 1."""
        events = [
            {
                "kind": "PayableCreated",
                "debtor": "A",
                "creditor": "B",
                "amount": "100",
                "due_day": 1,
                "payable_id": "p1",
                "alias": "loan1",
            }
        ]
        result = dues_for_day(events, 1)
        assert len(result) == 1
        assert result[0]["debtor"] == "A"
        assert result[0]["creditor"] == "B"
        assert result[0]["amount"] == Decimal("100")
        assert result[0]["due_day"] == 1
        assert result[0]["pid"] == "p1"
        assert result[0]["alias"] == "loan1"

    def test_multiple_dues_same_day(self):
        """Multiple PayableCreated events due on same day."""
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "B", "creditor": "C", "amount": "200", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "C", "creditor": "A", "amount": "300", "due_day": 2},
        ]
        result = dues_for_day(events, 1)
        assert len(result) == 2
        assert result[0]["amount"] == Decimal("100")
        assert result[1]["amount"] == Decimal("200")

    def test_filters_by_due_day(self):
        """Only returns dues matching the specified day."""
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "B", "creditor": "C", "amount": "200", "due_day": 2},
        ]
        result = dues_for_day(events, 2)
        assert len(result) == 1
        assert result[0]["amount"] == Decimal("200")

    def test_ignores_non_payable_created_events(self):
        """Ignores events that are not PayableCreated."""
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "PayableSettled", "debtor": "A", "creditor": "B", "amount": "100", "day": 1},
        ]
        result = dues_for_day(events, 1)
        assert len(result) == 1

    def test_fallback_to_from_to_fields(self):
        """Falls back to 'from' and 'to' fields if debtor/creditor missing."""
        events = [
            {
                "kind": "PayableCreated",
                "from": "A",
                "to": "B",
                "amount": "100",
                "due_day": 1,
            }
        ]
        result = dues_for_day(events, 1)
        assert len(result) == 1
        assert result[0]["debtor"] == "A"
        assert result[0]["creditor"] == "B"

    def test_handles_missing_fields(self):
        """Handles events with missing optional fields."""
        events = [
            {
                "kind": "PayableCreated",
                "debtor": "A",
                "creditor": "B",
                "due_day": 1,
            }
        ]
        result = dues_for_day(events, 1)
        assert len(result) == 1
        assert result[0]["amount"] == Decimal("0")
        assert result[0]["pid"] is None
        assert result[0]["alias"] is None


class TestNetVectors:
    """Test net_vectors function."""

    def test_empty_dues_returns_empty_dict(self):
        """Empty dues list returns empty dict."""
        assert net_vectors([]) == {}

    def test_single_obligation(self):
        """Single obligation between two agents."""
        dues = [
            {"debtor": "A", "creditor": "B", "amount": "100"}
        ]
        result = net_vectors(dues)

        assert "A" in result
        assert "B" in result
        assert result["A"]["F"] == Decimal("100")
        assert result["A"]["I"] == Decimal("0")
        assert result["A"]["n"] == Decimal("-100")
        assert result["B"]["F"] == Decimal("0")
        assert result["B"]["I"] == Decimal("100")
        assert result["B"]["n"] == Decimal("100")

    def test_multiple_obligations_same_agent(self):
        """Agent with multiple outflows and inflows."""
        dues = [
            {"debtor": "A", "creditor": "B", "amount": "100"},
            {"debtor": "A", "creditor": "C", "amount": "50"},
            {"debtor": "D", "creditor": "A", "amount": "200"},
        ]
        result = net_vectors(dues)

        assert result["A"]["F"] == Decimal("150")  # 100 + 50
        assert result["A"]["I"] == Decimal("200")
        assert result["A"]["n"] == Decimal("50")   # 200 - 150

    def test_circular_obligations(self):
        """Circular obligations: A->B->C->A."""
        dues = [
            {"debtor": "A", "creditor": "B", "amount": "100"},
            {"debtor": "B", "creditor": "C", "amount": "100"},
            {"debtor": "C", "creditor": "A", "amount": "100"},
        ]
        result = net_vectors(dues)

        for agent in ["A", "B", "C"]:
            assert result[agent]["F"] == Decimal("100")
            assert result[agent]["I"] == Decimal("100")
            assert result[agent]["n"] == Decimal("0")

    def test_fallback_to_from_to_fields(self):
        """Falls back to 'from' and 'to' fields."""
        dues = [
            {"from": "A", "to": "B", "amount": "100"}
        ]
        result = net_vectors(dues)

        assert result["A"]["F"] == Decimal("100")
        assert result["B"]["I"] == Decimal("100")


class TestRawMinimumLiquidity:
    """Test raw_minimum_liquidity function."""

    def test_empty_nets_returns_zero(self):
        """Empty nets dict returns zero."""
        assert raw_minimum_liquidity({}) == Decimal("0")

    def test_all_balanced_agents(self):
        """All agents have F == I, returns zero."""
        nets = {
            "A": {"F": Decimal("100"), "I": Decimal("100"), "n": Decimal("0")},
            "B": {"F": Decimal("200"), "I": Decimal("200"), "n": Decimal("0")},
        }
        assert raw_minimum_liquidity(nets) == Decimal("0")

    def test_single_net_debtor(self):
        """Single agent with F > I."""
        nets = {
            "A": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
            "B": {"F": Decimal("50"), "I": Decimal("100"), "n": Decimal("50")},
        }
        assert raw_minimum_liquidity(nets) == Decimal("50")

    def test_multiple_net_debtors(self):
        """Multiple agents with F > I."""
        nets = {
            "A": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
            "B": {"F": Decimal("200"), "I": Decimal("100"), "n": Decimal("-100")},
            "C": {"F": Decimal("50"), "I": Decimal("200"), "n": Decimal("150")},
        }
        # max(0, 100-50) + max(0, 200-100) + max(0, 50-200) = 50 + 100 + 0 = 150
        assert raw_minimum_liquidity(nets) == Decimal("150")

    def test_only_net_creditors(self):
        """All agents have I > F, returns zero."""
        nets = {
            "A": {"F": Decimal("50"), "I": Decimal("100"), "n": Decimal("50")},
            "B": {"F": Decimal("100"), "I": Decimal("200"), "n": Decimal("100")},
        }
        assert raw_minimum_liquidity(nets) == Decimal("0")


class TestSizeAndBunching:
    """Test size_and_bunching function."""

    def test_empty_dues_returns_zero(self):
        """Empty dues returns (0, 0)."""
        S_t, BI_t = size_and_bunching([])
        assert S_t == Decimal("0")
        assert BI_t == Decimal("0")

    def test_single_due_no_bin_fn(self):
        """Single due without bin function."""
        dues = [{"amount": "100"}]
        S_t, BI_t = size_and_bunching(dues)
        assert S_t == Decimal("100")
        assert BI_t == Decimal("0")

    def test_multiple_dues_total_size(self):
        """Multiple dues sum to correct total."""
        dues = [
            {"amount": "100"},
            {"amount": "200"},
            {"amount": "300"},
        ]
        S_t, BI_t = size_and_bunching(dues)
        assert S_t == Decimal("600")
        assert BI_t == Decimal("0")

    def test_with_bin_function(self):
        """Computes bunching index with bin function."""
        dues = [
            {"amount": "100", "bucket": "A"},
            {"amount": "200", "bucket": "A"},
            {"amount": "100", "bucket": "B"},
        ]
        bin_fn = lambda d: d["bucket"]
        S_t, BI_t = size_and_bunching(dues, bin_fn)

        assert S_t == Decimal("400")
        # Bucket A: 300, Bucket B: 100
        # Mean = 200, StdDev = 100, BI = 100/200 = 0.5
        assert BI_t == Decimal("0.5")

    def test_single_bucket_returns_zero_bi(self):
        """Single bucket returns BI = 0 (StdDev = 0)."""
        dues = [
            {"amount": "100", "bucket": "A"},
            {"amount": "200", "bucket": "A"},
        ]
        bin_fn = lambda d: d["bucket"]
        S_t, BI_t = size_and_bunching(dues, bin_fn)

        assert S_t == Decimal("300")
        assert BI_t == Decimal("0")  # StdDev of single value = 0

    def test_zero_mean_buckets(self):
        """Zero mean buckets returns BI = 0."""
        dues = [
            {"amount": "0", "bucket": "A"},
            {"amount": "0", "bucket": "B"},
        ]
        bin_fn = lambda d: d["bucket"]
        S_t, BI_t = size_and_bunching(dues, bin_fn)

        assert S_t == Decimal("0")
        assert BI_t == Decimal("0")


class TestPhiDelta:
    """Test phi_delta function."""

    def test_empty_dues_returns_none(self):
        """Empty dues returns (None, None)."""
        phi, delta = phi_delta([], [], 1)
        assert phi is None
        assert delta is None

    def test_zero_settlement_amount(self):
        """Zero dues amount returns (None, None)."""
        dues = [{"amount": "0", "pid": "p1", "due_day": 1}]
        events = []
        phi, delta = phi_delta(events, dues, 1)
        assert phi is None
        assert delta is None

    def test_full_settlement_on_time(self):
        """All dues settled on time."""
        dues = [
            {"amount": "100", "pid": "p1", "due_day": 1},
            {"amount": "200", "pid": "p2", "due_day": 1},
        ]
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p1"},
            {"kind": "PayableSettled", "day": 1, "amount": "200", "pid": "p2"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("1")
        assert delta == Decimal("0")

    def test_partial_settlement(self):
        """Partial settlement on due day."""
        dues = [
            {"amount": "100", "pid": "p1", "due_day": 1},
            {"amount": "200", "pid": "p2", "due_day": 1},
        ]
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p1"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        # 100/300 = 1/3
        assert phi == Decimal("100") / Decimal("300")
        assert delta == Decimal("200") / Decimal("300")

    def test_no_settlement(self):
        """No settlement on due day."""
        dues = [{"amount": "100", "pid": "p1", "due_day": 1}]
        events = []
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("0")
        assert delta == Decimal("1")

    def test_ignores_wrong_day_settlements(self):
        """Ignores settlements on different days."""
        dues = [{"amount": "100", "pid": "p1", "due_day": 1}]
        events = [
            {"kind": "PayableSettled", "day": 2, "amount": "100", "pid": "p1"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("0")
        assert delta == Decimal("1")

    def test_ignores_different_due_day_payables(self):
        """Ignores payables that originally had different due_day."""
        dues = [{"amount": "100", "pid": "p1", "due_day": 1}]
        events = [
            # This settlement is on day 1, and due_day from dues is 1 - matches
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p1"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("1")  # Correct match

        # Now test settlement on day 1 but payable was due on day 2 (shouldn't count)
        dues2 = [{"amount": "100", "pid": "p2", "due_day": 2}]
        events2 = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p2"},
        ]
        phi2, delta2 = phi_delta(events2, dues2, 1)
        # Payable is due on day 2, not day 1, so nothing settles on time
        # S_t uses total dues amount, phi=0/100=0, delta=1-0=1
        assert phi2 == Decimal("0")
        assert delta2 == Decimal("1")

    def test_uses_alias_for_matching(self):
        """Uses alias field for matching if pid missing."""
        dues = [{"amount": "100", "alias": "loan1", "due_day": 1}]
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "alias": "loan1"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("1")
        assert delta == Decimal("0")

    def test_uses_contract_id_for_matching(self):
        """Uses contract_id field for matching."""
        dues = [{"amount": "100", "pid": "p1", "due_day": 1}]
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "contract_id": "p1"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("1")
        assert delta == Decimal("0")


class TestReplayIntradayPeak:
    """Test replay_intraday_peak function."""

    def test_empty_events_returns_zero(self):
        """Empty events returns zero peak and empty steps."""
        peak, steps, gross = replay_intraday_peak([], 1)
        assert peak == Decimal("0")
        assert steps == []
        assert gross == Decimal("0")

    def test_single_payment(self):
        """Single payment creates one step."""
        events = [
            {
                "kind": "PayableSettled",
                "day": 1,
                "amount": "100",
                "debtor": "A",
                "creditor": "B",
            }
        ]
        peak, steps, gross = replay_intraday_peak(events, 1)

        assert peak == Decimal("100")
        assert gross == Decimal("100")
        assert len(steps) == 1
        assert steps[0]["payer"] == "A"
        assert steps[0]["payee"] == "B"
        assert steps[0]["amount"] == Decimal("100")
        assert steps[0]["P_prefix"] == Decimal("100")

    def test_multiple_payments_same_payer(self):
        """Multiple payments from same payer accumulate."""
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "debtor": "A", "creditor": "B"},
            {"kind": "PayableSettled", "day": 1, "amount": "50", "debtor": "A", "creditor": "C"},
        ]
        peak, steps, gross = replay_intraday_peak(events, 1)

        # After step 1: A = +100 (owes net 100), peak = 100
        # After step 2: A = +150 (owes net 150), peak = 150
        assert peak == Decimal("150")
        assert gross == Decimal("150")
        assert len(steps) == 2

    def test_offsetting_payments(self):
        """Payments in both directions offset."""
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "debtor": "A", "creditor": "B"},
            {"kind": "PayableSettled", "day": 1, "amount": "60", "debtor": "B", "creditor": "A"},
        ]
        peak, steps, gross = replay_intraday_peak(events, 1)

        # After step 1: A = +100, B = -100, P = 100
        # After step 2: A = +100-60 = +40, B = -100+60 = -40, P = 40
        assert peak == Decimal("100")
        assert gross == Decimal("160")
        assert len(steps) == 2

    def test_ignores_wrong_day(self):
        """Ignores events on different days."""
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "debtor": "A", "creditor": "B"},
            {"kind": "PayableSettled", "day": 2, "amount": "200", "debtor": "A", "creditor": "B"},
        ]
        peak, steps, gross = replay_intraday_peak(events, 1)

        assert peak == Decimal("100")
        assert gross == Decimal("100")
        assert len(steps) == 1

    def test_ignores_zero_amount(self):
        """Ignores zero-amount settlements."""
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "0", "debtor": "A", "creditor": "B"},
            {"kind": "PayableSettled", "day": 1, "amount": "100", "debtor": "A", "creditor": "B"},
        ]
        peak, steps, gross = replay_intraday_peak(events, 1)

        assert gross == Decimal("100")
        assert len(steps) == 1

    def test_uses_from_to_fields(self):
        """Falls back to 'from' and 'to' fields."""
        events = [
            {
                "kind": "PayableSettled",
                "day": 1,
                "amount": "100",
                "from": "A",
                "to": "B",
            }
        ]
        peak, steps, gross = replay_intraday_peak(events, 1)

        assert steps[0]["payer"] == "A"
        assert steps[0]["payee"] == "B"

    def test_steps_include_day_field(self):
        """Steps include day field."""
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "debtor": "A", "creditor": "B"},
        ]
        peak, steps, gross = replay_intraday_peak(events, 1)

        assert steps[0]["day"] == 1


class TestVelocity:
    """Test velocity function."""

    def test_zero_peak_returns_none(self):
        """Zero peak returns None."""
        assert velocity(Decimal("100"), Decimal("0")) is None

    def test_normal_case(self):
        """Normal case returns gross / peak."""
        result = velocity(Decimal("300"), Decimal("100"))
        assert result == Decimal("3")

    def test_equal_gross_and_peak(self):
        """Gross equals peak returns 1."""
        result = velocity(Decimal("100"), Decimal("100"))
        assert result == Decimal("1")

    def test_gross_less_than_peak(self):
        """Gross less than peak returns fraction."""
        result = velocity(Decimal("50"), Decimal("100"))
        assert result == Decimal("0.5")


class TestCreditorHHIPlus:
    """Test creditor_hhi_plus function."""

    def test_empty_nets_returns_none(self):
        """Empty nets returns None."""
        assert creditor_hhi_plus({}) is None

    def test_no_creditors_returns_none(self):
        """All agents have non-positive n, returns None."""
        nets = {
            "A": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
            "B": {"F": Decimal("100"), "I": Decimal("100"), "n": Decimal("0")},
        }
        assert creditor_hhi_plus(nets) is None

    def test_single_creditor(self):
        """Single creditor returns HHI = 1."""
        nets = {
            "A": {"F": Decimal("50"), "I": Decimal("100"), "n": Decimal("50")},
            "B": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
        }
        result = creditor_hhi_plus(nets)
        assert result == Decimal("1")

    def test_two_equal_creditors(self):
        """Two equal creditors returns HHI = 0.5."""
        nets = {
            "A": {"F": Decimal("0"), "I": Decimal("100"), "n": Decimal("100")},
            "B": {"F": Decimal("0"), "I": Decimal("100"), "n": Decimal("100")},
            "C": {"F": Decimal("200"), "I": Decimal("0"), "n": Decimal("-200")},
        }
        result = creditor_hhi_plus(nets)
        # Each creditor has 50% share: 0.5^2 + 0.5^2 = 0.5
        assert result == Decimal("0.5")

    def test_unequal_creditors(self):
        """Unequal creditor shares."""
        nets = {
            "A": {"F": Decimal("0"), "I": Decimal("100"), "n": Decimal("100")},
            "B": {"F": Decimal("0"), "I": Decimal("200"), "n": Decimal("200")},
            "C": {"F": Decimal("300"), "I": Decimal("0"), "n": Decimal("-300")},
        }
        result = creditor_hhi_plus(nets)
        # A: 100/300 = 1/3, B: 200/300 = 2/3
        # HHI = (1/3)^2 + (2/3)^2 = 1/9 + 4/9 = 5/9
        expected = Decimal("5") / Decimal("9")
        # Use quantize for comparison due to potential precision differences
        assert abs(result - expected) < Decimal("0.0001")


class TestDebtorShortfallShares:
    """Test debtor_shortfall_shares function."""

    def test_empty_nets_returns_empty(self):
        """Empty nets returns empty dict."""
        assert debtor_shortfall_shares({}) == {}

    def test_no_net_debtors_returns_none(self):
        """All agents have I >= F, returns None for all."""
        nets = {
            "A": {"F": Decimal("50"), "I": Decimal("100"), "n": Decimal("50")},
            "B": {"F": Decimal("100"), "I": Decimal("100"), "n": Decimal("0")},
        }
        result = debtor_shortfall_shares(nets)
        assert result["A"] is None
        assert result["B"] is None

    def test_single_debtor(self):
        """Single net debtor gets DS = 1, creditor gets 0."""
        nets = {
            "A": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
            "B": {"F": Decimal("50"), "I": Decimal("100"), "n": Decimal("50")},
        }
        result = debtor_shortfall_shares(nets)
        assert result["A"] == Decimal("1")
        assert result["B"] == Decimal("0")  # Creditor has 0 shortfall

    def test_two_equal_debtors(self):
        """Two equal debtors get DS = 0.5 each."""
        nets = {
            "A": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
            "B": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
            "C": {"F": Decimal("0"), "I": Decimal("100"), "n": Decimal("100")},
        }
        result = debtor_shortfall_shares(nets)
        assert result["A"] == Decimal("0.5")
        assert result["B"] == Decimal("0.5")
        assert result["C"] == Decimal("0")

    def test_unequal_debtors(self):
        """Unequal debtor shortfalls."""
        nets = {
            "A": {"F": Decimal("150"), "I": Decimal("50"), "n": Decimal("-100")},
            "B": {"F": Decimal("100"), "I": Decimal("50"), "n": Decimal("-50")},
            "C": {"F": Decimal("0"), "I": Decimal("150"), "n": Decimal("150")},
        }
        result = debtor_shortfall_shares(nets)
        # A shortfall: 100, B shortfall: 50, total: 150
        assert result["A"] == Decimal("100") / Decimal("150")
        assert result["B"] == Decimal("50") / Decimal("150")
        assert result["C"] == Decimal("0")


class TestStartOfDayMoney:
    """Test start_of_day_money function."""

    def test_empty_balances_returns_zero(self):
        """Empty balance rows returns zero."""
        assert start_of_day_money([], 1) == Decimal("0")

    def test_single_agent_single_mop(self):
        """Single agent with one means-of-payment."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "100",
                "assets_bank_deposit": "0",
                "assets_reserve_deposit": "0",
            }
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("100")

    def test_multiple_mops(self):
        """Agent with multiple means-of-payment types."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "100",
                "assets_bank_deposit": "200",
                "assets_reserve_deposit": "300",
            }
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("600")

    def test_multiple_agents(self):
        """Multiple agents sum correctly."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "100",
                "assets_bank_deposit": "0",
                "assets_reserve_deposit": "0",
            },
            {
                "agent_id": "Firm1",
                "assets_cash": "50",
                "assets_bank_deposit": "150",
                "assets_reserve_deposit": "0",
            },
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("300")

    def test_skips_system_row(self):
        """Skips the SYSTEM aggregate row."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "100",
                "assets_bank_deposit": "0",
                "assets_reserve_deposit": "0",
            },
            {
                "agent_id": "SYSTEM",
                "assets_cash": "1000",  # Should be skipped
                "assets_bank_deposit": "0",
                "assets_reserve_deposit": "0",
            },
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("100")

    def test_skips_rows_with_item_type(self):
        """Skips ad-hoc summary rows with item_type."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "100",
                "assets_bank_deposit": "0",
                "assets_reserve_deposit": "0",
            },
            {
                "agent_id": "Bank1",
                "item_type": "summary",
                "assets_cash": "999",  # Should be skipped
                "assets_bank_deposit": "0",
                "assets_reserve_deposit": "0",
            },
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("100")

    def test_handles_none_values(self):
        """Handles None values gracefully."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": None,
                "assets_bank_deposit": "100",
                "assets_reserve_deposit": None,
            }
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("100")

    def test_handles_empty_string_values(self):
        """Handles empty string values."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "",
                "assets_bank_deposit": "100",
                "assets_reserve_deposit": "",
            }
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("100")

    def test_handles_none_string_values(self):
        """Handles 'None' string values."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "None",
                "assets_bank_deposit": "100",
                "assets_reserve_deposit": "None",
            }
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("100")

    def test_handles_invalid_decimal_values(self):
        """Handles invalid decimal values gracefully."""
        bal_rows = [
            {
                "agent_id": "Bank1",
                "assets_cash": "invalid",
                "assets_bank_deposit": "100",
                "assets_reserve_deposit": "0",
            }
        ]
        assert start_of_day_money(bal_rows, 1) == Decimal("100")


class TestLiquidityGap:
    """Test liquidity_gap function."""

    def test_zero_gap(self):
        """M_t equals Mbar_t returns zero."""
        assert liquidity_gap(Decimal("100"), Decimal("100")) == Decimal("0")

    def test_positive_gap(self):
        """Mbar_t > M_t returns positive gap."""
        assert liquidity_gap(Decimal("150"), Decimal("100")) == Decimal("50")

    def test_negative_difference_returns_zero(self):
        """M_t > Mbar_t returns zero (no gap)."""
        assert liquidity_gap(Decimal("100"), Decimal("150")) == Decimal("0")

    def test_both_zero(self):
        """Both zero returns zero."""
        assert liquidity_gap(Decimal("0"), Decimal("0")) == Decimal("0")


class TestAlpha:
    """Test alpha function."""

    def test_zero_size_returns_none(self):
        """S_t = 0 returns None."""
        assert alpha(Decimal("50"), Decimal("0")) is None

    def test_zero_mbar(self):
        """Mbar_t = 0 returns alpha = 1."""
        assert alpha(Decimal("0"), Decimal("100")) == Decimal("1")

    def test_mbar_equals_size(self):
        """Mbar_t = S_t returns alpha = 0."""
        assert alpha(Decimal("100"), Decimal("100")) == Decimal("0")

    def test_mbar_half_of_size(self):
        """Mbar_t = S_t/2 returns alpha = 0.5."""
        assert alpha(Decimal("50"), Decimal("100")) == Decimal("0.5")

    def test_mbar_greater_than_size(self):
        """Mbar_t > S_t returns negative alpha."""
        result = alpha(Decimal("150"), Decimal("100"))
        assert result == Decimal("-0.5")


class TestMicrostructureGainLowerBound:
    """Test microstructure_gain_lower_bound function."""

    def test_zero_peak_returns_none(self):
        """Mpeak_rtgs = 0 returns None."""
        assert microstructure_gain_lower_bound(Decimal("50"), Decimal("0")) is None

    def test_mbar_equals_peak(self):
        """Mbar_t = Mpeak_rtgs returns 0."""
        result = microstructure_gain_lower_bound(Decimal("100"), Decimal("100"))
        assert result == Decimal("0")

    def test_mbar_less_than_peak(self):
        """Mbar_t < Mpeak_rtgs returns positive gain."""
        result = microstructure_gain_lower_bound(Decimal("50"), Decimal("100"))
        assert result == Decimal("0.5")

    def test_mbar_greater_than_peak(self):
        """Mbar_t > Mpeak_rtgs returns negative gain."""
        result = microstructure_gain_lower_bound(Decimal("150"), Decimal("100"))
        assert result == Decimal("-0.5")

    def test_zero_mbar(self):
        """Mbar_t = 0 returns 1."""
        result = microstructure_gain_lower_bound(Decimal("0"), Decimal("100"))
        assert result == Decimal("1")
