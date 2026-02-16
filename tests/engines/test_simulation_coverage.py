"""
Tests targeting uncovered branches in bilancio.engines.simulation.

Coverage gaps addressed:
- MonteCarloEngine: n_simulations parameter, random_seed, run(), set_num_simulations()
- _has_open_obligations: True branch (payable present)
- _impacted_today: counting impact events
- run_day Phase D: CB corridor loan repayment
- run_day Phase B1: scheduled actions execution
- run_until_stable: stability detection, max_days cap, rollover mode
"""

from decimal import Decimal

import pytest

from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.delivery import DeliveryObligation
from bilancio.engines.simulation import (
    DEFAULT_EVENTS,
    DayReport,
    MonteCarloEngine,
    _has_open_obligations,
    _impacted_today,
    run_day,
    run_until_stable,
)
from bilancio.engines.system import System

# =============================================================================
# MonteCarloEngine Tests
# =============================================================================


class TestMonteCarloEngine:
    """Tests for the MonteCarloEngine class."""

    def test_default_constructor(self):
        """Default constructor sets num_simulations=1000."""
        engine = MonteCarloEngine()
        assert engine.num_simulations == 1000
        assert engine.n_simulations == 1000

    def test_num_simulations_parameter(self):
        """num_simulations parameter is respected."""
        engine = MonteCarloEngine(num_simulations=500)
        assert engine.num_simulations == 500
        assert engine.n_simulations == 500

    def test_n_simulations_parameter(self):
        """n_simulations alternative parameter takes priority when given."""
        engine = MonteCarloEngine(n_simulations=42)
        assert engine.num_simulations == 42
        assert engine.n_simulations == 42

    def test_n_simulations_overrides_num_simulations(self):
        """When both are given, n_simulations takes priority."""
        engine = MonteCarloEngine(num_simulations=100, n_simulations=77)
        assert engine.num_simulations == 77
        assert engine.n_simulations == 77

    def test_random_seed_sets_deterministic(self):
        """random_seed produces reproducible results."""
        engine1 = MonteCarloEngine(num_simulations=10, random_seed=42)
        result1 = engine1.run("test_scenario")

        engine2 = MonteCarloEngine(num_simulations=10, random_seed=42)
        result2 = engine2.run("test_scenario")

        outcomes1 = [r["outcome"] for r in result1["results"]]
        outcomes2 = [r["outcome"] for r in result2["results"]]
        assert outcomes1 == outcomes2

    def test_run_returns_correct_structure(self):
        """run() returns dict with expected keys and values."""
        engine = MonteCarloEngine(num_simulations=5, random_seed=123)
        result = engine.run("my_scenario")

        assert result["num_simulations"] == 5
        assert "mean" in result
        assert "min" in result
        assert "max" in result
        assert len(result["results"]) == 5

        # Each run result has expected fields
        for r in result["results"]:
            assert "run_id" in r
            assert "outcome" in r
            assert r["scenario"] == "my_scenario"

    def test_run_statistics_are_consistent(self):
        """Summary statistics are consistent with individual results."""
        engine = MonteCarloEngine(num_simulations=20, random_seed=99)
        result = engine.run("stats_test")

        outcomes = [r["outcome"] for r in result["results"]]
        assert result["min"] == min(outcomes)
        assert result["max"] == max(outcomes)
        assert abs(result["mean"] - sum(outcomes) / len(outcomes)) < 1e-10

    def test_set_num_simulations(self):
        """set_num_simulations updates the count."""
        engine = MonteCarloEngine(num_simulations=10)
        assert engine.num_simulations == 10

        engine.set_num_simulations(50)
        assert engine.num_simulations == 50

        # Verify it actually runs with the new count
        result = engine.run("scenario")
        assert len(result["results"]) == 50


# =============================================================================
# DayReport Tests
# =============================================================================


class TestDayReport:
    """Test the DayReport dataclass."""

    def test_day_report_creation(self):
        report = DayReport(day=3, impacted=5, notes="test")
        assert report.day == 3
        assert report.impacted == 5
        assert report.notes == "test"

    def test_day_report_default_notes(self):
        report = DayReport(day=0, impacted=0)
        assert report.notes == ""


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHasOpenObligations:
    """Tests for _has_open_obligations helper."""

    def test_no_obligations_returns_false(self):
        """System with only cash/deposits has no open obligations."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 100)

        assert _has_open_obligations(sys) is False

    def test_payable_present_returns_true(self):
        """System with a payable has open obligations."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=5,
        )
        sys.add_contract(payable)

        assert _has_open_obligations(sys) is True

    def test_delivery_obligation_present_returns_true(self):
        """System with a delivery obligation has open obligations."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        f1 = Firm(id="F1", name="F1", kind="firm")
        f2 = Firm(id="F2", name="F2", kind="firm")
        sys.add_agent(cb)
        sys.add_agent(f1)
        sys.add_agent(f2)

        # Give F1 some stock to deliver
        sys.create_stock("F1", "widget", 10, Decimal("50"))

        delivery = DeliveryObligation(
            id="D1",
            kind=InstrumentKind.DELIVERY_OBLIGATION,
            amount=5,
            denom="X",
            asset_holder_id="F2",
            liability_issuer_id="F1",
            sku="widget",
            unit_price=Decimal("50"),
            due_day=3,
        )
        sys.add_contract(delivery)

        assert _has_open_obligations(sys) is True

    def test_empty_contracts_returns_false(self):
        """System with zero contracts has no open obligations."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        sys.add_agent(cb)
        assert _has_open_obligations(sys) is False


class TestImpactedToday:
    """Tests for _impacted_today helper."""

    def test_no_events_returns_zero(self):
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        sys.add_agent(cb)
        assert _impacted_today(sys, 0) == 0

    def test_counts_impact_events_only(self):
        """Only IMPACT_EVENTS are counted, not phase markers."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        sys.add_agent(cb)

        # Non-impact events (phase markers)
        sys.state.events.append({"kind": "PhaseA", "day": 0})
        sys.state.events.append({"kind": "PhaseB", "day": 0})
        sys.state.events.append({"kind": "PhaseC", "day": 0})

        # Impact events
        sys.state.events.append({"kind": "PayableSettled", "day": 0})
        sys.state.events.append({"kind": "InterbankCleared", "day": 0})

        # Impact event on different day
        sys.state.events.append({"kind": "PayableSettled", "day": 1})

        assert _impacted_today(sys, 0) == 2
        assert _impacted_today(sys, 1) == 1
        assert _impacted_today(sys, 2) == 0


# =============================================================================
# run_day Phase D: CB Corridor Tests
# =============================================================================


class TestRunDayPhaseD:
    """Tests for Phase D (CB corridor) within run_day."""

    def _setup_cb_system(self):
        """Create a system with CB, bank, and household for Phase D testing."""
        sys = System()
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(bank)
        sys.add_agent(h1)
        return sys

    def test_run_day_repays_due_cb_loans(self):
        """Phase D repays CB loans that are due on the current day."""
        sys = self._setup_cb_system()

        # Create a CB loan on day 0 (matures day 2)
        loan_id = sys.cb_lend_reserves("B1", 1000, day=0)
        assert loan_id in sys.state.contracts

        # Give bank extra reserves for interest (1000 * 0.03 = 30)
        sys.mint_reserves("B1", 50)

        # Advance to day 2 manually, then run_day
        sys.state.day = 2
        run_day(sys)

        # Loan should have been repaid
        assert loan_id not in sys.state.contracts

        # Check CBLoanRepaid event
        repaid_events = [e for e in sys.state.events if e.get("kind") == "CBLoanRepaid"]
        assert len(repaid_events) == 1
        assert repaid_events[0]["bank_id"] == "B1"
        assert repaid_events[0]["total_repaid"] == 1030

        sys.assert_invariants()

    def test_run_day_skips_phase_d_without_cb(self):
        """Without a central bank agent, Phase D is skipped entirely."""
        sys = System()
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(h1)
        sys.add_agent(h2)
        # Manually give them cash (no CB, so use a workaround)
        # Actually we need a CB to mint_cash, so let's just run with no contracts
        sys.state.day = 0

        # Should not raise even without a CB
        run_day(sys)
        assert sys.state.day == 1

    def test_run_day_cb_loan_not_yet_due(self):
        """CB loan not yet due is not repaid."""
        sys = self._setup_cb_system()

        # Create loan on day 0 (matures day 2)
        loan_id = sys.cb_lend_reserves("B1", 1000, day=0)

        # Give bank extra reserves
        sys.mint_reserves("B1", 50)

        # Day 0: loan matures on day 2, so it shouldn't be repaid today
        run_day(sys)
        assert sys.state.day == 1
        assert loan_id in sys.state.contracts  # Still active

        # Day 1: still not due
        run_day(sys)
        assert sys.state.day == 2
        # CB loan is_due checks >= maturity_day, and maturity_day = 0+2 = 2
        # On day 1 run_day processes day 1, but we called it when day was 1, so it processes day 1
        # Let's verify the loan is still there or was repaid
        # is_due(1) for a loan issued at day 0 with maturity_day 2: 1 >= 2 = False
        assert loan_id in sys.state.contracts

        # Day 2: now it's due
        run_day(sys)
        assert sys.state.day == 3
        assert loan_id not in sys.state.contracts

        sys.assert_invariants()

    def test_run_day_credits_reserve_interest(self):
        """Phase D credits interest on reserve deposits."""
        sys = System()
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
            reserves_accrue_interest=True,
        )
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        sys.add_agent(cb)
        sys.add_agent(bank)

        sys.mint_reserves_with_interest("B1", 10000, day=0)

        initial_reserves = sum(
            c.amount
            for c in sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        assert initial_reserves == 10000

        # Run days 0 and 1 — interest not due yet (due at day 2)
        run_day(sys)  # day 0 -> 1
        run_day(sys)  # day 1 -> 2

        # Day 2: interest should be credited
        run_day(sys)  # day 2 -> 3

        final_reserves = sum(
            c.amount
            for c in sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        # Should have earned interest: 10000 * 0.01 = 100
        assert final_reserves == 10100

        sys.assert_invariants()

    def test_run_day_multiple_cb_loans_due_same_day(self):
        """Multiple CB loans due on the same day are all repaid."""
        sys = self._setup_cb_system()

        # Two loans issued on day 0, both mature day 2
        loan1_id = sys.cb_lend_reserves("B1", 500, day=0)
        loan2_id = sys.cb_lend_reserves("B1", 300, day=0)

        # Give bank extra for interest: (500+300)*0.03 = 24
        sys.mint_reserves("B1", 50)

        sys.state.day = 2
        run_day(sys)

        assert loan1_id not in sys.state.contracts
        assert loan2_id not in sys.state.contracts

        repaid_events = [e for e in sys.state.events if e.get("kind") == "CBLoanRepaid"]
        assert len(repaid_events) == 2

        sys.assert_invariants()


# =============================================================================
# run_day Phase B1: Scheduled Actions Tests
# =============================================================================


class TestRunDayScheduledActions:
    """Tests for Phase B1 scheduled actions execution within run_day."""

    def test_scheduled_action_mint_cash(self):
        """Scheduled mint_cash action is executed at Phase B1."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)

        # Schedule a mint_cash action for day 0
        sys.state.scheduled_actions_by_day[0] = [
            {"mint_cash": {"to": "H1", "amount": 200}},
        ]

        # Verify H1 has no cash before
        h1_cash_before = sum(
            c.amount
            for c in sys.state.contracts.values()
            if c.kind == InstrumentKind.CASH and c.asset_holder_id == "H1"
        )
        assert h1_cash_before == 0

        run_day(sys)

        # H1 should have 200 cash
        h1_cash_after = sum(
            c.amount
            for c in sys.state.contracts.values()
            if c.kind == InstrumentKind.CASH and c.asset_holder_id == "H1"
        )
        assert h1_cash_after == 200

    def test_no_scheduled_actions_for_day(self):
        """Days with no scheduled actions run without error."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)

        # No scheduled actions at all
        run_day(sys)
        assert sys.state.day == 1

    def test_scheduled_create_payable(self):
        """Scheduled create_payable creates a payable and it's settled same day if due."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        # Give H1 enough to pay
        sys.mint_cash("H1", 500)

        # Schedule a payable due today
        sys.state.scheduled_actions_by_day[0] = [
            {"create_payable": {"from": "H1", "to": "H2", "amount": 100, "due_day": 0}},
        ]

        run_day(sys)

        # Payable should have been created and settled (due day 0, B1 creates, B2 settles)
        settled_events = [e for e in sys.state.events if e.get("kind") == "PayableSettled"]
        assert len(settled_events) == 1
        assert settled_events[0]["amount"] == 100


# =============================================================================
# run_until_stable Tests
# =============================================================================


class TestRunUntilStable:
    """Tests for the run_until_stable function."""

    def test_stable_system_stops_quickly(self):
        """A system with no obligations stabilizes after quiet_days."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 100)

        reports = run_until_stable(sys, max_days=100, quiet_days=2)

        # Should stop after 2 quiet days (no impact events, no obligations)
        assert len(reports) == 2
        assert all(r.impacted == 0 for r in reports)

    def test_obligations_prevent_early_stop(self):
        """System doesn't stop while obligations remain, even if days are quiet."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        sys.mint_cash("H1", 500)

        # Create payable due on day 5 (far in future)
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=5,
        )
        sys.add_contract(payable)

        # Run with quiet_days=2 and max_days=20
        reports = run_until_stable(sys, max_days=20, quiet_days=2)

        # The payable is settled on day 5, so system becomes stable after day 5 + quiet_days
        # Day 5: PayableSettled (impacted=1), resets consecutive_quiet
        # Day 6: quiet (1)
        # Day 7: quiet (2) → stable
        # Total: 8 reports (days 0-7)
        assert len(reports) >= 6  # At least runs through day 5
        # Payable should be settled
        assert "P1" not in sys.state.contracts

    def test_max_days_caps_simulation(self):
        """Simulation stops at max_days even if not stable."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        sys.mint_cash("H1", 500)

        # Create payable due very far out
        payable = Payable(
            id="P_FAR",
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=999,
        )
        sys.add_contract(payable)

        # max_days=5 should cap it even though obligation exists
        reports = run_until_stable(sys, max_days=5, quiet_days=2)
        assert len(reports) == 5

    def test_reports_contain_correct_day_and_impact(self):
        """DayReports track correct day and impact counts."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        sys.mint_cash("H1", 300)

        # Create payable due on day 0
        payable = Payable(
            id="P0",
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
        )
        sys.add_contract(payable)

        reports = run_until_stable(sys, max_days=10, quiet_days=2)

        # Day 0 should have impact (PayableSettled)
        assert reports[0].day == 0
        assert reports[0].impacted >= 1

        # Subsequent days should be quiet
        for r in reports[1:]:
            assert r.impacted == 0

    def test_rollover_enabled_stability(self):
        """With rollover enabled, stability is based only on consecutive quiet days."""
        sys = System()
        sys.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 100)

        # Even with no obligations, rollover mode skips the obligation check
        reports = run_until_stable(sys, max_days=50, quiet_days=3)

        # Should stop after exactly 3 quiet days
        assert len(reports) == 3
        assert all(r.impacted == 0 for r in reports)

    def test_quiet_days_resets_on_impact(self):
        """Consecutive quiet counter resets when an impact event occurs."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        sys.mint_cash("H1", 500)

        # Two payables: one due day 0, one due day 3
        p1 = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
        )
        p2 = Payable(
            id="P2",
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=3,
        )
        sys.add_contract(p1)
        sys.add_contract(p2)

        reports = run_until_stable(sys, max_days=20, quiet_days=2)

        # Day 0: PayableSettled (P1) → impacted, quiet=0
        # Day 1: quiet → quiet=1
        # Day 2: quiet → quiet=2, but P2 still open → no stability
        # Day 3: PayableSettled (P2) → impacted, quiet=0
        # Day 4: quiet → quiet=1
        # Day 5: quiet → quiet=2, no obligations → stable

        assert reports[0].impacted >= 1  # Day 0: P1 settled
        assert reports[3].impacted >= 1  # Day 3: P2 settled

        # All obligations gone
        assert "P1" not in sys.state.contracts
        assert "P2" not in sys.state.contracts

    def test_enable_dealer_false_by_default(self):
        """run_until_stable with dealer disabled doesn't try dealer trading."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 100)

        # Should work fine without dealer
        reports = run_until_stable(sys, max_days=10, quiet_days=2, enable_dealer=False)
        assert len(reports) >= 2

    def test_start_day_preserved(self):
        """run_until_stable works correctly when starting from a non-zero day."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 100)

        sys.state.day = 10  # Start from day 10

        reports = run_until_stable(sys, max_days=50, quiet_days=2)

        assert reports[0].day == 10
        assert reports[1].day == 11
        assert sys.state.day == 12  # Advanced by 2 quiet days

    def test_rollover_defaults_prevent_stability(self):
        """With rollover, defaults reset the consecutive no-defaults counter."""
        # Use expel-agent mode so defaults log events instead of raising
        sys = System(default_mode="expel-agent")
        sys.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        # H1 has NO cash — the payable will default
        # Create payable due on day 1 with maturity_distance for rollover
        payable = Payable(
            id="P_DEF",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=1,
            maturity_distance=5,
        )
        sys.add_contract(payable)

        # Run with quiet_days=2 and max_days=10
        reports = run_until_stable(sys, max_days=10, quiet_days=2)

        # Day 0: no default events (payable not due yet) → no_defaults=1
        # Day 1: default occurs (H1 can't pay) → no_defaults reset to 0
        # Day 2: quiet, no defaults → no_defaults=1
        # Day 3: quiet, no defaults → no_defaults=2 → stable
        # Total: 4 reports (days 0-3)

        # Verify a default event occurred
        default_events = [e for e in sys.state.events if e.get("kind") in DEFAULT_EVENTS]
        assert len(default_events) >= 1, "Expected at least one default event"

        # The simulation should NOT have stopped after just 2 days
        # (day 0 quiet + day 1 quiet) because the default on day 1
        # resets the counter
        assert len(reports) > 2, "Defaults should prevent premature stability"

    def test_rollover_settlements_dont_prevent_stability(self):
        """In rollover mode, settlement events alone don't block stability."""
        sys = System()
        sys.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        sys.mint_cash("H1", 500)

        # Create payable due on day 0 — will be settled successfully
        payable = Payable(
            id="P_SETTLE",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
            maturity_distance=5,
        )
        sys.add_contract(payable)

        # With quiet_days=2: settlements happen but no defaults
        # Day 0: PayableSettled (impact event!) but no defaults → no_defaults=1
        # Day 1: rollover payable may settle (impact) but no defaults → no_defaults=2 → stable
        reports = run_until_stable(sys, max_days=10, quiet_days=2)

        # Settlements should have occurred
        settled = [e for e in sys.state.events if e.get("kind") == "PayableSettled"]
        assert len(settled) >= 1, "Expected at least one settlement"

        # No defaults should have occurred
        default_events = [e for e in sys.state.events if e.get("kind") in DEFAULT_EVENTS]
        assert len(default_events) == 0, "No defaults should occur with sufficient cash"

        # Should stop quickly since no defaults (within quiet_days from start)
        assert len(reports) <= 4, "Should stabilize quickly without defaults"

    def test_rollover_mixed_defaults_and_settlements_same_day(self):
        """A day with both a settlement and a default resets the no-defaults counter."""
        # Use expel-agent mode so defaults log events instead of raising
        sys = System(default_mode="expel-agent")
        sys.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        h3 = Household(id="H3", name="H3", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)
        sys.add_agent(h3)

        # H1 can pay → settlement; H3 cannot pay → default; both due day 1
        sys.mint_cash("H1", 500)

        p_ok = Payable(
            id="P_OK",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=1,
            maturity_distance=5,
        )
        p_bad = Payable(
            id="P_BAD",
            kind=InstrumentKind.PAYABLE,
            amount=200,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H3",
            due_day=1,
            maturity_distance=5,
        )
        sys.add_contract(p_ok)
        sys.add_contract(p_bad)

        reports = run_until_stable(sys, max_days=10, quiet_days=2)

        # Day 1 should have both a settlement and a default
        settled = [e for e in sys.state.events if e.get("kind") == "PayableSettled"]
        defaults = [e for e in sys.state.events if e.get("kind") in DEFAULT_EVENTS]
        assert len(settled) >= 1, "Expected at least one settlement"
        assert len(defaults) >= 1, "Expected at least one default"

        # The default on day 1 should reset no_defaults counter, so
        # simulation shouldn't stop at day 2 (day 0 quiet + day 1 mixed)
        assert len(reports) > 2, "Mixed day with defaults should prevent premature stability"


# =============================================================================
# run_day Integration: Dealer Phase (disabled path)
# =============================================================================


class TestRunDayDealerDisabled:
    """Tests for run_day when dealer is disabled or no dealer_subsystem."""

    def test_enable_dealer_false(self):
        """enable_dealer=False skips dealer trading phase."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)

        run_day(sys, enable_dealer=False)
        assert sys.state.day == 1

        # No SubphaseB_Dealer event
        dealer_events = [e for e in sys.state.events if e.get("kind") == "SubphaseB_Dealer"]
        assert len(dealer_events) == 0

    def test_enable_dealer_true_no_subsystem(self):
        """enable_dealer=True but no dealer_subsystem still skips gracefully."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)

        assert sys.state.dealer_subsystem is None

        run_day(sys, enable_dealer=True)
        assert sys.state.day == 1

        # No SubphaseB_Dealer event since dealer_subsystem is None
        dealer_events = [e for e in sys.state.events if e.get("kind") == "SubphaseB_Dealer"]
        assert len(dealer_events) == 0


# =============================================================================
# run_day: Rollover Path Tests
# =============================================================================


class TestRunDayRollover:
    """Tests for the rollover path in run_day."""

    def test_rollover_disabled_no_rollover_phase(self):
        """With rollover_enabled=False, no SubphaseB_Rollover is logged."""
        sys = System()
        sys.state.rollover_enabled = False
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        sys.mint_cash("H1", 500)

        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
        )
        sys.add_contract(payable)

        run_day(sys)

        rollover_events = [e for e in sys.state.events if e.get("kind") == "SubphaseB_Rollover"]
        assert len(rollover_events) == 0

    def test_rollover_enabled_creates_rollover_phase(self):
        """With rollover_enabled=True and settled payables, SubphaseB_Rollover fires."""
        sys = System()
        sys.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)

        sys.mint_cash("H1", 500)

        # maturity_distance must be set for rollover to produce rollover_info
        payable = Payable(
            id="P_ROLL",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
            maturity_distance=5,
        )
        sys.add_contract(payable)

        run_day(sys)

        rollover_events = [e for e in sys.state.events if e.get("kind") == "SubphaseB_Rollover"]
        assert len(rollover_events) == 1


# =============================================================================
# run_day: All phases fire in order
# =============================================================================


class TestRunDayPhaseOrder:
    """Verify that all phases execute in the correct order."""

    def test_phase_events_in_order(self):
        """PhaseA, PhaseB, SubphaseB1, SubphaseB2, PhaseC all appear in order."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)

        run_day(sys)

        phase_kinds = [
            e["kind"]
            for e in sys.state.events
            if e["kind"] in ("PhaseA", "PhaseB", "SubphaseB1", "SubphaseB2", "PhaseC")
        ]
        assert phase_kinds == ["PhaseA", "PhaseB", "SubphaseB1", "SubphaseB2", "PhaseC"]

    def test_day_incremented(self):
        """Day counter increments by 1 after run_day."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        sys.add_agent(cb)

        assert sys.state.day == 0
        run_day(sys)
        assert sys.state.day == 1
        run_day(sys)
        assert sys.state.day == 2


# =============================================================================
# Edge case: Scheduled action error re-raise (lines 171-174)
# =============================================================================


class TestScheduledActionErrors:
    """Tests for error handling in scheduled actions during Phase B1."""

    def test_invalid_scheduled_action_raises_value_error(self):
        """A scheduled action that causes ValueError is re-raised."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)

        # Schedule an action that will fail: mint_cash to a nonexistent agent
        # apply_action wraps the KeyError into a ValueError
        sys.state.scheduled_actions_by_day[0] = [
            {"mint_cash": {"to": "NONEXISTENT", "amount": 100}},
        ]

        with pytest.raises(ValueError, match="Failed to apply mint_cash"):
            run_day(sys)

    def test_invalid_scheduled_action_raises_attribute_error(self):
        """A scheduled action dict with bad structure raises AttributeError."""
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)

        # Schedule a malformed action (unknown action type raises ValueError in parse_action)
        sys.state.scheduled_actions_by_day[0] = [
            {"bogus_action": {"foo": "bar"}},
        ]

        with pytest.raises(ValueError):
            run_day(sys)


# =============================================================================
# Edge case: CB loan vanishes between get_cb_loans_due and lookup (line 211)
# =============================================================================


class TestCBLoanVanishes:
    """Test the guard for loan disappearing between get_cb_loans_due and .get()."""

    def test_loan_none_guard_continues(self):
        """If a CB loan is removed before the loop body runs, the continue guard fires."""
        sys = System()
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        sys.add_agent(cb)
        sys.add_agent(bank)

        # Create a CB loan on day 0 (matures day 2)
        loan_id = sys.cb_lend_reserves("B1", 1000, day=0)
        sys.mint_reserves("B1", 50)

        # Monkey-patch get_cb_loans_due to return a stale loan_id
        # that has already been removed
        original_get_due = sys.get_cb_loans_due

        def patched_get_due(day):
            result = original_get_due(day)
            # Remove the loan from contracts before the loop body runs
            if result:
                for lid in result:
                    if lid in sys.state.contracts:
                        loan = sys.state.contracts[lid]
                        # Manually remove the loan to simulate a race/edge case
                        sys.state.agents[loan.asset_holder_id].asset_ids.remove(lid)
                        sys.state.agents[loan.liability_issuer_id].liability_ids.remove(lid)
                        # Maintain cb_loans_outstanding
                        sys.state.cb_loans_outstanding -= loan.amount
                        del sys.state.contracts[lid]
            return result

        sys.get_cb_loans_due = patched_get_due

        sys.state.day = 2
        # Should NOT raise even though the loan is gone when the loop body runs
        run_day(sys)
        assert sys.state.day == 3

        # Loan should be gone
        assert loan_id not in sys.state.contracts
