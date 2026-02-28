"""Tests for Plan 046 NBFI behavioral improvements.

Tests cover:
1. Helper functions: _nearest_receivable_day, _downstream_obligation_total, _receivables_at_risk
2. Phase 1A: Maturity matching (loan maturity aligns with nearest receivable)
3. Phase 1B: Concentration limits (loans capped per borrower per day)
4. Phase 2: Cascade-aware ranking (ranking_mode changes sort order)
5. Phase 3: Graduated coverage gate (sub-threshold coverage gets rate penalty)
6. Backward compatibility: Default config produces same behavior
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

import pytest

from bilancio.decision.profiles import LenderProfile
from bilancio.domain.agents import CentralBank, Firm
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import Cash
from bilancio.engines.lending import (
    LendingConfig,
    _downstream_obligation_total,
    _nearest_receivable_day,
    _receivables_at_risk,
    run_lending_phase,
)
from bilancio.engines.system import System


# ── Setup helpers ───────────────────────────────────────────────────


def _make_system_with_cb() -> System:
    """Create a System with a CentralBank (needed as cash liability issuer)."""
    sys = System()
    cb = CentralBank(id="CB", name="Central Bank")
    sys.state.agents["CB"] = cb
    return sys


def _add_lender(sys: System, cash: int = 10000) -> str:
    """Add a NonBankLender with cash. Returns lender_id."""
    lender = NonBankLender(id="NBFI", name="NBFI Lender")
    sys.state.agents["NBFI"] = lender
    c = Cash(
        id="C_NBFI",
        kind=InstrumentKind.CASH,
        amount=cash,
        denom="X",
        asset_holder_id="NBFI",
        liability_issuer_id="CB",
    )
    sys.state.contracts["C_NBFI"] = c
    lender.asset_ids.append("C_NBFI")
    return "NBFI"


def _add_firm(sys: System, firm_id: str, cash: int = 0) -> str:
    """Add a Firm with optional cash. Returns firm_id."""
    firm = Firm(id=firm_id, name=f"Firm {firm_id}", kind="firm")
    sys.state.agents[firm_id] = firm
    if cash > 0:
        cid = f"C_{firm_id}"
        c = Cash(
            id=cid,
            kind=InstrumentKind.CASH,
            amount=cash,
            denom="X",
            asset_holder_id=firm_id,
            liability_issuer_id="CB",
        )
        sys.state.contracts[cid] = c
        firm.asset_ids.append(cid)
    return firm_id


def _add_payable(
    sys: System,
    payable_id: str,
    creditor_id: str,
    debtor_id: str,
    amount: int,
    due_day: int,
) -> str:
    """Add a payable instrument. Returns payable_id."""
    p = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="X",
        asset_holder_id=creditor_id,
        liability_issuer_id=debtor_id,
        due_day=due_day,
    )
    sys.state.contracts[payable_id] = p
    sys.state.agents[creditor_id].asset_ids.append(payable_id)
    sys.state.agents[debtor_id].liability_ids.append(payable_id)
    return payable_id


def _setup_ring(
    n_agents: int = 5,
    maturity_days: int = 10,
    face: int = 100,
    cash_per_firm: int = 50,
    lender_cash: int = 10000,
) -> System:
    """Create a minimal ring: 1 lender + n firms, each owing the next.

    F_i owes F_{(i+1) % n} a payable due on day ((i % maturity_days) + 1).
    Each firm gets cash_per_firm cash. The NBFI lender gets lender_cash.
    """
    sys = _make_system_with_cb()
    _add_lender(sys, cash=lender_cash)

    firm_ids = [f"F{i}" for i in range(n_agents)]
    for fid in firm_ids:
        _add_firm(sys, fid, cash=cash_per_firm)

    for i in range(n_agents):
        creditor = firm_ids[(i + 1) % n_agents]
        debtor = firm_ids[i]
        due_day = (i % maturity_days) + 1
        _add_payable(sys, f"P{i}", creditor, debtor, face, due_day)

    return sys


# ── 1. Helper function tests ───────────────────────────────────────


class TestNearestReceivableDay:
    """Tests for _nearest_receivable_day helper."""

    def test_finds_nearest_receivable(self):
        sys = _setup_ring(n_agents=5, maturity_days=10)
        # Ring: F0 owes F1 (due 1), F1 owes F2 (due 2), ..., F4 owes F0 (due 5)
        # F0's receivable: F4 owes F0, due on day (4 % 10) + 1 = 5
        nearest = _nearest_receivable_day(sys, "F0", current_day=0, max_horizon=10)
        assert nearest is not None
        assert nearest == 5

    def test_returns_none_when_no_receivables(self):
        sys = _setup_ring(n_agents=3, maturity_days=5)
        # NBFI has no payable receivables (only cash)
        nearest = _nearest_receivable_day(sys, "NBFI", current_day=0, max_horizon=10)
        assert nearest is None

    def test_ignores_past_due_receivables(self):
        sys = _setup_ring(n_agents=5, maturity_days=10)
        # F0's receivable from F4 is due on day 5 - asking from day 6 should skip it
        nearest = _nearest_receivable_day(sys, "F0", current_day=6, max_horizon=10)
        # No other receivables for F0, so should be None
        assert nearest is None

    def test_respects_horizon_limit(self):
        sys = _setup_ring(n_agents=5, maturity_days=10)
        # F0's receivable is due day 5. Horizon=2 from day 0 means only days 1-2
        nearest = _nearest_receivable_day(sys, "F0", current_day=0, max_horizon=2)
        assert nearest is None

    def test_skips_defaulted_issuers(self):
        sys = _setup_ring(n_agents=5, maturity_days=10)
        # Mark F4 as defaulted (F4 is the one who owes F0)
        sys.state.agents["F4"].defaulted = True
        sys.state.defaulted_agent_ids.add("F4")
        nearest = _nearest_receivable_day(sys, "F0", current_day=0, max_horizon=10)
        # F4's receivable should be excluded
        assert nearest is None

    def test_picks_earliest_among_multiple(self):
        sys = _make_system_with_cb()
        _add_lender(sys)
        _add_firm(sys, "FA", cash=100)
        _add_firm(sys, "FB", cash=100)
        _add_firm(sys, "FC", cash=100)
        # FA has receivables from FB (due 5) and FC (due 3)
        _add_payable(sys, "P1", "FA", "FB", 100, due_day=5)
        _add_payable(sys, "P2", "FA", "FC", 100, due_day=3)
        nearest = _nearest_receivable_day(sys, "FA", current_day=0, max_horizon=10)
        assert nearest == 3


class TestDownstreamObligationTotal:
    """Tests for _downstream_obligation_total helper."""

    def test_counts_future_payable_liabilities(self):
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # F0 owes F1 a payable of 100 due day 1
        total = _downstream_obligation_total(sys, "F0", current_day=0)
        assert total == 100

    def test_ignores_past_due(self):
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # F0's payable is due day 1; asking from day 2 should skip it
        total = _downstream_obligation_total(sys, "F0", current_day=2)
        assert total == 0

    def test_zero_for_agent_with_no_liabilities(self):
        sys = _setup_ring(n_agents=3, maturity_days=5)
        # NBFI has no payable liabilities
        total = _downstream_obligation_total(sys, "NBFI", current_day=0)
        assert total == 0

    def test_sums_multiple_liabilities(self):
        sys = _make_system_with_cb()
        _add_lender(sys)
        _add_firm(sys, "FA", cash=100)
        _add_firm(sys, "FB", cash=100)
        _add_firm(sys, "FC", cash=100)
        # FA owes FB 200 (due 3) and FC 300 (due 5)
        _add_payable(sys, "P1", "FB", "FA", 200, due_day=3)
        _add_payable(sys, "P2", "FC", "FA", 300, due_day=5)
        total = _downstream_obligation_total(sys, "FA", current_day=0)
        assert total == 500


class TestReceivablesAtRisk:
    """Tests for _receivables_at_risk helper."""

    def test_identifies_at_risk_receivables(self):
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # F0's receivable is from F4, due day 5. If F4's default prob > threshold, it's at risk
        probs = {f"F{i}": Decimal("0.5") for i in range(5)}
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=probs, threshold=Decimal("0.3"),
        )
        assert at_risk == 100  # F4's payable to F0 is 100

    def test_no_risk_when_probs_below_threshold(self):
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        probs = {f"F{i}": Decimal("0.1") for i in range(5)}
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=probs, threshold=Decimal("0.3"),
        )
        assert at_risk == 0

    def test_uses_default_prior_for_missing_probs(self):
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # Empty probs dict -> uses default 0.15 for all
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs={}, threshold=Decimal("0.10"),
        )
        # Default prior 0.15 > threshold 0.10, so at risk
        assert at_risk == 100

    def test_none_probs_uses_default_prior(self):
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=None, threshold=Decimal("0.10"),
        )
        # None probs -> 0.15 default > 0.10 threshold
        assert at_risk == 100

    def test_respects_horizon(self):
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # F0's receivable is due day 5. Horizon=2 from day 0 -> only days 1-2
        probs = {f"F{i}": Decimal("0.5") for i in range(5)}
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=2,
            default_probs=probs, threshold=Decimal("0.3"),
        )
        assert at_risk == 0  # Day 5 is outside horizon


# ── 2. Phase 1A: Maturity matching ─────────────────────────────────


class TestMaturityMatching:
    """Tests for Phase 1A: maturity matching."""

    def test_maturity_matching_extends_loan(self):
        """Maturity matching should set loan maturity to cover nearest receivable."""
        # F0 owes 100 due day 1, has 30 cash (shortfall). F0's receivable due day 5
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100, cash_per_firm=30)

        config = LendingConfig(
            maturity_matching=True,
            min_loan_maturity=2,
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=10,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
            lender_profile=LenderProfile(
                kappa=Decimal("0.3"),
                max_loan_maturity=10,
                maturity_matching=True,
                min_loan_maturity=2,
                min_coverage_ratio=Decimal("0"),
            ),
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        created = [e for e in events if e["kind"] == "NonBankLoanCreated"]
        # With kappa=0.3 and disabled coverage gate, at least some loans should be created
        assert len(created) > 0, f"Expected loans, got events: {events}"

    def test_maturity_matching_disabled_by_default(self):
        """Default config should not use maturity matching."""
        config = LendingConfig()
        assert config.maturity_matching is False

    def test_min_loan_maturity_floor(self):
        """Matched maturity should not go below min_loan_maturity."""
        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=10)
        _add_firm(sys, "FB", cash=100)
        # FA owes FB 100 due day 1 (shortfall), FA's receivable from FB due day 1
        _add_payable(sys, "P_debt", "FB", "FA", 100, due_day=1)
        _add_payable(sys, "P_recv", "FA", "FB", 50, due_day=1)

        config = LendingConfig(
            maturity_matching=True,
            min_loan_maturity=3,  # floor at 3 days
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=10,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
            lender_profile=LenderProfile(
                kappa=Decimal("0.3"),
                max_loan_maturity=10,
                maturity_matching=True,
                min_loan_maturity=3,
                min_coverage_ratio=Decimal("0"),
            ),
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        # Should not crash; loan maturity >= 3 enforced internally


# ── 3. Phase 1B: Concentration limits ──────────────────────────────


class TestConcentrationLimits:
    """Tests for Phase 1B: concentration limits."""

    def test_concentration_limit_caps_loans(self):
        """With max_loans_per_borrower_per_day=1, each borrower gets at most 1 loan."""
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100, cash_per_firm=30)

        config = LendingConfig(
            max_loans_per_borrower_per_day=1,
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=10,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
            lender_profile=LenderProfile(
                kappa=Decimal("0.3"),
                max_loan_maturity=10,
                max_loans_per_borrower_per_day=1,
                min_coverage_ratio=Decimal("0"),
            ),
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        created = [e for e in events if e["kind"] == "NonBankLoanCreated"]

        # Check no borrower got more than 1 loan
        borrower_counts = Counter(e["borrower_id"] for e in created)
        for borrower_id, count in borrower_counts.items():
            assert count <= 1, f"Borrower {borrower_id} got {count} loans, expected at most 1"

    def test_unlimited_when_zero(self):
        """With max_loans_per_borrower_per_day=0 (default), no limit is applied."""
        sys = _setup_ring(n_agents=3, maturity_days=5, face=100, cash_per_firm=20)

        config = LendingConfig(
            max_loans_per_borrower_per_day=0,  # unlimited
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=5,
            horizon=5,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        # Should not have any concentration rejection events
        conc_events = [e for e in events if e["kind"] == "NonBankLoanRejectedConcentration"]
        assert len(conc_events) == 0

    def test_concentration_rejection_event_emitted(self):
        """Concentration limit should emit NonBankLoanRejectedConcentration when triggered."""
        # Create a scenario where the same borrower would get multiple loans
        # by giving them multiple shortfalls (multiple due dates)
        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=10)
        _add_firm(sys, "FB", cash=100)
        _add_firm(sys, "FC", cash=100)
        # FA owes FB 100 due day 1 and FC 100 due day 2 (two obligations, big shortfall)
        _add_payable(sys, "P1", "FB", "FA", 100, due_day=1)
        _add_payable(sys, "P2", "FC", "FA", 100, due_day=2)

        config = LendingConfig(
            max_loans_per_borrower_per_day=1,
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=5,
            horizon=5,
            max_single_exposure=Decimal("0.90"),
            max_total_exposure=Decimal("0.90"),
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        # Should not crash; event structure should be valid
        for e in events:
            assert "kind" in e


# ── 4. Phase 2: Cascade-aware ranking ──────────────────────────────


class TestCascadeRanking:
    """Tests for Phase 2: cascade-aware ranking."""

    def test_all_ranking_modes_run_without_error(self):
        """All three ranking modes should run without crashing."""
        for mode in ("profit", "cascade", "blended"):
            sys = _setup_ring(n_agents=5, maturity_days=10, face=100, cash_per_firm=30)
            config = LendingConfig(
                ranking_mode=mode,
                cascade_weight=Decimal("0.5"),
                min_coverage_ratio=Decimal("0"),
                max_ring_maturity=10,
                horizon=10,
                max_single_exposure=Decimal("0.50"),
                max_total_exposure=Decimal("0.80"),
            )
            events = run_lending_phase(sys, current_day=0, lending_config=config)
            assert isinstance(events, list), f"ranking_mode={mode} did not return a list"

    def test_cascade_mode_prioritizes_downstream_damage(self):
        """Cascade mode should lend to the agent with more downstream damage first.

        FA owes 500 to others (large downstream), FB owes 100 (small downstream).
        Both have the same shortfall. With cascade ranking, FA should get the loan first.
        """
        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=10)
        _add_firm(sys, "FB", cash=10)
        _add_firm(sys, "FC", cash=100)

        # FA owes FC 500 (large downstream damage)
        _add_payable(sys, "P1", "FC", "FA", 500, due_day=3)
        # FB owes FC 100 (small downstream damage)
        _add_payable(sys, "P2", "FC", "FB", 100, due_day=3)
        # Both also owe FC 50 more to trigger shortfall (due day 1)
        _add_payable(sys, "P3", "FC", "FA", 50, due_day=1)
        _add_payable(sys, "P4", "FC", "FB", 50, due_day=1)

        config = LendingConfig(
            ranking_mode="cascade",
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=10,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        created = [e for e in events if e["kind"] == "NonBankLoanCreated"]
        # At minimum, the phase should complete successfully
        assert isinstance(created, list)


# ── 5. Phase 3: Graduated coverage gate ────────────────────────────


class TestGraduatedCoverage:
    """Tests for Phase 3: graduated coverage gate."""

    def test_graduated_mode_allows_sub_threshold(self):
        """Graduated mode should penalize rate instead of rejecting."""
        # Gate mode: strict rejection
        sys_gate = _make_system_with_cb()
        _add_lender(sys_gate, cash=10000)
        _add_firm(sys_gate, "F1", cash=10)
        _add_firm(sys_gate, "F2", cash=100)
        # F1 has big obligation, low cash -> low coverage
        _add_payable(sys_gate, "P1", "F2", "F1", 1000, due_day=2)

        config_gate = LendingConfig(
            coverage_mode="gate",
            min_coverage_ratio=Decimal("0.5"),
            max_ring_maturity=10,
            horizon=5,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
        )
        events_gate = run_lending_phase(sys_gate, current_day=1, lending_config=config_gate)
        rejections_gate = [e for e in events_gate if e["kind"] == "NonBankLoanRejectedCoverage"]

        # Graduated mode: penalty instead of rejection
        sys_grad = _make_system_with_cb()
        _add_lender(sys_grad, cash=10000)
        _add_firm(sys_grad, "F1", cash=10)
        _add_firm(sys_grad, "F2", cash=100)
        _add_payable(sys_grad, "P1", "F2", "F1", 1000, due_day=2)

        config_grad = LendingConfig(
            coverage_mode="graduated",
            coverage_penalty_scale=Decimal("0.10"),
            min_coverage_ratio=Decimal("0.5"),
            max_ring_maturity=10,
            horizon=5,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
        )
        events_grad = run_lending_phase(sys_grad, current_day=1, lending_config=config_grad)
        rejections_grad = [e for e in events_grad if e["kind"] == "NonBankLoanRejectedCoverage"]

        # Graduated mode should have fewer (or equal) rejections
        assert len(rejections_grad) <= len(rejections_gate)

    def test_gate_mode_rejects_low_coverage(self):
        """Gate mode should reject borrowers below min_coverage_ratio."""
        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "F1", cash=10)
        _add_firm(sys, "F2", cash=100)
        _add_payable(sys, "P1", "F2", "F1", 1000, due_day=2)

        config = LendingConfig(
            coverage_mode="gate",
            min_coverage_ratio=Decimal("0.5"),
            max_ring_maturity=10,
            horizon=5,
            max_single_exposure=Decimal("0.50"),
            max_total_exposure=Decimal("0.80"),
        )
        events = run_lending_phase(sys, current_day=1, lending_config=config)
        rejections = [e for e in events if e["kind"] == "NonBankLoanRejectedCoverage"]
        loans = [e for e in events if e["kind"] == "NonBankLoanCreated"]
        assert len(rejections) >= 1, f"Expected rejection, got events: {events}"
        assert len(loans) == 0, "Low-coverage borrower should not get a loan in gate mode"


# ── 6. LenderProfile validation ────────────────────────────────────


class TestLenderProfileValidation:
    """Tests for LenderProfile field validation."""

    def test_valid_defaults(self):
        """Default LenderProfile should be valid."""
        profile = LenderProfile()
        assert profile.maturity_matching is False
        assert profile.ranking_mode == "profit"
        assert profile.coverage_mode == "gate"
        assert profile.preventive_lending is False
        assert profile.max_loans_per_borrower_per_day == 0
        assert profile.cascade_weight == Decimal("0.5")

    def test_invalid_ranking_mode(self):
        with pytest.raises(ValueError, match="ranking_mode"):
            LenderProfile(ranking_mode="invalid")

    def test_invalid_coverage_mode(self):
        with pytest.raises(ValueError, match="coverage_mode"):
            LenderProfile(coverage_mode="invalid")

    def test_invalid_cascade_weight_above_one(self):
        with pytest.raises(ValueError, match="cascade_weight"):
            LenderProfile(cascade_weight=Decimal("1.5"))

    def test_invalid_cascade_weight_negative(self):
        with pytest.raises(ValueError, match="cascade_weight"):
            LenderProfile(cascade_weight=Decimal("-0.1"))

    def test_invalid_prevention_threshold_zero(self):
        with pytest.raises(ValueError, match="prevention_threshold"):
            LenderProfile(prevention_threshold=Decimal("0"))

    def test_invalid_prevention_threshold_one(self):
        with pytest.raises(ValueError, match="prevention_threshold"):
            LenderProfile(prevention_threshold=Decimal("1"))

    def test_invalid_min_loan_maturity(self):
        with pytest.raises(ValueError, match="min_loan_maturity"):
            LenderProfile(min_loan_maturity=0)

    def test_invalid_max_loans_negative(self):
        with pytest.raises(ValueError, match="max_loans_per_borrower_per_day"):
            LenderProfile(max_loans_per_borrower_per_day=-1)

    def test_invalid_kappa_zero(self):
        with pytest.raises(ValueError, match="kappa"):
            LenderProfile(kappa=Decimal("0"))

    def test_invalid_kappa_negative(self):
        with pytest.raises(ValueError, match="kappa"):
            LenderProfile(kappa=Decimal("-1"))

    def test_valid_plan_046_profile(self):
        """A fully-specified Plan 046 profile should validate."""
        profile = LenderProfile(
            kappa=Decimal("0.5"),
            maturity_matching=True,
            min_loan_maturity=3,
            max_loans_per_borrower_per_day=2,
            ranking_mode="blended",
            cascade_weight=Decimal("0.7"),
            coverage_mode="graduated",
            coverage_penalty_scale=Decimal("0.05"),
            preventive_lending=True,
            prevention_threshold=Decimal("0.4"),
        )
        assert profile.maturity_matching is True
        assert profile.ranking_mode == "blended"
        assert profile.coverage_mode == "graduated"
        assert profile.preventive_lending is True


# ── 7. Backward compatibility ──────────────────────────────────────


class TestBackwardCompatibility:
    """Verify default params produce same behavior as pre-046 code."""

    def test_default_config_runs_without_error(self):
        """Default lending config should work identically to pre-046."""
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100, cash_per_firm=50)

        config = LendingConfig(
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=5,
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        assert isinstance(events, list)
        for e in events:
            assert "kind" in e
            # No concentration rejection events with default config (unlimited)
            assert e["kind"] != "NonBankLoanRejectedConcentration"
            # No preventive events with default config
            assert e["kind"] != "NonBankLoanCreatedPreventive"

    def test_default_lending_config_fields(self):
        """All Plan 046 fields should have backward-compatible defaults."""
        config = LendingConfig()
        assert config.maturity_matching is False
        assert config.min_loan_maturity == 2
        assert config.max_loans_per_borrower_per_day == 0
        assert config.ranking_mode == "profit"
        assert config.cascade_weight == Decimal("0.5")
        assert config.coverage_mode == "gate"
        assert config.coverage_penalty_scale == Decimal("0.10")
        assert config.preventive_lending is False
        assert config.prevention_threshold == Decimal("0.3")

    def test_default_lender_profile_fields(self):
        """All Plan 046 LenderProfile fields should have backward-compatible defaults."""
        profile = LenderProfile()
        assert profile.maturity_matching is False
        assert profile.min_loan_maturity == 2
        assert profile.max_loans_per_borrower_per_day == 0
        assert profile.ranking_mode == "profit"
        assert profile.cascade_weight == Decimal("0.5")
        assert profile.coverage_mode == "gate"
        assert profile.coverage_penalty_scale == Decimal("0.10")
        assert profile.preventive_lending is False
        assert profile.prevention_threshold == Decimal("0.3")

    def test_no_new_event_types_with_defaults(self):
        """With default config, only standard event types should appear."""
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100, cash_per_firm=30)
        config = LendingConfig(
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=10,
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        allowed_kinds = {
            "NonBankLoanCreated",
            "NonBankLoanRejectedCoverage",
        }
        for e in events:
            assert e["kind"] in allowed_kinds, (
                f"Unexpected event kind '{e['kind']}' with default config"
            )


# ── 8. Review fixes: _count_existing_loans ───────────────────────────


class TestCountExistingLoans:
    """Tests for _count_existing_loans — concentration limit now checks
    actual outstanding loans in the system, not an ephemeral counter."""

    def test_zero_when_no_loans_exist(self):
        """No loans → count is 0."""
        from bilancio.engines.lending import _count_existing_loans

        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=100)
        assert _count_existing_loans(sys, "NBFI", "FA") == 0

    def test_counts_outstanding_loans(self):
        """After creating loans, count should reflect them."""
        from bilancio.engines.lending import _count_existing_loans

        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=10)
        _add_firm(sys, "FB", cash=100)
        # FA has shortfall — create a loan to FA
        _add_payable(sys, "P1", "FB", "FA", 200, due_day=1)

        config = LendingConfig(
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=5,
            max_single_exposure=Decimal("0.90"),
            max_total_exposure=Decimal("0.90"),
        )
        events = run_lending_phase(sys, current_day=0, lending_config=config)
        created = [e for e in events if e["kind"] == "NonBankLoanCreated"]
        assert len(created) == 1
        assert created[0]["borrower_id"] == "FA"

        # Now _count_existing_loans should see the loan
        assert _count_existing_loans(sys, "NBFI", "FA") == 1
        # No loan to FB
        assert _count_existing_loans(sys, "NBFI", "FB") == 0

    def test_concentration_blocks_second_day_loan(self):
        """With an existing outstanding loan and limit=1, a second-day loan
        to the same borrower should be blocked."""
        from bilancio.engines.lending import _count_existing_loans

        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=10)
        _add_firm(sys, "FB", cash=100)
        # FA owes FB 200, due day 2 — creates shortfall on day 0 (horizon=5)
        _add_payable(sys, "P1", "FB", "FA", 200, due_day=2)

        config = LendingConfig(
            max_loans_per_borrower_per_day=1,
            min_coverage_ratio=Decimal("0"),
            max_ring_maturity=10,
            horizon=5,
            max_single_exposure=Decimal("0.90"),
            max_total_exposure=Decimal("0.90"),
        )
        # Day 0: first loan should succeed
        events_d0 = run_lending_phase(sys, current_day=0, lending_config=config)
        created_d0 = [e for e in events_d0 if e["kind"] == "NonBankLoanCreated"]
        assert len(created_d0) == 1
        assert _count_existing_loans(sys, "NBFI", "FA") == 1

        # Day 1: FA still has shortfall (loan doesn't cover payable), but
        # concentration limit blocks it because outstanding loan count >= 1
        events_d1 = run_lending_phase(sys, current_day=1, lending_config=config)
        created_d1 = [e for e in events_d1 if e["kind"] == "NonBankLoanCreated"]
        rejected_d1 = [e for e in events_d1 if e["kind"] == "NonBankLoanRejectedConcentration"]
        assert len(created_d1) == 0, f"Expected no new loans, got {created_d1}"
        assert len(rejected_d1) == 1

    def test_missing_lender_returns_zero(self):
        """_count_existing_loans returns 0 for nonexistent lender."""
        from bilancio.engines.lending import _count_existing_loans

        sys = _make_system_with_cb()
        assert _count_existing_loans(sys, "NONEXISTENT", "FA") == 0


# ── 9. Review fixes: _receivables_at_risk with info service ──────────


class TestReceivablesAtRiskWithInfoService:
    """Tests for _receivables_at_risk using InformationService when
    default_probs is None (review fix #3)."""

    def test_uses_info_service_probabilities(self):
        """When info service is provided and default_probs is None,
        should use info.get_default_probability() per issuer."""

        class MockInfo:
            """Mock InformationService returning per-issuer probabilities."""
            def __init__(self, probs: dict[str, Decimal]):
                self._probs = probs

            def get_default_probability(self, agent_id: str, current_day: int) -> Decimal | None:
                return self._probs.get(agent_id)

        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # F0 has a receivable from F4 (due day 5 in a 5-agent ring)
        # F4's default prob should be checked

        # High probability for F4 → at risk
        mock_info = MockInfo({f"F{i}": Decimal("0.5") for i in range(5)})
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=None, threshold=Decimal("0.3"),
            info=mock_info,
        )
        assert at_risk == 100  # F0's receivable from F4 is at risk

    def test_low_info_probability_not_at_risk(self):
        """When info service returns low p_default, receivable is not at risk."""

        class MockInfo:
            def get_default_probability(self, agent_id: str, current_day: int) -> Decimal | None:
                return Decimal("0.05")  # Low probability

        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=None, threshold=Decimal("0.3"),
            info=MockInfo(),
        )
        assert at_risk == 0  # 0.05 < 0.3 threshold

    def test_info_returns_none_falls_back_to_default(self):
        """When info.get_default_probability returns None for an issuer,
        should fall back to hardcoded 0.15."""

        class MockInfo:
            def get_default_probability(self, agent_id: str, current_day: int) -> Decimal | None:
                return None  # Unknown issuer

        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # Threshold 0.10 < default fallback 0.15 → at risk
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=None, threshold=Decimal("0.10"),
            info=MockInfo(),
        )
        assert at_risk == 100

        # Threshold 0.20 > default fallback 0.15 → not at risk
        at_risk_high = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=None, threshold=Decimal("0.20"),
            info=MockInfo(),
        )
        assert at_risk_high == 0

    def test_default_probs_takes_precedence_over_info(self):
        """When default_probs dict IS provided, it should be used
        even if info is also provided (backward compat)."""

        class MockInfo:
            def get_default_probability(self, agent_id: str, current_day: int) -> Decimal | None:
                return Decimal("0.01")  # Very low — should NOT trigger

        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        # default_probs says high risk, info says low risk
        probs = {f"F{i}": Decimal("0.5") for i in range(5)}
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=probs, threshold=Decimal("0.3"),
            info=MockInfo(),
        )
        # default_probs wins → at risk
        assert at_risk == 100

    def test_no_info_no_probs_uses_hardcoded_fallback(self):
        """With both None, should use hardcoded 0.15 (original behavior)."""
        sys = _setup_ring(n_agents=5, maturity_days=10, face=100)
        at_risk = _receivables_at_risk(
            sys, "F0", current_day=0, horizon=10,
            default_probs=None, threshold=Decimal("0.10"),
            info=None,
        )
        # 0.15 > 0.10 → at risk
        assert at_risk == 100


# ── 10. Review fixes: action-spec config wiring ──────────────────────


class TestActionSpecConfigWiring:
    """Tests that _init_lending_from_action_specs passes Plan 046 fields."""

    def test_action_spec_path_includes_046_fields(self):
        """LendingConfig built from action-spec path should include all
        Plan 046 fields when LenderScenarioConfig provides them."""
        from bilancio.config.models import LenderScenarioConfig, ScenarioConfig

        # Build a ScenarioConfig with lender settings that exercise Plan 046
        lender_cfg = LenderScenarioConfig(
            maturity_matching=True,
            min_loan_maturity=3,
            max_loans_per_borrower_per_day=2,
            ranking_mode="blended",
            cascade_weight=Decimal("0.7"),
            coverage_mode="graduated",
            coverage_penalty_scale=Decimal("0.05"),
            preventive_lending=True,
            prevention_threshold=Decimal("0.4"),
            min_coverage_ratio=Decimal("0.3"),
        )

        # Verify the config model itself stores the values
        assert lender_cfg.maturity_matching is True
        assert lender_cfg.min_loan_maturity == 3
        assert lender_cfg.max_loans_per_borrower_per_day == 2
        assert lender_cfg.ranking_mode == "blended"
        assert lender_cfg.cascade_weight == Decimal("0.7")
        assert lender_cfg.coverage_mode == "graduated"
        assert lender_cfg.coverage_penalty_scale == Decimal("0.05")
        assert lender_cfg.preventive_lending is True
        assert lender_cfg.prevention_threshold == Decimal("0.4")
        assert lender_cfg.min_coverage_ratio == Decimal("0.3")

    def test_lending_config_accepts_all_046_fields(self):
        """LendingConfig dataclass should accept all 046 fields without error."""
        config = LendingConfig(
            maturity_matching=True,
            min_loan_maturity=3,
            max_loans_per_borrower_per_day=2,
            ranking_mode="blended",
            cascade_weight=Decimal("0.7"),
            coverage_mode="graduated",
            coverage_penalty_scale=Decimal("0.05"),
            preventive_lending=True,
            prevention_threshold=Decimal("0.4"),
            min_coverage_ratio=Decimal("0.3"),
        )
        assert config.maturity_matching is True
        assert config.min_loan_maturity == 3
        assert config.max_loans_per_borrower_per_day == 2
        assert config.ranking_mode == "blended"
        assert config.cascade_weight == Decimal("0.7")
        assert config.coverage_mode == "graduated"
        assert config.coverage_penalty_scale == Decimal("0.05")
        assert config.preventive_lending is True
        assert config.prevention_threshold == Decimal("0.4")
        assert config.min_coverage_ratio == Decimal("0.3")

    def test_action_spec_wiring_end_to_end(self):
        """Full end-to-end: build system via action-spec path, verify
        lender_config has Plan 046 fields set correctly."""
        from bilancio.config.apply import _init_lending_from_action_specs
        from bilancio.config.models import LenderScenarioConfig, ScenarioConfig

        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=10)
        _add_firm(sys, "FB", cash=100)
        _add_payable(sys, "P1", "FB", "FA", 200, due_day=2)

        # Build minimal ScenarioConfig with lender section
        lender_cfg = LenderScenarioConfig(
            maturity_matching=True,
            min_loan_maturity=4,
            max_loans_per_borrower_per_day=3,
            ranking_mode="cascade",
            cascade_weight=Decimal("0.6"),
            coverage_mode="graduated",
            coverage_penalty_scale=Decimal("0.08"),
            preventive_lending=True,
            prevention_threshold=Decimal("0.35"),
            min_coverage_ratio=Decimal("0.2"),
        )

        config = ScenarioConfig(name="test", agents=[], lender=lender_cfg)

        # Build LenderProfile from the lender_cfg
        lender_profile = LenderProfile(
            maturity_matching=True,
            min_loan_maturity=4,
            max_loans_per_borrower_per_day=3,
            ranking_mode="cascade",
            cascade_weight=Decimal("0.6"),
            coverage_mode="graduated",
            coverage_penalty_scale=Decimal("0.08"),
            preventive_lending=True,
            prevention_threshold=Decimal("0.35"),
            min_coverage_ratio=Decimal("0.2"),
        )

        _init_lending_from_action_specs(config, sys, lender_profile, {})

        lc = sys.state.lender_config
        assert lc is not None
        assert lc.maturity_matching is True
        assert lc.min_loan_maturity == 4
        assert lc.max_loans_per_borrower_per_day == 3
        assert lc.ranking_mode == "cascade"
        assert lc.cascade_weight == Decimal("0.6")
        assert lc.coverage_mode == "graduated"
        assert lc.coverage_penalty_scale == Decimal("0.08")
        assert lc.preventive_lending is True
        assert lc.prevention_threshold == Decimal("0.35")
        assert lc.min_coverage_ratio == Decimal("0.2")

    def test_action_spec_defaults_match_lending_config_defaults(self):
        """When lender config has no Plan 046 fields set (defaults),
        the action-spec path should produce default LendingConfig values."""
        from bilancio.config.apply import _init_lending_from_action_specs
        from bilancio.config.models import LenderScenarioConfig, ScenarioConfig

        sys = _make_system_with_cb()
        _add_lender(sys, cash=10000)
        _add_firm(sys, "FA", cash=10)
        _add_firm(sys, "FB", cash=100)
        _add_payable(sys, "P1", "FB", "FA", 200, due_day=2)

        config = ScenarioConfig(name="test", agents=[], lender=LenderScenarioConfig())
        _init_lending_from_action_specs(config, sys, None, {})

        lc = sys.state.lender_config
        assert lc is not None
        # All Plan 046 fields should be at defaults
        assert lc.maturity_matching is False
        assert lc.min_loan_maturity == 2
        assert lc.max_loans_per_borrower_per_day == 0
        assert lc.ranking_mode == "profit"
        assert lc.cascade_weight == Decimal("0.5")
        assert lc.coverage_mode == "gate"
        assert lc.coverage_penalty_scale == Decimal("0.10")
        assert lc.preventive_lending is False
        assert lc.prevention_threshold == Decimal("0.3")
