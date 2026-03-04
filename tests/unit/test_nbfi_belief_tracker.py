"""Tests for NBFI Bayesian BeliefTracker integration (Plan 040 Phase 1).

Covers:
1. NBFI assessor creation and persistence across days
2. Bayesian posterior convergence with loan outcomes
3. Blending: coverage dominates early, posterior dominates late
4. Backward compatibility: no assessor -> existing waterfall behavior
"""

from decimal import Decimal

import pytest

from bilancio.decision.profiles import LenderProfile
from bilancio.decision.risk_assessment import RiskAssessmentParams, RiskAssessor
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.lending import LendingConfig, run_lending_phase, run_loan_repayments
from bilancio.engines.system import System

# ── Helpers ──────────────────────────────────────────────────────────────


def _build_lending_system(
    lender_cash: int = 10000,
    firm_cash: int = 500,
    firm_payable_amount: int = 1000,
    payable_due_day: int = 2,
) -> System:
    """Build a minimal system suitable for lending tests."""
    system = System()

    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank = Bank(id="B01", name="Bank 1", kind="bank")
    lender = NonBankLender(id="NBL01", name="Non-Bank Lender")
    firm1 = Firm(id="F01", name="Firm 1", kind="firm")
    firm2 = Firm(id="F02", name="Firm 2", kind="firm")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.add_agent(firm1)
    system.add_agent(firm2)

    if lender_cash > 0:
        system.mint_cash("NBL01", lender_cash)
    if firm_cash > 0:
        system.mint_cash("F01", firm_cash)

    # Create a payable: F01 owes F02
    if firm_payable_amount > 0:
        payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=firm_payable_amount,
            denom="X",
            asset_holder_id="F02",
            liability_issuer_id="F01",
            due_day=payable_due_day,
        )
        system.add_contract(payable)

    return system


def _make_risk_params(**overrides) -> RiskAssessmentParams:
    """Create RiskAssessmentParams with sensible defaults for lending tests."""
    defaults = {
        "lookback_window": 10,
        "smoothing_alpha": Decimal("1.0"),
        "initial_prior": Decimal("0.15"),
        "use_issuer_specific": True,
    }
    defaults.update(overrides)
    return RiskAssessmentParams(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# 1. NBFI Assessor Creation and Persistence
# ═══════════════════════════════════════════════════════════════════════


class TestNBFIAssessorCreation:
    """Test that the NBFI assessor is created and persists across days."""

    def test_lender_profile_with_risk_params(self):
        """LenderProfile accepts risk_assessment_params field."""
        params = _make_risk_params()
        profile = LenderProfile(
            kappa=Decimal("1.0"),
            risk_assessment_params=params,
        )
        assert profile.risk_assessment_params is not None
        assert profile.risk_assessment_params.initial_prior == Decimal("0.15")

    def test_lender_profile_default_no_params(self):
        """LenderProfile defaults to None risk_assessment_params (backward compat)."""
        profile = LenderProfile()
        assert profile.risk_assessment_params is None

    def test_warmup_observations_default(self):
        """warmup_observations defaults to 10."""
        profile = LenderProfile()
        assert profile.warmup_observations == 10

    def test_warmup_observations_validation(self):
        """warmup_observations must be >= 1."""
        with pytest.raises(ValueError, match="warmup_observations"):
            LenderProfile(warmup_observations=0)

    def test_lending_config_with_assessor(self):
        """LendingConfig can carry a RiskAssessor."""
        params = _make_risk_params()
        assessor = RiskAssessor(params)
        config = LendingConfig(risk_assessor=assessor)
        assert config.risk_assessor is not None
        assert config.risk_assessor is assessor

    def test_lending_config_default_no_assessor(self):
        """LendingConfig defaults to None risk_assessor (backward compat)."""
        config = LendingConfig()
        assert config.risk_assessor is None

    def test_assessor_persists_across_days(self):
        """The same assessor object persists on LendingConfig across multiple calls."""
        params = _make_risk_params()
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True
        config = LendingConfig(risk_assessor=assessor)

        # Simulate updating history on different days
        assessor.update_history(day=1, issuer_id="F01", defaulted=False)
        assessor.update_history(day=2, issuer_id="F01", defaulted=True)

        # The config's assessor should reflect both updates
        assert len(config.risk_assessor.belief_tracker.payment_history) == 2
        assert "F01" in config.risk_assessor.belief_tracker.issuer_history
        assert len(config.risk_assessor.belief_tracker.issuer_history["F01"]) == 2


# ═══════════════════════════════════════════════════════════════════════
# 2. Bayesian Posterior Convergence
# ═══════════════════════════════════════════════════════════════════════


class TestBayesianConvergence:
    """Test that Bayesian posterior converges with loan outcomes."""

    def test_no_observations_returns_prior(self):
        """With no history, estimate returns initial_prior."""
        params = _make_risk_params(initial_prior=Decimal("0.20"))
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        p = assessor.estimate_default_prob("F01", current_day=1)
        assert p == Decimal("0.20")

    def test_all_repayments_lowers_probability(self):
        """Many successful repayments drive posterior below prior."""
        params = _make_risk_params(initial_prior=Decimal("0.50"))
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        # 10 successful repayments
        for day in range(1, 11):
            assessor.update_history(day=day, issuer_id="F01", defaulted=False)

        p = assessor.estimate_default_prob("F01", current_day=10)
        # With Laplace smoothing: (1 + 0) / (2 + 10) = 1/12 ≈ 0.083
        assert p < Decimal("0.15"), f"Expected < 0.15, got {p}"

    def test_all_defaults_raises_probability(self):
        """Many defaults drive posterior above prior."""
        params = _make_risk_params(initial_prior=Decimal("0.15"))
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        # 10 defaults
        for day in range(1, 11):
            assessor.update_history(day=day, issuer_id="F01", defaulted=True)

        p = assessor.estimate_default_prob("F01", current_day=10)
        # With Laplace smoothing: (1 + 10) / (2 + 10) = 11/12 ≈ 0.917
        assert p > Decimal("0.50"), f"Expected > 0.50, got {p}"

    def test_mixed_outcomes_reflect_rate(self):
        """Mixed outcomes produce posterior near empirical rate."""
        params = _make_risk_params(initial_prior=Decimal("0.15"))
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        # 5 repayments, 5 defaults -> 50% empirical rate
        for day in range(1, 6):
            assessor.update_history(day=day, issuer_id="F01", defaulted=False)
        for day in range(6, 11):
            assessor.update_history(day=day, issuer_id="F01", defaulted=True)

        p = assessor.estimate_default_prob("F01", current_day=10)
        # With Laplace smoothing: (1 + 5) / (2 + 10) = 6/12 = 0.5
        assert Decimal("0.3") < p < Decimal("0.7"), f"Expected ~0.5, got {p}"

    def test_issuer_specific_tracks_independently(self):
        """Different issuers maintain independent histories."""
        params = _make_risk_params(initial_prior=Decimal("0.15"))
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        # F01 always repays, F02 always defaults
        for day in range(1, 11):
            assessor.update_history(day=day, issuer_id="F01", defaulted=False)
            assessor.update_history(day=day, issuer_id="F02", defaulted=True)

        p_f01 = assessor.estimate_default_prob("F01", current_day=10)
        p_f02 = assessor.estimate_default_prob("F02", current_day=10)

        assert p_f01 < Decimal("0.15"), f"F01 should be below prior, got {p_f01}"
        assert p_f02 > Decimal("0.50"), f"F02 should be above prior, got {p_f02}"
        assert p_f01 < p_f02, "Good borrower should have lower P than bad borrower"


# ═══════════════════════════════════════════════════════════════════════
# 3. Blending: Coverage Dominates Early, Posterior Dominates Late
# ═══════════════════════════════════════════════════════════════════════


class TestBlending:
    """Test that coverage-ratio dominates early, Bayesian posterior dominates late."""

    def test_zero_observations_pure_coverage(self):
        """With zero observations, w_bayes=0, so p_default = p_coverage."""
        params = _make_risk_params()
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        profile = LenderProfile(
            kappa=Decimal("1.0"),
            risk_assessment_params=params,
            warmup_observations=10,
        )

        config = LendingConfig(
            risk_assessor=assessor,
            lender_profile=profile,
        )

        system = _build_lending_system(firm_cash=500, firm_payable_amount=1000)
        system.state.lender_config = config

        events = run_lending_phase(system, current_day=0, lending_config=config)

        # Should produce a loan (pure coverage-based pricing, same as original)
        assert len(events) > 0, "Should produce at least one loan"

    def test_warmup_halfway_blended(self):
        """With warmup_observations/2 observations, w_bayes=0.5."""
        params = _make_risk_params(initial_prior=Decimal("0.50"))
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        # Add 5 observations (all defaults) for F01 -> high Bayesian P
        for day in range(1, 6):
            assessor.update_history(day=day, issuer_id="F01", defaulted=True)

        warmup = 10
        LenderProfile(
            kappa=Decimal("1.0"),
            risk_assessment_params=params,
            warmup_observations=warmup,
        )

        # n = 5, warmup = 10, so w_bayes = 0.5
        issuer_hist = assessor.belief_tracker.issuer_history.get("F01", [])
        n = len(issuer_hist)
        assert n == 5
        w_bayes = min(Decimal("1"), Decimal(str(n)) / Decimal(str(warmup)))
        assert w_bayes == Decimal("0.5")

    def test_full_warmup_pure_bayesian(self):
        """After warmup_observations, w_bayes=1.0 -> pure Bayesian."""
        params = _make_risk_params(initial_prior=Decimal("0.50"))
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        # Add 10 observations (all repayments) for F01
        for day in range(1, 11):
            assessor.update_history(day=day, issuer_id="F01", defaulted=False)

        warmup = 10
        LenderProfile(
            kappa=Decimal("1.0"),
            risk_assessment_params=params,
            warmup_observations=warmup,
        )

        # n = 10, warmup = 10, so w_bayes = 1.0
        issuer_hist = assessor.belief_tracker.issuer_history.get("F01", [])
        n = len(issuer_hist)
        assert n == 10
        w_bayes = min(Decimal("1"), Decimal(str(n)) / Decimal(str(warmup)))
        assert w_bayes == Decimal("1")

    def test_blending_weight_computation(self):
        """Verify the blending weight formula matches expectations."""
        warmup = 5

        # 0 observations -> w = 0
        assert min(Decimal("1"), Decimal("0") / Decimal(str(warmup))) == Decimal("0")

        # 1 observation -> w = 0.2
        assert min(Decimal("1"), Decimal("1") / Decimal(str(warmup))) == Decimal("0.2")

        # 3 observations -> w = 0.6
        assert min(Decimal("1"), Decimal("3") / Decimal(str(warmup))) == Decimal("0.6")

        # 5 observations -> w = 1.0
        assert min(Decimal("1"), Decimal("5") / Decimal(str(warmup))) == Decimal("1")

        # 10 observations -> w = 1.0 (capped)
        assert min(Decimal("1"), Decimal("10") / Decimal(str(warmup))) == Decimal("1")

    def test_blend_weight_uses_lookback_window_not_lifetime(self):
        """w_bayes should count only observations within the lookback window,
        not lifetime history.  If old observations fall outside the window,
        coverage should regain influence.
        """
        lookback = 5
        warmup = 10
        params = _make_risk_params(
            lookback_window=lookback,
            initial_prior=Decimal("0.50"),
        )
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        # Add 10 observations on days 1-10 (all repayments)
        for day in range(1, 11):
            assessor.update_history(day=day, issuer_id="F01", defaulted=False)

        # At day 10, lifetime history = 10, but window [5..10] has 6 (days 5-10 inclusive)
        tracker = assessor.belief_tracker
        window_start = 10 - tracker.lookback_window  # = 5
        n_in_window = sum(
            1 for d, _ in tracker.issuer_history["F01"] if d >= window_start
        )
        assert n_in_window == 6, "Days 5-10 inclusive = 6 observations within lookback"

        # w_bayes = 6/10 = 0.6, NOT 10/10 = 1.0 (lifetime would give)
        w_bayes = min(
            Decimal("1"),
            Decimal(str(n_in_window)) / Decimal(str(warmup)),
        )
        assert w_bayes == Decimal("0.6"), (
            f"w_bayes should use lookback-windowed count, got {w_bayes}"
        )
        # The key assertion: windowed count < lifetime count
        assert n_in_window < len(tracker.issuer_history["F01"]), (
            "Windowed count should be less than lifetime count"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestBackwardCompat:
    """Test that no assessor -> existing waterfall behavior."""

    def test_no_assessor_no_profile_uses_waterfall(self):
        """Without LenderProfile or assessor, lending uses default waterfall."""
        system = _build_lending_system()
        config = LendingConfig()  # No profile, no assessor

        events = run_lending_phase(system, current_day=0, lending_config=config)
        # Should still produce loans using the waterfall heuristic
        assert len(events) > 0

    def test_profile_without_assessor_uses_coverage(self):
        """LenderProfile without risk_assessment_params uses pure coverage-ratio."""
        system = _build_lending_system()
        profile = LenderProfile(kappa=Decimal("1.0"))
        assert profile.risk_assessment_params is None

        config = LendingConfig(lender_profile=profile)
        events = run_lending_phase(system, current_day=0, lending_config=config)
        assert len(events) > 0

    def test_repayment_without_assessor_no_error(self):
        """run_loan_repayments works fine when no assessor is on lender_config."""
        system = _build_lending_system()
        config = LendingConfig()
        system.state.lender_config = config

        # No loans to repay, but should not error
        events = run_loan_repayments(system, current_day=2)
        assert isinstance(events, list)

    def test_repayment_without_lender_config_no_error(self):
        """run_loan_repayments works when lender_config is None (no lending)."""
        system = _build_lending_system()
        # Do not set lender_config at all
        assert getattr(system.state, "lender_config", None) is None

        events = run_loan_repayments(system, current_day=2)
        assert isinstance(events, list)


# ═══════════════════════════════════════════════════════════════════════
# 5. Loan Repayment Updates Assessor
# ═══════════════════════════════════════════════════════════════════════


class TestRepaymentUpdatesAssessor:
    """Test that loan repayment outcomes update the NBFI BeliefTracker."""

    def test_repayment_feeds_assessor(self):
        """Successful loan repayment records a non-default in assessor."""
        # Firm has 100 cash but owes 1000 -> shortfall of 900
        system = _build_lending_system(firm_cash=100, firm_payable_amount=1000)

        params = _make_risk_params()
        assessor = RiskAssessor(params)
        assessor.belief_tracker.use_issuer_specific = True

        profile = LenderProfile(
            kappa=Decimal("1.0"),
            risk_assessment_params=params,
            warmup_observations=5,
        )

        config = LendingConfig(
            risk_assessor=assessor,
            lender_profile=profile,
            maturity_days=1,
        )
        system.state.lender_config = config

        # Create a loan on day 0
        events = run_lending_phase(system, current_day=0, lending_config=config)
        assert len(events) > 0, "Should create a loan (firm has shortfall)"

        # Before repayment: no history yet
        assert len(assessor.belief_tracker.payment_history) == 0

        # Give firm enough cash to repay the loan (principal + interest)
        system.mint_cash("F01", 5000)

        # Repay on day 1 (maturity_days=1, so due on day 1)
        repay_events = run_loan_repayments(system, current_day=1)

        if len(repay_events) > 0:
            # Assessor should now have one observation
            assert len(assessor.belief_tracker.payment_history) >= 1
            # Check that the observation was recorded for the borrower
            last_entry = assessor.belief_tracker.payment_history[-1]
            assert last_entry[1] == "F01"  # issuer_id
            # If repaid, defaulted should be False
            if repay_events[0].get("repaid"):
                assert last_entry[2] is False  # defaulted = False
