"""Tests for VBT forward-looking stress signal (Plan 040 Phase 2)."""

from decimal import Decimal

import pytest

from bilancio.decision.profiles import VBTProfile
from bilancio.decision.valuers import CreditAdjustedVBTPricing
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_sync import estimate_forward_stress
from bilancio.engines.system import System


def _make_system_with_agents_and_cash(
    n_agents: int = 3,
    cash_per_agent: int = 100,
) -> System:
    """Create a minimal system with agents and cash for testing."""
    sys = System()
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    sys.add_agent(cb)
    for i in range(n_agents):
        agent = Household(id=f"A{i}", name=f"Agent {i}", kind="household")
        sys.add_agent(agent)
        if cash_per_agent > 0:
            sys.mint_cash(to_agent_id=f"A{i}", amount=cash_per_agent)
    return sys


def _add_payable(
    sys: System,
    payable_id: str,
    issuer_id: str,
    holder_id: str,
    amount: int,
    due_day: int,
) -> None:
    """Add a payable obligation to the system."""
    p = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="X",
        asset_holder_id=holder_id,
        liability_issuer_id=issuer_id,
        due_day=due_day,
    )
    sys.add_contract(p)


class TestEstimateForwardStress:
    """Tests for the estimate_forward_stress function."""

    def test_forward_stress_zero_when_liquid(self):
        """When total_cash >= total_due, stress = 0."""
        sys = _make_system_with_agents_and_cash(n_agents=3, cash_per_agent=200)
        # Add payables totaling 300, cash is 600
        _add_payable(sys, "P1", "A0", "A1", 100, due_day=3)
        _add_payable(sys, "P2", "A1", "A2", 100, due_day=4)
        _add_payable(sys, "P3", "A2", "A0", 100, due_day=5)

        stress = estimate_forward_stress(sys, current_day=1, horizon=5)
        assert stress == Decimal(0)

    def test_forward_stress_positive_when_illiquid(self):
        """When obligations exceed cash, stress > 0."""
        sys = _make_system_with_agents_and_cash(n_agents=3, cash_per_agent=50)
        # Add payables totaling 300, cash is only 150
        _add_payable(sys, "P1", "A0", "A1", 100, due_day=3)
        _add_payable(sys, "P2", "A1", "A2", 100, due_day=4)
        _add_payable(sys, "P3", "A2", "A0", 100, due_day=5)

        stress = estimate_forward_stress(sys, current_day=1, horizon=5)
        # stress = 1 - 150/300 = 0.5
        assert stress == Decimal("0.5")

    def test_forward_stress_capped_at_one(self):
        """Stress is capped at 1.0 even when cash is zero."""
        sys = _make_system_with_agents_and_cash(n_agents=2, cash_per_agent=0)
        _add_payable(sys, "P1", "A0", "A1", 100, due_day=3)

        stress = estimate_forward_stress(sys, current_day=1, horizon=5)
        assert stress == Decimal(1)

    def test_forward_stress_zero_when_no_obligations(self):
        """When there are no upcoming obligations, stress = 0."""
        sys = _make_system_with_agents_and_cash(n_agents=2, cash_per_agent=100)
        # No payables added

        stress = estimate_forward_stress(sys, current_day=1, horizon=5)
        assert stress == Decimal(0)

    def test_forward_stress_ignores_obligations_outside_horizon(self):
        """Obligations beyond the horizon window are not counted."""
        sys = _make_system_with_agents_and_cash(n_agents=2, cash_per_agent=10)
        # Due on day 20, but horizon is 5 from current_day=1 (window: [1, 6])
        _add_payable(sys, "P1", "A0", "A1", 1000, due_day=20)

        stress = estimate_forward_stress(sys, current_day=1, horizon=5)
        assert stress == Decimal(0)

    def test_forward_stress_ignores_defaulted_agents(self):
        """Defaulted agents' cash and obligations are excluded."""
        sys = _make_system_with_agents_and_cash(n_agents=2, cash_per_agent=50)
        _add_payable(sys, "P1", "A0", "A1", 200, due_day=3)

        # Mark A0 as defaulted
        sys.state.agents["A0"].defaulted = True

        # Only A1 remains: cash=50, obligations=0 (A0's liability is excluded)
        stress = estimate_forward_stress(sys, current_day=1, horizon=5)
        assert stress == Decimal(0)


class TestComputeMidBlended:
    """Tests for the compute_mid_blended method on CreditAdjustedVBTPricing."""

    def test_compute_mid_blended_method(self):
        """Unit test the new method on CreditAdjustedVBTPricing."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1.0"),
            spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        p_default = Decimal("0.1")
        p_forward = Decimal("0.3")
        initial_prior = Decimal("0.15")
        forward_weight = Decimal("0.5")

        result = pricing.compute_mid_blended(
            p_default, p_forward, initial_prior, forward_weight
        )

        # p_blend = 0.5 * 0.1 + 0.5 * 0.3 = 0.2
        # compute_mid(0.2, 0.15) = 0.9 * (1 - 0.15) + 1.0 * (0.9*(1-0.2) - 0.9*(1-0.15))
        # = 0.765 + 1.0 * (0.72 - 0.765) = 0.765 - 0.045 = 0.72
        expected = pricing.compute_mid(Decimal("0.2"), initial_prior)
        assert result == expected

    def test_backward_compat_forward_weight_zero(self):
        """forward_weight=0 produces identical pricing to existing compute_mid."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("0.8"),
            spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        p_default = Decimal("0.15")
        p_forward = Decimal("0.5")  # should be ignored
        initial_prior = Decimal("0.10")

        blended = pricing.compute_mid_blended(
            p_default, p_forward, initial_prior, forward_weight=Decimal("0.0")
        )
        original = pricing.compute_mid(p_default, initial_prior)

        assert blended == original

    def test_blended_mid_lower_when_stressed(self):
        """With forward_weight > 0 and high stress, M is lower than without."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1.0"),
            spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        p_default = Decimal("0.1")
        p_forward = Decimal("0.5")  # high forward stress
        initial_prior = Decimal("0.15")

        mid_no_stress = pricing.compute_mid(p_default, initial_prior)
        mid_with_stress = pricing.compute_mid_blended(
            p_default, p_forward, initial_prior, forward_weight=Decimal("0.5")
        )

        # Forward stress is higher than p_default, so blended mid should be lower
        assert mid_with_stress < mid_no_stress

    def test_blended_equals_original_when_p_forward_equals_p_default(self):
        """When p_forward == p_default, blending has no effect regardless of weight."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1.0"),
            spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        p = Decimal("0.2")
        initial_prior = Decimal("0.15")

        blended = pricing.compute_mid_blended(
            p, p, initial_prior, forward_weight=Decimal("0.7")
        )
        original = pricing.compute_mid(p, initial_prior)

        assert blended == original


class TestVBTProfileDefaults:
    """Tests for backward compatibility of VBTProfile new fields."""

    def test_default_forward_weight_zero(self):
        """Default VBTProfile has forward_weight=0 (disabled)."""
        profile = VBTProfile()
        assert profile.forward_weight == Decimal("0.0")

    def test_default_stress_horizon_five(self):
        """Default VBTProfile has stress_horizon=5."""
        profile = VBTProfile()
        assert profile.stress_horizon == 5

    def test_custom_forward_weight(self):
        """VBTProfile can be created with custom forward_weight."""
        profile = VBTProfile(forward_weight=Decimal("0.3"), stress_horizon=10)
        assert profile.forward_weight == Decimal("0.3")
        assert profile.stress_horizon == 10
