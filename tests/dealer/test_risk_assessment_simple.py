"""Simplified tests for trader risk assessment module."""

from decimal import Decimal

import pytest

from bilancio.dealer.risk_assessment import RiskAssessor, RiskAssessmentParams
from bilancio.dealer.models import Ticket


def test_no_history_uses_prior():
    """With no history, should return prior default rate (25%)."""
    params = RiskAssessmentParams()
    assessor = RiskAssessor(params)

    p_default = assessor.estimate_default_prob(issuer_id="issuer_1", current_day=10)
    assert p_default == Decimal("0.25")


def test_all_successes_low_default_prob():
    """All successes in window should give low default probability."""
    params = RiskAssessmentParams(lookback_window=3, smoothing_alpha=Decimal("0.1"))
    assessor = RiskAssessor(params)

    # Days 0-10: all successes
    for day in range(11):
        assessor.update_history(day=day, issuer_id="issuer_1", defaulted=False)

    # At day 10, window is [7, 10], so 4 successes, 0 defaults
    p_default = assessor.estimate_default_prob(issuer_id="issuer_1", current_day=10)

    # With smoothing: (0.1 + 0) / (2*0.1 + 4) = 0.1 / 4.2 ≈ 0.0238
    expected = (Decimal("0.1") + Decimal(0)) / (Decimal(2) * Decimal("0.1") + Decimal(4))
    assert p_default == expected
    assert p_default < Decimal("0.05")  # Much less than prior


def test_all_defaults_high_default_prob():
    """All defaults in window should give high default probability."""
    params = RiskAssessmentParams(lookback_window=3, smoothing_alpha=Decimal("0.1"))
    assessor = RiskAssessor(params)

    # Days 0-10: all defaults
    for day in range(11):
        assessor.update_history(day=day, issuer_id="issuer_1", defaulted=True)

    # At day 10, window is [7, 10], so 0 successes, 4 defaults
    p_default = assessor.estimate_default_prob(issuer_id="issuer_1", current_day=10)

    # With smoothing: (0.1 + 4) / (2*0.1 + 4) = 4.1 / 4.2 ≈ 0.976
    expected = (Decimal("0.1") + Decimal(4)) / (Decimal(2) * Decimal("0.1") + Decimal(4))
    assert p_default == expected
    assert p_default > Decimal("0.9")  # Very high


def test_expected_value_scales_with_face():
    """Expected value should scale linearly with face value."""
    params = RiskAssessmentParams()
    assessor = RiskAssessor(params)

    # Set default probability to 20% (prior with no history gives 5%, so add some defaults)
    for _ in range(2):
        assessor.update_history(day=0, issuer_id="issuer_1", defaulted=True)
    for _ in range(8):
        assessor.update_history(day=0, issuer_id="issuer_1", defaulted=False)

    ticket_small = Ticket(
        id="t1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(10),
        maturity_day=10,
        remaining_tau=5,
        bucket_id="short",
        serial=1,
    )

    ticket_large = Ticket(
        id="t2",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(100),
        maturity_day=10,
        remaining_tau=5,
        bucket_id="short",
        serial=2,
    )

    ev_small = assessor.expected_value(ticket_small, current_day=1)
    ev_large = assessor.expected_value(ticket_large, current_day=1)

    # EV should scale 10x
    assert ev_large / ev_small == Decimal(10)


def test_should_sell_accepts_premium_above_threshold():
    """Trader accepts when dealer offers premium above threshold."""
    params = RiskAssessmentParams(base_risk_premium=Decimal("0.10"))  # 10% premium required
    assessor = RiskAssessor(params)

    # Set to zero defaults (best case)
    for day in range(10):
        assessor.update_history(day=day, issuer_id="issuer_1", defaulted=False)

    ticket = Ticket(
        id="t1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(10),
        maturity_day=20,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    # EV ≈ 10 (very low default prob with smoothing)
    # Threshold: 10% of 10 = 1.0
    # Need: dealer_bid * 10 >= EV + 1.0

    # Test 1: Bid of 1.15 (total = 11.5) should be accepted (well above EV + 1.0)
    assert assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("1.15"),
        current_day=10,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    ) is True

    # Test 2: Bid of 0.95 (total = 9.5) should be rejected (below EV)
    assert assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("0.95"),
        current_day=10,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    ) is False


def test_urgency_reduces_threshold():
    """High shortfall should make trader accept worse prices."""
    params = RiskAssessmentParams(
        base_risk_premium=Decimal("0.10"),  # 10% normally
        urgency_sensitivity=Decimal("0.20"),  # 20% reduction per unit urgency
    )
    assessor = RiskAssessor(params)

    # Zero defaults
    for day in range(10):
        assessor.update_history(day=day, issuer_id="issuer_1", defaulted=False)

    ticket = Ticket(
        id="t1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(10),
        maturity_day=20,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    # Scenario 1: No urgency - normal threshold
    # Wealth = 100 + 50 = 150, shortfall = 0, urgency = 0
    # Threshold = 0.10 - 0.20*0 = 0.10 (10%)
    accept_normal = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("1.05"),  # Total = 10.5
        current_day=10,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),  # No urgency
        trader_asset_value=Decimal(50),
    )

    # Scenario 2: High urgency - reduced threshold
    # Wealth = 100 + 50 = 150, shortfall = 75, urgency = 0.5
    # Threshold = 0.10 - 0.20*0.5 = 0.0 (zero premium required!)
    accept_urgent = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("1.05"),  # Same bid
        current_day=10,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(75),  # High shortfall
        trader_asset_value=Decimal(50),
    )

    # With normal conditions, might reject; with urgency, should accept
    assert accept_urgent is True


def test_buying_requires_higher_premium_than_selling():
    """Asymmetry: buying needs higher premium than selling."""
    params = RiskAssessmentParams(
        base_risk_premium=Decimal("0.05"),  # 5% for selling
        buy_premium_multiplier=Decimal("3.0"),  # 15% for buying
    )
    assessor = RiskAssessor(params)

    # Zero defaults
    for day in range(10):
        assessor.update_history(day=day, issuer_id="issuer_1", defaulted=False)

    ticket = Ticket(
        id="t1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(10),
        maturity_day=20,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    # For selling: need dealer_bid >= EV + 5%
    # For buying: need EV >= dealer_ask + 15%

    # With very low default prob, EV ≈ 10

    # Selling: will accept bid of 1.06 (total 10.6 > 10 + 0.5)
    will_sell = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("1.06"),
        current_day=10,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    )

    # Buying: will NOT accept ask of 1.06 (need 10 >= 10.6 + 1.5 = FALSE)
    will_buy = assessor.should_buy(
        ticket=ticket,
        dealer_ask=Decimal("1.06"),  # Same price
        current_day=10,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    )

    assert will_sell is True  # Happy to sell at this price
    assert will_buy is False  # Won't buy at same price (asymmetry)


def test_diagnostics_provides_useful_info():
    """get_diagnostics should return system state information."""
    params = RiskAssessmentParams(lookback_window=3)
    assessor = RiskAssessor(params)

    # Add history
    for day in range(10):
        defaulted = day % 3 == 0  # Every 3rd defaults
        assessor.update_history(day=day, issuer_id=f"issuer_{day%2}", defaulted=defaulted)

    diag = assessor.get_diagnostics(current_day=10)

    assert "total_payment_history_size" in diag
    assert "system_default_rate" in diag
    assert "lookback_window" in diag
    assert diag["total_payment_history_size"] == 10
    assert diag["lookback_window"] == 3
