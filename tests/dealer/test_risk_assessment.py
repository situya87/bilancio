"""Tests for trader risk assessment module."""

from decimal import Decimal

import pytest

from bilancio.dealer.risk_assessment import RiskAssessor, RiskAssessmentParams
from bilancio.dealer.models import Ticket


def test_default_probability_estimation_no_history():
    """With no history, should return prior default rate."""
    params = RiskAssessmentParams()
    assessor = RiskAssessor(params)

    p_default = assessor.estimate_default_prob(issuer_id="issuer_1", current_day=10)

    # Should return prior (15% - reflects moderate uncertainty with zero data)
    assert p_default == Decimal("0.15")


def test_default_probability_estimation_with_history():
    """Should estimate default rate from recent history."""
    params = RiskAssessmentParams(lookback_window=5)
    assessor = RiskAssessor(params)

    # Simulate 10 payments: 2 defaults, 8 successes
    for i in range(8):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=False)
    for i in range(8, 10):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=True)

    # Estimate at day 10 (lookback to day 5)
    # Recent window: days 5-9 inclusive
    # In that window: 3 successes (days 5,6,7) + 2 defaults (days 8,9) = 5 total, 2 defaults
    p_default = assessor.estimate_default_prob(issuer_id="issuer_1", current_day=10)

    # With Laplace smoothing (alpha=1): (1 + 2) / (2*1 + 5) = 3/7 ≈ 0.4286
    expected = (Decimal(1) + Decimal(2)) / (Decimal(2) + Decimal(5))
    assert p_default == expected


def test_expected_value_low_default_prob():
    """Expected value should be close to face value with low default probability."""
    params = RiskAssessmentParams(lookback_window=5)
    assessor = RiskAssessor(params)

    # Simulate low default rate within lookback window (days 15-19)
    # Add 19 successes (days 15-33) and 1 default (day 34)
    for i in range(15, 34):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=False)
    assessor.update_history(day=34, issuer_id="issuer_1", defaulted=True)

    ticket = Ticket(
        id="ticket_1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(20),
        maturity_day=50,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    ev = assessor.expected_value(ticket, current_day=35)

    # Lookback window: days 30-35 inclusive
    # In that window: 19 successes (days 30-33) + 1 default (day 34) = 5 events, 1 default
    # With smoothing: (1 + 1) / (2*1 + 5) = 2/7 ≈ 0.2857 default rate
    # But wait, day 35 window goes back to day 30, so: days 30,31,32,33,34
    # That's 4 successes + 1 default = 5 total
    p_default_expected = (Decimal(1) + Decimal(1)) / (Decimal(2) + Decimal(5))
    ev_expected = (Decimal(1) - p_default_expected) * Decimal(20)

    assert ev == ev_expected


def test_should_sell_accept_good_price():
    """Trader should accept dealer bid above expected value + threshold."""
    # Use lookback_window=25 to include all 20 events in calculation
    params = RiskAssessmentParams(base_risk_premium=Decimal("0.02"), lookback_window=25)
    assessor = RiskAssessor(params)

    # Low default rate: 1 default, 19 successes
    for i in range(19):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=False)
    assessor.update_history(day=19, issuer_id="issuer_1", defaulted=True)

    ticket = Ticket(
        id="ticket_1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(20),
        maturity_day=30,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    # With lookback_window=25, all 20 events are considered
    # P(default) = (1 + 1) / (2 + 20) = 2/22 ≈ 0.0909, EV = 0.9091 * 20 ≈ 18.18
    # With 2% threshold: need 18.18 + 0.02*20 = 18.58

    # Test 1: Bid of 1.0 (= 20 total) should be accepted
    should_accept = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("1.0"),  # Unit price
        current_day=20,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),  # No urgency
        trader_asset_value=Decimal(50),
    )
    assert should_accept is True

    # Test 2: Bid of 0.90 (= 18 total) should be rejected (below 18.58)
    should_accept = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("0.90"),
        current_day=20,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    )
    assert should_accept is False


def test_should_sell_urgency_lowers_threshold():
    """Liquidity urgency should lower acceptance threshold."""
    # Use lookback_window=25 to include all 20 events in calculation
    params = RiskAssessmentParams(
        base_risk_premium=Decimal("0.02"), urgency_sensitivity=Decimal("0.10"), lookback_window=25
    )
    assessor = RiskAssessor(params)

    # Low default rate
    for i in range(19):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=False)
    assessor.update_history(day=19, issuer_id="issuer_1", defaulted=True)

    ticket = Ticket(
        id="ticket_1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(20),
        maturity_day=30,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    # Scenario 1: No urgency - high threshold
    # Base threshold = 2%, need EV + 2% = 18.18 + 0.40 = 18.58
    should_accept_normal = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("0.92"),  # = 18.40 total
        current_day=20,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),  # No shortfall
        trader_asset_value=Decimal(50),
    )

    # Scenario 2: HIGH urgency - lower threshold
    # Wealth = 100 + 50 = 150, shortfall = 75, urgency = 0.5
    # Threshold = 0.02 - 0.10*0.5 = -0.03 (willing to take 3% loss!)
    # Now accept even worse prices
    should_accept_urgent = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("0.92"),  # Same bid as before
        current_day=20,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(75),  # Severe shortfall
        trader_asset_value=Decimal(50),
    )

    # Normal case: might reject (18.40 < 18.58)
    # Urgent case: should accept (18.40 > 18.18 - 0.60 = 17.58)
    assert should_accept_normal is False  # Rejects without urgency
    assert should_accept_urgent is True  # Accepts with urgency


def test_should_sell_high_default_rate():
    """With high default rate, should accept lower dealer bids."""
    params = RiskAssessmentParams(base_risk_premium=Decimal("0.02"))
    assessor = RiskAssessor(params)

    # High default rate: 12 defaults, 8 successes
    for i in range(8):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=False)
    for i in range(8, 20):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=True)

    ticket = Ticket(
        id="ticket_1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(20),
        maturity_day=30,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    # P(default) = (1 + 12) / (2 + 20) = 13/22 ≈ 0.5909
    # EV = (1 - 0.5909) * 20 ≈ 8.18
    # With 2% threshold: need 8.18 + 0.40 = 8.58

    # Bid of 0.50 (= 10 total) should be accepted (well above 8.58)
    should_accept = assessor.should_sell(
        ticket=ticket,
        dealer_bid=Decimal("0.50"),
        current_day=20,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    )
    assert should_accept is True


def test_should_buy_requires_higher_premium():
    """Buying should require higher premium than selling."""
    # Use lookback_window=25 to include all 20 events in calculation
    params = RiskAssessmentParams(
        base_risk_premium=Decimal("0.02"), buy_premium_multiplier=Decimal("2.0"), lookback_window=25
    )
    assessor = RiskAssessor(params)

    # Low default rate
    for i in range(19):
        assessor.update_history(day=i, issuer_id="issuer_1", defaulted=False)
    assessor.update_history(day=19, issuer_id="issuer_1", defaulted=True)

    ticket = Ticket(
        id="ticket_1",
        issuer_id="issuer_1",
        owner_id="trader_1",
        face=Decimal(20),
        maturity_day=30,
        remaining_tau=10,
        bucket_id="short",
        serial=1,
    )

    # EV ≈ 18.18
    # For buying: need EV >= cost + 4% (2x base premium)
    # So dealer must ask <= 18.18 - 0.80 = 17.38

    # Test 1: Ask of 0.85 (= 17.0 total) should be accepted
    should_accept = assessor.should_buy(
        ticket=ticket,
        dealer_ask=Decimal("0.85"),
        current_day=20,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    )
    assert should_accept is True

    # Test 2: Ask of 0.95 (= 19.0 total) should be rejected
    should_accept = assessor.should_buy(
        ticket=ticket,
        dealer_ask=Decimal("0.95"),
        current_day=20,
        trader_cash=Decimal(100),
        trader_shortfall=Decimal(0),
        trader_asset_value=Decimal(50),
    )
    assert should_accept is False


def test_diagnostics():
    """Should provide diagnostic information."""
    params = RiskAssessmentParams(lookback_window=3)
    assessor = RiskAssessor(params)

    # Add some history
    for i in range(10):
        defaulted = i % 3 == 0  # Every 3rd payment defaults
        assessor.update_history(day=i, issuer_id=f"issuer_{i%2}", defaulted=defaulted)

    diag = assessor.get_diagnostics(current_day=10)

    assert diag["total_payment_history_size"] == 10
    assert diag["lookback_window"] == 3
    assert diag["base_risk_premium"] == 0.02
    assert "system_default_rate" in diag
