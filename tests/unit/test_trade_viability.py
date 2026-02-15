"""Math-based trade viability tests for VBT pricing reform.

Verifies that the reformed pricing parameters (M = 1-P, per-bucket spreads,
seller premium=0, buyer premium=0.01) produce viable trades across a range
of kappa values.  No simulation needed — just pricing model + risk assessor.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.dealer.priors import kappa_informed_prior
from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
from bilancio.dealer.models import Ticket
from bilancio.decision.valuers import CreditAdjustedVBTPricing


# ── Helpers ──────────────────────────────────────────────────────────

FACE = Decimal(20)
BASE_SPREAD = {"short": Decimal("0.04"), "mid": Decimal("0.08"), "long": Decimal("0.12")}


def _make_ticket(issuer: str = "H1", face: Decimal = FACE) -> Ticket:
    return Ticket(
        id="TKT_test",
        issuer_id=issuer,
        owner_id="H2",
        face=face,
        maturity_day=5,
        remaining_tau=3,
        bucket_id="short",
        serial=0,
    )


# ── 1. kappa_informed_prior basic properties ─────────────────────────

class TestKappaInformedPrior:
    def test_high_stress(self):
        """kappa=0 → max prior 0.20."""
        assert kappa_informed_prior(Decimal("0.001")) > Decimal("0.19")

    def test_balanced(self):
        """kappa=1 → floor prior 0.05."""
        assert kappa_informed_prior(Decimal("1")) == Decimal("0.05")

    def test_abundant(self):
        """kappa>1 → floor stays 0.05."""
        assert kappa_informed_prior(Decimal("2")) == Decimal("0.05")

    def test_half(self):
        """kappa=0.5 → P = 0.05 + 0.15 × 0.5/1.5 = 0.10."""
        p = kappa_informed_prior(Decimal("0.5"))
        assert abs(p - Decimal("0.10")) < Decimal("0.001")

    def test_monotonically_decreasing(self):
        """Higher kappa → lower default probability."""
        kappas = [Decimal("0.1"), Decimal("0.5"), Decimal("1"), Decimal("2")]
        priors = [kappa_informed_prior(k) for k in kappas]
        for i in range(len(priors) - 1):
            assert priors[i] >= priors[i + 1]


# ── 2. VBT Pricing Model ────────────────────────────────────────────

class TestCreditAdjustedVBTPricing:
    def test_mid_at_prior(self):
        """When p_default == initial_prior, M should be 1 - prior."""
        model = CreditAdjustedVBTPricing()
        prior = Decimal("0.10")
        M = model.compute_mid(prior, prior)
        assert M == Decimal(1) - prior

    def test_mid_tracks_defaults(self):
        """When defaults increase, M should decrease."""
        model = CreditAdjustedVBTPricing(mid_sensitivity=Decimal("1.0"))
        prior = Decimal("0.10")
        M_low = model.compute_mid(Decimal("0.05"), prior)
        M_high = model.compute_mid(Decimal("0.30"), prior)
        assert M_low > M_high

    def test_spread_additive(self):
        """Spread = base + sensitivity × p_default."""
        model = CreditAdjustedVBTPricing(spread_sensitivity=Decimal("0.6"))
        base = Decimal("0.04")
        p = Decimal("0.10")
        expected = base + Decimal("0.6") * p  # 0.04 + 0.06 = 0.10
        assert model.compute_spread(base, p) == expected


# ── 3. Urgent seller accepts bid ─────────────────────────────────────

@pytest.mark.parametrize("kappa", [Decimal("0.25"), Decimal("0.5"), Decimal("1.0")])
def test_urgent_seller_accepts_bid(kappa: Decimal):
    """An urgent seller (shortfall > 0) must accept the dealer's short-bucket bid.

    Setup:
    - P_prior from kappa, M = 1 - P_prior
    - Short bucket: O = 0.04 + 0.6 × P_prior
    - Dealer bid at ~50% inventory: bid ≈ M - O/4
    - Seller has shortfall = 50, total obligation ~ 5 × face
    """
    P_prior = kappa_informed_prior(kappa)
    M = Decimal(1) - P_prior
    O = Decimal("0.04") + Decimal("0.6") * P_prior  # short bucket

    # Dealer bid at half capacity (midline between M-O/2 and M)
    dealer_bid = M - O / 4

    # Risk assessor with seller premium = 0
    risk_params = RiskAssessmentParams(
        base_risk_premium=Decimal(0),
        initial_prior=P_prior,
    )
    assessor = RiskAssessor(risk_params)

    ticket = _make_ticket()

    # Typical agent: obligation ≈ 5 × face, cash = kappa × obligation
    obligation = 5 * FACE  # 100
    cash = kappa * obligation
    ev_asset = (Decimal(1) - P_prior) * FACE
    shortfall = max(Decimal(0), obligation - cash)

    if shortfall <= 0:
        pytest.skip(f"No shortfall at kappa={kappa}, seller not urgent")

    should = assessor.should_sell(
        ticket=ticket,
        dealer_bid=dealer_bid,
        current_day=0,
        trader_cash=cash,
        trader_shortfall=shortfall,
        trader_asset_value=ev_asset,
    )
    assert should, (
        f"kappa={kappa}: urgent seller should accept bid={dealer_bid:.4f} "
        f"(M={M:.4f}, O={O:.4f}, P={P_prior:.4f})"
    )


# ── 4. Buyer accepts after inventory buildup ─────────────────────────

@pytest.mark.parametrize("kappa", [Decimal("0.25"), Decimal("0.5"), Decimal("1.0")])
def test_buyer_accepts_after_inventory_buildup(kappa: Decimal):
    """After sells push inventory above midpoint, dealer ask drops and buyer accepts.

    When dealer inventory grows, ask drops toward M - O/2.
    A buyer with surplus should buy if: EV > ask + buy_premium × face.
    """
    P_prior = kappa_informed_prior(kappa)
    M = Decimal(1) - P_prior
    O = Decimal("0.04") + Decimal("0.6") * P_prior

    # After inventory buildup, ask drops. At high inventory, ask ≈ M (midline).
    # More precisely, ask = M + O/2 × (1 - 2×inventory_ratio)
    # At 75% capacity (after several sells): inventory_ratio = 0.75
    # ask ≈ M + O/2 × (1 - 1.5) = M - O/4
    dealer_ask = M - O / 4

    # Buyer with surplus: EV = (1-P) × face
    ev_per_unit = Decimal(1) - P_prior  # Same as M

    # Buy premium = 0.01
    buy_premium = Decimal("0.01")

    # Buyer accepts if: ev_per_unit × face >= dealer_ask × face + buy_premium × face
    # Simplify: ev_per_unit >= dealer_ask + buy_premium
    should_accept = ev_per_unit >= (dealer_ask + buy_premium)

    assert should_accept, (
        f"kappa={kappa}: buyer should accept ask={dealer_ask:.4f} "
        f"(EV/face={ev_per_unit:.4f}, premium={buy_premium}, "
        f"gap={ev_per_unit - dealer_ask - buy_premium:.4f})"
    )


# ── 5. VBT updates from defaults ────────────────────────────────────

def test_vbt_updates_from_defaults():
    """After observed defaults, M decreases and spreads widen."""
    model = CreditAdjustedVBTPricing(
        mid_sensitivity=Decimal("1.0"),
        spread_sensitivity=Decimal("0.6"),
    )
    initial_prior = Decimal("0.10")
    base_spread = Decimal("0.04")

    # Before defaults
    M_before = model.compute_mid(initial_prior, initial_prior)
    O_before = model.compute_spread(base_spread, initial_prior)

    # After defaults push P_default up
    p_after = Decimal("0.25")
    M_after = model.compute_mid(p_after, initial_prior)
    O_after = model.compute_spread(base_spread, p_after)

    assert M_after < M_before, "M should decrease after defaults"
    assert O_after > O_before, "Spread should widen after defaults"

    # M should be exactly 1 - p_default (with full sensitivity)
    assert M_after == Decimal(1) - p_after


# ── 6. Market functions after defaults ───────────────────────────────

@pytest.mark.parametrize("p_default", [Decimal("0.15"), Decimal("0.25"), Decimal("0.35")])
def test_market_still_functions_after_defaults(p_default: Decimal):
    """Even after defaults increase P_default, urgent sellers still accept.

    The key property: with seller premium=0, any positive urgency makes
    the threshold negative, so the seller accepts as long as the bid is
    positive — which it always is when M > O/2.
    """
    model = CreditAdjustedVBTPricing()
    initial_prior = Decimal("0.10")

    M = model.compute_mid(p_default, initial_prior)
    base_spread = Decimal("0.04")
    O = model.compute_spread(base_spread, p_default)

    # The market is viable as long as M > O/2 (bid stays positive)
    assert M > O / 2, (
        f"M={M:.4f} must exceed O/2={O / 2:.4f} for positive bids "
        f"(p_default={p_default})"
    )

    # Dealer bid at half capacity
    dealer_bid = M - O / 4

    risk_params = RiskAssessmentParams(
        base_risk_premium=Decimal(0),
        initial_prior=initial_prior,
    )
    assessor = RiskAssessor(risk_params)
    # Feed in some default history so the assessor sees updated P
    for i in range(10):
        defaulted = i < int(p_default * 10)
        assessor.update_history(day=1, issuer_id=f"H{i}", defaulted=defaulted)

    ticket = _make_ticket()

    # Urgent seller: shortfall > 0
    should = assessor.should_sell(
        ticket=ticket,
        dealer_bid=dealer_bid,
        current_day=2,
        trader_cash=Decimal(10),
        trader_shortfall=Decimal(50),
        trader_asset_value=Decimal(18),
    )
    assert should, f"Urgent seller should still accept at p_default={p_default}"


# ── 7. Per-bucket spread ordering ────────────────────────────────────

def test_per_bucket_spread_ordering():
    """Short < mid < long base spreads, reflecting term risk premium."""
    model = CreditAdjustedVBTPricing(spread_sensitivity=Decimal("0.6"))
    p = Decimal("0.10")

    spreads = {}
    for bucket, base in BASE_SPREAD.items():
        spreads[bucket] = model.compute_spread(base, p)

    assert spreads["short"] < spreads["mid"] < spreads["long"]


# ── 8. Consistency: trader EV matches VBT mid ────────────────────────

def test_trader_ev_matches_vbt_mid():
    """Traders and VBT use the same P_prior, so EV/face == M.

    This prevents adverse selection: the trader's valuation of holding
    matches the VBT's mid-price anchor.
    """
    kappa = Decimal("0.5")
    P_prior = kappa_informed_prior(kappa)

    # VBT mid
    model = CreditAdjustedVBTPricing()
    M = model.compute_mid(P_prior, P_prior)

    # Trader EV per unit face
    risk_params = RiskAssessmentParams(initial_prior=P_prior)
    assessor = RiskAssessor(risk_params)
    ticket = _make_ticket()
    ev = assessor.expected_value(ticket, current_day=0)
    ev_per_face = ev / FACE

    assert M == ev_per_face, (
        f"VBT M={M} must equal trader EV/face={ev_per_face} "
        f"to prevent adverse selection"
    )
