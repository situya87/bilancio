"""Named information profiles for common agent archetypes.

Mirrors the pattern in ``bilancio.decision.presets``.
"""

from decimal import Decimal

from bilancio.information.levels import AccessLevel
from bilancio.information.noise import (
    AggregateOnlyNoise,
    EstimationNoise,
    SampleNoise,
)
from bilancio.information.profile import CategoryAccess, InformationProfile

# ── OMNISCIENT ─────────────────────────────────────────────────────────
# All fields PERFECT — backward-compatible default.
OMNISCIENT = InformationProfile()


# ── LENDER_REALISTIC ──────────────────────────────────────────────────
# A realistic non-bank lender: partial visibility into counterparties,
# no access to market prices or network topology.
LENDER_REALISTIC = InformationProfile(
    # I. Counterparty Balance Sheet — noisy
    counterparty_cash=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.15"))
    ),
    counterparty_assets=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    counterparty_liabilities=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    counterparty_net_worth=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.20"))
    ),
    counterparty_liquidity_ratio=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.20"))
    ),
    # II. Counterparty History — partial observation
    counterparty_default_history=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.7"))
    ),
    counterparty_settlement_history=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.7"))
    ),
    counterparty_track_record=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.7"))
    ),
    counterparty_partial_settlement=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    counterparty_avg_shortfall=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    # IV. Bilateral — own data always perfect
    bilateral_history=CategoryAccess(AccessLevel.PERFECT),
    # V. Market Prices — lender not in secondary market
    dealer_quotes=CategoryAccess(AccessLevel.NONE),
    vbt_anchors=CategoryAccess(AccessLevel.NONE),
    price_trends=CategoryAccess(AccessLevel.NONE),
    implied_default_prob=CategoryAccess(AccessLevel.NONE),
    # VII. Network — no access
    obligation_graph=CategoryAccess(AccessLevel.NONE),
    counterparty_connectivity=CategoryAccess(AccessLevel.NONE),
    cascade_risk=CategoryAccess(AccessLevel.NONE),
)


# ── TRADER_BASIC ──────────────────────────────────────────────────────
# A basic trader: can see market prices but not balance sheets.
TRADER_BASIC = InformationProfile(
    # I. Balance Sheet — no access
    counterparty_cash=CategoryAccess(AccessLevel.NONE),
    counterparty_assets=CategoryAccess(AccessLevel.NONE),
    counterparty_liabilities=CategoryAccess(AccessLevel.NONE),
    counterparty_net_worth=CategoryAccess(AccessLevel.NONE),
    counterparty_liquidity_ratio=CategoryAccess(AccessLevel.NONE),
    # II. History — partial
    counterparty_default_history=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.8"))
    ),
    counterparty_settlement_history=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.8"))
    ),
    counterparty_track_record=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.8"))
    ),
    counterparty_partial_settlement=CategoryAccess(AccessLevel.NONE),
    counterparty_avg_shortfall=CategoryAccess(AccessLevel.NONE),
    # V. Market Prices — full access
    dealer_quotes=CategoryAccess(AccessLevel.PERFECT),
    vbt_anchors=CategoryAccess(AccessLevel.PERFECT),
    price_trends=CategoryAccess(AccessLevel.PERFECT),
    implied_default_prob=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.10"))
    ),
    # VII. Network — no access
    obligation_graph=CategoryAccess(AccessLevel.NONE),
    counterparty_connectivity=CategoryAccess(AccessLevel.NONE),
    cascade_risk=CategoryAccess(AccessLevel.NONE),
)
