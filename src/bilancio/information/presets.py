"""Named information profiles for common agent archetypes.

Mirrors the pattern in ``bilancio.decision.presets``.
"""

from decimal import Decimal

from bilancio.information.hierarchy import (
    CounterpartyAccess,
    InstrumentAccess,
    TransactionAccess,
)
from bilancio.information.levels import AccessLevel
from bilancio.information.noise import (
    AggregateOnlyNoise,
    EstimationNoise,
    SampleNoise,
)
from bilancio.information.profile import CategoryAccess, InformationProfile
from bilancio.information.channels import (
    ChannelBinding,
    InstitutionalChannel,
    NetworkDerivedChannel,
    SelfDerivedChannel,
    category_from_channel,
)

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


# ── LENDER_CHANNEL_BASED ────────────────────────────────────────────────
# Equivalent to LENDER_REALISTIC but noise values are *derived* from
# channel properties instead of hand-tuned.  Structural constraints
# (AggregateOnlyNoise, NONE, PERFECT) remain as direct CategoryAccess
# because they describe *what form* the data takes, not signal quality.
LENDER_CHANNEL_BASED = InformationProfile(
    # I. Counterparty Balance Sheet
    counterparty_cash=category_from_channel(SelfDerivedChannel(sample_size=44)),
    counterparty_assets=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    counterparty_liabilities=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    counterparty_net_worth=category_from_channel(SelfDerivedChannel(sample_size=25)),
    counterparty_liquidity_ratio=category_from_channel(
        SelfDerivedChannel(sample_size=25)
    ),
    # II. Counterparty History — partial observation via network
    counterparty_default_history=category_from_channel(
        NetworkDerivedChannel(coverage=Decimal("0.7"))
    ),
    counterparty_settlement_history=category_from_channel(
        NetworkDerivedChannel(coverage=Decimal("0.7"))
    ),
    counterparty_track_record=category_from_channel(
        NetworkDerivedChannel(coverage=Decimal("0.7"))
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


# ── LENDER_REALISTIC_V2 ─────────────────────────────────────────────
# Same as LENDER_REALISTIC but constructed via from_hierarchy() to
# demonstrate and validate the hierarchical construction path.
LENDER_REALISTIC_V2 = InformationProfile.from_hierarchy(
    counterparty=CounterpartyAccess(
        cash=CategoryAccess(AccessLevel.NOISY, EstimationNoise(Decimal("0.15"))),
        assets=CategoryAccess(AccessLevel.NOISY, AggregateOnlyNoise()),
        liabilities=CategoryAccess(AccessLevel.NOISY, AggregateOnlyNoise()),
        net_worth=CategoryAccess(AccessLevel.NOISY, EstimationNoise(Decimal("0.20"))),
        liquidity_ratio=CategoryAccess(
            AccessLevel.NOISY, EstimationNoise(Decimal("0.20"))
        ),
        settlement_history=CategoryAccess(
            AccessLevel.NOISY, SampleNoise(Decimal("0.7"))
        ),
        default_history=CategoryAccess(
            AccessLevel.NOISY, SampleNoise(Decimal("0.7"))
        ),
        track_record=CategoryAccess(AccessLevel.NOISY, SampleNoise(Decimal("0.7"))),
        partial_settlement=CategoryAccess(AccessLevel.NOISY, AggregateOnlyNoise()),
        avg_shortfall=CategoryAccess(AccessLevel.NOISY, AggregateOnlyNoise()),
        connectivity=CategoryAccess(AccessLevel.NONE),
    ),
    instrument=InstrumentAccess(
        dealer_quotes=CategoryAccess(AccessLevel.NONE),
        vbt_anchors=CategoryAccess(AccessLevel.NONE),
        price_trends=CategoryAccess(AccessLevel.NONE),
        implied_default_prob=CategoryAccess(AccessLevel.NONE),
    ),
    transaction=TransactionAccess(
        bilateral_history=CategoryAccess(AccessLevel.PERFECT),
        obligation_graph=CategoryAccess(AccessLevel.NONE),
        cascade_risk=CategoryAccess(AccessLevel.NONE),
    ),
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


# ── RATING_AGENCY_REALISTIC ─────────────────────────────────────────
# Rating agency: good balance sheet access, good history, no market prices.
RATING_AGENCY_REALISTIC = InformationProfile(
    # I. Counterparty Balance Sheet — good access (NOISY 10%)
    counterparty_cash=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.10"))
    ),
    counterparty_assets=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.10"))
    ),
    counterparty_liabilities=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.10"))
    ),
    counterparty_net_worth=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.10"))
    ),
    counterparty_liquidity_ratio=CategoryAccess(
        AccessLevel.NOISY, EstimationNoise(Decimal("0.10"))
    ),
    # II. Counterparty History — good observation via sample
    counterparty_default_history=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.9"))
    ),
    counterparty_settlement_history=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.9"))
    ),
    counterparty_track_record=CategoryAccess(
        AccessLevel.NOISY, SampleNoise(Decimal("0.9"))
    ),
    counterparty_partial_settlement=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    counterparty_avg_shortfall=CategoryAccess(
        AccessLevel.NOISY, AggregateOnlyNoise()
    ),
    # IV. Bilateral — own data always perfect
    bilateral_history=CategoryAccess(AccessLevel.PERFECT),
    # V. Market Prices — rating agency not in secondary market
    dealer_quotes=CategoryAccess(AccessLevel.NONE),
    vbt_anchors=CategoryAccess(AccessLevel.NONE),
    price_trends=CategoryAccess(AccessLevel.NONE),
    implied_default_prob=CategoryAccess(AccessLevel.NONE),
    # VII. Network — no access
    obligation_graph=CategoryAccess(AccessLevel.NONE),
    counterparty_connectivity=CategoryAccess(AccessLevel.NONE),
    cascade_risk=CategoryAccess(AccessLevel.NONE),
)


# ── LENDER_WITH_RATINGS ─────────────────────────────────────────────
# Like LENDER_REALISTIC but default history uses InstitutionalChannel
# (from a rating agency) instead of network-derived observation.
LENDER_WITH_RATINGS = InformationProfile(
    # I. Counterparty Balance Sheet — noisy (same as LENDER_REALISTIC)
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
    # II. Counterparty History — institutional channel (rating agency)
    counterparty_default_history=category_from_channel(
        InstitutionalChannel(staleness_days=1, coverage=Decimal("0.8"))
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


# ── LENDER_RATINGS_BOUND ─────────────────────────────────────────
# Like LENDER_WITH_RATINGS but with explicit channel bindings that
# declare the lender's preferred information source order.
# Instead of the hard-coded waterfall, this preset says:
#   1. Use rating_registry first (institutional source)
#   2. Fall back to system_heuristic if no registry
# The dealer_risk_assessor is intentionally excluded — this lender
# relies on the rating agency, not the dealer's internal model.
LENDER_RATINGS_BOUND = InformationProfile(
    # I. Counterparty Balance Sheet — noisy (same as LENDER_WITH_RATINGS)
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
    # II. Counterparty History — institutional channel (rating agency)
    counterparty_default_history=category_from_channel(
        InstitutionalChannel(staleness_days=1, coverage=Decimal("0.8"))
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
    # Channel bindings: rating_registry first, then heuristic
    channel_bindings=(
        ChannelBinding(
            "default_prob", "rating_registry",
            InstitutionalChannel(staleness_days=1, coverage=Decimal("0.8")),
            priority=0,
        ),
        ChannelBinding(
            "default_prob", "system_heuristic",
            SelfDerivedChannel(sample_size=10),
            priority=1,
        ),
    ),
)
