"""Information profile: per-agent configuration of what can be observed.

An InformationProfile is a frozen dataclass (like TraderProfile in the
decision module) that specifies access levels and noise for each
information category.  Immutable during simulation; changed between
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from bilancio.information.levels import AccessLevel
from bilancio.information.noise import NoiseConfig


@dataclass(frozen=True)
class CategoryAccess:
    """Access configuration for one information element.

    Attributes:
        level: NONE / NOISY / PERFECT
        noise: Required when level == NOISY; ignored otherwise
    """

    level: AccessLevel = AccessLevel.PERFECT
    noise: Optional[NoiseConfig] = None

    def __post_init__(self) -> None:
        if self.level == AccessLevel.NOISY and self.noise is None:
            raise ValueError(
                "noise config is required when level is NOISY"
            )
        if self.level != AccessLevel.NOISY and self.noise is not None:
            raise ValueError(
                f"noise config must be None when level is {self.level.value}"
            )


# Sentinel for "use default" (PERFECT with no noise)
_DEFAULT = CategoryAccess()


@dataclass(frozen=True)
class InformationProfile:
    """Top-level information access profile for an agent.

    Groups ~32 information elements into 7 categories.  Each field
    is a CategoryAccess with a sensible default (PERFECT for full
    backward compatibility).

    Categories:
        I.   Counterparty Balance Sheet
        II.  Counterparty Payment/Default History
        III. Instrument-Type Statistics (System-Wide)
        IV.  Counterparty + Instrument Specific
        V.   Market Prices
        VI.  System Conditions
        VII. Network Topology
    """

    # ── I. Counterparty Balance Sheet ──────────────────────────────────
    counterparty_cash: CategoryAccess = _DEFAULT
    counterparty_assets: CategoryAccess = _DEFAULT
    counterparty_liabilities: CategoryAccess = _DEFAULT
    counterparty_net_worth: CategoryAccess = _DEFAULT
    counterparty_liquidity_ratio: CategoryAccess = _DEFAULT

    # ── II. Counterparty Payment / Default History ─────────────────────
    counterparty_settlement_history: CategoryAccess = _DEFAULT
    counterparty_default_history: CategoryAccess = _DEFAULT
    counterparty_track_record: CategoryAccess = _DEFAULT
    counterparty_partial_settlement: CategoryAccess = _DEFAULT
    counterparty_avg_shortfall: CategoryAccess = _DEFAULT

    # ── III. Instrument-Type Statistics (System-Wide) ──────────────────
    instrument_default_rate: CategoryAccess = _DEFAULT
    instrument_bucket_default_rate: CategoryAccess = _DEFAULT
    instrument_issuer_kind_rate: CategoryAccess = _DEFAULT
    instrument_recovery_rate: CategoryAccess = _DEFAULT

    # ── IV. Counterparty + Instrument Specific ─────────────────────────
    counterparty_instrument_history: CategoryAccess = _DEFAULT
    counterparty_bucket_default_rate: CategoryAccess = _DEFAULT
    bilateral_history: CategoryAccess = _DEFAULT
    own_exposure: CategoryAccess = _DEFAULT  # Always PERFECT (own data)

    # ── V. Market Prices ───────────────────────────────────────────────
    dealer_quotes: CategoryAccess = _DEFAULT
    vbt_anchors: CategoryAccess = _DEFAULT
    price_trends: CategoryAccess = _DEFAULT
    implied_default_prob: CategoryAccess = _DEFAULT

    # ── VI. System Conditions ──────────────────────────────────────────
    aggregate_default_rate: CategoryAccess = _DEFAULT
    system_liquidity: CategoryAccess = _DEFAULT
    # day_counter and maturity_calendar are always PERFECT (public info)

    # ── VII. Network Topology ──────────────────────────────────────────
    obligation_graph: CategoryAccess = _DEFAULT
    counterparty_connectivity: CategoryAccess = _DEFAULT
    cascade_risk: CategoryAccess = _DEFAULT
    exposure_concentration: CategoryAccess = _DEFAULT  # Always PERFECT (own data)
