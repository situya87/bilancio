"""Information profile: per-agent configuration of what can be observed.

An InformationProfile is a frozen dataclass (like TraderProfile in the
decision module) that specifies access levels and noise for each
information category.  Immutable during simulation; changed between
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from bilancio.information.levels import AccessLevel
from bilancio.information.noise import NoiseConfig

if TYPE_CHECKING:
    from bilancio.information.channels import ChannelBinding


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


#: Default field value for InformationProfile — PERFECT access with no noise.
#: Used as the default for all CategoryAccess fields so that an
#: ``InformationProfile()`` with no arguments is fully omniscient.
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

    # ── Channel Bindings (optional) ──────────────────────────────────
    channel_bindings: tuple[ChannelBinding, ...] = ()

    # ── Hierarchical sub-profile properties ──────────────────────────

    @property
    def system(self) -> "SystemAccess":
        """Level 1: System-wide information access."""
        from bilancio.information.hierarchy import SystemAccess

        return SystemAccess(
            aggregate_default_rate=self.aggregate_default_rate,
            system_liquidity=self.system_liquidity,
            instrument_default_rate=self.instrument_default_rate,
            instrument_bucket_default_rate=self.instrument_bucket_default_rate,
            instrument_issuer_kind_rate=self.instrument_issuer_kind_rate,
            instrument_recovery_rate=self.instrument_recovery_rate,
        )

    @property
    def counterparty(self) -> "CounterpartyAccess":
        """Level 2: Counterparty-specific information access."""
        from bilancio.information.hierarchy import CounterpartyAccess

        return CounterpartyAccess(
            cash=self.counterparty_cash,
            assets=self.counterparty_assets,
            liabilities=self.counterparty_liabilities,
            net_worth=self.counterparty_net_worth,
            liquidity_ratio=self.counterparty_liquidity_ratio,
            settlement_history=self.counterparty_settlement_history,
            default_history=self.counterparty_default_history,
            track_record=self.counterparty_track_record,
            partial_settlement=self.counterparty_partial_settlement,
            avg_shortfall=self.counterparty_avg_shortfall,
            connectivity=self.counterparty_connectivity,
        )

    @property
    def instrument(self) -> "InstrumentAccess":
        """Level 3: Instrument/market information access."""
        from bilancio.information.hierarchy import InstrumentAccess

        return InstrumentAccess(
            dealer_quotes=self.dealer_quotes,
            vbt_anchors=self.vbt_anchors,
            price_trends=self.price_trends,
            implied_default_prob=self.implied_default_prob,
        )

    @property
    def transaction(self) -> "TransactionAccess":
        """Level 4: Counterparty x Instrument specific access."""
        from bilancio.information.hierarchy import TransactionAccess

        return TransactionAccess(
            counterparty_instrument_history=self.counterparty_instrument_history,
            counterparty_bucket_default_rate=self.counterparty_bucket_default_rate,
            bilateral_history=self.bilateral_history,
            own_exposure=self.own_exposure,
            obligation_graph=self.obligation_graph,
            cascade_risk=self.cascade_risk,
            exposure_concentration=self.exposure_concentration,
        )

    @classmethod
    def from_hierarchy(
        cls,
        system: Optional["SystemAccess"] = None,
        counterparty: Optional["CounterpartyAccess"] = None,
        instrument: Optional["InstrumentAccess"] = None,
        transaction: Optional["TransactionAccess"] = None,
        channel_bindings: tuple[ChannelBinding, ...] = (),
    ) -> "InformationProfile":
        """Construct an InformationProfile from hierarchical sub-profiles.

        Any sub-profile that is ``None`` defaults to all-PERFECT access.
        """
        from bilancio.information.hierarchy import (
            CounterpartyAccess,
            InstrumentAccess,
            SystemAccess,
            TransactionAccess,
        )

        s = system or SystemAccess()
        c = counterparty or CounterpartyAccess()
        i = instrument or InstrumentAccess()
        t = transaction or TransactionAccess()

        return cls(
            # System (VI + III)
            aggregate_default_rate=s.aggregate_default_rate,
            system_liquidity=s.system_liquidity,
            instrument_default_rate=s.instrument_default_rate,
            instrument_bucket_default_rate=s.instrument_bucket_default_rate,
            instrument_issuer_kind_rate=s.instrument_issuer_kind_rate,
            instrument_recovery_rate=s.instrument_recovery_rate,
            # Counterparty (I + II + VII.connectivity)
            counterparty_cash=c.cash,
            counterparty_assets=c.assets,
            counterparty_liabilities=c.liabilities,
            counterparty_net_worth=c.net_worth,
            counterparty_liquidity_ratio=c.liquidity_ratio,
            counterparty_settlement_history=c.settlement_history,
            counterparty_default_history=c.default_history,
            counterparty_track_record=c.track_record,
            counterparty_partial_settlement=c.partial_settlement,
            counterparty_avg_shortfall=c.avg_shortfall,
            counterparty_connectivity=c.connectivity,
            # Instrument (V)
            dealer_quotes=i.dealer_quotes,
            vbt_anchors=i.vbt_anchors,
            price_trends=i.price_trends,
            implied_default_prob=i.implied_default_prob,
            # Transaction (IV + VII)
            counterparty_instrument_history=t.counterparty_instrument_history,
            counterparty_bucket_default_rate=t.counterparty_bucket_default_rate,
            bilateral_history=t.bilateral_history,
            own_exposure=t.own_exposure,
            obligation_graph=t.obligation_graph,
            cascade_risk=t.cascade_risk,
            exposure_concentration=t.exposure_concentration,
            channel_bindings=channel_bindings,
        )
