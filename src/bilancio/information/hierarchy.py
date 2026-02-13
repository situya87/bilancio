"""Hierarchical sub-profiles grouping the 28 flat CategoryAccess fields.

Organises information access into four levels that mirror the
information / decision hierarchy from Plan 033.
"""

from __future__ import annotations

from dataclasses import dataclass

from bilancio.information.profile import CategoryAccess

__all__ = [
    "SystemAccess",
    "CounterpartyAccess",
    "InstrumentAccess",
    "TransactionAccess",
]

_DEFAULT = CategoryAccess()  # PERFECT access, same pattern as profile.py


@dataclass(frozen=True)
class SystemAccess:
    """Level 1: System-wide information."""

    aggregate_default_rate: CategoryAccess = _DEFAULT  # from VI
    system_liquidity: CategoryAccess = _DEFAULT  # from VI
    instrument_default_rate: CategoryAccess = _DEFAULT  # from III
    instrument_bucket_default_rate: CategoryAccess = _DEFAULT  # from III
    instrument_issuer_kind_rate: CategoryAccess = _DEFAULT  # from III
    instrument_recovery_rate: CategoryAccess = _DEFAULT  # from III
    # 6 fields


@dataclass(frozen=True)
class CounterpartyAccess:
    """Level 2: Counterparty-specific information."""

    cash: CategoryAccess = _DEFAULT  # from I  (counterparty_cash)
    assets: CategoryAccess = _DEFAULT  # from I  (counterparty_assets)
    liabilities: CategoryAccess = _DEFAULT  # from I  (counterparty_liabilities)
    net_worth: CategoryAccess = _DEFAULT  # from I  (counterparty_net_worth)
    liquidity_ratio: CategoryAccess = _DEFAULT  # from I  (counterparty_liquidity_ratio)
    settlement_history: CategoryAccess = _DEFAULT  # from II (counterparty_settlement_history)
    default_history: CategoryAccess = _DEFAULT  # from II (counterparty_default_history)
    track_record: CategoryAccess = _DEFAULT  # from II (counterparty_track_record)
    partial_settlement: CategoryAccess = _DEFAULT  # from II (counterparty_partial_settlement)
    avg_shortfall: CategoryAccess = _DEFAULT  # from II (counterparty_avg_shortfall)
    connectivity: CategoryAccess = _DEFAULT  # from VII (counterparty_connectivity)
    # 11 fields


@dataclass(frozen=True)
class InstrumentAccess:
    """Level 3: Instrument/market information."""

    dealer_quotes: CategoryAccess = _DEFAULT  # from V
    vbt_anchors: CategoryAccess = _DEFAULT  # from V
    price_trends: CategoryAccess = _DEFAULT  # from V
    implied_default_prob: CategoryAccess = _DEFAULT  # from V
    # 4 fields


@dataclass(frozen=True)
class TransactionAccess:
    """Level 4: Counterparty x Instrument specific."""

    counterparty_instrument_history: CategoryAccess = _DEFAULT  # from IV
    counterparty_bucket_default_rate: CategoryAccess = _DEFAULT  # from IV
    bilateral_history: CategoryAccess = _DEFAULT  # from IV
    own_exposure: CategoryAccess = _DEFAULT  # from IV
    obligation_graph: CategoryAccess = _DEFAULT  # from VII
    cascade_risk: CategoryAccess = _DEFAULT  # from VII
    exposure_concentration: CategoryAccess = _DEFAULT  # from VII
    # 7 fields
