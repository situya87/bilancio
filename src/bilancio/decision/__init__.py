"""Decision module: trader and VBT behavioral profiles.

Provides parameterized profiles for trader risk behavior and VBT pricing
sensitivity, replacing hardcoded constants across the dealer subsystem.
"""

from bilancio.decision.profiles import RatingProfile, TraderProfile, VBTProfile
from bilancio.decision.presets import BASELINE, CAUTIOUS, AGGRESSIVE
from bilancio.decision.protocols import (
    CounterpartyScreener,
    FixedMaturitySelector,
    FixedPortfolioStrategy,
    InstrumentSelector,
    InstrumentValuer,
    LinearPricer,
    PortfolioStrategy,
    ThresholdScreener,
    TransactionPricer,
    VBTPricingModel,
)
from bilancio.decision.valuers import (
    CoverageRatioValuer,
    CreditAdjustedVBTPricing,
    EVHoldValuer,
)

__all__ = [
    "AGGRESSIVE",
    "BASELINE",
    "CAUTIOUS",
    "CounterpartyScreener",
    "CoverageRatioValuer",
    "CreditAdjustedVBTPricing",
    "EVHoldValuer",
    "FixedMaturitySelector",
    "FixedPortfolioStrategy",
    "InstrumentSelector",
    "InstrumentValuer",
    "LinearPricer",
    "PortfolioStrategy",
    "RatingProfile",
    "ThresholdScreener",
    "TraderProfile",
    "TransactionPricer",
    "VBTPricingModel",
    "VBTProfile",
]
