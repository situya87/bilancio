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
    LinearPricer,
    PortfolioStrategy,
    ThresholdScreener,
    TransactionPricer,
)

__all__ = [
    "AGGRESSIVE",
    "BASELINE",
    "CAUTIOUS",
    "CounterpartyScreener",
    "FixedMaturitySelector",
    "FixedPortfolioStrategy",
    "InstrumentSelector",
    "LinearPricer",
    "PortfolioStrategy",
    "RatingProfile",
    "ThresholdScreener",
    "TraderProfile",
    "TransactionPricer",
    "VBTProfile",
]
