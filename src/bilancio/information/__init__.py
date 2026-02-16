"""Information access framework for agent observation and noise.

Provides parameterized information profiles that control what each agent
can observe about the system, counterparties, and market conditions.
Mirrors the pattern of ``bilancio.decision`` for behavioral profiles.
"""

from bilancio.information.channels import (
    Channel,
    ChannelBinding,
    InstitutionalChannel,
    MarketDerivedChannel,
    NetworkDerivedChannel,
    SelfDerivedChannel,
    category_from_channel,
    derive_noise,
)
from bilancio.information.estimates import Estimate
from bilancio.information.hierarchy import (
    CounterpartyAccess,
    InstrumentAccess,
    SystemAccess,
    TransactionAccess,
)
from bilancio.information.levels import AccessLevel
from bilancio.information.noise import (
    AggregateOnlyNoise,
    BilateralOnlyNoise,
    EstimationNoise,
    LagNoise,
    NoiseConfig,
    SampleNoise,
)
from bilancio.information.presets import (
    DEALER_MARKET_OBSERVER,
    LENDER_CHANNEL_BASED,
    LENDER_RATINGS_BOUND,
    LENDER_REALISTIC,
    LENDER_REALISTIC_V2,
    OMNISCIENT,
    TRADER_BASIC,
)
from bilancio.information.profile import CategoryAccess, InformationProfile
from bilancio.information.service import InformationService
from bilancio.information.views import (
    CounterpartyView,
    InstrumentView,
    SystemView,
    TransactionView,
)

__all__ = [
    "AccessLevel",
    "AggregateOnlyNoise",
    "BilateralOnlyNoise",
    "CategoryAccess",
    "Channel",
    "ChannelBinding",
    "CounterpartyAccess",
    "CounterpartyView",
    "DEALER_MARKET_OBSERVER",
    "Estimate",
    "EstimationNoise",
    "InformationProfile",
    "InformationService",
    "InstitutionalChannel",
    "InstrumentAccess",
    "InstrumentView",
    "LENDER_CHANNEL_BASED",
    "LENDER_RATINGS_BOUND",
    "LENDER_REALISTIC",
    "LENDER_REALISTIC_V2",
    "LagNoise",
    "MarketDerivedChannel",
    "NetworkDerivedChannel",
    "NoiseConfig",
    "OMNISCIENT",
    "SampleNoise",
    "SelfDerivedChannel",
    "SystemAccess",
    "SystemView",
    "TRADER_BASIC",
    "TransactionAccess",
    "TransactionView",
    "category_from_channel",
    "derive_noise",
]
