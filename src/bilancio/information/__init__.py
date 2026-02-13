"""Information access framework for agent observation and noise.

Provides parameterized information profiles that control what each agent
can observe about the system, counterparties, and market conditions.
Mirrors the pattern of ``bilancio.decision`` for behavioral profiles.
"""

from bilancio.information.channels import (
    Channel,
    InstitutionalChannel,
    MarketDerivedChannel,
    NetworkDerivedChannel,
    SelfDerivedChannel,
    category_from_channel,
    derive_noise,
)
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
from bilancio.information.profile import CategoryAccess, InformationProfile
from bilancio.information.presets import (
    LENDER_CHANNEL_BASED,
    LENDER_REALISTIC,
    LENDER_REALISTIC_V2,
    OMNISCIENT,
    TRADER_BASIC,
)
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
    "CounterpartyAccess",
    "CounterpartyView",
    "EstimationNoise",
    "InformationProfile",
    "InformationService",
    "InstitutionalChannel",
    "InstrumentAccess",
    "InstrumentView",
    "LENDER_CHANNEL_BASED",
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
