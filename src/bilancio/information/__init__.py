"""Information access framework for agent observation and noise.

Provides parameterized information profiles that control what each agent
can observe about the system, counterparties, and market conditions.
Mirrors the pattern of ``bilancio.decision`` for behavioral profiles.
"""

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
from bilancio.information.presets import LENDER_REALISTIC, OMNISCIENT, TRADER_BASIC
from bilancio.information.service import InformationService

__all__ = [
    "AccessLevel",
    "AggregateOnlyNoise",
    "BilateralOnlyNoise",
    "CategoryAccess",
    "EstimationNoise",
    "InformationProfile",
    "InformationService",
    "LENDER_REALISTIC",
    "LagNoise",
    "NoiseConfig",
    "OMNISCIENT",
    "SampleNoise",
    "TRADER_BASIC",
]
