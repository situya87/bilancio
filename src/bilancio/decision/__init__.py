"""Decision module: trader and VBT behavioral profiles.

Provides parameterized profiles for trader risk behavior and VBT pricing
sensitivity, replacing hardcoded constants across the dealer subsystem.
"""

from bilancio.decision.profiles import TraderProfile, VBTProfile
from bilancio.decision.presets import BASELINE, CAUTIOUS, AGGRESSIVE

__all__ = [
    "TraderProfile",
    "VBTProfile",
    "BASELINE",
    "CAUTIOUS",
    "AGGRESSIVE",
]
