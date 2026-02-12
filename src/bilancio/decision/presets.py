"""Named presets for trader and VBT profiles."""

from decimal import Decimal

from bilancio.decision.profiles import TraderProfile, VBTProfile

BASELINE = (TraderProfile(), VBTProfile())

CAUTIOUS = (
    TraderProfile(
        risk_aversion=Decimal("0.5"),
        planning_horizon=15,
        aggressiveness=Decimal("0.5"),
    ),
    VBTProfile(),
)

AGGRESSIVE = (
    TraderProfile(
        risk_aversion=Decimal("0"),
        planning_horizon=5,
        aggressiveness=Decimal("1.0"),
    ),
    VBTProfile(
        mid_sensitivity=Decimal("1.0"),
        spread_sensitivity=Decimal("0.5"),
    ),
)
