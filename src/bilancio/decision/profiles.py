"""Frozen dataclasses for trader and VBT behavioral profiles."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TraderProfile:
    """Behavioral profile for trader risk assessment and eligibility.

    Attributes:
        risk_aversion: 0=risk-neutral, 1=max risk-averse
        planning_horizon: days to look ahead (1-20)
        aggressiveness: 0=conservative buyer, 1=current behavior
        default_observability: 0=ignore defaults, 1=full tracking
    """

    risk_aversion: Decimal = Decimal("0")
    planning_horizon: int = 10
    aggressiveness: Decimal = Decimal("1.0")
    default_observability: Decimal = Decimal("1.0")

    @property
    def base_risk_premium(self) -> Decimal:
        """Maps risk_aversion to base_risk_premium: 0.02 + 0.08 * risk_aversion."""
        return Decimal("0.02") + Decimal("0.08") * self.risk_aversion

    @property
    def buy_premium_multiplier(self) -> Decimal:
        """Risk-averse traders demand higher buy premium: 1.0 + risk_aversion."""
        return Decimal("1.0") + self.risk_aversion

    @property
    def sell_horizon(self) -> int:
        """Seller look-ahead horizon = planning_horizon."""
        return self.planning_horizon

    @property
    def buy_horizon(self) -> int:
        """Buyer look-ahead horizon = planning_horizon // 2."""
        return max(1, self.planning_horizon // 2)

    @property
    def surplus_threshold_factor(self) -> Decimal:
        """Factor for buyer surplus threshold: 1 - aggressiveness.

        aggressiveness=1: surplus > 0 (current behavior, buy eagerly)
        aggressiveness=0: surplus > face_value (very conservative)
        """
        return Decimal("1") - self.aggressiveness


@dataclass(frozen=True)
class VBTProfile:
    """Behavioral profile for VBT pricing sensitivity.

    Attributes:
        mid_sensitivity: 0=ignore defaults, 1=full tracking (current)
        spread_sensitivity: 0=fixed spread (current), 1=widen with defaults
    """

    mid_sensitivity: Decimal = Decimal("1.0")
    spread_sensitivity: Decimal = Decimal("0.0")
