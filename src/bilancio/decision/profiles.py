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

    def __post_init__(self) -> None:
        if not (1 <= self.planning_horizon <= 20):
            raise ValueError("planning_horizon must be between 1 and 20")

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


@dataclass(frozen=True)
class RatingProfile:
    """Behavioral profile for rating agency methodology.

    Attributes:
        lookback_window: Days of history to consider (1-30)
        balance_sheet_weight: Weight for balance sheet component (0-1)
        history_weight: Weight for default history component (0-1)
        conservatism_bias: Additive bias toward higher default probs (0-0.2)
        coverage_fraction: Fraction of eligible agents to rate each day (0-1)
        no_data_prior: Default probability when no data is available (0-1)
    """

    lookback_window: int = 5
    balance_sheet_weight: Decimal = Decimal("0.4")
    history_weight: Decimal = Decimal("0.6")
    conservatism_bias: Decimal = Decimal("0.02")
    coverage_fraction: Decimal = Decimal("0.8")
    no_data_prior: Decimal = Decimal("0.15")

    def __post_init__(self) -> None:
        if not (1 <= self.lookback_window <= 30):
            raise ValueError("lookback_window must be between 1 and 30")
        if not (Decimal("0") <= self.balance_sheet_weight <= Decimal("1")):
            raise ValueError("balance_sheet_weight must be between 0 and 1")
        if not (Decimal("0") <= self.history_weight <= Decimal("1")):
            raise ValueError("history_weight must be between 0 and 1")
        if not (Decimal("0") <= self.conservatism_bias <= Decimal("0.2")):
            raise ValueError("conservatism_bias must be between 0 and 0.2")
        if not (Decimal("0") < self.coverage_fraction <= Decimal("1")):
            raise ValueError("coverage_fraction must be in (0, 1]")
        if not (Decimal("0") < self.no_data_prior < Decimal("1")):
            raise ValueError("no_data_prior must be in (0, 1)")
