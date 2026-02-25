"""Frozen dataclasses for trader, VBT, rating, and lender behavioral profiles."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.decision.risk_assessment import RiskAssessmentParams


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
    buy_reserve_fraction: Decimal = Decimal("1.0")  # reserve 100% of upcoming dues before buying
    trading_motive: str = "liquidity_only"

    def __post_init__(self) -> None:
        if not (1 <= self.planning_horizon <= 20):
            raise ValueError("planning_horizon must be between 1 and 20")
        if not (Decimal("0") <= self.buy_reserve_fraction <= Decimal("1")):
            raise ValueError("buy_reserve_fraction must be between 0 and 1")
        if self.trading_motive not in ("liquidity_only", "liquidity_then_earning", "unrestricted"):
            raise ValueError(
                "trading_motive must be one of: liquidity_only, liquidity_then_earning, unrestricted"
            )

    @property
    def base_risk_premium(self) -> Decimal:
        """Seller premium: always 0 (selling converts uncertainty to certainty)."""
        return Decimal("0")

    @property
    def buy_risk_premium(self) -> Decimal:
        """Buyer premium: 0.01 + 0.02 * risk_aversion."""
        return Decimal("0.01") + Decimal("0.02") * self.risk_aversion

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
        """Buyer look-ahead horizon = planning_horizon.

        Uses the full planning horizon so the reserve calculation accounts
        for ALL upcoming obligations, not just near-term ones.
        """
        return self.planning_horizon

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
    spread_scale: Decimal = Decimal("1.0")  # multiplicative scale on base spreads
    forward_weight: Decimal = Decimal("0.0")  # 0 = disabled (backward compat)
    stress_horizon: int = 5  # days to look ahead for stress estimation
    flow_sensitivity: Decimal = Decimal("0.0")  # 0 = disabled (backward compat)


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


@dataclass(frozen=True)
class LenderProfile:
    """Behavioral profile for NBFI lender decision-making.

    The lender uses kappa (system liquidity ratio) as a prior for default
    estimation, then adjusts based on each borrower's balance-sheet
    coverage ratio.

    Attributes:
        kappa: System liquidity ratio (L0/S1), used for base default estimate
        risk_aversion: 0=aggressive lending, 1=conservative (widens risk premium)
        planning_horizon: Days to look ahead for obligations (1-20)
        profit_target: Target return rate on loans
        max_loan_maturity: Maximum loan term in days
    """

    kappa: Decimal = Decimal("1.0")
    risk_aversion: Decimal = Decimal("0.3")
    planning_horizon: int = 5
    profit_target: Decimal = Decimal("0.05")
    max_loan_maturity: int = 10
    risk_assessment_params: RiskAssessmentParams | None = None  # None = disable (backward compat)
    warmup_observations: int = 10  # data points before Bayesian dominates coverage

    def __post_init__(self) -> None:
        if self.kappa <= Decimal("0"):
            raise ValueError("kappa must be positive")
        if self.profit_target < Decimal("0"):
            raise ValueError("profit_target cannot be negative")
        if not (1 <= self.planning_horizon <= 20):
            raise ValueError("planning_horizon must be between 1 and 20")
        if not (Decimal("0") <= self.risk_aversion <= Decimal("1")):
            raise ValueError("risk_aversion must be between 0 and 1")
        if not (self.max_loan_maturity >= 1):
            raise ValueError("max_loan_maturity must be >= 1")
        if self.warmup_observations < 1:
            raise ValueError("warmup_observations must be >= 1")

    @property
    def base_default_estimate(self) -> Decimal:
        """Kappa-informed base default rate: p ~ 1/(1+kappa)."""
        return Decimal("1") / (Decimal("1") + self.kappa)

    @property
    def risk_premium_scale(self) -> Decimal:
        """Higher risk aversion -> higher premium: 0.1 + 0.4 * risk_aversion."""
        return Decimal("0.1") + self.risk_aversion * Decimal("0.4")


@dataclass(frozen=True)
class BankProfile:
    """Treynor pricing parameters for active banks in the Kalecki ring.

    The CB corridor is derived from kappa (system liquidity ratio).
    Banks price inside this corridor using the Treynor pricing kernel.

    Attributes:
        r_base: Base corridor midpoint when kappa >= 1 (no stress)
        r_stress: Stress sensitivity — how much mid rises as kappa falls
        omega_base: Base corridor width when kappa >= 1
        omega_stress: Stress sensitivity for corridor width
        reserve_target_ratio: Target reserves as fraction of total deposits
        symmetric_capacity_ratio: X* as multiple of reserve target
        alpha: Sensitivity of midline to cash-tightness L*
        gamma: Sensitivity of midline to risk index rho
        loan_maturity_fraction: Bank loan maturity = maturity_days * this
        interest_period: Days per deposit interest accrual period
    """

    # CB corridor base parameters
    r_base: Decimal = Decimal("0.01")
    r_stress: Decimal = Decimal("0.04")
    omega_base: Decimal = Decimal("0.01")
    omega_stress: Decimal = Decimal("0.02")

    # Treynor kernel parameters
    reserve_target_ratio: Decimal = Decimal("0.10")
    symmetric_capacity_ratio: Decimal = Decimal("2.0")
    alpha: Decimal = Decimal("0.005")
    gamma: Decimal = Decimal("0.002")

    # Loan and interest parameters
    loan_maturity_fraction: Decimal = Decimal("0.5")
    interest_period: int = 2

    # Per-borrower credit risk loading: r_L = treynor_base + credit_risk_loading × P_default
    credit_risk_loading: Decimal = Decimal("0")  # 0 = no per-borrower pricing (backward compat)
    # Credit rationing threshold: refuse loan when P_default > max_borrower_risk
    max_borrower_risk: Decimal = Decimal("1.0")  # 1.0 = never refuse (backward compat)

    def __post_init__(self) -> None:
        if self.r_base < Decimal("0"):
            raise ValueError("r_base must be non-negative")
        if self.r_stress < Decimal("0"):
            raise ValueError("r_stress must be non-negative")
        if self.omega_base < Decimal("0"):
            raise ValueError("omega_base must be non-negative")
        if self.omega_stress < Decimal("0"):
            raise ValueError("omega_stress must be non-negative")
        if not (Decimal("0") < self.reserve_target_ratio <= Decimal("1")):
            raise ValueError("reserve_target_ratio must be in (0, 1]")
        if self.symmetric_capacity_ratio <= Decimal("0"):
            raise ValueError("symmetric_capacity_ratio must be positive")
        if not (1 <= self.interest_period <= 10):
            raise ValueError("interest_period must be between 1 and 10")
        if not (Decimal("0") < self.loan_maturity_fraction <= Decimal("1")):
            raise ValueError("loan_maturity_fraction must be in (0, 1]")
        if self.credit_risk_loading < Decimal("0"):
            raise ValueError("credit_risk_loading must be non-negative")
        if not (Decimal("0") < self.max_borrower_risk <= Decimal("1")):
            raise ValueError("max_borrower_risk must be in (0, 1]")

    def _stress_factor(self, kappa: Decimal) -> Decimal:
        """Common stress factor: max(0, 1-kappa) / (1+kappa)."""
        return max(Decimal("0"), Decimal("1") - kappa) / (Decimal("1") + kappa)

    def corridor_mid(self, kappa: Decimal) -> Decimal:
        """r_mid = r_base + r_stress * max(0, 1-kappa) / (1+kappa)."""
        return self.r_base + self.r_stress * self._stress_factor(kappa)

    def corridor_width(self, kappa: Decimal) -> Decimal:
        """Omega = omega_base + omega_stress * max(0, 1-kappa) / (1+kappa)."""
        return self.omega_base + self.omega_stress * self._stress_factor(kappa)

    def r_floor(self, kappa: Decimal) -> Decimal:
        """Reserve remuneration rate (CB floor)."""
        return self.corridor_mid(kappa) - self.corridor_width(kappa) / 2

    def r_ceiling(self, kappa: Decimal) -> Decimal:
        """CB lending rate (corridor ceiling)."""
        return self.corridor_mid(kappa) + self.corridor_width(kappa) / 2

    def loan_maturity(self, maturity_days: int) -> int:
        """Bank loan maturity in days."""
        return max(2, int(maturity_days * self.loan_maturity_fraction))
