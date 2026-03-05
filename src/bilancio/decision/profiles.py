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

    # Adaptive flags (Plan 050)
    adaptive_planning_horizon: bool = False   # [PRE] scale planning_horizon with maturity_days
    adaptive_risk_aversion: bool = False      # [RUN] effective_ra responds to observed defaults
    adaptive_reserves: bool = False           # [RUN] buy_reserve_fraction responds to stress
    adaptive_ev_term_structure: bool = False  # [RUN] EV uses maturity-dependent p_default

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

    # Adaptive flags (Plan 050)
    adaptive_term_structure: bool = False     # [PRE] per-bucket M = rho * (1-h)^tau
    adaptive_base_spreads: bool = False       # [PRE] base spreads scale with kappa stress
    adaptive_stress_horizon: bool = False     # [PRE] stress_horizon scales with maturity_days
    adaptive_convex_spreads: bool = False     # [RUN] O = base + sens * p^2/(1-p)
    adaptive_per_bucket_tracking: bool = False  # [RUN] per-bucket default rates
    adaptive_issuer_pricing: bool = False     # [RUN] inventory-weighted issuer p_default
    term_strength: Decimal = Decimal("0.5")   # dampening for term-structure hazard rate


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
    min_coverage_ratio: Decimal = Decimal("0.5")  # balance-sheet coverage gate (Plan 044)

    # Phase 1A: Maturity matching (Plan 046)
    maturity_matching: bool = False  # match loan maturity to borrower's next receivable
    min_loan_maturity: int = 2  # floor for loan maturity when matching

    # Phase 1B: Concentration limits (Plan 046)
    max_loans_per_borrower_per_day: int = 0  # 0 = unlimited (backward compat)

    # Phase 2: Cascade-aware ranking (Plan 046)
    ranking_mode: str = "profit"  # "profit" | "cascade" | "blended"
    cascade_weight: Decimal = Decimal("0.5")  # weight in blended mode

    # Phase 3: Graduated coverage gate (Plan 046)
    coverage_mode: str = "gate"  # "gate" | "graduated"
    coverage_penalty_scale: Decimal = Decimal("0.10")  # rate premium per unit below threshold

    # Phase 4: Preventive lending (Plan 046)
    preventive_lending: bool = False
    prevention_threshold: Decimal = Decimal("0.3")  # min issuer p_default to trigger

    # Adaptive flags (Plan 050)
    adaptive_risk_aversion: bool = False      # [PRE] risk_aversion calibrates to kappa
    adaptive_profit_target: bool = False      # [PRE] profit_target anchors to CB corridor
    adaptive_loan_maturity: bool = False      # [PRE] max_loan_maturity scales with maturity_days
    adaptive_rates: bool = False              # [RUN] daily rate multiplier from portfolio losses
    adaptive_capital_conservation: bool = False  # [RUN] scale exposure limits by utilization
    adaptive_prevention: bool = False         # [RUN] dynamic prevention threshold

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
        if self.min_loan_maturity < 1:
            raise ValueError("min_loan_maturity must be >= 1")
        if self.max_loans_per_borrower_per_day < 0:
            raise ValueError("max_loans_per_borrower_per_day must be >= 0")
        if self.ranking_mode not in ("profit", "cascade", "blended"):
            raise ValueError("ranking_mode must be one of: profit, cascade, blended")
        if not (Decimal("0") <= self.cascade_weight <= Decimal("1")):
            raise ValueError("cascade_weight must be between 0 and 1")
        if self.coverage_mode not in ("gate", "graduated"):
            raise ValueError("coverage_mode must be one of: gate, graduated")
        if self.coverage_penalty_scale < Decimal("0"):
            raise ValueError("coverage_penalty_scale must be non-negative")
        if not (Decimal("0") < self.prevention_threshold < Decimal("1")):
            raise ValueError("prevention_threshold must be in (0, 1)")

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

    # Borrower assessment (Plan 042): bank examines borrower's balance sheet
    # coverage = (liquid + quality_receivables - obligations) / loan_repayment
    # 0.0 = never reject based on coverage (backward compat)
    min_coverage_ratio: Decimal = Decimal("0.0")

    # Exposure limits (Plan 042)
    max_single_exposure_ratio: Decimal = Decimal("0.20")  # Max lending to single borrower as fraction of total loan capacity
    max_total_exposure_ratio: Decimal = Decimal("1.50")  # Max total loans as multiple of initial reserves
    max_daily_lending_ratio: Decimal = Decimal("0.50")  # Max new loans per day as fraction of current reserves

    # Adaptive flags (Plan 050)
    adaptive_corridor: bool = False           # [PRE] corridor_mid/width incorporate mu, c

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
        if self.min_coverage_ratio < Decimal("0"):
            raise ValueError("min_coverage_ratio must be non-negative")
        if not (Decimal("0") < self.max_single_exposure_ratio <= Decimal("1")):
            raise ValueError("max_single_exposure_ratio must be in (0, 1]")
        if self.max_total_exposure_ratio <= Decimal("0"):
            raise ValueError("max_total_exposure_ratio must be positive")
        if not (Decimal("0") < self.max_daily_lending_ratio <= Decimal("1")):
            raise ValueError("max_daily_lending_ratio must be in (0, 1]")

    def _stress_factor(self, kappa: Decimal) -> Decimal:
        """Common stress factor: max(0, 1-kappa) / (1+kappa)."""
        return max(Decimal("0"), Decimal("1") - kappa) / (Decimal("1") + kappa)

    def _combined_stress(self, kappa: Decimal, mu: Decimal, c: Decimal) -> Decimal:
        """Combined stress factor incorporating mu (timing) and c (concentration)."""
        base = self._stress_factor(kappa)
        timing_stress = (Decimal("1") - mu) ** 2  # front-loaded = more stress
        concentration_stress = Decimal("1") / (Decimal("1") + c)  # lower c = more stress
        return base + Decimal("0.05") * timing_stress + Decimal("0.05") * concentration_stress

    def corridor_mid(self, kappa: Decimal, mu: Decimal | None = None, c: Decimal | None = None) -> Decimal:
        """r_mid = r_base + r_stress * stress_factor."""
        if self.adaptive_corridor and mu is not None and c is not None:
            return self.r_base + self.r_stress * self._combined_stress(kappa, mu, c)
        return self.r_base + self.r_stress * self._stress_factor(kappa)

    def corridor_width(self, kappa: Decimal, mu: Decimal | None = None, c: Decimal | None = None) -> Decimal:
        """Omega = omega_base + omega_stress * stress_factor."""
        if self.adaptive_corridor and mu is not None and c is not None:
            return self.omega_base + self.omega_stress * self._combined_stress(kappa, mu, c)
        return self.omega_base + self.omega_stress * self._stress_factor(kappa)

    def r_floor(self, kappa: Decimal, mu: Decimal | None = None, c: Decimal | None = None) -> Decimal:
        """Reserve remuneration rate (CB floor)."""
        return self.corridor_mid(kappa, mu, c) - self.corridor_width(kappa, mu, c) / 2

    def r_ceiling(self, kappa: Decimal, mu: Decimal | None = None, c: Decimal | None = None) -> Decimal:
        """CB lending rate (corridor ceiling)."""
        return self.corridor_mid(kappa, mu, c) + self.corridor_width(kappa, mu, c) / 2

    def loan_maturity(self, maturity_days: int) -> int:
        """Bank loan maturity in days."""
        return max(2, int(maturity_days * self.loan_maturity_fraction))
