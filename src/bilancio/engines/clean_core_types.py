"""Dataclasses shared by the clean-core scenario engine."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from bilancio.engines.termination import StabilitySnapshot, StopReason

ZERO = Decimal("0")


@dataclass(frozen=True)
class CleanAgent:
    id: str
    kind: str
    name: str


@dataclass
class CleanPayable:
    id: str
    debtor: str
    creditor: str
    amount: Decimal
    due_day: int
    maturity_distance: int
    alias: str | None = None
    settled: bool = False


@dataclass(frozen=True)
class CleanLenderConfig:
    base_rate: Decimal
    risk_premium_scale: Decimal
    max_single_exposure: Decimal
    max_total_exposure: Decimal
    maturity_days: int
    horizon: int
    min_shortfall: Decimal
    max_default_prob: Decimal
    kappa: Decimal | None
    risk_aversion: Decimal
    planning_horizon: int
    profit_target: Decimal
    max_loan_maturity: int
    initial_prior: Decimal
    max_ring_maturity: int | None
    min_coverage_ratio: Decimal
    maturity_matching: bool
    min_loan_maturity: int
    max_loans_per_borrower_per_day: int
    ranking_mode: str
    cascade_weight: Decimal
    coverage_mode: str
    coverage_penalty_scale: Decimal
    preventive_lending: bool
    prevention_threshold: Decimal
    marginal_relief_min_ratio: Decimal
    stress_risk_premium_scale: Decimal
    daily_expected_loss_budget_ratio: Decimal
    run_expected_loss_budget_ratio: Decimal
    stop_loss_realized_ratio: Decimal
    high_risk_default_threshold: Decimal
    high_risk_maturity_cap: int
    collateralized_terms: bool
    collateral_advance_rate: Decimal
    adaptive_capital_conservation: bool
    info_cash_visibility: str
    info_cash_noise: Decimal
    info_liabilities_visibility: str
    info_history_visibility: str
    info_history_sample_rate: Decimal
    info_network_visibility: str
    info_market_visibility: str


@dataclass(frozen=True)
class CleanRatingConfig:
    info_profile: str
    lookback_window: int
    balance_sheet_weight: Decimal
    history_weight: Decimal
    conservatism_bias: Decimal
    coverage_fraction: Decimal
    no_data_prior: Decimal = Decimal("0.15")


@dataclass(frozen=True)
class CleanDealerBucketConfig:
    name: str
    tau_min: int
    tau_max: int
    mid: Decimal
    spread: Decimal


@dataclass(frozen=True)
class CleanDealerConfig:
    ticket_size: Decimal
    buckets: tuple[CleanDealerBucketConfig, ...]
    dealer_share: Decimal
    vbt_share: Decimal
    risk_enabled: bool
    lookback_window: int
    smoothing_alpha: Decimal
    initial_prior: Decimal
    base_risk_premium: Decimal = Decimal("0")
    urgency_sensitivity: Decimal = Decimal("0.30")
    use_issuer_specific: bool = False
    buy_premium_multiplier: Decimal = Decimal("1.0")
    adaptive_lookback: bool = False
    adaptive_issuer_specific: bool = False
    adaptive_ev_term_structure: bool = False
    term_strength: Decimal = Decimal("0.5")
    balanced_passive: bool = False
    balanced_active: bool = False
    outside_mid_ratio: Decimal = Decimal("1")
    kappa: Decimal | None = None
    mu: Decimal | None = None
    trading_rounds: int = 100
    issuer_specific_pricing: bool = False
    dealer_concentration_limit: Decimal = ZERO
    spread_scale: Decimal = Decimal("1")
    trader_profile: Any | None = None
    vbt_profile: Any | None = None


@dataclass
class CleanCBLoan:
    id: str
    bank: str
    central_bank: str
    amount: Decimal
    rate: Decimal
    issuance_day: int
    alias: str | None = None
    settled: bool = False

    @property
    def repayment_amount(self) -> Decimal:
        return Decimal(int(self.amount * (Decimal("1") + self.rate)))

    @property
    def interest_amount(self) -> Decimal:
        return self.repayment_amount - self.amount


@dataclass
class CleanNonBankLoan:
    id: str
    lender: str
    borrower: str
    amount: Decimal
    rate: Decimal
    issuance_day: int
    maturity_days: int
    settled: bool = False

    @property
    def maturity_day(self) -> int:
        return self.issuance_day + self.maturity_days

    @property
    def repayment_amount(self) -> Decimal:
        return Decimal(int(self.amount * (Decimal("1") + self.rate)))

    @property
    def interest_amount(self) -> Decimal:
        return self.repayment_amount - self.amount


@dataclass
class CleanBankLoan:
    id: str
    bank: str
    borrower: str
    amount: Decimal
    rate: Decimal
    issuance_day: int
    maturity_day: int
    settled: bool = False

    @property
    def repayment_amount(self) -> Decimal:
        return Decimal(int(self.amount * (Decimal("1") + self.rate)))

    @property
    def interest_amount(self) -> Decimal:
        return self.repayment_amount - self.amount


@dataclass
class CleanStockLot:
    id: str
    owner: str
    sku: str
    quantity: int
    unit_price: Decimal

    @property
    def value(self) -> Decimal:
        return Decimal(self.quantity) * self.unit_price


@dataclass
class CleanDeliveryObligation:
    id: str
    debtor: str
    creditor: str
    sku: str
    quantity: int
    unit_price: Decimal
    due_day: int
    alias: str | None = None
    settled: bool = False


@dataclass
class CleanState:
    agents: dict[str, CleanAgent] = field(default_factory=dict)
    central_bank_id: str | None = None
    cash: Counter[str] = field(default_factory=Counter)
    cash_lots: defaultdict[str, list[Decimal]] = field(default_factory=lambda: defaultdict(list))
    reserves: Counter[str] = field(default_factory=Counter)
    deposits: Counter[tuple[str, str]] = field(default_factory=Counter)
    payables: list[CleanPayable] = field(default_factory=list)
    cb_loans: list[CleanCBLoan] = field(default_factory=list)
    non_bank_loans: list[CleanNonBankLoan] = field(default_factory=list)
    bank_loans: list[CleanBankLoan] = field(default_factory=list)
    bank_defaulted_borrowers: set[str] = field(default_factory=set)
    stocks: dict[str, CleanStockLot] = field(default_factory=dict)
    delivery_obligations: list[CleanDeliveryObligation] = field(default_factory=list)
    scheduled_actions_by_day: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    defaulted_agent_ids: set[str] = field(default_factory=set)
    rating_registry: dict[str, Decimal] = field(default_factory=dict)
    lender_run_expected_loss_spent: Decimal = ZERO
    cb_reserves_initial: Decimal = ZERO
    cb_reserves_outstanding: Decimal = ZERO
    cb_loans_outstanding: Decimal = ZERO
    cb_interest_total_paid: Decimal = ZERO
    cb_loans_created_count: int = 0
    cb_lending_frozen: bool = False
    default_mode: str = "fail-fast"
    rollover_enabled: bool = False
    estimate_logging_enabled: bool = False
    estimate_log: list[Any] = field(default_factory=list)
    dealer_config: CleanDealerConfig | None = None
    dealer_subsystem: Any | None = None
    dealer_metrics: Any | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    day: int = 0
    quiet_days: int = 0

    def log(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, "day": self.day, "phase": "simulation", **payload})

    def log_setup(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, "day": 0, "phase": "setup", **payload})


@dataclass(frozen=True)
class CleanBankingConfig:
    kappa: Decimal = Decimal("1")
    reserve_target_ratio: Decimal = Decimal("0.10")
    adaptive_corridor: bool = False
    mu: Decimal | None = None
    concentration: Decimal | None = None
    enable_bank_lending: bool = False
    maturity_days: int = 10
    credit_risk_loading: Decimal = ZERO
    max_borrower_risk: Decimal = Decimal("1.0")
    min_coverage_ratio: Decimal = ZERO
    cb_lending_cutoff_day: int | None = None
    trader_bank_assignments: dict[str, list[str]] = field(default_factory=dict)
    infra_bank_assignments: dict[str, str] = field(default_factory=dict)
    reserve_targets: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CleanRunResult:
    state: CleanState
    final_day: int
    reached_stable: bool
    stop_reason: StopReason | None = None
    stability_snapshots: tuple[StabilitySnapshot, ...] = ()

    def __post_init__(self) -> None:
        if self.stop_reason is None:
            object.__setattr__(
                self,
                "stop_reason",
                StopReason.STABILITY_REACHED
                if self.reached_stable
                else StopReason.MAX_DAYS_REACHED,
            )

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.state.events

    @property
    def stop_day(self) -> int:
        return self.final_day


@dataclass
class CleanStabilityTracker:
    consecutive_quiet: int = 0
    consecutive_no_defaults: int = 0
    snapshots: list[StabilitySnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class CleanScenarioRuntime:
    state: CleanState
    policy_order: dict[str, list[str]]
    lender_config: CleanLenderConfig | None
    rating_config: CleanRatingConfig | None
    banking_config: CleanBankingConfig | None = None
    dealer_config: CleanDealerConfig | None = None
