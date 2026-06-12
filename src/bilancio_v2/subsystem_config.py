"""Subsystem runtime configuration dataclasses.

These frozen configs (lender, rating agency, dealer, banking) are built
from the scenario schema by :mod:`bilancio_v2.scenario_gates` and consumed
by the phase plugins. They retain their historical ``Clean*`` names from
the clean-core engine they were extracted from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

ZERO = Decimal("0")


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


@dataclass(frozen=True)
class CleanClearingConfig:
    mode: str = "netting"


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
