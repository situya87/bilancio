"""Utilities for running balanced C vs D comparison experiments.

This module provides infrastructure for running paired passive/active
experiments comparing Kalecki ring simulations with:
- C (Passive): Big entities hold securities + cash but DON'T trade
- D (Active): Big entities hold securities + cash and CAN trade

The key output is the delta in defaults between conditions, measuring
the effect of market-making by dealers.

Reference: Plan 021 and Section 9.3 of the specification.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from bilancio.experiments.ring import RingSweepRunner, RingRunSummary, PreparedRun
from bilancio.runners import SimulationExecutor, LocalExecutor, RunOptions

logger = logging.getLogger(__name__)


@dataclass
class BalancedComparisonResult:
    """Result of a single C vs D comparison."""

    # Parameters
    kappa: Decimal
    concentration: Decimal
    mu: Decimal
    monotonicity: Decimal
    seed: int

    # Balanced mode parameters
    face_value: Decimal
    outside_mid_ratio: Decimal
    big_entity_share: Decimal  # DEPRECATED

    # C (Passive) metrics
    delta_passive: Optional[Decimal]
    phi_passive: Optional[Decimal]
    passive_run_id: str
    passive_status: str

    # D (Active) metrics
    delta_active: Optional[Decimal]
    phi_active: Optional[Decimal]
    active_run_id: str
    active_status: str

    # Cascade/contagion metrics
    n_defaults_passive: int = 0
    n_defaults_active: int = 0
    cascade_fraction_passive: Optional[Decimal] = None
    cascade_fraction_active: Optional[Decimal] = None

    # Balanced mode parameters with defaults (Plan 024)
    vbt_share_per_bucket: Decimal = Decimal("0.25")
    dealer_share_per_bucket: Decimal = Decimal("0.125")

    # Informedness parameters
    alpha_vbt: Decimal = Decimal("0")
    alpha_trader: Decimal = Decimal("0")

    # Decision module parameters
    risk_aversion: Decimal = Decimal("0")
    planning_horizon: int = 10
    aggressiveness: Decimal = Decimal("1.0")
    default_observability: Decimal = Decimal("1.0")
    vbt_mid_sensitivity: Decimal = Decimal("1.0")
    vbt_spread_sensitivity: Decimal = Decimal("0.0")

    # Dealer metrics from active run
    dealer_total_pnl: Optional[float] = None
    dealer_total_return: Optional[float] = None
    total_trades: Optional[int] = None

    # Big entity loss metrics
    big_entity_loss_passive: Optional[float] = None
    big_entity_pnl_active: Optional[float] = None

    # Modal call IDs for cloud execution debugging
    passive_modal_call_id: Optional[str] = None
    active_modal_call_id: Optional[str] = None

    # Comparative dealer PnL (passive vs active)
    dealer_passive_pnl: Optional[float] = None
    dealer_passive_return: Optional[float] = None
    dealer_trading_incremental_pnl: Optional[float] = None

    # E (Lender) metrics
    delta_lender: Optional[Decimal] = None
    phi_lender: Optional[Decimal] = None
    lender_run_id: str = ""
    lender_status: str = ""
    n_defaults_lender: int = 0
    cascade_fraction_lender: Optional[Decimal] = None
    lender_modal_call_id: Optional[str] = None

    # Lender-specific metrics
    lender_total_pnl: Optional[float] = None
    lender_total_return: Optional[float] = None
    total_loans: Optional[int] = None

    # F (Dealer+Lender) metrics
    delta_dealer_lender: Optional[Decimal] = None
    phi_dealer_lender: Optional[Decimal] = None
    dealer_lender_run_id: str = ""
    dealer_lender_status: str = ""
    n_defaults_dealer_lender: int = 0
    cascade_fraction_dealer_lender: Optional[Decimal] = None
    dealer_lender_modal_call_id: Optional[str] = None

    @staticmethod
    def _compute_incremental_pnl(
        active_metrics: Optional[Dict[str, Any]],
        passive_metrics: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        """Compute trading incremental PnL = active_pnl - passive_pnl.

        Returns None if either metric is unavailable.
        """
        active_pnl = (active_metrics or {}).get("dealer_total_pnl")
        passive_pnl = (passive_metrics or {}).get("dealer_total_pnl")
        if active_pnl is not None and passive_pnl is not None:
            result: float = active_pnl - passive_pnl
            return result
        return None

    @property
    def trading_effect(self) -> Optional[Decimal]:
        """Effect of trading = delta_passive - delta_active.

        Positive means trading reduced defaults.
        """
        if self.delta_passive is None or self.delta_active is None:
            return None
        return self.delta_passive - self.delta_active

    @property
    def trading_relief_ratio(self) -> Optional[Decimal]:
        """Percentage reduction in defaults from trading."""
        if self.delta_passive is None or self.delta_active is None:
            return None
        if self.delta_passive == 0:
            return Decimal("0")  # No defaults to reduce
        effect = self.trading_effect
        assert effect is not None  # guaranteed by checks above
        return effect / self.delta_passive

    @property
    def cascade_effect(self) -> Optional[Decimal]:
        """Effect of trading on cascade fraction = passive - active.

        Positive means trading reduced cascading defaults.
        """
        if self.cascade_fraction_passive is None or self.cascade_fraction_active is None:
            return None
        return self.cascade_fraction_passive - self.cascade_fraction_active

    @property
    def lending_effect(self) -> Optional[Decimal]:
        """Effect of lending = delta_passive - delta_lender.

        Positive means lending reduced defaults.
        """
        if self.delta_passive is None or self.delta_lender is None:
            return None
        return self.delta_passive - self.delta_lender

    @property
    def combined_effect(self) -> Optional[Decimal]:
        """Effect of combined dealer+lender = delta_passive - delta_dealer_lender.

        Positive means the combination reduced defaults.
        """
        if self.delta_passive is None or self.delta_dealer_lender is None:
            return None
        return self.delta_passive - self.delta_dealer_lender


class BalancedComparisonConfig(BaseModel):
    """Configuration for C vs D balanced comparison experiments."""

    # Ring parameters
    n_agents: int = Field(default=100, description="Number of agents in ring")
    maturity_days: int = Field(default=10, description="Maturity horizon in days")
    max_simulation_days: int = Field(default=15, description="Max days to run simulation")
    Q_total: Decimal = Field(default=Decimal("10000"), description="Total debt amount")
    liquidity_mode: str = Field(default="uniform", description="Liquidity allocation mode")
    base_seed: int = Field(default=42, description="Base random seed")
    name_prefix: str = Field(default="Balanced Comparison", description="Scenario name prefix")
    default_handling: str = Field(default="expel-agent", description="Default handling mode")

    # Detailed logging (Plan 022)
    detailed_logging: bool = Field(
        default=False,
        description="Enable detailed CSV logging (trades.csv, inventory_timeseries.csv, system_state_timeseries.csv)"
    )

    # Grid parameters
    kappas: List[Decimal] = Field(
        default_factory=lambda: [Decimal("0.25"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("4")]
    )
    concentrations: List[Decimal] = Field(
        default_factory=lambda: [Decimal("0.2"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("5")]
    )
    mus: List[Decimal] = Field(
        default_factory=lambda: [Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1")]
    )
    monotonicities: List[Decimal] = Field(default_factory=lambda: [Decimal("0")])

    # Balanced dealer parameters (Plan 024)
    face_value: Decimal = Field(default=Decimal("20"), description="Face value S (cashflow at maturity)")
    outside_mid_ratios: List[Decimal] = Field(
        default_factory=lambda: [Decimal("1.0")],
        description="M/S ratios to sweep (kept for backward compat; VBT pricing now uses kappa-informed prior)"
    )
    big_entity_share: Decimal = Field(default=Decimal("0.25"), description="DEPRECATED - use vbt/dealer shares")
    vbt_share_per_bucket: Decimal = Field(
        default=Decimal("0.25"),
        description="VBT holds 25% of claims per maturity bucket"
    )
    dealer_share_per_bucket: Decimal = Field(
        default=Decimal("0.125"),
        description="Dealer holds 12.5% of claims per maturity bucket"
    )
    rollover_enabled: bool = Field(
        default=True,
        description="Enable continuous rollover of matured claims"
    )

    # Plan 030: Quiet mode for faster sweeps
    quiet: bool = Field(
        default=True,
        description="Suppress verbose console output during sweeps"
    )

    # VBT configuration (for active mode)
    vbt_share: Decimal = Field(default=Decimal("0.50"), description="VBT capital as fraction of system cash")

    # Risk assessment configuration
    risk_assessment_enabled: bool = Field(
        default=True,
        description="Enable risk-based trader decision making"
    )
    risk_assessment_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "base_risk_premium": "0.02",
            "urgency_sensitivity": "0.10",
            "buy_premium_multiplier": "1.0",
            "lookback_window": 5,
        },
        description="Risk assessment parameters"
    )

    # Informedness parameters (credit-informed pricing)
    alpha_vbt: Decimal = Field(default=Decimal("0"), description="VBT informedness (0=naive, 1=fully informed)")
    alpha_trader: Decimal = Field(default=Decimal("0"), description="Trader informedness (0=naive, 1=fully informed)")

    # Decision module parameters
    risk_aversion: Decimal = Field(default=Decimal("0"), description="Trader risk aversion (0-1)")
    planning_horizon: int = Field(default=10, description="Trader planning horizon (1-20 days)")
    aggressiveness: Decimal = Field(default=Decimal("1.0"), description="Buyer aggressiveness (0-1)")
    default_observability: Decimal = Field(default=Decimal("1.0"), description="Trader default observability (0-1)")
    vbt_mid_sensitivity: Decimal = Field(default=Decimal("1.0"), description="VBT mid price sensitivity to defaults (0-1)")
    vbt_spread_sensitivity: Decimal = Field(default=Decimal("0.0"), description="VBT spread sensitivity to defaults (0-1)")
    trading_motive: str = Field(default="liquidity_then_earning", description="Trading motivation mode")

    # Non-bank lender parameters
    enable_lender: bool = Field(default=False, description="Enable third comparison arm with non-bank lender")
    enable_dealer_lender: bool = Field(
        default=False,
        description="Enable fourth arm: dealer trading + non-bank lending combined"
    )
    lender_share: Decimal = Field(default=Decimal("0.10"), description="Lender capital as fraction of system cash")
    lender_base_rate: Decimal = Field(default=Decimal("0.05"), description="Lender base interest rate")
    lender_risk_premium_scale: Decimal = Field(default=Decimal("0.20"), description="Rate = base + scale × P(default)")
    lender_max_single_exposure: Decimal = Field(default=Decimal("0.15"), description="Max fraction of capital to one borrower")
    lender_max_total_exposure: Decimal = Field(default=Decimal("0.80"), description="Max fraction of capital deployed")
    lender_maturity_days: int = Field(default=2, description="Loan term in days")
    lender_horizon: int = Field(default=3, description="Look-ahead for upcoming obligations")


class BalancedComparisonRunner:
    """
    Runs C vs D comparison experiments.

    For each parameter combination (κ, c, μ, ρ):
    1. Run C (passive): Big entities hold but don't trade
    2. Run D (active): Big entities can trade
    3. Compute comparison metrics

    Outputs:
    - passive/: All passive holder runs
    - active/: All active dealer runs
    - aggregate/comparison.csv: C vs D metrics
    - aggregate/summary.json: Aggregate statistics
    """

    COMPARISON_FIELDS = [
        "kappa",
        "concentration",
        "mu",
        "monotonicity",
        "seed",
        "face_value",
        "outside_mid_ratio",
        "big_entity_share",
        "delta_passive",
        "delta_active",
        "trading_effect",
        "trading_relief_ratio",
        "phi_passive",
        "phi_active",
        "passive_run_id",
        "passive_status",
        "active_run_id",
        "active_status",
        "dealer_total_pnl",
        "dealer_total_return",
        "total_trades",
        "n_defaults_passive",
        "n_defaults_active",
        "cascade_fraction_passive",
        "cascade_fraction_active",
        "cascade_effect",
        "alpha_vbt",
        "alpha_trader",
        "dealer_passive_pnl",
        "dealer_passive_return",
        "dealer_trading_incremental_pnl",
        "risk_aversion",
        "planning_horizon",
        "aggressiveness",
        "default_observability",
        "vbt_mid_sensitivity",
        "vbt_spread_sensitivity",
        "delta_lender",
        "lending_effect",
        "phi_lender",
        "lender_run_id",
        "lender_status",
        "n_defaults_lender",
        "cascade_fraction_lender",
        "lender_total_pnl",
        "lender_total_return",
        "total_loans",
        "delta_dealer_lender",
        "combined_effect",
        "phi_dealer_lender",
        "dealer_lender_run_id",
        "dealer_lender_status",
        "n_defaults_dealer_lender",
        "cascade_fraction_dealer_lender",
    ]

    def __init__(
        self,
        config: BalancedComparisonConfig,
        out_dir: Path,
        executor: Optional[SimulationExecutor] = None,
        job_id: Optional[str] = None,
        enable_supabase: bool = True,
    ) -> None:
        self.config = config
        self.base_dir = out_dir
        self.executor: SimulationExecutor = executor or LocalExecutor()

        # Cloud-only mode: skip local processing when using cloud executor
        # Modal already saves runs to Supabase, so no need to duplicate
        from bilancio.runners.cloud_executor import CloudExecutor
        self.skip_local_processing = isinstance(executor, CloudExecutor)

        self.passive_dir = self.base_dir / "passive"
        self.active_dir = self.base_dir / "active"
        self.lender_dir = self.base_dir / "nbfi"
        self.dealer_lender_dir = self.base_dir / "dealer_lender"
        self.aggregate_dir = self.base_dir / "aggregate"

        # Only create local directories if we're doing local processing
        if not self.skip_local_processing:
            self.passive_dir.mkdir(parents=True, exist_ok=True)
            self.active_dir.mkdir(parents=True, exist_ok=True)
            if config.enable_lender:
                self.lender_dir.mkdir(parents=True, exist_ok=True)
            if config.enable_dealer_lender:
                self.dealer_lender_dir.mkdir(parents=True, exist_ok=True)
            self.aggregate_dir.mkdir(parents=True, exist_ok=True)

        self.comparison_results: List[BalancedComparisonResult] = []
        self.comparison_path = self.aggregate_dir / "comparison.csv"
        self.summary_path = self.aggregate_dir / "summary.json"

        self._passive_runner: Optional[RingSweepRunner] = None
        self._active_runner: Optional[RingSweepRunner] = None
        self.seed_counter = config.base_seed

        # For progress tracking
        self._start_time: Optional[float] = None
        self._completed_keys: Set[Tuple[str, str, str, str, str]] = set()

        # Job tracking
        self.job_id = job_id

        # Supabase registry for persisting runs/metrics (only for local execution)
        self._supabase_store = None
        if enable_supabase and not self.skip_local_processing:
            try:
                from bilancio.storage.supabase_client import is_supabase_configured
                if is_supabase_configured():
                    from bilancio.storage.supabase_registry import SupabaseRegistryStore
                    self._supabase_store = SupabaseRegistryStore()
                    logger.info("Supabase registry enabled for run persistence")
            except Exception as e:  # Intentionally broad: external service init
                logger.warning(f"Failed to initialize Supabase registry: {e}")

        # Load existing results for resumption
        self._load_existing_results()

    def _load_existing_results(self) -> None:
        """Load existing results from CSV for resumption (skipped in cloud-only mode)."""
        if self.skip_local_processing:
            return  # Cloud-only mode: no local files to load
        if not self.comparison_path.exists():
            return

        try:
            with self.comparison_path.open("r") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    # Create key from parameters
                    key = (
                        row["kappa"],
                        row["concentration"],
                        row["mu"],
                        row["monotonicity"],
                        row["outside_mid_ratio"],
                    )
                    self._completed_keys.add(key)

                    # Also track the seed to resume from correct position
                    seed = int(row["seed"])
                    if seed >= self.seed_counter:
                        self.seed_counter = seed + 1

            if self._completed_keys:
                logger.info(
                    "Resuming sweep: found %d completed pairs, starting from seed %d",
                    len(self._completed_keys),
                    self.seed_counter,
                )
        except Exception as e:  # Intentionally broad: sweep resumption
            logger.warning("Could not load existing results: %s", e)

    def _make_key(
        self,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
        outside_mid_ratio: Decimal,
    ) -> Tuple[str, str, str, str, str]:
        """Create a key for tracking completed pairs."""
        return (str(kappa), str(concentration), str(mu), str(monotonicity), str(outside_mid_ratio))

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _next_seed(self) -> int:
        """Get next seed and increment counter."""
        seed = self.seed_counter
        self.seed_counter += 1
        return seed

    def _get_passive_runner(self, outside_mid_ratio: Decimal) -> RingSweepRunner:
        """Get or create passive runner (no dealer trading)."""
        # For passive mode, we use the balanced scenario but with dealers disabled
        # In balanced mode, VBT/Dealer have inventory but don't trade
        return RingSweepRunner(
            out_dir=self.passive_dir,
            name_prefix=f"{self.config.name_prefix} (Passive)",
            n_agents=self.config.n_agents,
            maturity_days=self.config.maturity_days,
            Q_total=self.config.Q_total,
            liquidity_mode=self.config.liquidity_mode,
            liquidity_agent=None,  # Not used with uniform mode
            base_seed=self.config.base_seed,
            default_handling=self.config.default_handling,
            dealer_enabled=False,  # No dealer trading in passive mode
            dealer_config=None,
            # Pass balanced mode config (Plan 024)
            balanced_mode=True,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            big_entity_share=self.config.big_entity_share,  # DEPRECATED
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            rollover_enabled=self.config.rollover_enabled,
            detailed_dealer_logging=self.config.detailed_logging,  # Plan 022
            executor=self.executor,  # Plan 028 cloud support
            quiet=self.config.quiet,  # Plan 030
            risk_assessment_enabled=self.config.risk_assessment_enabled,
            risk_assessment_config=self.config.risk_assessment_config if self.config.risk_assessment_enabled else None,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            trading_motive=self.config.trading_motive,
        )

    def _get_active_runner(self, outside_mid_ratio: Decimal) -> RingSweepRunner:
        """Get or create active runner (with dealer trading)."""
        dealer_config = {
            "ticket_size": int(self.config.face_value),
            "dealer_share": str(Decimal("0")),  # Dealers already have inventory from scenario
            "vbt_share": str(self.config.vbt_share),
        }

        return RingSweepRunner(
            out_dir=self.active_dir,
            name_prefix=f"{self.config.name_prefix} (Active)",
            n_agents=self.config.n_agents,
            maturity_days=self.config.maturity_days,
            Q_total=self.config.Q_total,
            liquidity_mode=self.config.liquidity_mode,
            liquidity_agent=None,  # Not used with uniform mode
            base_seed=self.config.base_seed,
            default_handling=self.config.default_handling,
            dealer_enabled=True,  # Dealer trading enabled
            dealer_config=dealer_config,
            # Pass balanced mode config (Plan 024)
            balanced_mode=True,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            big_entity_share=self.config.big_entity_share,  # DEPRECATED
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            rollover_enabled=self.config.rollover_enabled,
            detailed_dealer_logging=self.config.detailed_logging,  # Plan 022
            executor=self.executor,  # Plan 028 cloud support
            quiet=self.config.quiet,  # Plan 030
            risk_assessment_enabled=self.config.risk_assessment_enabled,
            risk_assessment_config=self.config.risk_assessment_config if self.config.risk_assessment_enabled else None,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            trading_motive=self.config.trading_motive,
        )

    def _get_lender_runner(self, outside_mid_ratio: Decimal) -> RingSweepRunner:
        """Get or create lender runner (lending enabled, no dealer trading)."""
        return RingSweepRunner(
            out_dir=self.base_dir / "lender",
            name_prefix=f"{self.config.name_prefix} (Lender)",
            n_agents=self.config.n_agents,
            maturity_days=self.config.maturity_days,
            Q_total=self.config.Q_total,
            liquidity_mode=self.config.liquidity_mode,
            liquidity_agent=None,
            base_seed=self.config.base_seed,
            default_handling=self.config.default_handling,
            dealer_enabled=False,  # No dealer trading in lender mode
            dealer_config=None,
            balanced_mode=True,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            big_entity_share=self.config.big_entity_share,
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            rollover_enabled=self.config.rollover_enabled,
            detailed_dealer_logging=self.config.detailed_logging,
            executor=self.executor,
            quiet=self.config.quiet,
            risk_assessment_enabled=self.config.risk_assessment_enabled,
            risk_assessment_config=self.config.risk_assessment_config if self.config.risk_assessment_enabled else None,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            trading_motive=self.config.trading_motive,
            # Lender-specific: pass lender mode and config
            lender_mode=True,
            lender_share=self.config.lender_share,
            balanced_mode_override="lender",
        )

    def _get_nbfi_runner(self, outside_mid_ratio: Decimal) -> RingSweepRunner:
        """Get NBFI runner (lending enabled, no dealer, NBFI gets all VBT/dealer liquidity).

        In NBFI mode, VBT/Dealer get zero cash, all their liquidity goes to NBFI.
        This uses balanced_mode_override="nbfi" so the scenario compiler allocates
        cash accordingly.
        """
        return RingSweepRunner(
            out_dir=self.base_dir / "nbfi",
            name_prefix=f"{self.config.name_prefix} (NBFI)",
            n_agents=self.config.n_agents,
            maturity_days=self.config.maturity_days,
            Q_total=self.config.Q_total,
            liquidity_mode=self.config.liquidity_mode,
            liquidity_agent=None,
            base_seed=self.config.base_seed,
            default_handling=self.config.default_handling,
            dealer_enabled=False,  # No dealer trading in NBFI mode
            dealer_config=None,
            balanced_mode=True,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            big_entity_share=self.config.big_entity_share,
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            rollover_enabled=self.config.rollover_enabled,
            detailed_dealer_logging=self.config.detailed_logging,
            executor=self.executor,
            quiet=self.config.quiet,
            risk_assessment_enabled=self.config.risk_assessment_enabled,
            risk_assessment_config=self.config.risk_assessment_config if self.config.risk_assessment_enabled else None,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            trading_motive=self.config.trading_motive,
            # NBFI mode: lender enabled, override mode to "nbfi"
            lender_mode=True,
            lender_share=self.config.lender_share,
            balanced_mode_override="nbfi",
        )

    def _get_dealer_lender_runner(self, outside_mid_ratio: Decimal) -> RingSweepRunner:
        """Get runner for combined dealer+lender arm (F).

        Both dealer trading and NBFI lending are enabled. Cash is split 50/50
        between VBT/Dealer and the lender using balanced_mode_override="nbfi_dealer".
        """
        dealer_config = {
            "ticket_size": int(self.config.face_value),
            "dealer_share": str(Decimal("0")),  # Dealers already have inventory from scenario
            "vbt_share": str(self.config.vbt_share),
        }

        return RingSweepRunner(
            out_dir=self.dealer_lender_dir,
            name_prefix=f"{self.config.name_prefix} (Dealer+Lender)",
            n_agents=self.config.n_agents,
            maturity_days=self.config.maturity_days,
            Q_total=self.config.Q_total,
            liquidity_mode=self.config.liquidity_mode,
            liquidity_agent=None,
            base_seed=self.config.base_seed,
            default_handling=self.config.default_handling,
            dealer_enabled=True,  # Dealer trading ON
            dealer_config=dealer_config,
            balanced_mode=True,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            big_entity_share=self.config.big_entity_share,
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            rollover_enabled=self.config.rollover_enabled,
            detailed_dealer_logging=self.config.detailed_logging,
            executor=self.executor,
            quiet=self.config.quiet,
            risk_assessment_enabled=self.config.risk_assessment_enabled,
            risk_assessment_config=self.config.risk_assessment_config if self.config.risk_assessment_enabled else None,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            trading_motive=self.config.trading_motive,
            # Combined mode: both dealer and lender enabled
            lender_mode=True,
            lender_share=self.config.lender_share,
            balanced_mode_override="nbfi_dealer",  # 50/50 cash split
        )

    def run_all(self) -> List[BalancedComparisonResult]:
        """Execute all passive/active pairs and return comparison results.

        Uses batch execution if the executor supports it (CloudExecutor),
        otherwise falls back to sequential execution (LocalExecutor).
        """
        # Check if executor supports batch execution
        if hasattr(self.executor, 'execute_batch'):
            return self._run_all_batch()
        else:
            return self._run_all_sequential()

    def _run_all_batch(self) -> List[BalancedComparisonResult]:
        """Execute all pairs using batch execution (parallel on Modal)."""
        # Lender/dealer+lender arms not yet optimized for batch - fall back to sequential
        if self.config.enable_lender or self.config.enable_dealer_lender:
            return self._run_all_sequential()

        total_pairs = (
            len(self.config.kappas)
            * len(self.config.concentrations)
            * len(self.config.mus)
            * len(self.config.monotonicities)
            * len(self.config.outside_mid_ratios)
        )

        skipped = len(self._completed_keys)
        remaining = total_pairs - skipped

        logger.info(
            "Starting BATCH balanced comparison sweep: %d kappas × %d concentrations × %d mus × %d ρ = %d pairs",
            len(self.config.kappas),
            len(self.config.concentrations),
            len(self.config.mus),
            len(self.config.outside_mid_ratios),
            total_pairs,
        )

        if skipped > 0:
            print(f"Resuming: {skipped} pairs already completed, {remaining} remaining", flush=True)

        self._start_time = time.time()

        # Phase 1: Prepare all runs
        print(f"Preparing {remaining * 2} runs...", flush=True)
        prepared_runs: List[Tuple[PreparedRun, PreparedRun, Decimal, Decimal, Decimal, Decimal, Decimal, int]] = []

        for outside_mid_ratio in self.config.outside_mid_ratios:
            passive_runner = self._get_passive_runner(outside_mid_ratio)
            active_runner = self._get_active_runner(outside_mid_ratio)

            for kappa in self.config.kappas:
                for concentration in self.config.concentrations:
                    for mu in self.config.mus:
                        for monotonicity in self.config.monotonicities:
                            key = self._make_key(kappa, concentration, mu, monotonicity, outside_mid_ratio)
                            if key in self._completed_keys:
                                continue

                            seed = self._next_seed()

                            # Prepare passive run
                            passive_prep = passive_runner._prepare_run(
                                phase="balanced_passive",
                                kappa=kappa,
                                concentration=concentration,
                                mu=mu,
                                monotonicity=monotonicity,
                                seed=seed,
                            )

                            # Prepare active run
                            active_prep = active_runner._prepare_run(
                                phase="balanced_active",
                                kappa=kappa,
                                concentration=concentration,
                                mu=mu,
                                monotonicity=monotonicity,
                                seed=seed,
                            )

                            prepared_runs.append((
                                passive_prep, active_prep,
                                kappa, concentration, mu, monotonicity, outside_mid_ratio, seed
                            ))

        if not prepared_runs:
            print("All pairs already completed!", flush=True)
            return self.comparison_results

        # Phase 2: Build batch and execute
        print(f"Submitting {len(prepared_runs) * 2} runs to Modal (parallel execution)...", flush=True)

        # Build flat list for batch execution
        batch_runs: List[Tuple[Dict[str, Any], str, RunOptions]] = []
        run_index_map: Dict[str, int] = {}  # run_id -> index in prepared_runs

        for idx, (passive_prep, active_prep, *_) in enumerate(prepared_runs):
            batch_runs.append((
                passive_prep.scenario_config,
                passive_prep.run_id,
                passive_prep.options,
            ))
            run_index_map[passive_prep.run_id] = idx * 2  # even indices are passive

            batch_runs.append((
                active_prep.scenario_config,
                active_prep.run_id,
                active_prep.options,
            ))
            run_index_map[active_prep.run_id] = idx * 2 + 1  # odd indices are active

        # Execute batch with progress callback
        completed = [0]

        def progress_callback(done: int, total: int) -> None:
            completed[0] = done
            assert self._start_time is not None
            elapsed = time.time() - self._start_time
            if done > 0:
                eta = elapsed / done * (total - done)
                print(f"\r  Progress: {done}/{total} runs ({done * 100 // total}%) - ETA: {self._format_time(eta)}    ", end="", flush=True)

        results = self.executor.execute_batch(  # type: ignore[attr-defined]
            [(config, run_id, opts) for config, run_id, opts, *_ in batch_runs],
            progress_callback=progress_callback,
        )
        print()  # newline after progress

        # Phase 3: Finalize runs and build results
        print("Finalizing results...", flush=True)

        for idx, (passive_prep, active_prep, kappa, concentration, mu, monotonicity, outside_mid_ratio, seed) in enumerate(prepared_runs):
            passive_result = results[idx * 2]
            active_result = results[idx * 2 + 1]

            # Get runners for finalization
            passive_runner = self._get_passive_runner(outside_mid_ratio)
            active_runner = self._get_active_runner(outside_mid_ratio)

            # Finalize runs
            passive_summary = passive_runner._finalize_run(passive_prep, passive_result)
            active_summary = active_runner._finalize_run(active_prep, active_result)

            # Extract dealer metrics
            dm = active_summary.dealer_metrics or {}

            result = BalancedComparisonResult(
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                monotonicity=monotonicity,
                seed=seed,
                face_value=self.config.face_value,
                outside_mid_ratio=outside_mid_ratio,
                big_entity_share=self.config.big_entity_share,
                vbt_share_per_bucket=self.config.vbt_share_per_bucket,
                dealer_share_per_bucket=self.config.dealer_share_per_bucket,
                delta_passive=passive_summary.delta_total,
                phi_passive=passive_summary.phi_total,
                passive_run_id=passive_summary.run_id,
                passive_status="completed" if passive_summary.delta_total is not None else "failed",
                delta_active=active_summary.delta_total,
                phi_active=active_summary.phi_total,
                active_run_id=active_summary.run_id,
                active_status="completed" if active_summary.delta_total is not None else "failed",
                dealer_total_pnl=dm.get("dealer_total_pnl"),
                dealer_total_return=dm.get("dealer_total_return"),
                total_trades=dm.get("total_trades"),
                n_defaults_passive=passive_summary.n_defaults,
                n_defaults_active=active_summary.n_defaults,
                cascade_fraction_passive=passive_summary.cascade_fraction,
                cascade_fraction_active=active_summary.cascade_fraction,
                passive_modal_call_id=passive_summary.modal_call_id,
                active_modal_call_id=active_summary.modal_call_id,
                alpha_vbt=self.config.alpha_vbt,
                alpha_trader=self.config.alpha_trader,
                risk_aversion=self.config.risk_aversion,
                planning_horizon=self.config.planning_horizon,
                aggressiveness=self.config.aggressiveness,
                default_observability=self.config.default_observability,
                vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
                vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
                dealer_passive_pnl=(passive_summary.dealer_metrics or {}).get("dealer_total_pnl"),
                dealer_passive_return=(passive_summary.dealer_metrics or {}).get("dealer_total_return"),
                dealer_trading_incremental_pnl=BalancedComparisonResult._compute_incremental_pnl(
                    dm, passive_summary.dealer_metrics,
                ),
            )

            self.comparison_results.append(result)
            key = self._make_key(kappa, concentration, mu, monotonicity, outside_mid_ratio)
            self._completed_keys.add(key)

            # Persist runs to Supabase (batch path)
            self._persist_run_to_supabase(passive_summary, "passive", kappa, concentration, mu, outside_mid_ratio, seed)
            self._persist_run_to_supabase(active_summary, "active", kappa, concentration, mu, outside_mid_ratio, seed)

            # Incremental CSV write
            self._write_comparison_csv()

        # Write final summary
        self._write_summary_json()

        # Compute aggregate metrics on Modal (if using cloud executor)
        if hasattr(self.executor, 'compute_aggregate_metrics'):
            all_run_ids = []
            for result in self.comparison_results:
                if result.passive_run_id:
                    all_run_ids.append(result.passive_run_id)
                if result.active_run_id:
                    all_run_ids.append(result.active_run_id)
            if all_run_ids:
                try:
                    self.executor.compute_aggregate_metrics(all_run_ids)
                except Exception as e:  # Intentionally broad: sweep orchestration
                    print(f"\nWarning: Aggregate metrics computation failed: {e}", flush=True)
                    print("Local comparison.csv is still available.", flush=True)

        total_time = time.time() - self._start_time
        print(f"\nSweep complete! {len(prepared_runs)} pairs in {self._format_time(total_time)}", flush=True)
        print(f"Results at: {self.aggregate_dir}", flush=True)

        return self.comparison_results

    def _run_all_sequential(self) -> List[BalancedComparisonResult]:
        """Execute all pairs sequentially (fallback for LocalExecutor)."""
        total_pairs = (
            len(self.config.kappas)
            * len(self.config.concentrations)
            * len(self.config.mus)
            * len(self.config.monotonicities)
            * len(self.config.outside_mid_ratios)
        )

        skipped = len(self._completed_keys)
        remaining = total_pairs - skipped

        logger.info(
            "Starting balanced comparison sweep: %d kappas × %d concentrations × %d mus × %d ρ = %d pairs",
            len(self.config.kappas),
            len(self.config.concentrations),
            len(self.config.mus),
            len(self.config.outside_mid_ratios),
            total_pairs,
        )

        if skipped > 0:
            print(f"Resuming: {skipped} pairs already completed, {remaining} remaining", flush=True)

        self._start_time = time.time()
        pair_idx = 0
        completed_this_run = 0

        for outside_mid_ratio in self.config.outside_mid_ratios:
            for kappa in self.config.kappas:
                for concentration in self.config.concentrations:
                    for mu in self.config.mus:
                        for monotonicity in self.config.monotonicities:
                            pair_idx += 1

                            # Check if already completed
                            key = self._make_key(kappa, concentration, mu, monotonicity, outside_mid_ratio)
                            if key in self._completed_keys:
                                continue

                            # Progress and ETA
                            if completed_this_run > 0:
                                elapsed = time.time() - self._start_time
                                avg_time = elapsed / completed_this_run
                                eta = avg_time * (remaining - completed_this_run)
                                progress_str = f"[{pair_idx}/{total_pairs}] ({completed_this_run}/{remaining} this run) ETA: {self._format_time(eta)}"
                            else:
                                progress_str = f"[{pair_idx}/{total_pairs}]"

                            print(
                                f"{progress_str} Running: κ={kappa}, c={concentration}, μ={mu}, ρ={outside_mid_ratio}",
                                flush=True,
                            )

                            result = self._run_pair(
                                kappa, concentration, mu, monotonicity, outside_mid_ratio
                            )
                            self.comparison_results.append(result)
                            self._completed_keys.add(key)
                            completed_this_run += 1

                            # Write incremental results
                            self._write_comparison_csv()

                            # Log completion with timing
                            elapsed = time.time() - self._start_time
                            if result.delta_passive is not None and result.delta_active is not None:
                                print(
                                    f"  Completed in {self._format_time(elapsed / completed_this_run)} avg | "
                                    f"δ_passive={result.delta_passive:.3f}, δ_active={result.delta_active:.3f}, "
                                    f"effect={result.trading_effect:.3f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"  Completed in {self._format_time(elapsed / completed_this_run)} avg | "
                                    f"(one or both runs failed)",
                                    flush=True,
                                )

        # Write final summary
        self._write_summary_json()

        total_time = time.time() - self._start_time
        print(f"\nSweep complete! {completed_this_run} pairs in {self._format_time(total_time)}", flush=True)
        if self.skip_local_processing:
            print(f"Results saved to Supabase. Query with: bilancio jobs get {self.job_id} --cloud", flush=True)
        else:
            print(f"Results at: {self.aggregate_dir}", flush=True)

        logger.info("Balanced comparison sweep complete. Job ID: %s", self.job_id)
        return self.comparison_results

    def _make_progress_callback(self, run_type: str) -> Callable[[int, int], None]:
        """Create a progress callback that prints day-by-day progress."""
        def callback(current_day: int, max_days: int) -> None:
            assert self._start_time is not None
            elapsed = time.time() - self._start_time
            print(f"    {run_type}: day {current_day}/{max_days} (elapsed: {self._format_time(elapsed)})", flush=True)
        return callback

    def _run_pair(
        self,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
        outside_mid_ratio: Decimal,
    ) -> BalancedComparisonResult:
        """Run one passive/active pair for given parameters."""
        passive_runner = self._get_passive_runner(outside_mid_ratio)
        active_runner = self._get_active_runner(outside_mid_ratio)

        # Use same seed for both runs
        seed = self._next_seed()

        # Run passive (no trading)
        logger.info("  Running passive (no trading)...")
        print("  Passive run:", flush=True)
        passive_result = passive_runner._execute_run(
            phase="balanced_passive",
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
            progress_callback=self._make_progress_callback("passive"),
        )

        # Run active (with trading)
        logger.info("  Running active (with trading)...")
        print("  Active run:", flush=True)
        active_result = active_runner._execute_run(
            phase="balanced_active",
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
            progress_callback=self._make_progress_callback("active"),
        )

        # Extract dealer metrics from active result
        dm = active_result.dealer_metrics or {}

        # Run lender (optional third arm — NBFI mode)
        lender_result_data: Dict[str, Any] = {}
        if self.config.enable_lender:
            logger.info("  Running NBFI lender...")
            print("  NBFI run:", flush=True)
            nbfi_runner = self._get_nbfi_runner(outside_mid_ratio)
            lender_result = nbfi_runner._execute_run(
                phase="balanced_nbfi",
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                monotonicity=monotonicity,
                seed=seed,
                progress_callback=self._make_progress_callback("nbfi"),
            )
            lender_result_data = {
                "delta_lender": lender_result.delta_total,
                "phi_lender": lender_result.phi_total,
                "lender_run_id": lender_result.run_id,
                "lender_status": "completed" if lender_result.delta_total is not None else "failed",
                "n_defaults_lender": lender_result.n_defaults,
                "cascade_fraction_lender": lender_result.cascade_fraction,
                "lender_modal_call_id": lender_result.modal_call_id,
            }

        # Run dealer+lender (optional fourth arm — combined mode)
        dealer_lender_data: Dict[str, Any] = {}
        if self.config.enable_dealer_lender:
            logger.info("  Running Dealer+Lender...")
            print("  Dealer+Lender run:", flush=True)
            dl_runner = self._get_dealer_lender_runner(outside_mid_ratio)
            dl_result = dl_runner._execute_run(
                phase="balanced_dealer_lender",
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                monotonicity=monotonicity,
                seed=seed,
                progress_callback=self._make_progress_callback("dealer+lender"),
            )
            dealer_lender_data = {
                "delta_dealer_lender": dl_result.delta_total,
                "phi_dealer_lender": dl_result.phi_total,
                "dealer_lender_run_id": dl_result.run_id,
                "dealer_lender_status": "completed" if dl_result.delta_total is not None else "failed",
                "n_defaults_dealer_lender": dl_result.n_defaults,
                "cascade_fraction_dealer_lender": dl_result.cascade_fraction,
                "dealer_lender_modal_call_id": dl_result.modal_call_id,
            }

        result = BalancedComparisonResult(
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            big_entity_share=self.config.big_entity_share,  # DEPRECATED
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            delta_passive=passive_result.delta_total,
            phi_passive=passive_result.phi_total,
            passive_run_id=passive_result.run_id,
            passive_status="completed" if passive_result.delta_total is not None else "failed",
            delta_active=active_result.delta_total,
            phi_active=active_result.phi_total,
            active_run_id=active_result.run_id,
            active_status="completed" if active_result.delta_total is not None else "failed",
            dealer_total_pnl=dm.get("dealer_total_pnl"),
            dealer_total_return=dm.get("dealer_total_return"),
            total_trades=dm.get("total_trades"),
            n_defaults_passive=passive_result.n_defaults,
            n_defaults_active=active_result.n_defaults,
            cascade_fraction_passive=passive_result.cascade_fraction,
            cascade_fraction_active=active_result.cascade_fraction,
            passive_modal_call_id=passive_result.modal_call_id,
            active_modal_call_id=active_result.modal_call_id,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            dealer_passive_pnl=(passive_result.dealer_metrics or {}).get("dealer_total_pnl"),
            dealer_passive_return=(passive_result.dealer_metrics or {}).get("dealer_total_return"),
            dealer_trading_incremental_pnl=BalancedComparisonResult._compute_incremental_pnl(
                dm, passive_result.dealer_metrics,
            ),
            **lender_result_data,
            **dealer_lender_data,
        )

        # Log comparison
        if result.trading_effect is not None:
            logger.info(
                "  Comparison: δ_passive=%s, δ_active=%s, effect=%s (%.1f%%)",
                result.delta_passive,
                result.delta_active,
                result.trading_effect,
                float(result.trading_relief_ratio or 0) * 100,
            )
        else:
            logger.warning("  Comparison: One or both runs failed")

        # Persist runs to Supabase
        self._persist_run_to_supabase(passive_result, "passive", kappa, concentration, mu, outside_mid_ratio, seed)
        self._persist_run_to_supabase(active_result, "active", kappa, concentration, mu, outside_mid_ratio, seed)
        if self.config.enable_lender and lender_result_data.get("lender_run_id"):
            self._persist_run_to_supabase(lender_result, "nbfi", kappa, concentration, mu, outside_mid_ratio, seed)
        if self.config.enable_dealer_lender and dealer_lender_data.get("dealer_lender_run_id"):
            self._persist_run_to_supabase(dl_result, "dealer_lender", kappa, concentration, mu, outside_mid_ratio, seed)

        return result

    def _persist_run_to_supabase(
        self,
        run_result: RingRunSummary,
        regime: str,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        outside_mid_ratio: Decimal,
        seed: int,
    ) -> None:
        """Persist a run and its metrics to Supabase.

        Args:
            run_result: The run summary from the simulation
            regime: 'passive' or 'active'
            kappa: Liquidity ratio parameter
            concentration: Dirichlet concentration parameter
            mu: Maturity timing parameter
            outside_mid_ratio: Outside money ratio
            seed: Random seed used
        """
        if self._supabase_store is None:
            return

        try:
            from bilancio.storage.models import RegistryEntry, RunStatus

            # Determine status
            status = RunStatus.COMPLETED if run_result.delta_total is not None else RunStatus.FAILED

            # Build parameters dict
            parameters = {
                "kappa": str(kappa),
                "concentration": str(concentration),
                "mu": str(mu),
                "outside_mid_ratio": str(outside_mid_ratio),
                "seed": seed,
                "regime": regime,
            }

            # Build metrics dict
            metrics: Dict[str, Any] = {}
            if run_result.delta_total is not None:
                metrics["delta_total"] = float(run_result.delta_total)
            if run_result.phi_total is not None:
                metrics["phi_total"] = float(run_result.phi_total)
            if hasattr(run_result, "n_defaults") and run_result.n_defaults is not None:
                metrics["n_defaults"] = run_result.n_defaults
            if hasattr(run_result, "n_clears") and run_result.n_clears is not None:
                metrics["n_clears"] = run_result.n_clears
            if hasattr(run_result, "time_to_stability") and run_result.time_to_stability is not None:
                metrics["time_to_stability"] = run_result.time_to_stability
            if hasattr(run_result, "dealer_metrics") and run_result.dealer_metrics:
                dm = run_result.dealer_metrics
                if "total_trades" in dm:
                    metrics["total_trades"] = dm["total_trades"]
                if "total_trade_volume" in dm:
                    metrics["total_trade_volume"] = dm["total_trade_volume"]

            # Build artifact paths (for cloud runs)
            artifact_paths: Dict[str, str] = {}
            if hasattr(run_result, "artifact_paths") and run_result.artifact_paths:
                artifact_paths = run_result.artifact_paths

            entry = RegistryEntry(
                run_id=run_result.run_id,
                experiment_id=self.job_id or "unknown",
                status=status,
                parameters=parameters,
                metrics=metrics,
                artifact_paths=artifact_paths,
                error=run_result.error if hasattr(run_result, "error") else None,
            )

            self._supabase_store.upsert(entry)
            logger.debug(f"Persisted run {run_result.run_id} to Supabase")

        except Exception as e:  # Intentionally broad: external service call
            logger.warning(f"Failed to persist run to Supabase: {e}")

    def _write_comparison_csv(self) -> None:
        """Write comparison results to CSV."""
        # Always write locally - even in cloud mode we want the aggregate CSV
        self.aggregate_dir.mkdir(parents=True, exist_ok=True)
        with self.comparison_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COMPARISON_FIELDS)
            writer.writeheader()
            for result in self.comparison_results:
                row = {
                    "kappa": str(result.kappa),
                    "concentration": str(result.concentration),
                    "mu": str(result.mu),
                    "monotonicity": str(result.monotonicity),
                    "seed": str(result.seed),
                    "face_value": str(result.face_value),
                    "outside_mid_ratio": str(result.outside_mid_ratio),
                    "big_entity_share": str(result.big_entity_share),
                    "delta_passive": str(result.delta_passive) if result.delta_passive is not None else "",
                    "delta_active": str(result.delta_active) if result.delta_active is not None else "",
                    "trading_effect": str(result.trading_effect) if result.trading_effect is not None else "",
                    "trading_relief_ratio": str(result.trading_relief_ratio) if result.trading_relief_ratio is not None else "",
                    "phi_passive": str(result.phi_passive) if result.phi_passive is not None else "",
                    "phi_active": str(result.phi_active) if result.phi_active is not None else "",
                    "passive_run_id": result.passive_run_id,
                    "passive_status": result.passive_status,
                    "active_run_id": result.active_run_id,
                    "active_status": result.active_status,
                    "dealer_total_pnl": str(result.dealer_total_pnl) if result.dealer_total_pnl is not None else "",
                    "dealer_total_return": str(result.dealer_total_return) if result.dealer_total_return is not None else "",
                    "total_trades": str(result.total_trades) if result.total_trades is not None else "",
                    "n_defaults_passive": str(result.n_defaults_passive),
                    "n_defaults_active": str(result.n_defaults_active),
                    "cascade_fraction_passive": str(result.cascade_fraction_passive) if result.cascade_fraction_passive is not None else "",
                    "cascade_fraction_active": str(result.cascade_fraction_active) if result.cascade_fraction_active is not None else "",
                    "cascade_effect": str(result.cascade_effect) if result.cascade_effect is not None else "",
                    "alpha_vbt": str(result.alpha_vbt),
                    "alpha_trader": str(result.alpha_trader),
                    "dealer_passive_pnl": str(result.dealer_passive_pnl) if result.dealer_passive_pnl is not None else "",
                    "dealer_passive_return": str(result.dealer_passive_return) if result.dealer_passive_return is not None else "",
                    "dealer_trading_incremental_pnl": str(result.dealer_trading_incremental_pnl) if result.dealer_trading_incremental_pnl is not None else "",
                    "risk_aversion": str(result.risk_aversion),
                    "planning_horizon": str(result.planning_horizon),
                    "aggressiveness": str(result.aggressiveness),
                    "default_observability": str(result.default_observability),
                    "vbt_mid_sensitivity": str(result.vbt_mid_sensitivity),
                    "vbt_spread_sensitivity": str(result.vbt_spread_sensitivity),
                    "delta_lender": str(result.delta_lender) if result.delta_lender is not None else "",
                    "lending_effect": str(result.lending_effect) if result.lending_effect is not None else "",
                    "phi_lender": str(result.phi_lender) if result.phi_lender is not None else "",
                    "lender_run_id": result.lender_run_id,
                    "lender_status": result.lender_status,
                    "n_defaults_lender": str(result.n_defaults_lender),
                    "cascade_fraction_lender": str(result.cascade_fraction_lender) if result.cascade_fraction_lender is not None else "",
                    "lender_total_pnl": str(result.lender_total_pnl) if result.lender_total_pnl is not None else "",
                    "lender_total_return": str(result.lender_total_return) if result.lender_total_return is not None else "",
                    "total_loans": str(result.total_loans) if result.total_loans is not None else "",
                    "delta_dealer_lender": str(result.delta_dealer_lender) if result.delta_dealer_lender is not None else "",
                    "combined_effect": str(result.combined_effect) if result.combined_effect is not None else "",
                    "phi_dealer_lender": str(result.phi_dealer_lender) if result.phi_dealer_lender is not None else "",
                    "dealer_lender_run_id": result.dealer_lender_run_id,
                    "dealer_lender_status": result.dealer_lender_status,
                    "n_defaults_dealer_lender": str(result.n_defaults_dealer_lender),
                    "cascade_fraction_dealer_lender": str(result.cascade_fraction_dealer_lender) if result.cascade_fraction_dealer_lender is not None else "",
                }
                writer.writerow(row)

    def _write_summary_json(self) -> None:
        """Write summary statistics to JSON (skipped in cloud-only mode)."""
        if self.skip_local_processing:
            return  # Cloud-only mode: no local files
        completed = [r for r in self.comparison_results if r.trading_effect is not None]

        if completed:
            delta_passives = [float(r.delta_passive) for r in completed if r.delta_passive is not None]
            delta_actives = [float(r.delta_active) for r in completed if r.delta_active is not None]
            trading_effects = [float(r.trading_effect) for r in completed if r.trading_effect is not None]
            relief_ratios = [float(r.trading_relief_ratio) for r in completed if r.trading_relief_ratio is not None]

            mean_delta_passive = sum(delta_passives) / len(delta_passives) if delta_passives else None
            mean_delta_active = sum(delta_actives) / len(delta_actives) if delta_actives else None
            mean_trading_effect = sum(trading_effects) / len(trading_effects) if trading_effects else None
            mean_relief_ratio = sum(relief_ratios) / len(relief_ratios) if relief_ratios else None

            # Cascade metrics
            n_defaults_passive_vals = [r.n_defaults_passive for r in completed]
            n_defaults_active_vals = [r.n_defaults_active for r in completed]
            mean_n_defaults_passive = sum(n_defaults_passive_vals) / len(n_defaults_passive_vals) if n_defaults_passive_vals else None
            mean_n_defaults_active = sum(n_defaults_active_vals) / len(n_defaults_active_vals) if n_defaults_active_vals else None

            cascade_passive_vals = [float(r.cascade_fraction_passive) for r in completed if r.cascade_fraction_passive is not None]
            cascade_active_vals = [float(r.cascade_fraction_active) for r in completed if r.cascade_fraction_active is not None]
            mean_cascade_passive = sum(cascade_passive_vals) / len(cascade_passive_vals) if cascade_passive_vals else None
            mean_cascade_active = sum(cascade_active_vals) / len(cascade_active_vals) if cascade_active_vals else None

            improved = sum(1 for r in completed if r.trading_effect and r.trading_effect > 0)
            unchanged = sum(1 for r in completed if r.trading_effect == 0)
            worsened = sum(1 for r in completed if r.trading_effect and r.trading_effect < 0)
        else:
            mean_delta_passive = None
            mean_delta_active = None
            mean_trading_effect = None
            mean_relief_ratio = None
            mean_n_defaults_passive = None
            mean_n_defaults_active = None
            mean_cascade_passive = None
            mean_cascade_active = None
            improved = 0
            unchanged = 0
            worsened = 0

        summary = {
            "total_pairs": len(self.comparison_results),
            "completed_pairs": len(completed),
            "mean_delta_passive": mean_delta_passive,
            "mean_delta_active": mean_delta_active,
            "mean_trading_effect": mean_trading_effect,
            "mean_relief_ratio": mean_relief_ratio,
            "pairs_with_improvement": improved,
            "pairs_unchanged": unchanged,
            "pairs_worsened": worsened,
            "mean_n_defaults_passive": mean_n_defaults_passive,
            "mean_n_defaults_active": mean_n_defaults_active,
            "mean_cascade_fraction_passive": mean_cascade_passive,
            "mean_cascade_fraction_active": mean_cascade_active,
            "config": {
                "n_agents": self.config.n_agents,
                "maturity_days": self.config.maturity_days,
                "Q_total": str(self.config.Q_total),
                "base_seed": self.config.base_seed,
                "face_value": str(self.config.face_value),
                "big_entity_share": str(self.config.big_entity_share),  # DEPRECATED
                "vbt_share_per_bucket": str(self.config.vbt_share_per_bucket),
                "dealer_share_per_bucket": str(self.config.dealer_share_per_bucket),
                "rollover_enabled": self.config.rollover_enabled,
                "kappas": [str(k) for k in self.config.kappas],
                "concentrations": [str(c) for c in self.config.concentrations],
                "mus": [str(m) for m in self.config.mus],
                "outside_mid_ratios": [str(r) for r in self.config.outside_mid_ratios],
            },
        }

        with self.summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2)


def run_balanced_comparison_sweep(
    out_dir: Path,
    *,
    n_agents: int = 100,
    maturity_days: int = 10,
    Q_total: Decimal = Decimal("10000"),
    kappas: Sequence[Decimal],
    concentrations: Sequence[Decimal],
    mus: Sequence[Decimal],
    monotonicities: Optional[Sequence[Decimal]] = None,
    face_value: Decimal = Decimal("20"),
    outside_mid_ratios: Sequence[Decimal],
    big_entity_share: Decimal = Decimal("0.25"),
    base_seed: int = 42,
    default_handling: str = "expel-agent",
    name_prefix: str = "Balanced Comparison",
) -> List[BalancedComparisonResult]:
    """
    Convenience function to run a balanced comparison sweep.

    Args:
        out_dir: Output directory for results
        n_agents: Number of agents in ring (default: 100)
        maturity_days: Maturity horizon (default: 10)
        Q_total: Total debt amount (default: 10000)
        kappas: List of kappa values to sweep
        concentrations: List of Dirichlet concentration values
        mus: List of mu (misalignment) values
        monotonicities: List of monotonicity values (default: [0])
        face_value: Face value S (default: 20)
        outside_mid_ratios: List of M/S ratios to sweep
        big_entity_share: Fraction of debt held by big entities (default: 0.25)
        base_seed: Base random seed
        default_handling: How to handle defaults
        name_prefix: Scenario name prefix

    Returns:
        List of BalancedComparisonResult objects
    """
    config = BalancedComparisonConfig(
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=Q_total,
        kappas=list(kappas),
        concentrations=list(concentrations),
        mus=list(mus),
        monotonicities=list(monotonicities or [Decimal("0")]),
        face_value=face_value,
        outside_mid_ratios=list(outside_mid_ratios),
        big_entity_share=big_entity_share,
        base_seed=base_seed,
        default_handling=default_handling,
        name_prefix=name_prefix,
    )

    runner = BalancedComparisonRunner(config, out_dir)
    return runner.run_all()


__all__ = [
    "BalancedComparisonResult",
    "BalancedComparisonConfig",
    "BalancedComparisonRunner",
    "run_balanced_comparison_sweep",
]
