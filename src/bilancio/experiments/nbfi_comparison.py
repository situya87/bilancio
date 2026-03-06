"""NBFI comparison runner: nbfi_idle vs nbfi_lend (Plan 043).

A simplified 2-arm comparison runner that isolates the effect of
non-bank lending. Both arms include VBT/Dealer infrastructure (passive,
no trading); the treatment arm enables NBFI lending.

Arms:
  - nbfi_idle (baseline): VBT+Dealer hold inventory passively, NBFI lender present but idle
  - nbfi_lend (treatment): Same, but NBFI lender actively lends

The key output is the lending_effect = delta_idle - delta_lend,
measuring how much NBFI lending reduces defaults.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bilancio.core.performance import PerformanceConfig
from bilancio.experiments.ring import PreparedRun, RingRunSummary, RingSweepRunner
from bilancio.runners import LocalExecutor, SimulationExecutor

logger = logging.getLogger(__name__)
EXTERNAL_OPERATION_ERRORS = (
    FileNotFoundError,
    ImportError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
)


@dataclass
class NBFIComparisonResult:
    """Result of a single nbfi_idle vs nbfi_lend comparison."""

    # Grid parameters
    kappa: Decimal
    concentration: Decimal
    mu: Decimal
    monotonicity: Decimal
    seed: int
    outside_mid_ratio: Decimal

    # Idle arm metrics
    delta_idle: Decimal | None
    phi_idle: Decimal | None
    idle_run_id: str
    idle_status: str

    # Lend arm metrics
    delta_lend: Decimal | None
    phi_lend: Decimal | None
    lend_run_id: str
    lend_status: str

    # Fields with defaults
    n_defaults_idle: int = 0
    cascade_fraction_idle: Decimal | None = None
    idle_modal_call_id: str | None = None

    n_defaults_lend: int = 0
    cascade_fraction_lend: Decimal | None = None
    lend_modal_call_id: str | None = None

    # Loss metrics
    total_loss_idle: int = 0
    total_loss_lend: int = 0
    total_loss_pct_idle: float | None = None
    total_loss_pct_lend: float | None = None
    intermediary_loss_idle: float = 0.0
    intermediary_loss_lend: float = 0.0
    intermediary_loss_pct_idle: float | None = None
    intermediary_loss_pct_lend: float | None = None
    system_loss_idle: float = 0.0
    system_loss_lend: float = 0.0
    system_loss_pct_idle: float | None = None
    system_loss_pct_lend: float | None = None

    @property
    def lending_effect(self) -> Decimal | None:
        """Effect of lending = delta_idle - delta_lend.

        Positive means lending reduced defaults.
        """
        if self.delta_idle is None or self.delta_lend is None:
            return None
        return self.delta_idle - self.delta_lend

    @property
    def lending_relief_ratio(self) -> Decimal | None:
        """Percentage reduction in defaults from lending."""
        if self.delta_idle is None or self.delta_lend is None:
            return None
        if self.delta_idle == 0:
            return Decimal("0")
        effect = self.lending_effect
        assert effect is not None
        return effect / self.delta_idle

    @property
    def system_loss_lending_effect(self) -> float | None:
        """Total system loss reduction from NBFI lending."""
        if self.system_loss_pct_idle is None or self.system_loss_pct_lend is None:
            return None
        return self.system_loss_pct_idle - self.system_loss_pct_lend


class NBFIComparisonConfig(BaseModel):
    """Configuration for NBFI idle vs lend comparison experiments."""

    # Ring parameters
    n_agents: int = Field(default=100, description="Number of agents in ring")
    maturity_days: int = Field(default=10, description="Maturity horizon in days")
    Q_total: Decimal = Field(default=Decimal("10000"), description="Total debt amount")
    liquidity_mode: str = Field(default="uniform", description="Liquidity allocation mode")
    base_seed: int = Field(default=42, description="Base random seed")
    n_replicates: int = Field(default=1, ge=1, description="Seeds per parameter cell")
    name_prefix: str = Field(default="NBFI Comparison", description="Scenario name prefix")
    default_handling: str = Field(default="expel-agent", description="Default handling mode")
    quiet: bool = Field(default=True, description="Suppress verbose console output")
    rollover_enabled: bool = Field(default=True, description="Enable rollover")

    # Grid parameters
    kappas: list[Decimal] = Field(
        default_factory=lambda: [Decimal("0.25"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("4")]
    )
    concentrations: list[Decimal] = Field(default_factory=lambda: [Decimal("1")])
    mus: list[Decimal] = Field(default_factory=lambda: [Decimal("0")])
    monotonicities: list[Decimal] = Field(default_factory=lambda: [Decimal("0")])
    outside_mid_ratios: list[Decimal] = Field(default_factory=lambda: [Decimal("0.90")])

    # Balanced dealer parameters (VBT/Dealer hold inventory passively in both arms)
    face_value: Decimal = Field(default=Decimal("20"), description="Face value per ticket")
    vbt_share_per_bucket: Decimal = Field(default=Decimal("0.20"), description="VBT claim share per bucket")
    dealer_share_per_bucket: Decimal = Field(default=Decimal("0.05"), description="Dealer claim share per bucket")

    # NBFI lender parameters
    lender_share: Decimal = Field(default=Decimal("0.10"), description="Lender capital fraction")
    lender_base_rate: Decimal = Field(default=Decimal("0.05"), description="Lender base interest rate")
    lender_risk_premium_scale: Decimal = Field(default=Decimal("0.20"), description="Rate = base + scale * P(default)")
    lender_max_single_exposure: Decimal = Field(default=Decimal("0.15"), description="Max fraction to single borrower")
    lender_max_total_exposure: Decimal = Field(default=Decimal("0.80"), description="Max fraction deployed")
    lender_maturity_days: int = Field(default=2, description="Loan term in days")
    lender_horizon: int = Field(default=3, description="Look-ahead for obligations")
    lender_min_coverage: Decimal = Field(
        default=Decimal("0.5"), description="NBFI min coverage ratio for borrower assessment"
    )
    lender_maturity_matching: bool = Field(
        default=False, description="Match NBFI loan maturity to borrower's next receivable"
    )
    lender_min_loan_maturity: int = Field(
        default=2, description="Floor for NBFI loan maturity when matching"
    )
    lender_max_loans_per_borrower_per_day: int = Field(
        default=0, description="Max NBFI loans per borrower per day (0=unlimited)"
    )
    lender_ranking_mode: str = Field(
        default="profit", description="NBFI ranking mode: profit, cascade, or blended"
    )
    lender_cascade_weight: Decimal = Field(
        default=Decimal("0.5"), description="Weight for cascade score in blended ranking"
    )
    lender_coverage_mode: str = Field(
        default="gate", description="NBFI coverage gate mode: gate or graduated"
    )
    lender_coverage_penalty_scale: Decimal = Field(
        default=Decimal("0.10"), description="Rate premium per unit below coverage threshold"
    )
    lender_preventive_lending: bool = Field(
        default=False, description="Enable NBFI proactive lending to at-risk agents"
    )
    lender_prevention_threshold: Decimal = Field(
        default=Decimal("0.3"), description="Min issuer default probability to trigger preventive lending"
    )
    lender_marginal_relief_min_ratio: Decimal = Field(
        default=Decimal("0"), description="Min expected default relief / expected intermediary loss"
    )
    lender_stress_risk_premium_scale: Decimal = Field(
        default=Decimal("0"), description="Additional convex risk premium scale in stress"
    )
    lender_high_risk_default_threshold: Decimal = Field(
        default=Decimal("0.70"), description="High-risk default-probability threshold for maturity cap"
    )
    lender_high_risk_maturity_cap: int = Field(
        default=2, description="Max maturity for high-risk borrowers"
    )
    lender_daily_expected_loss_budget_ratio: Decimal = Field(
        default=Decimal("0"), description="Per-day expected loss budget as fraction of lender capital"
    )
    lender_run_expected_loss_budget_ratio: Decimal = Field(
        default=Decimal("0"), description="Run-level expected loss budget as fraction of lender capital"
    )
    lender_stop_loss_realized_ratio: Decimal = Field(
        default=Decimal("0"), description="Pause lending when realized losses / capital exceed this ratio"
    )
    lender_collateralized_terms: bool = Field(
        default=False, description="Cap NBFI loan principal by receivable collateral value"
    )
    lender_collateral_advance_rate: Decimal = Field(
        default=Decimal("1.0"), description="Advance rate for collateralized NBFI terms"
    )

    # Risk assessment configuration
    risk_assessment_enabled: bool = Field(default=True, description="Enable risk-based decisions")
    risk_assessment_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "base_risk_premium": "0.02",
            "urgency_sensitivity": "0.30",
            "buy_premium_multiplier": "1.0",
            "lookback_window": 5,
        },
    )

    # Trader decision parameters
    risk_aversion: Decimal = Field(default=Decimal("0"), description="Trader risk aversion (0-1)")
    planning_horizon: int = Field(default=10, description="Trader planning horizon")
    aggressiveness: Decimal = Field(default=Decimal("1.0"), description="Buyer aggressiveness")
    default_observability: Decimal = Field(default=Decimal("1.0"), description="Default observability")
    trading_motive: str = Field(default="liquidity_then_earning", description="Trading motivation")

    # VBT parameters
    alpha_vbt: Decimal = Field(default=Decimal("0"), description="VBT informedness")
    alpha_trader: Decimal = Field(default=Decimal("0"), description="Trader informedness")
    vbt_mid_sensitivity: Decimal = Field(default=Decimal("1.0"), description="VBT mid sensitivity")
    vbt_spread_sensitivity: Decimal = Field(default=Decimal("0.0"), description="VBT spread sensitivity")
    flow_sensitivity: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="VBT flow-aware ask widening (0=disabled, 1=max)",
    )

    # Bank parameters (both arms use bank deposits as MoP)
    n_banks: int = Field(default=3, description="Number of banks")
    reserve_multiplier: float = Field(default=10.0, description="Reserve multiplier")

    # Spread scaling
    spread_scale: Decimal = Field(default=Decimal("1.0"), description="Spread scale")
    trading_rounds: int = Field(default=100, ge=1, description="Trading sub-rounds per day (exits early when no intentions)")

    # Detailed logging
    detailed_logging: bool = Field(default=False, description="Enable detailed CSV logging")

    # Performance optimization flags
    performance: dict[str, Any] = Field(
        default_factory=dict,
        description="PerformanceConfig flags dict (e.g. {'fast_atomic': True})",
    )


class NBFIComparisonRunner:
    """Runs nbfi_idle vs nbfi_lend comparison experiments.

    For each parameter combination:
    1. Run nbfi_idle: VBT+Dealer passive, NBFI present but idle
    2. Run nbfi_lend: VBT+Dealer passive, NBFI actively lends
    3. Compute lending_effect = delta_idle - delta_lend
    """

    COMPARISON_FIELDS = [
        "kappa",
        "concentration",
        "mu",
        "monotonicity",
        "seed",
        "outside_mid_ratio",
        "delta_idle",
        "delta_lend",
        "lending_effect",
        "lending_relief_ratio",
        "phi_idle",
        "phi_lend",
        "idle_run_id",
        "idle_status",
        "lend_run_id",
        "lend_status",
        "n_defaults_idle",
        "n_defaults_lend",
        "cascade_fraction_idle",
        "cascade_fraction_lend",
        "total_loss_idle",
        "total_loss_lend",
        "total_loss_pct_idle",
        "total_loss_pct_lend",
        "intermediary_loss_idle",
        "intermediary_loss_lend",
        "intermediary_loss_pct_idle",
        "intermediary_loss_pct_lend",
        "system_loss_idle",
        "system_loss_lend",
        "system_loss_pct_idle",
        "system_loss_pct_lend",
        "system_loss_lending_effect",
    ]

    def __init__(
        self,
        config: NBFIComparisonConfig,
        out_dir: Path,
        executor: SimulationExecutor | None = None,
        job_id: str | None = None,
        enable_supabase: bool = True,
    ) -> None:
        self.config = config
        self.base_dir = out_dir
        self.executor: SimulationExecutor = executor or LocalExecutor()

        from bilancio.runners.cloud_executor import CloudExecutor

        self.skip_local_processing = isinstance(executor, CloudExecutor)

        self.idle_dir = self.base_dir / "nbfi_idle"
        self.lend_dir = self.base_dir / "nbfi_lend"
        self.aggregate_dir = self.base_dir / "aggregate"

        if not self.skip_local_processing:
            self.idle_dir.mkdir(parents=True, exist_ok=True)
            self.lend_dir.mkdir(parents=True, exist_ok=True)
            self.aggregate_dir.mkdir(parents=True, exist_ok=True)

        self.comparison_results: list[NBFIComparisonResult] = []
        self.comparison_path = self.aggregate_dir / "comparison.csv"
        self.summary_path = self.aggregate_dir / "summary.json"

        self.seed_counter = config.base_seed
        self._start_time: float | None = None
        self._completed_counts: dict[tuple[str, str, str, str, str], int] = {}

        self.job_id = job_id

        # Supabase registry
        self._supabase_store = None
        if enable_supabase and not self.skip_local_processing:
            try:
                from bilancio.storage.supabase_client import is_supabase_configured

                if is_supabase_configured():
                    from bilancio.storage.supabase_registry import SupabaseRegistryStore

                    self._supabase_store = SupabaseRegistryStore()
                    logger.info("Supabase registry enabled for NBFI comparison")
            except EXTERNAL_OPERATION_ERRORS as e:
                logger.warning(f"Failed to initialize Supabase registry: {e}")

        self._load_existing_results()

    def _load_existing_results(self) -> None:
        """Load existing results from CSV for resumption."""
        if self.skip_local_processing:
            return
        if not self.comparison_path.exists():
            return
        try:
            with self.comparison_path.open("r") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    key = (
                        row["kappa"],
                        row["concentration"],
                        row["mu"],
                        row["monotonicity"],
                        row["outside_mid_ratio"],
                    )
                    self._completed_counts[key] = self._completed_counts.get(key, 0) + 1
                    seed = int(row["seed"])
                    if seed >= self.seed_counter:
                        self.seed_counter = seed + 1
            if self._completed_counts:
                total_completed = sum(self._completed_counts.values())
                logger.info(
                    "Resuming NBFI sweep: found %d completed runs, starting from seed %d",
                    total_completed,
                    self.seed_counter,
                )
        except EXTERNAL_OPERATION_ERRORS as e:
            logger.warning("Could not load existing results: %s", e)

    def _make_key(
        self,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
        outside_mid_ratio: Decimal,
    ) -> tuple[str, str, str, str, str]:
        return (str(kappa), str(concentration), str(mu), str(monotonicity), str(outside_mid_ratio))

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _next_seed(self) -> int:
        seed = self.seed_counter
        self.seed_counter += 1
        return seed

    def _get_idle_runner(self, outside_mid_ratio: Decimal) -> RingSweepRunner:
        """Idle arm: VBT+Dealer passive, NBFI present but not lending."""
        return RingSweepRunner(
            out_dir=self.idle_dir,
            name_prefix=f"{self.config.name_prefix} (NBFI Idle)",
            n_agents=self.config.n_agents,
            maturity_days=self.config.maturity_days,
            Q_total=self.config.Q_total,
            liquidity_mode=self.config.liquidity_mode,
            liquidity_agent=None,
            base_seed=self.config.base_seed,
            default_handling=self.config.default_handling,
            dealer_enabled=False,  # No dealer trading
            dealer_config=None,
            balanced_mode=True,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            rollover_enabled=self.config.rollover_enabled,
            detailed_dealer_logging=self.config.detailed_logging,
            executor=self.executor,
            quiet=self.config.quiet,
            risk_assessment_enabled=self.config.risk_assessment_enabled,
            risk_assessment_config=self.config.risk_assessment_config
            if self.config.risk_assessment_enabled
            else None,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            trading_motive=self.config.trading_motive,
            lender_mode=False,  # NBFI present but idle (no lending)
            lender_share=self.config.lender_share,
            balanced_mode_override="nbfi_idle",
            n_banks=self.config.n_banks,
            reserve_multiplier=self.config.reserve_multiplier,
            spread_scale=self.config.spread_scale,
            trading_rounds=self.config.trading_rounds,
            flow_sensitivity=self.config.flow_sensitivity,
            performance=PerformanceConfig.from_dict(self.config.performance) if self.config.performance else None,
        )

    def _get_lend_runner(self, outside_mid_ratio: Decimal) -> RingSweepRunner:
        """Lend arm: VBT+Dealer passive, NBFI actively lending."""
        return RingSweepRunner(
            out_dir=self.lend_dir,
            name_prefix=f"{self.config.name_prefix} (NBFI Lend)",
            n_agents=self.config.n_agents,
            maturity_days=self.config.maturity_days,
            Q_total=self.config.Q_total,
            liquidity_mode=self.config.liquidity_mode,
            liquidity_agent=None,
            base_seed=self.config.base_seed,
            default_handling=self.config.default_handling,
            dealer_enabled=False,  # No dealer trading
            dealer_config=None,
            balanced_mode=True,
            face_value=self.config.face_value,
            outside_mid_ratio=outside_mid_ratio,
            vbt_share_per_bucket=self.config.vbt_share_per_bucket,
            dealer_share_per_bucket=self.config.dealer_share_per_bucket,
            rollover_enabled=self.config.rollover_enabled,
            detailed_dealer_logging=self.config.detailed_logging,
            executor=self.executor,
            quiet=self.config.quiet,
            risk_assessment_enabled=self.config.risk_assessment_enabled,
            risk_assessment_config=self.config.risk_assessment_config
            if self.config.risk_assessment_enabled
            else None,
            alpha_vbt=self.config.alpha_vbt,
            alpha_trader=self.config.alpha_trader,
            risk_aversion=self.config.risk_aversion,
            planning_horizon=self.config.planning_horizon,
            aggressiveness=self.config.aggressiveness,
            default_observability=self.config.default_observability,
            vbt_mid_sensitivity=self.config.vbt_mid_sensitivity,
            vbt_spread_sensitivity=self.config.vbt_spread_sensitivity,
            trading_motive=self.config.trading_motive,
            lender_mode=True,  # NBFI actively lending
            lender_share=self.config.lender_share,
            lender_base_rate=self.config.lender_base_rate,
            lender_risk_premium_scale=self.config.lender_risk_premium_scale,
            lender_max_single_exposure=self.config.lender_max_single_exposure,
            lender_max_total_exposure=self.config.lender_max_total_exposure,
            lender_maturity_days=self.config.lender_maturity_days,
            lender_horizon=self.config.lender_horizon,
            lender_min_coverage=self.config.lender_min_coverage,
            lender_maturity_matching=self.config.lender_maturity_matching,
            lender_min_loan_maturity=self.config.lender_min_loan_maturity,
            lender_max_loans_per_borrower_per_day=self.config.lender_max_loans_per_borrower_per_day,
            lender_ranking_mode=self.config.lender_ranking_mode,
            lender_cascade_weight=self.config.lender_cascade_weight,
            lender_coverage_mode=self.config.lender_coverage_mode,
            lender_coverage_penalty_scale=self.config.lender_coverage_penalty_scale,
            lender_preventive_lending=self.config.lender_preventive_lending,
            lender_prevention_threshold=self.config.lender_prevention_threshold,
            lender_marginal_relief_min_ratio=self.config.lender_marginal_relief_min_ratio,
            lender_stress_risk_premium_scale=self.config.lender_stress_risk_premium_scale,
            lender_high_risk_default_threshold=self.config.lender_high_risk_default_threshold,
            lender_high_risk_maturity_cap=self.config.lender_high_risk_maturity_cap,
            lender_daily_expected_loss_budget_ratio=self.config.lender_daily_expected_loss_budget_ratio,
            lender_run_expected_loss_budget_ratio=self.config.lender_run_expected_loss_budget_ratio,
            lender_stop_loss_realized_ratio=self.config.lender_stop_loss_realized_ratio,
            lender_collateralized_terms=self.config.lender_collateralized_terms,
            lender_collateral_advance_rate=self.config.lender_collateral_advance_rate,
            balanced_mode_override="nbfi_lend",
            n_banks=self.config.n_banks,
            reserve_multiplier=self.config.reserve_multiplier,
            spread_scale=self.config.spread_scale,
            trading_rounds=self.config.trading_rounds,
            flow_sensitivity=self.config.flow_sensitivity,
            performance=PerformanceConfig.from_dict(self.config.performance) if self.config.performance else None,
        )

    def _get_enabled_arm_defs(
        self,
    ) -> list[tuple[str, str, str, str]]:
        """Return (arm_name, phase_name, runner_getter_name, supabase_regime) for each arm."""
        return [
            ("idle", "nbfi_idle", "_get_idle_runner", "nbfi_idle"),
            ("lend", "nbfi_lend", "_get_lend_runner", "nbfi_lend"),
        ]

    def _build_result_from_summaries(
        self,
        arm_summaries: dict[str, RingRunSummary],
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
        outside_mid_ratio: Decimal,
        seed: int,
    ) -> NBFIComparisonResult:
        """Build result from arm summaries."""
        idle = arm_summaries["idle"]
        lend = arm_summaries["lend"]

        # Compute system loss = trader loss + intermediary loss
        idle_system_loss = idle.total_loss + idle.intermediary_loss_total
        lend_system_loss = lend.total_loss + lend.intermediary_loss_total

        return NBFIComparisonResult(
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
            outside_mid_ratio=outside_mid_ratio,
            delta_idle=idle.delta_total,
            phi_idle=idle.phi_total,
            idle_run_id=idle.run_id,
            idle_status="completed" if idle.delta_total is not None else "failed",
            n_defaults_idle=idle.n_defaults,
            cascade_fraction_idle=idle.cascade_fraction,
            idle_modal_call_id=idle.modal_call_id,
            delta_lend=lend.delta_total,
            phi_lend=lend.phi_total,
            lend_run_id=lend.run_id,
            lend_status="completed" if lend.delta_total is not None else "failed",
            n_defaults_lend=lend.n_defaults,
            cascade_fraction_lend=lend.cascade_fraction,
            lend_modal_call_id=lend.modal_call_id,
            total_loss_idle=idle.total_loss,
            total_loss_lend=lend.total_loss,
            total_loss_pct_idle=idle.total_loss_pct,
            total_loss_pct_lend=lend.total_loss_pct,
            intermediary_loss_idle=idle.intermediary_loss_total,
            intermediary_loss_lend=lend.intermediary_loss_total,
            intermediary_loss_pct_idle=idle.intermediary_loss_total / idle.S_total if idle.S_total > 0 else None,
            intermediary_loss_pct_lend=lend.intermediary_loss_total / lend.S_total if lend.S_total > 0 else None,
            system_loss_idle=idle_system_loss,
            system_loss_lend=lend_system_loss,
            system_loss_pct_idle=idle_system_loss / idle.S_total if idle.S_total > 0 else None,
            system_loss_pct_lend=lend_system_loss / lend.S_total if lend.S_total > 0 else None,
        )

    # ── Execution ────────────────────────────────────────────────────────

    def run_all(self) -> list[NBFIComparisonResult]:
        """Execute all comparisons, batch or sequential."""
        if hasattr(self.executor, "execute_batch"):
            return self._run_all_batch()
        else:
            return self._run_all_sequential()

    def _run_all_batch(self) -> list[NBFIComparisonResult]:
        """Batch execution (parallel on Modal)."""
        arm_defs = self._get_enabled_arm_defs()
        n_arms = len(arm_defs)

        total_combos = (
            len(self.config.kappas)
            * len(self.config.concentrations)
            * len(self.config.mus)
            * len(self.config.monotonicities)
            * len(self.config.outside_mid_ratios)
        )

        cells_done = sum(
            1 for c in self._completed_counts.values() if c >= self.config.n_replicates
        )
        remaining = total_combos - cells_done

        arm_names = [a[0] for a in arm_defs]
        logger.info(
            "Starting BATCH NBFI comparison: %d combos x %d arms",
            total_combos,
            n_arms,
        )
        print(f"Arms: {', '.join(arm_names)}", flush=True)

        if cells_done > 0:
            print(f"Resuming: {cells_done} combos done, {remaining} remaining", flush=True)

        self._start_time = time.time()

        # Phase 1: Prepare all runs
        prepared_combos: list[
            tuple[dict[str, PreparedRun], Decimal, Decimal, Decimal, Decimal, Decimal, int]
        ] = []
        runners_cache: dict[tuple[Decimal, str], Any] = {}

        for outside_mid_ratio in self.config.outside_mid_ratios:
            for arm_name, _phase, getter_name, _regime in arm_defs:
                cache_key = (outside_mid_ratio, arm_name)
                if cache_key not in runners_cache:
                    runners_cache[cache_key] = getattr(self, getter_name)(outside_mid_ratio)

            for kappa in self.config.kappas:
                for concentration in self.config.concentrations:
                    for mu in self.config.mus:
                        for monotonicity in self.config.monotonicities:
                            key = self._make_key(kappa, concentration, mu, monotonicity, outside_mid_ratio)
                            completed = self._completed_counts.get(key, 0)
                            reps_needed = max(0, self.config.n_replicates - completed)

                            for _rep in range(reps_needed):
                                seed = self._next_seed()
                                arm_preps: dict[str, PreparedRun] = {}
                                for arm_name, phase, _getter_name, _regime in arm_defs:
                                    runner = runners_cache[(outside_mid_ratio, arm_name)]
                                    arm_preps[arm_name] = runner._prepare_run(
                                        phase=phase,
                                        kappa=kappa,
                                        concentration=concentration,
                                        mu=mu,
                                        monotonicity=monotonicity,
                                        seed=seed,
                                    )
                                prepared_combos.append(
                                    (arm_preps, kappa, concentration, mu, monotonicity, outside_mid_ratio, seed)
                                )

        if not prepared_combos:
            print("All combos already completed!", flush=True)
            return self.comparison_results

        # Phase 2: Execute batch
        total_runs = sum(len(arm_preps) for arm_preps, *_ in prepared_combos)
        print(f"Submitting {total_runs} runs to Modal...", flush=True)

        from bilancio.runners import RunOptions

        batch_runs: list[tuple[dict[str, Any], str, RunOptions]] = []
        run_id_to_location: dict[str, tuple[int, str]] = {}

        for combo_idx, (arm_preps, *_params) in enumerate(prepared_combos):
            for arm_name, prep in arm_preps.items():
                batch_runs.append((prep.scenario_config, prep.run_id, prep.options))
                run_id_to_location[prep.run_id] = (combo_idx, arm_name)

        def progress_callback(done: int, total: int) -> None:
            assert self._start_time is not None
            elapsed = time.time() - self._start_time
            if done > 0:
                eta = elapsed / done * (total - done)
                print(
                    f"\r  Progress: {done}/{total} runs ({done * 100 // total}%) - ETA: {self._format_time(eta)}    ",
                    end="",
                    flush=True,
                )

        results = self.executor.execute_batch(  # type: ignore[attr-defined]
            batch_runs,
            progress_callback=progress_callback,
        )
        print()

        # Route results
        combo_results: dict[int, dict[str, Any]] = {}
        for (_config, run_id, _opts), raw_result in zip(batch_runs, results, strict=False):
            combo_idx, arm_name = run_id_to_location[run_id]
            combo_results.setdefault(combo_idx, {})[arm_name] = raw_result

        # Phase 3: Finalize
        print("Finalizing results...", flush=True)

        for combo_idx, (
            arm_preps, kappa, concentration, mu, monotonicity, outside_mid_ratio, seed,
        ) in enumerate(prepared_combos):
            arm_summaries: dict[str, RingRunSummary] = {}
            for arm_name, prep in arm_preps.items():
                runner = runners_cache[(outside_mid_ratio, arm_name)]
                raw = combo_results[combo_idx][arm_name]
                arm_summaries[arm_name] = runner._finalize_run(prep, raw)

            result = self._build_result_from_summaries(
                arm_summaries, kappa, concentration, mu, monotonicity, outside_mid_ratio, seed
            )
            self.comparison_results.append(result)
            comp_key = self._make_key(kappa, concentration, mu, monotonicity, outside_mid_ratio)
            self._completed_counts[comp_key] = self._completed_counts.get(comp_key, 0) + 1

            for arm_name, _phase, _getter, regime in arm_defs:
                if arm_name in arm_summaries:
                    self._persist_run_to_supabase(
                        arm_summaries[arm_name], regime, kappa, concentration, mu, outside_mid_ratio, seed
                    )

            self._write_comparison_csv()

        self._write_summary_json()
        self._write_stats_analysis()
        self._write_activity_analysis()

        total_time = time.time() - self._start_time
        print(
            f"\nSweep complete! {len(prepared_combos)} combos ({total_runs} runs) in {self._format_time(total_time)}",
            flush=True,
        )
        print(f"Results at: {self.aggregate_dir}", flush=True)

        return self.comparison_results

    def _run_all_sequential(self) -> list[NBFIComparisonResult]:
        """Sequential execution (LocalExecutor fallback)."""
        total_combos = (
            len(self.config.kappas)
            * len(self.config.concentrations)
            * len(self.config.mus)
            * len(self.config.monotonicities)
            * len(self.config.outside_mid_ratios)
        )

        cells_done = sum(
            1 for c in self._completed_counts.values() if c >= self.config.n_replicates
        )
        remaining = total_combos - cells_done

        logger.info(
            "Starting NBFI comparison sweep: %d combos (%d remaining)",
            total_combos,
            remaining,
        )

        if cells_done > 0:
            print(f"Resuming: {cells_done} combos done, {remaining} remaining", flush=True)

        self._start_time = time.time()
        combo_idx = 0
        completed_this_run = 0

        for outside_mid_ratio in self.config.outside_mid_ratios:
            for kappa in self.config.kappas:
                for concentration in self.config.concentrations:
                    for mu in self.config.mus:
                        for monotonicity in self.config.monotonicities:
                            combo_idx += 1
                            key = self._make_key(kappa, concentration, mu, monotonicity, outside_mid_ratio)
                            completed = self._completed_counts.get(key, 0)
                            reps_needed = max(0, self.config.n_replicates - completed)
                            if reps_needed == 0:
                                continue

                            for rep in range(reps_needed):
                                rep_label = (
                                    f" rep {completed + rep + 1}/{self.config.n_replicates}"
                                    if self.config.n_replicates > 1
                                    else ""
                                )

                                if completed_this_run > 0:
                                    elapsed = time.time() - self._start_time
                                    avg_time = elapsed / completed_this_run
                                    eta = avg_time * (remaining * self.config.n_replicates - completed_this_run)
                                    progress_str = f"[{combo_idx}/{total_combos}] ({completed_this_run} done) ETA: {self._format_time(eta)}"
                                else:
                                    progress_str = f"[{combo_idx}/{total_combos}]"

                                print(
                                    f"{progress_str} Running{rep_label}: k={kappa}, c={concentration}, mu={mu}, rho={outside_mid_ratio}",
                                    flush=True,
                                )

                                result = self._run_pair(
                                    kappa, concentration, mu, monotonicity, outside_mid_ratio
                                )
                                self.comparison_results.append(result)
                                self._completed_counts[key] = self._completed_counts.get(key, 0) + 1
                                completed_this_run += 1

                                self._write_comparison_csv()

                                elapsed = time.time() - self._start_time
                                if result.lending_effect is not None:
                                    print(
                                        f"  Completed | delta_idle={result.delta_idle:.3f}, delta_lend={result.delta_lend:.3f}, "
                                        f"lending_effect={result.lending_effect:.3f}",
                                        flush=True,
                                    )
                                else:
                                    print("  Completed | (one or both runs failed)", flush=True)

        self._write_summary_json()
        self._write_stats_analysis()
        self._write_activity_analysis()

        total_time = time.time() - self._start_time
        print(
            f"\nSweep complete! {completed_this_run} combos in {self._format_time(total_time)}",
            flush=True,
        )
        if self.skip_local_processing:
            print(
                f"Results saved to Supabase. Query with: bilancio jobs get {self.job_id} --cloud",
                flush=True,
            )
        else:
            print(f"Results at: {self.aggregate_dir}", flush=True)

        return self.comparison_results

    def _run_pair(
        self,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
        outside_mid_ratio: Decimal,
    ) -> NBFIComparisonResult:
        """Run one idle/lend pair."""
        idle_runner = self._get_idle_runner(outside_mid_ratio)
        lend_runner = self._get_lend_runner(outside_mid_ratio)

        seed = self._next_seed()

        logger.info("  Running nbfi_idle...")
        print("  NBFI Idle run:", flush=True)
        idle_result = idle_runner._execute_run(
            phase="nbfi_idle",
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
        )

        logger.info("  Running nbfi_lend...")
        print("  NBFI Lend run:", flush=True)
        lend_result = lend_runner._execute_run(
            phase="nbfi_lend",
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
        )

        arm_summaries = {"idle": idle_result, "lend": lend_result}
        return self._build_result_from_summaries(
            arm_summaries, kappa, concentration, mu, monotonicity, outside_mid_ratio, seed
        )

    # ── Persistence ──────────────────────────────────────────────────────

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
        """Persist a run to Supabase."""
        if self._supabase_store is None:
            return
        try:
            from bilancio.storage.models import RegistryEntry, RunStatus

            status = RunStatus.COMPLETED if run_result.delta_total is not None else RunStatus.FAILED
            parameters = {
                "kappa": str(kappa),
                "concentration": str(concentration),
                "mu": str(mu),
                "outside_mid_ratio": str(outside_mid_ratio),
                "seed": seed,
                "regime": regime,
            }
            metrics: dict[str, Any] = {}
            if run_result.delta_total is not None:
                metrics["delta_total"] = float(run_result.delta_total)
            if run_result.phi_total is not None:
                metrics["phi_total"] = float(run_result.phi_total)
            metrics["n_defaults"] = run_result.n_defaults

            entry = RegistryEntry(
                run_id=run_result.run_id,
                experiment_id=self.job_id or "unknown",
                status=status,
                parameters=parameters,
                metrics=metrics,
                artifact_paths={},
            )
            self._supabase_store.upsert(entry)
            logger.debug(f"Persisted run {run_result.run_id} to Supabase")
        except EXTERNAL_OPERATION_ERRORS as e:
            logger.warning(f"Failed to persist run to Supabase: {e}")

    def _write_comparison_csv(self) -> None:
        """Write comparison results to CSV."""
        self.aggregate_dir.mkdir(parents=True, exist_ok=True)
        with self.comparison_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COMPARISON_FIELDS)
            writer.writeheader()
            for r in self.comparison_results:
                row = {
                    "kappa": str(r.kappa),
                    "concentration": str(r.concentration),
                    "mu": str(r.mu),
                    "monotonicity": str(r.monotonicity),
                    "seed": str(r.seed),
                    "outside_mid_ratio": str(r.outside_mid_ratio),
                    "delta_idle": str(r.delta_idle) if r.delta_idle is not None else "",
                    "delta_lend": str(r.delta_lend) if r.delta_lend is not None else "",
                    "lending_effect": str(r.lending_effect) if r.lending_effect is not None else "",
                    "lending_relief_ratio": str(r.lending_relief_ratio) if r.lending_relief_ratio is not None else "",
                    "phi_idle": str(r.phi_idle) if r.phi_idle is not None else "",
                    "phi_lend": str(r.phi_lend) if r.phi_lend is not None else "",
                    "idle_run_id": r.idle_run_id,
                    "idle_status": r.idle_status,
                    "lend_run_id": r.lend_run_id,
                    "lend_status": r.lend_status,
                    "n_defaults_idle": str(r.n_defaults_idle),
                    "n_defaults_lend": str(r.n_defaults_lend),
                    "cascade_fraction_idle": str(r.cascade_fraction_idle) if r.cascade_fraction_idle is not None else "",
                    "cascade_fraction_lend": str(r.cascade_fraction_lend) if r.cascade_fraction_lend is not None else "",
                    "total_loss_idle": str(r.total_loss_idle),
                    "total_loss_lend": str(r.total_loss_lend),
                    "total_loss_pct_idle": str(r.total_loss_pct_idle) if r.total_loss_pct_idle is not None else "",
                    "total_loss_pct_lend": str(r.total_loss_pct_lend) if r.total_loss_pct_lend is not None else "",
                    "intermediary_loss_idle": str(r.intermediary_loss_idle),
                    "intermediary_loss_lend": str(r.intermediary_loss_lend),
                    "intermediary_loss_pct_idle": str(r.intermediary_loss_pct_idle) if r.intermediary_loss_pct_idle is not None else "",
                    "intermediary_loss_pct_lend": str(r.intermediary_loss_pct_lend) if r.intermediary_loss_pct_lend is not None else "",
                    "system_loss_idle": str(r.system_loss_idle),
                    "system_loss_lend": str(r.system_loss_lend),
                    "system_loss_pct_idle": str(r.system_loss_pct_idle) if r.system_loss_pct_idle is not None else "",
                    "system_loss_pct_lend": str(r.system_loss_pct_lend) if r.system_loss_pct_lend is not None else "",
                    "system_loss_lending_effect": str(r.system_loss_lending_effect) if r.system_loss_lending_effect is not None else "",
                }
                writer.writerow(row)

    def _write_summary_json(self) -> None:
        """Write summary statistics to JSON."""
        if self.skip_local_processing:
            return
        completed = [r for r in self.comparison_results if r.lending_effect is not None]

        if completed:
            delta_idles = [float(r.delta_idle) for r in completed if r.delta_idle is not None]
            delta_lends = [float(r.delta_lend) for r in completed if r.delta_lend is not None]
            effects = [float(r.lending_effect) for r in completed if r.lending_effect is not None]

            mean_delta_idle = sum(delta_idles) / len(delta_idles) if delta_idles else None
            mean_delta_lend = sum(delta_lends) / len(delta_lends) if delta_lends else None
            mean_effect = sum(effects) / len(effects) if effects else None

            improved = sum(1 for r in completed if r.lending_effect and r.lending_effect > 0)
            unchanged = sum(1 for r in completed if r.lending_effect == 0)
            worsened = sum(1 for r in completed if r.lending_effect and r.lending_effect < 0)
        else:
            mean_delta_idle = None
            mean_delta_lend = None
            mean_effect = None
            improved = 0
            unchanged = 0
            worsened = 0

        summary = {
            "total_combos": len(self.comparison_results),
            "completed_combos": len(completed),
            "mean_delta_idle": mean_delta_idle,
            "mean_delta_lend": mean_delta_lend,
            "mean_lending_effect": mean_effect,
            "combos_improved": improved,
            "combos_unchanged": unchanged,
            "combos_worsened": worsened,
            "config": {
                "n_agents": self.config.n_agents,
                "maturity_days": self.config.maturity_days,
                "Q_total": str(self.config.Q_total),
                "base_seed": self.config.base_seed,
                "face_value": str(self.config.face_value),
                "lender_share": str(self.config.lender_share),
                "kappas": [str(k) for k in self.config.kappas],
                "concentrations": [str(c) for c in self.config.concentrations],
                "mus": [str(m) for m in self.config.mus],
                "outside_mid_ratios": [str(r) for r in self.config.outside_mid_ratios],
            },
        }

        with self.summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2)

    def _write_stats_analysis(self) -> None:
        """Write statistical analysis files using RingSweepAnalysis (skipped in cloud-only mode)."""
        if self.skip_local_processing:
            return
        if not self.comparison_results:
            return
        try:
            import pandas as pd

            from bilancio.experiments.sweep_analysis import RingSweepAnalysis

            comp_df = pd.read_csv(self.comparison_path)
            comp_df = comp_df.rename(columns={
                "delta_idle": "delta_passive",
                "delta_lend": "delta_active",
                "lending_effect": "trading_effect",
                "phi_idle": "phi_passive",
                "phi_lend": "phi_active",
            })
            analysis = RingSweepAnalysis(comp_df.to_dict("records"))
            if analysis.min_replicates() < 2:
                logger.info(
                    "Skipping statistical analysis: need >= 2 replicates per cell, have %d",
                    analysis.min_replicates(),
                )
                return
            paths = analysis.write_stats(self.aggregate_dir)
            for name, path in paths.items():
                logger.info("Stats %s written to %s", name, path)
            print(f"Statistical analysis written to {self.aggregate_dir}", flush=True)
        except EXTERNAL_OPERATION_ERRORS as e:
            logger.warning("Statistical analysis failed: %s", e)
            print(f"Warning: Statistical analysis failed: {e}", flush=True)

    def _write_activity_analysis(self) -> None:
        """Write mechanism activity analysis (skipped in cloud-only mode)."""
        if self.skip_local_processing:
            return
        if not self.comparison_results:
            return
        try:
            from bilancio.analysis.mechanism_activity import run_mechanism_activity_analysis

            analysis_dir = self.aggregate_dir / "analysis"
            paths = run_mechanism_activity_analysis(
                experiment_root=self.aggregate_dir.parent,
                sweep_type="nbfi",
                output_dir=analysis_dir,
            )
            for name, path in paths.items():
                logger.info("Activity analysis %s: %s", name, path)
            print(f"Mechanism activity analysis written to {analysis_dir}", flush=True)
        except EXTERNAL_OPERATION_ERRORS as e:
            logger.warning("Activity analysis failed: %s", e)
            print(f"Warning: Activity analysis failed: {e}", flush=True)
