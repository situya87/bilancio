"""Utilities for running Kalecki ring experiment sweeps."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from bilancio.analysis.metrics_computer import MetricsComputer
from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.experiments.sampling import (
    generate_frontier_params,
    generate_grid_params,
    generate_lhs_params,
)
from bilancio.runners import ExecutionResult, LocalExecutor, RunOptions
from bilancio.runners.protocols import SimulationExecutor
from bilancio.scenarios import compile_ring_explorer
from bilancio.storage import (
    FileRegistryStore,
    LocalArtifactLoader,
    ModalVolumeArtifactLoader,
    RegistryEntry,
)
from bilancio.storage.models import RunStatus
from bilancio.storage.protocols import RegistryStore

EXTERNAL_SERVICE_ERRORS = (
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
class RingRunSummary:
    run_id: str
    phase: str
    kappa: Decimal
    concentration: Decimal
    mu: Decimal
    monotonicity: Decimal
    delta_total: Decimal | None
    phi_total: Decimal | None
    time_to_stability: int
    # Cascade/contagion metrics
    n_defaults: int = 0
    cascade_fraction: Decimal | None = None
    # CB stress metrics (populated for banking runs)
    cb_loans_created_count: int = 0
    cb_interest_total_paid: int = 0
    cb_loans_outstanding_pre_final: int = 0
    bank_defaults_final: int = 0
    cb_reserve_destruction_pct: float = 0.0
    # Banking-specific default metrics (Plan 039)
    delta_bank: float | None = None
    deposit_loss_gross: int = 0
    deposit_loss_pct: float | None = None
    payable_default_loss: int = 0
    total_loss: int = 0
    total_loss_pct: float | None = None
    # Intermediary loss metrics (non-trader entity losses)
    intermediary_loss_total: float = 0.0
    dealer_vbt_loss: float = 0.0
    nbfi_loan_loss: float = 0.0
    bank_credit_loss: float = 0.0
    cb_backstop_loss: float = 0.0
    # Initial intermediary capital (for normalization)
    initial_intermediary_capital: float = 0.0
    initial_dealer_vbt_cash: float = 0.0
    initial_lender_cash: float = 0.0
    initial_bank_reserves: float = 0.0
    # Dealer metrics (only populated for treatment runs with dealer enabled)
    dealer_metrics: dict[str, Any] | None = None
    # Modal call ID for cloud execution debugging
    modal_call_id: str | None = None


@dataclass
class PreparedRun:
    """Data needed to execute and finalize a prepared run.

    Created by RingSweepRunner._prepare_run(), consumed by _finalize_run().
    """

    run_id: str
    phase: str
    kappa: Decimal
    concentration: Decimal
    mu: Decimal
    monotonicity: Decimal
    seed: int
    scenario_config: dict[str, Any]
    options: RunOptions
    run_dir: Path
    out_dir: Path
    scenario_path: Path
    base_params: dict[str, Any]
    S1: Decimal
    L0: Decimal


def _decimal_list(spec: str) -> list[Decimal]:
    out: list[Decimal] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(Decimal(part))
    return out


def _to_yaml_ready(obj: Any) -> Any:
    from decimal import Decimal as _D

    if isinstance(obj, dict):
        return {k: _to_yaml_ready(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_to_yaml_ready(v) for v in obj]
    if isinstance(obj, _D):
        norm = obj.normalize()
        if norm == norm.to_integral_value():
            return int(norm)
        return float(norm)
    return obj


class _RingSweepGridConfig(BaseModel):
    enabled: bool = True
    kappas: list[Decimal] = Field(default_factory=list)
    concentrations: list[Decimal] = Field(default_factory=list)
    mus: list[Decimal] = Field(default_factory=list)
    monotonicities: list[Decimal] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lists(self) -> _RingSweepGridConfig:
        if self.enabled:
            if not self.kappas:
                raise ValueError("grid.kappas must be provided when grid.enabled is true")
            if not self.concentrations:
                raise ValueError("grid.concentrations must be provided when grid.enabled is true")
            if not self.mus:
                raise ValueError("grid.mus must be provided when grid.enabled is true")
            if not self.monotonicities:
                self.monotonicities = [Decimal("0")]
        return self


class _RingSweepLHSConfig(BaseModel):
    count: int = 0
    kappa_range: tuple[Decimal, Decimal] | None = None
    concentration_range: tuple[Decimal, Decimal] | None = None
    mu_range: tuple[Decimal, Decimal] | None = None
    monotonicity_range: tuple[Decimal, Decimal] | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> _RingSweepLHSConfig:
        if self.count <= 0:
            return self
        if self.monotonicity_range is None:
            self.monotonicity_range = (Decimal("0"), Decimal("0"))
        for name, rng in (
            ("kappa_range", self.kappa_range),
            ("concentration_range", self.concentration_range),
            ("mu_range", self.mu_range),
            ("monotonicity_range", self.monotonicity_range),
        ):
            if rng is None or len(rng) != 2:
                raise ValueError(f"lhs.{name} must contain exactly two values when lhs.count > 0")
        return self


class _RingSweepFrontierConfig(BaseModel):
    enabled: bool = False
    kappa_low: Decimal | None = None
    kappa_high: Decimal | None = None
    tolerance: Decimal | None = None
    max_iterations: int | None = None

    @model_validator(mode="after")
    def validate_frontier(self) -> _RingSweepFrontierConfig:
        if not self.enabled:
            return self
        missing = [
            name
            for name in ("kappa_low", "kappa_high", "tolerance", "max_iterations")
            if getattr(self, name) is None
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"frontier fields missing when frontier.enabled is true: {missing_list}"
            )
        return self


class _RingSweepRunnerConfig(BaseModel):
    n_agents: int | None = None
    maturity_days: int | None = None
    q_total: Decimal | None = None
    liquidity_mode: str | None = None
    liquidity_agent: str | None = None
    base_seed: int | None = None
    name_prefix: str | None = None
    default_handling: str | None = None
    dealer_enabled: bool = False
    dealer_config: dict[str, Any] | None = None
    risk_assessment_enabled: bool = True
    risk_assessment_config: dict[str, Any] | None = None


class RingSweepConfig(BaseModel):
    version: int = Field(1, description="Configuration version")
    out_dir: str | None = None
    grid: _RingSweepGridConfig | None = None
    lhs: _RingSweepLHSConfig | None = None
    frontier: _RingSweepFrontierConfig | None = None
    runner: _RingSweepRunnerConfig | None = None

    @model_validator(mode="after")
    def ensure_version(self) -> RingSweepConfig:
        if self.version != 1:
            raise ValueError(f"Unsupported sweep config version: {self.version}")
        return self


def load_ring_sweep_config(path: Path | str) -> RingSweepConfig:
    """Load and validate a ring sweep configuration from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep configuration not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Failed to parse YAML from {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Sweep configuration must be a YAML mapping")

    try:
        return RingSweepConfig.model_validate(raw)
    except ValidationError as exc:
        messages = []
        for error in exc.errors():
            loc = " -> ".join(str(part) for part in error.get("loc", ()))
            messages.append(f"  - {loc}: {error.get('msg')}")
        details = "\n".join(messages)
        raise ValueError(f"Invalid sweep configuration:\n{details}") from exc


class RingSweepRunner:
    """Coordinator for running Kalecki ring experiments."""

    def __init__(
        self,
        out_dir: Path,
        *,
        name_prefix: str,
        n_agents: int,
        maturity_days: int,
        Q_total: Decimal,
        liquidity_mode: str,
        liquidity_agent: str | None,
        base_seed: int,
        default_handling: str = "fail-fast",
        dealer_enabled: bool = False,
        dealer_config: dict[str, Any] | None = None,
        risk_assessment_enabled: bool = True,
        risk_assessment_config: dict[str, Any] | None = None,
        balanced_mode: bool = False,
        face_value: Decimal | None = None,
        outside_mid_ratio: Decimal | None = None,
        big_entity_share: Decimal | None = None,  # DEPRECATED
        vbt_share_per_bucket: Decimal | None = None,
        dealer_share_per_bucket: Decimal | None = None,
        rollover_enabled: bool = True,
        detailed_dealer_logging: bool = False,  # Plan 022
        registry_store: RegistryStore | None = None,  # Plan 026
        executor: SimulationExecutor | None = None,  # Plan 027
        quiet: bool = True,  # Plan 030: suppress verbose output for sweeps
        alpha_vbt: Decimal = Decimal("0"),
        alpha_trader: Decimal = Decimal("0"),
        risk_aversion: Decimal = Decimal("0"),
        planning_horizon: int = 10,
        aggressiveness: Decimal = Decimal("1.0"),
        default_observability: Decimal = Decimal("1.0"),
        vbt_mid_sensitivity: Decimal = Decimal("1.0"),
        vbt_spread_sensitivity: Decimal = Decimal("0.0"),
        trading_motive: str = "liquidity_then_earning",
        lender_mode: bool = False,
        lender_share: Decimal = Decimal("0.10"),
        balanced_mode_override: str | None = None,
        n_banks: int = 0,
        reserve_multiplier: float = 10.0,
        equalize_capacity: bool = False,
        credit_risk_loading: Decimal = Decimal("0.5"),
        max_borrower_risk: Decimal = Decimal("0.4"),
        min_coverage_ratio: Decimal = Decimal("0"),
        cb_rate_escalation_slope: Decimal = Decimal("0.05"),
        cb_max_outstanding_ratio: Decimal = Decimal("2.0"),
        spread_scale: Decimal = Decimal("1.0"),
        cb_lending_cutoff_day: int | None = None,
        trading_rounds: int = 1,
    ) -> None:
        self.base_dir = out_dir
        self.registry_dir = self.base_dir / "registry"
        self.runs_dir = self.base_dir / "runs"
        self.aggregate_dir = self.base_dir / "aggregate"
        self.name_prefix = name_prefix
        self.n_agents = n_agents
        self.maturity_days = maturity_days
        self.Q_total = Q_total
        self.liquidity_mode = liquidity_mode
        self.liquidity_agent = liquidity_agent
        self.seed_counter = base_seed
        self.default_handling = default_handling
        self.dealer_enabled = dealer_enabled
        self.dealer_config = dealer_config
        self.risk_assessment_enabled = risk_assessment_enabled
        self.risk_assessment_config = risk_assessment_config
        self.balanced_mode = balanced_mode
        self.face_value = face_value or Decimal("20")
        self.outside_mid_ratio = outside_mid_ratio or Decimal("1.0")
        self.big_entity_share = big_entity_share or Decimal("0.25")  # DEPRECATED
        self.vbt_share_per_bucket = vbt_share_per_bucket or Decimal("0.20")
        self.dealer_share_per_bucket = dealer_share_per_bucket or Decimal("0.05")
        self.rollover_enabled = rollover_enabled
        self.detailed_dealer_logging = detailed_dealer_logging  # Plan 022
        self.quiet = quiet  # Plan 030: suppress verbose output
        self.alpha_vbt = alpha_vbt
        self.alpha_trader = alpha_trader
        self.risk_aversion = risk_aversion
        self.planning_horizon = planning_horizon
        self.aggressiveness = aggressiveness
        self.default_observability = default_observability
        self.vbt_mid_sensitivity = vbt_mid_sensitivity
        self.vbt_spread_sensitivity = vbt_spread_sensitivity
        self.trading_motive = trading_motive
        self.lender_mode = lender_mode
        self.lender_share = lender_share
        self.balanced_mode_override = balanced_mode_override
        self.n_banks = n_banks
        self.reserve_multiplier = reserve_multiplier
        self.equalize_capacity = equalize_capacity
        self.credit_risk_loading = credit_risk_loading
        self.max_borrower_risk = max_borrower_risk
        self.min_coverage_ratio = min_coverage_ratio
        self.cb_rate_escalation_slope = cb_rate_escalation_slope
        self.cb_max_outstanding_ratio = cb_max_outstanding_ratio
        self.spread_scale = spread_scale
        self.cb_lending_cutoff_day = cb_lending_cutoff_day
        self.trading_rounds = trading_rounds

        # Use provided registry store or create default file-based store
        self.registry_store: RegistryStore = registry_store or FileRegistryStore(self.base_dir)
        # Use provided executor or create default local executor (Plan 027)
        self.executor: SimulationExecutor = executor or LocalExecutor()
        self.experiment_id = ""  # Empty = use base_dir directly

        # Cloud-only mode: skip local processing when using cloud executor
        # This avoids downloading artifacts just to recompute metrics locally
        from bilancio.runners.cloud_executor import CloudExecutor

        self.skip_local_processing = isinstance(executor, CloudExecutor)

        # Only create local directories if we're doing local processing
        if not self.skip_local_processing:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            self.aggregate_dir.mkdir(parents=True, exist_ok=True)

            # Initialize empty registry file if it doesn't exist (backward compatible)
            registry_path = self.registry_dir / "experiments.csv"
            if not registry_path.exists():
                self._init_empty_registry(registry_path)

    def _init_empty_registry(self, registry_path: Path) -> None:
        """Create an empty registry file with headers."""
        import csv

        default_fields = [
            "run_id",
            "experiment_id",
            "phase",
            "status",
            "error",
            "seed",
            "n_agents",
            "kappa",
            "concentration",
            "mu",
            "monotonicity",
            "maturity_days",
            "Q_total",
            "S1",
            "L0",
            "default_handling",
            "dealer_enabled",
            "phi_total",
            "delta_total",
            "time_to_stability",
            "scenario_yaml",
            "events_jsonl",
            "balances_csv",
            "metrics_csv",
            "metrics_html",
            "run_html",
        ]
        with registry_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=default_fields)
            writer.writeheader()

    def _next_seed(self) -> int:
        value = self.seed_counter
        self.seed_counter += 1
        return value

    def _upsert_registry(
        self,
        run_id: str,
        phase: str,
        status: RunStatus,
        parameters: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        artifact_paths: dict[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        """Upsert a registry entry using the configured store."""
        entry = RegistryEntry(
            run_id=run_id,
            experiment_id=self.experiment_id,
            status=status,
            parameters=parameters,
            metrics=metrics or {},
            artifact_paths=artifact_paths or {},
            error=error,
        )
        self.registry_store.upsert(entry)

    def run_grid(
        self,
        kappas: Sequence[Decimal],
        concentrations: Sequence[Decimal],
        mus: Sequence[Decimal],
        monotonicities: Sequence[Decimal],
    ) -> list[RingRunSummary]:
        summaries: list[RingRunSummary] = []
        for kappa, concentration, mu, monotonicity in generate_grid_params(
            kappas, concentrations, mus, monotonicities
        ):
            seed = self._next_seed()
            summaries.append(
                self._execute_run(
                    "grid",
                    kappa,
                    concentration,
                    mu,
                    monotonicity,
                    seed,
                )
            )
        return summaries

    def run_lhs(
        self,
        count: int,
        *,
        kappa_range: tuple[Decimal, Decimal],
        concentration_range: tuple[Decimal, Decimal],
        mu_range: tuple[Decimal, Decimal],
        monotonicity_range: tuple[Decimal, Decimal],
    ) -> list[RingRunSummary]:
        if count <= 0:
            return []
        summaries: list[RingRunSummary] = []
        for kappa, concentration, mu, monotonicity in generate_lhs_params(
            count,
            kappa_range=kappa_range,
            concentration_range=concentration_range,
            mu_range=mu_range,
            monotonicity_range=monotonicity_range,
            seed=self.seed_counter,
        ):
            seed = self._next_seed()
            summaries.append(
                self._execute_run(
                    "lhs",
                    kappa,
                    concentration,
                    mu,
                    monotonicity,
                    seed,
                )
            )
        return summaries

    def run_frontier(
        self,
        concentrations: Sequence[Decimal],
        mus: Sequence[Decimal],
        monotonicities: Sequence[Decimal],
        *,
        kappa_low: Decimal,
        kappa_high: Decimal,
        tolerance: Decimal,
        max_iterations: int,
    ) -> list[RingRunSummary]:
        summaries: list[RingRunSummary] = []

        # Create execution function that captures self and returns delta_total
        def execute_fn(
            label: str,
            kappa: Decimal,
            concentration: Decimal,
            mu: Decimal,
            monotonicity: Decimal,
        ) -> Decimal | None:
            # Execute run with label
            summary = self._execute_run(
                "frontier",
                kappa,
                concentration,
                mu,
                monotonicity,
                self._next_seed(),
                label=label,
            )
            summaries.append(summary)
            return summary.delta_total

        # Use frontier sampling to execute runs with binary search
        # Unlike grid/LHS, frontier calls execute_fn directly for immediate feedback
        generate_frontier_params(
            concentrations,
            mus,
            monotonicities,
            kappa_low=kappa_low,
            kappa_high=kappa_high,
            tolerance=tolerance,
            max_iterations=max_iterations,
            execute_fn=execute_fn,
        )

        return summaries

    def _execute_run(
        self,
        phase: str,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
        seed: int,
        *,
        label: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> RingRunSummary:
        # Pre-simulation trade viability check for active runs
        if phase in ("active", "treatment") and self.balanced_mode:
            import logging

            _viability_logger = logging.getLogger(__name__)
            try:
                from bilancio.specification.trade_viability import check_trade_viability

                report = check_trade_viability(
                    kappa=kappa,
                    face_value=self.face_value,
                    n_agents=self.n_agents,
                    dealer_share=self.dealer_share_per_bucket,
                    vbt_share=self.vbt_share_per_bucket,
                    layoff_threshold=Decimal("0.7"),
                    buy_premium=Decimal("0.01"),
                    maturity_days=self.maturity_days,
                    outside_mid_ratio=self.outside_mid_ratio,
                )
                if not report.all_viable:
                    _viability_logger.warning(
                        "Trade viability check failed for kappa=%s: %s",
                        kappa,
                        report.diagnostics,
                    )
            except EXTERNAL_SERVICE_ERRORS as exc:
                _viability_logger.debug("Viability check skipped: %s", exc)

        run_uuid = uuid.uuid4().hex[:12]
        run_id = f"{phase}_{label}_{run_uuid}" if label else f"{phase}_{run_uuid}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_dir = run_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        scenario_path = run_dir / "scenario.yaml"
        run_html_path = run_dir / "run.html"
        balances_path = out_dir / "balances.csv"
        events_path = out_dir / "events.jsonl"

        # Common parameters for all registry updates
        base_params = {
            "phase": phase,
            "seed": seed,
            "n_agents": self.n_agents,
            "kappa": str(kappa),
            "concentration": str(concentration),
            "mu": str(mu),
            "monotonicity": str(monotonicity),
            "maturity_days": self.maturity_days,
            "Q_total": str(self.Q_total),
            "default_handling": self.default_handling,
            "dealer_enabled": self.dealer_enabled,
        }

        # Initial "running" status
        self._upsert_registry(
            run_id=run_id,
            phase=phase,
            status=RunStatus.RUNNING,
            parameters=base_params,
        )

        generator_data = {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": self.name_prefix,
            "params": {
                "n_agents": self.n_agents,
                "seed": seed,
                "kappa": str(kappa),
                "Q_total": str(self.Q_total),
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": str(concentration),
                    "monotonicity": str(monotonicity),
                },
                "maturity": {
                    "days": self.maturity_days,
                    "mode": "lead_lag",
                    "mu": str(mu),
                },
                "liquidity": {
                    "allocation": self._liquidity_allocation_dict(),
                },
            },
            "compile": {"emit_yaml": False},
        }

        generator_config = RingExplorerGeneratorConfig.model_validate(generator_data)

        if self.balanced_mode:
            # Use balanced generator for C vs D comparison scenarios (Plan 024)
            from bilancio.scenarios import compile_ring_explorer_balanced

            scenario = compile_ring_explorer_balanced(
                generator_config,
                face_value=self.face_value,
                outside_mid_ratio=self.outside_mid_ratio,
                big_entity_share=self.big_entity_share,  # DEPRECATED
                vbt_share_per_bucket=self.vbt_share_per_bucket,
                dealer_share_per_bucket=self.dealer_share_per_bucket,
                mode=self.balanced_mode_override
                or (
                    "lender"
                    if self.lender_mode
                    else ("active" if self.dealer_enabled else "passive")
                ),
                lender_share=self.lender_share,
                rollover_enabled=self.rollover_enabled,
                kappa=kappa,
                n_banks=self.n_banks,
                reserve_multiplier=self.reserve_multiplier,
                equalize_capacity=self.equalize_capacity,
                credit_risk_loading=self.credit_risk_loading,
                max_borrower_risk=self.max_borrower_risk,
                min_coverage_ratio=self.min_coverage_ratio,
                cb_rate_escalation_slope=self.cb_rate_escalation_slope,
                cb_max_outstanding_ratio=self.cb_max_outstanding_ratio,
                spread_scale=self.spread_scale,
                cb_lending_cutoff_day=self.cb_lending_cutoff_day,
                source_path=None,
            )
        else:
            scenario = compile_ring_explorer(generator_config, source_path=None)

        # Add dealer config: always for active mode, also for balanced passive
        # (passive balanced runs need the subsystem initialized for PnL tracking)
        if self.dealer_enabled or self.balanced_mode:
            dealer_section: dict[str, Any] = {"enabled": True}
            if self.dealer_config:
                dealer_section.update(self.dealer_config)
            else:
                dealer_section.update(
                    {
                        "ticket_size": 1,
                        "dealer_share": Decimal("0.25"),
                        "vbt_share": Decimal("0.50"),
                    }
                )
            # Add risk assessment config if enabled
            if self.risk_assessment_enabled:
                risk_section: dict[str, Any] = {"enabled": True}
                if self.risk_assessment_config:
                    risk_section.update(self.risk_assessment_config)
                dealer_section["risk_assessment"] = risk_section
            scenario["dealer"] = dealer_section

            if self.balanced_mode:
                scenario["balanced_dealer"] = {
                    "enabled": True,
                    "face_value": str(self.face_value),
                    "outside_mid_ratio": str(self.outside_mid_ratio),
                    "vbt_share_per_bucket": str(self.vbt_share_per_bucket),
                    "dealer_share_per_bucket": str(self.dealer_share_per_bucket),
                    "mode": self.balanced_mode_override
                    or (
                        "lender"
                        if self.lender_mode
                        else ("active" if self.dealer_enabled else "passive")
                    ),
                    "rollover_enabled": self.rollover_enabled,
                    "alpha_vbt": str(self.alpha_vbt),
                    "alpha_trader": str(self.alpha_trader),
                    "kappa": str(kappa),
                    "risk_aversion": str(self.risk_aversion),
                    "planning_horizon": self.planning_horizon,
                    "aggressiveness": str(self.aggressiveness),
                    "default_observability": str(self.default_observability),
                    "vbt_mid_sensitivity": str(self.vbt_mid_sensitivity),
                    "vbt_spread_sensitivity": str(self.vbt_spread_sensitivity),
                    "trading_motive": self.trading_motive,
                    "spread_scale": str(self.spread_scale),
                    "trading_rounds": self.trading_rounds,
                }

            if self.lender_mode:
                scenario["lender"] = {
                    "enabled": True,
                    "base_rate": "0.05",
                    "risk_premium_scale": "0.20",
                    "max_single_exposure": "0.15",
                    "max_total_exposure": "0.80",
                    "maturity_days": 2,
                    "horizon": 5,
                    "kappa": str(kappa),  # LenderProfile: kappa-aware pricing
                    "risk_aversion": "0.3",
                    "planning_horizon": 5,
                    "profit_target": "0.05",
                }

        if self.default_handling:
            scenario_run = scenario.setdefault("run", {})
            scenario_run["default_handling"] = self.default_handling

        # RingSweepRunner writes scenario.yaml itself for control
        with scenario_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(_to_yaml_ready(scenario), fh, sort_keys=False, allow_unicode=False)

        S1 = Decimal("0")
        L0 = Decimal("0")
        for action in scenario.get("initial_actions", []):
            if "create_payable" in action:
                S1 += action["create_payable"]["amount"]
            if "mint_cash" in action:
                L0 += action["mint_cash"]["amount"]

        # Determine regime for logging (Plan 022)
        regime = self.balanced_mode_override or (
            "lender" if self.lender_mode else ("active" if self.dealer_enabled else "passive")
        )

        # Build RunOptions from scenario configuration (Plan 027)
        options = RunOptions(
            mode="until_stable",
            max_days=scenario.get("run", {}).get("max_days", 90),
            quiet_days=scenario.get("run", {}).get("quiet_days", 2),
            check_invariants="daily",
            default_handling=self.default_handling,
            # Plan 030: Use "none" for quiet mode to suppress verbose console output
            show_events="none"
            if self.quiet
            else scenario.get("run", {}).get("show", {}).get("events", "detailed"),
            show_balances=scenario.get("run", {}).get("show", {}).get("balances"),
            t_account=False,
            detailed_dealer_logging=self.detailed_dealer_logging,
            run_id=run_id,
            regime=regime,
            # Run parameters for Supabase tracking
            kappa=float(kappa),
            concentration=float(concentration),
            mu=float(mu),
            outside_mid_ratio=float(self.outside_mid_ratio) if self.outside_mid_ratio else 1.0,
            seed=seed,
        )

        # Delegate simulation to executor (Plan 027)
        result = self.executor.execute(
            scenario_config=_to_yaml_ready(scenario),
            run_id=run_id,
            output_dir=run_dir,
            options=options,
        )

        # Handle failure case
        if result.status == RunStatus.FAILED:
            fail_params = {**base_params, "S1": str(S1), "L0": str(L0)}
            self._upsert_registry(
                run_id=run_id,
                phase=phase,
                status=RunStatus.FAILED,
                parameters=fail_params,
                artifact_paths={
                    "scenario_yaml": self._rel_path(scenario_path),
                    "run_html": self._rel_path(run_html_path),
                },
                error=result.error,
            )
            return RingRunSummary(
                run_id=run_id,
                phase=phase,
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                monotonicity=monotonicity,
                delta_total=None,
                phi_total=None,
                time_to_stability=0,
                modal_call_id=result.modal_call_id,
            )

        # Use MetricsComputer for analytics (Plan 027)
        # result.artifacts contains relative paths (e.g., "out/events.jsonl")
        artifacts: dict[str, str] = {}
        if "events_jsonl" in result.artifacts:
            artifacts["events_jsonl"] = result.artifacts["events_jsonl"]
        if "balances_csv" in result.artifacts:
            artifacts["balances_csv"] = result.artifacts["balances_csv"]

        loader = self._artifact_loader_for_result(result)
        computer = MetricsComputer(loader)
        bundle = computer.compute(artifacts)

        # Write metrics outputs
        output_paths = computer.write_outputs(bundle, out_dir)

        # Extract summary metrics
        delta_total = bundle.summary.get("delta_total")
        phi_total = bundle.summary.get("phi_total")
        time_to_stability = int(bundle.summary.get("max_day") or 0)
        n_defaults = int(bundle.summary.get("n_defaults", 0))
        cascade_fraction_val = bundle.summary.get("cascade_fraction")

        # CB stress metrics (Plan 038)
        cb_loans_created_count = int(
            bundle.summary.get("cb_loans_created_count", 0)
        )
        cb_interest_total_paid = int(
            bundle.summary.get("cb_interest_total_paid", 0)
        )
        cb_loans_outstanding_pre_final = int(
            bundle.summary.get("cb_loans_outstanding_pre_final", 0)
        )
        bank_defaults_final = int(
            bundle.summary.get("bank_defaults_final", 0)
        )
        cb_reserve_destruction_pct = float(
            bundle.summary.get("cb_reserve_destruction_pct", 0.0)
        )

        # Banking-specific default metrics (Plan 039)
        delta_bank = bundle.summary.get("delta_bank")
        deposit_loss_gross = int(
            bundle.summary.get("deposit_loss_gross", 0)
        )
        deposit_loss_pct = bundle.summary.get("deposit_loss_pct")

        # Total loss metric
        payable_default_loss = int(
            bundle.summary.get("payable_default_loss", 0)
        )
        total_loss = int(bundle.summary.get("total_loss", 0))
        S_total = float(bundle.summary.get("S_total", 0))
        total_loss_pct = total_loss / S_total if S_total > 0 else None

        # Read dealer metrics if available (treatment runs with dealer enabled)
        dealer_metrics: dict[str, Any] | None = None
        dealer_metrics_path = out_dir / "dealer_metrics.json"
        if dealer_metrics_path.exists():
            import json

            with dealer_metrics_path.open() as f:
                dealer_metrics = json.load(f)

        # Update registry with completed status
        success_params = {**base_params, "S1": str(S1), "L0": str(L0)}
        success_metrics = {
            "time_to_stability": time_to_stability,
            "phi_total": str(phi_total) if phi_total is not None else "",
            "delta_total": str(delta_total) if delta_total is not None else "",
            "n_defaults": str(n_defaults),
            "cascade_fraction": str(cascade_fraction_val)
            if cascade_fraction_val is not None
            else "",
        }
        self._upsert_registry(
            run_id=run_id,
            phase=phase,
            status=RunStatus.COMPLETED,
            parameters=success_params,
            metrics=success_metrics,
            artifact_paths={
                "scenario_yaml": self._rel_path(scenario_path),
                "events_jsonl": self._rel_path(events_path),
                "balances_csv": self._rel_path(balances_path),
                "metrics_csv": self._rel_path(output_paths["metrics_csv"]),
                "metrics_html": self._rel_path(output_paths["metrics_html"]),
                "run_html": self._rel_path(run_html_path),
            },
        )

        # Extract initial intermediary capitals from scenario config
        from bilancio.analysis.report import extract_initial_capitals
        capitals = extract_initial_capitals(scenario)

        return RingRunSummary(
            run_id=run_id,
            phase=phase,
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            delta_total=delta_total,
            phi_total=phi_total,
            time_to_stability=time_to_stability,
            n_defaults=n_defaults,
            cascade_fraction=cascade_fraction_val,
            dealer_metrics=dealer_metrics,
            modal_call_id=result.modal_call_id,
            cb_loans_created_count=cb_loans_created_count,
            cb_interest_total_paid=cb_interest_total_paid,
            cb_loans_outstanding_pre_final=cb_loans_outstanding_pre_final,
            bank_defaults_final=bank_defaults_final,
            cb_reserve_destruction_pct=cb_reserve_destruction_pct,
            delta_bank=delta_bank,
            deposit_loss_gross=deposit_loss_gross,
            deposit_loss_pct=deposit_loss_pct,
            payable_default_loss=payable_default_loss,
            total_loss=total_loss,
            total_loss_pct=total_loss_pct,
            nbfi_loan_loss=int(bundle.summary.get("nbfi_loan_loss", 0)),
            bank_credit_loss=int(bundle.summary.get("bank_credit_loss", 0)),
            cb_backstop_loss=int(bundle.summary.get("cb_backstop_loss", 0)),
            dealer_vbt_loss=max(0.0, -float((dealer_metrics or {}).get("dealer_total_pnl", 0))),
            intermediary_loss_total=(
                max(0.0, -float((dealer_metrics or {}).get("dealer_total_pnl", 0)))
                + int(bundle.summary.get("nbfi_loan_loss", 0))
                + int(bundle.summary.get("bank_credit_loss", 0))
                + int(bundle.summary.get("cb_backstop_loss", 0))
            ),
            initial_intermediary_capital=capitals["intermediary_capital"],
            initial_dealer_vbt_cash=capitals["dealer_capital"] + capitals["vbt_capital"],
            initial_lender_cash=capitals["lender_capital"],
            initial_bank_reserves=capitals["bank_capital"],
        )

    def _prepare_run(
        self,
        phase: str,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
        seed: int,
        label: str = "",
    ) -> PreparedRun:
        """Prepare a run without executing it.

        Creates directories, builds scenario config, writes scenario.yaml.
        Returns PreparedRun that can be passed to execute_batch and then _finalize_run.

        In cloud-only mode, skips local directory creation and file writes.
        """
        run_uuid = uuid.uuid4().hex[:12]
        run_id = f"{phase}_{label}_{run_uuid}" if label else f"{phase}_{run_uuid}"

        # For cloud-only mode, use placeholder paths (won't be used)
        if self.skip_local_processing:
            run_dir = Path(f"/tmp/bilancio/{run_id}")  # Placeholder, never created
            out_dir = run_dir / "out"
            scenario_path = run_dir / "scenario.yaml"
        else:
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            out_dir = run_dir / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            scenario_path = run_dir / "scenario.yaml"

        base_params = {
            "phase": phase,
            "seed": seed,
            "n_agents": self.n_agents,
            "kappa": str(kappa),
            "concentration": str(concentration),
            "mu": str(mu),
            "monotonicity": str(monotonicity),
            "maturity_days": self.maturity_days,
            "Q_total": str(self.Q_total),
            "default_handling": self.default_handling,
            "dealer_enabled": self.dealer_enabled,
        }

        # Initial "running" status (skip for cloud-only mode)
        if not self.skip_local_processing:
            self._upsert_registry(
                run_id=run_id,
                phase=phase,
                status=RunStatus.RUNNING,
                parameters=base_params,
            )

        generator_data = {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": self.name_prefix,
            "params": {
                "n_agents": self.n_agents,
                "seed": seed,
                "kappa": str(kappa),
                "Q_total": str(self.Q_total),
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": str(concentration),
                    "monotonicity": str(monotonicity),
                },
                "maturity": {
                    "days": self.maturity_days,
                    "mode": "lead_lag",
                    "mu": str(mu),
                },
                "liquidity": {
                    "allocation": self._liquidity_allocation_dict(),
                },
            },
            "compile": {"emit_yaml": False},
        }

        generator_config = RingExplorerGeneratorConfig.model_validate(generator_data)

        if self.balanced_mode:
            from bilancio.scenarios import compile_ring_explorer_balanced

            scenario = compile_ring_explorer_balanced(
                generator_config,
                face_value=self.face_value,
                outside_mid_ratio=self.outside_mid_ratio,
                big_entity_share=self.big_entity_share,
                vbt_share_per_bucket=self.vbt_share_per_bucket,
                dealer_share_per_bucket=self.dealer_share_per_bucket,
                mode=self.balanced_mode_override
                or (
                    "lender"
                    if self.lender_mode
                    else ("active" if self.dealer_enabled else "passive")
                ),
                lender_share=self.lender_share,
                rollover_enabled=self.rollover_enabled,
                kappa=kappa,
                n_banks=self.n_banks,
                reserve_multiplier=self.reserve_multiplier,
                equalize_capacity=self.equalize_capacity,
                credit_risk_loading=self.credit_risk_loading,
                max_borrower_risk=self.max_borrower_risk,
                min_coverage_ratio=self.min_coverage_ratio,
                cb_rate_escalation_slope=self.cb_rate_escalation_slope,
                cb_max_outstanding_ratio=self.cb_max_outstanding_ratio,
                spread_scale=self.spread_scale,
                cb_lending_cutoff_day=self.cb_lending_cutoff_day,
                source_path=None,
            )
        else:
            scenario = compile_ring_explorer(generator_config, source_path=None)

        # Add dealer config: always for active mode, also for balanced passive
        # (passive balanced runs need the subsystem initialized for PnL tracking)
        if self.dealer_enabled or self.balanced_mode:
            dealer_section: dict[str, Any] = {"enabled": True}
            if self.dealer_config:
                dealer_section.update(self.dealer_config)
            else:
                dealer_section.update(
                    {
                        "ticket_size": 1,
                        "dealer_share": Decimal("0.25"),
                        "vbt_share": Decimal("0.50"),
                    }
                )
            # Add risk assessment config if enabled
            if self.risk_assessment_enabled:
                risk_section: dict[str, Any] = {"enabled": True}
                if self.risk_assessment_config:
                    risk_section.update(self.risk_assessment_config)
                dealer_section["risk_assessment"] = risk_section
            scenario["dealer"] = dealer_section

            if self.balanced_mode:
                scenario["balanced_dealer"] = {
                    "enabled": True,
                    "face_value": str(self.face_value),
                    "outside_mid_ratio": str(self.outside_mid_ratio),
                    "vbt_share_per_bucket": str(self.vbt_share_per_bucket),
                    "dealer_share_per_bucket": str(self.dealer_share_per_bucket),
                    "mode": self.balanced_mode_override
                    or (
                        "lender"
                        if self.lender_mode
                        else ("active" if self.dealer_enabled else "passive")
                    ),
                    "rollover_enabled": self.rollover_enabled,
                    "alpha_vbt": str(self.alpha_vbt),
                    "alpha_trader": str(self.alpha_trader),
                    "kappa": str(kappa),
                    "risk_aversion": str(self.risk_aversion),
                    "planning_horizon": self.planning_horizon,
                    "aggressiveness": str(self.aggressiveness),
                    "default_observability": str(self.default_observability),
                    "vbt_mid_sensitivity": str(self.vbt_mid_sensitivity),
                    "vbt_spread_sensitivity": str(self.vbt_spread_sensitivity),
                    "trading_motive": self.trading_motive,
                    "spread_scale": str(self.spread_scale),
                    "trading_rounds": self.trading_rounds,
                }

            if self.lender_mode:
                scenario["lender"] = {
                    "enabled": True,
                    "base_rate": "0.05",
                    "risk_premium_scale": "0.20",
                    "max_single_exposure": "0.15",
                    "max_total_exposure": "0.80",
                    "maturity_days": 2,
                    "horizon": 5,
                    "kappa": str(kappa),  # LenderProfile: kappa-aware pricing
                    "risk_aversion": "0.3",
                    "planning_horizon": 5,
                    "profit_target": "0.05",
                }

        if self.default_handling:
            scenario_run = scenario.setdefault("run", {})
            scenario_run["default_handling"] = self.default_handling

        # Write scenario.yaml (skip for cloud-only mode)
        if not self.skip_local_processing:
            with scenario_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(_to_yaml_ready(scenario), fh, sort_keys=False, allow_unicode=False)

        S1 = Decimal("0")
        L0 = Decimal("0")
        for action in scenario.get("initial_actions", []):
            if "create_payable" in action:
                S1 += action["create_payable"]["amount"]
            if "mint_cash" in action:
                L0 += action["mint_cash"]["amount"]

        regime = self.balanced_mode_override or (
            "lender" if self.lender_mode else ("active" if self.dealer_enabled else "passive")
        )

        options = RunOptions(
            mode="until_stable",
            max_days=scenario.get("run", {}).get("max_days", 90),
            quiet_days=scenario.get("run", {}).get("quiet_days", 2),
            check_invariants="daily",
            default_handling=self.default_handling,
            show_events="none"
            if self.quiet
            else scenario.get("run", {}).get("show", {}).get("events", "detailed"),
            show_balances=scenario.get("run", {}).get("show", {}).get("balances"),
            t_account=False,
            detailed_dealer_logging=self.detailed_dealer_logging,
            run_id=run_id,
            regime=regime,
            # Run parameters for Supabase tracking
            kappa=float(kappa),
            concentration=float(concentration),
            mu=float(mu),
            outside_mid_ratio=float(self.outside_mid_ratio) if self.outside_mid_ratio else 1.0,
            seed=seed,
        )

        return PreparedRun(
            run_id=run_id,
            phase=phase,
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
            scenario_config=_to_yaml_ready(scenario),
            options=options,
            run_dir=run_dir,
            out_dir=out_dir,
            scenario_path=scenario_path,
            base_params=base_params,
            S1=S1,
            L0=L0,
        )

    def _finalize_run(
        self,
        prepared: PreparedRun,
        result: ExecutionResult,
    ) -> RingRunSummary:
        """Finalize a run after execution completes.

        For cloud execution with pre-computed metrics, uses those directly
        without downloading artifacts. For local execution, computes metrics
        from artifacts and updates local registry.
        """
        # Handle failure case
        if result.status == RunStatus.FAILED:
            if not self.skip_local_processing:
                run_html_path = prepared.run_dir / "run.html"
                fail_params = {
                    **prepared.base_params,
                    "S1": str(prepared.S1),
                    "L0": str(prepared.L0),
                }
                self._upsert_registry(
                    run_id=prepared.run_id,
                    phase=prepared.phase,
                    status=RunStatus.FAILED,
                    parameters=fail_params,
                    artifact_paths={
                        "scenario_yaml": self._rel_path(prepared.scenario_path),
                        "run_html": self._rel_path(run_html_path),
                    },
                    error=result.error,
                )
            return RingRunSummary(
                run_id=prepared.run_id,
                phase=prepared.phase,
                kappa=prepared.kappa,
                concentration=prepared.concentration,
                mu=prepared.mu,
                monotonicity=prepared.monotonicity,
                delta_total=None,
                phi_total=None,
                time_to_stability=0,
                modal_call_id=result.modal_call_id,
            )

        # Cloud-only path: use pre-computed metrics from Modal, skip local processing
        if self.skip_local_processing and result.metrics is not None:
            delta_total = result.metrics.get("delta_total")
            phi_total = result.metrics.get("phi_total")
            time_to_stability = int(result.metrics.get("max_day") or 0)
            n_defaults = int(result.metrics.get("n_defaults", 0))
            cascade_fraction_val = result.metrics.get("cascade_fraction")

            # Convert to Decimal for consistency
            if delta_total is not None:
                delta_total = Decimal(str(delta_total))
            if phi_total is not None:
                phi_total = Decimal(str(phi_total))
            if cascade_fraction_val is not None:
                cascade_fraction_val = Decimal(str(cascade_fraction_val))

            # Extract initial intermediary capitals from scenario config
            from bilancio.analysis.report import extract_initial_capitals
            capitals = extract_initial_capitals(prepared.scenario_config)

            return RingRunSummary(
                run_id=prepared.run_id,
                phase=prepared.phase,
                kappa=prepared.kappa,
                concentration=prepared.concentration,
                mu=prepared.mu,
                monotonicity=prepared.monotonicity,
                delta_total=delta_total,
                phi_total=phi_total,
                time_to_stability=time_to_stability,
                n_defaults=n_defaults,
                cascade_fraction=cascade_fraction_val,
                dealer_metrics=None,  # Dealer metrics not available in cloud path
                modal_call_id=result.modal_call_id,
                cb_loans_created_count=int(result.metrics.get("cb_loans_created_count", 0)),
                cb_interest_total_paid=int(result.metrics.get("cb_interest_total_paid", 0)),
                cb_loans_outstanding_pre_final=int(result.metrics.get("cb_loans_outstanding_pre_final", 0)),
                bank_defaults_final=int(result.metrics.get("bank_defaults_final", 0)),
                cb_reserve_destruction_pct=float(result.metrics.get("cb_reserve_destruction_pct", 0.0)),
                delta_bank=result.metrics.get("delta_bank"),
                deposit_loss_gross=int(result.metrics.get("deposit_loss_gross", 0)),
                deposit_loss_pct=result.metrics.get("deposit_loss_pct"),
                payable_default_loss=int(result.metrics.get("payable_default_loss", 0)),
                total_loss=int(result.metrics.get("total_loss", 0)),
                total_loss_pct=result.metrics.get("total_loss_pct"),
                nbfi_loan_loss=int(result.metrics.get("nbfi_loan_loss", 0)),
                bank_credit_loss=int(result.metrics.get("bank_credit_loss", 0)),
                cb_backstop_loss=int(result.metrics.get("cb_backstop_loss", 0)),
                dealer_vbt_loss=float(result.metrics.get("dealer_vbt_loss", 0.0)),
                intermediary_loss_total=float(result.metrics.get("intermediary_loss_total", 0.0)),
                initial_intermediary_capital=capitals["intermediary_capital"],
                initial_dealer_vbt_cash=capitals["dealer_capital"] + capitals["vbt_capital"],
                initial_lender_cash=capitals["lender_capital"],
                initial_bank_reserves=capitals["bank_capital"],
            )

        # Local path: load artifacts, compute metrics, update registry
        run_html_path = prepared.run_dir / "run.html"
        balances_path = prepared.out_dir / "balances.csv"
        events_path = prepared.out_dir / "events.jsonl"

        artifacts: dict[str, str] = {}
        if "events_jsonl" in result.artifacts:
            artifacts["events_jsonl"] = result.artifacts["events_jsonl"]
        if "balances_csv" in result.artifacts:
            artifacts["balances_csv"] = result.artifacts["balances_csv"]

        loader = self._artifact_loader_for_result(result)
        computer = MetricsComputer(loader)
        bundle = computer.compute(artifacts)

        output_paths = computer.write_outputs(bundle, prepared.out_dir)

        delta_total = bundle.summary.get("delta_total")
        phi_total = bundle.summary.get("phi_total")
        time_to_stability = int(bundle.summary.get("max_day") or 0)
        n_defaults = int(bundle.summary.get("n_defaults", 0))
        cascade_fraction_val = bundle.summary.get("cascade_fraction")

        dealer_metrics: dict[str, Any] | None = None
        dealer_metrics_path = prepared.out_dir / "dealer_metrics.json"
        if dealer_metrics_path.exists():
            import json

            with dealer_metrics_path.open() as f:
                dealer_metrics = json.load(f)

        success_params = {**prepared.base_params, "S1": str(prepared.S1), "L0": str(prepared.L0)}
        success_metrics = {
            "time_to_stability": time_to_stability,
            "phi_total": str(phi_total) if phi_total is not None else "",
            "delta_total": str(delta_total) if delta_total is not None else "",
            "n_defaults": str(n_defaults),
            "cascade_fraction": str(cascade_fraction_val)
            if cascade_fraction_val is not None
            else "",
        }
        self._upsert_registry(
            run_id=prepared.run_id,
            phase=prepared.phase,
            status=RunStatus.COMPLETED,
            parameters=success_params,
            metrics=success_metrics,
            artifact_paths={
                "scenario_yaml": self._rel_path(prepared.scenario_path),
                "events_jsonl": self._rel_path(events_path),
                "balances_csv": self._rel_path(balances_path),
                "metrics_csv": self._rel_path(output_paths["metrics_csv"]),
                "metrics_html": self._rel_path(output_paths["metrics_html"]),
                "run_html": self._rel_path(run_html_path),
            },
        )

        # Extract initial intermediary capitals from scenario config
        from bilancio.analysis.report import extract_initial_capitals
        capitals = extract_initial_capitals(prepared.scenario_config)

        return RingRunSummary(
            run_id=prepared.run_id,
            phase=prepared.phase,
            kappa=prepared.kappa,
            concentration=prepared.concentration,
            mu=prepared.mu,
            monotonicity=prepared.monotonicity,
            delta_total=delta_total,
            phi_total=phi_total,
            time_to_stability=time_to_stability,
            n_defaults=n_defaults,
            cascade_fraction=cascade_fraction_val,
            dealer_metrics=dealer_metrics,
            modal_call_id=result.modal_call_id,
            cb_loans_created_count=int(bundle.summary.get("cb_loans_created_count", 0)),
            cb_interest_total_paid=int(bundle.summary.get("cb_interest_total_paid", 0)),
            cb_loans_outstanding_pre_final=int(bundle.summary.get("cb_loans_outstanding_pre_final", 0)),
            bank_defaults_final=int(bundle.summary.get("bank_defaults_final", 0)),
            cb_reserve_destruction_pct=float(bundle.summary.get("cb_reserve_destruction_pct", 0.0)),
            delta_bank=bundle.summary.get("delta_bank"),
            deposit_loss_gross=int(bundle.summary.get("deposit_loss_gross", 0)),
            deposit_loss_pct=bundle.summary.get("deposit_loss_pct"),
            payable_default_loss=int(bundle.summary.get("payable_default_loss", 0)),
            total_loss=int(bundle.summary.get("total_loss", 0)),
            total_loss_pct=(
                int(bundle.summary.get("total_loss", 0)) / float(bundle.summary.get("S_total", 0))
                if float(bundle.summary.get("S_total", 0)) > 0 else None
            ),
            nbfi_loan_loss=int(bundle.summary.get("nbfi_loan_loss", 0)),
            bank_credit_loss=int(bundle.summary.get("bank_credit_loss", 0)),
            cb_backstop_loss=int(bundle.summary.get("cb_backstop_loss", 0)),
            dealer_vbt_loss=max(0.0, -float((dealer_metrics or {}).get("dealer_total_pnl", 0))),
            intermediary_loss_total=(
                max(0.0, -float((dealer_metrics or {}).get("dealer_total_pnl", 0)))
                + int(bundle.summary.get("nbfi_loan_loss", 0))
                + int(bundle.summary.get("bank_credit_loss", 0))
                + int(bundle.summary.get("cb_backstop_loss", 0))
            ),
            initial_intermediary_capital=capitals["intermediary_capital"],
            initial_dealer_vbt_cash=capitals["dealer_capital"] + capitals["vbt_capital"],
            initial_lender_cash=capitals["lender_capital"],
            initial_bank_reserves=capitals["bank_capital"],
        )

    def _rel_path(self, absolute: Path) -> str:
        try:
            return str(Path("..").joinpath(absolute.relative_to(self.base_dir)))
        except ValueError:
            return str(absolute)

    def _artifact_loader_for_result(self, result: ExecutionResult) -> Any:
        if result.storage_type == "modal_volume":
            return ModalVolumeArtifactLoader(base_path=result.storage_base)
        return LocalArtifactLoader(base_path=Path(result.storage_base))

    def _liquidity_allocation_dict(self) -> dict[str, Any]:
        allocation: dict[str, Any] = {"mode": self.liquidity_mode}
        if self.liquidity_mode == "single_at" and self.liquidity_agent:
            allocation["agent"] = self.liquidity_agent
        return allocation


__all__ = [
    "RingSweepRunner",
    "RingRunSummary",
    "PreparedRun",
    "RingSweepConfig",
    "load_ring_sweep_config",
    "_decimal_list",
]
