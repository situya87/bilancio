"""Utilities for running dealer comparison experiments.

This module provides infrastructure for running paired control/treatment
experiments comparing Kalecki ring simulations with and without dealer-mediated
secondary markets.

The key output is the delta in defaults between conditions, measuring
the effect of the dealer on settlement rates.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from bilancio.experiments.ring import RingSweepRunner, RingRunSummary, _decimal_list

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a single control/treatment comparison."""

    kappa: Decimal
    concentration: Decimal
    mu: Decimal
    monotonicity: Decimal
    seed: int

    # Control metrics (no dealer)
    delta_control: Optional[Decimal]
    phi_control: Optional[Decimal]
    control_run_id: str
    control_status: str

    # Treatment metrics (with dealer)
    delta_treatment: Optional[Decimal]
    phi_treatment: Optional[Decimal]
    treatment_run_id: str
    treatment_status: str

    # Dealer metrics (Section 8 of functional dealer specification)
    # From treatment run only (dealer_metrics dictionary)
    dealer_total_pnl: Optional[float] = None
    dealer_total_return: Optional[float] = None
    dealer_profitable: Optional[bool] = None
    spread_income_total: Optional[float] = None
    mean_trader_return: Optional[float] = None
    fraction_profitable_traders: Optional[float] = None
    liquidity_driven_sales: Optional[int] = None
    rescue_events: Optional[int] = None
    total_trades: Optional[int] = None
    unsafe_buy_count: Optional[int] = None
    fraction_unsafe_buys: Optional[float] = None

    # Plan 020 metrics: debt-to-money ratio and mid prices
    initial_total_debt: Optional[float] = None
    initial_total_money: Optional[float] = None
    debt_to_money_ratio: Optional[float] = None
    dealer_mid_final: Optional[Dict[str, float]] = None
    vbt_mid_final: Optional[Dict[str, float]] = None
    dealer_premium_final_pct: Optional[Dict[str, float]] = None
    vbt_premium_final_pct: Optional[Dict[str, float]] = None

    @property
    def delta_reduction(self) -> Optional[Decimal]:
        """Absolute reduction in default rate."""
        if self.delta_control is None or self.delta_treatment is None:
            return None
        return self.delta_control - self.delta_treatment

    @property
    def relief_ratio(self) -> Optional[Decimal]:
        """Percentage reduction in defaults (0 if no baseline defaults)."""
        if self.delta_control is None or self.delta_treatment is None:
            return None
        if self.delta_control == 0:
            return Decimal("0")  # No defaults to reduce
        reduction = self.delta_reduction
        assert reduction is not None  # guaranteed by checks above
        return reduction / self.delta_control


class ComparisonSweepConfig(BaseModel):
    """Configuration for dealer comparison experiments."""

    # Ring parameters
    n_agents: int = Field(default=100, description="Number of agents in ring")
    maturity_days: int = Field(default=10, description="Maturity horizon in days")
    max_simulation_days: int = Field(default=15, description="Max days to run simulation")
    Q_total: Decimal = Field(default=Decimal("10000"), description="Total debt amount")
    liquidity_mode: str = Field(default="uniform", description="Liquidity allocation mode")
    liquidity_agent: Optional[str] = Field(default=None, description="Target agent for single_at mode")
    base_seed: int = Field(default=42, description="Base random seed")
    name_prefix: str = Field(default="Dealer Comparison", description="Scenario name prefix")
    default_handling: str = Field(default="fail-fast", description="Default handling mode")

    # Grid parameters
    kappas: List[Decimal] = Field(default_factory=lambda: [Decimal("0.25"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("4")])
    concentrations: List[Decimal] = Field(default_factory=lambda: [Decimal("0.2"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("5")])
    mus: List[Decimal] = Field(default_factory=lambda: [Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1")])
    monotonicities: List[Decimal] = Field(default_factory=lambda: [Decimal("0")])

    # Dealer configuration for treatment runs
    # Per specification: Dealer and VBT bring NEW outside money (not taken from traders)
    # Traders keep 100% of their receivables. Dealer/VBT start empty and buy from traders.
    dealer_ticket_size: Decimal = Field(default=Decimal("1"), description="Ticket size for dealer")
    dealer_share: Decimal = Field(default=Decimal("0.25"), description="Dealer capital as fraction of system cash (NEW outside money)")
    vbt_share: Decimal = Field(default=Decimal("0.50"), description="VBT capital as fraction of system cash (NEW outside money)")


class ComparisonSweepRunner:
    """
    Runs paired control/treatment experiments for dealer comparison.

    For each parameter combination (κ, c, μ):
    1. Run WITHOUT dealer -> control metrics
    2. Run WITH dealer -> treatment metrics
    3. Compute and log comparison metrics

    Outputs:
    - control/registry/experiments.csv: All control runs
    - treatment/registry/experiments.csv: All treatment runs
    - aggregate/comparison.csv: Paired comparison metrics
    - aggregate/summary.json: Aggregate statistics
    """

    COMPARISON_FIELDS = [
        "kappa",
        "concentration",
        "mu",
        "monotonicity",
        "seed",
        "delta_control",
        "delta_treatment",
        "delta_reduction",
        "relief_ratio",
        "phi_control",
        "phi_treatment",
        "control_run_id",
        "control_status",
        "treatment_run_id",
        "treatment_status",
        # Dealer metrics (Section 8)
        "dealer_total_pnl",
        "dealer_total_return",
        "dealer_profitable",
        "spread_income_total",
        "mean_trader_return",
        "fraction_profitable_traders",
        "liquidity_driven_sales",
        "rescue_events",
        "total_trades",
        "unsafe_buy_count",
        "fraction_unsafe_buys",
        # Plan 020 metrics
        "initial_total_debt",
        "initial_total_money",
        "debt_to_money_ratio",
        "dealer_mid_final",
        "vbt_mid_final",
        "dealer_premium_final_pct",
        "vbt_premium_final_pct",
    ]

    def __init__(self, config: ComparisonSweepConfig, out_dir: Path) -> None:
        self.config = config
        self.base_dir = out_dir
        self.control_dir = self.base_dir / "control"
        self.treatment_dir = self.base_dir / "treatment"
        self.aggregate_dir = self.base_dir / "aggregate"

        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.treatment_dir.mkdir(parents=True, exist_ok=True)
        self.aggregate_dir.mkdir(parents=True, exist_ok=True)

        self.comparison_results: List[ComparisonResult] = []
        self.comparison_path = self.aggregate_dir / "comparison.csv"
        self.summary_path = self.aggregate_dir / "summary.json"

        # Build dealer config dict for treatment runs
        self.dealer_config = {
            "ticket_size": int(config.dealer_ticket_size),
            "dealer_share": str(config.dealer_share),
            "vbt_share": str(config.vbt_share),
        }

        # Initialize runners
        self._control_runner: Optional[RingSweepRunner] = None
        self._treatment_runner: Optional[RingSweepRunner] = None

    def _get_control_runner(self) -> RingSweepRunner:
        """Get or create control runner (no dealer)."""
        if self._control_runner is None:
            self._control_runner = RingSweepRunner(
                out_dir=self.control_dir,
                name_prefix=f"{self.config.name_prefix} (Control)",
                n_agents=self.config.n_agents,
                maturity_days=self.config.maturity_days,
                Q_total=self.config.Q_total,
                liquidity_mode=self.config.liquidity_mode,
                liquidity_agent=self.config.liquidity_agent,
                base_seed=self.config.base_seed,
                default_handling=self.config.default_handling,
                dealer_enabled=False,
                dealer_config=None,
            )
        return self._control_runner

    def _get_treatment_runner(self) -> RingSweepRunner:
        """Get or create treatment runner (with dealer)."""
        if self._treatment_runner is None:
            self._treatment_runner = RingSweepRunner(
                out_dir=self.treatment_dir,
                name_prefix=f"{self.config.name_prefix} (Treatment)",
                n_agents=self.config.n_agents,
                maturity_days=self.config.maturity_days,
                Q_total=self.config.Q_total,
                liquidity_mode=self.config.liquidity_mode,
                liquidity_agent=self.config.liquidity_agent,
                base_seed=self.config.base_seed,
                default_handling=self.config.default_handling,
                dealer_enabled=True,
                dealer_config=self.dealer_config,
            )
        return self._treatment_runner

    def run_all(self) -> List[ComparisonResult]:
        """Execute all control/treatment pairs and return comparison results."""
        logger.info(
            "Starting comparison sweep: %d kappas × %d concentrations × %d mus = %d pairs",
            len(self.config.kappas),
            len(self.config.concentrations),
            len(self.config.mus),
            len(self.config.kappas) * len(self.config.concentrations) * len(self.config.mus),
        )

        total_pairs = (
            len(self.config.kappas)
            * len(self.config.concentrations)
            * len(self.config.mus)
            * len(self.config.monotonicities)
        )
        pair_idx = 0

        for kappa in self.config.kappas:
            for concentration in self.config.concentrations:
                for mu in self.config.mus:
                    for monotonicity in self.config.monotonicities:
                        pair_idx += 1
                        logger.info(
                            "[%d/%d] Running pair: κ=%s, c=%s, μ=%s",
                            pair_idx,
                            total_pairs,
                            kappa,
                            concentration,
                            mu,
                        )
                        result = self._run_pair(kappa, concentration, mu, monotonicity)
                        self.comparison_results.append(result)

                        # Write incremental results
                        self._write_comparison_csv()

        # Write final summary
        self._write_summary_json()

        logger.info("Comparison sweep complete. Results at: %s", self.aggregate_dir)
        return self.comparison_results

    def _run_pair(
        self,
        kappa: Decimal,
        concentration: Decimal,
        mu: Decimal,
        monotonicity: Decimal,
    ) -> ComparisonResult:
        """Run one control/treatment pair for given parameters."""
        control_runner = self._get_control_runner()
        treatment_runner = self._get_treatment_runner()

        # Use same seed for both runs to ensure identical initial conditions
        seed = control_runner._next_seed()
        # Sync treatment runner seed
        treatment_runner.seed_counter = seed + 1

        # Run control (no dealer)
        logger.info("  Running control (no dealer)...")
        control_result = control_runner._execute_run(
            phase="comparison_control",
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
        )

        # Run treatment (with dealer) - use same seed
        logger.info("  Running treatment (with dealer)...")
        treatment_result = treatment_runner._execute_run(
            phase="comparison_treatment",
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
        )

        # Extract dealer metrics from treatment result
        dm = treatment_result.dealer_metrics or {}

        # Build comparison result
        result = ComparisonResult(
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            seed=seed,
            delta_control=control_result.delta_total,
            phi_control=control_result.phi_total,
            control_run_id=control_result.run_id,
            control_status="completed" if control_result.delta_total is not None else "failed",
            delta_treatment=treatment_result.delta_total,
            phi_treatment=treatment_result.phi_total,
            treatment_run_id=treatment_result.run_id,
            treatment_status="completed" if treatment_result.delta_total is not None else "failed",
            # Dealer metrics from treatment run
            dealer_total_pnl=dm.get("dealer_total_pnl"),
            dealer_total_return=dm.get("dealer_total_return"),
            dealer_profitable=dm.get("dealer_profitable"),
            spread_income_total=dm.get("spread_income_total"),
            mean_trader_return=dm.get("mean_trader_return"),
            fraction_profitable_traders=dm.get("fraction_profitable_traders"),
            liquidity_driven_sales=dm.get("liquidity_driven_sales"),
            rescue_events=dm.get("rescue_events"),
            total_trades=dm.get("total_trades"),
            unsafe_buy_count=dm.get("unsafe_buy_count"),
            fraction_unsafe_buys=dm.get("fraction_unsafe_buys"),
            # Plan 020 metrics
            initial_total_debt=dm.get("initial_total_debt"),
            initial_total_money=dm.get("initial_total_money"),
            debt_to_money_ratio=dm.get("debt_to_money_ratio"),
            dealer_mid_final=dm.get("dealer_mid_final"),
            vbt_mid_final=dm.get("vbt_mid_final"),
            dealer_premium_final_pct=dm.get("dealer_premium_final_pct"),
            vbt_premium_final_pct=dm.get("vbt_premium_final_pct"),
        )

        # Log comparison
        if result.delta_reduction is not None:
            logger.info(
                "  Comparison: δ_control=%s, δ_treatment=%s, reduction=%s (%.1f%%)",
                result.delta_control,
                result.delta_treatment,
                result.delta_reduction,
                float(result.relief_ratio or 0) * 100,
            )
        else:
            logger.warning("  Comparison: One or both runs failed, cannot compute reduction")

        return result

    def _write_comparison_csv(self) -> None:
        """Write comparison results to CSV."""
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
                    "delta_control": str(result.delta_control) if result.delta_control is not None else "",
                    "delta_treatment": str(result.delta_treatment) if result.delta_treatment is not None else "",
                    "delta_reduction": str(result.delta_reduction) if result.delta_reduction is not None else "",
                    "relief_ratio": str(result.relief_ratio) if result.relief_ratio is not None else "",
                    "phi_control": str(result.phi_control) if result.phi_control is not None else "",
                    "phi_treatment": str(result.phi_treatment) if result.phi_treatment is not None else "",
                    "control_run_id": result.control_run_id,
                    "control_status": result.control_status,
                    "treatment_run_id": result.treatment_run_id,
                    "treatment_status": result.treatment_status,
                    # Dealer metrics (Section 8)
                    "dealer_total_pnl": str(result.dealer_total_pnl) if result.dealer_total_pnl is not None else "",
                    "dealer_total_return": str(result.dealer_total_return) if result.dealer_total_return is not None else "",
                    "dealer_profitable": str(result.dealer_profitable) if result.dealer_profitable is not None else "",
                    "spread_income_total": str(result.spread_income_total) if result.spread_income_total is not None else "",
                    "mean_trader_return": str(result.mean_trader_return) if result.mean_trader_return is not None else "",
                    "fraction_profitable_traders": str(result.fraction_profitable_traders) if result.fraction_profitable_traders is not None else "",
                    "liquidity_driven_sales": str(result.liquidity_driven_sales) if result.liquidity_driven_sales is not None else "",
                    "rescue_events": str(result.rescue_events) if result.rescue_events is not None else "",
                    "total_trades": str(result.total_trades) if result.total_trades is not None else "",
                    "unsafe_buy_count": str(result.unsafe_buy_count) if result.unsafe_buy_count is not None else "",
                    "fraction_unsafe_buys": str(result.fraction_unsafe_buys) if result.fraction_unsafe_buys is not None else "",
                    # Plan 020 metrics
                    "initial_total_debt": str(result.initial_total_debt) if result.initial_total_debt is not None else "",
                    "initial_total_money": str(result.initial_total_money) if result.initial_total_money is not None else "",
                    "debt_to_money_ratio": str(result.debt_to_money_ratio) if result.debt_to_money_ratio is not None else "",
                    "dealer_mid_final": json.dumps(result.dealer_mid_final) if result.dealer_mid_final is not None else "",
                    "vbt_mid_final": json.dumps(result.vbt_mid_final) if result.vbt_mid_final is not None else "",
                    "dealer_premium_final_pct": json.dumps(result.dealer_premium_final_pct) if result.dealer_premium_final_pct is not None else "",
                    "vbt_premium_final_pct": json.dumps(result.vbt_premium_final_pct) if result.vbt_premium_final_pct is not None else "",
                }
                writer.writerow(row)

    def _write_summary_json(self) -> None:
        """Write summary statistics to JSON."""
        completed = [r for r in self.comparison_results if r.delta_reduction is not None]
        failed_control = [r for r in self.comparison_results if r.control_status == "failed"]
        failed_treatment = [r for r in self.comparison_results if r.treatment_status == "failed"]

        # Compute statistics
        if completed:
            delta_controls = [float(r.delta_control) for r in completed if r.delta_control is not None]
            delta_treatments = [float(r.delta_treatment) for r in completed if r.delta_treatment is not None]
            relief_ratios = [float(r.relief_ratio) for r in completed if r.relief_ratio is not None]

            mean_delta_control = sum(delta_controls) / len(delta_controls) if delta_controls else None
            mean_delta_treatment = sum(delta_treatments) / len(delta_treatments) if delta_treatments else None
            mean_relief_ratio = sum(relief_ratios) / len(relief_ratios) if relief_ratios else None

            improved = sum(1 for r in completed if r.delta_reduction and r.delta_reduction > 0)
            unchanged = sum(1 for r in completed if r.delta_reduction == 0)
            worsened = sum(1 for r in completed if r.delta_reduction and r.delta_reduction < 0)
        else:
            mean_delta_control = None
            mean_delta_treatment = None
            mean_relief_ratio = None
            improved = 0
            unchanged = 0
            worsened = 0

        summary = {
            "total_pairs": len(self.comparison_results),
            "completed_pairs": len(completed),
            "failed_control_runs": len(failed_control),
            "failed_treatment_runs": len(failed_treatment),
            "mean_delta_control": mean_delta_control,
            "mean_delta_treatment": mean_delta_treatment,
            "mean_relief_ratio": mean_relief_ratio,
            "pairs_with_improvement": improved,
            "pairs_unchanged": unchanged,
            "pairs_worsened": worsened,
            "config": {
                "n_agents": self.config.n_agents,
                "maturity_days": self.config.maturity_days,
                "Q_total": str(self.config.Q_total),
                "base_seed": self.config.base_seed,
                "kappas": [str(k) for k in self.config.kappas],
                "concentrations": [str(c) for c in self.config.concentrations],
                "mus": [str(m) for m in self.config.mus],
            },
        }

        with self.summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2)


def run_comparison_sweep(
    out_dir: Path,
    *,
    n_agents: int = 100,
    maturity_days: int = 10,
    Q_total: Decimal = Decimal("10000"),
    kappas: Sequence[Decimal],
    concentrations: Sequence[Decimal],
    mus: Sequence[Decimal],
    monotonicities: Optional[Sequence[Decimal]] = None,
    base_seed: int = 42,
    default_handling: str = "fail-fast",
    dealer_ticket_size: Decimal = Decimal("1"),
    dealer_share: Decimal = Decimal("0.25"),
    vbt_share: Decimal = Decimal("0.50"),
    liquidity_mode: str = "uniform",
    liquidity_agent: Optional[str] = None,
    name_prefix: str = "Dealer Comparison",
) -> List[ComparisonResult]:
    """
    Convenience function to run a comparison sweep.

    Args:
        out_dir: Output directory for results
        n_agents: Number of agents in ring (default: 100)
        maturity_days: Maturity horizon (default: 10)
        Q_total: Total debt amount (default: 10000)
        kappas: List of kappa values to sweep
        concentrations: List of Dirichlet concentration values
        mus: List of mu (misalignment) values
        monotonicities: List of monotonicity values (default: [0])
        base_seed: Base random seed
        default_handling: How to handle defaults
        dealer_ticket_size: Ticket size for dealer
        dealer_share: Dealer capital as fraction of system cash (NEW outside money)
        vbt_share: VBT capital as fraction of system cash (NEW outside money)
        liquidity_mode: Liquidity allocation mode
        liquidity_agent: Target agent for single_at mode
        name_prefix: Scenario name prefix

    Returns:
        List of ComparisonResult objects
    """
    config = ComparisonSweepConfig(
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=Q_total,
        kappas=list(kappas),
        concentrations=list(concentrations),
        mus=list(mus),
        monotonicities=list(monotonicities or [Decimal("0")]),
        base_seed=base_seed,
        default_handling=default_handling,
        dealer_ticket_size=dealer_ticket_size,
        dealer_share=dealer_share,
        vbt_share=vbt_share,
        liquidity_mode=liquidity_mode,
        liquidity_agent=liquidity_agent,
        name_prefix=name_prefix,
    )

    runner = ComparisonSweepRunner(config, out_dir)
    return runner.run_all()


__all__ = [
    "ComparisonResult",
    "ComparisonSweepConfig",
    "ComparisonSweepRunner",
    "run_comparison_sweep",
]
