"""Balanced sweep command implementation."""

from __future__ import annotations

from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

import click

from bilancio.experiments.balanced_comparison import (
    BalancedComparisonConfig,
    BalancedComparisonRunner,
)
from bilancio.jobs import JobConfig, JobManager

from ._common import (
    CLI_HANDLED_ERRORS,
    as_decimal_list,
    build_performance_config,
    invoke_subcommand,
)


def _deps() -> Any:
    return import_module("bilancio.ui.cli.sweep")


@click.command("balanced")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Output directory for results")
@click.option("--n-agents", type=int, default=100, help="Number of agents in ring")
@click.option("--maturity-days", type=int, default=10, help="Maturity horizon")
@click.option("--max-simulation-days", type=int, default=None, help="Max simulation days (default: auto from maturity)")
@click.option("--q-total", type=Decimal, default=Decimal("10000"), help="Total debt")
@click.option("--base-seed", type=int, default=42, help="Base random seed")
@click.option(
    "--n-replicates",
    type=click.IntRange(min=1),
    default=1,
    help="Number of replicate seeds per parameter cell (default: 1)",
)
@click.option("--kappas", type=str, default="0.25,0.5,1,2,4", help="Comma-separated kappa values")
@click.option(
    "--concentrations",
    type=str,
    default="0.2,0.5,1,2,5",
    help="Comma-separated concentration values",
)
@click.option("--mus", type=str, default="0,0.25,0.5,0.75,1", help="Comma-separated mu values")
@click.option("--face-value", type=Decimal, default=Decimal("20"), help="Face value S")
@click.option(
    "--outside-mid-ratios",
    type=str,
    default="0.90",
    help="Comma-separated M/S ratios to sweep",
)
@click.option(
    "--pool-scales",
    type=str,
    default="0.375",
    help="Comma-separated pool scale values (Plan 053: dose-response sweep)",
)
@click.option(
    "--big-entity-share",
    type=Decimal,
    default=Decimal("0.25"),
    help="Fraction of debt held by big entities (beta)",
)
@click.option(
    "--default-handling",
    type=click.Choice(["fail-fast", "expel-agent"]),
    default="expel-agent",
    help="Default handling mode",
)
@click.option(
    "--detailed-logging/--no-detailed-logging",
    default=True,
    help="Enable detailed CSV logging",
)
@click.option("--cloud", is_flag=True, help="Run simulations on Modal cloud")
@click.option("--job-id", type=str, default=None, help="Job ID (auto-generated if not provided)")
@click.option("--quiet/--verbose", default=True, help="Suppress verbose console output during sweeps")
@click.option("--rollover/--no-rollover", default=True, help="Enable continuous rollover")
@click.option(
    "--risk-assessment/--no-risk-assessment",
    default=True,
    help="Enable risk-based trader decision making",
)
@click.option("--risk-premium", type=Decimal, default=Decimal("0.02"), help="Base risk premium")
@click.option("--risk-urgency", type=Decimal, default=Decimal("0.30"), help="Urgency sensitivity")
@click.option(
    "--alpha-vbt",
    type=Decimal,
    default=Decimal("0"),
    help="[DEPRECATED] VBT informedness; prefer --adapt=calibrated",
)
@click.option(
    "--alpha-trader",
    type=Decimal,
    default=Decimal("0"),
    help="[DEPRECATED] Trader informedness; prefer --adapt=calibrated",
)
@click.option("--risk-aversion", type=Decimal, default=Decimal("0"), help="Trader risk aversion")
@click.option("--planning-horizon", type=int, default=10, help="Trader planning horizon in days")
@click.option("--aggressiveness", type=Decimal, default=Decimal("1.0"), help="Buyer aggressiveness")
@click.option(
    "--default-observability",
    type=Decimal,
    default=Decimal("1.0"),
    help="Trader default observability",
)
@click.option(
    "--vbt-mid-sensitivity",
    type=Decimal,
    default=Decimal("1.0"),
    help="VBT mid price sensitivity to defaults",
)
@click.option(
    "--vbt-spread-sensitivity",
    type=Decimal,
    default=Decimal("0.0"),
    help="VBT spread sensitivity to defaults",
)
@click.option(
    "--trading-motive",
    type=click.Choice(["liquidity_only", "liquidity_then_earning", "unrestricted"]),
    default="liquidity_then_earning",
    help="Trading motivation",
)
@click.option(
    "--enable-lender/--no-lender",
    default=False,
    help="Enable third comparison arm with non-bank lender",
)
@click.option("--lender-share", type=Decimal, default=Decimal("0.10"), help="Lender capital share")
@click.option(
    "--enable-dealer-lender/--no-dealer-lender",
    default=False,
    help="Enable fourth arm with dealer trading and non-bank lending",
)
@click.option(
    "--enable-bank-passive/--no-bank-passive",
    default=False,
    help="[DEPRECATED: use 'bilancio sweep bank'] Enable arm: banks + passive dealer",
)
@click.option(
    "--enable-bank-dealer/--no-bank-dealer",
    default=False,
    help="[DEPRECATED: use 'bilancio sweep bank'] Enable arm: banks + active dealer",
)
@click.option(
    "--enable-bank-dealer-nbfi/--no-bank-dealer-nbfi",
    default=False,
    help="[DEPRECATED: use 'bilancio sweep bank'] Enable arm: banks + active dealer + NBFI",
)
@click.option("--n-banks-for-banking", type=int, default=3, help="Number of banks in banking arms")
@click.option(
    "--bank-reserve-multiplier",
    type=float,
    default=0.5,
    help="Reserve multiplier for banking arms",
)
@click.option(
    "--min-coverage-ratio",
    type=Decimal,
    default=Decimal("0"),
    help="Borrower assessment: min coverage ratio to approve loan",
)
@click.option(
    "--cb-lending-cutoff-day",
    type=int,
    default=None,
    help="Day to freeze CB lending (default: auto = maturity_days)",
)
@click.option("--n-banks", type=int, default=0, help="Number of banks to add (0 = no banks)")
@click.option(
    "--reserve-multiplier",
    type=float,
    default=10.0,
    help="Bank reserves = reserve_multiplier * face_value",
)
@click.option(
    "--lender-min-coverage",
    type=Decimal,
    default=Decimal("0.5"),
    help="NBFI min coverage ratio for borrower assessment",
)
@click.option(
    "--lender-maturity-matching/--no-lender-maturity-matching",
    default=False,
    help="Match NBFI loan maturity to borrower's next receivable",
)
@click.option(
    "--lender-min-loan-maturity",
    type=int,
    default=2,
    help="Floor for NBFI loan maturity when matching",
)
@click.option(
    "--lender-max-loans-per-borrower-per-day",
    type=int,
    default=0,
    help="Max NBFI loans per borrower per day, 0=unlimited",
)
@click.option(
    "--lender-ranking-mode",
    type=click.Choice(["profit", "cascade", "blended"]),
    default="profit",
    help="NBFI ranking mode",
)
@click.option(
    "--lender-cascade-weight",
    type=Decimal,
    default=Decimal("0.5"),
    help="Weight for cascade score in blended ranking",
)
@click.option(
    "--lender-coverage-mode",
    type=click.Choice(["gate", "graduated"]),
    default="gate",
    help="NBFI coverage gate mode",
)
@click.option(
    "--lender-coverage-penalty-scale",
    type=Decimal,
    default=Decimal("0.10"),
    help="Rate premium per unit below coverage threshold",
)
@click.option(
    "--lender-preventive-lending/--no-lender-preventive-lending",
    default=False,
    help="Enable NBFI proactive lending to at-risk agents",
)
@click.option(
    "--lender-prevention-threshold",
    type=Decimal,
    default=Decimal("0.3"),
    help="Min issuer default probability to trigger preventive lending",
)
@click.option(
    "--lender-marginal-relief-min-ratio",
    type=Decimal,
    default=Decimal("0"),
    help="Min expected relief/loss ratio for NBFI lending",
)
@click.option(
    "--lender-stress-risk-premium-scale",
    type=Decimal,
    default=Decimal("0"),
    help="Stress risk premium convex scale",
)
@click.option(
    "--lender-high-risk-default-threshold",
    type=Decimal,
    default=Decimal("0.70"),
    help="High-risk default probability threshold",
)
@click.option(
    "--lender-high-risk-maturity-cap",
    type=int,
    default=2,
    help="Max maturity for high-risk borrowers",
)
@click.option(
    "--lender-daily-el-budget-ratio",
    type=Decimal,
    default=Decimal("0"),
    help="Daily expected loss budget divided by capital",
)
@click.option(
    "--lender-run-el-budget-ratio",
    type=Decimal,
    default=Decimal("0"),
    help="Run expected loss budget divided by capital",
)
@click.option(
    "--lender-stop-loss-ratio",
    type=Decimal,
    default=Decimal("0"),
    help="Stop lending when realized losses divided by capital exceed this",
)
@click.option(
    "--lender-collateralized-terms/--no-lender-collateralized-terms",
    default=False,
    help="Cap NBFI loan principal by receivable collateral value",
)
@click.option(
    "--lender-collateral-advance-rate",
    type=Decimal,
    default=Decimal("1.0"),
    help="Advance rate for collateralized NBFI terms",
)
@click.option(
    "--trading-rounds",
    type=click.IntRange(min=1),
    default=100,
    help="Max trading sub-rounds per day",
)
@click.option(
    "--issuer-specific-pricing/--no-issuer-specific-pricing",
    default=False,
    help="Enable per-issuer risk pricing",
)
@click.option(
    "--flow-sensitivity",
    type=Decimal,
    default=Decimal("0.0"),
    help="VBT flow-aware ask widening",
)
@click.option(
    "--dealer-concentration-limit",
    type=Decimal,
    default=Decimal("0"),
    help="Max fraction of dealer inventory from single issuer",
)
@click.option(
    "--equalize-bank-capacity/--no-equalize-bank-capacity",
    default=True,
    help="Equalize bank reserves to match non-bank intermediary capital",
)
@click.option(
    "--post-analysis",
    type=str,
    default=None,
    help=(
        "Post-sweep analysis: 'all', 'none', or comma-separated list. "
        "Valid: drilldowns, deltas, dynamics, narrative, strategy_outcomes, "
        "dealer_usage, mechanism_activity, treynor. Default: interactive prompt."
    ),
)
@click.option(
    "--perf-preset",
    type=click.Choice(["compatible", "fast", "aggressive"]),
    default=None,
    help="Performance preset",
)
@click.option("--fast-atomic", is_flag=True, default=False, help="Disable deepcopy in safe phases")
@click.option("--preview-buy", is_flag=True, default=False, help="Preview-then-commit buy path")
@click.option("--cache-dealer-quotes", is_flag=True, default=False, help="Snapshot/restore dealer state")
@click.option("--dirty-bucket-recompute", is_flag=True, default=False, help="Only recompute traded buckets")
@click.option("--prune-ineligible", is_flag=True, default=False, help="Skip zero-resource agents")
@click.option("--incremental-intentions", is_flag=True, default=False, help="Incremental intention queues")
@click.option("--matching-order", type=click.Choice(["random", "urgency"]), default=None, help="Matching order")
@click.option("--dealer-backend", type=click.Choice(["python", "native"]), default=None, help="Kernel backend")
@click.option(
    "--adaptive-planning-horizon/--no-adaptive-planning-horizon",
    default=None,
    help="Override adaptive planning horizon flag",
)
@click.option(
    "--adaptive-risk-aversion/--no-adaptive-risk-aversion",
    default=None,
    help="Override adaptive risk aversion flag",
)
@click.option("--adaptive-reserves/--no-adaptive-reserves", default=None, help="Override adaptive reserves flag")
@click.option("--adaptive-lookback/--no-adaptive-lookback", default=None, help="Override adaptive lookback flag")
@click.option(
    "--adaptive-issuer-specific/--no-adaptive-issuer-specific",
    default=None,
    help="Override adaptive issuer-specific pricing flag",
)
@click.option(
    "--adaptive-ev-term-structure/--no-adaptive-ev-term-structure",
    default=None,
    help="Override adaptive EV term structure flag",
)
@click.option(
    "--adaptive-term-structure/--no-adaptive-term-structure",
    default=None,
    help="Override adaptive VBT term structure flag",
)
@click.option(
    "--adaptive-base-spreads/--no-adaptive-base-spreads",
    default=None,
    help="Override adaptive base spreads flag",
)
@click.option(
    "--adaptive-convex-spreads/--no-adaptive-convex-spreads",
    default=None,
    help="Override adaptive convex spreads flag",
)
@click.option(
    "--adapt",
    type=click.Choice(["static", "calibrated", "responsive", "full"]),
    default="static",
    help="Adaptive profile preset (Plan 050)",
)
@click.option(
    "--preset",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Load preset YAML to override defaults",
)
def sweep_balanced(
    out_dir: Path,
    n_agents: int,
    maturity_days: int,
    max_simulation_days: int | None,
    q_total: Decimal,
    base_seed: int,
    n_replicates: int,
    kappas: str,
    concentrations: str,
    mus: str,
    face_value: Decimal,
    outside_mid_ratios: str,
    pool_scales: str,
    big_entity_share: Decimal,
    default_handling: str,
    detailed_logging: bool,
    cloud: bool,
    job_id: str | None,
    quiet: bool,
    rollover: bool,
    risk_assessment: bool,
    risk_premium: Decimal,
    risk_urgency: Decimal,
    alpha_vbt: Decimal,
    alpha_trader: Decimal,
    risk_aversion: Decimal,
    planning_horizon: int,
    aggressiveness: Decimal,
    default_observability: Decimal,
    vbt_mid_sensitivity: Decimal,
    vbt_spread_sensitivity: Decimal,
    trading_motive: str,
    enable_lender: bool,
    lender_share: Decimal,
    enable_dealer_lender: bool,
    enable_bank_passive: bool,
    enable_bank_dealer: bool,
    enable_bank_dealer_nbfi: bool,
    n_banks_for_banking: int,
    bank_reserve_multiplier: float,
    min_coverage_ratio: Decimal,
    cb_lending_cutoff_day: int | None,
    n_banks: int,
    reserve_multiplier: float,
    lender_min_coverage: Decimal,
    lender_maturity_matching: bool,
    lender_min_loan_maturity: int,
    lender_max_loans_per_borrower_per_day: int,
    lender_ranking_mode: str,
    lender_cascade_weight: Decimal,
    lender_coverage_mode: str,
    lender_coverage_penalty_scale: Decimal,
    lender_preventive_lending: bool,
    lender_prevention_threshold: Decimal,
    lender_marginal_relief_min_ratio: Decimal,
    lender_stress_risk_premium_scale: Decimal,
    lender_high_risk_default_threshold: Decimal,
    lender_high_risk_maturity_cap: int,
    lender_daily_el_budget_ratio: Decimal,
    lender_run_el_budget_ratio: Decimal,
    lender_stop_loss_ratio: Decimal,
    lender_collateralized_terms: bool,
    lender_collateral_advance_rate: Decimal,
    trading_rounds: int,
    issuer_specific_pricing: bool,
    flow_sensitivity: Decimal,
    dealer_concentration_limit: Decimal,
    equalize_bank_capacity: bool,
    post_analysis: str | None,
    perf_preset: str | None,
    fast_atomic: bool,
    preview_buy: bool,
    cache_dealer_quotes: bool,
    dirty_bucket_recompute: bool,
    prune_ineligible: bool,
    incremental_intentions: bool,
    matching_order: str | None,
    dealer_backend: str | None,
    adaptive_planning_horizon: bool | None,
    adaptive_risk_aversion: bool | None,
    adaptive_reserves: bool | None,
    adaptive_lookback: bool | None,
    adaptive_issuer_specific: bool | None,
    adaptive_ev_term_structure: bool | None,
    adaptive_term_structure: bool | None,
    adaptive_base_spreads: bool | None,
    adaptive_convex_spreads: bool | None,
    adapt: str,
    preset: Path | None,
) -> None:
    """Run balanced C vs D comparison experiments."""
    deps = _deps()

    if preset is not None:
        from bilancio.ui.sweep_setup import SweepSetupResult, build_cli_args, load_preset

        preset_data = load_preset(preset)
        setup_result = SweepSetupResult(
            sweep_type="balanced",
            cloud=cloud,
            params=preset_data.get("params", {}),
            out_dir=out_dir,
            launch=True,
        )
        cli_args = build_cli_args(setup_result)
        if "--out-dir" not in cli_args:
            cli_args = ["--out-dir", str(out_dir)] + cli_args
        click.echo(f"Loaded preset from: {preset}")
        invoke_subcommand(deps.sweep, click.get_current_context(), "balanced", cli_args)
        return

    out_dir = Path(out_dir)
    if job_id is None:
        job_id = deps.generate_job_id()

    performance_flags: dict[str, object] = {}
    if perf_preset:
        performance_flags["preset"] = perf_preset
    if fast_atomic:
        performance_flags["fast_atomic"] = True
    if preview_buy:
        performance_flags["preview_buy"] = True
    if cache_dealer_quotes:
        performance_flags["cache_dealer_quotes"] = True
    if dirty_bucket_recompute:
        performance_flags["dirty_bucket_recompute"] = True
    if prune_ineligible:
        performance_flags["prune_ineligible"] = True
    if incremental_intentions:
        performance_flags["incremental_intentions"] = True
    if matching_order:
        performance_flags["matching_order"] = matching_order
    if dealer_backend:
        performance_flags["dealer_backend"] = dealer_backend

    manager: JobManager | None = None
    try:
        manager = deps.create_job_manager(jobs_dir=out_dir, cloud=cloud, local=True)
        job_config = JobConfig(
            sweep_type="balanced",
            n_agents=n_agents,
            kappas=as_decimal_list(kappas),
            concentrations=as_decimal_list(concentrations),
            mus=as_decimal_list(mus),
            cloud=cloud,
            outside_mid_ratios=as_decimal_list(outside_mid_ratios),
            maturity_days=maturity_days,
            seeds=[base_seed],
            performance=performance_flags,
        )
        job = manager.create_job(
            description=f"Balanced comparison sweep (n={n_agents}, cloud={cloud})",
            config=job_config,
            job_id=job_id,
        )
        click.echo(f"Job ID: {job.job_id}")
        manager.start_job(job.job_id)
    except CLI_HANDLED_ERRORS as exc:
        click.echo(f"Warning: Job tracking initialization failed: {exc}")
        manager = None

    executor = None
    if cloud:
        from bilancio.runners import CloudExecutor

        executor = CloudExecutor(
            experiment_id=job_id,
            download_artifacts=False,
            local_output_dir=out_dir,
            job_id=job_id,
        )
        click.echo("Cloud execution enabled")

    if risk_assessment:
        click.echo(f"Risk assessment enabled (premium={risk_premium}, urgency={risk_urgency})")

    risk_config = {
        "base_risk_premium": str(risk_premium),
        "urgency_sensitivity": str(risk_urgency),
        "buy_premium_multiplier": "1.0",
        "lookback_window": 5,
    }
    if alpha_vbt > 0 or alpha_trader > 0:
        click.echo(f"Informedness enabled (alpha_vbt={alpha_vbt}, alpha_trader={alpha_trader})")

    performance = build_performance_config(performance_flags)
    adaptive_flags = {
        "adaptive_planning_horizon": adaptive_planning_horizon,
        "adaptive_risk_aversion": adaptive_risk_aversion,
        "adaptive_reserves": adaptive_reserves,
        "adaptive_lookback": adaptive_lookback,
        "adaptive_issuer_specific": adaptive_issuer_specific,
        "adaptive_ev_term_structure": adaptive_ev_term_structure,
        "adaptive_term_structure": adaptive_term_structure,
        "adaptive_base_spreads": adaptive_base_spreads,
        "adaptive_convex_spreads": adaptive_convex_spreads,
    }
    adaptive_overrides = {key: value for key, value in adaptive_flags.items() if value is not None}
    if adaptive_overrides:
        click.echo(f"Adaptive flag overrides: {adaptive_overrides}")

    config_kwargs: dict = {}
    if max_simulation_days is not None:
        config_kwargs["max_simulation_days"] = max_simulation_days

    config = BalancedComparisonConfig(
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=q_total,
        **config_kwargs,
        base_seed=base_seed,
        n_replicates=n_replicates,
        kappas=as_decimal_list(kappas),
        concentrations=as_decimal_list(concentrations),
        mus=as_decimal_list(mus),
        face_value=face_value,
        outside_mid_ratios=as_decimal_list(outside_mid_ratios),
        pool_scales=as_decimal_list(pool_scales),
        big_entity_share=big_entity_share,
        default_handling=default_handling,
        detailed_logging=detailed_logging,
        quiet=quiet,
        rollover_enabled=rollover,
        risk_assessment_enabled=risk_assessment,
        risk_assessment_config=risk_config,
        alpha_vbt=alpha_vbt,
        alpha_trader=alpha_trader,
        risk_aversion=risk_aversion,
        planning_horizon=planning_horizon,
        aggressiveness=aggressiveness,
        default_observability=default_observability,
        vbt_mid_sensitivity=vbt_mid_sensitivity,
        vbt_spread_sensitivity=vbt_spread_sensitivity,
        trading_motive=trading_motive,
        enable_lender=enable_lender,
        lender_share=lender_share,
        enable_dealer_lender=enable_dealer_lender,
        enable_bank_passive=enable_bank_passive,
        enable_bank_dealer=enable_bank_dealer,
        enable_bank_dealer_nbfi=enable_bank_dealer_nbfi,
        n_banks_for_banking=n_banks_for_banking,
        bank_reserve_multiplier=bank_reserve_multiplier,
        equalize_bank_capacity=equalize_bank_capacity,
        min_coverage_ratio=min_coverage_ratio,
        lender_min_coverage=lender_min_coverage,
        lender_maturity_matching=lender_maturity_matching,
        lender_min_loan_maturity=lender_min_loan_maturity,
        lender_max_loans_per_borrower_per_day=lender_max_loans_per_borrower_per_day,
        lender_ranking_mode=lender_ranking_mode,
        lender_cascade_weight=lender_cascade_weight,
        lender_coverage_mode=lender_coverage_mode,
        lender_coverage_penalty_scale=lender_coverage_penalty_scale,
        lender_preventive_lending=lender_preventive_lending,
        lender_prevention_threshold=lender_prevention_threshold,
        lender_marginal_relief_min_ratio=lender_marginal_relief_min_ratio,
        lender_stress_risk_premium_scale=lender_stress_risk_premium_scale,
        lender_high_risk_default_threshold=lender_high_risk_default_threshold,
        lender_high_risk_maturity_cap=lender_high_risk_maturity_cap,
        lender_daily_expected_loss_budget_ratio=lender_daily_el_budget_ratio,
        lender_run_expected_loss_budget_ratio=lender_run_el_budget_ratio,
        lender_stop_loss_realized_ratio=lender_stop_loss_ratio,
        lender_collateralized_terms=lender_collateralized_terms,
        lender_collateral_advance_rate=lender_collateral_advance_rate,
        cb_lending_cutoff_day=cb_lending_cutoff_day,
        n_banks=n_banks,
        reserve_multiplier=reserve_multiplier,
        trading_rounds=trading_rounds,
        issuer_specific_pricing=issuer_specific_pricing,
        flow_sensitivity=flow_sensitivity,
        dealer_concentration_limit=dealer_concentration_limit,
        adapt=adapt,
        adaptive_overrides=adaptive_overrides,
        performance=performance.to_dict() if performance else {},
    )

    runner = BalancedComparisonRunner(config, out_dir, executor=executor, job_id=job_id)

    preflight: Any | None = None
    try:
        from bilancio.scenarios.sweep_diagnostics import run_preflight_checks

        preflight = run_preflight_checks(config)
        preflight.print_summary()
    except CLI_HANDLED_ERRORS as exc:
        click.echo(f"Warning: Pre-flight checks failed: {exc}")

    try:
        results = runner.run_all()

        if manager is not None:
            try:
                for result in results:
                    if result.passive_run_id:
                        manager.record_progress(job_id, result.passive_run_id, modal_call_id=result.passive_modal_call_id)
                    if result.active_run_id:
                        manager.record_progress(job_id, result.active_run_id, modal_call_id=result.active_modal_call_id)
            except CLI_HANDLED_ERRORS as exc:
                click.echo(f"Warning: Failed to record run progress: {exc}")

        completed = sum(1 for result in results if result.trading_effect is not None)
        improved = sum(1 for result in results if result.trading_effect and result.trading_effect > 0)

        arm_summary: dict[str, dict[str, int]] = {}
        if enable_lender:
            ok = sum(1 for result in results if result.delta_lender is not None)
            arm_summary["lender"] = {"completed": ok, "failed": len(results) - ok}
        if enable_dealer_lender:
            ok = sum(1 for result in results if result.delta_dealer_lender is not None)
            arm_summary["dealer+lender"] = {"completed": ok, "failed": len(results) - ok}
        if enable_bank_passive:
            ok = sum(1 for result in results if result.delta_bank_passive is not None)
            arm_summary["bank+passive"] = {"completed": ok, "failed": len(results) - ok}
        if enable_bank_dealer:
            ok = sum(1 for result in results if result.delta_bank_dealer is not None)
            arm_summary["bank+dealer"] = {"completed": ok, "failed": len(results) - ok}
        if enable_bank_dealer_nbfi:
            ok = sum(1 for result in results if result.delta_bank_dealer_nbfi is not None)
            arm_summary["bank+dealer+nbfi"] = {"completed": ok, "failed": len(results) - ok}

        if manager is not None:
            try:
                manager.complete_job(
                    job_id,
                    {
                        "total_pairs": len(results),
                        "completed": completed,
                        "improved_with_trading": improved,
                        "arm_summary": arm_summary,
                    },
                )
            except CLI_HANDLED_ERRORS as exc:
                click.echo(f"Warning: Failed to complete job tracking: {exc}")

        click.echo("\nBalanced comparison complete!")
        click.echo(f"  Total pairs: {len(results)}")
        click.echo(f"  Completed (passive vs active): {completed}")
        click.echo(f"  Improved with trading: {improved}")
        for arm_name, counts in arm_summary.items():
            status = "OK" if counts["failed"] == 0 else f"{counts['failed']} FAILED"
            click.echo(f"  {arm_name}: {counts['completed']}/{len(results)} ({status})")
        click.echo(f"\nResults at: {out_dir / 'aggregate' / 'comparison.csv'}")

        if preflight is not None and results:
            try:
                from bilancio.scenarios.sweep_diagnostics import run_postsweep_validation

                postflight = run_postsweep_validation(preflight, results, aggregate_dir=out_dir / "aggregate")
                postflight.print_summary()
            except CLI_HANDLED_ERRORS as exc:
                click.echo(f"Warning: Post-sweep validation failed: {exc}")

        deps._offer_post_sweep_analysis(out_dir, "dealer", post_analysis, cloud=cloud)
    except CLI_HANDLED_ERRORS as exc:
        if manager is not None:
            try:
                manager.fail_job(job_id, str(exc))
            except CLI_HANDLED_ERRORS:
                pass
        raise
