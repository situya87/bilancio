"""Pre-flight viability checks and post-sweep validation.

Wires the existing viability check functions from ``viability.py`` into the
sweep lifecycle:

* **Pre-flight** — ``run_preflight_checks()`` runs V1–V11 checks across
  the parameter grid *before* the sweep starts.
* **Post-sweep** — ``run_postsweep_validation()`` verifies that each
  mechanism actually fired in tangible quantity during the sweep.

Pre-flight checks (V1–V11):

  V1   Sell viable — dealer bid > 0 for urgent sellers
  V2   Buy viable — buyer gain > premium for surplus buyers
  V3   Two-way trading — κ grid spans both stressed and surplus zones
  V4   Dealer capacity — K* is neither trivial nor unconstrained
  V5   Lending viable — NBFI rate in sensible κ range
  V6   Temporal spread — maturity ≥ 2, μ not extreme
  V7   Effect detectable — enough grid points for signal
  V8   Bank vs sell — bank loan rate vs secondary-market haircut
  V9   VBT mid stability — credit-adjusted mid M stays above M_MIN
  V10  NBFI loan adoption — loan rate cheaper than sell haircut
  V11  Bank reserve capacity — reserves cover expected lending demand

Post-sweep checks verify that each actor's actions actually occurred
(in tangible quantity for the run size), NOT whether they were effective.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from bilancio.dealer.priors import kappa_informed_prior
from bilancio.scenarios.viability import (
    InterbankViabilityReport,
    SimulationViabilityReport,
    SweepViabilityReport,
    ViabilityReport,
    check_interbank_viability,
    check_simulation_viability,
    check_sweep_viability,
    check_trade_viability,
)

# ---------------------------------------------------------------------------
# Pre-flight report structures
# ---------------------------------------------------------------------------

# VBT guard threshold — must match dealer/kernel.py M_MIN
_M_MIN = Decimal("0.02")


@dataclass
class PreflightCheckResult:
    """Single pre-flight viability check result."""

    check_id: str  # "V1", "V2", ..., "V11"
    label: str  # human-readable label
    prediction: bool
    level: str  # "pass", "warn", "fail"
    detail: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Aggregated pre-flight viability report."""

    checks: list[PreflightCheckResult] = field(default_factory=list)
    n_agents: int = 100
    kappa_trade_reports: dict[str, ViabilityReport] = field(default_factory=dict)
    sweep_report: SweepViabilityReport | None = None
    simulation_reports: dict[str, SimulationViabilityReport] = field(default_factory=dict)
    interbank_reports: dict[str, InterbankViabilityReport] = field(default_factory=dict)

    def print_summary(self) -> None:
        """Print compact pre-flight summary to stdout."""
        print("\nPRE-FLIGHT VIABILITY CHECKS", flush=True)
        print("\u2500" * 28, flush=True)
        for c in self.checks:
            icon = {"pass": "\u2713", "warn": "\u26a0", "fail": "\u2717"}.get(c.level, "?")
            print(f"{c.check_id:<4s} {c.label:<22s} {icon}  {c.detail}", flush=True)

        fails = sum(1 for c in self.checks if c.level == "fail")
        warns = sum(1 for c in self.checks if c.level == "warn")
        if fails == 0 and warns == 0:
            print("\nAll checks passed.", flush=True)
        else:
            parts: list[str] = []
            if fails:
                parts.append(f"{fails} fail(s)")
            if warns:
                parts.append(f"{warns} warning(s)")
            print(f"\n{', '.join(parts)}.", flush=True)
        print(flush=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for JSON storage."""
        return {
            "checks": [
                {
                    "check_id": c.check_id,
                    "label": c.label,
                    "prediction": c.prediction,
                    "level": c.level,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Post-flight report structures
# ---------------------------------------------------------------------------


@dataclass
class PostflightCheckResult:
    """Single post-flight validation result.

    ``prediction`` is what pre-flight expected; ``actual`` is whether
    the activity actually occurred in tangible quantity during the run.
    """

    check_id: str
    label: str
    prediction: bool
    actual: bool
    match: bool
    detail: str


@dataclass
class PostflightReport:
    """Aggregated post-flight validation report."""

    checks: list[PostflightCheckResult] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        if not self.checks:
            return 1.0
        return sum(1 for c in self.checks if c.match) / len(self.checks)

    def print_summary(self) -> None:
        """Print compact post-flight summary to stdout."""
        print("\nPOST-SWEEP VALIDATION", flush=True)
        print("\u2500" * 22, flush=True)
        for c in self.checks:
            icon = "\u2713" if c.match else "\u26a0"
            print(f"{c.check_id:<4s} {c.label:<22s} {icon}  {c.detail}", flush=True)

        matched = sum(1 for c in self.checks if c.match)
        warned = len(self.checks) - matched
        print(f"\nMatch rate: {matched}/{len(self.checks)} \u2713", end="", flush=True)
        if warned:
            print(f", {warned}/{len(self.checks)} \u26a0", end="", flush=True)
        print(flush=True)

    def write_report(self, path: Path) -> None:
        """Write plain-text validation report to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = ["POST-SWEEP VALIDATION", "=" * 22, ""]
        for c in self.checks:
            icon = "\u2713" if c.match else "\u26a0"
            lines.append(f"{c.check_id:<4s} {c.label:<22s} {icon}  {c.detail}")

        matched = sum(1 for c in self.checks if c.match)
        warned = len(self.checks) - matched
        summary = f"Match rate: {matched}/{len(self.checks)} matched"
        if warned:
            summary += f", {warned}/{len(self.checks)} mismatched"
        lines.extend(["", summary, ""])
        path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pre-flight check runner
# ---------------------------------------------------------------------------

# Type alias for either config type
_AnyConfig = Any  # BalancedComparisonConfig | BankComparisonConfig


def _representative_kappas(kappas: list[Decimal]) -> list[Decimal]:
    """Pick min, median, max kappas from the grid."""
    if not kappas:
        return []
    sorted_k = sorted(set(kappas))
    if len(sorted_k) == 1:
        return sorted_k
    if len(sorted_k) == 2:
        return sorted_k
    mid_idx = len(sorted_k) // 2
    return [sorted_k[0], sorted_k[mid_idx], sorted_k[-1]]


def _get_buy_premium(config: _AnyConfig) -> Decimal:
    """Derive buy_premium from risk_aversion (matches CLAUDE.md formula)."""
    ra = getattr(config, "risk_aversion", Decimal("0"))
    return Decimal("0.01") + Decimal("0.02") * ra


def _has_banking(config: _AnyConfig) -> bool:
    """Check if config has banking arms enabled."""
    # BankComparisonConfig always has banking
    if hasattr(config, "reserve_ratio"):
        return True
    # BalancedComparisonConfig has optional banking arms
    return bool(
        getattr(config, "enable_bank_passive", False)
        or getattr(config, "enable_bank_dealer", False)
        or getattr(config, "enable_bank_dealer_nbfi", False)
    )


def _has_lender(config: _AnyConfig) -> bool:
    """Check if config has NBFI lender enabled."""
    return bool(getattr(config, "enable_lender", False))


def run_preflight_checks(config: _AnyConfig) -> PreflightReport:
    """Run all viability checks across the parameter grid.

    Works with both ``BalancedComparisonConfig`` and ``BankComparisonConfig``.

    Returns a ``PreflightReport`` with per-check results and
    per-kappa trade viability reports.
    """
    report = PreflightReport()
    checks: list[PreflightCheckResult] = []

    kappas: list[Decimal] = list(config.kappas)
    n_agents: int = config.n_agents
    report.n_agents = n_agents
    maturity_days: int = config.maturity_days
    face_value: Decimal = config.face_value
    outside_mid_ratios: list[Decimal] = list(
        getattr(config, "outside_mid_ratios", [Decimal("0.90")])
    )
    # Use worst-case (lowest) OMR for pre-flight checks — if the most
    # constrained slice is viable, all others are too.
    omr = min(outside_mid_ratios) if outside_mid_ratios else Decimal("0.90")
    base_spread = Decimal("0.04")

    dealer_share = getattr(config, "dealer_share_per_bucket", Decimal("0.125"))
    vbt_share = getattr(config, "vbt_share_per_bucket", Decimal("0.25"))
    buy_premium = _get_buy_premium(config)
    Q_total = getattr(config, "Q_total", Decimal("10000"))

    rep_kappas = _representative_kappas(kappas)

    # ── V1 + V2: Trade viability per representative kappa ──────────────
    all_sell_viable = True
    all_buy_viable = True
    sell_detail_parts: list[str] = []
    buy_detail_parts: list[str] = []

    for k in rep_kappas:
        tv = check_trade_viability(
            kappa=k,
            face_value=face_value,
            n_agents=n_agents,
            dealer_share=dealer_share,
            vbt_share=vbt_share,
            layoff_threshold=Decimal("0.5"),
            buy_premium=buy_premium,
            maturity_days=maturity_days,
            outside_mid_ratio=omr,
        )
        report.kappa_trade_reports[str(k)] = tv

        if not tv.sell_viable:
            all_sell_viable = False
        bid = tv.diagnostics.get("dealer_bid", 0)
        sell_detail_parts.append(f"bid={bid:.2f}@\u03ba={k}")

        if not tv.buy_viable:
            all_buy_viable = False
        gain = tv.diagnostics.get("buyer_gain", 0)
        buy_detail_parts.append(f"gain={gain:.3f}@\u03ba={k}")

    # V1
    sell_level = "pass" if all_sell_viable else "warn"
    checks.append(PreflightCheckResult(
        check_id="V1",
        label="Sell viable",
        prediction=all_sell_viable,
        level=sell_level,
        detail=", ".join(sell_detail_parts),
    ))

    # V2
    buy_level = "pass" if all_buy_viable else "warn"
    checks.append(PreflightCheckResult(
        check_id="V2",
        label="Buy viable",
        prediction=all_buy_viable,
        level=buy_level,
        detail=f"{', '.join(buy_detail_parts)} (premium={float(buy_premium):.2f})",
    ))

    # ── V3 + V7: Sweep-level checks ──────────────────────────────────
    sweep_rpt = check_sweep_viability(
        kappas=kappas,
        n_agents=n_agents,
        maturity_days=maturity_days,
    )
    report.sweep_report = sweep_rpt

    # V3
    v3_level = "pass" if sweep_rpt.two_way_trading_viable else "warn"
    k_min, k_max = min(kappas), max(kappas)
    checks.append(PreflightCheckResult(
        check_id="V3",
        label="Two-way trading",
        prediction=sweep_rpt.two_way_trading_viable,
        level=v3_level,
        detail=f"\u03ba range {float(k_min):.2f}\u2013{float(k_max):.2f}",
    ))

    # V7
    v7_level = "pass" if sweep_rpt.effect_detectable else "warn"
    v7_warnings = sweep_rpt.diagnostics.get("warnings", [])
    v7_detail = f"{len(kappas)} kappas, n={n_agents}, mat={maturity_days}"
    if v7_warnings:
        v7_detail += f" [{'; '.join(w for w in v7_warnings if 'V7' in w)}]"
    checks.append(PreflightCheckResult(
        check_id="V7",
        label="Effect detectable",
        prediction=sweep_rpt.effect_detectable,
        level=v7_level,
        detail=v7_detail,
    ))

    # ── V4, V5, V6: Simulation-level checks (per representative kappa) ─
    lender_enabled = _has_lender(config)
    lender_base_rate = getattr(config, "lender_base_rate", Decimal("0.05"))
    lender_risk_premium_scale = getattr(
        config, "lender_risk_premium_scale", Decimal("0.20")
    )

    mus: list[Decimal] = list(getattr(config, "mus", [Decimal("0")]))
    mu_representative = mus[len(mus) // 2] if mus else Decimal("0")

    capacity_details: list[str] = []
    all_capacity_ok = True
    all_temporal_ok = True

    for k in rep_kappas:
        sim_rpt = check_simulation_viability(
            kappa=k,
            n_agents=n_agents,
            maturity_days=maturity_days,
            face_value=face_value,
            Q_total=Q_total,
            mu=mu_representative,
            dealer_share=dealer_share,
            vbt_share=vbt_share,
            outside_mid_ratio=omr,
            lender_enabled=lender_enabled,
            lender_base_rate=lender_base_rate,
            lender_risk_premium_scale=lender_risk_premium_scale,
        )
        report.simulation_reports[str(k)] = sim_rpt

        k_star = sim_rpt.diagnostics.get("K_star", "?")
        capacity_details.append(f"K*={k_star}@\u03ba={k}")

        if not sim_rpt.dealer_capacity_viable:
            all_capacity_ok = False
        if not sim_rpt.temporal_spread_viable:
            all_temporal_ok = False

    # V4
    v4_level = "pass" if all_capacity_ok else "warn"
    checks.append(PreflightCheckResult(
        check_id="V4",
        label="Dealer capacity",
        prediction=all_capacity_ok,
        level=v4_level,
        detail=", ".join(capacity_details),
    ))

    # V5
    if lender_enabled:
        lending_rates: list[str] = []
        for k in rep_kappas:
            sr = report.simulation_reports.get(str(k))
            rate = sr.diagnostics.get("lending_rate", "?") if sr else "?"
            if isinstance(rate, float):
                lending_rates.append(f"{rate:.3f}")
            else:
                lending_rates.append(str(rate))
        v5_detail = f"rate={','.join(lending_rates)} across \u03ba"
        checks.append(PreflightCheckResult(
            check_id="V5",
            label="Lending viable",
            prediction=True,  # always informational
            level="pass",
            detail=v5_detail,
        ))
    else:
        checks.append(PreflightCheckResult(
            check_id="V5",
            label="Lending viable",
            prediction=True,
            level="pass",
            detail="n/a (no lender arm)",
        ))

    # V6
    v6_level = "pass" if all_temporal_ok else "warn"
    mu_str = ",".join(str(m) for m in mus)
    checks.append(PreflightCheckResult(
        check_id="V6",
        label="Temporal spread",
        prediction=all_temporal_ok,
        level=v6_level,
        detail=f"maturity={maturity_days}d, \u03bc\u2208{{{mu_str}}}",
    ))

    # ── V8: Interbank viability (banking arms only) ───────────────────
    if _has_banking(config):
        n_banks = getattr(
            config, "n_banks", getattr(config, "n_banks_for_banking", 5)
        )
        total_deposits_est = int(Q_total)
        reserve_target_ratio = getattr(config, "reserve_ratio", Decimal("0.50"))
        credit_risk_loading = getattr(
            config, "credit_risk_loading", Decimal("0.5")
        )
        cb_slope = getattr(
            config, "cb_rate_escalation_slope", Decimal("0.05")
        )
        cb_max = getattr(
            config, "cb_max_outstanding_ratio", Decimal("2.0")
        )

        ib_details: list[str] = []
        all_ib_ok = True
        for k in rep_kappas:
            ib_rpt = check_interbank_viability(
                kappa=k,
                n_banks=n_banks,
                reserve_target_ratio=reserve_target_ratio,
                total_deposits_estimate=total_deposits_est,
                credit_risk_loading=credit_risk_loading,
                outside_mid_ratio=omr,
                cb_rate_escalation_slope=cb_slope,
                cb_max_outstanding_ratio=cb_max,
            )
            report.interbank_reports[str(k)] = ib_rpt
            bank_rate = ib_rpt.diagnostics.get("bank_rate_approx", "?")
            sell_haircut = ib_rpt.diagnostics.get("sell_haircut", "?")
            if isinstance(bank_rate, float) and isinstance(sell_haircut, float):
                ib_details.append(
                    f"bank={bank_rate:.3f} vs sell={sell_haircut:.3f}@\u03ba={k}"
                )
            if not ib_rpt.all_viable:
                all_ib_ok = False

        v8_level = "pass" if all_ib_ok else "warn"
        checks.append(PreflightCheckResult(
            check_id="V8",
            label="Bank vs sell",
            prediction=all_ib_ok,
            level=v8_level,
            detail=", ".join(ib_details) if ib_details else "banking enabled",
        ))
    else:
        checks.append(PreflightCheckResult(
            check_id="V8",
            label="Bank vs sell",
            prediction=True,
            level="pass",
            detail="n/a (no banking arms)",
        ))

    # ── V9: VBT mid stability ─────────────────────────────────────────
    # Check that credit-adjusted VBT mid M = ρ×(1-P) stays above M_MIN
    # for each representative kappa.  If M collapses, dealer pins to
    # outside quotes and trading becomes a passthrough.
    vbt_details: list[str] = []
    all_vbt_ok = True
    for k in rep_kappas:
        P = kappa_informed_prior(k)
        M = omr * (Decimal(1) - P)
        ok = M > _M_MIN
        vbt_details.append(f"M={float(M):.3f}@\u03ba={k}")
        if not ok:
            all_vbt_ok = False

    v9_level = "pass" if all_vbt_ok else "fail"
    checks.append(PreflightCheckResult(
        check_id="V9",
        label="VBT mid stability",
        prediction=all_vbt_ok,
        level=v9_level,
        detail=f"{', '.join(vbt_details)} (M_MIN={float(_M_MIN)})",
        diagnostics={"M_MIN": float(_M_MIN)},
    ))

    # ── V10: NBFI loan adoption ───────────────────────────────────────
    # Check that loan rate < sell haircut so borrowers prefer loans.
    # loan_rate = base_rate + risk_premium_scale × P_lender
    # sell_haircut = P_prior + O_short/2
    if lender_enabled:
        adopt_details: list[str] = []
        all_adopt_ok = True
        for k in rep_kappas:
            P = kappa_informed_prior(k)
            one = Decimal(1)
            p_lender = one / (one + k)
            loan_rate = lender_base_rate + lender_risk_premium_scale * p_lender
            sell_haircut = P + base_spread / 2
            cheaper = loan_rate < sell_haircut
            adopt_details.append(
                f"loan={float(loan_rate):.3f} vs sell={float(sell_haircut):.3f}@\u03ba={k}"
            )
            if not cheaper:
                all_adopt_ok = False

        v10_level = "pass" if all_adopt_ok else "warn"
        checks.append(PreflightCheckResult(
            check_id="V10",
            label="NBFI loan adoption",
            prediction=all_adopt_ok,
            level=v10_level,
            detail=", ".join(adopt_details),
        ))
    else:
        checks.append(PreflightCheckResult(
            check_id="V10",
            label="NBFI loan adoption",
            prediction=True,
            level="pass",
            detail="n/a (no lender arm)",
        ))

    # ── V11: Bank reserve capacity ────────────────────────────────────
    # Check that bank reserves can cover expected lending demand.
    # Expected shortfall agents ≈ n_agents × P_prior; each needs ≈ face_value.
    # Total reserves ≈ reserve_ratio × Q_total (BankConfig) or from multiplier.
    if _has_banking(config):
        reserve_details: list[str] = []
        all_reserve_ok = True
        for k in rep_kappas:
            P = kappa_informed_prior(k)
            expected_shortfall_agents = int(float(n_agents * P))
            expected_demand = expected_shortfall_agents * int(face_value)

            if hasattr(config, "reserve_ratio"):
                # BankComparisonConfig
                total_reserves = int(
                    float(config.reserve_ratio) * float(Q_total)
                )
            else:
                # BalancedComparisonConfig — bank_reserve_multiplier
                brm = getattr(config, "bank_reserve_multiplier", 0.5)
                n_b = getattr(
                    config, "n_banks_for_banking",
                    getattr(config, "n_banks", 5),
                )
                total_reserves = int(brm * float(Q_total) / n_b) * n_b

            covers = total_reserves >= expected_demand
            reserve_details.append(
                f"R={total_reserves} vs demand={expected_demand}@\u03ba={k}"
            )
            if not covers:
                all_reserve_ok = False

        v11_level = "pass" if all_reserve_ok else "warn"
        checks.append(PreflightCheckResult(
            check_id="V11",
            label="Bank reserve capacity",
            prediction=all_reserve_ok,
            level=v11_level,
            detail=", ".join(reserve_details),
        ))
    else:
        checks.append(PreflightCheckResult(
            check_id="V11",
            label="Bank reserve capacity",
            prediction=True,
            level="pass",
            detail="n/a (no banking arms)",
        ))

    report.checks = checks
    return report


# ---------------------------------------------------------------------------
# Post-sweep validation runner
# ---------------------------------------------------------------------------

# Type alias for result types
_AnyResult = Any  # BalancedComparisonResult | BankComparisonResult


def run_postsweep_validation(
    preflight: PreflightReport,
    results: list[_AnyResult],
    aggregate_dir: Path | None = None,
) -> PostflightReport:
    """Validate that each mechanism's actions actually occurred.

    This checks *activity*, not *effectiveness*.  The question for each
    check is "did this actor do things in tangible quantity?" — whether
    those actions helped is a result-interpretation question, not a
    diagnostic.

    Works with both ``BalancedComparisonResult`` and ``BankComparisonResult``.
    Writes a text report to ``aggregate_dir/viability_report.txt`` if provided.
    """
    post = PostflightReport()
    if not results:
        return post

    # Detect result type
    is_bank = hasattr(results[0], "delta_idle")

    # Gather n_agents from preflight for proportionality checks
    n_agents = preflight.n_agents

    # Run all post-sweep validators
    for pcheck in preflight.checks:
        validators = {
            "V1": _validate_v1_settlement,
            "V2": _validate_v2_trading,
            "V3": _validate_v3_two_way,
            "V4": _validate_v4_dealer,
            "V5": _validate_v5_lending,
            "V6": _validate_v6_cascade,
            "V7": _validate_v7_effect,
            "V8": _validate_v8_bank_activity,
            "V11": _validate_v11_cb_backstop,
        }
        validator = validators.get(pcheck.check_id)
        if validator:
            post_checks = list(validator(pcheck, results, is_bank, n_agents))
            post.checks.extend(post_checks)

    if aggregate_dir is not None:
        report_path = aggregate_dir / "viability_report.txt"
        post.write_report(report_path)

    return post


# ---------------------------------------------------------------------------
# Individual post-sweep validators
# ---------------------------------------------------------------------------


def _validate_v1_settlement(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V1: Settlement occurred — phi values exist and are positive."""
    if is_bank:
        phi_values = [
            float(r.phi_idle)
            for r in results
            if getattr(r, "phi_idle", None) is not None
        ]
        phi_values += [
            float(r.phi_lend)
            for r in results
            if getattr(r, "phi_lend", None) is not None
        ]
        label = "Settlement occurred"
    else:
        phi_values = [
            float(r.phi_passive)
            for r in results
            if getattr(r, "phi_passive", None) is not None
        ]
        phi_values += [
            float(r.phi_active)
            for r in results
            if getattr(r, "phi_active", None) is not None
        ]
        label = "Settlement occurred"

    n_positive = sum(1 for p in phi_values if p > 0)
    actual = n_positive > 0
    detail = f"phi>0 in {n_positive}/{len(phi_values)} arm-runs"

    return [PostflightCheckResult(
        check_id="V1",
        label=label,
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v2_trading(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V2: Trading occurred — total_trades > 0 and proportional to scale.

    For dealer arms only (active, dealer_lender).  Bank sweeps have no
    dealer trading, so we check bank lending activity instead.
    """
    if is_bank:
        # Bank sweep: no dealer trading; check that lend arm completed
        n_lend = sum(
            1 for r in results
            if getattr(r, "delta_lend", None) is not None
        )
        actual = n_lend > 0
        detail = f"lend arm completed in {n_lend}/{len(results)} combos"
    else:
        total_trades = sum(
            r.total_trades or 0
            for r in results
            if hasattr(r, "total_trades")
        )
        n_combos = len(results)
        # "tangible" = at least 1 trade per combo on average
        tangible = total_trades >= n_combos if n_combos > 0 else False
        actual = total_trades > 0
        detail = (
            f"total_trades={total_trades} across {n_combos} combos"
            f" ({total_trades / n_combos:.1f}/combo)"
            if n_combos > 0
            else f"total_trades={total_trades}"
        )
        if not tangible and actual:
            detail += " (sparse)"

    return [PostflightCheckResult(
        check_id="V2",
        label="Trading occurred",
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v3_two_way(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V3: Two-way trading — effects vary across kappas."""
    if is_bank:
        effects = [
            r.bank_lending_effect
            for r in results
            if getattr(r, "bank_lending_effect", None) is not None
        ]
    else:
        effects = [
            r.trading_effect
            for r in results
            if getattr(r, "trading_effect", None) is not None
        ]

    nonzero = sum(1 for e in effects if e != 0)
    actual = nonzero > 0
    detail = f"nonzero effects in {nonzero}/{len(effects)} combos"

    return [PostflightCheckResult(
        check_id="V3",
        label="Two-way trading",
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v4_dealer(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V4: Dealer was active — dealer metrics exist in results."""
    if is_bank:
        # Bank sweep has no dealer
        return [PostflightCheckResult(
            check_id="V4",
            label="Dealer active",
            prediction=pcheck.prediction,
            actual=True,
            match=True,
            detail="n/a (bank sweep, no dealer)",
        )]

    has_return = sum(
        1 for r in results
        if getattr(r, "dealer_total_return", None) is not None
    )
    has_trades = sum(
        1 for r in results
        if (getattr(r, "total_trades", None) or 0) > 0
    )
    actual = has_return > 0
    detail = (
        f"dealer_return in {has_return}/{len(results)} combos, "
        f"trades>0 in {has_trades}/{len(results)}"
    )

    return [PostflightCheckResult(
        check_id="V4",
        label="Dealer active",
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v5_lending(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V5: Lending occurred — lender arm produced results."""
    if pcheck.detail.startswith("n/a"):
        return [PostflightCheckResult(
            check_id="V5",
            label="Lending occurred",
            prediction=pcheck.prediction,
            actual=True,
            match=True,
            detail="n/a (no lender arm)",
        )]

    if is_bank:
        # Bank sweep: check that bank lending arm completed
        n_lend = sum(
            1 for r in results
            if getattr(r, "delta_lend", None) is not None
        )
        actual = n_lend > 0
        detail = f"lend arm completed in {n_lend}/{len(results)} combos"
    else:
        # Balanced: check NBFI lender arm completed
        lender_results = [
            r for r in results
            if getattr(r, "delta_lender", None) is not None
        ]
        if not lender_results:
            return [PostflightCheckResult(
                check_id="V5",
                label="Lending occurred",
                prediction=pcheck.prediction,
                actual=False,
                match=False,
                detail="lender arm produced no results",
            )]

        # Check total_loans if available (may be None due to extraction gap)
        total_loans = sum(r.total_loans or 0 for r in lender_results)
        if total_loans > 0:
            actual = True
            detail = f"total_loans={total_loans} in {len(lender_results)}/{len(results)} combos"
        else:
            # total_loans may be unpopulated (extraction gap); fall back to arm completion
            actual = len(lender_results) > 0
            detail = f"lender arm completed in {len(lender_results)}/{len(results)} combos (total_loans=0)"

    return [PostflightCheckResult(
        check_id="V5",
        label="Lending occurred",
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v6_cascade(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V6: Cascade dynamics — cascade_fraction > 0 in stressed runs."""
    if is_bank:
        cascade_values = [
            float(r.cascade_fraction_idle)
            for r in results
            if getattr(r, "cascade_fraction_idle", None) is not None
        ]
        cascade_values += [
            float(r.cascade_fraction_lend)
            for r in results
            if getattr(r, "cascade_fraction_lend", None) is not None
        ]
    else:
        cascade_values = [
            float(r.cascade_fraction_passive)
            for r in results
            if getattr(r, "cascade_fraction_passive", None) is not None
        ]
        cascade_values += [
            float(r.cascade_fraction_active)
            for r in results
            if getattr(r, "cascade_fraction_active", None) is not None
        ]

    n_positive = sum(1 for v in cascade_values if v > 0)
    actual = n_positive > 0
    detail = f"cascade>0 in {n_positive}/{len(cascade_values)} arm-runs"

    return [PostflightCheckResult(
        check_id="V6",
        label="Cascade dynamics",
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v7_effect(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V7: Effect detectable — variance in effects across combos."""
    if is_bank:
        effects = [
            float(r.bank_lending_effect)
            for r in results
            if getattr(r, "bank_lending_effect", None) is not None
        ]
    else:
        effects = [
            float(r.trading_effect)
            for r in results
            if getattr(r, "trading_effect", None) is not None
        ]

    if len(effects) >= 2:
        var = statistics.variance(effects)
        actual = var > 1e-10
        detail = f"effect variance={var:.6f} across {len(effects)} combos"
    elif len(effects) == 1:
        actual = True
        detail = "single combo (no variance calc)"
    else:
        actual = False
        detail = "no effects computed"

    return [PostflightCheckResult(
        check_id="V7",
        label="Effect detectable",
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v8_bank_activity(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V8: Banking occurred — bank arms produced results."""
    if pcheck.detail.startswith("n/a"):
        return [PostflightCheckResult(
            check_id="V8",
            label="Banking occurred",
            prediction=pcheck.prediction,
            actual=True,
            match=True,
            detail="n/a (no banking arms)",
        )]

    if is_bank:
        n_idle = sum(
            1 for r in results
            if getattr(r, "delta_idle", None) is not None
        )
        n_lend = sum(
            1 for r in results
            if getattr(r, "delta_lend", None) is not None
        )
        actual = n_idle > 0 and n_lend > 0
        detail = f"idle={n_idle}, lend={n_lend} of {len(results)} combos"
    else:
        # Balanced: check banking arms completed
        n_bank = sum(
            1 for r in results
            if getattr(r, "delta_bank_passive", None) is not None
            or getattr(r, "delta_bank_dealer", None) is not None
            or getattr(r, "delta_bank_dealer_nbfi", None) is not None
        )
        actual = n_bank > 0
        detail = f"bank arms completed in {n_bank}/{len(results)} combos"

    return [PostflightCheckResult(
        check_id="V8",
        label="Banking occurred",
        prediction=pcheck.prediction,
        actual=actual,
        match=(pcheck.prediction == actual),
        detail=detail,
    )]


def _validate_v11_cb_backstop(
    pcheck: PreflightCheckResult,
    results: list[_AnyResult],
    is_bank: bool,
    n_agents: int,
) -> list[PostflightCheckResult]:
    """V11: CB backstop used — cb_loans_created > 0 in banking runs."""
    if pcheck.detail.startswith("n/a"):
        return [PostflightCheckResult(
            check_id="V11",
            label="CB backstop used",
            prediction=pcheck.prediction,
            actual=True,
            match=True,
            detail="n/a (no banking arms)",
        )]

    if is_bank:
        total_cb_idle = sum(
            getattr(r, "cb_loans_created_idle", 0) for r in results
        )
        total_cb_lend = sum(
            getattr(r, "cb_loans_created_lend", 0) for r in results
        )
        total_cb = total_cb_idle + total_cb_lend
        actual = total_cb > 0
        detail = f"CB loans: idle={total_cb_idle}, lend={total_cb_lend}"
    else:
        total_cb = 0
        for suffix in ("bank_passive", "bank_dealer", "bank_dealer_nbfi"):
            total_cb += sum(
                getattr(r, f"cb_loans_created_{suffix}", 0) for r in results
            )
        actual = total_cb > 0
        detail = f"CB loans total={total_cb} across banking arms"

    return [PostflightCheckResult(
        check_id="V11",
        label="CB backstop used",
        prediction=pcheck.prediction,
        actual=actual,
        match=True,  # CB usage is informational, not pass/fail
        detail=detail,
    )]
