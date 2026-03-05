"""Adaptive preset system for decision profiles (Plan 050).

The 2x2 preset system enables factorial experiments:
- static:     all flags False (current behavior, bit-identical)
- calibrated: pre-run calibration only (flags with [PRE] tag)
- responsive: within-run adaptation only (flags with [RUN] tag)
- full:       both pre-run and within-run adaptation
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum


class AdaptPreset(str, Enum):
    """Adaptation preset levels."""

    STATIC = "static"
    CALIBRATED = "calibrated"
    RESPONSIVE = "responsive"
    FULL = "full"


def build_adaptive_overrides(
    preset: AdaptPreset | str,
    kappa: Decimal,
    mu: Decimal,
    c: Decimal,
    maturity_days: int,
    n_agents: int,
) -> dict[str, dict]:
    """Return dict of profile overrides for the given adaptation preset.

    The returned dict has keys: trader, risk_params, vbt, bank, lender, cb.
    Each contains field name -> value mappings to merge into profile construction.

    Args:
        preset: Adaptation level.
        kappa: System liquidity ratio.
        mu: Maturity timing skew.
        c: Dirichlet concentration.
        maturity_days: Scenario maturity days.
        n_agents: Number of agents in ring.

    Returns:
        Nested dict of overrides per profile type.
    """
    if isinstance(preset, str):
        preset = AdaptPreset(preset)

    pre_run = preset in (AdaptPreset.CALIBRATED, AdaptPreset.FULL)
    in_run = preset in (AdaptPreset.RESPONSIVE, AdaptPreset.FULL)

    overrides: dict[str, dict] = {
        "trader": {},
        "risk_params": {},
        "vbt": {},
        "bank": {},
        "lender": {},
        "cb": {},
    }

    if not pre_run and not in_run:
        return overrides  # static: no changes

    # ── Pre-run calibration flags ──
    if pre_run:
        # Trader: scale planning horizon with maturity
        scaled_ph = max(1, min(20, maturity_days))
        overrides["trader"]["adaptive_planning_horizon"] = True
        overrides["trader"]["planning_horizon"] = scaled_ph

        # Risk params: scale lookback with maturity, enable issuer-specific
        scaled_lookback = max(3, min(30, maturity_days))
        overrides["risk_params"]["adaptive_lookback"] = True
        overrides["risk_params"]["lookback_window"] = scaled_lookback
        overrides["risk_params"]["adaptive_issuer_specific"] = True
        overrides["risk_params"]["use_issuer_specific"] = True

        # VBT: term structure, adaptive base spreads, stress horizon
        overrides["vbt"]["adaptive_term_structure"] = True
        overrides["vbt"]["adaptive_base_spreads"] = True
        overrides["vbt"]["adaptive_stress_horizon"] = True
        overrides["vbt"]["stress_horizon"] = max(3, min(20, maturity_days))

        # Bank: corridor incorporates mu, c
        overrides["bank"]["adaptive_corridor"] = True

        # Lender: calibrate risk aversion, profit target, loan maturity
        # Risk aversion: lower when kappa is high (less stress)
        lender_ra = max(Decimal("0.1"), min(Decimal("0.9"),
            Decimal("0.3") + Decimal("0.3") * max(Decimal(0), Decimal(1) - kappa)))
        overrides["lender"]["adaptive_risk_aversion"] = True
        overrides["lender"]["risk_aversion"] = lender_ra
        overrides["lender"]["adaptive_profit_target"] = True
        overrides["lender"]["adaptive_loan_maturity"] = True
        overrides["lender"]["max_loan_maturity"] = max(2, min(maturity_days, 10))

        # CB: adaptive betas
        overrides["cb"]["adaptive_betas"] = True

    # ── Within-run adaptation flags ──
    if in_run:
        overrides["trader"]["adaptive_risk_aversion"] = True
        overrides["trader"]["adaptive_reserves"] = True
        overrides["trader"]["adaptive_ev_term_structure"] = True

        overrides["risk_params"]["adaptive_ev_term_structure"] = True

        overrides["vbt"]["adaptive_convex_spreads"] = True
        overrides["vbt"]["adaptive_per_bucket_tracking"] = True
        overrides["vbt"]["adaptive_issuer_pricing"] = True

        overrides["lender"]["adaptive_rates"] = True
        overrides["lender"]["adaptive_capital_conservation"] = True
        overrides["lender"]["adaptive_prevention"] = True

        overrides["cb"]["adaptive_early_warning"] = True

    return overrides


__all__ = ["AdaptPreset", "build_adaptive_overrides"]
