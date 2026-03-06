"""Sequential tuning helpers for intermediary support experiments.

These utilities score paired frontier runs against a narrow objective:
increase default relief without adding intermediary loss, preserve
non-negative net system relief, and compress loss per unit of help.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import inf
from typing import Any

from bilancio.analysis.intermediary_frontier import FrontierArtifact, FrontierPair


def _decimal(value: str) -> Decimal:
    return Decimal(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _share(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


def _metric_or_neg_inf(value: float | None) -> float:
    return -inf if value is None else value


def _loss_metric(value: float | None) -> float:
    return 0.0 if value is None else -value


@dataclass(frozen=True)
class TuningCandidate:
    """Named candidate override set for one intermediary mechanism."""

    name: str
    rationale: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class FrontierScore:
    """Compact scorecard for a single treatment arm."""

    treatment_arm: str
    n_pairs: int
    n_complete: int
    n_help: int
    n_help_no_extra_inst_loss: int
    n_help_nonnegative_net_system: int
    n_safe_help: int
    share_help: float | None
    share_help_no_extra_inst_loss: float | None
    share_help_nonnegative_net_system: float | None
    share_safe_help: float | None
    mean_default_relief: float | None
    mean_safe_default_relief: float | None
    mean_inst_loss_shift: float | None
    mean_net_system_relief: float | None
    mean_loss_per_help: float | None

    def ranking_key(self) -> tuple[float, float, float, float, float]:
        """Lexicographic score aligned to the Task 4 objective."""
        return (
            _metric_or_neg_inf(self.share_safe_help),
            _metric_or_neg_inf(self.share_help_nonnegative_net_system),
            _loss_metric(self.mean_loss_per_help),
            _metric_or_neg_inf(self.mean_safe_default_relief),
            _metric_or_neg_inf(self.mean_net_system_relief),
        )


def score_frontier_pairs(
    pairs: list[FrontierPair],
    *,
    treatment_arm: str,
) -> FrontierScore:
    """Score a treatment arm using safe-help and loss-efficiency criteria."""
    arm_pairs = [pair for pair in pairs if pair.treatment_arm == treatment_arm]
    complete = [pair for pair in arm_pairs if pair.default_relief is not None]
    help_pairs = [pair for pair in complete if pair.default_relief > 0]
    no_extra_pairs = [pair for pair in help_pairs if pair.inst_loss_shift <= 0]
    nonnegative_net_pairs = [pair for pair in help_pairs if pair.net_system_relief >= 0]
    safe_pairs = [pair for pair in help_pairs if pair.inst_loss_shift <= 0 and pair.net_system_relief >= 0]

    loss_per_help = [
        max(pair.inst_loss_shift, 0.0) / pair.default_relief
        for pair in help_pairs
    ]

    return FrontierScore(
        treatment_arm=treatment_arm,
        n_pairs=len(arm_pairs),
        n_complete=len(complete),
        n_help=len(help_pairs),
        n_help_no_extra_inst_loss=len(no_extra_pairs),
        n_help_nonnegative_net_system=len(nonnegative_net_pairs),
        n_safe_help=len(safe_pairs),
        share_help=_share(len(help_pairs), len(complete)),
        share_help_no_extra_inst_loss=_share(len(no_extra_pairs), len(complete)),
        share_help_nonnegative_net_system=_share(len(nonnegative_net_pairs), len(complete)),
        share_safe_help=_share(len(safe_pairs), len(complete)),
        mean_default_relief=_mean(
            [pair.default_relief for pair in complete if pair.default_relief is not None]
        ),
        mean_safe_default_relief=_mean(
            [pair.default_relief for pair in safe_pairs if pair.default_relief is not None]
        ),
        mean_inst_loss_shift=_mean([pair.inst_loss_shift for pair in complete]),
        mean_net_system_relief=_mean([pair.net_system_relief for pair in complete]),
        mean_loss_per_help=_mean(loss_per_help),
    )


def score_frontier_artifact(
    artifact: FrontierArtifact,
    *,
    treatment_arm: str,
) -> FrontierScore:
    """Convenience wrapper around ``score_frontier_pairs``."""
    return score_frontier_pairs(artifact.pairs, treatment_arm=treatment_arm)


def choose_best_candidate(scores: list[tuple[TuningCandidate, FrontierScore]]) -> tuple[TuningCandidate, FrontierScore]:
    """Select the highest-ranked candidate, keeping input order on ties."""
    if not scores:
        raise ValueError("scores must not be empty")
    best_candidate, best_score = scores[0]
    for candidate, score in scores[1:]:
        if score.ranking_key() > best_score.ranking_key():
            best_candidate = candidate
            best_score = score
    return best_candidate, best_score


def dealer_candidates() -> tuple[TuningCandidate, ...]:
    """Reasonable dealer-side candidates using only trader/VBT adjustments."""
    return (
        TuningCandidate(
            name="baseline",
            rationale="Current dealer-side comparison defaults.",
            overrides={},
        ),
        TuningCandidate(
            name="flow_buffered_vbt",
            rationale="Slightly wider VBT response to stress and order-flow pressure.",
            overrides={
                "vbt_spread_sensitivity": _decimal("0.15"),
                "flow_sensitivity": _decimal("0.15"),
                "spread_scale": _decimal("1.05"),
            },
        ),
        TuningCandidate(
            name="prudent_traders",
            rationale="Moderately longer planning and lower urgency from traders.",
            overrides={
                "risk_aversion": _decimal("0.15"),
                "planning_horizon": 12,
                "aggressiveness": _decimal("0.90"),
            },
        ),
        TuningCandidate(
            name="coordinated_buffering",
            rationale="Combine modest trader prudence with modest VBT stress buffering.",
            overrides={
                "risk_aversion": _decimal("0.12"),
                "planning_horizon": 12,
                "aggressiveness": _decimal("0.92"),
                "vbt_spread_sensitivity": _decimal("0.12"),
                "flow_sensitivity": _decimal("0.12"),
                "spread_scale": _decimal("1.04"),
            },
        ),
    )


def bank_candidates() -> tuple[TuningCandidate, ...]:
    """Reasonable bank-side candidates using CB/trader adjustments only."""
    return (
        TuningCandidate(
            name="baseline",
            rationale="Current bank comparison defaults.",
            overrides={},
        ),
        TuningCandidate(
            name="cb_liquidity_buffer",
            rationale="Softer CB escalation and slightly deeper backstop capacity.",
            overrides={
                "cb_rate_escalation_slope": _decimal("0.03"),
                "cb_max_outstanding_ratio": _decimal("2.50"),
            },
        ),
        TuningCandidate(
            name="trader_buffer",
            rationale="Moderately more conservative trader purchasing under stress.",
            overrides={
                "risk_aversion": _decimal("0.15"),
                "planning_horizon": 12,
                "aggressiveness": _decimal("0.90"),
            },
        ),
        TuningCandidate(
            name="cb_plus_trader_buffer",
            rationale="Combine a softer CB backstop with modest trader prudence.",
            overrides={
                "cb_rate_escalation_slope": _decimal("0.03"),
                "cb_max_outstanding_ratio": _decimal("2.50"),
                "risk_aversion": _decimal("0.12"),
                "planning_horizon": 12,
                "aggressiveness": _decimal("0.92"),
            },
        ),
    )


def nbfi_candidates() -> tuple[TuningCandidate, ...]:
    """Reasonable NBFI-side candidates using lender/VBT/trader adjustments."""
    return (
        TuningCandidate(
            name="baseline",
            rationale="Current NBFI comparison defaults.",
            overrides={},
        ),
        TuningCandidate(
            name="measured_guardrails",
            rationale="Light-touch NBFI loss discipline that aims to preserve support capacity.",
            overrides={
                "lender_max_single_exposure": _decimal("0.14"),
                "lender_max_total_exposure": _decimal("0.75"),
                "lender_marginal_relief_min_ratio": _decimal("0.50"),
                "lender_stress_risk_premium_scale": _decimal("0.05"),
                "lender_high_risk_default_threshold": _decimal("0.65"),
                "lender_daily_expected_loss_budget_ratio": _decimal("0.05"),
                "lender_run_expected_loss_budget_ratio": _decimal("0.12"),
                "lender_stop_loss_realized_ratio": _decimal("0.15"),
            },
        ),
        TuningCandidate(
            name="discipline_buffer",
            rationale="Tighter underwriting, shorter risky terms, and collateral discipline.",
            overrides={
                "lender_min_coverage": _decimal("0.60"),
                "lender_maturity_matching": True,
                "lender_ranking_mode": "blended",
                "lender_cascade_weight": _decimal("0.35"),
                "lender_coverage_mode": "graduated",
                "lender_coverage_penalty_scale": _decimal("0.15"),
                "lender_marginal_relief_min_ratio": _decimal("1.20"),
                "lender_stress_risk_premium_scale": _decimal("0.12"),
                "lender_high_risk_default_threshold": _decimal("0.60"),
                "lender_high_risk_maturity_cap": 1,
                "lender_daily_expected_loss_budget_ratio": _decimal("0.03"),
                "lender_run_expected_loss_budget_ratio": _decimal("0.08"),
                "lender_stop_loss_realized_ratio": _decimal("0.10"),
                "lender_collateralized_terms": True,
                "lender_collateral_advance_rate": _decimal("0.80"),
                "lender_max_single_exposure": _decimal("0.12"),
                "lender_max_total_exposure": _decimal("0.70"),
            },
        ),
        TuningCandidate(
            name="preventive_guardrails",
            rationale="Allow selective preventive support, but only with explicit loss guardrails.",
            overrides={
                "lender_min_coverage": _decimal("0.60"),
                "lender_coverage_mode": "graduated",
                "lender_coverage_penalty_scale": _decimal("0.15"),
                "lender_preventive_lending": True,
                "lender_prevention_threshold": _decimal("0.45"),
                "lender_marginal_relief_min_ratio": _decimal("1.10"),
                "lender_stress_risk_premium_scale": _decimal("0.10"),
                "lender_daily_expected_loss_budget_ratio": _decimal("0.03"),
                "lender_run_expected_loss_budget_ratio": _decimal("0.08"),
                "lender_stop_loss_realized_ratio": _decimal("0.10"),
                "lender_collateralized_terms": True,
                "lender_collateral_advance_rate": _decimal("0.78"),
            },
        ),
        TuningCandidate(
            name="guardrails_plus_prudent_flow",
            rationale="Pair NBFI guardrails with modest trader and VBT stress-damping.",
            overrides={
                "risk_aversion": _decimal("0.12"),
                "planning_horizon": 12,
                "aggressiveness": _decimal("0.92"),
                "vbt_spread_sensitivity": _decimal("0.10"),
                "flow_sensitivity": _decimal("0.10"),
                "lender_min_coverage": _decimal("0.60"),
                "lender_maturity_matching": True,
                "lender_ranking_mode": "blended",
                "lender_cascade_weight": _decimal("0.35"),
                "lender_coverage_mode": "graduated",
                "lender_coverage_penalty_scale": _decimal("0.12"),
                "lender_marginal_relief_min_ratio": _decimal("1.15"),
                "lender_stress_risk_premium_scale": _decimal("0.10"),
                "lender_high_risk_default_threshold": _decimal("0.60"),
                "lender_high_risk_maturity_cap": 1,
                "lender_daily_expected_loss_budget_ratio": _decimal("0.03"),
                "lender_run_expected_loss_budget_ratio": _decimal("0.08"),
                "lender_stop_loss_realized_ratio": _decimal("0.10"),
                "lender_collateralized_terms": True,
                "lender_collateral_advance_rate": _decimal("0.80"),
                "lender_max_single_exposure": _decimal("0.12"),
                "lender_max_total_exposure": _decimal("0.70"),
            },
        ),
    )


__all__ = [
    "FrontierScore",
    "TuningCandidate",
    "bank_candidates",
    "choose_best_candidate",
    "dealer_candidates",
    "nbfi_candidates",
    "score_frontier_artifact",
    "score_frontier_pairs",
]
