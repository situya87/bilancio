from __future__ import annotations

from bilancio.analysis.intermediary_frontier import FrontierPair
from bilancio.analysis.intermediary_support_tuning import (
    FrontierScore,
    TuningCandidate,
    choose_best_candidate,
    dealer_candidates,
    nbfi_candidates,
    score_frontier_pairs,
)


def _pair(
    *,
    treatment_arm: str = "active",
    default_relief: float | None,
    inst_loss_shift: float,
    net_system_relief: float,
) -> FrontierPair:
    return FrontierPair(
        cell_id="kappa=1,c=1,mu=0",
        kappa=1.0,
        concentration=1.0,
        mu=0.0,
        seed=1,
        treatment_arm=treatment_arm,
        control_arm="passive",
        treatment_run_id="treat",
        control_run_id="control",
        delta_treatment=0.2,
        delta_control=0.3,
        default_relief=default_relief,
        inst_loss_shift=inst_loss_shift,
        net_system_relief=net_system_relief,
        pareto_label="pareto_improving",
        help_no_extra_inst_loss=bool(default_relief is not None and default_relief > 0 and inst_loss_shift <= 0),
        help_positive_net_system=bool(default_relief is not None and default_relief > 0 and net_system_relief > 0),
        pareto_improving=bool(
            default_relief is not None
            and default_relief > 0
            and inst_loss_shift <= 0
            and net_system_relief >= 0
        ),
    )


def test_score_frontier_pairs_computes_safe_help_and_loss_per_help():
    score = score_frontier_pairs(
        [
            _pair(default_relief=0.10, inst_loss_shift=-0.01, net_system_relief=0.05),
            _pair(default_relief=0.08, inst_loss_shift=0.02, net_system_relief=0.03),
            _pair(default_relief=0.06, inst_loss_shift=0.00, net_system_relief=-0.01),
            _pair(default_relief=-0.02, inst_loss_shift=-0.03, net_system_relief=0.01),
        ],
        treatment_arm="active",
    )

    assert score.n_pairs == 4
    assert score.n_complete == 4
    assert score.n_help == 3
    assert score.n_help_no_extra_inst_loss == 2
    assert score.n_help_nonnegative_net_system == 2
    assert score.n_safe_help == 1
    assert score.share_help == 0.75
    assert score.share_help_no_extra_inst_loss == 0.5
    assert score.share_help_nonnegative_net_system == 0.5
    assert score.share_safe_help == 0.25
    assert score.mean_loss_per_help == ((0.0 / 0.10) + (0.02 / 0.08) + (0.0 / 0.06)) / 3


def test_choose_best_candidate_prefers_higher_safe_help_share():
    baseline = TuningCandidate(name="baseline", rationale="base", overrides={})
    challenger = TuningCandidate(name="challenger", rationale="alt", overrides={})

    baseline_score = FrontierScore(
        treatment_arm="active",
        n_pairs=10,
        n_complete=10,
        n_help=4,
        n_help_no_extra_inst_loss=2,
        n_help_nonnegative_net_system=2,
        n_safe_help=1,
        share_help=0.4,
        share_help_no_extra_inst_loss=0.2,
        share_help_nonnegative_net_system=0.2,
        share_safe_help=0.1,
        mean_default_relief=0.04,
        mean_safe_default_relief=0.03,
        mean_inst_loss_shift=0.01,
        mean_net_system_relief=0.01,
        mean_loss_per_help=0.20,
    )
    challenger_score = FrontierScore(
        treatment_arm="active",
        n_pairs=10,
        n_complete=10,
        n_help=4,
        n_help_no_extra_inst_loss=3,
        n_help_nonnegative_net_system=3,
        n_safe_help=2,
        share_help=0.4,
        share_help_no_extra_inst_loss=0.3,
        share_help_nonnegative_net_system=0.3,
        share_safe_help=0.2,
        mean_default_relief=0.04,
        mean_safe_default_relief=0.035,
        mean_inst_loss_shift=0.0,
        mean_net_system_relief=0.02,
        mean_loss_per_help=0.05,
    )

    candidate, score = choose_best_candidate(
        [(baseline, baseline_score), (challenger, challenger_score)]
    )

    assert candidate.name == "challenger"
    assert score is challenger_score


def test_candidate_factories_expose_baseline_and_guardrail_options():
    assert dealer_candidates()[0].name == "baseline"
    names = [candidate.name for candidate in nbfi_candidates()]
    assert "discipline_buffer" in names
    assert "guardrails_plus_prudent_flow" in names


def test_score_frontier_pairs_empty_list():
    score = score_frontier_pairs([], treatment_arm="active")
    assert score.n_pairs == 0
    assert score.n_complete == 0
    assert score.n_help == 0
    assert score.share_help is None
    assert score.mean_loss_per_help is None


def test_score_frontier_pairs_no_matching_arm():
    pairs = [_pair(treatment_arm="active", default_relief=0.1, inst_loss_shift=0.0, net_system_relief=0.1)]
    score = score_frontier_pairs(pairs, treatment_arm="nonexistent")
    assert score.n_pairs == 0


def test_score_frontier_pairs_all_negative_relief():
    pairs = [
        _pair(default_relief=-0.05, inst_loss_shift=0.0, net_system_relief=0.0),
        _pair(default_relief=-0.02, inst_loss_shift=-0.01, net_system_relief=0.01),
    ]
    score = score_frontier_pairs(pairs, treatment_arm="active")
    assert score.n_help == 0
    assert score.mean_loss_per_help is None


def test_choose_best_candidate_single_candidate():
    candidate = TuningCandidate(name="only", rationale="only option", overrides={})
    score = FrontierScore(
        treatment_arm="active", n_pairs=5, n_complete=5, n_help=2,
        n_help_no_extra_inst_loss=1, n_help_nonnegative_net_system=1,
        n_safe_help=1, share_help=0.4, share_help_no_extra_inst_loss=0.2,
        share_help_nonnegative_net_system=0.2, share_safe_help=0.2,
        mean_default_relief=0.05, mean_safe_default_relief=0.03,
        mean_inst_loss_shift=0.0, mean_net_system_relief=0.01,
        mean_loss_per_help=0.1,
    )
    best_candidate, best_score = choose_best_candidate([(candidate, score)])
    assert best_candidate.name == "only"
    assert best_score is score


def test_choose_best_candidate_tied_scores_prefers_first():
    c1 = TuningCandidate(name="first", rationale="first", overrides={})
    c2 = TuningCandidate(name="second", rationale="second", overrides={})
    score = FrontierScore(
        treatment_arm="active", n_pairs=5, n_complete=5, n_help=2,
        n_help_no_extra_inst_loss=1, n_help_nonnegative_net_system=1,
        n_safe_help=1, share_help=0.4, share_help_no_extra_inst_loss=0.2,
        share_help_nonnegative_net_system=0.2, share_safe_help=0.2,
        mean_default_relief=0.05, mean_safe_default_relief=0.03,
        mean_inst_loss_shift=0.0, mean_net_system_relief=0.01,
        mean_loss_per_help=0.1,
    )
    best_candidate, _ = choose_best_candidate([(c1, score), (c2, score)])
    assert best_candidate.name == "first"


def test_bank_candidates_factory():
    from bilancio.analysis.intermediary_support_tuning import bank_candidates
    candidates = bank_candidates()
    assert candidates[0].name == "baseline"
    assert len(candidates) >= 2
