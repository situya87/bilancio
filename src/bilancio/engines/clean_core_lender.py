"""Non-bank lender analysis helpers for the clean-core scenario engine."""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

from bilancio.decision.profiles import BankProfile
from bilancio.engines.clean_core_types import (
    ZERO,
    CleanBankingConfig,
    CleanLenderConfig,
    CleanState,
)


def rank_lending_opportunities(
    opportunities: list[dict[str, Any]],
    config: CleanLenderConfig,
) -> None:
    if config.ranking_mode == "profit":
        opportunities.sort(key=lambda item: item["expected_profit"], reverse=True)
        return

    max_downstream = max((int(item.get("downstream", 0)) for item in opportunities), default=1) or 1
    if config.ranking_mode == "cascade":
        for opportunity in opportunities:
            opportunity["cascade_score"] = lending_cascade_score(opportunity, max_downstream)
        opportunities.sort(key=lambda item: item["cascade_score"], reverse=True)
        return

    for opportunity in opportunities:
        cascade_score = lending_cascade_score(opportunity, max_downstream)
        opportunity["blended_score"] = (
            float(config.cascade_weight) * cascade_score
            + (1.0 - float(config.cascade_weight)) * opportunity["expected_profit"]
        )
    opportunities.sort(key=lambda item: item["blended_score"], reverse=True)


def lending_cascade_score(opportunity: dict[str, Any], max_downstream: int) -> float:
    coverage = Decimal(str(opportunity.get("coverage_ratio", Decimal("0.5"))))
    norm_downstream = int(opportunity.get("downstream", 0)) / max_downstream
    return float(coverage) * norm_downstream * (1.0 - float(opportunity["p_default"]))


def active_lender_id(state: CleanState) -> str | None:
    for agent_id, agent in state.agents.items():
        if agent.kind == "non_bank_lender" and agent_id not in state.defaulted_agent_ids:
            return agent_id
    return None


def agent_liquid_assets(state: CleanState, agent_id: str) -> Decimal:
    deposits = sum(
        amount
        for (customer_id, _bank_id), amount in state.deposits.items()
        if customer_id == agent_id
    )
    return state.cash[agent_id] + deposits


def upcoming_obligations(
    state: CleanState,
    agent_id: str,
    horizon: int,
    *,
    include_bank_loans: bool = True,
) -> Decimal:
    latest_day = state.day + horizon
    total = ZERO
    for payable in state.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        if state.day <= payable.due_day <= latest_day:
            total += payable.amount
    for loan in state.non_bank_loans:
        if loan.settled or loan.borrower != agent_id:
            continue
        if state.day <= loan.maturity_day <= latest_day:
            total += loan.repayment_amount
    if include_bank_loans:
        for loan in state.bank_loans:
            if loan.settled or loan.borrower != agent_id:
                continue
            if state.day <= loan.maturity_day <= latest_day:
                total += loan.repayment_amount
    return total


def quality_adjusted_receivables(
    state: CleanState,
    agent_id: str,
    horizon: int,
) -> Decimal:
    latest_day = state.day + horizon
    total = ZERO
    for payable in state.payables:
        if payable.settled or payable.creditor != agent_id:
            continue
        if payable.debtor in state.defaulted_agent_ids:
            continue
        if state.day <= payable.due_day <= latest_day:
            total += payable.amount
    return total


def assess_non_bank_borrower(
    state: CleanState,
    agent_id: str,
    loan_amount: Decimal,
    rate: Decimal,
    horizon: int,
) -> Decimal:
    repayment = Decimal(int(loan_amount * (Decimal("1") + rate)))
    if repayment <= ZERO:
        return Decimal("999")

    liquid = agent_liquid_assets(state, agent_id)
    receivables = quality_adjusted_receivables(state, agent_id, horizon)
    obligations = upcoming_obligations(
        state,
        agent_id,
        horizon,
        include_bank_loans=False,
    )
    return (liquid + receivables - obligations) / repayment


def lender_uses_information(config: CleanLenderConfig) -> bool:
    return (
        config.info_cash_visibility != "perfect"
        or config.info_liabilities_visibility != "perfect"
        or config.info_history_visibility != "perfect"
    )


def observe_lender_counterparty_liquidity(
    state: CleanState,
    config: CleanLenderConfig,
    agent_id: str,
) -> tuple[Decimal, Decimal] | None:
    if not lender_uses_information(config):
        return (
            upcoming_obligations(
                state,
                agent_id,
                config.horizon,
                include_bank_loans=False,
            ),
            agent_liquid_assets(state, agent_id),
        )

    if config.info_liabilities_visibility == "none":
        return None
    upcoming_due = upcoming_obligations(
        state,
        agent_id,
        config.horizon,
        include_bank_loans=False,
    )

    if config.info_cash_visibility == "none":
        liquid = ZERO
    elif config.info_cash_visibility == "noisy":
        liquid = max(
            ZERO,
            lender_noisy_decimal(
                state,
                agent_id,
                state.cash[agent_id],
                "cash",
                config.info_cash_noise,
            ),
        )
    else:
        liquid = state.cash[agent_id]
    return upcoming_due, liquid


def lender_observed_default_probability(
    state: CleanState,
    config: CleanLenderConfig,
    agent_id: str,
) -> Decimal:
    if not lender_uses_information(config):
        return lender_signal_default_probability(state, agent_id)
    if config.info_history_visibility == "none":
        return config.initial_prior

    raw = lender_signal_default_probability(state, agent_id)
    if config.info_history_visibility == "noisy":
        return max(ZERO, min(Decimal("1"), raw * config.info_history_sample_rate))
    return raw


def lender_noisy_decimal(
    state: CleanState,
    agent_id: str,
    value: Decimal,
    channel: str,
    error_fraction: Decimal,
) -> Decimal:
    seed = (
        state.day * 131_071
        + sum(ord(ch) for ch in agent_id) * 389
        + sum(ord(ch) for ch in channel)
        + len(state.non_bank_loans) * 37
    )
    rng = random.Random(seed)
    sigma = float(abs(value) * error_fraction)
    noisy = float(value) + rng.gauss(0, max(sigma, 0.01))
    return Decimal(int(round(noisy)))


def lender_default_probability(
    state: CleanState,
    config: CleanLenderConfig,
    agent_id: str,
    upcoming_due: Decimal,
    liquid: Decimal,
) -> Decimal:
    if config.kappa is None:
        return lender_signal_default_probability(state, agent_id)

    return lender_profile_default_probability(state, config, agent_id, upcoming_due, liquid)


def lender_profile_default_probability(
    state: CleanState,
    config: CleanLenderConfig,
    agent_id: str,
    upcoming_due: Decimal,
    liquid: Decimal,
) -> Decimal:
    assert config.kappa is not None
    base_default_estimate = Decimal("1") / (Decimal("1") + config.kappa)
    receivables = quality_adjusted_receivables(state, agent_id, config.planning_horizon)
    resources = max(liquid + receivables, ZERO)
    coverage = resources / max(upcoming_due, Decimal("1"))
    p_default = base_default_estimate * (Decimal("1") / max(coverage, Decimal("0.01")))
    return max(Decimal("0.01"), min(Decimal("0.95"), p_default))


def lender_loan_rate(
    config: CleanLenderConfig,
    p_default: Decimal,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> Decimal:
    if config.kappa is not None:
        risk_premium_scale = Decimal("0.1") + config.risk_aversion * Decimal("0.4")
        rate = config.profit_target + risk_premium_scale * p_default
    else:
        rate = config.base_rate + config.risk_premium_scale * p_default

    if banking_config is not None:
        bank_profile = _clean_bank_profile(banking_config)
        r_floor = bank_profile.r_floor(banking_config.kappa)
        omega = bank_profile.corridor_width(banking_config.kappa)
        p_0 = Decimal("1") / (Decimal("1") + banking_config.kappa)
        if p_0 > ZERO:
            rate = r_floor + omega * (p_default / p_0)
        else:
            rate = r_floor + omega

    if config.stress_risk_premium_scale > ZERO:
        denom = max(Decimal("0.01"), Decimal("1") - p_default)
        convex_component = (p_default * p_default) / denom
        rate += config.stress_risk_premium_scale * convex_component
    return rate


def preventive_lender_loan_rate(config: CleanLenderConfig, p_default: Decimal) -> Decimal:
    if config.kappa is not None:
        risk_premium_scale = Decimal("0.1") + config.risk_aversion * Decimal("0.4")
        return config.profit_target + risk_premium_scale * p_default
    return config.base_rate + config.risk_premium_scale * p_default


def resolve_non_bank_loan_terms(
    state: CleanState,
    config: CleanLenderConfig,
    opportunity: dict[str, Any],
) -> tuple[Decimal, int]:
    rate = Decimal(str(opportunity["rate"]))
    maturity_days = config.maturity_days

    if config.kappa is not None:
        maturity_days = min(
            config.max_loan_maturity,
            config.max_ring_maturity or config.max_loan_maturity,
        )

    if config.maturity_matching and config.kappa is not None:
        nearest_day = nearest_receivable_day(
            state,
            opportunity["borrower_id"],
            max_horizon=config.max_loan_maturity,
        )
        if nearest_day is not None:
            matched = nearest_day - state.day + 1
            maturity_days = max(
                config.min_loan_maturity,
                min(matched, config.max_loan_maturity),
            )
            if maturity_days > 2:
                rate = rate * (
                    Decimal("1") + Decimal("0.01") * Decimal(str(maturity_days - 2))
                )

    if opportunity["p_default"] >= config.high_risk_default_threshold:
        maturity_days = min(maturity_days, config.high_risk_maturity_cap)

    return rate, maturity_days


def resolve_preventive_non_bank_loan_maturity(
    state: CleanState,
    config: CleanLenderConfig,
    borrower_id: str,
    p_default: Decimal,
) -> int:
    maturity_days = config.max_loan_maturity
    if config.maturity_matching:
        nearest_day = nearest_receivable_day(
            state,
            borrower_id,
            max_horizon=config.max_loan_maturity,
        )
        if nearest_day is not None:
            matched = nearest_day - state.day + 1
            maturity_days = max(
                config.min_loan_maturity,
                min(matched, config.max_loan_maturity),
            )
    if p_default >= config.high_risk_default_threshold:
        maturity_days = min(maturity_days, config.high_risk_maturity_cap)
    return maturity_days


def nearest_receivable_day(
    state: CleanState,
    agent_id: str,
    *,
    max_horizon: int,
) -> int | None:
    nearest: int | None = None
    for payable in state.payables:
        if payable.settled or payable.creditor != agent_id:
            continue
        if payable.debtor in state.defaulted_agent_ids:
            continue
        if payable.due_day <= state.day:
            continue
        if payable.due_day > state.day + max_horizon:
            continue
        if nearest is None or payable.due_day < nearest:
            nearest = payable.due_day
    return nearest


def downstream_obligation_total(state: CleanState, agent_id: str) -> Decimal:
    return sum(
        (
            payable.amount
            for payable in state.payables
            if not payable.settled and payable.debtor == agent_id and payable.due_day > state.day
        ),
        ZERO,
    )


def receivables_at_risk(
    state: CleanState,
    config: CleanLenderConfig,
    agent_id: str,
    horizon: int,
    threshold: Decimal,
) -> Decimal:
    latest_day = state.day + horizon
    total = ZERO
    for payable in state.payables:
        if payable.settled or payable.creditor != agent_id:
            continue
        if payable.due_day <= state.day or payable.due_day > latest_day:
            continue
        p_default = lender_observed_default_probability(state, config, payable.debtor)
        if p_default >= threshold:
            total += payable.amount
    return total


def lender_signal_default_probability(state: CleanState, agent_id: str) -> Decimal:
    if agent_id in state.defaulted_agent_ids:
        return Decimal("1.0")
    subsystem = state.dealer_subsystem
    risk_assessor = getattr(subsystem, "risk_assessor", None)
    if risk_assessor is not None:
        p_default = risk_assessor.estimate_default_prob(agent_id, state.day)
        if p_default is not None:
            return Decimal(str(p_default))
    if state.dealer_config is not None and state.dealer_config.risk_enabled:
        if state.dealer_config.kappa is not None:
            from bilancio.dealer.priors import kappa_informed_prior

            return kappa_informed_prior(state.dealer_config.kappa)
        return state.dealer_config.initial_prior
    if state.rating_registry:
        return state.rating_registry.get(agent_id, Decimal("0.15"))
    base_rate = Decimal(len(state.defaulted_agent_ids)) / Decimal(max(len(state.agents), 1))
    return max(Decimal("0.01"), min(Decimal("0.99"), base_rate + Decimal("0.05")))


def count_existing_non_bank_loans(
    state: CleanState,
    lender_id: str,
    borrower_id: str,
) -> int:
    return sum(
        1
        for loan in state.non_bank_loans
        if not loan.settled and loan.lender == lender_id and loan.borrower == borrower_id
    )


def _clean_bank_profile(config: CleanBankingConfig) -> BankProfile:
    return BankProfile(
        reserve_target_ratio=config.reserve_target_ratio,
        credit_risk_loading=config.credit_risk_loading,
        max_borrower_risk=config.max_borrower_risk,
        min_coverage_ratio=config.min_coverage_ratio,
        adaptive_corridor=config.adaptive_corridor,
    )
