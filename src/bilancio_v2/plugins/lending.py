"""Non-bank lender phase (Subphase B_Lending) and loan servicing.

The lender screens non-defaulted households/firms for liquidity shortfalls
over its horizon, prices loans from an observed default probability
(kappa-aware profile or signal-based), ranks opportunities (profit, cascade,
or blended), and disburses cash loans subject to exposure limits,
concentration limits, coverage screens, and expected-loss budgets.
Preventive lending extends credit to agents whose *receivables* are at risk.
Matured loans are serviced in Phase C: repaid from borrower liquidity or
written off (the lender absorbs the loss — no cascade).

Observable behavior matches ``clean_core_lending_phase`` exactly, including
the duplicate ``NonBankLoanCreated`` events (one phase-tagged at creation,
one phase-less decision record appended at the end of the phase) and the
event-less cash movement on disbursal and cash repayment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from bilancio_v2.ledger import ZERO, InsufficientFundsError, Ledger, NonBankLoan
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.plugins.settlement import pay_with_deposit
from bilancio_v2.subsystem_config import CleanBankingConfig, CleanLenderConfig


@dataclass(frozen=True)
class LendingPhase:
    name = "SubphaseB_Lending"
    config: CleanLenderConfig

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        run_lending_phase(ledger, self.config, banking_config=ctx.banking_config)
        # Lending decisions never count toward the stability impact signal,
        # matching the existing engine (which discards this phase's result).
        return False


# ---------------------------------------------------------------------------
# Phase B_Lending
# ---------------------------------------------------------------------------


def run_lending_phase(
    ledger: Ledger,
    config: CleanLenderConfig,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> bool:
    lender_id = active_lender_id(ledger)
    if lender_id is None:
        return False
    if config.adaptive_capital_conservation:
        config = apply_lender_capital_conservation(ledger, config, lender_id)

    lender_liquid = ledger.agent_liquid_assets(lender_id)
    performing_exposure = sum(
        (
            loan.amount
            for loan in ledger.non_bank_loans
            if not loan.settled and loan.lender == lender_id and loan.borrower not in ledger.defaulted_agent_ids
        ),
        ZERO,
    )
    initial_capital = lender_liquid + performing_exposure
    if initial_capital <= ZERO:
        return False

    existing_exposure = sum(
        (loan.amount for loan in ledger.non_bank_loans if not loan.settled and loan.lender == lender_id),
        ZERO,
    )
    max_total = Decimal(int(initial_capital * config.max_total_exposure))
    available = min(lender_liquid, max_total - existing_exposure)
    if available <= ZERO:
        return False

    if config.stop_loss_realized_ratio > ZERO:
        realized_loss = realized_non_bank_loan_loss(ledger)
        realized_ratio = realized_loss / max(initial_capital, Decimal("1"))
        if realized_ratio >= config.stop_loss_realized_ratio:
            ledger.log_raw(
                "NonBankLendingPausedStopLoss",
                lender_id=lender_id,
                realized_loss=str(realized_loss),
                realized_ratio=str(realized_ratio),
                threshold=str(config.stop_loss_realized_ratio),
            )
            return False

    opportunities = collect_lending_opportunities(ledger, config, lender_id, initial_capital, banking_config=banking_config)
    rank_lending_opportunities(opportunities, config)

    remaining = available
    created = False
    decision_events: list[tuple[str, dict[str, Any]]] = []
    daily_expected_loss_spent = ZERO
    daily_expected_loss_cap = (
        initial_capital * config.daily_expected_loss_budget_ratio if config.daily_expected_loss_budget_ratio > ZERO else None
    )
    run_expected_loss_cap = (
        initial_capital * config.run_expected_loss_budget_ratio if config.run_expected_loss_budget_ratio > ZERO else None
    )
    for opportunity in opportunities:
        if remaining <= ZERO:
            break
        amount = min(opportunity["amount"], remaining)
        if amount <= ZERO:
            continue

        expected_loss = Decimal(str(opportunity.get("expected_loss", 0.0)))
        if opportunity["amount"] > ZERO:
            expected_loss = expected_loss * amount / Decimal(str(opportunity["amount"]))

        if daily_expected_loss_cap is not None and daily_expected_loss_spent + expected_loss > daily_expected_loss_cap:
            decision_events.append(
                (
                    "NonBankLoanRejectedBudget",
                    {
                        "lender_id": lender_id,
                        "borrower_id": opportunity["borrower_id"],
                        "scope": "daily",
                        "expected_loss": str(expected_loss),
                        "budget_cap": str(daily_expected_loss_cap),
                        "budget_used": str(daily_expected_loss_spent),
                    },
                )
            )
            continue

        if run_expected_loss_cap is not None and ledger.lender_run_expected_loss_spent + expected_loss > run_expected_loss_cap:
            decision_events.append(
                (
                    "NonBankLoanRejectedBudget",
                    {
                        "lender_id": lender_id,
                        "borrower_id": opportunity["borrower_id"],
                        "scope": "run",
                        "expected_loss": str(expected_loss),
                        "budget_cap": str(run_expected_loss_cap),
                        "budget_used": str(ledger.lender_run_expected_loss_spent),
                    },
                )
            )
            continue

        if config.max_loans_per_borrower_per_day > 0:
            borrower_id = opportunity["borrower_id"]
            count = count_existing_non_bank_loans(ledger, lender_id, borrower_id)
            if count >= config.max_loans_per_borrower_per_day:
                ledger.log_raw(
                    "NonBankLoanRejectedConcentration",
                    lender_id=lender_id,
                    borrower_id=borrower_id,
                    count=count,
                    limit=config.max_loans_per_borrower_per_day,
                )
                continue

        rate, maturity_days = resolve_non_bank_loan_terms(ledger, config, opportunity)
        loan = create_non_bank_loan(
            ledger,
            lender_id=lender_id,
            borrower_id=opportunity["borrower_id"],
            amount=amount,
            rate=rate,
            maturity_days=maturity_days,
            banking_config=banking_config,
        )
        remaining -= amount
        created = True
        decision_events.append(
            (
                "NonBankLoanCreated",
                {
                    "lender_id": lender_id,
                    "borrower_id": opportunity["borrower_id"],
                    "amount": amount,
                    "rate": str(rate),
                    "loan_id": loan.id,
                    "p_default": str(opportunity["p_default"]),
                },
            )
        )
        daily_expected_loss_spent += expected_loss
        ledger.lender_run_expected_loss_spent += expected_loss

    if config.preventive_lending and remaining > ZERO and config.kappa is not None:
        preventive = collect_preventive_lending_opportunities(ledger, config, lender_id, initial_capital)
        preventive.sort(
            key=lambda item: int(item.get("downstream", 0)) * (1.0 - float(item["p_default"])),
            reverse=True,
        )
        remaining, daily_expected_loss_spent, preventive_created = execute_preventive_lending_opportunities(
            ledger,
            config,
            lender_id,
            preventive,
            initial_capital,
            remaining,
            daily_expected_loss_spent,
            decision_events,
        )
        created = created or preventive_created

    for kind, data in decision_events:
        ledger.log_raw(kind, **data)
    return created


def apply_lender_capital_conservation(ledger: Ledger, config: CleanLenderConfig, lender_id: str) -> CleanLenderConfig:
    total_assets = ledger.cash[lender_id]
    total_loans = ZERO
    for loan in ledger.non_bank_loans:
        if loan.settled or loan.lender != lender_id:
            continue
        total_assets += loan.amount
        total_loans += loan.amount
    if total_assets <= ZERO:
        return config

    utilization = total_loans / total_assets
    conservation = max(Decimal("0.2"), Decimal("1") - utilization)
    return replace(
        config,
        max_single_exposure=config.max_single_exposure * conservation,
        max_total_exposure=config.max_total_exposure * conservation,
    )


def realized_non_bank_loan_loss(ledger: Ledger) -> Decimal:
    total = ZERO
    for event in ledger.journal:
        if event.kind != "NonBankLoanDefaulted":
            continue
        amount_owed = Decimal(str(event.data.get("amount_owed", 0)))
        cash_available = Decimal(str(event.data.get("cash_available", 0)))
        total += max(ZERO, amount_owed - cash_available)
    return total


def collect_lending_opportunities(
    ledger: Ledger,
    config: CleanLenderConfig,
    lender_id: str,
    initial_capital: Decimal,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    for agent_id, agent in ledger.agents.items():
        if agent_id in ledger.defaulted_agent_ids:
            continue
        if agent.kind not in ("household", "firm"):
            continue

        observed = observe_lender_counterparty_liquidity(ledger, config, agent_id)
        if observed is None:
            continue
        upcoming_due, liquid = observed
        shortfall = upcoming_due - liquid
        if shortfall < config.min_shortfall:
            continue

        p_for_screen = lender_observed_default_probability(ledger, config, agent_id)
        if p_for_screen > config.max_default_prob:
            continue
        p_default = (
            lender_profile_default_probability(ledger, config, agent_id, upcoming_due, liquid) if config.kappa is not None else p_for_screen
        )

        max_single = Decimal(int(config.max_single_exposure * initial_capital))
        coverage_rate_penalty = ZERO
        if config.min_coverage_ratio > ZERO:
            coverage = assess_non_bank_borrower(
                ledger,
                agent_id,
                min(shortfall, max_single),
                config.profit_target if config.kappa is not None else config.base_rate,
                config.horizon,
            )
            if config.coverage_mode == "graduated" and coverage >= Decimal("-1"):
                if coverage < config.min_coverage_ratio:
                    coverage_rate_penalty = config.coverage_penalty_scale * (config.min_coverage_ratio - coverage)
            elif coverage < config.min_coverage_ratio:
                ledger.log_raw(
                    "NonBankLoanRejectedCoverage",
                    lender_id=lender_id,
                    borrower_id=agent_id,
                    coverage=str(coverage),
                    min_coverage=str(config.min_coverage_ratio),
                )
                continue

        borrower_existing = sum(
            (loan.amount for loan in ledger.non_bank_loans if not loan.settled and loan.lender == lender_id and loan.borrower == agent_id),
            ZERO,
        )
        max_to_this_borrower = max_single - borrower_existing
        if max_to_this_borrower <= ZERO:
            continue

        loan_amount = min(shortfall, max_to_this_borrower)
        if config.collateralized_terms:
            collateral_value = quality_adjusted_receivables(ledger, agent_id, config.horizon)
            collateral_cap = Decimal(int(collateral_value * config.collateral_advance_rate))
            loan_amount = min(loan_amount, collateral_cap)
            if loan_amount <= ZERO:
                continue

        expected_loss = loan_amount * p_default
        expected_relief = shortfall * (Decimal("1") - p_default)
        if config.marginal_relief_min_ratio > ZERO:
            ratio = expected_relief / expected_loss if expected_loss > ZERO else Decimal("999")
            if ratio < config.marginal_relief_min_ratio:
                ledger.log_raw(
                    "NonBankLoanRejectedMarginalBenefit",
                    lender_id=lender_id,
                    borrower_id=agent_id,
                    expected_relief=str(expected_relief),
                    expected_loss=str(expected_loss),
                    ratio=str(ratio),
                    threshold=str(config.marginal_relief_min_ratio),
                )
                continue

        rate = lender_loan_rate(config, p_default, banking_config=banking_config) + coverage_rate_penalty
        opportunities.append(
            {
                "borrower_id": agent_id,
                "amount": loan_amount,
                "rate": rate,
                "p_default": p_default,
                "expected_profit": float(rate) * (1.0 - float(p_default)),
                "expected_loss": float(expected_loss),
                "expected_relief": float(expected_relief),
                "downstream": downstream_obligation_total(ledger, agent_id),
                "coverage_ratio": max(liquid, ZERO) / max(upcoming_due, Decimal("1")),
            }
        )
    return opportunities


def collect_preventive_lending_opportunities(
    ledger: Ledger,
    config: CleanLenderConfig,
    lender_id: str,
    initial_capital: Decimal,
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    for agent_id, agent in ledger.agents.items():
        if agent_id in ledger.defaulted_agent_ids:
            continue
        if agent.kind not in ("household", "firm"):
            continue

        observed = observe_lender_counterparty_liquidity(ledger, config, agent_id)
        if observed is None:
            continue
        upcoming_due, liquid = observed
        shortfall = upcoming_due - liquid
        if shortfall >= config.min_shortfall:
            continue

        at_risk = receivables_at_risk(ledger, config, agent_id, config.horizon, config.prevention_threshold)
        if at_risk <= ZERO:
            continue

        p_default = lender_observed_default_probability(ledger, config, agent_id)
        if p_default > config.max_default_prob:
            continue

        borrower_existing = sum(
            (loan.amount for loan in ledger.non_bank_loans if not loan.settled and loan.lender == lender_id and loan.borrower == agent_id),
            ZERO,
        )
        max_single = Decimal(int(config.max_single_exposure * initial_capital))
        max_to_this_borrower = max_single - borrower_existing
        if max_to_this_borrower <= ZERO:
            continue

        if config.max_loans_per_borrower_per_day > 0:
            count = count_existing_non_bank_loans(ledger, lender_id, agent_id)
            if count >= config.max_loans_per_borrower_per_day:
                continue

        loan_amount = min(at_risk, max_to_this_borrower)
        if config.collateralized_terms:
            collateral_value = quality_adjusted_receivables(ledger, agent_id, config.horizon)
            collateral_cap = Decimal(int(collateral_value * config.collateral_advance_rate))
            loan_amount = min(loan_amount, collateral_cap)
            if loan_amount <= ZERO:
                continue

        rate = preventive_lender_loan_rate(config, p_default)
        expected_loss = loan_amount * p_default
        expected_relief = at_risk * (Decimal("1") - p_default)
        if config.marginal_relief_min_ratio > ZERO:
            ratio = expected_relief / expected_loss if expected_loss > ZERO else Decimal("999")
            if ratio < config.marginal_relief_min_ratio:
                continue

        opportunities.append(
            {
                "borrower_id": agent_id,
                "amount": loan_amount,
                "rate": rate,
                "p_default": p_default,
                "expected_profit": float(rate) * (1.0 - float(p_default)),
                "expected_loss": float(expected_loss),
                "expected_relief": float(expected_relief),
                "shortfall": ZERO,
                "downstream": downstream_obligation_total(ledger, agent_id),
                "coverage_ratio": max(liquid, ZERO) / max(upcoming_due, Decimal("1")),
                "preventive": True,
            }
        )
    return opportunities


def execute_preventive_lending_opportunities(
    ledger: Ledger,
    config: CleanLenderConfig,
    lender_id: str,
    opportunities: list[dict[str, Any]],
    initial_capital: Decimal,
    remaining_capital: Decimal,
    daily_expected_loss_spent: Decimal,
    decision_events: list[tuple[str, dict[str, Any]]],
    *,
    banking_config: CleanBankingConfig | None = None,
) -> tuple[Decimal, Decimal, bool]:
    daily_expected_loss_cap = (
        initial_capital * config.daily_expected_loss_budget_ratio if config.daily_expected_loss_budget_ratio > ZERO else None
    )
    run_expected_loss_cap = (
        initial_capital * config.run_expected_loss_budget_ratio if config.run_expected_loss_budget_ratio > ZERO else None
    )
    created = False

    for opportunity in opportunities:
        if remaining_capital <= ZERO:
            break
        amount = min(opportunity["amount"], remaining_capital)
        if amount <= ZERO:
            continue

        expected_loss = Decimal(str(opportunity.get("expected_loss", 0.0)))
        if opportunity["amount"] > ZERO:
            expected_loss = expected_loss * amount / Decimal(str(opportunity["amount"]))

        if daily_expected_loss_cap is not None and daily_expected_loss_spent + expected_loss > daily_expected_loss_cap:
            decision_events.append(
                (
                    "NonBankLoanRejectedBudget",
                    {
                        "lender_id": lender_id,
                        "borrower_id": opportunity["borrower_id"],
                        "scope": "daily",
                        "expected_loss": str(expected_loss),
                        "budget_cap": str(daily_expected_loss_cap),
                        "budget_used": str(daily_expected_loss_spent),
                        "preventive": True,
                    },
                )
            )
            continue

        if run_expected_loss_cap is not None and ledger.lender_run_expected_loss_spent + expected_loss > run_expected_loss_cap:
            decision_events.append(
                (
                    "NonBankLoanRejectedBudget",
                    {
                        "lender_id": lender_id,
                        "borrower_id": opportunity["borrower_id"],
                        "scope": "run",
                        "expected_loss": str(expected_loss),
                        "budget_cap": str(run_expected_loss_cap),
                        "budget_used": str(ledger.lender_run_expected_loss_spent),
                        "preventive": True,
                    },
                )
            )
            continue

        maturity_days = resolve_preventive_non_bank_loan_maturity(ledger, config, opportunity["borrower_id"], opportunity["p_default"])
        loan = ledger.disburse_non_bank_loan(
            lender_id=lender_id,
            borrower_id=opportunity["borrower_id"],
            amount=amount,
            rate=opportunity["rate"],
            maturity_days=maturity_days,
        )
        remaining_capital -= amount
        created = True
        decision_events.append(
            (
                "NonBankLoanCreatedPreventive",
                {
                    "lender_id": lender_id,
                    "borrower_id": opportunity["borrower_id"],
                    "amount": amount,
                    "rate": str(opportunity["rate"]),
                    "loan_id": loan.id,
                    "p_default": str(opportunity["p_default"]),
                    "at_risk_receivables": opportunity["amount"],
                },
            )
        )
        daily_expected_loss_spent += expected_loss
        ledger.lender_run_expected_loss_spent += expected_loss

    return remaining_capital, daily_expected_loss_spent, created


def create_non_bank_loan(
    ledger: Ledger,
    *,
    lender_id: str,
    borrower_id: str,
    amount: Decimal,
    rate: Decimal,
    maturity_days: int,
    banking_config: CleanBankingConfig | None = None,
) -> NonBankLoan:
    """Disburse a loan: routed deposits in banking mode, silent cash otherwise."""
    from bilancio.core.errors import DefaultError
    from bilancio_v2.plugins.banking import agent_deposits_total
    from bilancio_v2.plugins.settlement import pay_with_routed_deposits

    if banking_config is not None and agent_deposits_total(ledger, lender_id) > ZERO:
        paid = pay_with_routed_deposits(ledger, lender_id, borrower_id, amount, banking_config)
        if paid != amount:
            raise DefaultError(f"Insufficient deposits to create non-bank loan from {lender_id}: {amount - paid} still needed")
        return ledger.record_non_bank_loan(
            lender_id=lender_id,
            borrower_id=borrower_id,
            amount=amount,
            rate=rate,
            maturity_days=maturity_days,
        )
    return ledger.disburse_non_bank_loan(
        lender_id=lender_id,
        borrower_id=borrower_id,
        amount=amount,
        rate=rate,
        maturity_days=maturity_days,
    )


# ---------------------------------------------------------------------------
# Phase C servicing
# ---------------------------------------------------------------------------


def repay_due_non_bank_loans(ledger: Ledger) -> None:
    for loan in list(ledger.non_bank_loans):
        if loan.settled or ledger.day < loan.maturity_day:
            continue
        borrower_liquid = ledger.agent_liquid_assets(loan.borrower)
        repayment = loan.repayment_amount
        if borrower_liquid < repayment:
            ledger.default_non_bank_loan(loan, borrower_liquid=borrower_liquid)
            continue

        if ledger.cash[loan.borrower] >= repayment:
            ledger.repay_non_bank_loan_with_cash(loan)
        else:
            paid = pay_with_deposit(ledger, loan.borrower, loan.lender, repayment)
            if paid < repayment:
                raise InsufficientFundsError(f"insufficient {loan.borrower} liquid assets: required {repayment}, available {paid}")
            ledger.mark_non_bank_loan_repaid(loan)


# ---------------------------------------------------------------------------
# Lender analysis (pure reads)
# ---------------------------------------------------------------------------


def active_lender_id(ledger: Ledger) -> str | None:
    for agent_id, agent in ledger.agents.items():
        if agent.kind == "non_bank_lender" and agent_id not in ledger.defaulted_agent_ids:
            return agent_id
    return None


def rank_lending_opportunities(opportunities: list[dict[str, Any]], config: CleanLenderConfig) -> None:
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
            float(config.cascade_weight) * cascade_score + (1.0 - float(config.cascade_weight)) * opportunity["expected_profit"]
        )
    opportunities.sort(key=lambda item: item["blended_score"], reverse=True)


def lending_cascade_score(opportunity: dict[str, Any], max_downstream: int) -> float:
    coverage = Decimal(str(opportunity.get("coverage_ratio", Decimal("0.5"))))
    norm_downstream = int(opportunity.get("downstream", 0)) / max_downstream
    return float(coverage) * norm_downstream * (1.0 - float(opportunity["p_default"]))


def upcoming_obligations(ledger: Ledger, agent_id: str, horizon: int, *, include_bank_loans: bool = True) -> Decimal:
    latest_day = ledger.day + horizon
    total = ZERO
    for payable in ledger.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        if ledger.day <= payable.due_day <= latest_day:
            total += payable.amount
    for loan in ledger.non_bank_loans:
        if loan.settled or loan.borrower != agent_id:
            continue
        if ledger.day <= loan.maturity_day <= latest_day:
            total += loan.repayment_amount
    if include_bank_loans:
        for bank_loan in ledger.bank_loans:
            if bank_loan.settled or bank_loan.borrower != agent_id:
                continue
            if ledger.day <= bank_loan.maturity_day <= latest_day:
                total += bank_loan.repayment_amount
    return total


def quality_adjusted_receivables(ledger: Ledger, agent_id: str, horizon: int) -> Decimal:
    latest_day = ledger.day + horizon
    total = ZERO
    for payable in ledger.payables:
        if payable.settled or payable.creditor != agent_id:
            continue
        if payable.debtor in ledger.defaulted_agent_ids:
            continue
        if ledger.day <= payable.due_day <= latest_day:
            total += payable.amount
    return total


def assess_non_bank_borrower(ledger: Ledger, agent_id: str, loan_amount: Decimal, rate: Decimal, horizon: int) -> Decimal:
    repayment = Decimal(int(loan_amount * (Decimal("1") + rate)))
    if repayment <= ZERO:
        return Decimal("999")

    liquid = ledger.agent_liquid_assets(agent_id)
    receivables = quality_adjusted_receivables(ledger, agent_id, horizon)
    obligations = upcoming_obligations(ledger, agent_id, horizon, include_bank_loans=False)
    return (liquid + receivables - obligations) / repayment


def lender_uses_information(config: CleanLenderConfig) -> bool:
    return (
        config.info_cash_visibility != "perfect"
        or config.info_liabilities_visibility != "perfect"
        or config.info_history_visibility != "perfect"
    )


def observe_lender_counterparty_liquidity(ledger: Ledger, config: CleanLenderConfig, agent_id: str) -> tuple[Decimal, Decimal] | None:
    if not lender_uses_information(config):
        return (
            upcoming_obligations(ledger, agent_id, config.horizon, include_bank_loans=False),
            ledger.agent_liquid_assets(agent_id),
        )

    if config.info_liabilities_visibility == "none":
        return None
    upcoming_due = upcoming_obligations(ledger, agent_id, config.horizon, include_bank_loans=False)

    if config.info_cash_visibility == "none":
        liquid = ZERO
    elif config.info_cash_visibility == "noisy":
        liquid = max(
            ZERO,
            lender_noisy_decimal(ledger, agent_id, ledger.cash[agent_id], "cash", config.info_cash_noise),
        )
    else:
        liquid = ledger.cash[agent_id]
    return upcoming_due, liquid


def lender_observed_default_probability(ledger: Ledger, config: CleanLenderConfig, agent_id: str) -> Decimal:
    if not lender_uses_information(config):
        return lender_signal_default_probability(ledger, agent_id)
    if config.info_history_visibility == "none":
        return config.initial_prior

    raw = lender_signal_default_probability(ledger, agent_id)
    if config.info_history_visibility == "noisy":
        return max(ZERO, min(Decimal("1"), raw * config.info_history_sample_rate))
    return raw


def lender_noisy_decimal(ledger: Ledger, agent_id: str, value: Decimal, channel: str, error_fraction: Decimal) -> Decimal:
    seed = ledger.day * 131_071 + sum(ord(ch) for ch in agent_id) * 389 + sum(ord(ch) for ch in channel) + len(ledger.non_bank_loans) * 37
    rng = random.Random(seed)
    sigma = float(abs(value) * error_fraction)
    noisy = float(value) + rng.gauss(0, max(sigma, 0.01))
    return Decimal(int(round(noisy)))


def lender_profile_default_probability(
    ledger: Ledger,
    config: CleanLenderConfig,
    agent_id: str,
    upcoming_due: Decimal,
    liquid: Decimal,
) -> Decimal:
    assert config.kappa is not None
    base_default_estimate = Decimal("1") / (Decimal("1") + config.kappa)
    receivables = quality_adjusted_receivables(ledger, agent_id, config.planning_horizon)
    resources = max(liquid + receivables, ZERO)
    coverage = resources / max(upcoming_due, Decimal("1"))
    p_default = base_default_estimate * (Decimal("1") / max(coverage, Decimal("0.01")))
    return max(Decimal("0.01"), min(Decimal("0.95"), p_default))


def lender_signal_default_probability(ledger: Ledger, agent_id: str) -> Decimal:
    if agent_id in ledger.defaulted_agent_ids:
        return Decimal("1.0")
    subsystem = ledger.dealer_subsystem
    risk_assessor = getattr(subsystem, "risk_assessor", None)
    if risk_assessor is not None:
        p_default = risk_assessor.estimate_default_prob(agent_id, ledger.day)
        if p_default is not None:
            return Decimal(str(p_default))
    if ledger.dealer_config is not None and ledger.dealer_config.risk_enabled:
        if ledger.dealer_config.kappa is not None:
            from bilancio.dealer.priors import kappa_informed_prior

            return Decimal(kappa_informed_prior(ledger.dealer_config.kappa))
        return Decimal(ledger.dealer_config.initial_prior)
    if ledger.rating_registry:
        return ledger.rating_registry.get(agent_id, Decimal("0.15"))
    base_rate = Decimal(len(ledger.defaulted_agent_ids)) / Decimal(max(len(ledger.agents), 1))
    return max(Decimal("0.01"), min(Decimal("0.99"), base_rate + Decimal("0.05")))


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
        from bilancio_v2.plugins.banking import bank_profile

        profile = bank_profile(banking_config)
        r_floor = profile.r_floor(banking_config.kappa)
        omega = profile.corridor_width(banking_config.kappa)
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


def resolve_non_bank_loan_terms(ledger: Ledger, config: CleanLenderConfig, opportunity: dict[str, Any]) -> tuple[Decimal, int]:
    rate = Decimal(str(opportunity["rate"]))
    maturity_days = config.maturity_days

    if config.kappa is not None:
        maturity_days = min(
            config.max_loan_maturity,
            config.max_ring_maturity or config.max_loan_maturity,
        )

    if config.maturity_matching and config.kappa is not None:
        nearest_day = nearest_receivable_day(ledger, opportunity["borrower_id"], max_horizon=config.max_loan_maturity)
        if nearest_day is not None:
            matched = nearest_day - ledger.day + 1
            maturity_days = max(config.min_loan_maturity, min(matched, config.max_loan_maturity))
            if maturity_days > 2:
                rate = rate * (Decimal("1") + Decimal("0.01") * Decimal(str(maturity_days - 2)))

    if opportunity["p_default"] >= config.high_risk_default_threshold:
        maturity_days = min(maturity_days, config.high_risk_maturity_cap)

    return rate, maturity_days


def resolve_preventive_non_bank_loan_maturity(ledger: Ledger, config: CleanLenderConfig, borrower_id: str, p_default: Decimal) -> int:
    maturity_days = config.max_loan_maturity
    if config.maturity_matching:
        nearest_day = nearest_receivable_day(ledger, borrower_id, max_horizon=config.max_loan_maturity)
        if nearest_day is not None:
            matched = nearest_day - ledger.day + 1
            maturity_days = max(config.min_loan_maturity, min(matched, config.max_loan_maturity))
    if p_default >= config.high_risk_default_threshold:
        maturity_days = min(maturity_days, config.high_risk_maturity_cap)
    return maturity_days


def nearest_receivable_day(ledger: Ledger, agent_id: str, *, max_horizon: int) -> int | None:
    nearest: int | None = None
    for payable in ledger.payables:
        if payable.settled or payable.creditor != agent_id:
            continue
        if payable.debtor in ledger.defaulted_agent_ids:
            continue
        if payable.due_day <= ledger.day:
            continue
        if payable.due_day > ledger.day + max_horizon:
            continue
        if nearest is None or payable.due_day < nearest:
            nearest = payable.due_day
    return nearest


def downstream_obligation_total(ledger: Ledger, agent_id: str) -> Decimal:
    return sum(
        (
            payable.amount
            for payable in ledger.payables
            if not payable.settled and payable.debtor == agent_id and payable.due_day > ledger.day
        ),
        ZERO,
    )


def receivables_at_risk(
    ledger: Ledger,
    config: CleanLenderConfig,
    agent_id: str,
    horizon: int,
    threshold: Decimal,
) -> Decimal:
    latest_day = ledger.day + horizon
    total = ZERO
    for payable in ledger.payables:
        if payable.settled or payable.creditor != agent_id:
            continue
        if payable.due_day <= ledger.day or payable.due_day > latest_day:
            continue
        p_default = lender_observed_default_probability(ledger, config, payable.debtor)
        if p_default >= threshold:
            total += payable.amount
    return total


def count_existing_non_bank_loans(ledger: Ledger, lender_id: str, borrower_id: str) -> int:
    return sum(1 for loan in ledger.non_bank_loans if not loan.settled and loan.lender == lender_id and loan.borrower == borrower_id)
