"""Operational non-bank lending phase for the clean-core engine."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

from bilancio.core.errors import DefaultError
from bilancio.engines.clean_core_banking import agent_deposits_total as _agent_deposits_total
from bilancio.engines.clean_core_cash import add_cash_lot as _add_cash_lot
from bilancio.engines.clean_core_cash import require_at_least as _require_at_least
from bilancio.engines.clean_core_cash import take_cash_lots as _take_cash_lots
from bilancio.engines.clean_core_lender import active_lender_id as _active_lender_id
from bilancio.engines.clean_core_lender import agent_liquid_assets as _agent_liquid_assets
from bilancio.engines.clean_core_lender import assess_non_bank_borrower as _assess_non_bank_borrower
from bilancio.engines.clean_core_lender import count_existing_non_bank_loans as _count_existing_non_bank_loans
from bilancio.engines.clean_core_lender import downstream_obligation_total as _downstream_obligation_total
from bilancio.engines.clean_core_lender import lender_loan_rate as _lender_loan_rate
from bilancio.engines.clean_core_lender import lender_observed_default_probability as _lender_observed_default_probability
from bilancio.engines.clean_core_lender import lender_profile_default_probability as _lender_profile_default_probability
from bilancio.engines.clean_core_lender import observe_lender_counterparty_liquidity as _observe_lender_counterparty_liquidity
from bilancio.engines.clean_core_lender import preventive_lender_loan_rate as _preventive_lender_loan_rate
from bilancio.engines.clean_core_lender import quality_adjusted_receivables as _quality_adjusted_receivables
from bilancio.engines.clean_core_lender import rank_lending_opportunities as _rank_lending_opportunities
from bilancio.engines.clean_core_lender import receivables_at_risk as _receivables_at_risk
from bilancio.engines.clean_core_lender import resolve_non_bank_loan_terms as _resolve_non_bank_loan_terms
from bilancio.engines.clean_core_lender import resolve_preventive_non_bank_loan_maturity as _resolve_preventive_non_bank_loan_maturity
from bilancio.engines.clean_core_settlement import _pay_with_deposit, _pay_with_routed_deposits
from bilancio.engines.clean_core_types import (
    ZERO,
    CleanBankingConfig,
    CleanLenderConfig,
    CleanNonBankLoan,
    CleanState,
)


def _run_lending_phase(
    state: CleanState,
    config: CleanLenderConfig,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> bool:
    lender_id = _active_lender_id(state)
    if lender_id is None:
        return False
    if config.adaptive_capital_conservation:
        config = _apply_lender_capital_conservation(state, config, lender_id)

    lender_liquid = _agent_liquid_assets(state, lender_id)
    performing_exposure = sum(
        loan.amount
        for loan in state.non_bank_loans
        if not loan.settled
        and loan.lender == lender_id
        and loan.borrower not in state.defaulted_agent_ids
    )
    initial_capital = lender_liquid + performing_exposure
    if initial_capital <= ZERO:
        return False

    existing_exposure = sum(
        loan.amount
        for loan in state.non_bank_loans
        if not loan.settled and loan.lender == lender_id
    )
    max_total = Decimal(int(initial_capital * config.max_total_exposure))
    available = min(lender_liquid, max_total - existing_exposure)
    if available <= ZERO:
        return False

    if config.stop_loss_realized_ratio > ZERO:
        realized_loss = _realized_non_bank_loan_loss(state)
        realized_ratio = realized_loss / max(initial_capital, Decimal("1"))
        if realized_ratio >= config.stop_loss_realized_ratio:
            state.events.append(
                {
                    "kind": "NonBankLendingPausedStopLoss",
                    "day": state.day,
                    "lender_id": lender_id,
                    "realized_loss": str(realized_loss),
                    "realized_ratio": str(realized_ratio),
                    "threshold": str(config.stop_loss_realized_ratio),
                }
            )
            return False

    opportunities = _collect_lending_opportunities(
        state,
        config,
        lender_id,
        initial_capital,
        banking_config=banking_config,
    )
    _rank_lending_opportunities(opportunities, config)

    remaining = available
    created = False
    lending_events: list[dict[str, Any]] = []
    daily_expected_loss_spent = ZERO
    daily_expected_loss_cap = (
        initial_capital * config.daily_expected_loss_budget_ratio
        if config.daily_expected_loss_budget_ratio > ZERO
        else None
    )
    run_expected_loss_cap = (
        initial_capital * config.run_expected_loss_budget_ratio
        if config.run_expected_loss_budget_ratio > ZERO
        else None
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

        if (
            daily_expected_loss_cap is not None
            and daily_expected_loss_spent + expected_loss > daily_expected_loss_cap
        ):
            lending_events.append(
                {
                    "kind": "NonBankLoanRejectedBudget",
                    "day": state.day,
                    "lender_id": lender_id,
                    "borrower_id": opportunity["borrower_id"],
                    "scope": "daily",
                    "expected_loss": str(expected_loss),
                    "budget_cap": str(daily_expected_loss_cap),
                    "budget_used": str(daily_expected_loss_spent),
                }
            )
            continue

        if (
            run_expected_loss_cap is not None
            and state.lender_run_expected_loss_spent + expected_loss > run_expected_loss_cap
        ):
            lending_events.append(
                {
                    "kind": "NonBankLoanRejectedBudget",
                    "day": state.day,
                    "lender_id": lender_id,
                    "borrower_id": opportunity["borrower_id"],
                    "scope": "run",
                    "expected_loss": str(expected_loss),
                    "budget_cap": str(run_expected_loss_cap),
                    "budget_used": str(state.lender_run_expected_loss_spent),
                }
            )
            continue

        if config.max_loans_per_borrower_per_day > 0:
            borrower_id = opportunity["borrower_id"]
            count = _count_existing_non_bank_loans(state, lender_id, borrower_id)
            if count >= config.max_loans_per_borrower_per_day:
                state.events.append(
                    {
                        "kind": "NonBankLoanRejectedConcentration",
                        "day": state.day,
                        "lender_id": lender_id,
                        "borrower_id": borrower_id,
                        "count": count,
                        "limit": config.max_loans_per_borrower_per_day,
                    }
                )
                continue

        rate, maturity_days = _resolve_non_bank_loan_terms(state, config, opportunity)
        loan_id = _create_non_bank_loan(
            state,
            lender_id=lender_id,
            borrower_id=opportunity["borrower_id"],
            amount=amount,
            rate=rate,
            maturity_days=maturity_days,
            banking_config=banking_config,
        )
        remaining -= amount
        created = True
        lending_events.append(
            {
                "kind": "NonBankLoanCreated",
                "day": state.day,
                "lender_id": lender_id,
                "borrower_id": opportunity["borrower_id"],
                "amount": amount,
                "rate": str(rate),
                "loan_id": loan_id,
                "p_default": str(opportunity["p_default"]),
            }
        )
        daily_expected_loss_spent += expected_loss
        state.lender_run_expected_loss_spent += expected_loss

    if config.preventive_lending and remaining > ZERO and config.kappa is not None:
        preventive_opportunities = _collect_preventive_lending_opportunities(
            state,
            config,
            lender_id,
            initial_capital,
        )
        preventive_opportunities.sort(
            key=lambda item: int(item.get("downstream", 0))
            * (1.0 - float(item["p_default"])),
            reverse=True,
        )
        remaining, daily_expected_loss_spent, preventive_created = (
            _execute_preventive_lending_opportunities(
                state,
                config,
                lender_id,
                preventive_opportunities,
                initial_capital,
                remaining,
                daily_expected_loss_spent,
                lending_events,
                banking_config=banking_config,
            )
        )
        created = created or preventive_created

    state.events.extend(lending_events)
    return created


def _apply_lender_capital_conservation(
    state: CleanState,
    config: CleanLenderConfig,
    lender_id: str,
) -> CleanLenderConfig:
    total_assets = state.cash[lender_id]
    total_loans = ZERO
    for loan in state.non_bank_loans:
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


def _realized_non_bank_loan_loss(state: CleanState) -> Decimal:
    total = ZERO
    for event in state.events:
        if event.get("kind") != "NonBankLoanDefaulted":
            continue
        amount_owed = Decimal(str(event.get("amount_owed", 0)))
        cash_available = Decimal(str(event.get("cash_available", 0)))
        total += max(ZERO, amount_owed - cash_available)
    return total


def _collect_lending_opportunities(
    state: CleanState,
    config: CleanLenderConfig,
    lender_id: str,
    initial_capital: Decimal,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    for agent_id, agent in state.agents.items():
        if agent_id in state.defaulted_agent_ids:
            continue
        if agent.kind not in ("household", "firm"):
            continue

        observed = _observe_lender_counterparty_liquidity(state, config, agent_id)
        if observed is None:
            continue
        upcoming_due, liquid = observed
        shortfall = upcoming_due - liquid
        if shortfall < config.min_shortfall:
            continue

        p_for_screen = _lender_observed_default_probability(state, config, agent_id)
        if p_for_screen > config.max_default_prob:
            continue
        p_default = (
            _lender_profile_default_probability(state, config, agent_id, upcoming_due, liquid)
            if config.kappa is not None
            else p_for_screen
        )

        max_single = Decimal(int(config.max_single_exposure * initial_capital))
        coverage_rate_penalty = ZERO
        if config.min_coverage_ratio > ZERO:
            coverage = _assess_non_bank_borrower(
                state,
                agent_id,
                min(shortfall, max_single),
                config.profit_target if config.kappa is not None else config.base_rate,
                config.horizon,
            )
            if config.coverage_mode == "graduated" and coverage >= Decimal("-1"):
                if coverage < config.min_coverage_ratio:
                    coverage_rate_penalty = config.coverage_penalty_scale * (
                        config.min_coverage_ratio - coverage
                    )
            elif coverage < config.min_coverage_ratio:
                state.events.append(
                    {
                        "kind": "NonBankLoanRejectedCoverage",
                        "day": state.day,
                        "lender_id": lender_id,
                        "borrower_id": agent_id,
                        "coverage": str(coverage),
                        "min_coverage": str(config.min_coverage_ratio),
                    }
                )
                continue

        borrower_existing = sum(
            loan.amount
            for loan in state.non_bank_loans
            if not loan.settled and loan.lender == lender_id and loan.borrower == agent_id
        )
        max_to_this_borrower = max_single - borrower_existing
        if max_to_this_borrower <= ZERO:
            continue

        loan_amount = min(shortfall, max_to_this_borrower)
        if config.collateralized_terms:
            collateral_value = _quality_adjusted_receivables(state, agent_id, config.horizon)
            collateral_cap = Decimal(int(collateral_value * config.collateral_advance_rate))
            loan_amount = min(loan_amount, collateral_cap)
            if loan_amount <= ZERO:
                continue

        expected_loss = loan_amount * p_default
        expected_relief = shortfall * (Decimal("1") - p_default)
        if config.marginal_relief_min_ratio > ZERO:
            ratio = expected_relief / expected_loss if expected_loss > ZERO else Decimal("999")
            if ratio < config.marginal_relief_min_ratio:
                state.events.append(
                    {
                        "kind": "NonBankLoanRejectedMarginalBenefit",
                        "day": state.day,
                        "lender_id": lender_id,
                        "borrower_id": agent_id,
                        "expected_relief": str(expected_relief),
                        "expected_loss": str(expected_loss),
                        "ratio": str(ratio),
                        "threshold": str(config.marginal_relief_min_ratio),
                    }
                )
                continue

        rate = _lender_loan_rate(config, p_default, banking_config=banking_config) + coverage_rate_penalty
        opportunities.append(
            {
                "borrower_id": agent_id,
                "amount": loan_amount,
                "rate": rate,
                "p_default": p_default,
                "expected_profit": float(rate) * (1.0 - float(p_default)),
                "expected_loss": float(expected_loss),
                "expected_relief": float(expected_relief),
                "downstream": _downstream_obligation_total(state, agent_id),
                "coverage_ratio": max(liquid, ZERO) / max(upcoming_due, Decimal("1")),
            }
        )
    return opportunities


def _collect_preventive_lending_opportunities(
    state: CleanState,
    config: CleanLenderConfig,
    lender_id: str,
    initial_capital: Decimal,
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    for agent_id, agent in state.agents.items():
        if agent_id in state.defaulted_agent_ids:
            continue
        if agent.kind not in ("household", "firm"):
            continue

        observed = _observe_lender_counterparty_liquidity(state, config, agent_id)
        if observed is None:
            continue
        upcoming_due, liquid = observed
        shortfall = upcoming_due - liquid
        if shortfall >= config.min_shortfall:
            continue

        at_risk = _receivables_at_risk(
            state,
            config,
            agent_id,
            config.horizon,
            config.prevention_threshold,
        )
        if at_risk <= ZERO:
            continue

        p_default = _lender_observed_default_probability(state, config, agent_id)
        if p_default > config.max_default_prob:
            continue

        borrower_existing = sum(
            loan.amount
            for loan in state.non_bank_loans
            if not loan.settled and loan.lender == lender_id and loan.borrower == agent_id
        )
        max_single = Decimal(int(config.max_single_exposure * initial_capital))
        max_to_this_borrower = max_single - borrower_existing
        if max_to_this_borrower <= ZERO:
            continue

        if config.max_loans_per_borrower_per_day > 0:
            count = _count_existing_non_bank_loans(state, lender_id, agent_id)
            if count >= config.max_loans_per_borrower_per_day:
                continue

        loan_amount = min(at_risk, max_to_this_borrower)
        if config.collateralized_terms:
            collateral_value = _quality_adjusted_receivables(state, agent_id, config.horizon)
            collateral_cap = Decimal(int(collateral_value * config.collateral_advance_rate))
            loan_amount = min(loan_amount, collateral_cap)
            if loan_amount <= ZERO:
                continue

        rate = _preventive_lender_loan_rate(config, p_default)
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
                "downstream": _downstream_obligation_total(state, agent_id),
                "coverage_ratio": max(liquid, ZERO) / max(upcoming_due, Decimal("1")),
                "preventive": True,
            }
        )
    return opportunities


def _execute_preventive_lending_opportunities(
    state: CleanState,
    config: CleanLenderConfig,
    lender_id: str,
    opportunities: list[dict[str, Any]],
    initial_capital: Decimal,
    remaining_capital: Decimal,
    daily_expected_loss_spent: Decimal,
    events: list[dict[str, Any]],
    *,
    banking_config: CleanBankingConfig | None = None,
) -> tuple[Decimal, Decimal, bool]:
    daily_expected_loss_cap = (
        initial_capital * config.daily_expected_loss_budget_ratio
        if config.daily_expected_loss_budget_ratio > ZERO
        else None
    )
    run_expected_loss_cap = (
        initial_capital * config.run_expected_loss_budget_ratio
        if config.run_expected_loss_budget_ratio > ZERO
        else None
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

        if (
            daily_expected_loss_cap is not None
            and daily_expected_loss_spent + expected_loss > daily_expected_loss_cap
        ):
            events.append(
                {
                    "kind": "NonBankLoanRejectedBudget",
                    "day": state.day,
                    "lender_id": lender_id,
                    "borrower_id": opportunity["borrower_id"],
                    "scope": "daily",
                    "expected_loss": str(expected_loss),
                    "budget_cap": str(daily_expected_loss_cap),
                    "budget_used": str(daily_expected_loss_spent),
                    "preventive": True,
                }
            )
            continue

        if (
            run_expected_loss_cap is not None
            and state.lender_run_expected_loss_spent + expected_loss > run_expected_loss_cap
        ):
            events.append(
                {
                    "kind": "NonBankLoanRejectedBudget",
                    "day": state.day,
                    "lender_id": lender_id,
                    "borrower_id": opportunity["borrower_id"],
                    "scope": "run",
                    "expected_loss": str(expected_loss),
                    "budget_cap": str(run_expected_loss_cap),
                    "budget_used": str(state.lender_run_expected_loss_spent),
                    "preventive": True,
                }
            )
            continue

        maturity_days = _resolve_preventive_non_bank_loan_maturity(
            state,
            config,
            opportunity["borrower_id"],
            opportunity["p_default"],
        )
        loan_id = _create_non_bank_loan(
            state,
            lender_id=lender_id,
            borrower_id=opportunity["borrower_id"],
            amount=amount,
            rate=opportunity["rate"],
            maturity_days=maturity_days,
            banking_config=banking_config,
        )
        remaining_capital -= amount
        created = True
        events.append(
            {
                "kind": "NonBankLoanCreatedPreventive",
                "day": state.day,
                "lender_id": lender_id,
                "borrower_id": opportunity["borrower_id"],
                "amount": amount,
                "rate": str(opportunity["rate"]),
                "loan_id": loan_id,
                "p_default": str(opportunity["p_default"]),
                "at_risk_receivables": opportunity["amount"],
            }
        )
        daily_expected_loss_spent += expected_loss
        state.lender_run_expected_loss_spent += expected_loss

    return remaining_capital, daily_expected_loss_spent, created


def _create_non_bank_loan(
    state: CleanState,
    *,
    lender_id: str,
    borrower_id: str,
    amount: Decimal,
    rate: Decimal,
    maturity_days: int,
    banking_config: CleanBankingConfig | None = None,
) -> str:
    if banking_config is not None and _agent_deposits_total(state, lender_id) > ZERO:
        paid = _pay_with_routed_deposits(
            state,
            lender_id,
            borrower_id,
            amount,
            banking_config,
        )
        if paid != amount:
            raise DefaultError(
                f"Insufficient deposits to create non-bank loan from {lender_id}: "
                f"{amount - paid} still needed"
            )
    else:
        _require_at_least(state.cash[lender_id], amount, f"{lender_id} cash")
        _take_cash_lots(state, lender_id, amount)
        state.cash[lender_id] -= amount
        state.cash[borrower_id] += amount
        _add_cash_lot(state, borrower_id, amount)
    loan_id = f"NBL_{len(state.non_bank_loans)}"
    loan = CleanNonBankLoan(
        id=loan_id,
        lender=lender_id,
        borrower=borrower_id,
        amount=amount,
        rate=rate,
        issuance_day=state.day,
        maturity_days=maturity_days,
    )
    state.non_bank_loans.append(loan)
    state.log(
        "NonBankLoanCreated",
        lender_id=lender_id,
        borrower_id=borrower_id,
        amount=amount,
        loan_id=loan_id,
        rate=str(rate),
        maturity_day=loan.maturity_day,
    )
    return loan_id


def _repay_due_non_bank_loans(state: CleanState) -> bool:
    impactful = False
    for loan in list(state.non_bank_loans):
        if loan.settled or state.day < loan.maturity_day:
            continue
        borrower_liquid = _agent_liquid_assets(state, loan.borrower)
        repayment = loan.repayment_amount
        if borrower_liquid < repayment:
            loan.settled = True
            state.log(
                "NonBankLoanDefaulted",
                loan_id=loan.id,
                borrower_id=loan.borrower,
                lender_id=loan.lender,
                amount_owed=repayment,
                cash_available=borrower_liquid,
            )
            impactful = True
            continue

        if state.cash[loan.borrower] >= repayment:
            _take_cash_lots(state, loan.borrower, repayment)
            state.cash[loan.borrower] -= repayment
            state.cash[loan.lender] += repayment
            _add_cash_lot(state, loan.lender, repayment)
        else:
            paid = _pay_with_deposit(state, loan.borrower, loan.lender, repayment)
            _require_at_least(paid, repayment, f"{loan.borrower} liquid assets")
        loan.settled = True
        state.log(
            "NonBankLoanRepaid",
            loan_id=loan.id,
            borrower_id=loan.borrower,
            lender_id=loan.lender,
            principal=loan.amount,
            interest=loan.interest_amount,
            total_repaid=repayment,
        )
        impactful = True
    return impactful
