"""Bank lending and quote helpers for the clean-core scenario engine."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from bilancio.banking.pricing_kernel import (
    PricingParams,
    compute_integrated_rate,
    compute_inventory,
    compute_quotes,
)
from bilancio.decision.profiles import BankProfile
from bilancio.engines.clean_core_interbank import primary_bank_for_customer
from bilancio.engines.clean_core_lender import agent_liquid_assets, upcoming_obligations
from bilancio.engines.clean_core_types import (
    ZERO,
    CleanBankingConfig,
    CleanBankLoan,
    CleanState,
)


def clean_bank_borrower_rate(
    base_rate: Decimal,
    borrower_id: str,
    profile: BankProfile,
    current_day: int,
) -> Decimal | None:
    del borrower_id, current_day
    loading = profile.credit_risk_loading
    max_risk = profile.max_borrower_risk
    if loading == ZERO:
        return base_rate

    p_default = Decimal("0.15")
    if p_default > max_risk:
        return None
    return base_rate + loading * p_default


def find_clean_bank_borrowers(
    state: CleanState,
    horizon: int,
) -> list[tuple[str, Decimal]]:
    eligible: list[tuple[str, Decimal]] = []
    for agent_id, agent in state.agents.items():
        if agent.kind not in {"household", "firm"}:
            continue
        if agent_id in state.defaulted_agent_ids:
            continue
        shortfall = upcoming_obligations(state, agent_id, horizon) - agent_liquid_assets(
            state,
            agent_id,
        )
        if shortfall > ZERO:
            eligible.append((agent_id, shortfall))
    eligible.sort(key=lambda item: -item[1])
    return eligible


def clean_cheapest_loan_bank(
    state: CleanState,
    borrower_id: str,
    config: CleanBankingConfig,
) -> tuple[str, Any, PricingParams] | None:
    profile = clean_bank_profile(config)
    candidates: list[tuple[Decimal, str, Any, PricingParams]] = []
    for bank_id in clean_agent_banks(state, borrower_id, config):
        bank = state.agents.get(bank_id)
        if bank is None or bank.kind != "bank" or bank_id in state.defaulted_agent_ids:
            continue
        quote, params = clean_bank_quote(state, bank_id, config, profile)
        candidates.append((quote.loan_rate, bank_id, quote, params))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, bank_id, quote, params = candidates[0]
    return bank_id, quote, params


def clean_agent_banks(
    state: CleanState,
    agent_id: str,
    config: CleanBankingConfig,
) -> list[str]:
    if agent_id in config.trader_bank_assignments:
        return config.trader_bank_assignments[agent_id]
    if agent_id in config.infra_bank_assignments:
        return [config.infra_bank_assignments[agent_id]]
    return [
        bank_id
        for bank_id, agent in state.agents.items()
        if agent.kind == "bank"
    ]


def clean_bank_profile(config: CleanBankingConfig) -> BankProfile:
    return BankProfile(
        reserve_target_ratio=config.reserve_target_ratio,
        credit_risk_loading=config.credit_risk_loading,
        max_borrower_risk=config.max_borrower_risk,
        min_coverage_ratio=config.min_coverage_ratio,
        adaptive_corridor=config.adaptive_corridor,
    )


def clean_bank_loan_maturity(config: CleanBankingConfig) -> int:
    return clean_bank_profile(config).loan_maturity(config.maturity_days)


def clean_bank_quote(
    state: CleanState,
    bank_id: str,
    config: CleanBankingConfig,
    profile: BankProfile,
) -> tuple[Any, PricingParams]:
    reserve_target = config.reserve_targets.get(bank_id)
    if reserve_target is None:
        deposits = int(clean_bank_deposits_total(state, bank_id))
        reserve_target = max(1, int(profile.reserve_target_ratio * deposits))
    symmetric_capacity = max(1, int(profile.symmetric_capacity_ratio * reserve_target))
    initial_deposits = int(Decimal(reserve_target) / profile.reserve_target_ratio)
    ticket_size = max(100, initial_deposits // 10) if initial_deposits > 0 else 100
    reserve_floor = max(1, reserve_target // 2)
    params = PricingParams(
        reserve_remuneration_rate=profile.r_floor(config.kappa, config.mu, config.concentration),
        cb_borrowing_rate=profile.r_ceiling(config.kappa, config.mu, config.concentration),
        reserve_target=reserve_target,
        symmetric_capacity=symmetric_capacity,
        ticket_size=ticket_size,
        reserve_floor=reserve_floor,
        alpha=profile.alpha,
        gamma=profile.gamma,
    )

    n_banks = max(1, sum(1 for agent in state.agents.values() if agent.kind == "bank"))
    withdrawal_forecast = clean_bank_withdrawal_forecast(state, bank_id, n_banks)
    settlement_drain = max(ZERO, clean_bank_settlement_forecast(state, bank_id))
    path: list[int] = [0] * 11
    path[0] = int(state.reserves[bank_id] - withdrawal_forecast - settlement_drain)
    for offset in range(1, 11):
        projected_day = state.day + offset
        delta = ZERO
        for loan in state.cb_loans:
            if loan.settled or loan.bank != bank_id:
                continue
            if loan.issuance_day + 2 == projected_day:
                delta -= loan.repayment_amount
        path[offset] = int(Decimal(path[offset - 1]) + delta)

    min_path = min(path)
    cash_tightness = (
        max(ZERO, Decimal(reserve_floor - min_path) / Decimal(reserve_floor))
        if reserve_floor > 0
        else ZERO
    )
    risk_index = cash_tightness
    inventory = compute_inventory(path[min(2, len(path) - 1)], reserve_target)
    quote = compute_quotes(
        inventory=inventory,
        cash_tightness=cash_tightness,
        risk_index=risk_index,
        params=params,
        day=state.day,
    )
    return quote, params


def clean_bank_deposits_total(state: CleanState, bank_id: str) -> Decimal:
    return sum(
        amount
        for (_customer_id, candidate_bank), amount in state.deposits.items()
        if candidate_bank == bank_id
    )


def clean_bank_withdrawal_forecast(
    state: CleanState,
    bank_id: str,
    n_banks: int,
) -> Decimal:
    if n_banks <= 1:
        return ZERO
    cross_bank_fraction = Decimal(n_banks - 1) / Decimal(n_banks)
    seen_borrowers: set[str] = set()
    total_loan_deposits = ZERO
    for loan in state.bank_loans:
        if loan.settled or loan.bank != bank_id or loan.borrower in seen_borrowers:
            continue
        seen_borrowers.add(loan.borrower)
        total_loan_deposits += state.deposits[(loan.borrower, bank_id)]
    return Decimal(int(total_loan_deposits * cross_bank_fraction))


def clean_bank_settlement_forecast(state: CleanState, bank_id: str) -> Decimal:
    net = ZERO
    for payable in state.payables:
        if payable.settled or payable.due_day != state.day:
            continue
        payer_bank = primary_bank_for_customer(state, payable.debtor)
        payee_bank = primary_bank_for_customer(state, payable.creditor) or payer_bank
        if payer_bank == payee_bank:
            continue
        if payer_bank == bank_id:
            net += payable.amount
        if payee_bank == bank_id:
            net -= payable.amount
    return net


def assess_clean_bank_borrower(
    state: CleanState,
    borrower_id: str,
    amount: Decimal,
    rate: Decimal,
    loan_maturity: int,
) -> Decimal:
    loan_repayment = Decimal(int(amount * (Decimal("1") + rate)))
    maturity_day = state.day + loan_maturity
    liquid = agent_liquid_assets(state, borrower_id)
    obligations = ZERO
    for payable in state.payables:
        if payable.settled or payable.debtor != borrower_id:
            continue
        if state.day <= payable.due_day <= maturity_day:
            obligations += payable.amount
    for loan in [*state.non_bank_loans, *state.bank_loans]:
        if loan.settled or loan.borrower != borrower_id:
            continue
        if state.day <= loan.maturity_day <= maturity_day:
            obligations += loan.repayment_amount
    receivables = ZERO
    for payable in state.payables:
        if payable.settled or payable.creditor != borrower_id:
            continue
        if (
            state.day <= payable.due_day <= maturity_day
            and payable.debtor not in state.defaulted_agent_ids
        ):
            receivables += payable.amount
    if loan_repayment <= ZERO:
        return Decimal("999")
    return (liquid - obligations + receivables) / loan_repayment


def clean_bank_can_lend(
    state: CleanState,
    bank_id: str,
    borrower_id: str,
    amount: Decimal,
    profile: BankProfile,
    params: PricingParams,
) -> bool:
    reserves = state.reserves[bank_id]
    deposits = clean_bank_deposits_total(state, bank_id)
    target_ratio = Decimal(params.reserve_target) / Decimal(max(1, int(deposits)))
    post_loan_ratio = reserves / Decimal(max(1, int(deposits + amount)))
    if post_loan_ratio <= target_ratio * Decimal("0.75"):
        return False

    n_banks = sum(1 for agent in state.agents.values() if agent.kind == "bank")
    if n_banks > 1:
        cross_bank_fraction = Decimal(n_banks - 1) / Decimal(n_banks)
        expected_outflow = Decimal(int(amount * cross_bank_fraction))
        if Decimal(params.reserve_floor) > state.reserves[bank_id] - expected_outflow:
            return False

    max_total_capacity = Decimal(int(reserves * profile.max_total_exposure_ratio))
    max_single = Decimal(int(max_total_capacity * profile.max_single_exposure_ratio))
    existing_to_borrower = ZERO
    for loan in state.bank_loans:
        if loan.settled or loan.bank != bank_id:
            continue
        if loan.borrower == borrower_id:
            existing_to_borrower += loan.amount
    if existing_to_borrower + amount > max_single:
        return False
    total_principal = sum(
        (loan.amount for loan in state.bank_loans if not loan.settled and loan.bank == bank_id),
        ZERO,
    )
    if total_principal + amount > max_total_capacity:
        return False
    today_lending = sum(
        (
            loan.amount
            for loan in state.bank_loans
            if not loan.settled and loan.bank == bank_id and loan.issuance_day == state.day
        ),
        ZERO,
    )
    max_daily = Decimal(int(reserves * profile.max_daily_lending_ratio))
    return today_lending + amount <= max_daily


def run_bank_lending_phase(state: CleanState, config: CleanBankingConfig) -> None:
    maturity = clean_bank_loan_maturity(config)
    for borrower_id, shortfall in find_clean_bank_borrowers(state, maturity):
        if borrower_id in state.bank_defaulted_borrowers:
            continue
        if any(
            not loan.settled and loan.borrower == borrower_id
            for loan in state.bank_loans
        ):
            continue
        bank_choice = clean_cheapest_loan_bank(state, borrower_id, config)
        if bank_choice is None:
            continue
        bank_id, quote, params = bank_choice
        profile = clean_bank_profile(config)
        _, loan_rate = compute_integrated_rate(
            current_inventory=quote.inventory,
            amount=int(shortfall),
            direction=-1,
            cash_tightness=quote.cash_tightness,
            risk_index=quote.risk_index,
            params=params,
        )
        adjusted_rate = clean_bank_borrower_rate(
            loan_rate,
            borrower_id,
            profile,
            state.day,
        )
        if adjusted_rate is None:
            state.events.append(
                {
                    "kind": "BankLoanRationed",
                    "day": state.day,
                    "bank": bank_id,
                    "borrower": borrower_id,
                    "shortfall": shortfall,
                }
            )
            continue
        loan_rate = adjusted_rate
        if profile.min_coverage_ratio > ZERO:
            coverage = assess_clean_bank_borrower(
                state,
                borrower_id,
                shortfall,
                loan_rate,
                maturity,
            )
            if coverage < profile.min_coverage_ratio:
                state.events.append(
                    {
                        "kind": "BankLoanRejectedCoverage",
                        "day": state.day,
                        "bank": bank_id,
                        "borrower": borrower_id,
                        "shortfall": shortfall,
                        "coverage": str(coverage),
                        "min_coverage": str(profile.min_coverage_ratio),
                    }
                )
                continue

        if not clean_bank_can_lend(
            state,
            bank_id,
            borrower_id,
            shortfall,
            profile,
            params,
        ):
            continue
        loan_id = create_clean_bank_loan(
            state,
            bank_id=bank_id,
            borrower_id=borrower_id,
            amount=shortfall,
            rate=loan_rate,
            maturity=maturity,
        )
        state.events.append(
            {
                "kind": "BankLoanIssued",
                "day": state.day,
                "bank": bank_id,
                "borrower": borrower_id,
                "amount": shortfall,
                "rate": str(loan_rate),
                "maturity_day": state.day + maturity,
                "loan_id": loan_id,
                "n_tickets": max(1, math.ceil(int(shortfall) / params.ticket_size)),
            }
        )


def create_clean_bank_loan(
    state: CleanState,
    *,
    bank_id: str,
    borrower_id: str,
    amount: Decimal,
    rate: Decimal,
    maturity: int,
) -> str:
    loan_id = f"BL_{len(state.bank_loans)}"
    state.bank_loans.append(
        CleanBankLoan(
            id=loan_id,
            bank=bank_id,
            borrower=borrower_id,
            amount=amount,
            rate=rate,
            issuance_day=state.day,
            maturity_day=state.day + maturity,
        )
    )
    state.deposits[(borrower_id, bank_id)] += amount
    return loan_id


def repay_due_bank_loans(state: CleanState, *, include_overdue: bool = False) -> None:
    for loan in list(state.bank_loans):
        if loan.settled:
            continue
        if include_overdue:
            if loan.maturity_day > state.day:
                continue
        elif loan.maturity_day != state.day:
            continue

        total_deposits = agent_deposits_total(state, loan.borrower)
        repayment = loan.repayment_amount
        if total_deposits >= repayment:
            debit_clean_bank_loan_repayment(state, loan, repayment)
            loan.settled = True
            state.events.append(
                {
                    "kind": "BankLoanRepaid",
                    "day": state.day,
                    "bank": loan.bank,
                    "borrower": loan.borrower,
                    "principal": loan.amount,
                    "repayment": repayment,
                    "interest": loan.interest_amount,
                    "loan_id": loan.id,
                }
            )
            continue

        recovered = ZERO
        if total_deposits > ZERO:
            recovered = debit_clean_bank_loan_repayment(state, loan, total_deposits)
        loan.settled = True
        state.bank_defaulted_borrowers.add(loan.borrower)
        state.events.append(
            {
                "kind": "BankLoanDefault",
                "day": state.day,
                "bank": loan.bank,
                "borrower": loan.borrower,
                "principal": loan.amount,
                "repayment_due": repayment,
                "recovered": recovered,
                "loan_id": loan.id,
            }
        )


def agent_deposits_total(state: CleanState, agent_id: str) -> Decimal:
    return sum(
        amount
        for (customer_id, _bank_id), amount in state.deposits.items()
        if customer_id == agent_id
    )


def debit_clean_bank_loan_repayment(
    state: CleanState,
    loan: CleanBankLoan,
    amount: Decimal,
) -> Decimal:
    remaining = amount
    debited_total = ZERO
    debited = decrease_clean_deposit(state, loan.borrower, loan.bank, remaining)
    remaining -= debited
    debited_total += debited
    if remaining <= ZERO:
        return debited_total

    for (customer_id, bank_id), balance in sorted(state.deposits.items()):
        if remaining <= ZERO:
            break
        if customer_id != loan.borrower or bank_id == loan.bank or balance <= ZERO:
            continue
        debited = decrease_clean_deposit(state, loan.borrower, bank_id, remaining)
        if debited <= ZERO:
            continue
        if state.reserves[bank_id] >= debited:
            state.reserves[bank_id] -= debited
            state.reserves[loan.bank] += debited
            state.log("ReservesTransferred", frm=bank_id, to=loan.bank, amount=debited)
            debited_total += debited
            remaining -= debited
        else:
            state.deposits[(loan.borrower, bank_id)] += debited
    return debited_total


def decrease_clean_deposit(
    state: CleanState,
    agent_id: str,
    bank_id: str,
    amount: Decimal,
) -> Decimal:
    debited = min(state.deposits[(agent_id, bank_id)], amount)
    if debited > ZERO:
        state.deposits[(agent_id, bank_id)] -= debited
    return debited


def run_clean_bank_loan_winddown(state: CleanState) -> int:
    if not has_outstanding_clean_bank_loans(state):
        return 0
    initial_loans = sum(1 for loan in state.bank_loans if not loan.settled)
    state.log("BankLoanWinddownStart", outstanding_loans=initial_loans)
    max_maturity = max(loan.maturity_day for loan in state.bank_loans if not loan.settled)
    max_winddown = max(max_maturity - state.day + 5, 10)
    winddown_days = 0
    for _ in range(max_winddown):
        if not has_outstanding_clean_bank_loans(state):
            break
        state.log("BankLoanWinddownDay")
        repay_due_bank_loans(state, include_overdue=True)
        state.day += 1
        winddown_days += 1
    remaining_loans = sum(1 for loan in state.bank_loans if not loan.settled)
    state.log(
        "BankLoanWinddownEnd",
        winddown_days=winddown_days,
        initial_loans=initial_loans,
        remaining_loans=remaining_loans,
    )
    return winddown_days


def has_outstanding_clean_bank_loans(state: CleanState) -> bool:
    return any(not loan.settled for loan in state.bank_loans)
