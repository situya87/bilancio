"""Banking subsystem: bank quotes, bank lending, loan servicing, resolution.

Banks quote deposit/loan rates from the shared pricing kernel
(``bilancio.banking.pricing_kernel``) using reserve-path forecasts; the
cheapest bank lends to liquidity-short households/firms by crediting
deposits (credit creation). Matured loans are debited from borrower
deposits, with cross-bank debits settled in reserves. When the CB lending
freeze is active, banks that cannot repay CB loans default and are
resolved: deposits are written off, reserves distributed pro-rata to
depositors (to a surviving bank, or as cash).

End-of-run, ``finalize_banking`` winds down outstanding bank loans and
runs the final CB settlement, matching ``finalize_banking_marker_events``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bilancio.banking.pricing_kernel import (
    PricingParams,
    compute_integrated_rate,
    compute_inventory,
    compute_quotes,
)
from bilancio.decision.profiles import BankProfile
from bilancio_v2.ledger import ZERO, BankLoan, CBLoan, Ledger, NonBankLoan
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.plugins.lending import upcoming_obligations
from bilancio_v2.subsystem_config import CleanBankingConfig


class BankQuotesPhase:
    """Subphase B_BankQuotes: marker only (quotes are computed on demand)."""

    name = "SubphaseB_BankQuotes"

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        return False


@dataclass(frozen=True)
class BankLendingPhase:
    name = "SubphaseB_BankLending"
    config: CleanBankingConfig

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        run_bank_lending_phase(ledger, self.config)
        # Like the lender, bank lending decisions never count as impactful.
        return False


# ---------------------------------------------------------------------------
# Quotes & pricing
# ---------------------------------------------------------------------------


def bank_profile(config: CleanBankingConfig) -> BankProfile:
    return BankProfile(
        reserve_target_ratio=config.reserve_target_ratio,
        credit_risk_loading=config.credit_risk_loading,
        max_borrower_risk=config.max_borrower_risk,
        min_coverage_ratio=config.min_coverage_ratio,
        adaptive_corridor=config.adaptive_corridor,
    )


def agent_banks(ledger: Ledger, agent_id: str, config: CleanBankingConfig) -> list[str]:
    if agent_id in config.trader_bank_assignments:
        return config.trader_bank_assignments[agent_id]
    if agent_id in config.infra_bank_assignments:
        return [config.infra_bank_assignments[agent_id]]
    return [bank_id for bank_id, agent in ledger.agents.items() if agent.kind == "bank"]


def bank_deposits_total(ledger: Ledger, bank_id: str) -> Decimal:
    return sum(
        (amount for (_customer_id, candidate_bank), amount in ledger.deposits.items() if candidate_bank == bank_id),
        ZERO,
    )


def agent_deposits_total(ledger: Ledger, agent_id: str) -> Decimal:
    return sum(
        (amount for (customer_id, _bank_id), amount in ledger.deposits.items() if customer_id == agent_id),
        ZERO,
    )


def bank_withdrawal_forecast(ledger: Ledger, bank_id: str, n_banks: int) -> Decimal:
    if n_banks <= 1:
        return ZERO
    cross_bank_fraction = Decimal(n_banks - 1) / Decimal(n_banks)
    seen_borrowers: set[str] = set()
    total_loan_deposits = ZERO
    for loan in ledger.bank_loans:
        if loan.settled or loan.bank != bank_id or loan.borrower in seen_borrowers:
            continue
        seen_borrowers.add(loan.borrower)
        total_loan_deposits += ledger.deposits[(loan.borrower, bank_id)]
    return Decimal(int(total_loan_deposits * cross_bank_fraction))


def bank_settlement_forecast(ledger: Ledger, bank_id: str) -> Decimal:
    net = ZERO
    for payable in ledger.payables:
        if payable.settled or payable.due_day != ledger.day:
            continue
        payer_bank = ledger.primary_bank_for_customer(payable.debtor)
        payee_bank = ledger.primary_bank_for_customer(payable.creditor) or payer_bank
        if payer_bank == payee_bank:
            continue
        if payer_bank == bank_id:
            net += payable.amount
        if payee_bank == bank_id:
            net -= payable.amount
    return net


def bank_quote(ledger: Ledger, bank_id: str, config: CleanBankingConfig, profile: BankProfile) -> tuple[Any, PricingParams]:
    reserve_target = config.reserve_targets.get(bank_id)
    if reserve_target is None:
        deposits = int(bank_deposits_total(ledger, bank_id))
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

    n_banks = max(1, sum(1 for agent in ledger.agents.values() if agent.kind == "bank"))
    withdrawal_forecast = bank_withdrawal_forecast(ledger, bank_id, n_banks)
    settlement_drain = max(ZERO, bank_settlement_forecast(ledger, bank_id))
    path: list[int] = [0] * 11
    path[0] = int(ledger.reserves[bank_id] - withdrawal_forecast - settlement_drain)
    for offset in range(1, 11):
        projected_day = ledger.day + offset
        delta = ZERO
        for loan in ledger.cb_loans:
            if loan.settled or loan.bank != bank_id:
                continue
            if loan.issuance_day + 2 == projected_day:
                delta -= loan.repayment_amount
        path[offset] = int(Decimal(path[offset - 1]) + delta)

    min_path = min(path)
    cash_tightness = max(ZERO, Decimal(reserve_floor - min_path) / Decimal(reserve_floor)) if reserve_floor > 0 else ZERO
    risk_index = cash_tightness
    inventory = compute_inventory(path[min(2, len(path) - 1)], reserve_target)
    quote = compute_quotes(
        inventory=inventory,
        cash_tightness=cash_tightness,
        risk_index=risk_index,
        params=params,
        day=ledger.day,
    )
    return quote, params


def cheapest_loan_bank(ledger: Ledger, borrower_id: str, config: CleanBankingConfig) -> tuple[str, Any, PricingParams] | None:
    profile = bank_profile(config)
    candidates: list[tuple[Decimal, str, Any, PricingParams]] = []
    for bank_id in agent_banks(ledger, borrower_id, config):
        bank = ledger.agents.get(bank_id)
        if bank is None or bank.kind != "bank" or bank_id in ledger.defaulted_agent_ids:
            continue
        quote, params = bank_quote(ledger, bank_id, config, profile)
        candidates.append((quote.loan_rate, bank_id, quote, params))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, bank_id, quote, params = candidates[0]
    return bank_id, quote, params


def bank_borrower_rate(base_rate: Decimal, borrower_id: str, profile: BankProfile, current_day: int) -> Decimal | None:
    del borrower_id, current_day
    loading = profile.credit_risk_loading
    max_risk = profile.max_borrower_risk
    if loading == ZERO:
        return base_rate

    p_default = Decimal("0.15")
    if p_default > max_risk:
        return None
    return base_rate + loading * p_default


def initial_banking_reserve_targets(ledger: Ledger, config: CleanBankingConfig) -> dict[str, int]:
    targets: dict[str, int] = {}
    for bank_id, agent in ledger.agents.items():
        if agent.kind != "bank":
            continue
        deposits = sum(
            (amount for (_customer_id, candidate_bank), amount in ledger.deposits.items() if candidate_bank == bank_id),
            ZERO,
        )
        targets[bank_id] = max(1, int(config.reserve_target_ratio * deposits))
    return targets


# ---------------------------------------------------------------------------
# Bank lending
# ---------------------------------------------------------------------------


def find_bank_borrowers(ledger: Ledger, horizon: int) -> list[tuple[str, Decimal]]:
    eligible: list[tuple[str, Decimal]] = []
    for agent_id, agent in ledger.agents.items():
        if agent.kind not in {"household", "firm"}:
            continue
        if agent_id in ledger.defaulted_agent_ids:
            continue
        shortfall = upcoming_obligations(ledger, agent_id, horizon, include_bank_loans=True) - ledger.agent_liquid_assets(agent_id)
        if shortfall > ZERO:
            eligible.append((agent_id, shortfall))
    eligible.sort(key=lambda item: -item[1])
    return eligible


def assess_bank_borrower(ledger: Ledger, borrower_id: str, amount: Decimal, rate: Decimal, loan_maturity: int) -> Decimal:
    loan_repayment = Decimal(int(amount * (Decimal("1") + rate)))
    maturity_day = ledger.day + loan_maturity
    liquid = ledger.agent_liquid_assets(borrower_id)
    obligations = ZERO
    for payable in ledger.payables:
        if payable.settled or payable.debtor != borrower_id:
            continue
        if ledger.day <= payable.due_day <= maturity_day:
            obligations += payable.amount
    mixed_loans: list[NonBankLoan | BankLoan] = [*ledger.non_bank_loans, *ledger.bank_loans]
    for loan in mixed_loans:
        if loan.settled or loan.borrower != borrower_id:
            continue
        if ledger.day <= loan.maturity_day <= maturity_day:
            obligations += loan.repayment_amount
    receivables = ZERO
    for payable in ledger.payables:
        if payable.settled or payable.creditor != borrower_id:
            continue
        if ledger.day <= payable.due_day <= maturity_day and payable.debtor not in ledger.defaulted_agent_ids:
            receivables += payable.amount
    if loan_repayment <= ZERO:
        return Decimal("999")
    return (liquid - obligations + receivables) / loan_repayment


def bank_can_lend(
    ledger: Ledger,
    bank_id: str,
    borrower_id: str,
    amount: Decimal,
    profile: BankProfile,
    params: PricingParams,
) -> bool:
    reserves = ledger.reserves[bank_id]
    deposits = bank_deposits_total(ledger, bank_id)
    target_ratio = Decimal(params.reserve_target) / Decimal(max(1, int(deposits)))
    post_loan_ratio = reserves / Decimal(max(1, int(deposits + amount)))
    if post_loan_ratio <= target_ratio * Decimal("0.75"):
        return False

    n_banks = sum(1 for agent in ledger.agents.values() if agent.kind == "bank")
    if n_banks > 1:
        cross_bank_fraction = Decimal(n_banks - 1) / Decimal(n_banks)
        expected_outflow = Decimal(int(amount * cross_bank_fraction))
        if Decimal(params.reserve_floor) > ledger.reserves[bank_id] - expected_outflow:
            return False

    max_total_capacity = Decimal(int(reserves * profile.max_total_exposure_ratio))
    max_single = Decimal(int(max_total_capacity * profile.max_single_exposure_ratio))
    existing_to_borrower = ZERO
    for loan in ledger.bank_loans:
        if loan.settled or loan.bank != bank_id:
            continue
        if loan.borrower == borrower_id:
            existing_to_borrower += loan.amount
    if existing_to_borrower + amount > max_single:
        return False
    total_principal = sum(
        (loan.amount for loan in ledger.bank_loans if not loan.settled and loan.bank == bank_id),
        ZERO,
    )
    if total_principal + amount > max_total_capacity:
        return False
    today_lending = sum(
        (loan.amount for loan in ledger.bank_loans if not loan.settled and loan.bank == bank_id and loan.issuance_day == ledger.day),
        ZERO,
    )
    max_daily = Decimal(int(reserves * profile.max_daily_lending_ratio))
    return today_lending + amount <= max_daily


def run_bank_lending_phase(ledger: Ledger, config: CleanBankingConfig) -> None:
    maturity = bank_profile(config).loan_maturity(config.maturity_days)
    for borrower_id, shortfall in find_bank_borrowers(ledger, maturity):
        if borrower_id in ledger.bank_defaulted_borrowers:
            continue
        if any(not loan.settled and loan.borrower == borrower_id for loan in ledger.bank_loans):
            continue
        bank_choice = cheapest_loan_bank(ledger, borrower_id, config)
        if bank_choice is None:
            continue
        bank_id, quote, params = bank_choice
        profile = bank_profile(config)
        _, loan_rate = compute_integrated_rate(
            current_inventory=quote.inventory,
            amount=int(shortfall),
            direction=-1,
            cash_tightness=quote.cash_tightness,
            risk_index=quote.risk_index,
            params=params,
        )
        adjusted_rate = bank_borrower_rate(loan_rate, borrower_id, profile, ledger.day)
        if adjusted_rate is None:
            ledger.log_raw("BankLoanRationed", bank=bank_id, borrower=borrower_id, shortfall=shortfall)
            continue
        loan_rate = adjusted_rate
        if profile.min_coverage_ratio > ZERO:
            coverage = assess_bank_borrower(ledger, borrower_id, shortfall, loan_rate, maturity)
            if coverage < profile.min_coverage_ratio:
                ledger.log_raw(
                    "BankLoanRejectedCoverage",
                    bank=bank_id,
                    borrower=borrower_id,
                    shortfall=shortfall,
                    coverage=str(coverage),
                    min_coverage=str(profile.min_coverage_ratio),
                )
                continue

        if not bank_can_lend(ledger, bank_id, borrower_id, shortfall, profile, params):
            continue
        loan = ledger.create_bank_loan(
            bank_id=bank_id,
            borrower_id=borrower_id,
            amount=shortfall,
            rate=loan_rate,
            maturity=maturity,
        )
        ledger.log_raw(
            "BankLoanIssued",
            bank=bank_id,
            borrower=borrower_id,
            amount=shortfall,
            rate=str(loan_rate),
            maturity_day=ledger.day + maturity,
            loan_id=loan.id,
            n_tickets=max(1, math.ceil(int(shortfall) / params.ticket_size)),
        )


# ---------------------------------------------------------------------------
# Loan servicing & winddown
# ---------------------------------------------------------------------------


def repay_due_bank_loans(ledger: Ledger, *, include_overdue: bool = False) -> None:
    for loan in list(ledger.bank_loans):
        if loan.settled:
            continue
        if include_overdue:
            if loan.maturity_day > ledger.day:
                continue
        elif loan.maturity_day != ledger.day:
            continue

        total_deposits = agent_deposits_total(ledger, loan.borrower)
        repayment = loan.repayment_amount
        if total_deposits >= repayment:
            debit_bank_loan_repayment(ledger, loan, repayment)
            loan.settled = True
            ledger.log_raw(
                "BankLoanRepaid",
                bank=loan.bank,
                borrower=loan.borrower,
                principal=loan.amount,
                repayment=repayment,
                interest=loan.interest_amount,
                loan_id=loan.id,
            )
            continue

        recovered = ZERO
        if total_deposits > ZERO:
            recovered = debit_bank_loan_repayment(ledger, loan, total_deposits)
        loan.settled = True
        ledger.bank_defaulted_borrowers.add(loan.borrower)
        ledger.log_raw(
            "BankLoanDefault",
            bank=loan.bank,
            borrower=loan.borrower,
            principal=loan.amount,
            repayment_due=repayment,
            recovered=recovered,
            loan_id=loan.id,
        )


def debit_bank_loan_repayment(ledger: Ledger, loan: BankLoan, amount: Decimal) -> Decimal:
    remaining = amount
    debited_total = ZERO
    debited = ledger.decrease_deposit(loan.borrower, loan.bank, remaining)
    remaining -= debited
    debited_total += debited
    if remaining <= ZERO:
        return debited_total

    for (customer_id, bank_id), balance in sorted(ledger.deposits.items()):
        if remaining <= ZERO:
            break
        if customer_id != loan.borrower or bank_id == loan.bank or balance <= ZERO:
            continue
        debited = ledger.decrease_deposit(loan.borrower, bank_id, remaining)
        if debited <= ZERO:
            continue
        if ledger.reserves[bank_id] >= debited:
            ledger.move_reserves_logged(bank_id, loan.bank, debited)
            debited_total += debited
            remaining -= debited
        else:
            ledger.credit_deposit(loan.borrower, bank_id, debited)
    return debited_total


def run_bank_loan_winddown(ledger: Ledger) -> int:
    if not has_outstanding_bank_loans(ledger):
        return 0
    initial_loans = sum(1 for loan in ledger.bank_loans if not loan.settled)
    ledger.log("BankLoanWinddownStart", outstanding_loans=initial_loans)
    max_maturity = max(loan.maturity_day for loan in ledger.bank_loans if not loan.settled)
    max_winddown = max(max_maturity - ledger.day + 5, 10)
    winddown_days = 0
    for _ in range(max_winddown):
        if not has_outstanding_bank_loans(ledger):
            break
        ledger.log("BankLoanWinddownDay")
        repay_due_bank_loans(ledger, include_overdue=True)
        ledger.day += 1
        winddown_days += 1
    remaining_loans = sum(1 for loan in ledger.bank_loans if not loan.settled)
    ledger.log(
        "BankLoanWinddownEnd",
        winddown_days=winddown_days,
        initial_loans=initial_loans,
        remaining_loans=remaining_loans,
    )
    return winddown_days


def has_outstanding_bank_loans(ledger: Ledger) -> bool:
    return any(not loan.settled for loan in ledger.bank_loans)


# ---------------------------------------------------------------------------
# CB freeze write-off & bank resolution
# ---------------------------------------------------------------------------


def write_off_frozen_cb_loan(ledger: Ledger, loan: CBLoan) -> None:
    ledger.log("CBLendingFrozen", bank_id=loan.bank, amount=loan.repayment_amount, day=ledger.day)
    if loan.bank in ledger.agents and loan.bank not in ledger.defaulted_agent_ids:
        ledger.defaulted_agent_ids.add(loan.bank)
        ledger.log("BankDefaultCBFreeze", bank_id=loan.bank, loan_id=loan.id, amount=loan.amount)
    loan.settled = True
    ledger.cb_loans_outstanding -= loan.amount
    ledger.log("CBLoanFreezeWrittenOff", bank_id=loan.bank, loan_id=loan.id, amount=loan.amount)


def write_off_final_cb_loan(ledger: Ledger, loan: CBLoan, *, reason: str) -> None:
    loan.settled = True
    ledger.cb_loans_outstanding -= loan.amount
    ledger.log(
        "CBFinalSettlementWrittenOff",
        loan_id=loan.id,
        bank_id=loan.bank,
        amount=loan.amount,
        reason=reason,
    )


def find_surviving_bank(ledger: Ledger, depositor_id: str, failed_bank_id: str) -> str | None:
    for customer_id, candidate_bank in ledger.deposits:
        if customer_id != depositor_id or candidate_bank == failed_bank_id:
            continue
        bank = ledger.agents.get(candidate_bank)
        if bank is not None and candidate_bank not in ledger.defaulted_agent_ids:
            return candidate_bank
    return None


def resolve_failed_bank(ledger: Ledger, bank_id: str) -> None:
    total_reserves = ledger.reserves[bank_id]
    if total_reserves <= ZERO:
        ledger.log(
            "BankResolutionCompleted",
            bank_id=bank_id,
            total_reserves=0,
            cb_claims_cancelled=0,
            depositor_distributions=0,
        )
        return

    remaining_reserves = total_reserves
    depositor_distributions: list[dict[str, Any]] = []
    deposit_claims = [
        (customer_id, amount)
        for (customer_id, candidate_bank), amount in ledger.deposits.items()
        if candidate_bank == bank_id and amount > ZERO
    ]
    total_deposit_claims = sum((amount for _, amount in deposit_claims), ZERO)

    if total_deposit_claims > ZERO:
        total_distributed = ZERO
        for depositor_id, claim_amount in deposit_claims:
            share = Decimal(round((claim_amount / total_deposit_claims) * remaining_reserves))
            share = min(share, remaining_reserves - total_distributed)
            if share <= ZERO:
                continue
            surviving_bank_id = find_surviving_bank(ledger, depositor_id, bank_id)
            total_distributed += share
            if surviving_bank_id is not None:
                ledger.credit_deposit(depositor_id, surviving_bank_id, share)
                ledger.move_reserves_logged(
                    bank_id,
                    surviving_bank_id,
                    share,
                    instr_id=f"resolution:{bank_id}:{surviving_bank_id}:{depositor_id}",
                )
                depositor_distributions.append(
                    {
                        "depositor": depositor_id,
                        "amount": share,
                        "method": "deposit_at_surviving_bank",
                        "bank": surviving_bank_id,
                    }
                )
            else:
                ledger.convert_reserves_to_cash(
                    bank_id,
                    depositor_id,
                    share,
                    instr_id=f"resolution:{bank_id}:{depositor_id}",
                )
                depositor_distributions.append(
                    {
                        "depositor": depositor_id,
                        "amount": share,
                        "method": "cash_no_surviving_bank",
                    }
                )

    ledger.log(
        "BankResolutionCompleted",
        bank_id=bank_id,
        total_reserves=total_reserves,
        remaining_reserves=remaining_reserves,
        cb_claims_cancelled=0,
        depositor_distributions=depositor_distributions,
        num_depositors=len(depositor_distributions),
    )


def write_off_bank_liabilities(ledger: Ledger, bank_id: str) -> None:
    for key, amount in list(ledger.deposits.items()):
        customer_id, candidate_bank = key
        if candidate_bank != bank_id or amount <= ZERO:
            continue
        ledger.log(
            "ObligationWrittenOff",
            contract_id=f"deposit:{customer_id}:{bank_id}",
            alias=None,
            debtor=bank_id,
            creditor=customer_id,
            contract_kind="bank_deposit",
            amount=amount,
        )
        del ledger.deposits[key]


# ---------------------------------------------------------------------------
# Final CB settlement & finalize
# ---------------------------------------------------------------------------


def run_final_cb_settlement(ledger: Ledger) -> dict[str, Decimal | int]:
    loans_attempted = 0
    loans_repaid = 0
    loans_written_off = 0
    bank_defaults = 0
    total_written_off_amount = ZERO
    defaulted_banks: set[str] = set()

    for loan in list(ledger.cb_loans):
        if loan.settled:
            continue
        if loan.bank in defaulted_banks:
            loans_attempted += 1
            loans_written_off += 1
            total_written_off_amount += loan.amount
            write_off_final_cb_loan(ledger, loan, reason="bank_already_defaulted")
            continue

        loans_attempted += 1
        repayment = loan.amount + loan.interest_amount
        if ledger.reserves[loan.bank] >= repayment:
            ledger.repay_cb_loan(loan)
            loans_repaid += 1
            ledger.log("CBFinalSettlementRepaid", loan_id=loan.id, bank_id=loan.bank)
            continue

        loans_written_off += 1
        total_written_off_amount += loan.amount
        defaulted_banks.add(loan.bank)
        if loan.bank in ledger.agents and loan.bank not in ledger.defaulted_agent_ids:
            ledger.defaulted_agent_ids.add(loan.bank)
            bank_defaults += 1
            ledger.log(
                "CBFinalSettlementBankDefault",
                bank_id=loan.bank,
                loan_id=loan.id,
                shortfall=loan.amount,
            )

        write_off_final_cb_loan(ledger, loan, reason="insufficient_reserves")
        for other_loan in list(ledger.cb_loans):
            if other_loan.settled or other_loan.bank != loan.bank:
                continue
            loans_attempted += 1
            loans_written_off += 1
            total_written_off_amount += other_loan.amount
            write_off_final_cb_loan(ledger, other_loan, reason="bank_already_defaulted")
        resolve_failed_bank(ledger, loan.bank)
        write_off_bank_liabilities(ledger, loan.bank)

    return {
        "loans_attempted": loans_attempted,
        "loans_repaid": loans_repaid,
        "loans_written_off": loans_written_off,
        "bank_defaults": bank_defaults,
        "total_written_off_amount": total_written_off_amount,
    }


def finalize_banking(
    ledger: Ledger,
    *,
    final_day: int,
    reached_stable: bool,
    banking_config: CleanBankingConfig | None,
) -> None:
    """End-of-run banking shutdown (mirrors ``finalize_banking_marker_events``)."""
    if banking_config is None:
        return

    ledger.day = final_day
    if reached_stable and not ledger.cb_lending_frozen:
        ledger.cb_lending_frozen = True
        ledger.log("CBLendingFreezeStability")

    run_bank_loan_winddown(ledger)

    outstanding = ledger.cb_loans_outstanding
    ledger.log("CBFinalSettlementStart", cb_loans_outstanding=outstanding)
    settlement = run_final_cb_settlement(ledger)
    ledger.log(
        "CBFinalSettlementEnd",
        loans_attempted=settlement["loans_attempted"],
        loans_repaid=settlement["loans_repaid"],
        loans_written_off=settlement["loans_written_off"],
        bank_defaults=settlement["bank_defaults"],
        total_written_off_amount=settlement["total_written_off_amount"],
        cb_loans_outstanding_pre_final=outstanding,
        cb_loans_outstanding_post_final=ledger.cb_loans_outstanding,
        cb_reserves_initial=ledger.cb_reserves_initial,
        cb_reserves_final=ledger.cb_reserves_outstanding,
        cb_interest_total_paid=ledger.cb_interest_total_paid,
        cb_loans_created_count=ledger.cb_loans_created_count,
    )
