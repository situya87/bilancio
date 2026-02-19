"""Bank lending phase for the Kalecki ring.

Banks offer loans to traders with upcoming shortfalls.
Traders compare r_L across their assigned banks and borrow from the cheapest.
Only traders (Households/Firms) can borrow — dealers, VBTs, and NBFIs cannot.

Loan origination creates deposits (money creation, no reserve movement).
Loan repayment destroys deposits (money destruction).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from bilancio.core.atomic_tx import atomic
from bilancio.domain.agent import AgentKind
from bilancio.domain.instruments.base import InstrumentKind

if TYPE_CHECKING:
    from bilancio.engines.banking_subsystem import BankLoanRecord, BankingSubsystem, BankTreynorState
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)

# Agent kinds eligible for bank borrowing
BORROWER_KINDS = {AgentKind.HOUSEHOLD, AgentKind.FIRM}


def run_bank_lending_phase(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """Run bank lending phase: banks offer loans to traders with shortfalls.

    Steps:
        1. Identify traders with upcoming shortfalls.
        2. For each, find the cheapest bank (lowest r_L).
        3. Check bank capacity.
        4. Execute loan (creates deposit, no reserve movement).

    Returns list of event dicts.
    """
    events: list[dict] = []

    # Find eligible borrowers with shortfalls
    eligible = _find_eligible_borrowers(system, banking, current_day)

    for borrower_id, shortfall in eligible:
        # Find cheapest bank
        bank_id = banking.cheapest_loan_bank(borrower_id)
        if bank_id is None:
            continue

        bank_state = banking.banks[bank_id]
        quote = bank_state.current_quote
        if quote is None:
            continue

        r_L = quote.loan_rate

        # Per-borrower credit risk pricing
        adjusted_rate = _per_borrower_rate(r_L, borrower_id, banking, current_day)
        if adjusted_rate is None:
            # Borrower is credit-rationed
            events.append({
                "kind": "BankLoanRationed",
                "day": current_day,
                "bank": bank_id,
                "borrower": borrower_id,
                "shortfall": shortfall,
            })
            continue
        r_L = adjusted_rate

        # Check bank has lending capacity
        if not _bank_can_lend(system, bank_state, shortfall):
            continue

        # Borrow-vs-sell decision: skip if selling is cheaper
        if _prefer_selling(system, borrower_id, shortfall, r_L, banking.loan_maturity, current_day):
            continue

        # Execute loan
        loan_id = _execute_bank_loan(
            system, banking, bank_state, borrower_id, shortfall, r_L,
            current_day, banking.loan_maturity,
        )

        if loan_id:
            events.append({
                "kind": "BankLoanIssued",
                "day": current_day,
                "bank": bank_id,
                "borrower": borrower_id,
                "amount": shortfall,
                "rate": str(r_L),
                "maturity_day": current_day + banking.loan_maturity,
                "loan_id": loan_id,
            })

    return events


def run_bank_loan_repayments(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """Process bank loan repayments due today.

    For each loan maturing today:
        1. Check if borrower has sufficient deposits.
        2. If yes, debit deposit and retire loan.
        3. If no, log default event.

    Returns list of event dicts.
    """
    from bilancio.engines.banking_subsystem import _get_deposit_at_bank, _get_total_deposits

    events: list[dict] = []

    for bank_state in banking.banks.values():
        repaid_loan_ids = []

        for loan_id, loan in list(bank_state.outstanding_loans.items()):
            if loan.maturity_day != current_day:
                continue

            borrower_id = loan.borrower_id
            bank_id = loan.bank_id
            repayment = loan.repayment_amount

            # Try to repay from borrower's deposits
            total_deposits = _get_total_deposits(system, borrower_id)

            if total_deposits >= repayment:
                # Repay: debit deposit at this bank first, then others
                _repay_loan(system, banking, loan, repayment)
                repaid_loan_ids.append(loan_id)

                events.append({
                    "kind": "BankLoanRepaid",
                    "day": current_day,
                    "bank": bank_id,
                    "borrower": borrower_id,
                    "principal": loan.principal,
                    "repayment": repayment,
                    "interest": repayment - loan.principal,
                    "loan_id": loan_id,
                })
            else:
                # Borrower can't fully repay — partial repayment + default
                if total_deposits > 0:
                    _repay_loan(system, banking, loan, total_deposits)

                events.append({
                    "kind": "BankLoanDefault",
                    "day": current_day,
                    "bank": bank_id,
                    "borrower": borrower_id,
                    "principal": loan.principal,
                    "repayment_due": repayment,
                    "recovered": min(total_deposits, repayment),
                    "loan_id": loan_id,
                })
                repaid_loan_ids.append(loan_id)

        # Remove repaid/defaulted loans from book
        for loan_id in repaid_loan_ids:
            loan = bank_state.outstanding_loans.pop(loan_id, None)
            if loan:
                bank_state.total_loan_principal -= loan.principal

                # Remove BankLoan instrument from system
                contract = system.state.contracts.get(loan_id)
                if contract:
                    # Remove from agent registries
                    bank_agent = system.state.agents.get(loan.bank_id)
                    borrower_agent = system.state.agents.get(loan.borrower_id)
                    if bank_agent and loan_id in bank_agent.asset_ids:
                        bank_agent.asset_ids.remove(loan_id)
                    if borrower_agent and loan_id in borrower_agent.liability_ids:
                        borrower_agent.liability_ids.remove(loan_id)
                    del system.state.contracts[loan_id]

    return events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _per_borrower_rate(
    base_rate: Decimal,
    borrower_id: str,
    banking: BankingSubsystem,
    current_day: int,
) -> Decimal | None:
    """Compute per-borrower rate: r_L = base + loading × P_default.

    Returns None if the borrower is credit-rationed (P_default > max_borrower_risk).
    Returns base_rate unchanged when credit_risk_loading == 0 (backward compat).
    """
    profile = banking.bank_profile
    loading = profile.credit_risk_loading
    max_risk = profile.max_borrower_risk

    # If neither pricing nor rationing is configured, skip risk lookup
    if loading == 0 and max_risk >= Decimal("1"):
        return base_rate

    assessor = banking.risk_assessor
    if assessor is None:
        return base_rate  # no risk assessor available

    p = assessor.estimate_default_prob(borrower_id, current_day)

    # Credit rationing (independent of loading)
    if p > max_risk:
        return None  # credit rationed

    # Per-borrower pricing
    if loading > 0:
        return base_rate + loading * p

    return base_rate


def _find_eligible_borrowers(
    system: System,
    banking: BankingSubsystem,
    current_day: int,
) -> list[tuple[str, int]]:
    """Find traders with upcoming shortfalls.

    Returns list of (agent_id, shortfall_amount) sorted by shortfall descending.
    """
    eligible = []
    horizon = banking.loan_maturity  # Look ahead by loan maturity

    for agent_id, agent in system.state.agents.items():
        if agent.kind not in BORROWER_KINDS:
            continue
        if agent.defaulted:
            continue

        # Calculate upcoming obligations within horizon
        obligations = _get_upcoming_obligations(system, agent_id, current_day, horizon)
        cash = _get_agent_liquidity(system, agent_id)

        shortfall = obligations - cash
        if shortfall > 0:
            eligible.append((agent_id, shortfall))

    # Sort by shortfall descending (most urgent first)
    eligible.sort(key=lambda x: -x[1])
    return eligible


def _get_upcoming_obligations(
    system: System,
    agent_id: str,
    current_day: int,
    horizon: int,
) -> int:
    """Get total obligations due within horizon days."""
    total = 0
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0

    for cid in agent.liability_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue
        if contract.kind == InstrumentKind.PAYABLE:
            due_day = getattr(contract, "due_day", None)
            if due_day is not None and current_day <= due_day <= current_day + horizon:
                total += contract.amount
        elif contract.kind == InstrumentKind.NON_BANK_LOAN:
            maturity_day = getattr(contract, "maturity_day", None)
            if maturity_day is not None and current_day <= maturity_day <= current_day + horizon:
                total += getattr(contract, "repayment_amount", contract.amount)
        elif contract.kind == InstrumentKind.BANK_LOAN:
            maturity_day = getattr(contract, "maturity_day", None)
            if maturity_day is not None and current_day <= maturity_day <= current_day + horizon:
                total += getattr(contract, "repayment_amount", contract.amount)

    return total


def _get_agent_liquidity(system: System, agent_id: str) -> int:
    """Get agent's total liquid assets (cash + deposits)."""
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    total = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue
        if contract.kind in (InstrumentKind.CASH, InstrumentKind.BANK_DEPOSIT):
            total += contract.amount
    return total


def _bank_can_lend(system: System, bank_state: BankTreynorState, amount: int) -> bool:
    """Check if bank has capacity to lend this amount.

    A bank can lend if its projected reserves after the loan
    would remain above the reserve floor. Since bank lending
    creates deposits (not reserve outflow), we check the
    reserve-to-deposit ratio won't fall too low.

    After the loan, deposits increase by ``amount`` (money creation) while
    reserves stay unchanged. The post-loan reserve ratio must remain above
    half the *current* target ratio (derived from pre-loan deposits).
    """
    from bilancio.engines.banking_subsystem import _get_bank_deposits_total, _get_bank_reserves

    reserves = _get_bank_reserves(system, bank_state.bank_id)
    deposits = _get_bank_deposits_total(system, bank_state.bank_id)

    # Current target ratio (from initialization: reserve_target = ratio * deposits)
    target_ratio = Decimal(bank_state.pricing_params.reserve_target) / Decimal(max(1, deposits))

    # After loan: deposits increase by amount (new deposit created)
    # Reserves unchanged (no reserve movement in money creation)
    new_deposits = deposits + amount
    post_loan_ratio = Decimal(reserves) / Decimal(max(1, new_deposits))

    # Floor: half the target ratio
    min_ratio = target_ratio / 2

    return post_loan_ratio > min_ratio


def _prefer_selling(
    system: System,
    borrower_id: str,
    shortfall: int,
    r_L: Decimal,
    loan_maturity: int,
    current_day: int,
) -> bool:
    """Compare cost of borrowing vs cost of selling to dealer.

    Borrow cost = shortfall * r_L (interest paid at maturity).
    Sell cost   = shortfall * (1 - avg_bid) (haircut from face value).

    The average bid is computed from actual dealer quotes when available.

    Returns True if selling to dealer is cheaper than borrowing.
    """
    borrow_cost = int(Decimal(shortfall) * r_L)

    # Estimate selling cost from dealer bid
    dealer_sub = system.state.dealer_subsystem
    if dealer_sub is None:
        return False  # No dealer available, must borrow

    # Compute average bid from all dealers that have a current bid price.
    # Falls back to the outside_mid_ratio from the subsystem config when
    # no dealer has quoted (day-0 edge case).
    bid_sum = Decimal("0")
    bid_count = 0
    for dealer in dealer_sub.dealers.values():
        if hasattr(dealer, "bid") and dealer.bid is not None:
            bid_sum += dealer.bid
            bid_count += 1

    if bid_count > 0:
        avg_bid = bid_sum / bid_count
    else:
        # Use outside_mid_ratio from subsystem config as a reasonable proxy
        avg_bid = Decimal(str(getattr(dealer_sub, "outside_mid_ratio", "0.85")))

    sell_cost = int(Decimal(shortfall) * (Decimal("1") - avg_bid))

    return sell_cost < borrow_cost


def _execute_bank_loan(
    system: System,
    banking: BankingSubsystem,
    bank_state: BankTreynorState,
    borrower_id: str,
    amount: int,
    rate: Decimal,
    current_day: int,
    maturity: int,
) -> str | None:
    """Execute a bank loan: create BankLoan instrument + credit borrower deposit.

    This is money creation:
    - Bank gains an asset (BankLoan)
    - Borrower gains a liability (BankLoan) AND an asset (deposit increase)
    - Bank gains a liability (deposit increase)
    - No reserve movement.
    """
    from bilancio.domain.instruments.bank_loan import BankLoan as BankLoanInstr
    from bilancio.engines.banking_subsystem import BankLoanRecord

    bid = bank_state.bank_id

    # 1. Create BankLoan instrument
    loan_id = system.new_contract_id(prefix="BL")
    loan_instrument = BankLoanInstr(
        id=loan_id,
        kind=InstrumentKind.BANK_LOAN,
        amount=amount,
        denom="USD",
        asset_holder_id=bid,  # Bank is the asset holder (creditor)
        liability_issuer_id=borrower_id,  # Borrower is the liability issuer
        rate=rate,
        issuance_day=current_day,
        maturity_days=maturity,
    )

    # Add to system contracts and agent registries
    system.state.contracts[loan_id] = loan_instrument
    bank_agent = system.state.agents.get(bid)
    borrower_agent = system.state.agents.get(borrower_id)
    if bank_agent:
        bank_agent.asset_ids.append(loan_id)
    else:
        logger.warning("Bank agent %s not found when issuing loan %s", bid, loan_id)
    if borrower_agent:
        borrower_agent.liability_ids.append(loan_id)
    else:
        logger.warning("Borrower agent %s not found when issuing loan %s", borrower_id, loan_id)

    # 2. Credit borrower's deposit at the lending bank (money creation)
    _increase_deposit(system, borrower_id, bid, amount)

    # 3. Track in bank state
    record = BankLoanRecord(
        loan_id=loan_id,
        bank_id=bid,
        borrower_id=borrower_id,
        principal=amount,
        rate=rate,
        issuance_day=current_day,
        maturity_day=current_day + maturity,
    )
    bank_state.outstanding_loans[loan_id] = record
    bank_state.total_loan_principal += amount

    # 4. Refresh bank's quote (lending changes balance sheet)
    bank_state.refresh_quote(system, current_day)

    system.log(
        "BankLoanIssued",
        bank=bid,
        borrower=borrower_id,
        amount=amount,
        rate=str(rate),
        maturity_day=current_day + maturity,
        loan_id=loan_id,
    )

    return loan_id


def _increase_deposit(system: System, agent_id: str, bank_id: str, amount: int) -> None:
    """Increase an agent's deposit at a specific bank.

    If the agent already has a deposit at this bank, increase it.
    Otherwise, create a new deposit instrument.
    """
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return

    # Find existing deposit at this bank
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if (
            contract
            and contract.kind == InstrumentKind.BANK_DEPOSIT
            and contract.liability_issuer_id == bank_id
        ):
            contract.amount += amount
            return

    # No existing deposit — create a new one
    from bilancio.domain.instruments.means_of_payment import BankDeposit

    deposit_id = system.new_contract_id(prefix="DEP")
    deposit = BankDeposit(
        id=deposit_id,
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=amount,
        denom="USD",
        asset_holder_id=agent_id,
        liability_issuer_id=bank_id,
    )
    system.state.contracts[deposit_id] = deposit
    agent.asset_ids.append(deposit_id)
    bank_agent = system.state.agents.get(bank_id)
    if bank_agent:
        bank_agent.liability_ids.append(deposit_id)


def _decrease_deposit(system: System, agent_id: str, bank_id: str, amount: int) -> int:
    """Decrease an agent's deposit at a specific bank.

    Returns the amount actually debited (may be less if insufficient).
    """
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0

    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if (
            contract
            and contract.kind == InstrumentKind.BANK_DEPOSIT
            and contract.liability_issuer_id == bank_id
        ):
            debit = min(amount, contract.amount)
            contract.amount -= debit
            return debit

    return 0


def _repay_loan(
    system: System,
    banking: BankingSubsystem,
    loan: BankLoanRecord,
    amount: int,
) -> None:
    """Repay a bank loan by debiting borrower deposits.

    Tries the lending bank first, then other banks (lowest r_D first).
    When deposits at a *different* bank are used, reserves must be
    transferred from that bank to the lending bank to settle the
    interbank claim.
    """
    borrower_id = loan.borrower_id
    bank_id = loan.bank_id

    remaining = amount

    # First, try to debit from the lending bank (no reserve movement needed)
    debited = _decrease_deposit(system, borrower_id, bank_id, remaining)
    remaining -= debited

    # If still remaining, debit from other banks (lowest r_D first)
    if remaining > 0:
        other_banks = []
        for bid in banking.banks:
            if bid != bank_id:
                state = banking.banks[bid]
                rate = state.current_quote.deposit_rate if state.current_quote else Decimal("0")
                other_banks.append((rate, bid))
        other_banks.sort()  # Lowest rate first

        for _, other_bid in other_banks:
            if remaining <= 0:
                break
            debited = _decrease_deposit(system, borrower_id, other_bid, remaining)
            if debited > 0:
                # Cross-bank settlement: transfer reserves from the bank
                # whose deposit was debited to the lending bank.
                try:
                    system.transfer_reserves(other_bid, bank_id, debited)
                except Exception:
                    # Reserve transfer failed — try CB refinancing atomically
                    try:
                        with atomic(system):
                            system.cb_lend_reserves(other_bid, debited, system.state.day)
                            system.transfer_reserves(other_bid, bank_id, debited)
                    except Exception:
                        # Both operations rolled back — reverse deposit debit
                        logger.warning(
                            "Reserve transfer failed (even with CB): %s -> %s amount=%d; "
                            "reversing deposit debit",
                            other_bid, bank_id, debited,
                        )
                        _increase_deposit(system, borrower_id, other_bid, debited)
                        continue  # skip this bank, don't reduce remaining
            remaining -= debited
