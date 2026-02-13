from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from bilancio.core.atomic_tx import atomic
from bilancio.core.errors import ValidationError
from bilancio.core.ids import AgentId, InstrId, new_id
from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.instruments.base import Instrument, InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan
from bilancio.domain.instruments.non_bank_loan import NonBankLoan
from bilancio.domain.instruments.means_of_payment import Cash, ReserveDeposit
from bilancio.domain.instruments.delivery import DeliveryObligation
from bilancio.domain.goods import StockLot
from bilancio.domain.policy import PolicyEngine
from bilancio.ops.primitives import consume, merge, split
from bilancio.ops.primitives_stock import split_stock, merge_stock

logger = logging.getLogger(__name__)


@dataclass
class State:
    agents: dict[AgentId, Agent] = field(default_factory=dict)
    contracts: dict[InstrId, Instrument] = field(default_factory=dict)
    stocks: dict[InstrId, StockLot] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    day: int = 0
    cb_cash_outstanding: int = 0
    cb_reserves_outstanding: int = 0
    cb_loans_outstanding: int = 0  # Total CB loans to banks (principal)
    phase: str = "simulation"
    # Aliases for created contracts (alias -> contract_id)
    aliases: dict[str, str] = field(default_factory=dict)
    # Scheduled actions to run at Phase B1 by day (day -> list of action dicts)
    scheduled_actions_by_day: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    # Track agents that have defaulted and been expelled from future activity
    defaulted_agent_ids: set[AgentId] = field(default_factory=set)
    # Plan 024: Enable continuous rollover of settled payables
    rollover_enabled: bool = False
    # Index contracts by due_day for fast lookup (preserves insertion order)
    contracts_by_due_day: dict[int, list[str]] = field(default_factory=dict)
    dealer_subsystem: Any = None
    jurisdictions: dict[str, Any] = field(default_factory=dict)  # str -> Jurisdiction
    fx_market: Any = None  # FXMarket instance
    lender_config: Any = None

class System:
    def __init__(self, policy: PolicyEngine | None = None, default_mode: str = "fail-fast"):
        self.policy = policy or PolicyEngine.default()
        self.state = State()
        self.default_mode = default_mode

    # ---- ID helpers
    def new_agent_id(self, prefix: str = "A") -> AgentId: return new_id(prefix)
    def new_contract_id(self, prefix: str = "C") -> InstrId: return new_id(prefix)

    # ---- phase management
    @contextmanager
    def setup(self) -> Generator[None, None, None]:
        """Context manager to temporarily set phase to 'setup'."""
        old_phase = self.state.phase
        self.state.phase = "setup"
        try:
            yield
        finally:
            self.state.phase = old_phase

    # ---- registry gateway
    def add_agent(self, agent: Agent) -> None:
        logger.debug("add_agent: %s (%s)", agent.id, agent.kind)
        self.state.agents[agent.id] = agent

    def add_contract(self, c: Instrument) -> None:
        # type invariants
        c.validate_type_invariants()
        # policy checks
        holder = self.state.agents[c.asset_holder_id]
        issuer = self.state.agents[c.liability_issuer_id]
        if not self.policy.can_hold(holder, c):
            raise ValidationError(f"{holder.kind} cannot hold {c.kind}")
        if not self.policy.can_issue(issuer, c):
            raise ValidationError(f"{issuer.kind} cannot issue {c.kind}")

        self.state.contracts[c.id] = c
        holder.asset_ids.append(c.id)
        issuer.liability_ids.append(c.id)
        logger.debug("add_contract: %s (kind=%s)", c.id, c.kind)

        # Maintain due_day index
        due_day = c.due_day
        if due_day is not None:
            if due_day not in self.state.contracts_by_due_day:
                self.state.contracts_by_due_day[due_day] = []
            self.state.contracts_by_due_day[due_day].append(c.id)

    # ---- events
    def log(self, kind: str, **payload: object) -> None:
        self.state.events.append({"kind": kind, "day": self.state.day, "phase": self.state.phase, **payload})

    # ---- invariants (MVP)
    def assert_invariants(self) -> None:
        from bilancio.core.invariants import (
            assert_cb_cash_matches_outstanding,
            assert_cb_reserves_match,
            assert_double_entry_numeric,
            assert_no_negative_balances,
            assert_no_duplicate_refs,
            assert_all_stock_ids_owned,
            assert_no_negative_stocks,
            assert_no_duplicate_stock_refs,
        )
        for cid, c in self.state.contracts.items():
            # For secondary market transfers (e.g., payables sold to dealers),
            # check the effective holder, not the original asset_holder_id
            effective_holder_id = c.effective_creditor
            assert cid in self.state.agents[effective_holder_id].asset_ids, f"{cid} missing on asset holder {effective_holder_id}"
            assert cid in self.state.agents[c.liability_issuer_id].liability_ids, f"{cid} missing on issuer"
        assert_no_duplicate_refs(self)
        assert_cb_cash_matches_outstanding(self)
        assert_cb_reserves_match(self)
        assert_no_negative_balances(self)
        assert_double_entry_numeric(self)
        # Stock-related invariants
        assert_all_stock_ids_owned(self)
        assert_no_negative_stocks(self)
        assert_no_duplicate_stock_refs(self)

    # ---- bootstrap helper
    def bootstrap_cb(self, cb: Agent) -> None:
        self.add_agent(cb)
        logger.info("bootstrapped central bank: %s", cb.id)
        self.log("BootstrapCB", cb_id=cb.id)
    
    def add_agents(self, agents: list[Agent]) -> None:
        """Add multiple agents to the system at once."""
        for agent in agents:
            self.add_agent(agent)

    # ---- cash operations
    def mint_cash(self, to_agent_id: AgentId, amount: int, denom: str = "X", alias: str | None = None) -> str:
        cb_id = next((aid for aid,a in self.state.agents.items() if a.kind == AgentKind.CENTRAL_BANK), None)
        assert cb_id, "CentralBank must exist"
        instr_id = self.new_contract_id("C")
        c = Cash(
            id=instr_id, kind=InstrumentKind.CASH, amount=amount, denom=denom,
            asset_holder_id=to_agent_id, liability_issuer_id=cb_id
        )
        with atomic(self):
            self.add_contract(c)
            self.state.cb_cash_outstanding += amount
            # Include alias if provided (for UI linking)
            if alias is not None:
                self.log("CashMinted", to=to_agent_id, amount=amount, instr_id=instr_id, alias=alias)
            else:
                self.log("CashMinted", to=to_agent_id, amount=amount, instr_id=instr_id)
        return instr_id

    def retire_cash(self, from_agent_id: AgentId, amount: int) -> None:
        # pull from holder's cash instruments (simple greedy)
        with atomic(self):
            remaining = amount
            cash_ids = [cid for cid in self.state.agents[from_agent_id].asset_ids
                        if cid in self.state.contracts and self.state.contracts[cid].kind == InstrumentKind.CASH]
            for cid in list(cash_ids):
                instr = self.state.contracts[cid]
                take = min(instr.amount, remaining)
                consume(self, cid, take)
                remaining -= take
                if remaining == 0: break
            if remaining != 0:
                raise ValidationError("insufficient cash to retire")
            self.state.cb_cash_outstanding -= amount
            self.log("CashRetired", frm=from_agent_id, amount=amount)

    def transfer_cash(self, from_agent_id: AgentId, to_agent_id: AgentId, amount: int) -> str:
        if from_agent_id == to_agent_id:
            raise ValidationError("no-op transfer")
        with atomic(self):
            remaining = amount
            # collect cash pieces and split as needed
            for cid in list(self.state.agents[from_agent_id].asset_ids):
                instr = self.state.contracts.get(cid)
                if not instr or instr.kind != InstrumentKind.CASH: continue
                piece_id = cid
                if instr.amount > remaining:
                    piece_id = split(self, cid, remaining)
                piece = self.state.contracts[piece_id]
                # move holder
                self.state.agents[from_agent_id].asset_ids.remove(piece_id)
                self.state.agents[to_agent_id].asset_ids.append(piece_id)
                piece.asset_holder_id = to_agent_id
                self.log("CashTransferred", frm=from_agent_id, to=to_agent_id, amount=min(remaining, piece.amount), instr_id=piece_id)
                remaining -= piece.amount
                if remaining == 0: break
            if remaining != 0:
                raise ValidationError("insufficient cash")
            # optional coalesce at receiver (merge duplicates)
            rx_ids = [cid for cid in self.state.agents[to_agent_id].asset_ids
                      if cid in self.state.contracts and self.state.contracts[cid].kind == InstrumentKind.CASH]
            # naive coalesce: pairwise merge same-key
            seen: dict[tuple[str, str], str] = {}
            for cid in rx_ids:
                if cid not in self.state.contracts:
                    continue  # may have been merged in prior iteration
                k = (self.state.contracts[cid].denom, self.state.contracts[cid].liability_issuer_id)
                keep = seen.get(k)
                if keep and keep != cid:
                    merge(self, keep, cid)
                else:
                    seen[k] = cid
        return "ok"

    # ---- reserve operations
    def _central_bank_id(self) -> str:
        """Find and return the central bank agent ID"""
        cb_id = next((aid for aid, a in self.state.agents.items() if a.kind == AgentKind.CENTRAL_BANK), None)
        if not cb_id:
            raise ValidationError("CentralBank must exist")
        return cb_id

    def mint_reserves(self, to_bank_id: str, amount: int, denom: str = "X", alias: str | None = None) -> str:
        """Mint reserves to a bank"""
        cb_id = self._central_bank_id()
        instr_id = self.new_contract_id("R")
        c = ReserveDeposit(
            id=instr_id, kind=InstrumentKind.RESERVE_DEPOSIT, amount=amount, denom=denom,
            asset_holder_id=to_bank_id, liability_issuer_id=cb_id
        )
        with atomic(self):
            self.add_contract(c)
            self.state.cb_reserves_outstanding += amount
            if alias is not None:
                self.log("ReservesMinted", to=to_bank_id, amount=amount, instr_id=instr_id, alias=alias)
            else:
                self.log("ReservesMinted", to=to_bank_id, amount=amount, instr_id=instr_id)
        return instr_id

    def transfer_reserves(self, from_bank_id: str, to_bank_id: str, amount: int) -> None:
        """Transfer reserves between banks"""
        if from_bank_id == to_bank_id:
            raise ValidationError("no-op transfer")
        with atomic(self):
            remaining = amount
            # collect reserve pieces and split as needed
            for cid in list(self.state.agents[from_bank_id].asset_ids):
                instr = self.state.contracts.get(cid)
                if not instr or instr.kind != InstrumentKind.RESERVE_DEPOSIT: continue
                piece_id = cid
                if instr.amount > remaining:
                    piece_id = split(self, cid, remaining)
                piece = self.state.contracts[piece_id]
                # move holder
                self.state.agents[from_bank_id].asset_ids.remove(piece_id)
                self.state.agents[to_bank_id].asset_ids.append(piece_id)
                piece.asset_holder_id = to_bank_id
                self.log("ReservesTransferred", frm=from_bank_id, to=to_bank_id, amount=min(remaining, piece.amount), instr_id=piece_id)
                remaining -= piece.amount
                if remaining == 0: break
            if remaining != 0:
                raise ValidationError("insufficient reserves")
            # optional coalesce at receiver (merge duplicates)
            rx_ids = [cid for cid in self.state.agents[to_bank_id].asset_ids
                      if cid in self.state.contracts and self.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT]
            # naive coalesce: pairwise merge same-key
            seen: dict[tuple[str, str], str] = {}
            for cid in rx_ids:
                if cid not in self.state.contracts:
                    continue
                k = (self.state.contracts[cid].denom, self.state.contracts[cid].liability_issuer_id)
                keep = seen.get(k)
                if keep and keep != cid:
                    merge(self, keep, cid)
                else:
                    seen[k] = cid

    def convert_reserves_to_cash(self, bank_id: str, amount: int) -> None:
        """Convert reserves to cash"""
        with atomic(self):
            # consume reserves
            remaining = amount
            reserve_ids = [cid for cid in self.state.agents[bank_id].asset_ids
                          if cid in self.state.contracts and self.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT]
            for cid in list(reserve_ids):
                instr = self.state.contracts[cid]
                take = min(instr.amount, remaining)
                consume(self, cid, take)
                remaining -= take
                if remaining == 0: break
            if remaining != 0:
                raise ValidationError("insufficient reserves to convert")
            # update outstanding reserves
            self.state.cb_reserves_outstanding -= amount
            # mint equivalent cash
            self.state.cb_cash_outstanding += amount
            cb_id = self._central_bank_id()
            instr_id = self.new_contract_id("C")
            c = Cash(
                id=instr_id, kind=InstrumentKind.CASH, amount=amount, denom="X",
                asset_holder_id=bank_id, liability_issuer_id=cb_id
            )
            self.add_contract(c)
            self.log("ReservesToCash", bank_id=bank_id, amount=amount, instr_id=instr_id)

    def convert_cash_to_reserves(self, bank_id: str, amount: int) -> None:
        """Convert cash to reserves"""
        with atomic(self):
            # consume cash
            remaining = amount
            cash_ids = [cid for cid in self.state.agents[bank_id].asset_ids
                       if cid in self.state.contracts and self.state.contracts[cid].kind == InstrumentKind.CASH]
            for cid in list(cash_ids):
                instr = self.state.contracts[cid]
                take = min(instr.amount, remaining)
                consume(self, cid, take)
                remaining -= take
                if remaining == 0: break
            if remaining != 0:
                raise ValidationError("insufficient cash to convert")
            # update outstanding cash
            self.state.cb_cash_outstanding -= amount
            # mint equivalent reserves
            self.state.cb_reserves_outstanding += amount
            cb_id = self._central_bank_id()
            instr_id = self.new_contract_id("R")
            c = ReserveDeposit(
                id=instr_id, kind=InstrumentKind.RESERVE_DEPOSIT, amount=amount, denom="X",
                asset_holder_id=bank_id, liability_issuer_id=cb_id
            )
            self.add_contract(c)
            self.log("CashToReserves", bank_id=bank_id, amount=amount, instr_id=instr_id)

    # ---- CB lending facility
    def cb_lend_reserves(self, bank_id: str, amount: int, day: int, denom: str = "X") -> str:
        """
        Central Bank lends reserves to a bank.

        This creates:
        1. New reserves (CB liability, bank asset)
        2. A CBLoan (CB asset, bank liability)

        The loan matures at day + 2 with interest at cb_lending_rate.

        Args:
            bank_id: The borrowing bank
            amount: Loan principal
            day: Current day (for maturity calculation)
            denom: Currency denomination

        Returns:
            The CBLoan instrument ID
        """
        cb_id = self._central_bank_id()
        cb = self.state.agents[cb_id]

        # Get the CB lending rate
        cb_rate = cb.cb_lending_rate if isinstance(cb, CentralBank) else Decimal("0.03")

        with atomic(self):
            # 1. Create new reserves for the bank
            reserve_id = self.new_contract_id("R")
            reserve = ReserveDeposit(
                id=reserve_id,
                kind=InstrumentKind.RESERVE_DEPOSIT,
                amount=amount,
                denom=denom,
                asset_holder_id=bank_id,
                liability_issuer_id=cb_id,
                remuneration_rate=cb.reserve_remuneration_rate if isinstance(cb, CentralBank) else None,
                issuance_day=day,
            )
            self.add_contract(reserve)
            self.state.cb_reserves_outstanding += amount

            # 2. Create the CB loan (bank's liability to CB)
            loan_id = self.new_contract_id("L")
            loan = CBLoan(
                id=loan_id,
                kind=InstrumentKind.CB_LOAN,
                amount=amount,
                denom=denom,
                asset_holder_id=cb_id,  # CB holds the loan as asset
                liability_issuer_id=bank_id,  # Bank is the borrower
                cb_rate=cb_rate,
                issuance_day=day,
            )
            self.add_contract(loan)
            self.state.cb_loans_outstanding += amount

            self.log("CBLoanCreated",
                     bank_id=bank_id,
                     amount=amount,
                     loan_id=loan_id,
                     reserve_id=reserve_id,
                     cb_rate=str(cb_rate),
                     maturity_day=day + 2)

        return loan_id

    def cb_repay_loan(self, loan_id: str, bank_id: str) -> int:
        """
        Bank repays a CB loan with interest.

        The bank pays principal + interest by having reserves consumed.
        The CBLoan is then cancelled.

        Args:
            loan_id: The CBLoan to repay
            bank_id: The borrowing bank (must match loan issuer)

        Returns:
            The total amount repaid (principal + interest)
        """
        if loan_id not in self.state.contracts:
            raise ValidationError(f"Loan {loan_id} not found")

        loan_instr = self.state.contracts[loan_id]
        if loan_instr.kind != InstrumentKind.CB_LOAN:
            raise ValidationError(f"{loan_id} is not a CB loan")
        if loan_instr.liability_issuer_id != bank_id:
            raise ValidationError(f"Bank {bank_id} is not the borrower of {loan_id}")

        assert isinstance(loan_instr, CBLoan)
        loan = loan_instr
        repayment_amount = loan.repayment_amount
        principal = loan.principal

        with atomic(self):
            # 1. Consume reserves from bank (repayment amount)
            remaining = repayment_amount
            reserve_ids = [cid for cid in self.state.agents[bank_id].asset_ids
                          if cid in self.state.contracts and self.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT]

            for cid in list(reserve_ids):
                instr = self.state.contracts[cid]
                take = min(instr.amount, remaining)
                consume(self, cid, take)
                remaining -= take
                if remaining == 0:
                    break

            if remaining != 0:
                raise ValidationError(f"Insufficient reserves to repay CB loan: needed {repayment_amount}, short by {remaining}")

            self.state.cb_reserves_outstanding -= repayment_amount

            # 2. Cancel the loan
            cb_id = loan.asset_holder_id
            self.state.agents[cb_id].asset_ids.remove(loan_id)
            self.state.agents[bank_id].liability_ids.remove(loan_id)

            # Maintain due_day index
            loan_due_day = loan.due_day
            if loan_due_day is not None:
                bucket = self.state.contracts_by_due_day.get(loan_due_day)
                if bucket:
                    try:
                        bucket.remove(loan_id)
                    except ValueError:
                        pass
                    if not bucket:
                        del self.state.contracts_by_due_day[loan_due_day]

            del self.state.contracts[loan_id]
            self.state.cb_loans_outstanding -= principal

            self.log("CBLoanRepaid",
                     bank_id=bank_id,
                     loan_id=loan_id,
                     principal=principal,
                     interest=repayment_amount - principal,
                     total_repaid=repayment_amount)

        return repayment_amount

    def get_cb_loans_due(self, day: int) -> list[str]:
        """Get all CB loans that are due on the given day."""
        due_loans = []
        for cid, contract in self.state.contracts.items():
            if contract.kind == InstrumentKind.CB_LOAN and isinstance(contract, CBLoan) and contract.is_due(day):
                due_loans.append(cid)
        return due_loans

    # ---- Non-bank lending facility
    def nonbank_lend_cash(
        self,
        lender_id: str,
        borrower_id: str,
        amount: int,
        rate: Decimal,
        day: int,
        maturity_days: int = 2,
        denom: str = "X",
    ) -> str:
        """Non-bank lender transfers existing cash to a borrower as a loan.

        Unlike CB lending (which creates new reserves), this transfers
        existing cash from the lender to the borrower.

        Args:
            lender_id: The lending agent
            borrower_id: The borrowing agent
            amount: Loan principal
            rate: Interest rate
            day: Current day (for maturity calculation)
            maturity_days: Days until maturity
            denom: Currency denomination

        Returns:
            The NonBankLoan instrument ID
        """
        # Find lender's cash and consume amount
        remaining = amount
        cash_ids = [
            cid for cid in self.state.agents[lender_id].asset_ids
            if cid in self.state.contracts
            and self.state.contracts[cid].kind == InstrumentKind.CASH
        ]

        cb_id = self._central_bank_id()

        with atomic(self):
            # 1. Consume cash from lender
            for cid in list(cash_ids):
                instr = self.state.contracts.get(cid)
                if instr is None:
                    continue
                take = min(instr.amount, remaining)
                consume(self, cid, take)
                remaining -= take
                if remaining == 0:
                    break

            if remaining != 0:
                raise ValidationError(
                    f"Lender {lender_id} has insufficient cash: "
                    f"needed {amount}, short by {remaining}"
                )

            # 2. Create new cash for borrower
            cash_id = self.new_contract_id("C")
            new_cash = Cash(
                id=cash_id,
                kind=InstrumentKind.CASH,
                amount=amount,
                denom=denom,
                asset_holder_id=borrower_id,
                liability_issuer_id=cb_id,
            )
            self.add_contract(new_cash)

            # 3. Create the NonBankLoan (lender's asset, borrower's liability)
            loan_id = self.new_contract_id("NBL")
            loan = NonBankLoan(
                id=loan_id,
                kind=InstrumentKind.NON_BANK_LOAN,
                amount=amount,
                denom=denom,
                asset_holder_id=lender_id,
                liability_issuer_id=borrower_id,
                rate=rate,
                issuance_day=day,
                maturity_days=maturity_days,
            )
            self.add_contract(loan)

            self.log(
                "NonBankLoanCreated",
                lender_id=lender_id,
                borrower_id=borrower_id,
                amount=amount,
                loan_id=loan_id,
                cash_id=cash_id,
                rate=str(rate),
                maturity_day=day + maturity_days,
            )

        return loan_id

    def nonbank_repay_loan(self, loan_id: str, borrower_id: str) -> bool:
        """Borrower repays a non-bank loan with interest.

        If borrower has sufficient cash, repays principal + interest.
        If insufficient, the loan defaults (lender takes the loss).

        Args:
            loan_id: The NonBankLoan to repay
            borrower_id: The borrowing agent

        Returns:
            True if repaid successfully, False if defaulted
        """
        if loan_id not in self.state.contracts:
            raise ValidationError(f"Loan {loan_id} not found")

        loan_instr = self.state.contracts[loan_id]
        if loan_instr.kind != InstrumentKind.NON_BANK_LOAN:
            raise ValidationError(f"{loan_id} is not a NonBankLoan")
        if loan_instr.liability_issuer_id != borrower_id:
            raise ValidationError(f"{borrower_id} is not the borrower of {loan_id}")

        assert isinstance(loan_instr, NonBankLoan)
        repayment_amount = loan_instr.repayment_amount
        lender_id = loan_instr.asset_holder_id
        principal = loan_instr.principal

        # Check if borrower has enough cash
        borrower_cash = sum(
            c.amount for cid in self.state.agents[borrower_id].asset_ids
            if (c := self.state.contracts.get(cid)) is not None
            and c.kind == InstrumentKind.CASH
        )

        if borrower_cash < repayment_amount:
            # Default: write off the loan
            self._remove_loan(loan_id, lender_id, borrower_id)
            self.log(
                "NonBankLoanDefaulted",
                loan_id=loan_id,
                borrower_id=borrower_id,
                lender_id=lender_id,
                amount_owed=repayment_amount,
                cash_available=borrower_cash,
            )
            return False

        cb_id = self._central_bank_id()

        with atomic(self):
            # 1. Consume cash from borrower
            remaining = repayment_amount
            cash_ids = [
                cid for cid in self.state.agents[borrower_id].asset_ids
                if cid in self.state.contracts
                and self.state.contracts[cid].kind == InstrumentKind.CASH
            ]
            for cid in list(cash_ids):
                instr = self.state.contracts.get(cid)
                if instr is None:
                    continue
                take = min(instr.amount, remaining)
                consume(self, cid, take)
                remaining -= take
                if remaining == 0:
                    break

            # 2. Create cash for lender (the repayment)
            cash_id = self.new_contract_id("C")
            new_cash = Cash(
                id=cash_id,
                kind=InstrumentKind.CASH,
                amount=repayment_amount,
                denom=loan_instr.denom,
                asset_holder_id=lender_id,
                liability_issuer_id=cb_id,
            )
            self.add_contract(new_cash)

            # 3. Remove the loan
            self._remove_loan(loan_id, lender_id, borrower_id)

            self.log(
                "NonBankLoanRepaid",
                loan_id=loan_id,
                borrower_id=borrower_id,
                lender_id=lender_id,
                principal=principal,
                interest=repayment_amount - principal,
                total_repaid=repayment_amount,
            )

        return True

    def _remove_loan(self, loan_id: str, lender_id: str, borrower_id: str) -> None:
        """Remove a NonBankLoan from the system."""
        self.state.agents[lender_id].asset_ids.remove(loan_id)
        self.state.agents[borrower_id].liability_ids.remove(loan_id)

        loan = self.state.contracts.get(loan_id)
        if loan is not None:
            due_day = loan.due_day
            if due_day is not None:
                bucket = self.state.contracts_by_due_day.get(due_day)
                if bucket:
                    try:
                        bucket.remove(loan_id)
                    except ValueError:
                        pass
                    if not bucket:
                        del self.state.contracts_by_due_day[due_day]

        del self.state.contracts[loan_id]

    def get_nonbank_loans_due(self, day: int) -> list[str]:
        """Get all NonBankLoans that are due on the given day."""
        due_loans = []
        for cid, contract in self.state.contracts.items():
            if (contract.kind == InstrumentKind.NON_BANK_LOAN
                    and isinstance(contract, NonBankLoan)
                    and contract.is_due(day)):
                due_loans.append(cid)
        return due_loans

    def credit_reserve_interest(self, day: int) -> int:
        """
        Credit interest on all reserve deposits that are due for interest.

        Interest is credited every 2 days (at issuance_day + 2, + 4, etc.).
        New reserves are minted to pay the interest.

        Args:
            day: Current day

        Returns:
            Total interest credited across all reserves
        """
        cb_id = self._central_bank_id()
        cb = self.state.agents[cb_id]

        # Check if CB has interest enabled
        if not (cb.reserves_accrue_interest if isinstance(cb, CentralBank) else True):
            return 0

        total_interest = 0

        with atomic(self):
            # Find all reserve deposits due for interest
            for cid in list(self.state.contracts.keys()):
                contract = self.state.contracts.get(cid)
                if contract is None or contract.kind != InstrumentKind.RESERVE_DEPOSIT:
                    continue

                # Must be a ReserveDeposit to check interest
                if not isinstance(contract, ReserveDeposit):
                    continue

                # Check if this reserve has interest and is due
                if not contract.is_interest_due(day):
                    continue

                interest = contract.compute_interest()
                if interest <= 0:
                    continue

                bank_id = contract.asset_holder_id

                # Mint new reserves for the interest
                interest_id = self.new_contract_id("R")
                interest_reserve = ReserveDeposit(
                    id=interest_id,
                    kind=InstrumentKind.RESERVE_DEPOSIT,
                    amount=interest,
                    denom=contract.denom,
                    asset_holder_id=bank_id,
                    liability_issuer_id=cb_id,
                    remuneration_rate=contract.remuneration_rate,
                    issuance_day=day,
                )
                self.add_contract(interest_reserve)
                self.state.cb_reserves_outstanding += interest

                # Update the original contract's last interest day
                contract.last_interest_day = day

                total_interest += interest

                self.log("ReserveInterestCredited",
                         bank_id=bank_id,
                         reserve_id=cid,
                         interest_reserve_id=interest_id,
                         interest=interest,
                         day=day)

        return total_interest

    def mint_reserves_with_interest(
        self,
        to_bank_id: str,
        amount: int,
        day: int,
        denom: str = "X",
        alias: str | None = None
    ) -> str:
        """
        Mint reserves to a bank with interest accrual enabled.

        This is an enhanced version of mint_reserves that sets up the reserve
        to accrue interest at the CB's reserve remuneration rate.

        Args:
            to_bank_id: Bank to receive reserves
            amount: Amount of reserves
            day: Current day (for interest tracking)
            denom: Currency denomination
            alias: Optional alias for UI

        Returns:
            The reserve instrument ID
        """
        cb_id = self._central_bank_id()
        cb = self.state.agents[cb_id]

        remuneration_rate = None
        if cb.reserves_accrue_interest if isinstance(cb, CentralBank) else True:
            remuneration_rate = cb.reserve_remuneration_rate if isinstance(cb, CentralBank) else Decimal("0.01")

        instr_id = self.new_contract_id("R")
        reserve = ReserveDeposit(
            id=instr_id,
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=amount,
            denom=denom,
            asset_holder_id=to_bank_id,
            liability_issuer_id=cb_id,
            remuneration_rate=remuneration_rate,
            issuance_day=day,
        )

        with atomic(self):
            self.add_contract(reserve)
            self.state.cb_reserves_outstanding += amount
            if alias is not None:
                self.log("ReservesMinted", to=to_bank_id, amount=amount, instr_id=instr_id, alias=alias)
            else:
                self.log("ReservesMinted", to=to_bank_id, amount=amount, instr_id=instr_id)

        return instr_id

    # ---- deposit helpers
    def deposit_ids(self, customer_id: str, bank_id: str) -> list[str]:
        """Filter customer assets for bank_deposit issued by bank_id"""
        out = []
        for cid in self.state.agents[customer_id].asset_ids:
            c = self.state.contracts.get(cid)
            if c is None:
                continue
            if c.kind == InstrumentKind.BANK_DEPOSIT and c.liability_issuer_id == bank_id:
                out.append(cid)
        return out

    def total_deposit(self, customer_id: str, bank_id: str) -> int:
        """Calculate total deposit amount for customer at bank"""
        return sum(self.state.contracts[cid].amount for cid in self.deposit_ids(customer_id, bank_id))

    # ---- obligation settlement

    def settle_obligation(self, contract_id: InstrId) -> None:
        """
        Settle and extinguish a bilateral obligation.
        
        This removes a matched asset-liability pair when the obligation has been fulfilled,
        such as after delivering goods or services that were promised.
        
        Args:
            contract_id: The ID of the contract to settle
            
        Raises:
            ValidationError: If the contract doesn't exist
        """
        with atomic(self):
            # Validate contract exists
            if contract_id not in self.state.contracts:
                raise ValidationError(f"Contract {contract_id} not found")
            
            contract = self.state.contracts[contract_id]
            
            # Remove from holder's assets
            holder = self.state.agents[contract.asset_holder_id]
            if contract_id not in holder.asset_ids:
                raise ValidationError(f"Contract {contract_id} not in holder's assets")
            holder.asset_ids.remove(contract_id)
            
            # Remove from issuer's liabilities
            issuer = self.state.agents[contract.liability_issuer_id]
            if contract_id not in issuer.liability_ids:
                raise ValidationError(f"Contract {contract_id} not in issuer's liabilities")
            issuer.liability_ids.remove(contract_id)

            # Maintain due_day index
            due_day = contract.due_day
            if due_day is not None:
                bucket = self.state.contracts_by_due_day.get(due_day)
                if bucket:
                    try:
                        bucket.remove(contract_id)
                    except ValueError:
                        pass
                    if not bucket:
                        del self.state.contracts_by_due_day[due_day]

            # Remove contract from registry
            del self.state.contracts[contract_id]

            # Log the settlement
            self.log("ObligationSettled",
                    contract_id=contract_id,
                    holder_id=contract.asset_holder_id,
                    issuer_id=contract.liability_issuer_id,
                    contract_kind=contract.kind,
                    amount=contract.amount)

    # ---- stock operations (inventory)
    def create_stock(self, owner_id: AgentId, sku: str, quantity: int, unit_price: Decimal, divisible: bool=True) -> InstrId:
        """Create a new stock lot (inventory)."""
        stock_id = new_id("S")
        stock = StockLot(
            id=stock_id,
            kind="stock_lot",
            sku=sku,
            quantity=quantity,
            unit_price=unit_price,
            owner_id=owner_id,
            divisible=divisible
        )
        with atomic(self):
            self.state.stocks[stock_id] = stock
            self.state.agents[owner_id].stock_ids.append(stock_id)
            self.log("StockCreated", owner=owner_id, sku=sku, qty=quantity, unit_price=unit_price, stock_id=stock_id)
        return stock_id

    def split_stock(self, stock_id: InstrId, quantity: int) -> InstrId:
        """Split a stock lot. Returns ID of the new split piece."""
        with atomic(self):
            return split_stock(self, stock_id, quantity)

    def merge_stock(self, stock_id_keep: InstrId, stock_id_into: InstrId) -> InstrId:
        """Merge two stock lots. Returns the ID of the kept lot."""
        with atomic(self):
            return merge_stock(self, stock_id_keep, stock_id_into)

    def _transfer_stock_internal(self, stock_id: InstrId, from_owner: AgentId, to_owner: AgentId, quantity: int | None = None) -> InstrId:
        """Internal helper for stock transfer without atomic wrapper."""
        stock = self.state.stocks[stock_id]
        if stock.owner_id != from_owner:
            raise ValidationError("Stock owner mismatch")
        
        moving_id = stock_id
        if quantity is not None:
            if not stock.divisible:
                raise ValidationError("Stock lot is not divisible")
            if quantity <= 0 or quantity > stock.quantity:
                raise ValidationError("Invalid transfer quantity")
            if quantity < stock.quantity:
                moving_id = split_stock(self, stock_id, quantity)
        
        # Transfer ownership
        moving_stock = self.state.stocks[moving_id]
        self.state.agents[from_owner].stock_ids.remove(moving_id)
        self.state.agents[to_owner].stock_ids.append(moving_id)
        moving_stock.owner_id = to_owner
        
        self.log("StockTransferred", 
                frm=from_owner, 
                to=to_owner, 
                stock_id=moving_id, 
                sku=moving_stock.sku,
                qty=moving_stock.quantity)
        return moving_id

    def transfer_stock(self, stock_id: InstrId, from_owner: AgentId, to_owner: AgentId, quantity: int | None = None) -> InstrId:
        """Transfer stock from one owner to another."""
        with atomic(self):
            return self._transfer_stock_internal(stock_id, from_owner, to_owner, quantity)

    # ---- delivery obligation operations
    def create_delivery_obligation(self, from_agent: AgentId, to_agent: AgentId, sku: str, quantity: int, unit_price: Decimal, due_day: int, alias: str | None = None) -> InstrId:
        """Create a delivery obligation (bilateral promise to deliver goods)."""
        obligation_id = self.new_contract_id("D")
        obligation = DeliveryObligation(
            id=obligation_id,
            kind=InstrumentKind.DELIVERY_OBLIGATION,
            amount=quantity,
            denom="N/A",
            asset_holder_id=to_agent,
            liability_issuer_id=from_agent,
            sku=sku,
            unit_price=unit_price,
            due_day=due_day
        )
        with atomic(self):
            self.add_contract(obligation)
            if alias is not None:
                self.log("DeliveryObligationCreated", 
                        id=obligation_id, 
                        frm=from_agent, 
                        to=to_agent, 
                        sku=sku, 
                        qty=quantity, 
                        due_day=due_day, 
                        unit_price=unit_price,
                        alias=alias)
            else:
                self.log("DeliveryObligationCreated", 
                        id=obligation_id, 
                        frm=from_agent, 
                        to=to_agent, 
                        sku=sku, 
                        qty=quantity, 
                        due_day=due_day, 
                        unit_price=unit_price)
        return obligation_id

    def _cancel_delivery_obligation_internal(self, obligation_id: InstrId) -> None:
        """Internal helper for cancelling delivery obligation without atomic wrapper."""
        # Validate contract exists and is a delivery obligation
        if obligation_id not in self.state.contracts:
            raise ValidationError(f"Contract {obligation_id} not found")
        
        contract_instr = self.state.contracts[obligation_id]
        if contract_instr.kind != InstrumentKind.DELIVERY_OBLIGATION:
            raise ValidationError(f"Contract {obligation_id} is not a delivery obligation")

        assert isinstance(contract_instr, DeliveryObligation)
        contract = contract_instr

        # Remove from holder's assets
        holder = self.state.agents[contract.asset_holder_id]
        if obligation_id not in holder.asset_ids:
            raise ValidationError(f"Contract {obligation_id} not in holder's assets")
        holder.asset_ids.remove(obligation_id)
        
        # Remove from issuer's liabilities
        issuer = self.state.agents[contract.liability_issuer_id]
        if obligation_id not in issuer.liability_ids:
            raise ValidationError(f"Contract {obligation_id} not in issuer's liabilities")
        issuer.liability_ids.remove(obligation_id)

        # Maintain due_day index
        due_day = contract.due_day
        if due_day is not None:
            bucket = self.state.contracts_by_due_day.get(due_day)
            if bucket:
                try:
                    bucket.remove(obligation_id)
                except ValueError:
                    pass
                if not bucket:
                    del self.state.contracts_by_due_day[due_day]

        # Remove contract from registry
        del self.state.contracts[obligation_id]

        # Log the cancellation with alias (if any) and contract_id for UI consistency
        from bilancio.ops.aliases import get_alias_for_id
        alias = get_alias_for_id(self, obligation_id)
        self.log("DeliveryObligationCancelled",
                obligation_id=obligation_id,
                contract_id=obligation_id,
                alias=alias,
                debtor=contract.liability_issuer_id,
                creditor=contract.asset_holder_id,
                sku=contract.sku,
                qty=contract.amount)

    def cancel_delivery_obligation(self, obligation_id: InstrId) -> None:
        """Cancel (extinguish) a delivery obligation. Used by settlement engine after fulfillment."""
        with atomic(self):
            self._cancel_delivery_obligation_internal(obligation_id)
