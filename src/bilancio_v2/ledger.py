"""The v2 ledger: single source of truth for all simulation state.

Design rules
============
1. Every balance mutation goes through a ledger operation that records the
   corresponding event in the journal. Plugins never touch raw balances.
2. There is exactly one copy of every balance — no subsystem shadow state,
   no sync functions.
3. Conservation invariants are checked after every simulated day:
   cash in circulation equals cash minted minus burned, reserves equal the
   central bank's outstanding issuance, every cash balance equals the sum
   of its lots, and no balance is negative.

The observable behavior (event kinds, payloads, and balance arithmetic,
including cash-lot granularity of ``CashTransferred`` events) matches the
existing clean-core engine exactly; the parity suite enforces this.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from bilancio_v2.events import EventJournal

ZERO = Decimal("0")


class InsufficientFundsError(ValueError):
    """A ledger operation required more funds than the agent holds."""


class InvariantViolation(AssertionError):
    """A conservation or integrity invariant failed after a ledger operation."""


# ---------------------------------------------------------------------------
# Open-contract records (projections of contract-creation events)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentInfo:
    id: str
    kind: str
    name: str


@dataclass
class Payable:
    id: str
    debtor: str
    creditor: str
    amount: Decimal
    due_day: int
    maturity_distance: int
    alias: str | None = None
    settled: bool = False
    netted_amount: Decimal = ZERO


@dataclass
class CBLoan:
    id: str
    bank: str
    central_bank: str
    amount: Decimal
    rate: Decimal
    issuance_day: int
    alias: str | None = None
    settled: bool = False

    @property
    def repayment_amount(self) -> Decimal:
        return Decimal(int(self.amount * (Decimal("1") + self.rate)))

    @property
    def interest_amount(self) -> Decimal:
        return self.repayment_amount - self.amount


@dataclass
class NonBankLoan:
    id: str
    lender: str
    borrower: str
    amount: Decimal
    rate: Decimal
    issuance_day: int
    maturity_days: int
    settled: bool = False

    @property
    def maturity_day(self) -> int:
        return self.issuance_day + self.maturity_days

    @property
    def repayment_amount(self) -> Decimal:
        return Decimal(int(self.amount * (Decimal("1") + self.rate)))

    @property
    def interest_amount(self) -> Decimal:
        return self.repayment_amount - self.amount


@dataclass
class BankLoan:
    id: str
    bank: str
    borrower: str
    amount: Decimal
    rate: Decimal
    issuance_day: int
    maturity_day: int
    settled: bool = False

    @property
    def repayment_amount(self) -> Decimal:
        return Decimal(int(self.amount * (Decimal("1") + self.rate)))

    @property
    def interest_amount(self) -> Decimal:
        return self.repayment_amount - self.amount


@dataclass
class StockLot:
    id: str
    owner: str
    sku: str
    quantity: int
    unit_price: Decimal

    @property
    def value(self) -> Decimal:
        return Decimal(self.quantity) * self.unit_price


@dataclass
class DeliveryObligation:
    id: str
    debtor: str
    creditor: str
    sku: str
    quantity: int
    unit_price: Decimal
    due_day: int
    alias: str | None = None
    settled: bool = False


@dataclass
class Checkpoint:
    """Snapshot for fail-fast atomic rollback of a settlement attempt."""

    cash: defaultdict[str, Decimal]
    cash_lots: dict[str, list[Decimal]]
    deposits: defaultdict[tuple[str, str], Decimal]
    stocks: dict[str, StockLot]
    journal_length: int


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    agents: dict[str, AgentInfo] = field(default_factory=dict)
    central_bank_id: str | None = None

    cash: defaultdict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    cash_lots: defaultdict[str, list[Decimal]] = field(default_factory=lambda: defaultdict(list))
    reserves: defaultdict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    # (customer, bank) -> amount
    deposits: defaultdict[tuple[str, str], Decimal] = field(default_factory=lambda: defaultdict(Decimal))

    payables: list[Payable] = field(default_factory=list)
    cb_loans: list[CBLoan] = field(default_factory=list)
    non_bank_loans: list[NonBankLoan] = field(default_factory=list)
    bank_loans: list[BankLoan] = field(default_factory=list)
    bank_defaulted_borrowers: set[str] = field(default_factory=set)
    stocks: dict[str, StockLot] = field(default_factory=dict)
    delivery_obligations: list[DeliveryObligation] = field(default_factory=list)

    scheduled_actions_by_day: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    defaulted_agent_ids: set[str] = field(default_factory=set)

    # (debtor, creditor, original face, maturity distance) entries queued by
    # the clearing phase for fully-netted payables, drained into the rollover
    # flow by the settlement phase ("net-settle, gross-roll", Plan 059).
    netted_rollover_queue: list[tuple[str, str, Decimal, int]] = field(default_factory=list)

    rating_registry: dict[str, Decimal] = field(default_factory=dict)
    lender_run_expected_loss_spent: Decimal = ZERO
    estimate_logging_enabled: bool = False
    estimate_log: list[Any] = field(default_factory=list)

    # Dealer subsystem working state (active mode reuses the shared trading
    # machinery from bilancio.dealer; system-side balances stay ledger-owned
    # and are reconciled at the same points as the existing engine).
    dealer_config: Any | None = None
    dealer_subsystem: Any | None = None
    dealer_metrics: Any | None = None

    cash_minted_total: Decimal = ZERO
    cash_burned_total: Decimal = ZERO
    cash_converted_from_reserves: Decimal = ZERO
    cb_reserves_initial: Decimal = ZERO
    cb_reserves_outstanding: Decimal = ZERO
    cb_loans_outstanding: Decimal = ZERO
    cb_interest_total_paid: Decimal = ZERO
    cb_loans_created_count: int = 0
    cb_lending_frozen: bool = False

    day: int = 0
    quiet_days: int = 0
    journal: EventJournal = field(default_factory=EventJournal)

    # -- journal helpers ----------------------------------------------------

    def log(self, kind: str, **data: Any) -> None:
        self.journal.append(kind, self.day, "simulation", **data)

    def log_setup(self, kind: str, **data: Any) -> None:
        self.journal.append(kind, 0, "setup", **data)

    def record(self, kind: str, *, setup: bool, **data: Any) -> None:
        if setup:
            self.log_setup(kind, **data)
        else:
            self.log(kind, **data)

    def log_raw(self, kind: str, **data: Any) -> None:
        """Record an informational event with no phase key (legacy subsystem shape)."""
        self.journal.append(kind, self.day, None, **data)

    # -- agent registration -------------------------------------------------

    def register_agent(self, agent_id: str, kind: str, name: str) -> None:
        self.agents[agent_id] = AgentInfo(id=agent_id, kind=kind, name=name)
        if self.central_bank_id is None and kind == "central_bank":
            self.central_bank_id = agent_id

    # -- cash operations ----------------------------------------------------

    def _require(self, actual: Decimal, required: Decimal, label: str) -> None:
        if actual < required:
            raise InsufficientFundsError(f"insufficient {label}: required {required}, available {actual}")

    def _add_cash_lot(self, agent_id: str, amount: Decimal) -> None:
        if amount > ZERO:
            self.cash_lots[agent_id].append(amount)

    def _take_cash_lots(self, agent_id: str, amount: Decimal) -> list[Decimal]:
        remaining = amount
        pieces: list[Decimal] = []
        lots = self.cash_lots[agent_id]
        if not lots and self.cash[agent_id] > ZERO:
            lots.append(self.cash[agent_id])
        while remaining > ZERO and lots:
            lot = lots.pop(0)
            take = min(lot, remaining)
            pieces.append(take)
            leftover = lot - take
            if leftover > ZERO:
                lots.insert(0, leftover)
            remaining -= take
        if remaining != ZERO:
            raise InsufficientFundsError(f"insufficient {agent_id} cash lots: short by {remaining}")
        return pieces

    def _merge_cash_lots(self, agent_id: str, *, setup: bool) -> None:
        lots = self.cash_lots[agent_id]
        if len(lots) <= 1:
            return
        for index in range(len(lots) - 1):
            self.record(
                "InstrumentMerged",
                setup=setup,
                keep=f"cash:{agent_id}",
                removed=f"cash:{agent_id}:merged:{self.day}:{index}",
            )
        total = sum(lots, ZERO)
        self.cash_lots[agent_id] = [total] if total > ZERO else []

    def mint_cash(self, agent_id: str, amount: Decimal, *, alias: str | None = None, setup: bool = False) -> None:
        self.cash[agent_id] += amount
        self._add_cash_lot(agent_id, amount)
        self.cash_minted_total += amount
        data: dict[str, Any] = {"to": agent_id, "amount": amount}
        if alias is not None:
            data["alias"] = alias
        self.record("CashMinted", setup=setup, **data)

    def transfer_cash(self, from_agent: str, to_agent: str, amount: Decimal, *, setup: bool = False) -> Decimal:
        if amount <= ZERO:
            return ZERO
        self._require(self.cash[from_agent], amount, f"{from_agent} cash")
        pieces = self._take_cash_lots(from_agent, amount)
        self.cash[from_agent] -= amount
        self.cash[to_agent] += amount
        for piece in pieces:
            self._add_cash_lot(to_agent, piece)
            self.record("CashTransferred", setup=setup, frm=from_agent, to=to_agent, amount=piece)
        self._merge_cash_lots(to_agent, setup=setup)
        return amount

    def deposit_cash(self, customer_id: str, bank_id: str, amount: Decimal, *, setup: bool = False) -> None:
        self._require(self.cash[customer_id], amount, f"{customer_id} cash")
        self._take_cash_lots(customer_id, amount)
        self.cash[customer_id] -= amount
        self.cash[bank_id] += amount
        self._add_cash_lot(bank_id, amount)
        self.deposits[(customer_id, bank_id)] += amount
        self.record("CashDeposited", setup=setup, customer=customer_id, bank=bank_id, amount=amount)

    def withdraw_cash(self, customer_id: str, bank_id: str, amount: Decimal, *, setup: bool = False) -> None:
        self._require(self.deposits[(customer_id, bank_id)], amount, f"{customer_id} deposit at {bank_id}")
        self._require(self.cash[bank_id], amount, f"{bank_id} cash")
        self._take_cash_lots(bank_id, amount)
        self.deposits[(customer_id, bank_id)] -= amount
        self.cash[bank_id] -= amount
        self.cash[customer_id] += amount
        self._add_cash_lot(customer_id, amount)
        self.record("CashWithdrawn", setup=setup, customer=customer_id, bank=bank_id, amount=amount)

    def burn_bank_cash(self, bank_id: str, *, setup: bool = False) -> None:
        burned = self.cash[bank_id]
        if burned:
            self.cash[bank_id] = ZERO
            self.cash_lots[bank_id].clear()
            self.cash_burned_total += burned
            self.record("BankCashBurned", setup=setup, bank_id=bank_id, amount=burned)

    # -- reserve operations ---------------------------------------------------

    def mint_reserves(self, bank_id: str, amount: Decimal, *, alias: str | None = None, setup: bool = False) -> None:
        self.reserves[bank_id] += amount
        self.cb_reserves_outstanding += amount
        data: dict[str, Any] = {"to": bank_id, "amount": amount}
        if alias is not None:
            data["alias"] = alias
        self.record("ReservesMinted", setup=setup, **data)

    def transfer_reserves(
        self,
        from_bank: str,
        to_bank: str,
        amount: Decimal,
        *,
        always_merge: bool = False,
        setup: bool = False,
    ) -> None:
        if from_bank == to_bank:
            raise ValueError("no-op transfer")
        receiver_before = self.reserves[to_bank]
        self._require(self.reserves[from_bank], amount, f"{from_bank} reserves")
        self.reserves[from_bank] -= amount
        self.reserves[to_bank] += amount
        self.record("ReservesTransferred", setup=setup, frm=from_bank, to=to_bank, amount=amount)
        if always_merge or receiver_before:
            self.record(
                "InstrumentMerged",
                setup=setup,
                keep=f"reserve:{to_bank}",
                removed=f"transfer:{from_bank}:{to_bank}:{self.day}",
            )

    # -- deposit transfers ----------------------------------------------------

    def primary_bank_for_customer(self, customer_id: str) -> str | None:
        for (candidate_customer, bank_id), amount in self.deposits.items():
            if candidate_customer == customer_id and amount > 0:
                return bank_id
        return None

    def move_deposit(
        self,
        payer: str,
        payer_bank: str,
        payee: str,
        payee_bank: str,
        amount: Decimal,
        *,
        setup: bool = False,
    ) -> None:
        """Move a deposit balance between customers, emitting the payment event."""
        self.deposits[(payer, payer_bank)] -= amount
        self.deposits[(payee, payee_bank)] += amount
        if payer_bank == payee_bank:
            self.record("IntraBankPayment", setup=setup, payer=payer, payee=payee, bank=payer_bank, amount=amount)
        else:
            self.record(
                "ClientPayment",
                setup=setup,
                payer=payer,
                payer_bank=payer_bank,
                payee=payee,
                payee_bank=payee_bank,
                amount=amount,
            )

    # -- contract creation ------------------------------------------------------

    def unique_contract_id(self, prefix: str, preferred_index: int) -> str:
        used = {
            *self.stocks.keys(),
            *(payable.id for payable in self.payables),
            *(obligation.id for obligation in self.delivery_obligations),
            *(loan.id for loan in self.cb_loans),
            *(loan.id for loan in self.non_bank_loans),
            *(loan.id for loan in self.bank_loans),
        }
        candidate = f"{prefix}_{preferred_index}"
        if candidate not in used:
            return candidate
        next_index = 0
        while True:
            candidate = f"{prefix}_{next_index}"
            if candidate not in used:
                return candidate
            next_index += 1

    def create_payable(
        self,
        *,
        payable_id: str,
        debtor: str,
        creditor: str,
        amount: Decimal,
        due_day: int,
        maturity_distance: int,
        alias: str | None = None,
        setup: bool = False,
        reason: str | None = None,
    ) -> Payable:
        payable = Payable(
            id=payable_id,
            debtor=debtor,
            creditor=creditor,
            amount=amount,
            due_day=due_day,
            maturity_distance=maturity_distance,
            alias=alias,
        )
        self.payables.append(payable)
        # Two observable payload shapes exist: scenario-created payables and
        # receivable-reassignment payables. Keep both exact.
        if reason is None:
            data: dict[str, Any] = {
                "debtor": debtor,
                "creditor": creditor,
                "amount": amount,
                "due_day": due_day,
                "maturity_distance": maturity_distance,
                "payable_id": payable_id,
                "alias": alias,
            }
        else:
            data = {
                "contract_id": payable_id,
                "debtor": debtor,
                "creditor": creditor,
                "amount": amount,
                "due_day": due_day,
                "maturity_distance": maturity_distance,
                "reason": reason,
            }
        self.record("PayableCreated", setup=setup, **data)
        return payable

    def create_cb_loan(
        self,
        *,
        loan_id: str,
        bank: str,
        amount: Decimal,
        rate: Decimal,
        issuance_day: int,
        alias: str | None = None,
        setup: bool = False,
    ) -> CBLoan:
        if self.central_bank_id is None:
            raise ValueError("No central bank found for create_cb_loan")
        loan = CBLoan(
            id=loan_id,
            bank=bank,
            central_bank=self.central_bank_id,
            amount=amount,
            rate=rate,
            issuance_day=issuance_day,
            alias=alias,
        )
        self.cb_loans.append(loan)
        self.record(
            "CBLoanCreated",
            setup=setup,
            bank=loan.bank,
            amount=loan.amount,
            rate=str(loan.rate),
            issuance_day=loan.issuance_day,
            loan_id=loan.id,
            alias=loan.alias,
        )
        return loan

    def create_stock(
        self,
        *,
        stock_id: str,
        owner: str,
        sku: str,
        quantity: int,
        unit_price: Decimal,
        setup: bool = False,
    ) -> StockLot:
        lot = StockLot(id=stock_id, owner=owner, sku=sku, quantity=quantity, unit_price=unit_price)
        self.stocks[stock_id] = lot
        self.record(
            "StockCreated",
            setup=setup,
            owner=owner,
            sku=sku,
            qty=quantity,
            unit_price=unit_price,
            stock_id=stock_id,
        )
        return lot

    def create_delivery_obligation(
        self,
        *,
        obligation_id: str,
        debtor: str,
        creditor: str,
        sku: str,
        quantity: int,
        unit_price: Decimal,
        due_day: int,
        alias: str | None = None,
        setup: bool = False,
    ) -> DeliveryObligation:
        obligation = DeliveryObligation(
            id=obligation_id,
            debtor=debtor,
            creditor=creditor,
            sku=sku,
            quantity=quantity,
            unit_price=unit_price,
            due_day=due_day,
            alias=alias,
        )
        self.delivery_obligations.append(obligation)
        data: dict[str, Any] = {
            "id": obligation_id,
            "frm": debtor,
            "to": creditor,
            "sku": sku,
            "qty": quantity,
            "due_day": due_day,
            "unit_price": unit_price,
        }
        if alias is not None:
            data["alias"] = alias
        self.record("DeliveryObligationCreated", setup=setup, **data)
        return obligation

    # -- central-bank loan servicing -------------------------------------------

    def refinance_cb_loan(self, bank_id: str, repayment: Decimal, *, rate: Decimal) -> CBLoan:
        """Mint reserves against a fresh CB loan to cover a repayment shortfall."""
        if self.central_bank_id is None:
            raise ValueError("No central bank found for CB refinancing")
        loan_id = f"L_{len(self.cb_loans)}"
        reserve_id = f"R_{self.day}_{len(self.cb_loans)}"
        self.reserves[bank_id] += repayment
        self.cb_reserves_outstanding += repayment
        self.cb_loans_outstanding += repayment
        self.cb_loans_created_count += 1
        loan = CBLoan(
            id=loan_id,
            bank=bank_id,
            central_bank=self.central_bank_id,
            amount=repayment,
            rate=rate,
            issuance_day=self.day,
        )
        self.cb_loans.append(loan)
        self.log(
            "CBLoanCreated",
            bank_id=bank_id,
            amount=repayment,
            loan_id=loan_id,
            reserve_id=reserve_id,
            cb_rate=str(rate),
            maturity_day=self.day + 2,
        )
        return loan

    def repay_cb_loan(self, loan: CBLoan) -> None:
        repayment = loan.repayment_amount
        self._require(self.reserves[loan.bank], repayment, f"{loan.bank} reserves to repay CB loan")
        self.reserves[loan.bank] -= repayment
        loan.settled = True
        self.cb_reserves_outstanding -= repayment
        self.cb_loans_outstanding -= loan.amount
        self.cb_interest_total_paid += loan.interest_amount
        self.log(
            "CBLoanRepaid",
            bank_id=loan.bank,
            loan_id=loan.id,
            principal=loan.amount,
            interest=loan.interest_amount,
            total_repaid=repayment,
        )

    def net_payable(self, payable: Payable, reduction: Decimal) -> None:
        """Extinguish ``reduction`` of a payable's face by netting (no cash moves)."""
        if reduction <= ZERO or reduction > payable.amount:
            raise InvariantViolation(f"invalid netting reduction {reduction} for payable {payable.id} with face {payable.amount}")
        original_amount = payable.amount + payable.netted_amount
        payable.amount -= reduction
        payable.netted_amount += reduction
        if payable.amount == ZERO:
            payable.settled = True
        self.log(
            "PayableNetted",
            pid=payable.id,
            contract_id=payable.id,
            alias=payable.alias,
            debtor=payable.debtor,
            creditor=payable.creditor,
            original_amount=original_amount,
            netted_amount=reduction,
            remaining_amount=payable.amount,
        )

    def add_rollover_payable(
        self,
        *,
        debtor: str,
        creditor: str,
        amount: Decimal,
        due_day: int,
        maturity_distance: int,
    ) -> Payable:
        """Append a rollover-refinanced payable without a creation event.

        Rollover is observable through ``PayableRolledOver``/``RolloverPartial``
        (emitted by the settlement plugin after the cash return-flow), not
        ``PayableCreated`` — matching the existing engine.
        """
        payable = Payable(
            id=f"PAY_rollover_{len(self.payables)}",
            debtor=debtor,
            creditor=creditor,
            amount=amount,
            due_day=due_day,
            maturity_distance=maturity_distance,
        )
        self.payables.append(payable)
        return payable

    # -- dealer cash reconciliation ----------------------------------------------

    def sync_mint_cash(self, agent_id: str, amount: Decimal) -> None:
        """Dealer-subsystem reconciliation: trading gains materialize as cash."""
        self.cash[agent_id] += amount
        self._add_cash_lot(agent_id, amount)
        self.cash_minted_total += amount
        self.log("CashMinted", to=agent_id, amount=amount)

    def sync_retire_cash(self, agent_id: str, amount: Decimal) -> None:
        """Dealer-subsystem reconciliation: trading losses retire cash."""
        self._take_cash_lots(agent_id, amount)
        self.cash[agent_id] -= amount
        self.cash_burned_total += amount
        self.log("CashRetired", frm=agent_id, amount=amount)

    # -- bank loan / resolution operations --------------------------------------

    def create_bank_loan(
        self,
        *,
        bank_id: str,
        borrower_id: str,
        amount: Decimal,
        rate: Decimal,
        maturity: int,
    ) -> BankLoan:
        """Issue a bank loan by crediting the borrower's deposit (no event;
        observable through ``BankLoanIssued``, emitted by the banking plugin)."""
        loan = BankLoan(
            id=f"BL_{len(self.bank_loans)}",
            bank=bank_id,
            borrower=borrower_id,
            amount=amount,
            rate=rate,
            issuance_day=self.day,
            maturity_day=self.day + maturity,
        )
        self.bank_loans.append(loan)
        self.deposits[(borrower_id, bank_id)] += amount
        return loan

    def decrease_deposit(self, agent_id: str, bank_id: str, amount: Decimal) -> Decimal:
        """Debit up to ``amount`` from a deposit balance (no event)."""
        debited = min(self.deposits[(agent_id, bank_id)], amount)
        if debited > ZERO:
            self.deposits[(agent_id, bank_id)] -= debited
        return debited

    def credit_deposit(self, agent_id: str, bank_id: str, amount: Decimal) -> None:
        """Credit a deposit balance (no event; caller emits the domain event)."""
        self.deposits[(agent_id, bank_id)] += amount

    def move_reserves_logged(self, from_bank: str, to_bank: str, amount: Decimal, **extra: Any) -> None:
        """Move reserves with a bare ``ReservesTransferred`` event (no merge event)."""
        self.reserves[from_bank] -= amount
        self.reserves[to_bank] += amount
        self.log("ReservesTransferred", frm=from_bank, to=to_bank, amount=amount, **extra)

    def convert_reserves_to_cash(self, bank_id: str, depositor_id: str, amount: Decimal, *, instr_id: str) -> None:
        """Bank-resolution payout: failed bank's reserves become depositor cash."""
        self.reserves[bank_id] -= amount
        self.cb_reserves_outstanding -= amount
        self.cash_converted_from_reserves += amount
        self.log("ReservesToCash", bank_id=bank_id, amount=amount, instr_id=instr_id)
        self.cash[depositor_id] += amount
        self._add_cash_lot(depositor_id, amount)
        self.log("CashTransferred", frm=bank_id, to=depositor_id, amount=amount, instr_id=instr_id)

    # -- non-bank loan operations ----------------------------------------------

    def disburse_non_bank_loan(
        self,
        *,
        lender_id: str,
        borrower_id: str,
        amount: Decimal,
        rate: Decimal,
        maturity_days: int,
    ) -> NonBankLoan:
        """Create a non-bank loan, moving lender cash to the borrower.

        The cash movement is deliberately silent (no ``CashTransferred``
        events) — loan disbursement is observable only through
        ``NonBankLoanCreated``, matching the existing engine.
        """
        self._require(self.cash[lender_id], amount, f"{lender_id} cash")
        self._take_cash_lots(lender_id, amount)
        self.cash[lender_id] -= amount
        self.cash[borrower_id] += amount
        self._add_cash_lot(borrower_id, amount)
        return self.record_non_bank_loan(
            lender_id=lender_id,
            borrower_id=borrower_id,
            amount=amount,
            rate=rate,
            maturity_days=maturity_days,
        )

    def record_non_bank_loan(
        self,
        *,
        lender_id: str,
        borrower_id: str,
        amount: Decimal,
        rate: Decimal,
        maturity_days: int,
    ) -> NonBankLoan:
        """Record a disbursed loan (funding already moved by the caller)."""
        loan_id = f"NBL_{len(self.non_bank_loans)}"
        loan = NonBankLoan(
            id=loan_id,
            lender=lender_id,
            borrower=borrower_id,
            amount=amount,
            rate=rate,
            issuance_day=self.day,
            maturity_days=maturity_days,
        )
        self.non_bank_loans.append(loan)
        self.log(
            "NonBankLoanCreated",
            lender_id=lender_id,
            borrower_id=borrower_id,
            amount=amount,
            loan_id=loan_id,
            rate=str(rate),
            maturity_day=loan.maturity_day,
        )
        return loan

    def default_non_bank_loan(self, loan: NonBankLoan, *, borrower_liquid: Decimal) -> None:
        """Write off a matured loan the borrower cannot repay (lender absorbs)."""
        loan.settled = True
        self.log(
            "NonBankLoanDefaulted",
            loan_id=loan.id,
            borrower_id=loan.borrower,
            lender_id=loan.lender,
            amount_owed=loan.repayment_amount,
            cash_available=borrower_liquid,
        )

    def repay_non_bank_loan_with_cash(self, loan: NonBankLoan) -> None:
        """Repay a matured loan from borrower cash (silent movement, like disbursal)."""
        repayment = loan.repayment_amount
        self._take_cash_lots(loan.borrower, repayment)
        self.cash[loan.borrower] -= repayment
        self.cash[loan.lender] += repayment
        self._add_cash_lot(loan.lender, repayment)
        self.mark_non_bank_loan_repaid(loan)

    def mark_non_bank_loan_repaid(self, loan: NonBankLoan) -> None:
        loan.settled = True
        self.log(
            "NonBankLoanRepaid",
            loan_id=loan.id,
            borrower_id=loan.borrower,
            lender_id=loan.lender,
            principal=loan.amount,
            interest=loan.interest_amount,
            total_repaid=loan.repayment_amount,
        )

    # -- inventory operations ------------------------------------------------

    def first_stock_lot_by_sku(self, owner: str, sku: str) -> StockLot | None:
        for stock in self.stocks.values():
            if stock.owner == owner and stock.sku == sku:
                return stock
        return None

    def move_stock_lot(
        self,
        stock: StockLot,
        from_agent: str,
        to_agent: str,
        quantity: int,
        *,
        setup: bool = False,
    ) -> str:
        moving_id = stock.id
        if quantity < stock.quantity:
            original_quantity = stock.quantity
            moving_id = f"S_split_{stock.id}_{self.day}_{len(self.stocks)}"
            self.stocks[moving_id] = StockLot(
                id=moving_id,
                owner=stock.owner,
                sku=stock.sku,
                quantity=quantity,
                unit_price=stock.unit_price,
            )
            stock.quantity -= quantity
            self.record(
                "StockSplit",
                setup=setup,
                original_id=stock.id,
                new_id=moving_id,
                sku=stock.sku,
                original_qty=original_quantity,
                split_qty=quantity,
                remaining_qty=stock.quantity,
            )
        moving_stock = self.stocks[moving_id]
        if moving_stock.owner != from_agent:
            raise ValueError("Stock owner mismatch")
        moving_stock.owner = to_agent
        return moving_id

    # -- contract queries -------------------------------------------------------

    def contract_id_for_alias(self, alias: str | None) -> str | None:
        if alias is None:
            return None
        for payable in self.payables:
            if payable.alias == alias:
                return payable.id
        for obligation in self.delivery_obligations:
            if obligation.alias == alias:
                return obligation.id
        return None

    def agent_liquid_assets(self, agent_id: str) -> Decimal:
        deposits = sum(
            (amount for (customer_id, _bank_id), amount in self.deposits.items() if customer_id == agent_id),
            ZERO,
        )
        return self.cash[agent_id] + deposits

    # -- checkpoint / rollback ----------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            cash=self.cash.copy(),
            cash_lots={agent_id: list(lots) for agent_id, lots in self.cash_lots.items()},
            deposits=self.deposits.copy(),
            stocks={
                stock_id: StockLot(
                    id=stock.id,
                    owner=stock.owner,
                    sku=stock.sku,
                    quantity=stock.quantity,
                    unit_price=stock.unit_price,
                )
                for stock_id, stock in self.stocks.items()
            },
            journal_length=len(self.journal),
        )

    def restore(self, checkpoint: Checkpoint, *, restore_stocks: bool = False) -> None:
        self.cash = checkpoint.cash
        self.cash_lots = defaultdict(list, checkpoint.cash_lots)
        self.deposits = checkpoint.deposits
        if restore_stocks:
            self.stocks = checkpoint.stocks
        self.journal.truncate(checkpoint.journal_length)

    # -- invariants -----------------------------------------------------------

    def check_invariants(self) -> None:
        for agent_id, balance in self.cash.items():
            if balance < ZERO:
                raise InvariantViolation(f"negative cash for {agent_id}: {balance}")
            lots_total = sum(self.cash_lots.get(agent_id, []), ZERO)
            if lots_total != balance:
                raise InvariantViolation(f"cash lots for {agent_id} sum to {lots_total}, balance is {balance}")
        for bank_id, balance in self.reserves.items():
            if balance < ZERO:
                raise InvariantViolation(f"negative reserves for {bank_id}: {balance}")
        for (customer, bank), balance in self.deposits.items():
            if balance < ZERO:
                raise InvariantViolation(f"negative deposit for {customer} at {bank}: {balance}")
        cash_in_circulation = sum(self.cash.values(), ZERO)
        expected_cash = self.cash_minted_total - self.cash_burned_total + self.cash_converted_from_reserves
        if cash_in_circulation != expected_cash:
            raise InvariantViolation(
                f"cash conservation broken: in circulation {cash_in_circulation}, minted-burned+converted {expected_cash}"
            )
        reserves_in_circulation = sum(self.reserves.values(), ZERO)
        if reserves_in_circulation != self.cb_reserves_outstanding:
            raise InvariantViolation(
                f"reserve conservation broken: in circulation {reserves_in_circulation}, outstanding {self.cb_reserves_outstanding}"
            )
