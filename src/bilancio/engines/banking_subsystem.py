"""Banking subsystem for integrating Treynor pricing into the main simulation.

Bridges the banking kernel (bilancio.banking) into the main engine,
providing active bank state management, quote computation, and
multi-bank routing for the Kalecki ring.

Analogous to DealerSubsystem for dealer/VBT integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bilancio.banking.pricing_kernel import (
    PricingParams,
    compute_inventory,
    compute_quotes,
)
from bilancio.banking.types import Quote
from bilancio.decision.profiles import BankProfile
from bilancio.domain.instruments.base import InstrumentKind

if TYPE_CHECKING:
    from bilancio.domain.agents.central_bank import CentralBank
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)


@dataclass
class BankLoanRecord:
    """A loan from bank to trader, tracked in the banking subsystem."""

    loan_id: str
    bank_id: str
    borrower_id: str
    principal: int
    rate: Decimal
    issuance_day: int
    maturity_day: int

    @property
    def repayment_amount(self) -> int:
        return int(self.principal * (1 + self.rate))


@dataclass
class InterbankLoan:
    """Interbank loan between two banks."""

    lender_bank: str
    borrower_bank: str
    amount: int
    rate: Decimal
    issuance_day: int
    maturity_day: int

    @property
    def repayment_amount(self) -> int:
        return int(self.amount * (1 + self.rate))


@dataclass
class BankTreynorState:
    """Active bank state for Treynor pricing within the main simulation."""

    bank_id: str
    pricing_params: PricingParams

    # Cached quotes (refreshed each day and after significant events)
    current_quote: Quote | None = None

    # Lending book (loan_id -> BankLoanRecord)
    outstanding_loans: dict[str, BankLoanRecord] = field(default_factory=dict)
    total_loan_principal: int = 0

    # Deposit outflow forecast (W_t^rem per paper Section 6.1)
    withdrawal_forecast: int = 0  # W_t^c — expected total withdrawals today
    realized_withdrawals: int = 0  # W_t^real — cumulative realized withdrawals today

    # Settlement outflow forecast (set by BankingSubsystem before refresh_quote)
    _settlement_net_outflow: int = 0  # net cross-bank outflow from ring settlements today

    # Reserve projection cache (set by refresh_quote)
    min_projected_reserves: int = 0  # min(path) from last refresh_quote

    def compute_withdrawal_forecast(self, system: System, n_banks: int) -> int:
        """Compute expected deposit outflows from loan-origin deposits.

        W_t^c = sum of loan-origin deposits still at this bank × cross_bank_fraction.
        Cross-bank fraction = (n_banks - 1) / n_banks — the probability that
        the borrower's next payment goes to a client of another bank.
        """
        if n_banks <= 1:
            return 0

        cross_bank_fraction = Decimal(n_banks - 1) / Decimal(n_banks)
        total_loan_deposits = 0

        # Deduplicate by borrower: a borrower with multiple loans still has
        # only one deposit balance, so count each borrower's deposit once.
        seen_borrowers: set[str] = set()
        for loan_rec in self.outstanding_loans.values():
            if loan_rec.borrower_id in seen_borrowers:
                continue
            seen_borrowers.add(loan_rec.borrower_id)
            deposit_at_bank = _get_deposit_at_bank(
                system, loan_rec.borrower_id, self.bank_id
            )
            total_loan_deposits += deposit_at_bank

        return int(total_loan_deposits * cross_bank_fraction)

    def record_withdrawal(self, amount: int) -> None:
        """Record a realized deposit withdrawal (cross-bank payment outflow)."""
        self.realized_withdrawals += amount

    def reset_daily_tracking(self) -> None:
        """Reset intraday withdrawal tracking at start of new day."""
        self.realized_withdrawals = 0
        self.withdrawal_forecast = 0

    def refresh_quote(
        self,
        system: System,
        current_day: int,
        n_banks: int = 1,
        interbank_loans: list[InterbankLoan] | None = None,
    ) -> Quote:
        """Recompute (r_D, r_L) from bank's current balance sheet.

        Uses a 10-day reserve projection path (per paper Section 6.1.3):
        1. Build path[t] = effective_reserves (after W_t^rem)
        2. For s in 1..10, add scheduled legs (CB loan repayments, bank loan repayments)
        3. Cash-tightness L* = max(0, reserve_floor - min(path)) / reserve_floor
        4. Scale L* by CB pressure (Plan 042): when CB is near its lending cap,
           the bank cannot assume CB backstop → amplify cash-tightness
        5. Risk index rho = L* (simple version, matches simple_risk_index)
        6. Inventory x = path[t+2] - reserve_target (projected t+2, not current)
        """
        from bilancio.domain.instruments.cb_loan import CBLoan

        reserves = _get_bank_reserves(system, self.bank_id)
        reserve_target = self.pricing_params.reserve_target
        reserve_floor = self.pricing_params.reserve_floor

        # --- Deposit outflow forecast (W_t^rem) ---
        W_t_c = self.compute_withdrawal_forecast(system, n_banks)
        self.withdrawal_forecast = W_t_c
        W_t_rem = max(0, W_t_c - self.realized_withdrawals)

        # --- Build 10-day reserve projection path ---
        # Include settlement outflow: net cross-bank drain from ring payments
        # due today. This is the dominant reserve drain at low κ.
        settlement_drain = max(0, self._settlement_net_outflow)
        path = [0] * 11  # path[0] = today, path[1..10] = future days
        path[0] = reserves - W_t_rem - settlement_drain

        for s in range(1, 11):
            proj_day = current_day + s
            delta = 0

            # CB loan repayments due on proj_day (outflow: bank pays back principal + interest)
            bank_agent = system.state.agents.get(self.bank_id)
            if bank_agent:
                for cid in bank_agent.liability_ids:
                    contract = system.state.contracts.get(cid)
                    if (
                        contract is not None
                        and contract.kind == InstrumentKind.CB_LOAN
                        and isinstance(contract, CBLoan)
                        and contract.maturity_day == proj_day
                    ):
                        delta -= contract.repayment_amount

            # Bank loan repayments are NOT counted in the reserve projection.
            #
            # Rationale: In the paper, the projection counts "old contract legs"
            # from relatively safe interbank deposits and wholesale funding.
            # Our bank loans are to stressed Kalecki ring agents who frequently
            # default (19%+ default rate observed). If we count repayments as
            # certain inflows, each loan *improves* the projected path, driving
            # rates to deeply negative values (the broken feedback loop).
            #
            # By excluding loan repayments, the bank conservatively prices as if
            # it won't get repaid. Each new loan only has costs (deposit outflow
            # via W_t^rem, CB repayment obligations) with no offsetting inflows.
            # This makes rates rise with lending volume — the correct behavior.
            # When repayments DO arrive, actual reserves improve, and the next
            # quote naturally reflects the better position.

            # Interbank loan legs (overnight, risk-free between banks)
            if interbank_loans:
                for ib_loan in interbank_loans:
                    if ib_loan.maturity_day == proj_day:
                        if ib_loan.lender_bank == self.bank_id:
                            delta += ib_loan.repayment_amount  # inflow
                        elif ib_loan.borrower_bank == self.bank_id:
                            delta -= ib_loan.repayment_amount  # outflow

            path[s] = path[s - 1] + delta

        # --- Cash-tightness: L* = max(0, reserve_floor - min(path)) / reserve_floor ---
        min_path = min(path)
        self.min_projected_reserves = min_path
        if reserve_floor > 0:
            cash_tightness = max(Decimal("0"), Decimal(reserve_floor - min_path) / Decimal(reserve_floor))
        else:
            cash_tightness = Decimal("0")

        # --- CB pressure overlay (Plan 042) ---
        # When the CB is near its lending cap or has frozen, the bank cannot
        # assume the CB backstop will be available to top up reserves.
        # Scale L* by (1 + cb_pressure) to reflect this uncertainty.
        # cb_pressure = 0 when CB is unconstrained → no change (backward compat)
        # cb_pressure → ∞ when CB is at cap or frozen → L* dominates pricing
        cb_pressure = _compute_cb_pressure(system)
        if cb_pressure > 0 and cash_tightness > 0:
            cash_tightness = cash_tightness * (1 + cb_pressure)

        # --- Risk index: rho = L* (simple version) ---
        risk_index = cash_tightness

        # --- Inventory: x = path[t+2] - reserve_target (projected, not current) ---
        projected_t2 = path[min(2, len(path) - 1)]
        inventory = compute_inventory(projected_t2, reserve_target)

        # Compute quote from Treynor kernel
        self.current_quote = compute_quotes(
            inventory=inventory,
            cash_tightness=cash_tightness,
            risk_index=risk_index,
            params=self.pricing_params,
            day=current_day,
        )
        return self.current_quote


@dataclass
class BankingSubsystem:
    """Active banking state within the main simulation engine.

    Holds all bank states, assignment maps, and configuration.
    Analogous to DealerSubsystem for dealer/VBT integration.
    """

    banks: dict[str, BankTreynorState]  # bank_id -> state
    bank_profile: BankProfile
    kappa: Decimal

    # Assignment maps
    trader_banks: dict[str, list[str]] = field(
        default_factory=dict
    )  # trader_id -> [bank_id, ...]
    infra_banks: dict[str, str] = field(
        default_factory=dict
    )  # infra_agent_id -> bank_id

    # Interbank lending book
    interbank_loans: list[InterbankLoan] = field(default_factory=list)

    # Configuration (derived from BankProfile + scenario)
    loan_maturity: int = 5  # Days
    interest_period: int = 2  # Days per interest accrual

    # Risk assessor for credit-risk-adjusted lending rates (optional)
    risk_assessor: Any = None

    # "Fool me once" — borrowers who defaulted on a bank loan are blocked
    defaulted_borrowers: set[str] = field(default_factory=set)

    def best_deposit_bank(self, agent_id: str) -> str | None:
        """Return bank_id with highest r_D among this agent's banks."""
        bank_ids = self._get_agent_banks(agent_id)
        if not bank_ids:
            return None

        best_rate = Decimal("-1")
        best_bank = bank_ids[0]
        for bank_id in bank_ids:
            state = self.banks.get(bank_id)
            if state and state.current_quote:
                if state.current_quote.deposit_rate > best_rate:
                    best_rate = state.current_quote.deposit_rate
                    best_bank = bank_id
        return best_bank

    def cheapest_loan_bank(self, agent_id: str) -> str | None:
        """Return bank_id with lowest r_L among this agent's banks."""
        bank_ids = self._get_agent_banks(agent_id)
        if not bank_ids:
            return None

        best_rate = Decimal("999")
        best_bank = bank_ids[0]
        for bank_id in bank_ids:
            state = self.banks.get(bank_id)
            if state and state.current_quote:
                if state.current_quote.loan_rate < best_rate:
                    best_rate = state.current_quote.loan_rate
                    best_bank = bank_id
        return best_bank

    def cheapest_pay_bank(self, agent_id: str) -> str | None:
        """Return bank_id with lowest r_D (min opportunity cost for payer)."""
        bank_ids = self._get_agent_banks(agent_id)
        if not bank_ids:
            return None

        best_rate = Decimal("999")
        best_bank = bank_ids[0]
        for bank_id in bank_ids:
            state = self.banks.get(bank_id)
            if state and state.current_quote:
                if state.current_quote.deposit_rate < best_rate:
                    best_rate = state.current_quote.deposit_rate
                    best_bank = bank_id
        return best_bank

    def refresh_all_quotes(self, system: System, current_day: int) -> None:
        """Refresh quotes for all banks."""
        n_banks = len(self.banks)
        # Compute settlement forecasts first so refresh_quote can use them
        settlement = self.compute_settlement_forecasts(system, current_day)
        for bank_state in self.banks.values():
            bank_state._settlement_net_outflow = settlement.get(bank_state.bank_id, 0)
            bank_state.refresh_quote(
                system, current_day, n_banks,
                interbank_loans=self.interbank_loans,
            )

    def compute_settlement_forecasts(
        self, system: System, current_day: int
    ) -> dict[str, int]:
        """Estimate net reserve outflow per bank from upcoming ring settlements.

        Mirrors the actual settlement routing in ``_pay_with_deposits``:
        - Debtor pays from banks sorted by ascending r_D (cheapest first),
          split by actual deposit balance at each bank.
        - Creditor receives at bank with highest r_D.
        - Uses ``effective_creditor`` for payables transferred in secondary
          market (matches settlement's use of ``payable.effective_creditor``).

        Returns {bank_id: net_outflow} where positive = reserves leave.
        """
        net: dict[str, int] = dict.fromkeys(self.banks, 0)

        for contract in system.state.contracts.values():
            if contract.kind != InstrumentKind.PAYABLE:
                continue
            due_day = getattr(contract, "due_day", None)
            if due_day is None or due_day != current_day:
                continue

            debtor_id = contract.liability_issuer_id
            creditor_id = contract.effective_creditor
            amount = contract.amount

            # Skip if debtor is defaulted/missing
            debtor = system.state.agents.get(debtor_id)
            if debtor is None or debtor.defaulted:
                continue

            # Creditor bank: highest r_D (mirrors _select_receive_bank)
            creditor_bank = self._forecast_creditor_bank(system, creditor_id)
            if creditor_bank is None:
                continue

            # Debtor bank splits: ascending r_D (mirrors _pay_with_deposits)
            debtor_splits = self._forecast_debtor_bank_splits(
                system, debtor_id, amount
            )
            if not debtor_splits:
                continue

            # Record reserve flows for each split chunk
            for pay_bank, chunk in debtor_splits:
                if pay_bank == creditor_bank:
                    continue  # intra-bank, no reserve movement
                if pay_bank in net:
                    net[pay_bank] += chunk
                if creditor_bank in net:
                    net[creditor_bank] -= chunk

        return net

    def _forecast_creditor_bank(
        self, system: System, creditor_id: str
    ) -> str | None:
        """Select creditor's receive bank (highest r_D).

        Mirrors ``_select_receive_bank`` in settlement: scans the creditor's
        actual BANK_DEPOSIT contracts and picks the issuing bank with the
        highest deposit rate.  Falls back to the ``trader_banks`` mapping
        if the creditor has no deposit contracts.
        """
        agent = system.state.agents.get(creditor_id)
        if agent is None:
            return None

        best_rate = Decimal("-1")
        best_bank: str | None = None

        for cid in agent.asset_ids:
            contract = system.state.contracts.get(cid)
            if contract is None or contract.kind != InstrumentKind.BANK_DEPOSIT:
                continue
            bid = contract.liability_issuer_id
            bank_state = self.banks.get(bid)
            r_d = Decimal("0")
            if bank_state and bank_state.current_quote:
                r_d = bank_state.current_quote.deposit_rate
            if r_d > best_rate:
                best_rate = r_d
                best_bank = bid

        if best_bank is not None:
            return best_bank

        # Fallback: use bank assignment mapping
        return self.best_deposit_bank(creditor_id)

    def _forecast_debtor_bank_splits(
        self, system: System, debtor_id: str, amount: int
    ) -> list[tuple[str, int]]:
        """Estimate how settlement splits a payment across debtor's banks.

        Mirrors ``_pay_with_deposits``: groups the debtor's deposit contracts
        by issuing bank, sorts by ascending r_D (cheapest to withdraw first),
        and splits the payment amount across banks in that order.

        Returns list of ``(bank_id, chunk)`` tuples.
        """
        agent = system.state.agents.get(debtor_id)
        if agent is None:
            return []

        # Group deposit balances by bank
        bank_balances: dict[str, int] = {}
        for cid in agent.asset_ids:
            contract = system.state.contracts.get(cid)
            if contract is None or contract.kind != InstrumentKind.BANK_DEPOSIT:
                continue
            bid = contract.liability_issuer_id
            bank_balances[bid] = bank_balances.get(bid, 0) + contract.amount

        if not bank_balances:
            return []

        # Sort by ascending r_D (cheapest to withdraw first)
        sorted_banks: list[tuple[Decimal, str, int]] = []
        for bid, balance in bank_balances.items():
            if balance <= 0:
                continue
            bank_state = self.banks.get(bid)
            r_d = Decimal("0")
            if bank_state and bank_state.current_quote:
                r_d = bank_state.current_quote.deposit_rate
            sorted_banks.append((r_d, bid, balance))
        sorted_banks.sort(key=lambda x: x[0])

        # Split payment across banks
        splits: list[tuple[str, int]] = []
        remaining = amount
        for _, bid, balance in sorted_banks:
            if remaining <= 0:
                break
            chunk = min(balance, remaining)
            splits.append((bid, chunk))
            remaining -= chunk

        return splits

    def has_outstanding_loan(self, borrower_id: str) -> bool:
        """Check if a borrower has any outstanding loan at any bank."""
        for bank_state in self.banks.values():
            for loan in bank_state.outstanding_loans.values():
                if loan.borrower_id == borrower_id:
                    return True
        return False

    def all_outstanding_loans(self) -> list[tuple[str, BankLoanRecord]]:
        """Return all outstanding bank loans across all banks."""
        result = []
        for bank_state in self.banks.values():
            for loan_id, loan in bank_state.outstanding_loans.items():
                result.append((loan_id, loan))
        return result

    def update_cb_corridor(self, system: System) -> None:
        """Sync CB corridor into PricingParams.

        Handles two independent corridor adjustment mechanisms:
        1. Rate escalation: CB lending rate rises with outstanding CB loans.
        2. Deviation from expectation (Plan 041): corridor mid shifts up
           and width widens when realized defaults exceed the kappa-informed
           prior (P_0).

        Both mechanisms stack: escalation raises the ceiling further after
        deviation-from-expectation has set the base corridor.
        """
        cb = _find_central_bank(system)
        if cb is None:
            return

        # --- Plan 041: Deviation-from-expectation corridor ---
        if cb.kappa_prior > 0:
            n_total = len(system.state.agents)
            n_defaulted = len(system.state.defaulted_agent_ids)
            r_floor, r_ceiling = cb.compute_corridor(n_defaulted, n_total)
            for bank_state in self.banks.values():
                bank_state.pricing_params.reserve_remuneration_rate = r_floor
                bank_state.pricing_params.cb_borrowing_rate = r_ceiling
        else:
            # No kappa prior — reset to BankProfile's base corridor each day
            # to prevent escalation from compounding across calls.
            r_floor = self.bank_profile.r_floor(self.kappa)
            r_ceiling = self.bank_profile.r_ceiling(self.kappa)
            for bank_state in self.banks.values():
                bank_state.pricing_params.reserve_remuneration_rate = r_floor
                bank_state.pricing_params.cb_borrowing_rate = r_ceiling

        # --- Existing: rate escalation stacked on top ---
        if cb.rate_escalation_slope > 0 and cb.escalation_base_amount > 0:
            outstanding = system.state.cb_loans_outstanding
            utilization = Decimal(outstanding) / Decimal(cb.escalation_base_amount)
            escalation_increment = cb.rate_escalation_slope * utilization
            for bank_state in self.banks.values():
                bank_state.pricing_params.cb_borrowing_rate += escalation_increment

    def _get_agent_banks(self, agent_id: str) -> list[str]:
        """Get list of bank_ids for an agent."""
        # Check trader assignment first
        if agent_id in self.trader_banks:
            return self.trader_banks[agent_id]
        # Check infra assignment
        if agent_id in self.infra_banks:
            return [self.infra_banks[agent_id]]
        # Fallback: return all banks
        return list(self.banks.keys())


# ---------------------------------------------------------------------------
# Helper functions for reading bank state from the main System
# ---------------------------------------------------------------------------


def _get_bank_reserves(system: System, bank_id: str) -> int:
    """Get total reserves held by a bank."""
    agent = system.state.agents.get(bank_id)
    if agent is None:
        return 0
    total = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == InstrumentKind.RESERVE_DEPOSIT:
            total += contract.amount
    return total


def _get_bank_deposits_total(system: System, bank_id: str) -> int:
    """Get total deposit liabilities of a bank."""
    agent = system.state.agents.get(bank_id)
    if agent is None:
        return 0
    total = 0
    for cid in agent.liability_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == InstrumentKind.BANK_DEPOSIT:
            total += contract.amount
    return total


def _get_deposit_at_bank(system: System, agent_id: str, bank_id: str) -> int:
    """Get an agent's deposit balance at a specific bank."""
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    total = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if (
            contract
            and contract.kind == InstrumentKind.BANK_DEPOSIT
            and contract.liability_issuer_id == bank_id
        ):
            total += contract.amount
    return total


def _get_total_deposits(system: System, agent_id: str) -> int:
    """Get an agent's total deposits across all banks."""
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    total = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == InstrumentKind.BANK_DEPOSIT:
            total += contract.amount
    return total


def _find_central_bank(system: System) -> CentralBank | None:
    """Find the CentralBank agent in the system."""
    from bilancio.domain.agents.central_bank import CentralBank

    for agent in system.state.agents.values():
        if isinstance(agent, CentralBank):
            return agent
    return None


def _compute_cb_pressure(system: System) -> Decimal:
    """Compute CB pressure: how constrained is the CB's lending capacity?

    Returns a non-negative scalar:
    - 0 when CB is unconstrained (no cap, backward compat)
    - Rises as CB utilization approaches the cap
    - Very large when CB lending is frozen

    The formula is:  pressure = utilization^2 / (1 - utilization)
    This gives:
    - utilization=0.0 → pressure=0.00 (no stress)
    - utilization=0.5 → pressure=0.50
    - utilization=0.8 → pressure=3.20
    - utilization=0.9 → pressure=8.10
    - utilization=1.0 → pressure capped at 10.0

    When CB is frozen: pressure = 10.0 (maximum).
    """
    # Frozen = maximum pressure
    if system.state.cb_lending_frozen:
        return Decimal("10")

    cb = _find_central_bank(system)
    if cb is None:
        return Decimal("0")

    # No cap configured → no pressure (backward compat)
    if cb.max_outstanding_ratio <= 0 or cb.escalation_base_amount <= 0:
        return Decimal("0")

    cap = cb.max_outstanding_ratio * Decimal(cb.escalation_base_amount)
    if cap <= 0:
        return Decimal("0")

    utilization = min(Decimal("1"), Decimal(system.state.cb_loans_outstanding) / cap)

    if utilization >= Decimal("1"):
        return Decimal("10")

    # u^2 / (1 - u): convex, rises sharply near capacity
    return utilization * utilization / (Decimal("1") - utilization)


def cb_can_backstop(system: System, needed_amount: int) -> bool:
    """Check if the CB has room to backstop a reserve shortfall of this size.

    Used by banks in lending decisions: before issuing a loan that will
    create a deposit (and an expected cross-bank reserve outflow), the bank
    checks whether the CB can cover the resulting reserve drain.

    Returns True when:
    - CB lending is not frozen, AND
    - CB has enough room under its lending cap

    Returns True (no constraint) when CB has no cap configured (backward compat).
    """
    if system.state.cb_lending_frozen:
        return False

    cb = _find_central_bank(system)
    if cb is None:
        return True  # No CB → no constraint

    return cb.can_lend(system.state.cb_loans_outstanding, needed_amount)


def initialize_banking_subsystem(
    system: System,
    bank_profile: BankProfile,
    kappa: Decimal,
    maturity_days: int,
    trader_banks: dict[str, list[str]] | None = None,
    infra_banks: dict[str, str] | None = None,
    risk_assessor: Any = None,
) -> BankingSubsystem:
    """Initialize the banking subsystem from the current system state.

    Creates BankTreynorState for each bank agent, computes initial
    PricingParams from BankProfile + kappa, and returns the subsystem.

    Args:
        system: The main system with bank agents already created.
        bank_profile: Treynor pricing configuration.
        kappa: System liquidity ratio (for corridor calibration).
        maturity_days: Scenario maturity days (for loan maturity calc).
        trader_banks: Trader -> bank assignment map.
        infra_banks: Infrastructure agent -> bank assignment map.
    """
    from bilancio.domain.agent import AgentKind

    # Find all bank agents
    bank_ids = [
        aid
        for aid, agent in system.state.agents.items()
        if agent.kind == AgentKind.BANK
    ]

    if not bank_ids:
        raise ValueError("No bank agents found in system")

    # Derive corridor rates from kappa
    r_floor = bank_profile.r_floor(kappa)
    r_ceiling = bank_profile.r_ceiling(kappa)

    # Compute loan maturity
    loan_mat = bank_profile.loan_maturity(maturity_days)
    interest_period = bank_profile.interest_period

    # Create BankTreynorState for each bank
    banks: dict[str, BankTreynorState] = {}
    for bank_id in bank_ids:
        _get_bank_reserves(system, bank_id)
        deposits = _get_bank_deposits_total(system, bank_id)

        # Reserve target = ratio * deposits
        reserve_target = max(1, int(bank_profile.reserve_target_ratio * deposits))

        # Symmetric capacity = ratio * reserve_target
        symmetric_capacity = max(
            1, int(bank_profile.symmetric_capacity_ratio * reserve_target)
        )

        # Ticket size: use a reasonable default based on system scale
        # Use average deposit / 10 as ticket size, minimum 100
        ticket_size = max(100, deposits // 10) if deposits > 0 else 100

        # Reserve floor: minimum reserves before CB borrowing
        reserve_floor = max(1, reserve_target // 2)

        pricing_params = PricingParams(
            reserve_remuneration_rate=r_floor,
            cb_borrowing_rate=r_ceiling,
            reserve_target=reserve_target,
            symmetric_capacity=symmetric_capacity,
            ticket_size=ticket_size,
            reserve_floor=reserve_floor,
            alpha=bank_profile.alpha,
            gamma=bank_profile.gamma,
        )

        banks[bank_id] = BankTreynorState(
            bank_id=bank_id,
            pricing_params=pricing_params,
        )

    # Build subsystem
    subsystem = BankingSubsystem(
        banks=banks,
        bank_profile=bank_profile,
        kappa=kappa,
        trader_banks=trader_banks or {},
        infra_banks=infra_banks or {},
        interbank_loans=[],
        loan_maturity=loan_mat,
        interest_period=interest_period,
        risk_assessor=risk_assessor,
    )

    # Compute initial quotes
    current_day = system.state.day
    subsystem.refresh_all_quotes(system, current_day)

    logger.info(
        "Banking subsystem initialized: %d banks, kappa=%.2f, "
        "corridor=[%.4f, %.4f], loan_maturity=%d",
        len(banks),
        float(kappa),
        float(r_floor),
        float(r_ceiling),
        loan_mat,
    )

    return subsystem
