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

    def refresh_quote(self, system: System, current_day: int) -> Quote:
        """Recompute (r_D, r_L) from bank's current balance sheet.

        Uses a 10-day reserve projection path (per paper Section 6.1.3):
        1. Build path[t] = current_reserves
        2. For s in 1..10, add scheduled legs (CB loan repayments, bank loan repayments)
        3. Cash-tightness L* = max(0, reserve_floor - min(path)) / reserve_floor
        4. Risk index rho = L* (simple version, matches simple_risk_index)
        5. Inventory x = path[t+2] - reserve_target (projected t+2, not current)
        """
        from bilancio.domain.instruments.cb_loan import CBLoan

        reserves = _get_bank_reserves(system, self.bank_id)
        reserve_target = self.pricing_params.reserve_target
        reserve_floor = self.pricing_params.reserve_floor

        # --- Build 10-day reserve projection path ---
        path = [0] * 11  # path[0] = today, path[1..10] = future days
        path[0] = reserves

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

            # Bank loan repayments due on proj_day (inflow: borrowers repay to bank)
            for loan_rec in self.outstanding_loans.values():
                if loan_rec.maturity_day == proj_day:
                    delta += loan_rec.repayment_amount

            path[s] = path[s - 1] + delta

        # --- Cash-tightness: L* = max(0, reserve_floor - min(path)) / reserve_floor ---
        min_path = min(path)
        if reserve_floor > 0:
            cash_tightness = max(Decimal("0"), Decimal(reserve_floor - min_path) / Decimal(reserve_floor))
        else:
            cash_tightness = Decimal("0")

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
        for bank_state in self.banks.values():
            bank_state.refresh_quote(system, current_day)

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
        reserves = _get_bank_reserves(system, bank_id)
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
