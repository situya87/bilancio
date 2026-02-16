from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.agents.treasury import Treasury
from bilancio.domain.instruments.base import Instrument, InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.delivery import DeliveryObligation
from bilancio.domain.instruments.means_of_payment import BankDeposit, Cash, ReserveDeposit
from bilancio.domain.instruments.non_bank_loan import NonBankLoan

AgentType = type[Agent]
InstrType = type[Instrument]


@dataclass
class PolicyEngine:
    # who may issue / hold each instrument type (MVP: static sets)
    issuers: dict[InstrType, Sequence[AgentType]]
    holders: dict[InstrType, Sequence[AgentType]]
    # means-of-payment ranking per agent kind (least-preferred to keep first)
    mop_rank: dict[str, list[str]]

    @classmethod
    def default(cls) -> PolicyEngine:
        return cls(
            issuers={
                Cash: (CentralBank,),
                BankDeposit: (Bank,),
                ReserveDeposit: (CentralBank,),
                CBLoan: (Bank,),  # banks issue (borrow from CB)
                Payable: (Agent,),  # any agent can issue a payable
                DeliveryObligation: (Agent,),  # any agent can promise to deliver
                NonBankLoan: (Agent,),  # any agent can be a borrower (issuer of liability)
            },
            holders={
                Cash: (Agent,),
                BankDeposit: (
                    Household,
                    Firm,
                    Treasury,
                    Bank,
                ),  # banks may hold but not for interbank settlement
                ReserveDeposit: (Bank, Treasury),
                CBLoan: (CentralBank,),  # CB holds loans as assets
                Payable: (Agent,),
                DeliveryObligation: (Agent,),  # any agent can hold a delivery claim
                NonBankLoan: (NonBankLender,),  # only non-bank lenders hold loans as assets
            },
            mop_rank={
                AgentKind.HOUSEHOLD: [InstrumentKind.BANK_DEPOSIT, InstrumentKind.CASH],
                AgentKind.FIRM: [InstrumentKind.CASH, InstrumentKind.BANK_DEPOSIT],
                AgentKind.BANK: [InstrumentKind.RESERVE_DEPOSIT],
                AgentKind.TREASURY: [InstrumentKind.RESERVE_DEPOSIT],
                AgentKind.CENTRAL_BANK: [InstrumentKind.RESERVE_DEPOSIT],
                AgentKind.NON_BANK_LENDER: [InstrumentKind.CASH],
                AgentKind.RATING_AGENCY: [],  # No settlement activity
            },
        )

    def can_issue(self, agent: Agent, instr: Instrument) -> bool:
        return any(isinstance(agent, t) for t in self.issuers.get(type(instr), ()))

    def can_hold(self, agent: Agent, instr: Instrument) -> bool:
        return any(isinstance(agent, t) for t in self.holders.get(type(instr), ()))

    def settlement_order(self, agent: Agent) -> Sequence[str]:
        return self.mop_rank.get(agent.kind, [])
