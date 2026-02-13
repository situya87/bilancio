"""
Non-bank lender agent for direct lending.

A NonBankLender provides cash loans to traders before settlement.
It holds only cash and loan assets (no claims on the ring).
It uses a profit-seeking strategy: earn interest while managing risk.
"""
from dataclasses import dataclass, field

from bilancio.domain.agent import Agent, AgentKind


@dataclass
class NonBankLender(Agent):
    """
    Non-bank lender that provides cash loans to traders.

    The lender starts with a pool of cash and lends to traders
    who need liquidity to settle upcoming obligations. Loans are
    priced based on borrower risk (estimated default probability).

    Attributes inherited from Agent:
        id: Unique identifier (e.g., "lender")
        name: Display name (e.g., "Non-Bank Lender")
        kind: Always "non_bank_lender"
        asset_ids: Cash instruments and NonBankLoan claims
        liability_ids: Currently unused for lenders
        stock_ids: Currently unused for lenders
        defaulted: Whether lender has failed (typically never)
    """

    kind: str = field(default=AgentKind.NON_BANK_LENDER, init=False)
