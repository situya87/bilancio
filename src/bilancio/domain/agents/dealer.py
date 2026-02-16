"""
Dealer agent for secondary market trading.

A Dealer is a market maker in the dealer ring subsystem that provides
liquidity by maintaining bid/ask quotes for claims in a specific
maturity bucket. Dealers hold inventory and trade with both customers
(traders) and the VBT (Value-Based Trader).
"""

from dataclasses import dataclass, field

from bilancio.domain.agent import Agent, AgentKind


@dataclass
class Dealer(Agent):
    """
    Market maker agent for a maturity bucket in the dealer ring.

    Dealers provide two-sided markets (bid/ask) for claims within their
    assigned maturity bucket. They maintain inventory and cash positions,
    adjusting quotes based on inventory levels via the kernel formula.

    Attributes inherited from Agent:
        id: Unique identifier (e.g., "dealer_short", "dealer_mid")
        name: Display name (e.g., "Dealer (short)")
        kind: Always "dealer"
        asset_ids: Claims owned (bought from sellers)
        liability_ids: Currently unused for dealers
        stock_ids: Currently unused for dealers
        defaulted: Whether dealer has failed

    Note:
        The detailed trading state (inventory list, quotes, cash) is
        maintained in DealerState within the dealer subsystem. This
        Agent class provides the main system representation for
        ownership tracking during claim transfers.
    """

    kind: str = field(default=AgentKind.DEALER, init=False)
