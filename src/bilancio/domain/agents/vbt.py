"""
Value-Based Trader (VBT) agent for outside liquidity.

A VBT is an outside liquidity provider in the dealer ring subsystem that
trades based on a fundamental value model (M, O anchors). VBTs provide
liquidity when dealer inventory is exhausted, allowing the market to
continue functioning even under stress.
"""

from dataclasses import dataclass, field

from bilancio.domain.agent import Agent, AgentKind


@dataclass
class VBT(Agent):
    """
    Value-Based Trader providing outside liquidity to the dealer ring.

    VBTs represent external market participants who trade based on
    fundamental value assessments rather than inventory management.
    They have unlimited capacity to absorb or provide claims at their
    quoted prices, ensuring market continuity.

    Attributes inherited from Agent:
        id: Unique identifier (e.g., "vbt_short", "vbt_mid")
        name: Display name (e.g., "VBT (short)")
        kind: Always "vbt"
        asset_ids: Claims owned (bought from dealers or sellers)
        liability_ids: Currently unused for VBTs
        stock_ids: Currently unused for VBTs
        defaulted: Whether VBT has failed (typically never)

    Note:
        The detailed trading state (inventory list, M/O anchors, quotes)
        is maintained in VBTState within the dealer subsystem. This
        Agent class provides the main system representation for
        ownership tracking during claim transfers.
    """

    kind: str = field(default=AgentKind.VBT, init=False)
