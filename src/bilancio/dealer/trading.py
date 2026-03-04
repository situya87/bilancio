"""
Trade execution logic for the dealer ring.

This module implements the TradeExecutor class, which handles customer trades
with dealers in the dealer ring model. It implements:

- Event 1: Customer SELL (interior) - Customer sells ticket to dealer
- Event 2: Customer BUY (interior) - Customer buys ticket from dealer
- Event 9: Customer SELL (passthrough) - Routed to VBT at outside bid B
- Event 10: Customer BUY (passthrough) - Routed to VBT at outside ask A

The executor determines whether trades can be executed by the dealer (interior)
or must be routed to the VBT (passthrough), executes balance sheet updates,
and enforces programmatic assertions C1, C3, and C4.

All arithmetic uses Decimal for precision - never float.

References:
- Specification Section 6: Event specifications
- Specification Section 8.6: Feasibility checks
- Examples Doc Section 1: Programmatic Assertions
"""

import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bilancio.core.ids import AgentId

from .assertions import (
    assert_c1_double_entry,
    assert_c3_feasibility,
    assert_c4_passthrough_invariant,
)
from .kernel import (
    ExecutionResult,
    KernelParams,
    can_interior_buy,
    can_interior_sell,
    recompute_dealer_state,
)
from .models import DealerState, Ticket, VBTState

logger = logging.getLogger(__name__)


@dataclass
class BuyPreview:
    """Preview of a customer buy trade -- no state mutation."""

    feasible: bool
    ticket: Ticket | None
    price: Decimal  # unit price
    scaled_price: Decimal  # price * face (set by caller after preview)
    is_passthrough: bool
    is_interior: bool
    bucket_id: str
    # Snapshot of dealer state BEFORE the hypothetical trade
    dealer_a_before: Decimal = Decimal(0)
    dealer_x_before: Decimal = Decimal(0)


class TradeExecutor:
    """
    Executes trades between customers and dealers.

    Implements Events 1-2 (interior trades) and Events 9-10 (passthrough)
    from the specification Section 6.

    The executor handles:
    1. Feasibility determination (interior vs passthrough)
    2. Balance sheet updates for dealers and VBT
    3. Ticket ownership transfers
    4. Quote recomputation after trades
    5. Programmatic assertion checking

    Attributes:
        params: Kernel parameters including ticket size S
        rng: Random number generator (currently unused, reserved for future stochastic features)

    References:
        - Specification Section 6.1-6.2: Interior trades (Events 1-2)
        - Specification Section 6.9-6.10: Passthrough trades (Events 9-10)
        - Specification Section 8.6: Feasibility checks
    """

    def __init__(
        self,
        params: KernelParams,
        rng: random.Random | None = None,
        layoff_threshold: Decimal = Decimal("0"),
        recompute_fn: Any = None,
    ):
        """
        Initialize trade executor.

        Args:
            params: Kernel parameters including ticket size S
            rng: Optional random number generator (default: new Random instance)
            layoff_threshold: Inventory ratio below which VBT injects cash into dealer (0 = disabled)
            recompute_fn: Kernel recompute callable (default: Python recompute_dealer_state).
                Swapped to the Rust native version via PerformanceConfig.dealer_backend.
        """
        self.params = params
        self.rng = rng or random.Random()
        self.layoff_threshold = layoff_threshold
        self._recompute_fn = recompute_fn or recompute_dealer_state

    def execute_customer_sell(
        self,
        dealer: DealerState,
        vbt: VBTState,
        ticket: Ticket,
        check_assertions: bool = True,
    ) -> ExecutionResult:
        """
        Customer sells ticket to dealer (dealer buys).

        Implements Event 1 (interior) or Event 9 (passthrough at outside bid).

        Flow:
        1. Check if interior BUY is feasible (can_interior_buy)
        2. If feasible: Execute at dealer bid b_c(x)
           - Dealer inventory increases by 1
           - Dealer cash decreases by price
           - Ticket ownership transfers to dealer
        3. If not feasible: Passthrough at outside bid B
           - VBT absorbs the ticket
           - Dealer state unchanged (C4 assertion)
        4. Recompute dealer quotes after trade
        5. Run C1 double-entry assertion

        Args:
            dealer: Dealer state (modified in-place)
            vbt: VBT state (modified in-place if passthrough)
            ticket: Ticket being sold by customer
            check_assertions: Whether to run assertions (default True)

        Returns:
            ExecutionResult with price, passthrough flag, and ticket

        References:
            - Event 1: Interior customer sell at b_c(x) (Section 6.1)
            - Event 9: Passthrough customer sell at B (Section 6.9)
            - Feasibility: Section 8.6
        """
        # Store customer ID before modifying ticket
        customer_id = ticket.owner_id

        # Determine if interior execution is feasible
        is_interior = can_interior_buy(dealer, self.params)
        logger.debug(
            "customer_sell: ticket=%s customer=%s interior=%s", ticket.id, customer_id, is_interior
        )

        # VBT credit facility: when dealer can't buy interior and inventory
        # is below layoff threshold, VBT injects cash to expand dealer capacity.
        # Economically: repo/credit facility from VBT to market maker.
        if not is_interior and self.layoff_threshold > 0:
            inventory_ratio = (
                Decimal(dealer.a) / Decimal(dealer.K_star) if dealer.K_star > 0 else Decimal(0)
            )
            if inventory_ratio < self.layoff_threshold:
                needed_cash = dealer.bid * self.params.S
                if vbt.cash >= needed_cash:
                    vbt.cash -= needed_cash
                    dealer.cash += needed_cash
                    self._recompute_fn(dealer, vbt, self.params)
                    is_interior = can_interior_buy(dealer, self.params)

        if is_interior:
            # Event 1: Interior execution at dealer bid
            execution_price = dealer.bid

            # Pre-check feasibility (C3 assertion)
            if check_assertions:
                assert_c3_feasibility(dealer, side="BUY", params=self.params)

            # Update dealer balance sheet
            dealer.inventory.append(ticket)
            dealer.cash -= execution_price

            # Transfer ownership
            ticket.owner_id = dealer.agent_id

            # Recompute dealer state after trade
            self._recompute_fn(dealer, vbt, self.params)

            # C1: Double-entry assertion
            if check_assertions:
                assert_c1_double_entry(
                    cash_changes={
                        customer_id: execution_price,  # Customer receives cash
                        dealer.agent_id: -execution_price,  # Dealer pays cash
                    },
                    qty_changes={
                        customer_id: -1,  # Customer gives ticket
                        dealer.agent_id: 1,  # Dealer receives ticket
                    },
                )

            return ExecutionResult(
                executed=True,
                price=execution_price,
                is_passthrough=False,
                ticket=ticket,
            )

        else:
            # Event 9: Passthrough to VBT at outside bid
            execution_price = vbt.B

            # Snapshot dealer state for C4 assertion
            if check_assertions:
                dealer_snapshot = deepcopy(dealer)

            # VBT absorbs the ticket
            vbt.inventory.append(ticket)
            vbt.cash -= execution_price

            # Transfer ownership to VBT
            ticket.owner_id = vbt.agent_id

            # Dealer state unchanged - recompute to update quotes based on VBT state
            # (VBT anchors may have changed, affecting dealer quotes)
            self._recompute_fn(dealer, vbt, self.params)

            # C4: Verify dealer balance sheet unchanged
            if check_assertions:
                assert_c4_passthrough_invariant(dealer_snapshot, dealer)

            # C1: Double-entry assertion (dealer not involved in passthrough)
            if check_assertions:
                assert_c1_double_entry(
                    cash_changes={
                        customer_id: execution_price,  # Customer receives cash
                        vbt.agent_id: -execution_price,  # VBT pays cash
                    },
                    qty_changes={
                        customer_id: -1,  # Customer gives ticket
                        vbt.agent_id: 1,  # VBT receives ticket
                    },
                )

            return ExecutionResult(
                executed=True,
                price=execution_price,
                is_passthrough=True,
                ticket=ticket,
            )

    def execute_customer_buy(
        self,
        dealer: DealerState,
        vbt: VBTState,
        buyer_id: AgentId,
        issuer_preference: AgentId | None = None,
        check_assertions: bool = True,
    ) -> ExecutionResult:
        """
        Customer buys ticket from dealer (dealer sells).

        Implements Event 2 (interior) or Event 10 (passthrough at outside ask).

        Flow:
        1. Check if interior SELL is feasible (can_interior_sell - dealer has inventory)
        2. If feasible: Execute at dealer ask a_c(x)
           - Select ticket from dealer inventory (deterministic tie-breaker)
           - Dealer inventory decreases by 1
           - Dealer cash increases by price
           - Ticket ownership transfers to buyer
        3. If not feasible: Passthrough at outside ask A
           - VBT provides ticket
           - Dealer state unchanged (C4 assertion)
        4. Recompute dealer quotes after trade
        5. Run C1 double-entry assertion

        Ticket selection (deterministic tie-breaker):
        - Prefer issuer_preference if specified
        - Then lowest maturity_day
        - Then lowest serial number

        Args:
            dealer: Dealer state (modified in-place)
            vbt: VBT state (modified in-place if passthrough)
            buyer_id: Agent ID of the buyer
            issuer_preference: Optional preferred issuer (for single-issuer constraint)
            check_assertions: Whether to run assertions (default True)

        Returns:
            ExecutionResult with price, passthrough flag, and ticket

        References:
            - Event 2: Interior customer buy at a_c(x) (Section 6.2)
            - Event 10: Passthrough customer buy at A (Section 6.10)
            - Feasibility: Section 8.6
            - Tie-breaker: Section 6.2 (deterministic selection)
        """
        # Determine if interior execution is feasible
        is_interior = can_interior_sell(dealer, self.params)
        logger.debug("customer_buy: buyer=%s interior=%s", buyer_id, is_interior)

        if is_interior:
            # Event 2: Interior execution at dealer ask
            execution_price = dealer.ask

            # Pre-check feasibility (C3 assertion)
            if check_assertions:
                assert_c3_feasibility(dealer, side="SELL", params=self.params)

            # Select ticket to sell using deterministic tie-breaker
            ticket = self._select_ticket_to_sell(dealer.inventory, issuer_preference)

            # Update dealer balance sheet
            dealer.inventory.remove(ticket)
            dealer.cash += execution_price

            # Transfer ownership
            ticket.owner_id = buyer_id

            # Recompute dealer state after trade
            self._recompute_fn(dealer, vbt, self.params)

            # C1: Double-entry assertion
            if check_assertions:
                assert_c1_double_entry(
                    cash_changes={
                        buyer_id: -execution_price,  # Buyer pays cash
                        dealer.agent_id: execution_price,  # Dealer receives cash
                    },
                    qty_changes={
                        buyer_id: 1,  # Buyer receives ticket
                        dealer.agent_id: -1,  # Dealer gives ticket
                    },
                )

            return ExecutionResult(
                executed=True,
                price=execution_price,
                is_passthrough=False,
                ticket=ticket,
            )

        else:
            # Event 10: Passthrough to VBT at outside ask
            execution_price = vbt.A

            # Snapshot dealer state for C4 assertion
            if check_assertions:
                dealer_snapshot = deepcopy(dealer)

            # Select ticket from VBT inventory
            # If VBT has no inventory, the buy cannot proceed
            if not vbt.inventory:
                return ExecutionResult(executed=False, price=Decimal(0), is_passthrough=True)

            ticket = self._select_ticket_to_sell(vbt.inventory, issuer_preference)

            # VBT provides the ticket
            vbt.inventory.remove(ticket)
            vbt.cash += execution_price

            # Transfer ownership to buyer
            ticket.owner_id = buyer_id

            # Dealer state unchanged - recompute to update quotes based on VBT state
            self._recompute_fn(dealer, vbt, self.params)

            # C4: Verify dealer balance sheet unchanged
            if check_assertions:
                assert_c4_passthrough_invariant(dealer_snapshot, dealer)

            # C1: Double-entry assertion (dealer not involved in passthrough)
            if check_assertions:
                assert_c1_double_entry(
                    cash_changes={
                        buyer_id: -execution_price,  # Buyer pays cash
                        vbt.agent_id: execution_price,  # VBT receives cash
                    },
                    qty_changes={
                        buyer_id: 1,  # Buyer receives ticket
                        vbt.agent_id: -1,  # VBT gives ticket
                    },
                )

            return ExecutionResult(
                executed=True,
                price=execution_price,
                is_passthrough=True,
                ticket=ticket,
            )

    def preview_customer_buy(
        self,
        dealer: DealerState,
        vbt: VBTState,
        buyer_id: AgentId,
        issuer_preference: AgentId | None = None,
    ) -> BuyPreview:
        """Compute a customer buy trade preview WITHOUT mutating state.

        Returns a ``BuyPreview`` describing what *would* happen if the buy
        were committed.  The dealer, VBT, and ticket inventories are
        untouched.

        The caller should inspect the preview, run all rejection gates,
        and then call ``commit_customer_buy`` only if every gate passes.
        """
        is_interior = can_interior_sell(dealer, self.params)

        if is_interior:
            execution_price = dealer.ask
            # Check that inventory is non-empty (same as execute path)
            if not dealer.inventory:
                return BuyPreview(
                    feasible=False,
                    ticket=None,
                    price=Decimal(0),
                    scaled_price=Decimal(0),
                    is_passthrough=False,
                    is_interior=True,
                    bucket_id=dealer.bucket_id,
                    dealer_a_before=Decimal(dealer.a),
                    dealer_x_before=dealer.x,
                )
            # Select ticket (read-only peek -- does not remove from list)
            ticket = self._select_ticket_to_sell(dealer.inventory, issuer_preference)
            return BuyPreview(
                feasible=True,
                ticket=ticket,
                price=execution_price,
                scaled_price=Decimal(0),  # set by caller
                is_passthrough=False,
                is_interior=True,
                bucket_id=dealer.bucket_id,
                dealer_a_before=Decimal(dealer.a),
                dealer_x_before=dealer.x,
            )
        else:
            # Passthrough at VBT ask
            execution_price = vbt.A
            if not vbt.inventory:
                return BuyPreview(
                    feasible=False,
                    ticket=None,
                    price=Decimal(0),
                    scaled_price=Decimal(0),
                    is_passthrough=True,
                    is_interior=False,
                    bucket_id=dealer.bucket_id,
                    dealer_a_before=Decimal(dealer.a),
                    dealer_x_before=dealer.x,
                )
            ticket = self._select_ticket_to_sell(vbt.inventory, issuer_preference)
            return BuyPreview(
                feasible=True,
                ticket=ticket,
                price=execution_price,
                scaled_price=Decimal(0),  # set by caller
                is_passthrough=True,
                is_interior=False,
                bucket_id=dealer.bucket_id,
                dealer_a_before=Decimal(dealer.a),
                dealer_x_before=dealer.x,
            )

    def commit_customer_buy(
        self,
        preview: BuyPreview,
        dealer: DealerState,
        vbt: VBTState,
        buyer_id: AgentId,
    ) -> ExecutionResult:
        """Commit a previously-previewed buy trade, mutating state.

        The *preview* MUST have ``feasible=True`` and a non-None ticket.
        The price used is the one recorded in the preview (not re-read
        from dealer/VBT state) so the committed trade matches exactly.

        Between ``preview_customer_buy`` and this call, no other trade
        may execute in the same bucket (same loop iteration guarantee).
        """
        assert preview.feasible and preview.ticket is not None

        ticket = preview.ticket
        execution_price = preview.price

        if preview.is_interior:
            # Interior buy: dealer sells from own inventory
            dealer.inventory.remove(ticket)
            dealer.cash += execution_price
            ticket.owner_id = buyer_id
            self._recompute_fn(dealer, vbt, self.params)
        else:
            # Passthrough: VBT provides the ticket
            vbt.inventory.remove(ticket)
            vbt.cash += execution_price
            ticket.owner_id = buyer_id
            self._recompute_fn(dealer, vbt, self.params)

        return ExecutionResult(
            executed=True,
            price=execution_price,
            is_passthrough=preview.is_passthrough,
            ticket=ticket,
        )

    def _select_ticket_to_sell(
        self,
        inventory: list[Ticket],
        issuer_preference: AgentId | None,
    ) -> Ticket:
        """
        Select ticket to sell from inventory using deterministic tie-breaker.

        Priority:
        1. Match issuer_preference if specified
        2. Lowest maturity_day
        3. Lowest serial number

        This ensures reproducible behavior across runs and respects the
        single-issuer constraint when specified.

        Args:
            inventory: List of tickets available for sale
            issuer_preference: Optional preferred issuer ID

        Returns:
            Selected ticket

        Raises:
            ValueError: If inventory is empty or no matching ticket found

        References:
            - Specification Section 6.2: Deterministic tie-breaker for Event 2
            - Specification Section 10.4: Single-issuer constraint for ring traders
        """
        if not inventory:
            raise ValueError("Cannot select ticket from empty inventory")

        # Filter by issuer preference if specified
        if issuer_preference is not None:
            candidates = [t for t in inventory if t.issuer_id == issuer_preference]
            if not candidates:
                raise ValueError(
                    f"No tickets from preferred issuer {issuer_preference} in inventory. "
                    f"Available issuers: { {t.issuer_id for t in inventory} }"
                )
        else:
            candidates = inventory

        # Sort by (maturity_day, serial) and return first
        # This implements the deterministic tie-breaker:
        # 1. Lowest maturity_day (soonest to mature)
        # 2. Lowest serial number (stable ordering)
        selected = min(candidates, key=lambda t: (t.maturity_day, t.serial))

        return selected
