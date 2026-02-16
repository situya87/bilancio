"""
Event logging system for dealer ring simulation.

Records all events during simulation for observability:
- Trades (interior and passthrough)
- Settlements (full and partial recovery)
- Defaults
- Rebucketing (dealer-to-dealer and VBT-to-VBT)
- Quote updates
- VBT anchor updates
"""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass
class EventLog:
    """
    Structured event log for dealer ring simulation.

    Records all events during simulation for observability:
    - Trades (interior and passthrough)
    - Settlements (full and partial recovery)
    - Defaults
    - Rebucketing (dealer-to-dealer and VBT-to-VBT)
    - Quote updates
    - VBT anchor updates
    """

    events: list[dict[str, Any]] = field(default_factory=list)

    # Indexed structures for efficient lookup
    defaults_by_day: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    trades_by_day: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    settlements_by_day: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    def _serialize_value(self, value: Any) -> Any:
        """
        Convert value to JSON-serializable format.

        Decimals are stored as strings to preserve precision.
        """
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            return [self._serialize_value(v) for v in value]
        return value

    def log(self, kind: str, day: int, **payload: Any) -> None:
        """
        Append event to log with kind, day, and arbitrary payload.

        Args:
            kind: Event type identifier
            day: Simulation day when event occurred
            **payload: Additional event data
        """
        event = {
            "kind": kind,
            "day": day,
            **self._serialize_value(payload),
        }
        self.events.append(event)

        # Update indices for specific event types
        if kind == "default":
            if day not in self.defaults_by_day:
                self.defaults_by_day[day] = []
            self.defaults_by_day[day].append(event)
        elif kind == "trade":
            if day not in self.trades_by_day:
                self.trades_by_day[day] = []
            self.trades_by_day[day].append(event)
        elif kind == "settlement":
            if day not in self.settlements_by_day:
                self.settlements_by_day[day] = []
            self.settlements_by_day[day].append(event)

    def log_day_start(self, day: int) -> None:
        """
        Log start of a simulation day.

        Args:
            day: Day number
        """
        self.log("day_start", day)

    def log_trade(
        self,
        day: int,
        side: str,
        trader_id: str,
        ticket_id: str,
        bucket: str,
        price: Decimal,
        is_passthrough: bool,
    ) -> None:
        """
        Log a trade event (Events 1-2 or 9-10).

        Args:
            day: Simulation day
            side: "BUY" or "SELL" (from customer's perspective)
            trader_id: Agent ID of the trader
            ticket_id: ID of the ticket being traded
            bucket: Maturity bucket
            price: Trade price
            is_passthrough: True if this is a dealer-to-VBT passthrough trade
        """
        self.log(
            "trade",
            day,
            side=side,
            trader_id=trader_id,
            ticket_id=ticket_id,
            bucket=bucket,
            price=price,
            is_passthrough=is_passthrough,
        )

    def log_sell_rejected(
        self,
        day: int,
        trader_id: str,
        ticket_id: str,
        bucket: str,
        offered_price: Decimal,
        expected_value: Decimal,
        threshold: Decimal,
        reason: str,
    ) -> None:
        """
        Log a rejected SELL trade (trader rejected dealer bid).

        Args:
            day: Simulation day
            trader_id: Agent ID of the trader
            ticket_id: ID of the ticket that was not sold
            bucket: Maturity bucket
            offered_price: Dealer's bid price
            expected_value: Expected value of holding ticket
            threshold: Risk premium threshold
            reason: Rejection reason (e.g., "price_below_ev_plus_threshold")
        """
        self.log(
            "sell_rejected",
            day,
            trader_id=trader_id,
            ticket_id=ticket_id,
            bucket=bucket,
            offered_price=offered_price,
            expected_value=expected_value,
            threshold=threshold,
            reason=reason,
        )

    def log_buy_rejected(
        self,
        day: int,
        trader_id: str,
        ticket_id: str,
        bucket: str,
        offered_price: Decimal,
        expected_value: Decimal,
        threshold: Decimal,
        reason: str,
    ) -> None:
        """
        Log a rejected BUY trade (trader rejected dealer ask).

        Args:
            day: Simulation day
            trader_id: Agent ID of the trader
            ticket_id: ID of the ticket that was not bought
            bucket: Maturity bucket
            offered_price: Dealer's ask price
            expected_value: Expected value of the ticket
            threshold: Risk premium threshold
            reason: Rejection reason (e.g., "ev_below_price_plus_threshold")
        """
        self.log(
            "buy_rejected",
            day,
            trader_id=trader_id,
            ticket_id=ticket_id,
            bucket=bucket,
            offered_price=offered_price,
            expected_value=expected_value,
            threshold=threshold,
            reason=reason,
        )

    def log_quote(
        self,
        day: int,
        bucket: str,
        dealer_bid: Decimal,
        dealer_ask: Decimal,
        vbt_bid: Decimal,
        vbt_ask: Decimal,
        inventory: int,
        capacity: Decimal,
        is_pinned_bid: bool,
        is_pinned_ask: bool,
    ) -> None:
        """
        Log dealer quote state after recomputation.

        Args:
            day: Simulation day
            bucket: Maturity bucket
            dealer_bid: Dealer's bid price
            dealer_ask: Dealer's ask price
            vbt_bid: VBT's bid price
            vbt_ask: VBT's ask price
            inventory: Number of tickets in dealer inventory
            capacity: One-sided capacity in face units
            is_pinned_bid: True if dealer bid is pinned at VBT bid
            is_pinned_ask: True if dealer ask is pinned at VBT ask
        """
        self.log(
            "quote",
            day,
            bucket=bucket,
            dealer_bid=dealer_bid,
            dealer_ask=dealer_ask,
            vbt_bid=vbt_bid,
            vbt_ask=vbt_ask,
            inventory=inventory,
            capacity=capacity,
            is_pinned_bid=is_pinned_bid,
            is_pinned_ask=is_pinned_ask,
        )

    def log_settlement(
        self,
        day: int,
        issuer_id: str,
        total_paid: Decimal,
        n_tickets: int,
    ) -> None:
        """
        Log full settlement (recovery_rate = 1).

        Args:
            day: Simulation day
            issuer_id: Agent ID of the issuer who settled
            total_paid: Total amount paid
            n_tickets: Number of tickets settled
        """
        self.log(
            "settlement",
            day,
            issuer_id=issuer_id,
            total_paid=total_paid,
            n_tickets=n_tickets,
            recovery_rate=Decimal(1),
        )

    def log_default(
        self,
        day: int,
        issuer_id: str,
        recovery_rate: Decimal,
        total_due: Decimal,
        total_paid: Decimal,
        n_tickets: int,
        bucket: str,
    ) -> None:
        """
        Log default event with partial recovery (Events 6-8).

        Args:
            day: Simulation day
            issuer_id: Agent ID of the issuer who defaulted
            recovery_rate: Recovery rate R (0 <= R < 1)
            total_due: Total face value due
            total_paid: Total amount actually paid
            n_tickets: Number of tickets involved
            bucket: Maturity bucket of the defaulted tickets
        """
        self.log(
            "default",
            day,
            issuer_id=issuer_id,
            recovery_rate=recovery_rate,
            total_due=total_due,
            total_paid=total_paid,
            n_tickets=n_tickets,
            bucket=bucket,
        )

    def log_rebucket(
        self,
        day: int,
        ticket_id: str,
        old_bucket: str,
        new_bucket: str,
        price: Decimal,
        holder_type: str,
    ) -> None:
        """
        Log rebucketing event (Events 11-12).

        Args:
            day: Simulation day
            ticket_id: ID of the ticket being rebucketed
            old_bucket: Previous bucket
            new_bucket: New bucket
            price: Price at which ticket is transferred
            holder_type: "dealer" or "vbt" - who holds the ticket
        """
        self.log(
            "rebucket",
            day,
            ticket_id=ticket_id,
            old_bucket=old_bucket,
            new_bucket=new_bucket,
            price=price,
            holder_type=holder_type,
        )

    def log_vbt_anchor_update(
        self,
        day: int,
        bucket: str,
        M_old: Decimal,
        M_new: Decimal,
        O_old: Decimal,
        O_new: Decimal,
        loss_rate: Decimal,
    ) -> None:
        """
        Log VBT anchor update after defaults.

        Args:
            day: Simulation day
            bucket: Maturity bucket
            M_old: Previous mid price anchor
            M_new: New mid price anchor
            O_old: Previous spread anchor
            O_new: New spread anchor
            loss_rate: Loss rate that triggered the update
        """
        self.log(
            "vbt_anchor_update",
            day,
            bucket=bucket,
            M_old=M_old,
            M_new=M_new,
            O_old=O_old,
            O_new=O_new,
            loss_rate=loss_rate,
        )

    def get_bucket_loss_rate(self, day: int, bucket_id: str) -> Decimal:
        """
        Compute loss rate for bucket from today's defaults.

        l_t = sum(face * (1 - R)) / sum(face) for maturing tickets.

        Returns Decimal(0) if no defaults in this bucket today.

        Args:
            day: Simulation day
            bucket_id: Bucket identifier

        Returns:
            Loss rate as Decimal
        """
        if day not in self.defaults_by_day:
            return Decimal(0)

        total_loss = Decimal(0)
        total_face = Decimal(0)

        for event in self.defaults_by_day[day]:
            if event.get("bucket") == bucket_id:
                # Parse Decimal values from string
                total_due = Decimal(event["total_due"])
                total_paid = Decimal(event["total_paid"])

                total_face += total_due
                total_loss += total_due - total_paid

        if total_face == 0:
            return Decimal(0)

        return total_loss / total_face

    def get_trades_for_day(self, day: int) -> list[dict[str, Any]]:
        """
        Get all trades that occurred on a given day.

        Args:
            day: Simulation day

        Returns:
            List of trade events
        """
        return self.trades_by_day.get(day, [])

    def get_defaults_for_day(self, day: int) -> list[dict[str, Any]]:
        """
        Get all defaults that occurred on a given day.

        Args:
            day: Simulation day

        Returns:
            List of default events
        """
        return self.defaults_by_day.get(day, [])

    def get_settlements_for_day(self, day: int) -> list[dict[str, Any]]:
        """
        Get all settlements that occurred on a given day.

        Args:
            day: Simulation day

        Returns:
            List of settlement events
        """
        return self.settlements_by_day.get(day, [])

    def get_events_for_day(self, day: int) -> list[dict[str, Any]]:
        """
        Get all events that occurred on a given day.

        Args:
            day: Simulation day

        Returns:
            List of events
        """
        return [event for event in self.events if event.get("day") == day]

    def get_all_events(self) -> list[dict[str, Any]]:
        """
        Get all events in chronological order.

        Returns:
            List of all events
        """
        return self.events.copy()

    def to_jsonl(self, path: str) -> None:
        """
        Export events to JSONL file (one JSON object per line).

        Args:
            path: File path to write to
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")

    def to_dataframe(self) -> Any:
        """
        Convert events to pandas DataFrame for analysis.

        Returns:
            DataFrame with one row per event

        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for DataFrame export. Install with: uv add pandas"
            ) from e

        return pd.DataFrame(self.events)
