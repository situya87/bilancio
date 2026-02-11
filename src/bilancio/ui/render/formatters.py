"""Event formatters for bilancio - Improved version with no redundancy."""

from typing import Dict, Any, Tuple, List


class EventFormatterRegistry:
    """Registry for event formatters."""

    def __init__(self) -> None:
        self._formatters: Dict[str, Any] = {}

    def register(self, event_kind: str) -> Any:
        """Decorator to register a formatter for an event kind."""
        def decorator(func: Any) -> Any:
            self._formatters[event_kind] = func
            return func
        return decorator
    
    def format(self, event: Dict[str, Any]) -> Tuple[str, List[str], str]:
        """Format an event using the registered formatter."""
        kind = event.get("kind", "Unknown")
        formatter = self._formatters.get(kind)

        if formatter:
            result: Tuple[str, List[str], str] = formatter(event)
            return result
        else:
            # Generic fallback for unknown events
            return self._format_generic(event)
    
    def _format_generic(self, event: Dict[str, Any]) -> Tuple[str, List[str], str]:
        """Generic formatter for unknown event kinds."""
        kind = event.get("kind", "Unknown")
        
        # Build a generic representation
        title = f"{kind} Event"
        lines = []
        
        # Add key fields, skipping meta fields
        skip_fields = {"kind", "day", "phase", "type"}
        for key, value in event.items():
            if key not in skip_fields:
                lines.append(f"{key}: {value}")
        
        return title, lines[:3], "❓"  # Limit to 3 lines


# Create global registry instance
registry = EventFormatterRegistry()


# Register specific formatters for common event kinds
@registry.register("CashTransferred")
def format_cash_transferred(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format cash transfer events."""
    amount = event.get("amount", 0)
    frm = event.get("frm", "Unknown")
    to = event.get("to", "Unknown")
    
    title = f"Cash Transfer: ${amount:,}"
    lines = [f"{frm} → {to}"]

    return title, lines, "💰"


@registry.register("ReservesTransferred")
def format_reserves_transferred(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format reserves transfer events."""
    amount = event.get("amount", 0)
    frm = event.get("frm", "Unknown")
    to = event.get("to", "Unknown")
    
    title = f"[BANK] Reserves Transfer: ${amount:,}"
    lines = [f"{frm} → {to}"]
    
    return title, lines, "[BANK]"


@registry.register("StockTransferred")
def format_stock_transferred(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format stock transfer events."""
    sku = event.get("sku", "Unknown")
    quantity = event.get("qty", event.get("quantity", 0))
    frm = event.get("frm", "Unknown") 
    to = event.get("to", "Unknown")
    unit_price = event.get("unit_price", None)
    
    title = f"[PKG] Stock Transfer: {quantity} {sku}"
    lines = [f"{frm} → {to}"]
    if unit_price:
        lines.append(f"@ ${unit_price:,}/unit")

    return title, lines, "📦"


@registry.register("DeliveryObligationCreated")
def format_delivery_obligation_created(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format delivery obligation creation events."""
    sku = event.get("sku", "Unknown")
    quantity = event.get("quantity", event.get("qty", 0))
    frm = event.get("frm", "Unknown")
    to = event.get("to", "Unknown")
    due_day = event.get("due_day", None)
    
    title = f"[DOC] Delivery Obligation: {quantity} {sku}"
    lines = [f"{frm} → {to}"]
    if due_day:
        lines.append(f"Due: Day {due_day}")
    
    return title, lines, "[DOC]"


@registry.register("DeliveryObligationSettled")
def format_delivery_obligation_settled(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format delivery obligation settlement events."""
    sku = event.get("sku", "Unknown")
    quantity = event.get("quantity", event.get("qty", 0))
    debtor = event.get("debtor", "Unknown")
    creditor = event.get("creditor", "Unknown")
    
    title = f"[OK] Delivery Settled: {quantity} {sku}"
    lines = [f"{debtor} → {creditor}"]
    
    return title, lines, "[OK]"


@registry.register("PayableCreated")
def format_payable_created(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format payable creation events."""
    amount = event.get("amount", 0)
    debtor = event.get("debtor", event.get("frm", "Unknown"))
    creditor = event.get("creditor", event.get("to", "Unknown"))
    due_day = event.get("due_day", None)
    
    title = f"[PAY] Payable Created: ${amount:,}"
    lines = [f"{debtor} owes {creditor}"]
    if due_day is not None:
        lines.append(f"Due: Day {due_day}")
    
    return title, lines, "[PAY]"


@registry.register("PayableSettled")
def format_payable_settled(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format payable settlement events."""
    amount = event.get("amount", 0)
    debtor = event.get("debtor", "Unknown")
    creditor = event.get("creditor", "Unknown")
    
    title = f"$ Payable Settled: ${amount:,}"
    lines = [f"{debtor} → {creditor}"]
    
    return title, lines, "$"


@registry.register("CashDeposited")
def format_cash_deposited(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format cash deposit events."""
    customer = event.get("customer", "Unknown")
    bank = event.get("bank", "Unknown")
    amount = event.get("amount", 0)
    
    title = f"[ATM] Cash Deposit: ${amount:,}"
    lines = [f"{customer} → {bank}"]
    
    return title, lines, "[ATM]"


@registry.register("CashWithdrawn")
def format_cash_withdrawn(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format cash withdrawal events."""
    customer = event.get("customer", "Unknown")
    bank = event.get("bank", "Unknown") 
    amount = event.get("amount", 0)
    
    title = f"[PAY] Cash Withdrawal: ${amount:,}"
    lines = [f"{customer} <- {bank}"]
    
    return title, lines, "[PAY]"


@registry.register("ClientPayment")
def format_client_payment(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format inter-bank client payment events."""
    payer = event.get("payer", "Unknown")
    payee = event.get("payee", "Unknown")
    amount = event.get("amount", 0)
    payer_bank = event.get("payer_bank", "Unknown")
    payee_bank = event.get("payee_bank", "Unknown")
    
    title = f"[CARD] Inter-Bank Payment: ${amount:,}"
    lines = [
        f"{payer} → {payee}",
        f"via {payer_bank} → {payee_bank}"
    ]
    
    return title, lines, "[CARD]"


@registry.register("IntraBankPayment")
def format_intra_bank_payment(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format intra-bank payment events."""
    payer = event.get("payer", "Unknown")
    payee = event.get("payee", "Unknown")
    amount = event.get("amount", 0)
    bank = event.get("bank", "Unknown")
    
    title = f"[BANK] Intra-Bank Payment: ${amount:,}"
    lines = [
        f"{payer} → {payee}",
        f"at {bank}"
    ]
    
    return title, lines, "[BANK]"


@registry.register("CashPayment")
def format_cash_payment(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format cash payment events."""
    payer = event.get("payer", "Unknown")
    payee = event.get("payee", "Unknown")
    amount = event.get("amount", 0)
    
    title = f"[CASH] Cash Payment: ${amount:,}"
    lines = [f"{payer} → {payee}"]
    
    return title, lines, "[CASH]"


@registry.register("InstrumentMerged")
def format_instrument_merged(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format instrument merge events (cash consolidation)."""
    keep = event.get("keep", "Unknown")
    removed = event.get("removed", "Unknown")
    
    # Extract short IDs for readability
    keep_short = keep.split('_')[-1][:8] if keep != "Unknown" else keep
    removed_short = removed.split('_')[-1][:8] if removed != "Unknown" else removed
    
    title = f"[MRG] Cash Consolidation"
    lines = [
        f"Merged: {removed_short} → {keep_short}",
        f"(Reduces fragmentation)"
    ]
    
    return title, lines, "[MRG]"


@registry.register("InterbankCleared")
def format_interbank_cleared(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format interbank clearing events."""
    debtor_bank = event.get("debtor_bank", "Unknown")
    creditor_bank = event.get("creditor_bank", "Unknown")
    amount = event.get("amount", 0)
    
    title = f"[CLR] Interbank Clearing: ${amount:,}"
    lines = [f"{debtor_bank} → {creditor_bank}"]
    
    return title, lines, "[CLR]"


@registry.register("CashMinted")
def format_cash_minted(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format cash minting events."""
    to = event.get("to", "Unknown")
    amount = event.get("amount", 0)
    
    title = f"[MINT] Cash Minted: ${amount:,}"
    lines = [f"To: {to}"]
    
    return title, lines, "[MINT]"


@registry.register("ReservesMinted")
def format_reserves_minted(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format reserves minting events."""
    to = event.get("to", "Unknown")
    amount = event.get("amount", 0)
    
    title = f"[RSV] Reserves Minted: ${amount:,}"
    lines = [f"Bank: {to}"]
    
    return title, lines, "[RSV]"


@registry.register("StockSplit")
def format_stock_split(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format stock split events."""
    sku = event.get("sku", "Unknown")
    original_qty = event.get("original_qty", 0)
    split_qty = event.get("split_qty", 0)
    remaining_qty = event.get("remaining_qty", 0)
    
    title = f"[SPLIT] Stock Split: {split_qty} {sku}"
    lines = [
        f"From lot of {original_qty} → {remaining_qty} remain",
        f"(Preparing transfer)"
    ]
    
    return title, lines, "[SPLIT]"


@registry.register("DeliveryObligationCancelled")
def format_delivery_cancelled(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format delivery obligation cancellation."""
    obligation_id = event.get("obligation_id", "Unknown")
    debtor = event.get("debtor", "Unknown")
    
    # Short ID for readability
    short_id = obligation_id.split('_')[-1][:8] if obligation_id != "Unknown" else obligation_id
    
    title = f"[OK] Obligation Cleared"
    lines = [
        f"By: {debtor}",
        f"ID: ...{short_id}"
    ]

    return title, lines, "[OK]"


@registry.register("StockCreated")
def format_stock_created(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format stock creation events."""
    owner = event.get("owner", "Unknown")
    sku = event.get("sku", "Unknown")
    qty = event.get("qty", event.get("quantity", 0))
    unit_price = event.get("unit_price", None)
    
    title = f"[DOC] Stock Created: {qty} {sku}"
    lines = [f"Owner: {owner}"]
    if unit_price:
        if isinstance(qty, (int, float)) and isinstance(unit_price, (int, float)):
            total_value = qty * unit_price
            lines.append(f"Value: ${unit_price:,}/unit (${total_value:,} total)")
        else:
            lines.append(f"Value: ${unit_price}/unit")
    
    return title, lines, "[DOC]"


# Phase markers
@registry.register("PhaseA")
def format_phase_a(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format phase A markers."""
    day = event.get("day", "?")
    return f"[TIME] Day {day} begins", ["Morning activities"], "[TIME]"


@registry.register("PhaseB")
def format_phase_b(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format phase B markers."""
    return "[MORN] Business hours", ["Main economic activity"], "[MORN]"


@registry.register("PhaseC")
def format_phase_c(event: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Format phase C markers."""
    return "[NIGHT] End of day", ["Settlements and clearing"], "[NIGHT]"