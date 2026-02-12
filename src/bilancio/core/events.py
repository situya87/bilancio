"""Event kind enumeration for typed system events."""

from enum import Enum


class EventKind(str, Enum):
    """Enumeration of all system event kinds.

    Using str mixin ensures EventKind values work as dict keys
    and are JSON-serializable.
    """

    # ── Phase markers ────────────────────────────────────────────────
    PHASE_A = "PhaseA"
    PHASE_B = "PhaseB"
    PHASE_C = "PhaseC"
    SUBPHASE_B1 = "SubphaseB1"
    SUBPHASE_B2 = "SubphaseB2"
    SUBPHASE_B_DEALER = "SubphaseB_Dealer"
    SUBPHASE_B_ROLLOVER = "SubphaseB_Rollover"

    # ── Payable / settlement events ──────────────────────────────────
    PAYABLE_CREATED = "PayableCreated"
    PAYABLE_SETTLED = "PayableSettled"
    PAYABLE_ROLLED_OVER = "PayableRolledOver"
    PARTIAL_SETTLEMENT = "PartialSettlement"
    ROLLOVER_PARTIAL = "RolloverPartial"

    # ── Delivery obligation events ───────────────────────────────────
    DELIVERY_OBLIGATION_SETTLED = "DeliveryObligationSettled"

    # ── Default handling ─────────────────────────────────────────────
    OBLIGATION_DEFAULTED = "ObligationDefaulted"
    OBLIGATION_WRITTEN_OFF = "ObligationWrittenOff"
    AGENT_DEFAULTED = "AgentDefaulted"
    SCHEDULED_ACTION_CANCELLED = "ScheduledActionCancelled"

    # ── Ring topology events ─────────────────────────────────────────
    RING_RECONNECTED = "RingReconnected"
    RING_COLLAPSED = "RingCollapsed"

    # ── Interbank / clearing events ──────────────────────────────────
    INTERBANK_CLEARED = "InterbankCleared"
    INTERBANK_OVERNIGHT_CREATED = "InterbankOvernightCreated"

    # ── Banking / payment events ─────────────────────────────────────
    CASH_DEPOSITED = "CashDeposited"
    CASH_WITHDRAWN = "CashWithdrawn"
    INTRA_BANK_PAYMENT = "IntraBankPayment"
    CLIENT_PAYMENT = "ClientPayment"
    CASH_PAYMENT = "CashPayment"

    # ── Instrument primitives ────────────────────────────────────────
    INSTRUMENT_MERGED = "InstrumentMerged"
    CLAIM_TRANSFERRED = "ClaimTransferred"

    # ── Stock events ─────────────────────────────────────────────────
    STOCK_SPLIT = "StockSplit"
    STOCK_MERGED = "StockMerged"
    STOCK_CONSUMED = "StockConsumed"

    # ── Dealer / secondary-market events ─────────────────────────────
    CLAIM_TRANSFERRED_DEALER = "ClaimTransferredDealer"
    DEALER_TRADE = "dealer_trade"
    SELL_REJECTED = "sell_rejected"
    BUY_REJECTED = "buy_rejected"

    def __str__(self) -> str:
        return self.value
