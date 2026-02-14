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
    SUBPHASE_B_RATING = "SubphaseB_Rating"

    # ── Bootstrap / setup events ─────────────────────────────────────
    BOOTSTRAP_CB = "BootstrapCB"

    # ── Cash events ──────────────────────────────────────────────────
    CASH_MINTED = "CashMinted"
    CASH_RETIRED = "CashRetired"
    CASH_TRANSFERRED = "CashTransferred"

    # ── Reserves events ──────────────────────────────────────────────
    RESERVES_MINTED = "ReservesMinted"
    RESERVES_TRANSFERRED = "ReservesTransferred"
    RESERVES_TO_CASH = "ReservesToCash"
    CASH_TO_RESERVES = "CashToReserves"
    RESERVE_INTEREST_CREDITED = "ReserveInterestCredited"

    # ── Central bank loan events ─────────────────────────────────────
    CB_LOAN_CREATED = "CBLoanCreated"
    CB_LOAN_REPAID = "CBLoanRepaid"

    # ── Payable / settlement events ──────────────────────────────────
    PAYABLE_CREATED = "PayableCreated"
    PAYABLE_SETTLED = "PayableSettled"
    PAYABLE_ROLLED_OVER = "PayableRolledOver"
    PARTIAL_SETTLEMENT = "PartialSettlement"
    ROLLOVER_PARTIAL = "RolloverPartial"
    OBLIGATION_SETTLED = "ObligationSettled"

    # ── Delivery obligation events ───────────────────────────────────
    DELIVERY_OBLIGATION_CREATED = "DeliveryObligationCreated"
    DELIVERY_OBLIGATION_SETTLED = "DeliveryObligationSettled"
    DELIVERY_OBLIGATION_CANCELLED = "DeliveryObligationCancelled"

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
    STOCK_CREATED = "StockCreated"
    STOCK_SPLIT = "StockSplit"
    STOCK_MERGED = "StockMerged"
    STOCK_CONSUMED = "StockConsumed"
    STOCK_TRANSFERRED = "StockTransferred"

    # ── Dealer / secondary-market events ─────────────────────────────
    CLAIM_TRANSFERRED_DEALER = "ClaimTransferredDealer"
    DEALER_TRADE = "dealer_trade"
    SELL_REJECTED = "sell_rejected"
    BUY_REJECTED = "buy_rejected"

    # ── Non-bank lending events ────────────────────────────────────────
    SUBPHASE_B_LENDING = "SubphaseB_Lending"
    NONBANK_LOAN_CREATED = "NonBankLoanCreated"
    NONBANK_LOAN_REPAID = "NonBankLoanRepaid"
    NONBANK_LOAN_DEFAULTED = "NonBankLoanDefaulted"

    # ── Rating agency events ──────────────────────────────────────────
    RATINGS_PUBLISHED = "RatingsPublished"

    # ── Jurisdiction / FX events ──────────────────────────────────────
    FX_CONVERSION = "FXConversion"
    CAPITAL_CONTROL_BLOCKED = "CapitalControlBlocked"
    CAPITAL_CONTROL_TAXED = "CapitalControlTaxed"
    RESERVE_REQUIREMENT_BREACH = "ReserveRequirementBreach"

    def __str__(self) -> str:
        return self.value
