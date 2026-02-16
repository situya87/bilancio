"""
Specification of the current bilancio system.

This file defines all agents and instruments in the current implementation,
serving as the source of truth for the system's structure.

Run validation with:
    from bilancio.specification.current_system import get_current_registry, validate_current_system
    registry = get_current_registry()
    result = validate_current_system()
"""

from .models import (
    AgentRelation,
    AgentSpec,
    BalanceSheetPosition,
    DecisionSpec,
    InstrumentInteraction,
    InstrumentRelation,
    InstrumentSpec,
    LifecycleSpec,
)
from .registry import SpecificationRegistry
from .validators import ValidationResult, validate_all_relationships


def create_trader_spec() -> AgentSpec:
    """Create specification for the Trader agent."""
    return AgentSpec(
        name="Trader",
        description="Ring participant who issues and holds payables, maintains deposits at a bank",
        state_fields={
            "agent_id": "str - Unique identifier (H1, H2, ...)",
            "bank_id": "str - Which bank holds this trader's deposits",
            "deposit_cohorts": "list[DepositCohort] - Interest-bearing deposit tranches",
            "loans": "list[TraderLoan] - Outstanding bank loans",
            "tickets_owned": "list[Ticket] - Payables held as assets",
            "obligations": "list[Ticket] - Payables issued as liabilities",
            "defaulted": "bool - Whether trader has defaulted",
            "asset_issuer_id": "str - Single-issuer constraint for tickets",
        },
        has_bank_account=True,
        bank_assignment_rule="Round-robin: trader i → bank_(((i-1) % n_banks) + 1)",
        instrument_relations={
            "Payable": InstrumentRelation(
                instrument_name="Payable",
                position=BalanceSheetPosition.ASSET,  # Can also be LIABILITY
                balance_sheet_entry="tickets_owned (asset) / obligations (liability)",
                can_create=True,
                can_hold=True,
                can_transfer=True,
                settlement_role="issuer or holder",
                settlement_action="As issuer: pay from deposits. As holder: receive to deposits.",
                affects_decisions=True,
                decision_description="Compare payable yield vs deposit rate for buy/sell decisions",
            ),
            "Deposit": InstrumentRelation(
                instrument_name="Deposit",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="deposit_cohorts",
                can_create=False,  # Bank creates when trader deposits
                can_hold=True,
                can_transfer=False,  # Deposits don't transfer, payments do
                settlement_role="depositor",
                settlement_action="Withdraw to make payments, receive payments as new cohorts",
                affects_decisions=True,
                decision_description="Deposit rate vs payable yield determines trading decisions",
            ),
            "Loan": InstrumentRelation(
                instrument_name="Loan",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="loans",
                can_create=False,  # Bank creates
                can_hold=True,  # Holds as borrower (liability)
                can_transfer=False,
                settlement_role="borrower",
                settlement_action="Repay principal + interest from deposits at maturity",
                affects_decisions=True,
                decision_description="Loan cost vs sell cost determines borrow vs sell decision",
            ),
            "Reserve": InstrumentRelation(
                instrument_name="Reserve",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "CBLoan": InstrumentRelation(
                instrument_name="CBLoan",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
        },
        decisions=[
            DecisionSpec(
                name="cover_shortfall",
                trigger="shortfall > 0 at start of trading phase",
                inputs=["deposit_balance", "payment_due", "tickets_owned", "loan_rate"],
                instruments_involved=["Payable", "Deposit", "Loan"],
                outputs=["SELL tickets", "BORROW from bank", "DEFAULT if still short"],
                logic_description="""
                1. Calculate shortfall = payment_due - deposit_balance
                2. If has tickets: sell to cover (shortest maturity first)
                3. If still short: borrow remainder from bank
                4. If can't borrow enough: DEFAULT
                """,
            ),
            DecisionSpec(
                name="yield_based_sell",
                trigger="deposit_rate > payable_yield and has tickets",
                inputs=["deposit_rate", "payable_expected_value", "dealer_bid"],
                instruments_involved=["Payable", "Deposit"],
                outputs=["SELL ticket if deposit rate exceeds payable yield"],
                logic_description="""
                Compare: keeping payable (earns yield to maturity)
                vs: selling and holding deposits (earns deposit rate)
                Sell if deposit rate > implied payable yield
                """,
            ),
            DecisionSpec(
                name="yield_based_buy",
                trigger="payable_yield > deposit_rate and has excess deposits",
                inputs=["deposit_rate", "payable_expected_value", "dealer_ask", "horizon"],
                instruments_involved=["Payable", "Deposit"],
                outputs=["BUY ticket if payable yield exceeds deposit rate"],
                logic_description="""
                Compare: holding deposits (earns deposit rate)
                vs: buying payable (earns yield to maturity)
                Buy if payable yield > deposit rate and horizon >= H
                """,
            ),
        ],
        lifecycle=LifecycleSpec(
            creation_trigger="Simulation setup (traders created with initial deposits and tickets)",
            interest_accrual="Deposits accrue interest after 2-day grace period, every 2 days",
            maturity_trigger="Obligations due when ticket.maturity_day == current_day",
            full_settlement_action="Pay obligations from deposits, receive payments to deposits",
            partial_settlement_action="Partial recovery distributed to claimants pro-rata by maturity weight",
            default_trigger="Cannot cover obligations + loan repayments from deposits + liquidation",
            default_priority="Loans and payables rank equally, weighted by maturity",
            recovery_mechanism="Liquidate tickets at dealer bid, pool with deposits, distribute pro-rata",
        ),
    )


def create_bank_spec() -> AgentSpec:
    """Create specification for the Bank agent."""
    return AgentSpec(
        name="Bank",
        description="Holds trader deposits, issues loans, participates in interbank settlement",
        state_fields={
            "bank_id": "str - Unique identifier (bank_1, bank_2, ...)",
            "reserves": "Decimal - Reserves held at central bank",
            "current_deposit_rate": "Decimal - Rate offered on deposits",
            "current_loan_rate": "Decimal - Rate charged on loans",
            "loans_outstanding": "dict[str, TraderLoan] - Loans issued to traders",
            "cb_borrowing": "list - Borrowing from central bank",
            "client_ids": "set[str] - Traders who bank here",
        },
        has_bank_account=True,  # At central bank (reserves)
        bank_assignment_rule="Reserves held at CentralBank",
        instrument_relations={
            "Payable": InstrumentRelation(
                instrument_name="Payable",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,  # Banks don't hold payables in current model
                can_transfer=False,
            ),
            "Deposit": InstrumentRelation(
                instrument_name="Deposit",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="implicit (sum of trader deposits)",
                can_create=True,  # Creates deposits when issuing loans
                can_hold=False,  # Bank is issuer, not holder
                can_transfer=False,
                settlement_role="issuer",
                settlement_action="Credit/debit trader accounts, adjust reserves for interbank",
                affects_decisions=True,
                decision_description="Deposit rate affects trader behavior and bank profitability",
            ),
            "Loan": InstrumentRelation(
                instrument_name="Loan",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="loans_outstanding",
                can_create=True,
                can_hold=True,
                can_transfer=False,
                settlement_role="lender",
                settlement_action="Receive repayment from trader at maturity",
                affects_decisions=True,
                decision_description="Loan rate set by pricing kernel based on inventory",
            ),
            "Reserve": InstrumentRelation(
                instrument_name="Reserve",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="reserves",
                can_create=False,  # CB creates
                can_hold=True,
                can_transfer=True,  # Via interbank settlement
                settlement_role="holder",
                settlement_action="Used for interbank settlement",
                affects_decisions=True,
                decision_description="Reserve position affects rate setting via pricing kernel",
            ),
            "CBLoan": InstrumentRelation(
                instrument_name="CBLoan",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="cb_borrowing",
                can_create=False,  # CB creates
                can_hold=True,  # As borrower
                can_transfer=False,
                settlement_role="borrower",
                settlement_action="Repay to CB",
                affects_decisions=True,
                decision_description="CB borrowing as last resort when reserves insufficient",
            ),
        },
        decisions=[
            DecisionSpec(
                name="set_rates",
                trigger="Start of each day",
                inputs=["reserves", "reserve_target", "cb_deposit_rate", "cb_lending_rate"],
                instruments_involved=["Reserve", "Deposit", "Loan"],
                outputs=["current_deposit_rate", "current_loan_rate"],
                logic_description="""
                Use Treynor pricing kernel:
                1. Compute inventory = reserves - reserve_target
                2. Midline moves with inventory (more reserves → lower rates)
                3. Rates bounded by CB corridor
                4. Spread reflects risk and capacity
                """,
            ),
            DecisionSpec(
                name="issue_loan",
                trigger="Trader requests loan to cover shortfall",
                inputs=["trader_shortfall", "current_loan_rate"],
                instruments_involved=["Loan", "Deposit"],
                outputs=["Issue loan, create deposit (endogenous money)"],
                logic_description="""
                1. Trader requests loan ≤ shortfall (purpose-bound)
                2. Bank issues loan at current_loan_rate
                3. Simultaneously create deposit for trader (money creation)
                """,
            ),
        ],
        lifecycle=LifecycleSpec(
            creation_trigger="Simulation setup (banks created with initial reserves)",
            interest_accrual="Earn interest on reserves (CB deposit rate), pay on deposits, earn on loans",
            maturity_trigger="Loans mature when loan.maturity_day == current_day",
            full_settlement_action="Receive loan repayment, adjust reserves",
            partial_settlement_action="Claim on defaulted trader in waterfall",
            default_trigger="Bank default not modeled (CB backstop)",
            recovery_mechanism="CB provides reserves as needed (lending facility)",
        ),
    )


def create_dealer_spec() -> AgentSpec:
    """Create specification for the Dealer agent."""
    return AgentSpec(
        name="Dealer",
        description="Market maker for payables, quotes bid/ask, holds inventory",
        state_fields={
            "bucket_id": "str - Which maturity bucket (short, mid, long)",
            "agent_id": "str - Unique identifier",
            "cash": "Decimal - Cash holdings",
            "inventory": "list[Ticket] - Payables held in inventory",
            "bid": "Decimal - Current bid price",
            "ask": "Decimal - Current ask price",
        },
        has_bank_account=True,
        bank_assignment_rule="All dealers use dealer_bank (bank_4)",
        instrument_relations={
            "Payable": InstrumentRelation(
                instrument_name="Payable",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="inventory",
                can_create=False,
                can_hold=True,
                can_transfer=True,
                settlement_role="holder (market maker)",
                settlement_action="Receive payment when held tickets mature",
                affects_decisions=True,
                decision_description="Inventory level affects bid/ask quotes via pricing kernel",
            ),
            "Deposit": InstrumentRelation(
                instrument_name="Deposit",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,  # Dealer uses cash, not deposits in current model
                can_transfer=False,
            ),
            "Loan": InstrumentRelation(
                instrument_name="Loan",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "Reserve": InstrumentRelation(
                instrument_name="Reserve",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "CBLoan": InstrumentRelation(
                instrument_name="CBLoan",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
        },
        decisions=[
            DecisionSpec(
                name="quote_prices",
                trigger="Start of each trading day",
                inputs=["inventory", "VBT_anchor", "capacity"],
                instruments_involved=["Payable"],
                outputs=["bid", "ask"],
                logic_description="""
                Use kernel pricing:
                1. Compute inventory position x = tickets - target
                2. Midline m(x) moves with inventory
                3. Bid = midline - inside_spread/2
                4. Ask = midline + inside_spread/2
                5. Clip to VBT quotes (B, A)
                """,
            ),
        ],
        lifecycle=LifecycleSpec(
            creation_trigger="Simulation setup (one dealer per bucket)",
            interest_accrual="N/A (uses cash, not deposits)",
            maturity_trigger="Held tickets mature at maturity_day",
            full_settlement_action="Receive face value for matured tickets",
            partial_settlement_action="Receive recovery amount if issuer defaults",
        ),
    )


def create_vbt_spec() -> AgentSpec:
    """Create specification for the VBT (Value-Based Trader) agent."""
    return AgentSpec(
        name="VBT",
        description="Provides liquidity backstop, quotes based on anchor values",
        state_fields={
            "bucket_id": "str - Which maturity bucket",
            "agent_id": "str - Unique identifier",
            "M": "Decimal - Mid anchor",
            "O": "Decimal - Spread anchor",
            "A": "Decimal - Ask quote (M + O/2)",
            "B": "Decimal - Bid quote (M - O/2)",
            "cash": "Decimal - Cash holdings",
            "inventory": "list[Ticket] - Tickets held",
        },
        has_bank_account=False,  # VBT uses cash only
        instrument_relations={
            "Payable": InstrumentRelation(
                instrument_name="Payable",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="inventory",
                can_create=False,
                can_hold=True,
                can_transfer=True,
                settlement_role="holder (backstop)",
                settlement_action="Receive payment when held tickets mature",
                affects_decisions=True,
                decision_description="Anchors adjust based on bucket loss rates",
            ),
            "Deposit": InstrumentRelation(
                instrument_name="Deposit",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "Loan": InstrumentRelation(
                instrument_name="Loan",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "Reserve": InstrumentRelation(
                instrument_name="Reserve",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "CBLoan": InstrumentRelation(
                instrument_name="CBLoan",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
        },
        lifecycle=LifecycleSpec(
            creation_trigger="Simulation setup (one VBT per bucket)",
            interest_accrual="N/A",
            maturity_trigger="Held tickets mature at maturity_day",
            full_settlement_action="Receive face value",
            partial_settlement_action="Receive recovery, adjust anchors based on loss",
        ),
    )


def create_central_bank_spec() -> AgentSpec:
    """Create specification for the CentralBank agent."""
    return AgentSpec(
        name="CentralBank",
        description="Sets corridor rates, provides lending facility to banks",
        state_fields={
            "deposit_rate": "Decimal - Floor of corridor (rate on reserves)",
            "lending_rate": "Decimal - Ceiling of corridor (rate on CB loans)",
        },
        has_bank_account=False,  # CB issues reserves
        instrument_relations={
            "Payable": InstrumentRelation(
                instrument_name="Payable",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "Deposit": InstrumentRelation(
                instrument_name="Deposit",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "Loan": InstrumentRelation(
                instrument_name="Loan",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_create=False,
                can_hold=False,
                can_transfer=False,
            ),
            "Reserve": InstrumentRelation(
                instrument_name="Reserve",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="implicit (sum of bank reserves)",
                can_create=True,
                can_hold=False,
                can_transfer=False,
                settlement_role="issuer",
                settlement_action="Pay interest to banks holding reserves",
            ),
            "CBLoan": InstrumentRelation(
                instrument_name="CBLoan",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="implicit (sum of bank borrowing)",
                can_create=True,
                can_hold=True,  # CB is lender
                can_transfer=False,
                settlement_role="lender",
                settlement_action="Receive repayment from banks",
            ),
        },
        lifecycle=LifecycleSpec(
            creation_trigger="System initialization",
            maturity_trigger="N/A (CB provides standing facilities)",
            full_settlement_action="Process reserve transfers in interbank settlement",
        ),
    )


# =============================================================================
# Instrument Specifications
# =============================================================================


def create_payable_spec() -> InstrumentSpec:
    """Create specification for the Payable (Ticket) instrument."""
    return InstrumentSpec(
        name="Payable",
        description="Promise to pay face value at maturity (the core tradeable claim)",
        attributes={
            "id": "str - Unique identifier",
            "issuer_id": "str - Who issued this payable",
            "owner_id": "str - Current holder",
            "face": "Decimal - Face value (payment at maturity)",
            "maturity_day": "int - When payment is due",
            "remaining_tau": "int - Days until maturity",
            "bucket_id": "str - Current maturity bucket",
        },
        tradeable=True,
        interest_bearing=False,  # Traded at discount instead
        secured=False,
        agent_relations={
            "Trader": AgentRelation(
                agent_name="Trader",
                position=BalanceSheetPosition.ASSET,  # When owned; LIABILITY when issued
                balance_sheet_entry="tickets_owned / obligations",
                can_issue=True,
                can_hold=True,
                settlement_action="As issuer: pay face. As holder: receive face.",
                creation_trigger="Initial ring setup, or future rollover",
                creation_effect="Creates liability for issuer, asset for holder",
            ),
            "Bank": AgentRelation(
                agent_name="Bank",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "Dealer": AgentRelation(
                agent_name="Dealer",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="inventory",
                can_issue=False,
                can_hold=True,
                settlement_action="Receive face value at maturity",
                creation_trigger="N/A (acquires via trading)",
            ),
            "VBT": AgentRelation(
                agent_name="VBT",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="inventory",
                can_issue=False,
                can_hold=True,
                settlement_action="Receive face value at maturity",
                creation_trigger="N/A (acquires via passthrough)",
            ),
            "CentralBank": AgentRelation(
                agent_name="CentralBank",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
        },
        lifecycle=LifecycleSpec(
            creation_trigger="Initial ring setup (each trader issues to successor)",
            creation_effect="Creates ring of claims: H1→H2→H3→...→Hn→H1",
            transferable=True,
            transfer_mechanism="Dealer market (sell at bid, buy at ask)",
            maturity_trigger="remaining_tau == 0 (current_day == maturity_day)",
            full_settlement_action="Issuer pays face value to holder",
            partial_settlement_action="Issuer pays recovery_rate × face to holder",
            can_rollover=False,  # Currently no rollover
            rollover_trigger="NOT IMPLEMENTED",
            default_trigger="Issuer deposits + liquidation < obligations",
            default_priority="Pari passu with loans of same maturity",
            recovery_mechanism="Pro-rata weighted by maturity",
        ),
        instrument_interactions={
            "Deposit": InstrumentInteraction(
                other_instrument="Deposit",
                relationship="substitutes",
                decision_tradeoff="""
                Trader chooses between holding payable vs deposit:
                - Payable: earns implicit yield (face - price) / price over remaining_tau
                - Deposit: earns deposit_rate
                Sell payable if deposit_rate > payable_yield
                Buy payable if payable_yield > deposit_rate
                """,
            ),
            "Loan": InstrumentInteraction(
                other_instrument="Loan",
                relationship="complements",
                decision_tradeoff="""
                To cover shortfall:
                - Sell payable: lose EV - bid (opportunity cost)
                - Borrow: pay principal × rate (interest cost)
                Choose whichever is cheaper
                """,
            ),
        },
    )


def create_deposit_spec() -> InstrumentSpec:
    """Create specification for the Deposit instrument."""
    return InstrumentSpec(
        name="Deposit",
        description="Claim on bank, earns interest, used for payments",
        attributes={
            "amount": "Decimal - Principal",
            "rate": "Decimal - Interest rate (locked at deposit time)",
            "deposit_day": "int - When deposited",
        },
        tradeable=False,
        interest_bearing=True,
        secured=False,  # No deposit insurance in model
        agent_relations={
            "Trader": AgentRelation(
                agent_name="Trader",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="deposit_cohorts",
                can_issue=False,
                can_hold=True,
                settlement_action="Withdraw to make payments, receive payments as new deposits",
                creation_trigger="Payment received, loan proceeds credited",
                creation_effect="New deposit cohort with current bank rate",
            ),
            "Bank": AgentRelation(
                agent_name="Bank",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="implicit (sum of client deposits)",
                can_issue=True,  # Creates when crediting accounts
                can_hold=False,
                settlement_action="Credit/debit accounts, adjust reserves for interbank",
                creation_trigger="Loan issuance (money creation) or payment receipt",
                creation_effect="Increase liability to depositor",
            ),
            "Dealer": AgentRelation(
                agent_name="Dealer",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,  # Uses cash in current model
            ),
            "VBT": AgentRelation(
                agent_name="VBT",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "CentralBank": AgentRelation(
                agent_name="CentralBank",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
        },
        lifecycle=LifecycleSpec(
            creation_trigger="Payment received or loan issued",
            creation_effect="New cohort with rate locked at creation",
            interest_accrual="After 2-day grace period, accrues every 2 days at locked rate",
            transferable=False,
            maturity_trigger="On demand (no fixed maturity)",
            full_settlement_action="Withdraw amount from cohorts (FIFO)",
        ),
        instrument_interactions={
            "Payable": InstrumentInteraction(
                other_instrument="Payable",
                relationship="substitutes",
                decision_tradeoff="See Payable specification",
            ),
            "Loan": InstrumentInteraction(
                other_instrument="Loan",
                relationship="created_together",
                decision_tradeoff="""
                When bank issues loan:
                - Loan appears as asset on bank's books
                - Deposit appears as liability on bank's books
                - Deposit appears as asset on trader's books
                This is endogenous money creation.
                """,
            ),
        },
    )


def create_loan_spec() -> InstrumentSpec:
    """Create specification for the Loan instrument."""
    return InstrumentSpec(
        name="Loan",
        description="Bank credit to trader, fixed rate and maturity",
        attributes={
            "loan_id": "str - Unique identifier",
            "borrower_id": "str - Trader who borrowed",
            "bank_id": "str - Lending bank",
            "principal": "Decimal - Amount borrowed",
            "rate": "Decimal - Interest rate",
            "issuance_day": "int - When issued",
            "maturity_day": "int - When repayment due",
        },
        tradeable=False,
        interest_bearing=True,
        secured=False,
        agent_relations={
            "Trader": AgentRelation(
                agent_name="Trader",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="loans",
                can_issue=False,
                can_hold=True,  # Holds as borrower (liability)
                settlement_action="Repay principal + interest from deposits",
                creation_trigger="Request loan to cover shortfall",
                creation_effect="Receive deposit, incur loan liability",
            ),
            "Bank": AgentRelation(
                agent_name="Bank",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="loans_outstanding",
                can_issue=True,
                can_hold=True,
                settlement_action="Receive repayment, credit to reserves",
                creation_trigger="Trader requests loan, bank approves",
                creation_effect="Create loan asset, create deposit liability",
            ),
            "Dealer": AgentRelation(
                agent_name="Dealer",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "VBT": AgentRelation(
                agent_name="VBT",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "CentralBank": AgentRelation(
                agent_name="CentralBank",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
        },
        lifecycle=LifecycleSpec(
            creation_trigger="Trader needs to borrow to cover shortfall",
            creation_effect="Loan + Deposit created simultaneously (money creation)",
            interest_accrual="Simple interest, computed at maturity",
            transferable=False,
            maturity_trigger="current_day == maturity_day",
            full_settlement_action="Trader repays principal + interest",
            partial_settlement_action="Bank claims in trader default waterfall",
            default_trigger="Trader cannot repay at maturity",
            default_priority="Pari passu with payables of same maturity",
            recovery_mechanism="Share in trader's liquidation proceeds",
        ),
        instrument_interactions={
            "Payable": InstrumentInteraction(
                other_instrument="Payable",
                relationship="complements",
                decision_tradeoff="See Payable specification",
            ),
            "Deposit": InstrumentInteraction(
                other_instrument="Deposit",
                relationship="created_together",
                decision_tradeoff="See Deposit specification",
            ),
        },
    )


def create_reserve_spec() -> InstrumentSpec:
    """Create specification for the Reserve instrument."""
    return InstrumentSpec(
        name="Reserve",
        description="Bank's claim on central bank, used for interbank settlement",
        attributes={
            "amount": "Decimal - Reserve balance",
        },
        tradeable=False,
        interest_bearing=True,  # Earns CB deposit rate
        secured=True,  # Claim on CB is risk-free
        agent_relations={
            "Trader": AgentRelation(
                agent_name="Trader",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "Bank": AgentRelation(
                agent_name="Bank",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="reserves",
                can_issue=False,
                can_hold=True,
                settlement_action="Transfer to other banks in interbank settlement",
                creation_trigger="Initial endowment or CB loan",
                creation_effect="Increase reserve balance",
            ),
            "Dealer": AgentRelation(
                agent_name="Dealer",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "VBT": AgentRelation(
                agent_name="VBT",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "CentralBank": AgentRelation(
                agent_name="CentralBank",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="implicit",
                can_issue=True,
                can_hold=False,
                settlement_action="Process reserve transfers",
                creation_trigger="Initial setup or CB lending",
                creation_effect="Increase CB liability, increase bank asset",
            ),
        },
        lifecycle=LifecycleSpec(
            creation_trigger="Initial bank setup or CB lending facility",
            interest_accrual="Earns CB deposit rate on balance",
            transferable=True,
            transfer_mechanism="Interbank settlement at end of day",
            maturity_trigger="N/A (no fixed maturity)",
            full_settlement_action="Transfer between banks to clear interbank positions",
        ),
        instrument_interactions={
            "CBLoan": InstrumentInteraction(
                other_instrument="CBLoan",
                relationship="complements",
                decision_tradeoff="""
                If bank needs reserves for settlement but has insufficient:
                - Borrow from CB at lending rate
                - Receive reserves
                Cost = CB_lending_rate (ceiling of corridor)
                """,
            ),
        },
    )


def create_cb_loan_spec() -> InstrumentSpec:
    """Create specification for the CBLoan instrument."""
    return InstrumentSpec(
        name="CBLoan",
        description="Central bank lending facility loan to commercial bank",
        attributes={
            "amount": "Decimal - Borrowed amount",
            "rate": "Decimal - CB lending rate",
            "day": "int - When borrowed",
        },
        tradeable=False,
        interest_bearing=True,
        secured=True,  # Implicitly secured by bank's assets
        agent_relations={
            "Trader": AgentRelation(
                agent_name="Trader",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "Bank": AgentRelation(
                agent_name="Bank",
                position=BalanceSheetPosition.LIABILITY,
                balance_sheet_entry="cb_borrowing",
                can_issue=False,
                can_hold=True,  # Holds as borrower
                settlement_action="Repay CB",
                creation_trigger="Insufficient reserves for interbank settlement",
                creation_effect="Receive reserves, incur CB debt",
            ),
            "Dealer": AgentRelation(
                agent_name="Dealer",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "VBT": AgentRelation(
                agent_name="VBT",
                position=BalanceSheetPosition.NOT_APPLICABLE,
                can_issue=False,
                can_hold=False,
            ),
            "CentralBank": AgentRelation(
                agent_name="CentralBank",
                position=BalanceSheetPosition.ASSET,
                balance_sheet_entry="implicit",
                can_issue=True,
                can_hold=True,  # CB is lender
                settlement_action="Receive repayment",
                creation_trigger="Bank requests lending facility access",
                creation_effect="Create reserves for bank, create CB asset",
            ),
        },
        lifecycle=LifecycleSpec(
            creation_trigger="Bank's reserves insufficient for interbank settlement",
            creation_effect="CB creates reserves for bank",
            interest_accrual="At CB lending rate (ceiling of corridor)",
            transferable=False,
            maturity_trigger="End of day (overnight facility)",
            full_settlement_action="Bank repays CB",
        ),
        instrument_interactions={
            "Reserve": InstrumentInteraction(
                other_instrument="Reserve",
                relationship="complements",
                decision_tradeoff="See Reserve specification",
            ),
        },
    )


# =============================================================================
# Registry Builder
# =============================================================================


def get_current_registry() -> SpecificationRegistry:
    """
    Build and return a registry with all current system specifications.

    This is the source of truth for what the system should contain.
    """
    registry = SpecificationRegistry()

    # Register agents first
    registry.register_agent(create_trader_spec())
    registry.register_agent(create_bank_spec())
    registry.register_agent(create_dealer_spec())
    registry.register_agent(create_vbt_spec())
    registry.register_agent(create_central_bank_spec())

    # Register instruments
    registry.register_instrument(create_payable_spec())
    registry.register_instrument(create_deposit_spec())
    registry.register_instrument(create_loan_spec())
    registry.register_instrument(create_reserve_spec())
    registry.register_instrument(create_cb_loan_spec())

    return registry


def validate_current_system() -> ValidationResult:
    """
    Validate the current system specification.

    Returns ValidationResult with any errors or warnings.
    """
    registry = get_current_registry()
    return validate_all_relationships(registry)
