"""
Sandbox simulation for Banks-as-Dealers.

Demonstrates the banking kernel with:
- 2 BankDealers (Bank A and Bank B)
- Central Bank parameters (corridor)
- Client actions: deposits, loans, inter-bank payments, withdrawals

Run with: uv run python -m bilancio.banking.sandbox
"""

from decimal import Decimal

from bilancio.banking.day_runner import DayRunner, MultiBankDayRunner
from bilancio.banking.pricing_kernel import PricingParams
from bilancio.banking.state import BankDealerState, CentralBankParams
from bilancio.banking.ticket_processor import (
    TicketProcessor,
    process_inter_bank_payment,
    process_intra_bank_payment,
)
from bilancio.banking.types import Quote, Ticket, TicketType


def create_standard_cb_params() -> CentralBankParams:
    """Create standard central bank parameters (2% corridor)."""
    return CentralBankParams(
        reserve_remuneration_rate=Decimal("0.01"),  # 1% floor
        cb_borrowing_rate=Decimal("0.03"),  # 3% ceiling
    )


def create_standard_pricing_params(
    cb_params: CentralBankParams,
    reserve_target: int = 100_000,
    symmetric_capacity: int = 50_000,
    ticket_size: int = 10_000,
    reserve_floor: int = 10_000,
) -> PricingParams:
    """Create standard pricing parameters."""
    return PricingParams(
        reserve_remuneration_rate=cb_params.reserve_remuneration_rate,
        cb_borrowing_rate=cb_params.cb_borrowing_rate,
        reserve_target=reserve_target,
        symmetric_capacity=symmetric_capacity,
        ticket_size=ticket_size,
        reserve_floor=reserve_floor,
    )


def setup_bank(
    bank_id: str,
    initial_reserves: int,
    initial_deposits: int,
    cb_params: CentralBankParams,
    pricing_params: PricingParams,
) -> tuple[BankDealerState, TicketProcessor, DayRunner]:
    """Set up a bank with initial state."""
    state = BankDealerState(
        bank_id=bank_id,
        reserves=initial_reserves,
    )

    # Add initial deposits as payment-origin (day 0)
    if initial_deposits > 0:
        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=initial_deposits,
            rate=Decimal("0.015"),  # Initial rate
        )

    processor = TicketProcessor(
        state=state,
        cb_params=cb_params,
        pricing_params=pricing_params,
    )

    runner = DayRunner(processor=processor)

    return state, processor, runner


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)


def print_bank_state(state: BankDealerState) -> None:
    """Print bank balance sheet."""
    print(state)


def print_quote(quote: Quote, label: str = "") -> None:
    """Print a quote."""
    if label:
        print(f"\n{label}:")
    print(f"  Deposit rate (r_D): {quote.deposit_rate:.4%}")
    print(f"  Loan rate (r_L):    {quote.loan_rate:.4%}")
    print(f"  Inventory (x):      {quote.inventory:,}")
    print(f"  Cash tightness (L*): {quote.cash_tightness:.4f}")
    print(f"  Midline:            {quote.midline:.4%}")


def run_sandbox() -> None:
    """Run the sandbox simulation."""
    print_separator("Banks-as-Dealers Sandbox Simulation")

    # =========================================================================
    # Setup
    # =========================================================================
    print_separator("Setup")

    cb_params = create_standard_cb_params()
    print("Central Bank Corridor:")
    print(f"  Floor (i_R):   {cb_params.reserve_remuneration_rate:.2%}")
    print(f"  Ceiling (i_B): {cb_params.cb_borrowing_rate:.2%}")
    print(f"  Width (Omega): {cb_params.corridor_width:.2%}")

    pricing_params = create_standard_pricing_params(
        cb_params,
        reserve_target=100_000,
        symmetric_capacity=50_000,
        ticket_size=10_000,
        reserve_floor=10_000,
    )

    print("\nPricing Parameters:")
    print(f"  Reserve target (R^tar): {pricing_params.reserve_target:,}")
    print(f"  Symmetric capacity (X*): {pricing_params.symmetric_capacity:,}")
    print(f"  Ticket size (S_fund):    {pricing_params.ticket_size:,}")
    print(f"  Reserve floor (R_min):   {pricing_params.reserve_floor:,}")
    print(f"  Inside width (I^(2)):    {pricing_params.inside_width:.4%}")
    print(f"  Layoff prob (lambda):    {pricing_params.layoff_probability:.4f}")

    # Create two banks
    print("\nCreating banks...")
    state_a, processor_a, runner_a = setup_bank(
        bank_id="BankA",
        initial_reserves=100_000,
        initial_deposits=150_000,
        cb_params=cb_params,
        pricing_params=pricing_params,
    )

    state_b, processor_b, runner_b = setup_bank(
        bank_id="BankB",
        initial_reserves=80_000,
        initial_deposits=120_000,
        cb_params=cb_params,
        pricing_params=pricing_params,
    )

    print("\n--- Bank A Initial State ---")
    print_bank_state(state_a)

    print("\n--- Bank B Initial State ---")
    print_bank_state(state_b)

    # =========================================================================
    # Day 1: Basic operations
    # =========================================================================
    print_separator("Day 1: Basic Operations")

    # Initialize day 1
    day = 1
    quote_a = processor_a.initialize_day(day, withdrawal_forecast=20_000)
    quote_b = processor_b.initialize_day(day, withdrawal_forecast=15_000)

    print_quote(quote_a, "Bank A Opening Quote")
    print_quote(quote_b, "Bank B Opening Quote")

    # --- Test 1: Loan issuance at Bank A ---
    print_separator("Test 1: Loan Issuance at Bank A")
    loan_ticket = Ticket(
        id=state_a.new_ticket_id(),
        ticket_type=TicketType.LOAN,
        amount=30_000,
        client_id="client_alice",
        created_day=day,
    )

    result = processor_a.process_ticket(loan_ticket)
    print(f"Loan ticket result: {result.message}")
    print(f"  Reserve delta: {result.reserve_delta:,}")
    print(f"  Deposit delta: {result.deposit_delta:,}")
    print(f"  Loan delta:    {result.loan_delta:,}")

    print("\n--- Bank A After Loan ---")
    print_bank_state(state_a)
    if result.new_quote is not None:
        print_quote(result.new_quote, "Bank A Updated Quote")

    # --- Test 2: Inter-bank payment (Alice at A pays Bob at B) ---
    print_separator("Test 2: Inter-bank Payment (A -> B)")
    print("Alice (Bank A) pays Bob (Bank B) 25,000")

    settlement = process_inter_bank_payment(
        payer_processor=processor_a,
        payee_processor=processor_b,
        payer_client_id="client_alice",
        payee_client_id="client_bob",
        amount=25_000,
    )

    print(f"Settlement success: {settlement.success}")
    assert settlement.payer_result is not None
    assert settlement.payee_result is not None
    print(f"Payer (Bank A) result: {settlement.payer_result.message}")
    print(f"  Reserve delta: {settlement.payer_result.reserve_delta:,}")
    print(f"Payee (Bank B) result: {settlement.payee_result.message}")
    print(f"  Reserve delta: {settlement.payee_result.reserve_delta:,}")

    print("\n--- Bank A After Inter-bank Payment ---")
    print_bank_state(state_a)

    print("\n--- Bank B After Inter-bank Payment ---")
    print_bank_state(state_b)

    # Verify reserve conservation
    total_reserves = state_a.reserves + state_b.reserves
    print(f"\nTotal system reserves: {total_reserves:,}")
    print("(Should equal initial: 100,000 + 80,000 = 180,000)")

    # --- Test 3: Intra-bank payment at Bank B ---
    print_separator("Test 3: Intra-bank Payment at Bank B")
    print("Bob (Bank B) pays Carol (Bank B) 10,000")

    wd_result, cr_result = process_intra_bank_payment(
        processor=processor_b,
        payer_client_id="client_bob",
        payee_client_id="client_carol",
        amount=10_000,
    )

    print(f"Withdrawal result: {wd_result.message}")
    print(f"  Reserve delta: {wd_result.reserve_delta:,}")
    print(f"Credit result: {cr_result.message}")
    print(f"  Reserve delta: {cr_result.reserve_delta:,}")

    print("\n--- Bank B After Intra-bank Payment ---")
    print_bank_state(state_b)
    print("(Deposits unchanged, reserves unchanged - internal transfer)")

    # =========================================================================
    # Day 2-3: Day runner test
    # =========================================================================
    print_separator("Day 2-3: Using Day Runner")

    multi_runner = MultiBankDayRunner(runners={"BankA": runner_a, "BankB": runner_b})

    # Day 2: Process some tickets
    day2_tickets_a = [
        Ticket(
            id="manual_d2_1",
            ticket_type=TicketType.LOAN,
            amount=15_000,
            client_id="client_dave",
            created_day=2,
        ),
    ]

    day2_tickets_b = [
        Ticket(
            id="manual_d2_2",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=5_000,
            client_id="client_eve",
            created_day=2,
            counterparty_bank_id="external",  # Simulates external payment
        ),
    ]

    results_d2 = multi_runner.run_day(
        day=2,
        tickets_by_bank={"BankA": day2_tickets_a, "BankB": day2_tickets_b},
        withdrawal_forecasts={"BankA": 10_000, "BankB": 5_000},
    )

    print("\n--- Day 2 Results ---")
    for bank_id, day_result in results_d2.items():
        print(f"\n{bank_id}:")
        print(f"  Tickets processed: {day_result.tickets_processed}")
        print(f"  Opening reserves: {day_result.opening_reserves:,}")
        print(f"  Closing reserves: {day_result.closing_reserves:,}")
        print(f"  Opening deposits: {day_result.opening_deposits:,}")
        print(f"  Closing deposits: {day_result.closing_deposits:,}")

    # Day 3: Test deposit interest (should kick in for day 0 deposits)
    # Day 0 deposits get interest at day 2 (issuance + 2)
    # So by day 3, we should see interest credited

    results_d3 = multi_runner.run_day(
        day=3,
        tickets_by_bank={"BankA": [], "BankB": []},
        withdrawal_forecasts={"BankA": 5_000, "BankB": 5_000},
    )

    print("\n--- Day 3 Results ---")
    for bank_id, day_result in results_d3.items():
        print(f"\n{bank_id}:")
        print(f"  Deposit interest credited: {day_result.deposit_interest_credited:,}")
        for event in day_result.events:
            if event.event_type == "deposit_interest":
                print(f"  Event: {event.description}")

    # =========================================================================
    # Final State
    # =========================================================================
    print_separator("Final System State")

    system_state = multi_runner.get_system_state()
    print(f"Total reserves:     {system_state['total_reserves']:,}")
    print(f"Total deposits:     {system_state['total_deposits']:,}")
    print(f"Total loans:        {system_state['total_loans']:,}")
    print(f"Total CB borrowing: {system_state['total_cb_borrowing']:,}")

    for bank_id, bs in system_state["banks"].items():
        print(f"\n{bank_id}:")
        print(f"  Reserves:     {bs['reserves']:>12,}")
        print(f"  Deposits:     {bs['deposits']:>12,}")
        print(f"  Loans:        {bs['loans']:>12,}")
        print(f"  CB Borrowing: {bs['cb_borrowing']:>12,}")
        print(f"  Equity:       {bs['equity']:>12,}")

    # =========================================================================
    # Stress Test: Large withdrawal causing overdraft
    # =========================================================================
    print_separator("Stress Test: Overdraft & CB Top-up")

    # Create a large withdrawal that will cause overdraft
    print("Creating large withdrawal at Bank A (will cause overdraft)...")

    # First, add a large deposit to withdraw from
    state_a.add_deposit_cohort(
        day=3,
        origin="payment",
        amount=200_000,
        rate=Decimal("0.015"),
    )
    print("Added 200,000 deposit at Bank A")
    print(f"Bank A reserves before large withdrawal: {state_a.reserves:,}")

    # Now process a large withdrawal that exceeds reserves
    large_wd_ticket = Ticket(
        id=state_a.new_ticket_id(),
        ticket_type=TicketType.WITHDRAWAL,
        amount=150_000,
        client_id="client_alice",
        created_day=3,
        counterparty_bank_id="BankB",  # Goes to Bank B
    )

    result = processor_a.process_ticket(large_wd_ticket)
    print(f"\nLarge withdrawal result: {result.message}")
    print(f"Bank A reserves after withdrawal: {state_a.reserves:,}")

    if state_a.reserves < 0:
        print(f"\n[OVERDRAFT DETECTED] Bank A reserves negative: {state_a.reserves:,}")
        print("Running end-of-day sequence to trigger CB top-up...")

        # Run day 4 to trigger CB top-up
        result_d4 = runner_a.run_day(
            day=4,
            tickets=[],
            withdrawal_forecast=0,
        )

        print("\nAfter CB top-up:")
        print(f"  Reserves: {state_a.reserves:,}")
        print(f"  CB Borrowing: {state_a.total_cb_borrowing:,}")
        print(f"  CB top-up amount: {result_d4.cb_topup_amount:,}")

        for event in result_d4.events:
            if event.part == "E":
                print(f"  Event: {event.description}")

    print_separator("Sandbox Complete")


if __name__ == "__main__":
    run_sandbox()
