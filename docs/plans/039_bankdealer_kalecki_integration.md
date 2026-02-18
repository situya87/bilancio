# Plan 039: BankDealer Integration for Kalecki Ring

## Overview

Integrate the BankDealer model (Treynor pricing, rate-setting, lending) into the existing bank infrastructure for the Kalecki ring scenario plugin. Banks become active participants that compete on deposit rates to attract reserves and set loan rates to deploy capital.

**Current State**: Plan 038 added banks as passive deposit issuers with ample reserves. The standalone BankDealer kernel exists in `src/bilancio/banking/` with Treynor pricing, cohort tracking, and rate computation, but is not wired into the main simulation engine.

**Goal**: Banks in the Kalecki ring actively set rates `(r_D, r_L)` via the Treynor pricing kernel, compete for incoming payment flows via deposit rates, lend to traders, and manage reserves through interbank lending — all anchored to a CB corridor derived from run parameters.

---

## Key Design Decisions (From Discussion)

1. **CB corridor as universal anchor**: `(r_floor, r_ceiling)` derived from κ. Banks price inside this corridor. NBFIs and VBTs also anchor to it, but are NOT bounded by it.
2. **Only banks access the CB**: The corridor ceiling is not a ceiling for NBFIs or their clients.
3. **Multi-bank deposits**: Each trader is a client of 3 banks. Each bank has at least 1 infrastructure agent (dealer/VBT/NBFI).
4. **Payment routing by r_D**: Incoming payments go to the trader's highest-r_D bank. Outgoing payments drawn from the lowest-r_D bank.
5. **Loan shopping**: Traders borrow from the bank offering the lowest r_L. Only traders can borrow — dealers, VBTs, and NBFIs cannot.
6. **Loan maturity**: `maturity_days // 2` (default ~5 days), matching NBFI loan maturity.
7. **Deposit interest**: Accrues per 2-day period at the rate stamped at deposit time.
8. **Interbank lending**: Banks with excess reserves lend to banks with deficits, inside the corridor.
9. **VBT spreads derived from corridor**: `O_bucket = Ω × τ_avg_bucket`.
10. **NBFI rate derived from corridor**: Anchored to corridor but not bounded by ceiling.

---

## Layer 1: BankProfile & Corridor Calibration

### 1.1 BankProfile Dataclass

**File**: `src/bilancio/decision/profiles.py`

Add `BankProfile` alongside existing `TraderProfile`, `VBTProfile`, `LenderProfile`:

```python
@dataclass(frozen=True)
class BankProfile:
    """Treynor pricing parameters for active banks."""
    # CB corridor base parameters
    r_base: Decimal = Decimal("0.01")       # Base corridor midpoint
    r_stress: Decimal = Decimal("0.04")     # Stress sensitivity for mid
    omega_base: Decimal = Decimal("0.01")   # Base corridor width
    omega_stress: Decimal = Decimal("0.02") # Stress sensitivity for width

    # Treynor kernel parameters
    reserve_target_ratio: Decimal = Decimal("0.10")  # R_tar / total_deposits
    symmetric_capacity_ratio: Decimal = Decimal("2.0")  # X* / R_tar
    alpha: Decimal = Decimal("0.005")  # Sensitivity to cash-tightness L*
    gamma: Decimal = Decimal("0.002")  # Sensitivity to risk index ρ

    # Loan parameters
    loan_maturity_fraction: Decimal = Decimal("0.5")  # loan_maturity = maturity_days × this
    interest_period: int = 2  # Days per interest accrual period

    def corridor_mid(self, kappa: Decimal) -> Decimal:
        """r_mid = r_base + r_stress × max(0, 1-κ) / (1+κ)"""
        stress = max(Decimal(0), Decimal(1) - kappa) / (Decimal(1) + kappa)
        return self.r_base + self.r_stress * stress

    def corridor_width(self, kappa: Decimal) -> Decimal:
        """Ω = omega_base + omega_stress × max(0, 1-κ) / (1+κ)"""
        stress = max(Decimal(0), Decimal(1) - kappa) / (Decimal(1) + kappa)
        return self.omega_base + self.omega_stress * stress

    def r_floor(self, kappa: Decimal) -> Decimal:
        return self.corridor_mid(kappa) - self.corridor_width(kappa) / 2

    def r_ceiling(self, kappa: Decimal) -> Decimal:
        return self.corridor_mid(kappa) + self.corridor_width(kappa) / 2

    def loan_maturity(self, maturity_days: int) -> int:
        return max(2, int(maturity_days * self.loan_maturity_fraction))
```

### 1.2 Corridor Calibration Table

| κ   | r_mid | Ω     | r_floor | r_ceiling | Character |
|-----|-------|-------|---------|-----------|-----------|
| 0.3 | 3.2%  | 2.1%  | 2.2%    | 4.3%      | Stressed  |
| 0.5 | 2.3%  | 1.7%  | 1.5%    | 3.1%      | Moderate  |
| 1.0 | 1.0%  | 1.0%  | 0.5%    | 1.5%      | Balanced  |
| 2.0 | 1.0%  | 1.0%  | 0.5%    | 1.5%      | Abundant  |

### 1.3 Update CentralBank Agent

**File**: `src/bilancio/domain/agents/central_bank.py`

The CentralBank agent already has `reserve_remuneration_rate` and `cb_lending_rate`. These should be set from `BankProfile.r_floor(kappa)` and `BankProfile.r_ceiling(kappa)` during compilation.

---

## Layer 2: Multi-Bank Assignment & Active Bank State

### 2.1 Multi-Bank Assignment in Compiler

**File**: `src/bilancio/scenarios/ring/compiler.py`

**Current** (Plan 038, lines 508-514): Round-robin single-bank assignment.

**New**: Each trader assigned to 3 banks (or all banks if n_banks ≤ 3). Infrastructure agents assigned 1 per bank, distributed evenly.

```python
# Trader assignment: each trader at min(3, n_banks) banks
# For n_banks=3: every trader is at all banks
# For n_banks=5: sliding window [i%5, (i+1)%5, (i+2)%5]
banks_per_trader = min(3, n_banks)
trader_bank_assignments: dict[str, list[str]] = {}
for idx, trader_id in enumerate(trader_ids):
    assigned = []
    for b in range(banks_per_trader):
        bank_idx = ((idx + b) % n_banks) + 1
        assigned.append(f"bank_{bank_idx}")
    trader_bank_assignments[trader_id] = assigned

# Infrastructure assignment: round-robin, one bank each
infra_agents = dealer_ids + vbt_ids + (nbfi_ids if nbfi_ids else [])
infra_bank_assignments: dict[str, str] = {}
for idx, agent_id in enumerate(infra_agents):
    infra_bank_assignments[agent_id] = f"bank_{(idx % n_banks) + 1}"
```

### 2.2 Initial Deposit Distribution

**File**: `src/bilancio/scenarios/ring/compiler.py`

**Current**: After `mint_cash`, a single `deposit_cash` action deposits all cash at one bank.

**New**: After `mint_cash`, split deposit equally across the trader's assigned banks.

```python
# For traders: split initial deposits across assigned banks
for action in initial_actions:
    if "mint_cash" in action:
        agent_id = action["mint_cash"]["to"]
        amount = action["mint_cash"]["amount"]
        if agent_id in trader_bank_assignments:
            banks = trader_bank_assignments[agent_id]
            per_bank = amount // len(banks)
            remainder = amount - per_bank * len(banks)
            for i, bank_id in enumerate(banks):
                dep_amount = per_bank + (1 if i < remainder else 0)
                if dep_amount > 0:
                    deposit_actions.append({
                        "deposit_cash": {
                            "customer": agent_id,
                            "bank": bank_id,
                            "amount": dep_amount,
                        }
                    })
        elif agent_id in infra_bank_assignments:
            # Infrastructure: all at one bank
            deposit_actions.append({
                "deposit_cash": {
                    "customer": agent_id,
                    "bank": infra_bank_assignments[agent_id],
                    "amount": amount,
                }
            })
```

### 2.3 Active Bank State (BankTreynorState)

**File**: NEW `src/bilancio/engines/bank_state.py`

This is the bridge between the existing `banking/` kernel and the main simulation engine. Each bank gets a live state object that tracks its Treynor pricing.

```python
@dataclass
class BankTreynorState:
    """Active bank state for Treynor pricing within the main simulation."""
    bank_id: str
    pricing_params: PricingParams  # From banking/pricing_kernel.py

    # Cached quotes (refreshed each day and after significant events)
    current_quote: Quote | None = None

    # Lending book
    outstanding_loans: dict[str, BankLoan] = field(default_factory=dict)
    total_loan_principal: int = 0

    # Reserve projection
    projection: ReserveProjection | None = None

    def refresh_quote(self, system: System, current_day: int) -> Quote:
        """Recompute (r_D, r_L) from bank's current balance sheet."""
        reserves = _get_bank_reserves(system, self.bank_id)
        deposits = _get_bank_deposits(system, self.bank_id)

        # Reserve target from profile
        reserve_target = int(self.pricing_params.reserve_target * deposits)

        # Build projection
        self.projection = build_reserve_projection(...)

        # Compute inventory
        inventory = compute_inventory(
            self.projection.at_horizon(2), reserve_target
        )

        # Compute cash-tightness
        cash_tightness = compute_cash_tightness(self.projection, ...)

        # Get quote from Treynor kernel
        self.current_quote = compute_quotes(
            inventory, cash_tightness, Decimal(0),  # risk_index TBD
            self.pricing_params, current_day
        )
        return self.current_quote


@dataclass
class BankLoan:
    """A loan from bank to trader, tracked in the main simulation."""
    loan_id: str
    bank_id: str
    borrower_id: str
    principal: int
    rate: Decimal  # Stamped at issuance
    issuance_day: int
    maturity_day: int

    @property
    def repayment_amount(self) -> int:
        return int(self.principal * (1 + self.rate))
```

### 2.4 BankingSubsystem

**File**: NEW `src/bilancio/engines/banking_subsystem.py`

Analogous to `DealerSubsystem` — holds all active bank state, wired into `system.state`.

```python
@dataclass
class BankingSubsystem:
    """Active banking state within the main simulation engine."""
    banks: dict[str, BankTreynorState]  # bank_id → state
    bank_profile: BankProfile
    kappa: Decimal

    # Assignment maps
    trader_banks: dict[str, list[str]]    # trader_id → [bank_id, ...]
    infra_banks: dict[str, str]           # infra_agent_id → bank_id

    # Interbank lending book
    interbank_loans: list[InterbankLoan] = field(default_factory=list)

    # Configuration
    loan_maturity: int = 5  # Days
    interest_period: int = 2  # Days per interest accrual

    def best_deposit_bank(self, agent_id: str) -> str:
        """Return bank_id with highest r_D among this agent's banks."""
        ...

    def cheapest_loan_bank(self, agent_id: str) -> str:
        """Return bank_id with lowest r_L among this agent's banks."""
        ...

    def cheapest_pay_bank(self, agent_id: str) -> str:
        """Return bank_id with lowest r_D among this agent's banks (min opportunity cost)."""
        ...
```

Add to `system.state`:

**File**: `src/bilancio/engines/system.py`

```python
# Add to State dataclass
banking_subsystem: BankingSubsystem | None = None
```

---

## Layer 3: Payment Routing

### 3.1 Settlement Engine: Route by r_D

**File**: `src/bilancio/engines/settlement.py`

**Current** (`_pay_with_deposits`, lines 78-127): Finds the debtor's first bank deposit and pays from there.

**New**: When banking_subsystem is active, use routing rules:

- **Debtor (payer)**: Draw from the bank with the lowest r_D among debtor's banks that have sufficient balance.
- **Creditor (payee)**: Credit to the bank with the highest r_D among creditor's banks.

```python
def _pay_with_deposits_routed(
    system: System,
    debtor_id: str,
    creditor_id: str,
    amount: int,
) -> int:
    """Pay using deposits with r_D-based routing."""
    banking = system.state.banking_subsystem

    # Find debtor's bank with lowest r_D that has sufficient balance
    debtor_banks = _get_agent_bank_ids(system, debtor_id)
    debtor_bank = _select_pay_bank(system, banking, debtor_id, debtor_banks, amount)
    if debtor_bank is None:
        return 0  # Insufficient deposits across all banks

    # Find creditor's bank with highest r_D
    creditor_banks = _get_agent_bank_ids(system, creditor_id)
    creditor_bank = _select_receive_bank(system, banking, creditor_id, creditor_banks)

    # Execute payment
    client_payment(system, debtor_id, debtor_bank, creditor_id, creditor_bank, amount)
    return amount


def _select_pay_bank(system, banking, agent_id, bank_ids, amount) -> str | None:
    """Select bank with lowest r_D that has sufficient balance."""
    candidates = []
    for bank_id in bank_ids:
        balance = _get_deposit_at_bank(system, agent_id, bank_id)
        if balance >= amount:
            r_d = banking.banks[bank_id].current_quote.deposit_rate
            candidates.append((r_d, balance, bank_id))
    if not candidates:
        # Try combining across banks (split payment)
        return _select_pay_bank_split(system, banking, agent_id, bank_ids, amount)
    candidates.sort(key=lambda x: x[0])  # Lowest r_D first
    return candidates[0][2]


def _select_receive_bank(system, banking, agent_id, bank_ids) -> str:
    """Select bank with highest r_D."""
    best_rate = Decimal("-1")
    best_bank = bank_ids[0]
    for bank_id in bank_ids:
        r_d = banking.banks[bank_id].current_quote.deposit_rate
        if r_d > best_rate:
            best_rate = r_d
            best_bank = bank_id
    return best_bank
```

### 3.2 Multi-Bank Deposit Tracking

**File**: `src/bilancio/ops/banking.py`

**Current**: `deposit_cash()` coalesces deposits into a single instrument per (customer, bank).

**No change needed** — coalescing per (customer, bank) already supports multi-bank deposits naturally. A trader with 3 banks will have up to 3 `BankDeposit` instruments, one per bank. The `_pay_with_deposits_routed()` function selects which to use.

### 3.3 Dealer Trading Cash Sync

**File**: `src/bilancio/engines/dealer_sync.py`

**Current** (`_sync_trader_cash_to_system`, lines 458-609): Applies trader cash deltas to the agent's deposits or cash.

**Update**: When applying a cash delta (from a dealer trade), route the deposit change to the appropriate bank:

- **Cash increase** (trader sold a payable → received cash): Credit to highest-r_D bank
- **Cash decrease** (trader bought a payable → paid cash): Debit from lowest-r_D bank

```python
# In _sync_trader_cash_to_system():
if banking_subsystem is not None:
    if delta > 0:
        # Trader received cash from sale → deposit at best-rate bank
        bank_id = banking_subsystem.best_deposit_bank(agent_id)
        _increase_deposit(system, agent_id, bank_id, delta)
    elif delta < 0:
        # Trader paid cash for purchase → withdraw from worst-rate bank
        bank_id = banking_subsystem.cheapest_pay_bank(agent_id)
        _decrease_deposit(system, agent_id, bank_id, abs(delta))
```

---

## Layer 4: Bank Lending

### 4.1 Lending Phase for Banks

**File**: NEW `src/bilancio/engines/bank_lending.py`

A new subphase that runs AFTER the existing NBFI lending phase (SubphaseB_Lending) and BEFORE dealer trading (SubphaseB_Dealer).

```python
def run_bank_lending_phase(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """
    Banks offer loans to traders with shortfalls.
    Traders compare bank r_L across their assigned banks and pick the cheapest.
    Only traders can borrow (not dealers, VBTs, or NBFIs).
    """
    events = []

    # 1. Identify traders with upcoming shortfalls
    eligible_borrowers = _find_eligible_borrowers(
        system, banking, current_day, banking.loan_maturity
    )

    # 2. For each eligible borrower, find cheapest bank
    for borrower_id, shortfall in eligible_borrowers:
        bank_id = banking.cheapest_loan_bank(borrower_id)
        bank_state = banking.banks[bank_id]
        quote = bank_state.current_quote

        if quote is None:
            continue

        r_L = quote.loan_rate

        # 3. Check bank has capacity (reserve projection)
        if not _bank_can_lend(system, bank_state, shortfall):
            continue

        # 4. Borrow-vs-sell decision (compare with dealer bid)
        if _prefer_selling(system, borrower_id, shortfall, r_L, banking.loan_maturity, current_day):
            continue  # Trader will sell to dealer instead

        # 5. Execute loan
        loan = _execute_bank_loan(
            system, bank_state, borrower_id, shortfall, r_L,
            current_day, banking.loan_maturity
        )
        events.append({"bank_loan_issued": {...}})

    return events


def _execute_bank_loan(system, bank_state, borrower_id, amount, rate, day, maturity):
    """
    Create loan: bank gains asset (loan), borrower gains deposit.
    No reserve movement — this is money creation.
    """
    # 1. Create BankLoan instrument (new instrument kind)
    loan_id = system.create_bank_loan(
        bank_id=bank_state.bank_id,
        borrower_id=borrower_id,
        amount=amount,
        rate=rate,
        issuance_day=day,
        maturity_day=day + maturity,
    )

    # 2. Credit borrower's deposit at the lending bank
    _increase_deposit(system, borrower_id, bank_state.bank_id, amount)

    # 3. Track in bank state
    bank_state.outstanding_loans[loan_id] = BankLoan(...)
    bank_state.total_loan_principal += amount

    # 4. Refresh bank's quote (lending changes inventory outlook)
    bank_state.refresh_quote(system, day)

    return loan_id
```

### 4.2 BankLoan Instrument

**File**: `src/bilancio/domain/instruments/`

Add `BankLoan` instrument kind to `InstrumentKind` enum and create the instrument dataclass.

```python
# In base.py, add to InstrumentKind:
BANK_LOAN = "bank_loan"

# New file: bank_loan.py
@dataclass
class BankLoan(Instrument):
    """Loan from commercial bank to trader."""
    kind: str = InstrumentKind.BANK_LOAN
    rate: Decimal = Decimal(0)
    issuance_day: int = 0
    maturity_day: int = 0

    @property
    def repayment_amount(self) -> int:
        return int(self.amount * (1 + self.rate))

    def is_due(self, current_day: int) -> bool:
        return current_day >= self.maturity_day
```

### 4.3 Policy Updates

**File**: `src/bilancio/domain/policy.py`

```python
# BankLoan: issued by Bank (asset), held by Agent (liability)
InstrumentKind.BANK_LOAN: {
    "issuers": [Bank],          # Bank is the asset holder (creditor)
    "holders": [Household, Firm],  # Traders are the liability issuers (debtors)
}
```

### 4.4 Loan Repayment Phase

**File**: `src/bilancio/engines/bank_lending.py`

Add to Phase D (after CB interest, before or alongside non-bank loan repayments):

```python
def run_bank_loan_repayments(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """Process bank loan repayments due today."""
    events = []

    for loan_id, loan in list(banking.all_outstanding_loans()):
        if loan.maturity_day != current_day:
            continue

        borrower_id = loan.borrower_id
        bank_id = loan.bank_id
        repayment = loan.repayment_amount

        # Try to repay from borrower's deposit at this bank
        deposit_balance = _get_deposit_at_bank(system, borrower_id, bank_id)

        if deposit_balance >= repayment:
            # Debit deposit, retire loan
            _decrease_deposit(system, borrower_id, bank_id, repayment)
            _retire_bank_loan(system, banking, loan_id)
            events.append({"bank_loan_repaid": {...}})
        else:
            # Borrower can't repay — try cross-bank deposits
            total_deposits = _get_total_deposits(system, borrower_id)
            if total_deposits >= repayment:
                # Pay from multiple banks (lowest r_D first)
                _repay_from_multiple_banks(system, banking, borrower_id, bank_id, repayment)
                _retire_bank_loan(system, banking, loan_id)
                events.append({"bank_loan_repaid": {...}})
            else:
                # Default on loan
                events.append({"bank_loan_default": {...}})
                # Trigger trader default handling

    return events
```

### 4.5 Borrow-vs-Sell Decision

Reuse logic from `bank_integration.py:compute_borrow_vs_sell_decision()`:

```python
def _prefer_selling(system, borrower_id, shortfall, r_L, loan_maturity, current_day):
    """Compare: cost of borrowing vs cost of selling to dealer."""
    borrow_cost = shortfall * r_L  # Simple interest for one period

    # Estimate selling cost from dealer bid
    dealer_sub = system.state.dealer_subsystem
    if dealer_sub is None:
        return False  # No dealer available, must borrow

    sell_cost = _estimate_selling_cost(dealer_sub, borrower_id, shortfall, current_day)
    return sell_cost < borrow_cost
```

---

## Layer 5: Deposit Interest Accrual

### 5.1 Interest Accrual Phase

**File**: `src/bilancio/engines/bank_interest.py`

Runs in Phase D (after settlement, before day increment). Every 2 days, each deposit earns interest at its stamped rate.

```python
def accrue_deposit_interest(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """
    Credit interest on deposits every 2 days.
    Interest is deposit-only (increases deposit balance, no reserve movement).
    Uses the r_D that was stamped when the deposit was created/last credited.
    """
    events = []
    period = banking.interest_period  # Default: 2 days

    if current_day % period != 0:
        return events  # Only accrue on even days (or per period)

    for agent_id, agent in system.state.agents.items():
        for deposit_id in agent.asset_ids:
            contract = system.state.contracts.get(deposit_id)
            if contract is None or contract.kind != InstrumentKind.BANK_DEPOSIT:
                continue

            bank_id = contract.liability_issuer_id
            bank_state = banking.banks.get(bank_id)
            if bank_state is None:
                continue

            # Use current r_D for this bank (or stamped rate from creation)
            r_D = bank_state.current_quote.deposit_rate if bank_state.current_quote else Decimal(0)

            interest = int(contract.amount * r_D)
            if interest > 0:
                contract.amount += interest
                events.append({"deposit_interest": {
                    "agent": agent_id,
                    "bank": bank_id,
                    "interest": interest,
                    "rate": str(r_D),
                    "day": current_day,
                }})

    return events
```

**Note**: Interest increases the deposit (bank liability) without moving reserves. This is the standard banking convention — interest is a book entry. It increases the bank's liabilities, reducing equity. The bank earns its margin from loans (assets at r_L) exceeding deposit costs (liabilities at r_D).

---

## Layer 6: Interbank Lending

### 6.1 Interbank Lending Phase

**File**: NEW `src/bilancio/engines/interbank.py`

Runs after Phase C (intraday clearing) and before Phase D. Banks with excess reserves lend to banks with deficits.

```python
def run_interbank_lending(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """
    After client payment netting (Phase C), redistribute reserves
    between banks via bilateral interbank lending.

    Rate: borrowing bank's Treynor midline (the rate at which it's
    willing to borrow). Lending bank accepts if this exceeds its
    deposit rate (opportunity cost of reserves).
    """
    events = []

    # 1. Compute each bank's reserve surplus/deficit
    positions = {}
    for bank_id, bank_state in banking.banks.items():
        reserves = _get_bank_reserves(system, bank_id)
        deposits = _get_bank_deposits_total(system, bank_id)
        target = int(bank_state.pricing_params.reserve_target * deposits)
        positions[bank_id] = reserves - target

    # 2. Sort: surplus banks (lenders) and deficit banks (borrowers)
    surplus_banks = [(bid, pos) for bid, pos in positions.items() if pos > 0]
    deficit_banks = [(bid, pos) for bid, pos in positions.items() if pos < 0]

    surplus_banks.sort(key=lambda x: -x[1])  # Largest surplus first
    deficit_banks.sort(key=lambda x: x[1])   # Largest deficit first

    # 3. Match: deficit banks borrow from surplus banks
    for def_bank_id, deficit in deficit_banks:
        remaining_need = abs(deficit)
        def_state = banking.banks[def_bank_id]

        for i, (sur_bank_id, surplus) in enumerate(surplus_banks):
            if remaining_need <= 0 or surplus <= 0:
                break

            # Rate: midpoint of borrower's and lender's midlines
            borrower_mid = def_state.current_quote.midline if def_state.current_quote else Decimal(0)
            lender_rate = banking.banks[sur_bank_id].current_quote.deposit_rate if banking.banks[sur_bank_id].current_quote else Decimal(0)

            # Lender accepts if offered rate > their deposit rate (opportunity cost)
            if borrower_mid <= lender_rate:
                continue  # Lender won't lend below their own deposit rate

            interbank_rate = (borrower_mid + lender_rate) / 2

            transfer = min(remaining_need, surplus)

            # Transfer reserves
            system.transfer_reserves(sur_bank_id, def_bank_id, transfer)

            # Record interbank loan (matures in 2 days, like CB borrowing)
            ib_loan = InterbankLoan(
                lender_bank=sur_bank_id,
                borrower_bank=def_bank_id,
                amount=transfer,
                rate=interbank_rate,
                issuance_day=current_day,
                maturity_day=current_day + 2,
            )
            banking.interbank_loans.append(ib_loan)

            surplus_banks[i] = (sur_bank_id, surplus - transfer)
            remaining_need -= transfer

            events.append({"interbank_loan": {
                "lender": sur_bank_id,
                "borrower": def_bank_id,
                "amount": transfer,
                "rate": str(interbank_rate),
                "day": current_day,
            }})

    return events


def run_interbank_repayments(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """Process interbank loan repayments due today."""
    events = []
    remaining = []

    for loan in banking.interbank_loans:
        if loan.maturity_day == current_day:
            repayment = int(loan.amount * (1 + loan.rate))
            system.transfer_reserves(loan.borrower_bank, loan.lender_bank, repayment)
            events.append({"interbank_repaid": {...}})
        else:
            remaining.append(loan)

    banking.interbank_loans = remaining
    return events
```

---

## Layer 7: VBT & NBFI Alignment to Corridor

### 7.1 VBT Spread from Corridor

**File**: `src/bilancio/engines/dealer_wiring.py`

**Current** (lines 340-344): Hardcoded base spreads per bucket.

**New**: Derive from corridor width when banking_subsystem is active.

```python
def _compute_base_spreads_from_corridor(
    banking: BankingSubsystem,
    bucket_configs: list[BucketConfig],
) -> dict[str, Decimal]:
    """O_bucket = Ω × τ_avg_bucket"""
    omega = banking.bank_profile.corridor_width(banking.kappa)
    spreads = {}
    for bc in bucket_configs:
        tau_avg = (bc.tau_min + (bc.tau_max or bc.tau_min)) / 2
        spreads[bc.bucket_id] = omega * Decimal(str(tau_avg))
    return spreads
```

When `banking_subsystem` is active, use corridor-derived spreads instead of hardcoded values. When not active (n_banks=0), fall back to existing hardcoded spreads for backward compatibility.

### 7.2 NBFI Rate from Corridor

**File**: `src/bilancio/engines/lending.py`

**Current**: `rate = base_rate + risk_premium_scale × p_default` with `base_rate=0.05`.

**New**: When banking_subsystem is active, anchor to corridor:

```python
def _compute_nbfi_rate(
    lending_config: LendingConfig,
    p_default: Decimal,
    banking: BankingSubsystem | None,
) -> Decimal:
    if banking is not None:
        # Anchor to corridor floor + risk premium scaled by corridor width
        r_floor = banking.bank_profile.r_floor(banking.kappa)
        omega = banking.bank_profile.corridor_width(banking.kappa)
        p_0 = kappa_informed_prior(banking.kappa)
        # NBFI rate: floor + corridor_width × (p_default / p_0)
        # At day 0 (p=p_0): rate = floor + omega ≈ ceiling
        # As defaults rise: rate exceeds ceiling (NBFI is not bounded)
        return r_floor + omega * (p_default / p_0) if p_0 > 0 else r_floor + omega
    else:
        # Existing formula (backward compatible)
        return lending_config.base_rate + lending_config.risk_premium_scale * p_default
```

### 7.3 NBFI Loan Maturity Alignment

**File**: `src/bilancio/engines/lending.py`

When banking_subsystem is active, NBFI loan maturity should match bank loan maturity:

```python
def _get_loan_maturity(lending_config, banking):
    if banking is not None:
        return banking.loan_maturity  # Same as bank: maturity_days // 2
    return lending_config.maturity_days  # Existing default: 2
```

---

## Layer 8: Phase Timing Integration

### 8.1 Updated Daily Cycle

**File**: `src/bilancio/engines/simulation.py` → `run_day()`

```
Phase A: [no change]

Phase B1: Scheduled actions [no change]

Phase B_Rating: Rating agency [no change]

NEW → SubphaseB_BankQuotes: Refresh bank quotes
    - For each bank: bank_state.refresh_quote(system, current_day)
    - Provides current (r_D, r_L) for routing and lending decisions

Phase B_Lending: NBFI lending [updated: rate from corridor]

NEW → SubphaseB_BankLending: Bank lending to traders
    - run_bank_lending_phase(system, current_day, banking)
    - Traders compare r_L across banks, borrow from cheapest
    - Borrow-vs-sell decision factors in dealer bids

Phase B_Dealer: Dealer trading [updated: cash sync uses routing]

Phase B2: settle_due() [updated: payment routing by r_D]

Phase B_Rollover: [no change]

Phase C: settle_intraday_nets() [no change — handles reserve transfers]

NEW → SubphaseC_Interbank: Interbank lending
    - run_interbank_lending(system, current_day, banking)
    - Surplus banks lend to deficit banks
    - Remaining deficits → CB borrowing (existing mechanism)

NEW → SubphaseC_InterbankRepay: Interbank loan repayments
    - run_interbank_repayments(system, current_day, banking)

Phase D: [updated]
    - CB interest on reserves [no change]
    - CB loan repayments [no change]
    - Non-bank loan repayments [no change]
    - NEW: Bank loan repayments
        - run_bank_loan_repayments(system, current_day, banking)
    - NEW: Deposit interest accrual
        - accrue_deposit_interest(system, current_day, banking)

Day increment [no change]
```

### 8.2 Signature Update

```python
def run_day(
    system: System,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,  # NEW
) -> None:
```

---

## Layer 9: Deposit Yield as Trading Decision Factor

### 9.1 Yield-Sell Decision

**File**: `src/bilancio/engines/dealer_integration.py` (eligibility computation)

When banking_subsystem is active, traders consider deposit yield when deciding to sell:

```python
def _is_yield_sell_eligible(trader_state, dealer_bid, ticket, banking, current_day):
    """Sell payable if deposit yield exceeds holding yield."""
    if banking is None:
        return False  # No deposit yield without active banks

    bank_id = banking.best_deposit_bank(trader_state.agent_id)
    r_D = banking.banks[bank_id].current_quote.deposit_rate

    tau = ticket.remaining_tau
    periods = max(1, tau // banking.interest_period)

    # Holding yield: EV at maturity
    ev = _expected_value(ticket, current_day)  # (1 - p) × face

    # Sell + deposit yield: bid_price × (1 + r_D)^periods
    sell_deposit_value = dealer_bid * ticket.face * (1 + r_D) ** periods

    return sell_deposit_value > ev
```

### 9.2 Yield-Buy Hurdle

When banking_subsystem is active, the deposit rate is the hurdle for buying:

```python
def _is_yield_buy_eligible(trader_state, dealer_ask, ticket, banking, current_day):
    """Buy payable only if its yield exceeds deposit yield."""
    if banking is None:
        return True  # Without banks, use existing buy_risk_premium

    bank_id = banking.best_deposit_bank(trader_state.agent_id)
    r_D = banking.banks[bank_id].current_quote.deposit_rate

    tau = ticket.remaining_tau
    periods = max(1, tau // banking.interest_period)

    ev = _expected_value(ticket, current_day)
    cost = dealer_ask * ticket.face

    # Payable yield must exceed deposit yield
    deposit_alternative = cost * (1 + r_D) ** periods
    return ev > deposit_alternative
```

---

## Layer 10: CLI & Compiler Integration

### 10.1 CLI Parameters

**File**: `src/bilancio/cli/sweep.py`

Add to `sweep balanced` command:

```
--enable-banking         Enable active BankDealer (Treynor pricing)
--r-base DECIMAL         CB corridor base mid (default: 0.01)
--r-stress DECIMAL       CB corridor stress sensitivity (default: 0.04)
--omega-base DECIMAL     CB corridor base width (default: 0.01)
--omega-stress DECIMAL   CB corridor stress width sensitivity (default: 0.02)
--loan-maturity-fraction DECIMAL  Bank/NBFI loan maturity as fraction of maturity_days (default: 0.5)
```

When `--enable-banking` is set and `--n-banks` is not explicitly provided, default `n_banks=3`.

### 10.2 Compiler Wiring

**File**: `src/bilancio/scenarios/ring/compiler.py`

When `enable_banking=True`:
1. Create banks with multi-bank assignment (Layer 2)
2. Set CB corridor rates from BankProfile + κ
3. Store `BankProfile` parameters in `_balanced_config`
4. Set `enable_banking: true` in run config

### 10.3 Simulation Bootstrap

**File**: `src/bilancio/engines/simulation.py` or new `src/bilancio/engines/bank_bootstrap.py`

When `enable_banking` and `n_banks > 0`:
1. Read bank_profile from scenario config
2. Initialize `BankTreynorState` per bank using current reserves/deposits
3. Compute initial `PricingParams` from BankProfile + κ
4. Create `BankingSubsystem` and attach to `system.state`
5. Compute initial quotes for all banks

---

## Backward Compatibility

All changes are gated behind `enable_banking` / `n_banks > 0`:

| Condition | Behavior |
|-----------|----------|
| `n_banks=0` (default) | Identical to pre-039 behavior. No banking subsystem. |
| `n_banks>0, enable_banking=False` | Plan 038 passive banks (ample reserves, no pricing). |
| `n_banks>0, enable_banking=True` | Full BankDealer: Treynor pricing, multi-bank, lending, interbank. |

---

## New Files Summary

| File | Purpose |
|------|---------|
| `src/bilancio/engines/bank_state.py` | BankTreynorState, BankLoan dataclasses |
| `src/bilancio/engines/banking_subsystem.py` | BankingSubsystem (analogous to DealerSubsystem) |
| `src/bilancio/engines/bank_lending.py` | Bank lending phase + loan repayments |
| `src/bilancio/engines/bank_interest.py` | Deposit interest accrual |
| `src/bilancio/engines/interbank.py` | Interbank lending + repayments |
| `src/bilancio/domain/instruments/bank_loan.py` | BankLoan instrument |

## Modified Files Summary

| File | Change |
|------|--------|
| `decision/profiles.py` | Add `BankProfile` |
| `domain/instruments/base.py` | Add `BANK_LOAN` to `InstrumentKind` |
| `domain/policy.py` | Add BankLoan to issuers/holders |
| `scenarios/ring/compiler.py` | Multi-bank assignment, corridor calibration, deposit splitting |
| `engines/simulation.py` | New subphases (BankQuotes, BankLending, Interbank, BankLoanRepay, DepositInterest) |
| `engines/settlement.py` | Payment routing by r_D |
| `engines/dealer_sync.py` | Cash sync uses routing |
| `engines/dealer_wiring.py` | VBT spread from corridor |
| `engines/lending.py` | NBFI rate from corridor, maturity alignment |
| `engines/dealer_integration.py` | Yield-sell and yield-buy with deposit rate |
| `engines/system.py` | Add `banking_subsystem` to State |
| `cli/sweep.py` | Add banking CLI flags |

---

## Testing Strategy

### Unit Tests
- `tests/unit/test_bank_profile.py` — Corridor calibration, rate formulas
- `tests/unit/test_payment_routing.py` — r_D-based routing logic
- `tests/unit/test_bank_lending.py` — Loan origination, repayment, borrow-vs-sell
- `tests/unit/test_deposit_interest.py` — Accrual mechanics
- `tests/unit/test_interbank.py` — Reserve redistribution, rate negotiation

### Integration Tests
- `tests/integration/test_bankdealer_ring.py` — Full ring with active banks, verify:
  - Banks quote different rates based on balance sheet
  - Payments route to highest-r_D bank
  - Traders borrow from cheapest bank
  - Interbank lending redistributes reserves
  - Deposit interest affects trading decisions
  - VBT spreads and NBFI rates anchored to corridor

### Regression Tests
- `tests/regression/test_backward_compat.py` — Verify `n_banks=0` and `enable_banking=False` produce identical results to pre-039 code.

---

## Implementation Order

1. **Layer 1**: BankProfile + corridor calibration (standalone, no integration)
2. **Layer 2**: Multi-bank assignment in compiler + BankingSubsystem scaffold
3. **Layer 3**: Payment routing in settlement engine
4. **Layer 8**: Phase timing (wire new subphases into run_day, initially as no-ops)
5. **Layer 4**: Bank lending (the core new capability)
6. **Layer 5**: Deposit interest accrual
7. **Layer 6**: Interbank lending
8. **Layer 7**: VBT/NBFI alignment to corridor
9. **Layer 9**: Deposit yield as trading factor
10. **Layer 10**: CLI integration + end-to-end testing
