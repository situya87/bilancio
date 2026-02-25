# Plan 036: Decision Profile Architecture

**Status**: Design document
**Depends on**: Plan 033 (Information & Decision Architecture), Plan 034 (Valuation as Heuristics)
**Goal**: Define the three-layer architecture (balance sheet / agent type / decision profile), enumerate the concrete decision profiles, and specify per-agent heterogeneity and composition rules

---

## Relationship to Plans 033 and 034

This plan builds on two prior design documents:

- **Plan 033** defines the information hierarchy (access, channels, estimation quality), the decision hierarchy (portfolio → screening → selection → pricing), and the progressive complexity model for the Kalecki Ring.

- **Plan 034** establishes that settlement outcomes are the sole ground truth, that all valuation is agent-held belief with provenance, and that the simulation loop has six stages (state → information → decision → action → agreement → settlement).

This plan adds:
1. The **three-layer architecture** that separates balance sheet, agent type, and decision profile
2. The **concrete decision profiles** for each behavioral entity (trader, Treynor dealer, VBT, Treynor bank, CB, NBFI lender)
3. **Per-agent heterogeneity** — different agents of the same type can use different profiles
4. **Composition rules** — how multiple activity profiles combine for a single agent
5. **Instrument binding** — how general profiles are parameterized for specific scenarios

---

## Part 1: The Three-Layer Architecture

### Layer 1: Balance Sheet

Pure accounting structure. Assets, liabilities, equity. No behavior, no constraints, no decisions.

This is the `State` object with its `Agent`, `Instrument`, and contract tracking. A balance sheet records what an entity owns and owes — nothing more.

Balance sheets are views over the bilateral instrument set (Plan 034, Layer 1). They are always exact from the system's perspective. Individual agents may not see other agents' balance sheets perfectly.

**Current implementation**: Clean and correct. No changes needed.

### Layer 2: Agent Type (Capacities and Limitations)

Domain rules that define what an entity CAN do. These are structural properties of the monetary/financial architecture, not behavioral choices.

**What agent type defines**:

1. **Instrument capacities** — what instruments this agent can issue and hold
   - A Bank can issue BankDeposit (money creation). An NBFI cannot.
   - Only a CentralBank can issue Cash and ReserveDeposit.
   - Only a Bank can issue CBLoan (borrow from CB).

2. **Means of payment** — what this agent uses to settle obligations
   - Households: bank deposits, then cash
   - Banks: reserve deposits
   - NBFI: cash only

3. **Structural information access** — what this agent can see by virtue of its position in the system
   - A bank can see its depositors' balances (it holds the deposit liability)
   - A dealer can see its own inventory and the VBT's quotes
   - An agent always sees its own balance sheet perfectly
   - These are capacities, not choices — they follow from the monetary architecture

4. **Action space envelope** — what activities this agent can participate in
   - Banks can lend, take deposits, access CB facilities, and potentially make markets
   - Firms can trade in secondary markets, borrow, issue payables
   - Dealers can quote bid/ask, hold inventory, intermediate between buyers and sellers
   - The CB can set corridor rates, provide standing facilities, act as lender of last resort

**Current implementation**: `PolicyEngine` in `domain/policy.py` captures items 1 and 2. Item 3 (structural information access) is not yet formalized — it's implicit in how subsystems are wired. Item 4 (action space) is implicit in the phase structure.

**Changes needed**:
- Add structural information access rules to `PolicyEngine` (or a companion `TypeCapabilities` dataclass)
- Formalize the action space per type, so that decision profiles can enumerate available actions

### Layer 3: Decision Profile (Behavioral Strategy)

HOW an entity chooses to act within the capacities defined by its type. This is the behavioral layer — configurable per-agent, potentially heterogeneous within a single run.

**What a decision profile defines**:

Given the action space from the agent type, the decision profile specifies:

1. **Information preferences** — what the agent chooses to look at and how it weights sources (within the structural access granted by its type)
2. **Valuation method** — what heuristics the agent uses to assess things beyond cash flow (Plan 034: valuation is always a heuristic, not a fact)
3. **Risk assessment** — how the agent evaluates its position, exposure, and uncertainty
4. **Choice function** — how the agent picks from available actions

All of this is subject to the **universal cash flow constraint**: every agent, regardless of profile, must meet its payment obligations or default. Cash flow assessment is not a behavioral choice — it's a hard constraint from the accounting layer.

**Current implementation**: Partially exists as `TraderProfile`, `VBTProfile`, `BankProfile`, `LenderProfile` in `decision/profiles.py`. These cover some behavioral parameters but:
- Are shared per-type, not per-agent
- Don't implement a common protocol
- Mix information preferences with valuation logic
- Don't compose

---

## Part 2: The Decision Pipeline

Every decision profile, regardless of the activity it covers, runs the same conceptual pipeline. This is the `ActivityProfile` protocol.

### Step 0: Cash Flow Assessment (Universal, Non-Negotiable)

Before any behavioral logic runs, the agent assesses its cash flow position:
- What do I owe? When? How much?
- What am I owed? When? How much?
- What cash (and cash-equivalent means of payment) do I have?
- Am I facing a shortfall within my planning horizon?

This is always based on perfect information about the agent's own state. It determines the **hard constraint** — the agent cannot choose to default; default happens mechanically when cash flow is insufficient (Plan 034: settlement is always mechanical).

Cash flow assessment also provides the factual baseline for decision-making: actual prices paid, actual amounts received, actual payment schedules. These are facts, not valuations (see Part 4).

### Step 1: Observe (Information Gathering)

The agent queries the `InformationService` for whatever it needs, filtered by:
- **Structural access** (from agent type): what CAN be seen
- **Profile preferences** (from decision profile): what the agent CHOOSES to look at

Some information arrives pre-valued — market prices are someone else's valuation, ratings are the rating agency's estimate. The agent receives these as information inputs, not as its own beliefs.

### Step 2: Value (Heuristic Assessment)

When cash flow information alone is insufficient for the decision at hand, the agent applies valuation heuristics:
- Estimating whether a counterparty will pay (default probability)
- Estimating what an asset would fetch if sold (fair value)
- Projecting future cash flows under uncertainty

Different agents can use different valuation methods on the same information. This is a key dimension of profile heterogeneity.

Valuation produces `Estimate` objects with provenance (Plan 034): who computed it, using what information, through which channel, with what confidence.

### Step 3: Assess (Risk Evaluation)

Given valuations and the agent's current position:
- How does this potential action affect my cash flow position?
- What is my exposure? Concentration?
- How urgent is my liquidity need?
- How uncertain are my valuations?

Risk assessment operates on both cash flow facts AND valuations. It answers "how does this trade affect my position?" — distinct from valuation's "what is this thing worth?"

### Step 4: Choose (Action Selection)

Given the risk-adjusted view and the available action set:
- Enumerate feasible actions (constrained by type capacities and cash flow)
- Evaluate each action using the risk-adjusted view
- Select an action (or choose to hold/do nothing)

The choice function can be: maximize expected utility, satisfice, follow a heuristic rule, or any other decision procedure. This is where profile heterogeneity has its most direct effect.

### Pipeline Interface

```python
class ActivityProfile(Protocol):
    """One coherent decision domain for an agent."""

    @property
    def activity_type(self) -> str:
        """What kind of activity: 'trading', 'market_making', 'lending', etc."""
        ...

    @property
    def instrument_class(self) -> str | None:
        """What instrument class this profile operates on, if bound.
        None means the profile is general (not yet scenario-bound)."""
        ...

    def observe(self, info: InformationService, position: CashFlowPosition) -> ObservedState:
        """Gather filtered information relevant to this activity."""
        ...

    def value(self, observed: ObservedState) -> Valuations:
        """Apply valuation heuristics where cash flow facts aren't enough."""
        ...

    def assess(self, valuations: Valuations, position: CashFlowPosition) -> RiskView:
        """Evaluate risk given valuations and current position."""
        ...

    def choose(self, risk_view: RiskView, action_set: ActionSet) -> Action | None:
        """Pick from available actions, or None to hold/pass."""
        ...
```

---

## Part 3: The Concrete Decision Profiles

Each behavioral entity in the system gets a decision profile that implements `ActivityProfile`. These are general — they define behavior in terms of abstract instrument classes and action types, not Kalecki-specific concepts.

### Profile 1: TradingProfile

**Activity**: Buy/sell instruments in a secondary market

**Current basis**: `TraderProfile` + `RiskAssessor` + `LiquidityDrivenSeller` / `SurplusBuyer`

**Pipeline**:
- **Observe**: Own cash flow position, own holdings, dealer quotes, system/issuer default rates (filtered by `default_observability`)
- **Value**: Estimate holding value of owned instruments using configured valuation method (currently: EV = (1 - P_default) x face). Could be swapped for other methods.
- **Assess**: Compute urgency (shortfall / wealth), liquidity position, concentration of holdings. Apply risk aversion to buying thresholds.
- **Choose**: From action set {sell instrument X at dealer bid, buy instrument Y at dealer ask, hold}. Sellers are liquidity-driven (shortfall triggers selling). Buyers are surplus-driven (excess cash triggers buying).

**Configurable parameters** (current TraderProfile fields):
- `risk_aversion` (0-1): affects buy premium thresholds
- `planning_horizon` (1-20 days): look-ahead for cash flow assessment
- `aggressiveness` (0-1): buy eagerness threshold
- `buy_reserve_fraction` (0-1): reserves before buying
- `default_observability` (0-1): trust in observed default data
- `trading_motive` (liquidity_only / liquidity_then_earning / unrestricted)

**Instrument binding**: In the Kalecki Ring, the instrument class is ring payables. The profile doesn't hardcode "tickets" — it operates on whatever tradeable instruments the scenario defines.

### Profile 2: TreynorDealerProfile

**Activity**: Market-making (two-way quoting, inventory management)

**Current basis**: The dealer pricing kernel in `dealer/kernel.py` + `DealerState`

**Pipeline**:
- **Observe**: Own inventory (instruments held), own capital, VBT/outside reference quotes (via market-derived channel), own equity (mark-to-mid)
- **Value**: Mark inventory to mid-price. Compute capacity (K* = max fundable units). Compute layoff probability (lambda = risk of hitting capacity). This is the Treynor kernel's valuation logic.
- **Assess**: Inventory-sensitive midline p(x) — deviation from target inventory drives risk assessment. Wide inventory → lower bid, higher ask (incentive to shed). Low inventory → higher bid, lower ask (incentive to accumulate).
- **Choose**: From action set {set bid B, set ask A} for each instrument bucket. The Treynor kernel determines these mechanically from the assessment. Passthrough to VBT when quotes hit outside bounds.

**Configurable parameters**:
- `dealer_share_per_bucket`: capital allocation as fraction of system
- The Treynor kernel parameters are implicit in the pricing formula but could be made explicit
- Spread policy (how much to widen/narrow inside spread)

**Instrument binding**: In the Kalecki Ring, the dealer makes markets in ring payables bucketed by maturity (short/mid/long). In another scenario, it could be bonds bucketed by credit rating, or any other instrument class with a natural bucketing.

**Generality note**: The Treynor kernel is one possible market-making decision model. Another scenario might use a simpler spread-based model, or a more sophisticated model with adverse selection. The profile protocol allows this — `TreynorDealerProfile` is one implementation of a market-making `ActivityProfile`.

### Profile 3: VBTProfile (Outside Liquidity Provider)

**Activity**: Reference pricing and outside liquidity provision

**Current basis**: `VBTProfile` + `VBTState` — provides mid M and spread O, absorbs passthrough trades

**Pipeline**:
- **Observe**: System-wide default rate (via risk assessor or published statistics), own capital and inventory
- **Value**: Compute credit-adjusted mid price: M = f(outside_mid_ratio, P_default). This is the VBT's own valuation heuristic — its belief about what instruments are worth.
- **Assess**: Adjust spreads based on stress. `mid_sensitivity` controls how much M reacts to defaults. `spread_sensitivity` controls whether O widens.
- **Choose**: From action set {set bid B = M - O/2, set ask A = M + O/2}. The VBT stands ready to transact at these prices — it's a passive market participant that provides the outside anchor.

**Configurable parameters** (current VBTProfile fields):
- `mid_sensitivity` (0-1): reaction to observed defaults
- `spread_sensitivity` (0-1): spread widening under stress
- `spread_scale` (multiplier): global spread scaling

**Balance sheet note**: The VBT is an entity with its own balance sheet (Plan 034). It holds instruments, has cash, and transacts. It is not an abstract price oracle.

### Profile 4: TreynorBankProfile

**Activity**: Bank lending and treasury management

**Current basis**: `BankProfile` + `BankingSubsystem` — CB corridor pricing, Treynor-derived lending rates, credit risk loading, reserve management

This is actually **two composable activities**:

#### Sub-profile 4a: BankLendingProfile

- **Observe**: Borrower needs (who is requesting credit), own reserve position, CB corridor rates, borrower creditworthiness (via structural access — bank sees depositors' balances)
- **Value**: Price loans using Treynor-derived rate. corridor_mid(kappa) gives the base. credit_risk_loading x P_default gives the per-borrower premium. This is the bank's valuation of credit risk.
- **Assess**: Check credit rationing thresholds (max_borrower_risk). Check own capacity (reserves vs. target). Check portfolio concentration.
- **Choose**: From action set {extend loan at rate r to borrower B, refuse, adjust rate}. Loans create deposits (money creation — a type capacity, not a profile choice).

#### Sub-profile 4b: BankTreasuryProfile

- **Observe**: Own reserve position relative to target. CB facility rates (floor for deposits, ceiling for borrowing). Interbank market conditions.
- **Value**: Compare cost of CB borrowing vs. interbank vs. reserve buffer value.
- **Assess**: Reserve adequacy. Liquidity coverage. Distance from targets.
- **Choose**: From action set {borrow from CB, deposit at CB, lend interbank, borrow interbank, hold}.

**Configurable parameters** (current BankProfile fields):
- `r_base`, `r_stress`: corridor midpoint dynamics
- `omega_base`, `omega_stress`: corridor width dynamics
- `reserve_target_ratio`: target reserves as fraction of deposits
- `credit_risk_loading`: per-borrower pricing sensitivity
- `max_borrower_risk`: credit rationing threshold
- `loan_maturity_fraction`: loan term setting

### Profile 5: CBProfile (Central Bank)

**Activity**: Central banking — corridor setting, standing facilities, lender of last resort

**Current basis**: Hardcoded in CB operations phase (`engines/cb.py` or equivalent)

**Pipeline**:
- **Observe**: System-wide reserve positions, aggregate bank borrowing, corridor utilization
- **Value**: Not applicable in the simple case — CB sets policy rates, doesn't value credit risk. In a richer scenario, the CB might assess systemic risk.
- **Assess**: Monitor system stability indicators. Are banks using the ceiling (stress)? Is reserve distribution adequate?
- **Choose**: From action set {maintain corridor, adjust corridor, extend emergency lending}. In the current Kalecki Ring, the CB's choices are rule-based (fixed corridor, automatic backstop). A richer profile could have adaptive policy.

**Configurable parameters**:
- Corridor rates (currently derived from BankProfile, should be CB's own parameters)
- Backstop lending rules (currently automatic)
- Reserve remuneration rate

**Design note**: Even a rule-based CB is a decision profile — a rule is a simple choice function. Making it a proper profile means we can experiment with different CB policy rules without changing the architecture.

### Profile 6: NBFILendingProfile

**Activity**: Non-bank lending (credit extension from own assets)

**Current basis**: `LenderProfile` + lending engine

**Pipeline**:
- **Observe**: Borrower needs, own cash position, system default rates. Unlike a bank, the NBFI does NOT have structural access to borrower deposit balances — it must use other channels (ratings, published statistics, own bilateral history).
- **Value**: Estimate borrower default probability. Price loans: rate = profit_target + risk_premium_scale x P_default. This is the NBFI's valuation heuristic — currently hardcoded, should be a pluggable valuation method.
- **Assess**: Check portfolio limits (max_single_exposure, max_total_exposure). Check own liquidity — lending depletes cash (no money creation).
- **Choose**: From action set {extend loan at rate r to borrower B, refuse}. Loans come from own cash — a type constraint, not a profile choice.

**Configurable parameters** (current LenderProfile fields):
- `risk_aversion` (0-1): risk premium scaling
- `planning_horizon`: look-ahead for borrower obligations
- `profit_target`: target return rate
- `max_loan_maturity`: maximum loan term

**Known gap**: This profile is underdeveloped (see MEMORY.md). It needs a proper information model (currently borrows the dealer's RiskAssessor or falls back to a crude heuristic) and a proper decision-making model (currently hardcoded rate formula).

---

## Part 4: Cash Flow vs. Valuation

This distinction (discussed in conversation, formalized in Plan 034) is foundational to how decision profiles work.

### Cash Flow Is Fact

At any point in time, an agent has:
- Contractual cash flow obligations with dates and amounts (what it owes)
- Contractual cash flow entitlements with dates and amounts (what it's owed)
- A history of actual cash flows (what was paid, received, at what prices)
- Current cash on hand

These are not valuations. They are accounting facts — part of the information set. Any agent sees its own cash flows with perfect information.

Prices at which things were previously bought and sold are also facts — historical transaction data. The current bid and ask from a dealer are observable facts. These are information, not valuation.

### Valuation Is Heuristic

Valuation enters when cash flow information alone is insufficient for the decision at hand:
- When future cash flows are uncertain (will the counterparty pay?)
- When comparing across time horizons (is a payment in 10 days worth more than one in 3?)
- When estimating the worth of something that might be sold before maturity
- When projecting states of the world that haven't materialized

Valuation is always an agent's belief — a heuristic applied to information. Different agents can apply different heuristics to the same information and reach different conclusions.

### The Universal Hard Constraint

Every decision profile, regardless of its valuation methods or risk attitudes, shares the same bottom line: **meet payment obligations or default**. This is not a behavioral choice — it's a mechanical consequence of the accounting layer (Plan 034: settlement is always mechanical).

This means:
- Cash flow assessment (Step 0 of the pipeline) is universal and non-negotiable
- No profile can choose to default — default happens when cash flow is insufficient
- All behavioral choices (valuation, risk assessment, trading) operate ABOVE the cash flow constraint
- A trader with brilliant valuation who can't pay still defaults
- A trader with no valuation who has enough cash does not default

### Relationship Between Cash Flow and Valuation in the Pipeline

```
Cash flow facts (own obligations, own entitlements, prices paid)
    → Available to all profiles as perfect information
    → Determines the hard constraint (don't default)
    → Provides the baseline for decisions (do I have a shortfall? surplus?)

Valuation heuristics (applied when cash flow isn't enough)
    → Different methods for different profiles
    → Produces estimates with provenance
    → Informs risk assessment and choice
    → Never overrides the cash flow constraint
```

Cash flow and valuation are layered, not independent. An agent's cash flow entitlements include "firm_A owes me 20 on day 12" — but whether firm_A will actually pay is uncertain. The cash flow fact is the contractual picture; valuation tells the agent what to expect from that picture given uncertainty.

---

## Part 5: Per-Agent Heterogeneity

### Current State

All traders in a `DealerSubsystem` share one `TraderProfile` instance. All VBTs share one `VBTProfile`. This means every firm in the ring behaves identically (same risk aversion, same horizon, same aggressiveness).

### Target State

Each agent has its own profile instance. Different agents of the same type can use:
- **Different parameter values**: Firm_3 has risk_aversion=0.8, Firm_7 has risk_aversion=0.2
- **Different profile types**: Firm_3 uses an EV-based valuation, Firm_7 uses a momentum-based valuation

### Configuration

In YAML, profiles can be assigned per-agent or as defaults for a type:

```yaml
# Default profile for all firms (used when no per-agent override)
defaults:
  firm:
    profile:
      type: trading
      risk_aversion: 0.5
      planning_horizon: 10

# Per-agent overrides
agents:
  firm_3:
    profile:
      type: trading
      risk_aversion: 0.8
      planning_horizon: 15

  firm_7:
    profile:
      type: trading
      risk_aversion: 0.2
      aggressiveness: 1.0
```

### Implementation

- Move `trader_profile` from `DealerSubsystem` to individual `TraderState`
- Intention collectors (`collect_sell_intentions`, `collect_buy_intentions`) pull profile from individual trader, not subsystem
- Risk assessment gates (`should_sell`, `should_buy`) use the individual trader's profile
- The subsystem stores a default profile for agents not explicitly configured

### Heterogeneity as Experimental Dimension

Per-agent heterogeneity enables new experiments:
- What happens when 20% of agents are aggressive and 80% are cautious?
- Do aggressive traders cause more defaults or absorb liquidity shocks?
- Does heterogeneity increase or decrease systemic risk?
- How do different valuation methods interact in the same market?

---

## Part 6: Composition Rules

### When Composition Applies

An agent with multiple activities needs multiple profiles. A bank that both lends and makes markets has a lending profile and a market-making profile. These compose into the agent's complete decision specification.

### The Simple Case

An agent with one activity has one profile. That profile IS the complete decision specification. No composition needed.

```python
# Simple: one profile is the whole thing
agent_spec = TradingProfile(risk_aversion=0.5, ...)
```

### The Composed Case

An agent with multiple activities has an ordered list of activity profiles.

```python
# Composed: multiple profiles
agent_spec = ComposedProfile(
    activities=[
        BankLendingProfile(credit_risk_loading=0.02, ...),
        BankTreasuryProfile(reserve_target=0.10, ...),
    ]
)
```

### Composition Rule 1: Activation by Phase

Each activity profile activates during its relevant phase in the simulation's daily cycle. The lending profile activates during the lending phase; the trading profile during the trading phase; the treasury profile during the CB operations phase.

This is already how the code works — phases are the temporal structure of the simulation. Each phase invokes the relevant activity for each agent that participates in that phase.

```
Day N:
  RatingPhase       → activates rating profiles
  LendingPhase      → activates lending profiles (bank + NBFI)
  BankLendingPhase  → activates bank lending profiles
  TradingPhase      → activates trading profiles + market-making profiles
  SettlementPhase   → no profile needed (mechanical)
  CBPhase           → activates treasury profiles + CB profile
```

### Composition Rule 2: Shared Balance Sheet

All activity profiles for the same agent read from and write to the **same balance sheet**. There are no separate cash pools or asset segregation (unless the scenario explicitly defines it, e.g., ring-fenced entities).

This means:
- The order of phases matters — earlier phases get first claim on resources
- Each profile should be aware it's not the only claimant (via the cash flow assessment, which reflects ALL obligations and ALL assets, not just those relevant to one activity)
- A lending profile that deploys too much cash leaves less for the trading profile

Resource awareness is a behavioral parameter within each profile, not an architectural constraint. The `buy_reserve_fraction` concept generalizes: before acting, each profile can reserve resources for other activities. How much to reserve is a profile choice.

### Composition Rule 3: Sequential Information Handoff

Profiles run in phase order. Later profiles can see earlier profiles' outputs:
- Actions taken by earlier profiles change the balance sheet — later profiles see the updated state
- Estimates produced by earlier profiles are logged to the `estimate_log` — later profiles can query them
- Events generated by earlier phases are available in `events_by_day`

No special coordination mechanism needed. The handoff happens through shared state (balance sheet, estimate log, event log).

### Composition Rule 4: Cash Flow Constraint Binds All

The universal cash flow constraint applies across all composed activities. No single activity profile can commit to actions that would cause the agent to default. The constraint is checked at the balance sheet level, not within individual profiles.

In practice:
- Each profile's Step 0 (cash flow assessment) sees ALL obligations, not just those related to its activity
- If the lending profile extended a loan that created a deposit obligation, the trading profile's cash flow assessment reflects that
- The agent cannot fragment its solvency assessment across activities

---

## Part 7: Instrument Binding

### The Problem

Current profiles are entangled with Kalecki-specific instrument types:
- `TraderProfile` assumes "tickets" derived from ring payables
- `LiquidityDrivenSeller` checks shortfall against payable obligations specifically
- The dealer kernel assumes tickets bucketed by maturity
- VBT provides mid/spread for payable-based instruments

### The Solution

Profiles are defined in terms of abstract instrument classes. The scenario binds the specific instrument type.

A `TradingProfile` says: "I trade in [tradeable instruments] using [valuation method]."
The Kalecki Ring scenario says: "tradeable instruments = ring payables, bucketed by maturity."
A different scenario says: "tradeable instruments = corporate bonds, bucketed by credit rating."

A `TreynorDealerProfile` says: "I make two-way markets in [instrument class], with inventory bucketed by [bucket criterion]."
The Kalecki Ring says: "instrument class = payables, bucket criterion = remaining maturity (short/mid/long)."
Another scenario: "instrument class = bonds, bucket criterion = credit grade (AAA/AA/A/BBB)."

### Binding Mechanism

The scenario configuration specifies bindings:

```yaml
scenario:
  instrument_bindings:
    tradeable: Payable           # what's traded in the secondary market
    bucket_by: remaining_maturity  # how instruments are bucketed
    lendable: BankLoan           # what's created when banks lend
    depositable: BankDeposit     # what deposits look like
```

Profiles reference these bindings by role, not by concrete type:

```python
class TradingProfile:
    def choose(self, risk_view, action_set):
        # action_set contains actions like:
        #   Sell(instrument=<whatever 'tradeable' is bound to>, price=bid)
        #   Buy(instrument=<whatever 'tradeable' is bound to>, price=ask)
        # The profile doesn't need to know it's a Payable vs a Bond
        ...
```

### Migration Path

This is a significant refactoring. The migration is incremental:

1. **Phase A**: Extract instrument-specific logic from profiles into binding-aware helpers. Profiles call helpers parameterized by instrument class.
2. **Phase B**: Define the binding configuration in YAML. Kalecki Ring scenarios explicitly declare their bindings (currently implicit).
3. **Phase C**: Test with a second scenario type to verify the binding mechanism works for non-Kalecki instruments.

---

## Part 8: Implementation Plan

### Phase 1: ActivityProfile Protocol and Cash Flow Position

Define the core interfaces without changing any behavior.

**New files**:
- `decision/activity.py`: `ActivityProfile` protocol, `CashFlowPosition` dataclass, `ObservedState`, `Valuations`, `RiskView`, `Action`, `ActionSet` types

**Changes**:
- No behavioral changes. This is pure interface definition.

### Phase 2: Per-Agent Profile Storage

Move profile from subsystem-level to agent-level.

**Changes**:
- `TraderState` gets a `profile: TraderProfile` field (instead of `DealerSubsystem.trader_profile`)
- Intention collectors take profile from individual trader
- Risk assessment gates take profile from individual trader
- Default profile still available for agents without explicit configuration
- Support per-agent profile configuration in YAML

**Behavioral change**: None if all agents get the default profile (current behavior preserved).

### Phase 3: Decompose RiskAssessor

Split the current `RiskAssessor` into the pipeline stages.

**Current**: `RiskAssessor` does belief updating + EV valuation + urgency assessment + accept/reject decision in one class.

**Target**:
- **Belief updater** (Bayesian/Laplace estimation of P_default) → part of observe step (Information layer, Plan 033 Phase 1)
- **Valuer** (EV = (1-p) x face) → part of value step (Plan 034 Phase 2)
- **Position assessor** (urgency, concentration, liquidity) → part of assess step
- **Choice gate** (threshold comparison, accept/reject) → part of choose step

**Migration**: The decomposed components can be wired together to reproduce current behavior exactly. The `RiskAssessor` class becomes a convenience wrapper that runs the four steps in sequence.

### Phase 4: Wire InformationService into Trading

Connect the existing InformationProfile/InformationService to trader decisions.

**Changes**:
- Instantiate `InformationService` per trader (using trader's information profile)
- Trader's observe step queries through `InformationService` instead of directly accessing private state
- Default information profile = current behavior (PERFECT access to own state, system default rate via risk assessor)
- New profiles can restrict or degrade information access

**Depends on**: Plan 033 Phase 1 (hierarchical profile) and Phase 3 (channel-declared sources)

### Phase 5: Wrap Existing Profiles as ActivityProfile Implementations

Retrofit current profiles to implement the `ActivityProfile` protocol.

**Changes**:
- `TraderProfile` → `TradingActivity(ActivityProfile)` wrapping current logic
- Dealer kernel → `TreynorDealerActivity(ActivityProfile)` wrapping current kernel
- `VBTProfile` → `VBTActivity(ActivityProfile)` wrapping current VBT logic
- `BankProfile` → `BankLendingActivity(ActivityProfile)` + `BankTreasuryActivity(ActivityProfile)`
- CB logic → `CBActivity(ActivityProfile)` wrapping current CB phase
- `LenderProfile` → `NBFILendingActivity(ActivityProfile)` wrapping current lending logic

Each wrapper reproduces current behavior exactly while conforming to the common protocol.

### Phase 6: Composition

Enable agents with multiple activity profiles.

**Changes**:
- `AgentDecisionSpec` dataclass holding ordered list of `ActivityProfile`
- Phase dispatch mechanism: each phase activates the relevant profile for each agent
- Shared balance sheet access across all profiles for the same agent
- Test with a bank that has both lending and treasury profiles

### Phase 7: Instrument Binding (Future)

Abstract away Kalecki-specific instrument types.

**Changes**:
- Define instrument binding configuration in YAML
- Profiles reference bindings by role, not by concrete type
- Test with a second scenario type

---

## Part 9: Verification Criteria

### Phase 1
- `ActivityProfile` protocol defined with all four steps + cash flow precondition
- `CashFlowPosition` dataclass captures own obligations, entitlements, cash on hand
- No behavioral changes; existing tests pass

### Phase 2
- Each `TraderState` holds its own profile
- Heterogeneous profiles in a single run produce different behavior
- Default profile reproduces current behavior exactly
- YAML supports per-agent profile configuration

### Phase 3
- `RiskAssessor` decomposed into four separable stages
- Stages can be individually swapped (e.g., different valuation method)
- Reassembled pipeline reproduces current behavior exactly
- New tests verify each stage independently

### Phase 4
- Traders query through `InformationService`, not direct state access
- Default information profile reproduces current behavior
- Degraded profiles produce different (worse-informed) decisions
- Cross-phase interaction tests pass (Plan 032 regression tests)

### Phase 5
- All six profile types implement `ActivityProfile` protocol
- Each produces the same behavior as current implementation
- Profiles are interchangeable where type capacities allow

### Phase 6
- Composed agent with two activities (e.g., bank lending + treasury) works correctly
- Phase ordering respected — earlier activities have resource priority
- Cash flow constraint checked at agent level, not per-activity
- Single-activity agents work unchanged (one profile = whole profile)

### Phase 7
- Profiles reference instrument classes by role binding
- Kalecki Ring scenarios declare bindings explicitly
- (Future) Second scenario type validates the binding mechanism

---

## Design Principles Summary

1. **Three layers, clear boundaries**: Balance sheet (accounting) / Agent type (capacities) / Decision profile (behavior). Each has a distinct role and interface.

2. **Profiles are per-agent**: Not per-type, not per-run. Each agent instance can have its own profile, enabling heterogeneity within a single simulation.

3. **Cash flow is the universal constraint**: Every profile, every agent, every scenario — the bottom line is meeting payment obligations. This is mechanical, not behavioral.

4. **Valuation is heuristic, cash flow is fact**: Prices paid and amounts owed are information. Estimated values and projected cash flows are beliefs. The distinction is fundamental (Plan 034).

5. **One protocol, many implementations**: All profiles implement `ActivityProfile` with the same four-step pipeline. The differences are in HOW each step is executed, not in the pipeline structure.

6. **Composition through shared state**: Multiple activity profiles for one agent compose through the balance sheet, estimate log, and phase ordering. No special coordination mechanism needed.

7. **Generality through binding**: Profiles are defined in terms of abstract instrument classes. Scenarios bind concrete instrument types. The same profile type works across different scenarios.

8. **Complexity is opt-in**: A basic Kalecki Ring firm has one simple trading profile. A complex bank has composed lending + treasury + market-making profiles. The architecture handles both without special cases.

9. **Structural access ≠ behavioral choice**: What an agent CAN see (type capacity) is distinct from what it CHOOSES to look at (profile preference). Both feed into the information pipeline.

10. **Every entity is a balance sheet with a profile**: Dealers, VBTs, banks, the CB — all are balance sheets with decision profiles. None are special constructs outside the architecture.
