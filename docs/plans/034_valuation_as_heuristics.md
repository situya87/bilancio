# Plan 034: Valuation as Heuristics, Cashflow as Ground Truth

**Status**: Design document
**Depends on**: Plan 033 (Information & Decision Architecture, Phases 0–4)
**Goal**: Establish that settlement outcomes are the sole determinant of what happens in the simulation, and that all valuation is agent-held belief subject to the agent's information and decision model

---

## Philosophical Foundation

The simulation models a world where **obligations mature and agents either pay or default**. This is the only thing that actually happens. Everything else — credit ratings, expected values, risk premia — exists to support agent decisions that shape the instrument positions which determine those outcomes.

**The ontological claim**: There are exactly two layers of reality, and a precise causal chain connecting beliefs to outcomes.

### Layer 1: Instrument-Level Balance Sheet Reality

The ground truth of the system is the set of bilateral instruments — who holds what, who owes what, when it's due, for how much. This is not a model, not an estimate, not a heuristic. It is the state of the world.

```
Instrument (the fact):
  - Payable PAY-007: Firm_A owes Firm_B 500 units, due day 5
  - Cash CASH-003: Firm_A holds 300 units of cash
  - Deposit DEP-012: Firm_A has 100 units at Bank_1
```

Balance sheets are just views over this instrument set — they are always exact, always up-to-date from the system's perspective. The system knows every instrument perfectly. Individual agents may not.

### Layer 2: Cashflow/Settlement Reality

When day 5 arrives, the settlement engine asks: can Firm_A mobilize 500 units using its means of payment (deposits, then cash, then reserves)? If yes, the payable is settled. If no, Firm_A defaults. This binary outcome is determined entirely by Layer 1 — the instrument positions at that moment.

No heuristic, no valuation model, no credit rating can override this. An agent rated AAA with a "low" default probability still defaults if it doesn't have the cash. An agent rated CCC with "high" default probability still settles if it does.

### The Simulation Loop

If we freeze time at any point, the world consists of:

- **Instruments**: The bilateral contracts — who holds what, who owes what, when it's due. These are facts.
- **Balance sheets**: Views over the instrument set, per agent. Always exact from the system's perspective. Instruments implicitly encode relationships *between* balance sheets (a payable is on one agent's asset side and another's liability side).
- **Agent rules**: What each agent type can do — its means of payment, its role in the system, its policy constraints. These are structural, not beliefs.

From this frozen state, the simulation loop proceeds:

```
┌─ STATE ──────────────────────────────────────────────────────────┐
│  Instruments, balance sheets (views over instruments),           │
│  relationships between balance sheets (via bilateral instruments)│
│  agent-type rules (what each can do)                             │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─ INFORMATION ────────────────────────────────────────────────────┐
│  What the agent can observe of the state (via channels)          │
│  + estimations about what it can't observe:                      │
│    Direct: "I estimate Firm_B has 400 cash" (balance sheet est.) │
│    Indirect: "I estimate this asset's value at 0.85 × face"     │
│    (still an estimation about cashflows — will it be settled?)   │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─ DECISION ───────────────────────────────────────────────────────┐
│  Information + goals + strategy → actions                        │
│  Goals & strategy may themselves be set in different ways:       │
│    Dealers: essentially fixed (market-making mandate)            │
│    Lenders: configurable (risk appetite, return targets)         │
│    Future agents: may have degrees of freedom in strategy-setting│
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─ ACTIONS (offers) ───────────────────────────────────────────────┐
│  Bids, asks, loan offers, sell orders                            │
│  Directed to: specific counterparty / group / open market        │
│  These are commitments to transact, not beliefs                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─ AGREEMENT / TRANSACTION ────────────────────────────────────────┐
│  When offers are accepted: instruments are created or modified   │
│  A loan offer accepted → NonBankLoan instrument created          │
│  A bid accepted → Payable changes hands (claim transfer)         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─ SETTLEMENT ─────────────────────────────────────────────────────┐
│  Performing a balance sheet operation on a particular instrument │
│  as previously agreed: debtor mobilizes means of payment,       │
│  creditor receives. If debtor can't → default.                   │
│  Pure mechanical resolution — no beliefs consulted.              │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
                     NEW STATE (go again)
```

**Key observations about this loop:**

1. **"Value" is always an estimation about cashflows.** Even indirect estimations like "fundamental value of this asset" ultimately reduce to: "will the obligations backing this asset be settled?" There is no abstract "value" floating in the system — only beliefs about whether instruments will perform, derived from whatever the agent can observe about balance sheet positions.

2. **Beliefs never directly cause outcomes.** They do so only indirectly: beliefs shape decisions, decisions produce actions, actions create transactions, transactions modify instruments, instruments determine settlements. A belief that "Firm_A is safe" matters only if it leads to an action (lend to A, buy A's claims) that changes instrument positions.

3. **Strategy is itself a variable.** Some agents (dealers) have essentially fixed strategies. Others (lenders, future agent types) may have degrees of freedom in setting goals and strategy. How strategy-setting is affected by information and experience is a dimension for future exploration — but it sits in the DECISION box, upstream of actions.

4. **Actions are offers, not transactions.** A bid is an offer — it becomes a transaction only upon agreement. This matters for market microstructure: an agent may offer to buy at 0.82, but if no counterparty accepts, no instrument changes. The offer itself doesn't alter the balance sheet.

### What This Means for the Codebase

Code in this system falls into one of the loop stages:

**STATE code** (instrument operations, balance sheet queries): Reads and writes facts. Always exact from the system's perspective. Examples: `add_contract()`, `agent_balance()`, `_remove_contract()`.

**INFORMATION code** (channels, estimation, noise): Takes state facts the agent can access and produces estimations. Every "value" or "probability" computation is in this category — including indirect estimations like "fundamental value," which is still an estimation about whether cashflows will materialize. Examples: `InformationService`, `RiskAssessor.estimate_default_prob()`, `_compute_rating()`.

**DECISION code** (protocols, profiles, strategy): Takes information + goals/strategy and produces actions. Examples: `CounterpartyScreener`, `TransactionPricer`, `should_sell()`, `should_buy()`.

**ACTION code** (offers, order generation): Expresses decisions as offers directed at counterparties. A bid, an ask, a loan offer. These are commitments valid at the moment of expression (the agent's balance sheet supports fulfillment). Examples: dealer quote-setting, lending offer generation.

**AGREEMENT code** (matching, transaction execution): When offers are accepted, instruments are created or modified. Examples: `execute_customer_sell()`, `execute_customer_buy()`, loan origination in `run_lending_phase()`.

**SETTLEMENT code** (mechanical resolution): Performs the balance sheet operations previously agreed upon. Never consults beliefs, never evaluates offers. Reads actual positions, transfers funds, produces binary settle/default outcomes. Examples: `settle_due()`, `_settle_single_payable()`, `_pay_with_deposits()`.

Every function in the codebase should be classifiable into one of these stages. If a function mixes stages (e.g., a settlement function that also computes a "value"), that's a design smell to be refactored.

---

## Current Architecture Assessment

### What Already Embodies This Philosophy

**Settlement engine** (`engines/settlement.py`): Pure Layer 2. Checks actual balance sheet positions, attempts payment using means-of-payment hierarchy, produces binary settle/default outcome. No valuations, no heuristics. This is correct and should not change.

**Information framework** (Plan 033): Correctly models that agents have different access to information and derive understanding through channels with structural degradation. The framework already supports the idea that "truth + noise" is a simplification — the real model is "observations through channels → estimation."

**Decision protocols** (Plan 033 Phase 3): The four-level hierarchy (portfolio → screening → selection → pricing) correctly separates the decision chain. Each level takes information views as input and produces action parameters as output.

**RiskAssessor** (`dealer/risk_assessment.py`): A self-derived channel implementation. It observes settlement outcomes, applies Bayesian updating (Laplace smoothing), and produces default probability estimates. This is a heuristic with clear provenance — but it's wired as infrastructure, not as an agent's belief.

**Rating agency** (Plan 033 Phase 4): An information-producing agent that observes balance sheets and histories, applies a methodology, and publishes ratings. Other agents consume via InstitutionalChannel. This is close to the ideal model.

### What Violates This Philosophy

#### Problem 1: Valuation Without an Agent

The `RiskAssessor.expected_value()` method computes `(1 - P_default) × face_value`. But whose belief is this? The RiskAssessor is attached to the `DealerSubsystem` — it's shared infrastructure, not an individual agent's model. Every trader sees the same default probabilities and computes the same expected values.

**Should be**: Each trader has its own estimation capability (possibly shared infrastructure, but parameterized by the trader's own profile — risk aversion, observability, prior).

#### Problem 2: VBT Mid as External Parameter

The VBT mid price `M = outside_mid_ratio × (1 - P_default)` combines a free parameter (`outside_mid_ratio`) with a system-derived estimate. But `outside_mid_ratio` doesn't trace to any information source — it's an externally injected number.

**Should be**: The dealer *observes* the VBT's bid and ask quotes through a MarketDerived channel — the VBT mid is simply `(bid + ask) / 2` of the observed external market. This is how real market-makers work: they look at where comparable instruments trade elsewhere and anchor their own quotes to that. The quality of this observation depends on channel properties (market thickness, staleness, informativeness), which the information framework already models. The `outside_mid_ratio` then becomes a property of the external market that the dealer reads, not a free parameter we inject.

#### Problem 3: The Default Probability Waterfall

In `InformationService._raw_default_probs()` and `lending._estimate_default_probs()`, there's a priority waterfall: dealer RiskAssessor → rating registry → crude heuristic. This waterfall is hard-coded — it's not something any agent chose. It's the system deciding which oracle to consult.

**Should be**: Each agent declares which channels it uses for default probability estimation. The lender's InformationProfile specifies: "I use the rating agency's published ratings (InstitutionalChannel, coverage 0.8, staleness 1 day)." Another lender might specify: "I use my own bilateral history (SelfDerivedChannel, sample_size 15)."

#### Problem 4: No Belief Provenance

When a lender prices a loan at 8%, we can't trace why. Was it `base_rate 5% + risk_premium_scale 0.20 × P_default 0.15`? Where did 0.15 come from? The rating agency? The RiskAssessor? A heuristic? And what was the rating agency's confidence? None of this is tracked.

**Should be**: Valuations are first-class objects with provenance — who computed them, using what information, through which channel, with what confidence.

#### Problem 5: The Stub Valuation Engine

`engines/valuation.py` defines a `ValuationEngine` protocol and a `SimpleValuationEngine` that just returns face value. This isn't used anywhere. Its design assumes valuation is a system service ("value this instrument for me") rather than an agent activity ("I believe this instrument is worth X because...").

**Should be**: Replaced with a framework where valuation is an agent action, not a system service.

---

## Design: Priors, Partial Settlement, and Simplifications

### Priors Are Scenario Parameters, Not Information-Derived

Agents start the simulation with prior beliefs — e.g., `initial_prior = 0.15` in the RiskAssessor. These don't trace to any information channel; they are initial conditions, analogous to how initial cash endowments define the starting balance sheet. Over time, observations overwhelm the prior, but at t=0 it's scenario setup.

**Design rule**: The user must explicitly confirm priors before any simulation. The configuration should make it clear what is a prior (fixed starting belief) versus what is derived from channels. The user should have the choice to set specific quantities as priors rather than derived values, depending on the simulation's needs. For example:

```yaml
# User chooses: "I want default_prob to be a fixed prior for this simulation"
information:
  counterparty_default_prob:
    mode: prior          # Fixed value, not derived from channels
    value: 0.15

# Or: "I want it derived from the rating agency"
information:
  counterparty_default_prob:
    mode: derived         # Comes from declared channel
    channel: institutional
    source: rating_agency
```

This prior-vs-derived choice is itself a simulation design decision that the user makes, not something the framework imposes.

### Settlement Outcomes Are Richer Than Binary

The settlement engine already handles partial settlement — pay what you can, default on the rest with a specific shortfall amount. The actual outcome space is:

- **Full settlement**: debtor pays 100% of obligation
- **Partial settlement + default**: debtor pays X%, defaults on remainder
- **Full default**: debtor pays 0%

This matters for the INFORMATION stage — agents observing settlement outcomes should potentially see **recovery rates**, not just boolean defaulted/not-defaulted. The RiskAssessor currently tracks only the boolean (`defaulted: bool`), which is a simplification. A 60% recovery is more informative than "defaulted." Future phases should enrich settlement observation to include recovery information.

### Offer and Agreement Collapse by Agent Type

In the current implementation, the ACTION and AGREEMENT stages are collapsed — offers and acceptance happen atomically. This is a valid simplification, but it's agent-type-dependent:

- **Dealers**: Stand ready to trade. They post quotes (offers) and any eligible counterparty can accept immediately. The collapse is natural — the dealer's whole purpose is to make this seamless.
- **Lenders**: Evaluate borrowers and offer loan terms. The borrower accepts in the same phase. Collapsed, but could be separated if we wanted borrower negotiation.
- **Basic ring agents**: No offers at all — obligations exist from setup. The ACTION and AGREEMENT stages are entirely absent (the loop goes STATE → SETTLEMENT → NEW STATE).
- **Future agents**: May have separated offer/agreement phases (e.g., limit orders that persist across days, auctions with bidding rounds, loan syndication with multiple lender offers).

The framework should acknowledge this collapse as a simplification that different agent types can override when needed.

### Strategy-Setting as a Future Dimension

Some agents have essentially fixed strategies (dealers: market-making mandate). Others have configurable strategies set at simulation start (lenders: risk appetite, return targets via profile dataclasses). A future extension would allow agents with **degrees of freedom in strategy-setting** — agents that adapt their goals and strategy based on experience during the simulation.

This sits in the DECISION stage as a meta-decision: "given what I've learned, should I change how I decide?" Examples:
- A lender that tightens credit standards after observing rising defaults
- A trader that becomes more conservative after losses
- A fund manager that shifts allocation between asset classes

The current profile-based system (TraderProfile, RatingProfile, LendingConfig) is the right foundation — profiles are the "strategy" and they're configurable per experiment. Adaptive strategy-setting would mean profiles that change during the simulation. This is noted here as a future dimension to be taken up when needed for specific simulations.

---

## Design: The Belief Model

### Core Concept: Estimate

An `Estimate` is a belief held by an agent about a quantity. It records what was estimated, by whom, using what information, and with what confidence.

```python
@dataclass(frozen=True)
class Estimate:
    """A belief held by an agent about a quantity in the system.

    Not a fact — a fact is an instrument on a balance sheet.
    An Estimate is what an agent thinks, given what it can observe.
    """
    quantity: str           # What is being estimated (e.g., "default_prob", "fair_value")
    subject_id: str         # About whom/what (e.g., agent_id, instrument_id)
    value: Decimal          # The estimated value
    confidence: Decimal     # How confident (0=no idea, 1=certain), derived from channel
    observer_id: str        # Who made this estimate
    day: int                # When it was computed
    channel_type: str       # How (e.g., "self_derived", "institutional", "market")
    sample_size: int | None = None   # Observations behind this (if applicable)
    staleness_days: int = 0          # How old the underlying data is
```

### Where Estimates Are Produced

| Producer | Quantity | Channel | Consumer |
|----------|----------|---------|----------|
| RiskAssessor | `default_prob` | SelfDerived (observed settlements) | Traders (sell/buy decisions) |
| Rating Agency | `default_prob` | Institutional (balance sheet analysis) | Lenders, traders |
| Dealer (VBT) | `fair_value` | MarketDerived (transaction history) | Traders (price comparison) |
| Lender | `default_prob` | SelfDerived (own borrower outcomes) | Lender itself (pricing) |
| Trader | `expected_payoff` | Composite (RiskAssessor + own info) | Trader itself (hold/sell) |

### Where Estimates Are Consumed

The decision protocols are the natural consumption points:

| Decision Level | Estimates Needed | Decision Output |
|----------------|-----------------|-----------------|
| L1: Portfolio Strategy | System-wide default rate, own balance sheet | Max exposure, risk budget |
| L2: Counterparty Screening | Counterparty default_prob | Eligible set |
| L3: Instrument Selection | Default_prob by maturity bucket | Which instrument type |
| L4: Transaction Pricing | Counterparty-specific default_prob, fair_value | Price, rate, accept/reject |

### What Is Not an Estimate

The settlement engine does not use estimates. It reads actual instrument positions:

```python
# This is NOT an estimate — it's a balance sheet query
cash_available = sum(c.amount for c in agent.asset_ids if c.kind == CASH)

# This IS an estimate — it's a belief
expected_recovery = (1 - estimated_default_prob) * face_value
```

The distinction matters because balance sheet queries are always exact (Layer 1), while estimates can be wrong.

---

## Design: Valuation as Decision Protocol

### Replace Free-Floating Valuations With Protocol Methods

Currently, valuation code lives in scattered locations:
- `RiskAssessor.expected_value()` — trader EV computation
- `dealer_sync._update_vbt_credit_mids()` — VBT pricing
- `lending.py: rate = base_rate + scale * p_default` — lender pricing
- `rating.py: _compute_rating()` — rating methodology

Each of these is a heuristic. Each should be a method on a decision protocol, parameterized by a profile, using information from a service.

### The Valuation Protocol

```python
class InstrumentValuer(Protocol):
    """Produce an Estimate of an instrument's value.

    This is Level 4 of the decision hierarchy, specialized for valuation.
    Every valuation is an agent's belief, not a system fact.
    """
    def value(
        self,
        instrument_id: str,
        info: InformationService,
        day: int,
    ) -> Estimate: ...
```

Concrete implementations:

```python
class EVHoldValuer:
    """Trader heuristic: EV = (1 - P_default) × face.

    Simple expected value computation. The quality of this
    valuation depends entirely on the quality of P_default,
    which depends on the agent's information channels.
    """

class CoverageRatioValuer:
    """Rating agency heuristic: score based on net_worth / obligations.

    Maps balance sheet coverage to implied default probability.
    Quality depends on how accurately the agency observes balance sheets.
    """

class SpreadImpliedValuer:
    """Market-derived heuristic: infer value from observed bid/ask spreads.

    Uses market prices as information about credit quality.
    Quality depends on market thickness and price informativeness.
    """
```

### VBT as an Entity With Its Own Balance Sheet

The VBT is not an abstract "price oracle" — it is an entity with its own balance sheet that quotes bid and ask prices and transacts when the dealer needs to lay off risk. The dealer *observes* the VBT's bid and ask through a MarketDerived information channel (currently with PERFECT access, degradable in future simulations), and can also *transact* with the VBT.

```python
class DealerMarketObserver:
    """Dealer's observation of the VBT's quotes.

    The VBT is an entity that:
    - Has its own balance sheet (holds instruments, has inventory)
    - Quotes its own bid and ask prices (using its own heuristic)
    - Transacts with the dealer when the dealer lays off risk

    The dealer has a MarketDerived channel to the VBT:
    - Observes VBT bid and ask prices (currently PERFECT, degradable later)
    - Derives mid = (bid + ask) / 2
    - Uses observed mid to anchor its own quotes to customers

    The dealer then sets its own quotes as actions:
    - dealer_bid = observed_mid - half_spread
    - dealer_ask = observed_mid + half_spread
    """
```

The `outside_mid_ratio` parameter in the current implementation corresponds to a property of the VBT itself — how the VBT prices instruments it's willing to trade. The VBT's pricing is itself a heuristic (simplest version: `(1 - system_default_rate) × face`). More sophisticated VBT models could use richer information. The dealer reads the VBT's quotes and anchors to them.

The key shift: the dealer's quotes are **actions** (decision outputs), not beliefs. The dealer decides to bid 0.82 and ask 0.88 — this is a commitment to transact at those prices. The *belief* is the dealer's estimate of fair value (informed by VBT observation + its own settlement history). The *action* is the bid/ask, which also incorporates the dealer's spread policy, risk appetite, and inventory position.

### Bids and Asks Are Actions, Not Beliefs

This distinction matters throughout the system:

| Thing | Category | Example |
|-------|----------|---------|
| "I estimate P_default = 0.15" | **Belief** (estimate) | RiskAssessor output |
| "I estimate fair value = 0.85 × face" | **Belief** (estimate) | Derived from P_default |
| "I will buy at 0.82" | **Action** (bid) | Decision output: belief + risk tolerance + liquidity |
| "I will sell at 0.88" | **Action** (ask) | Decision output: belief + spread policy + inventory |
| "Firm_A has 300 cash" | **Fact** (instrument) | Balance sheet reality |
| "Firm_A paid its obligation" | **Fact** (settlement) | Cashflow reality |

Beliefs inform actions. Actions create transactions. Transactions modify facts. Facts determine settlements. Settlements are observable and feed back into beliefs. The loop is complete, and every link in the chain has a clear ontological status.

---

## Design: Channel-Declared Information Sources

### Current: Implicit Waterfall

```python
# In _raw_default_probs():
if dealer_risk_assessor:     # Try this first
    return assessor_probs
elif rating_registry:         # Then this
    return registry_probs
else:                         # Last resort
    return crude_heuristic
```

### Proposed: Agent Declares Its Channels

Each agent's InformationProfile already declares what it can access (AccessLevel per category). The missing piece is: for a given information category, which *source* does the agent use?

```python
@dataclass
class ChannelBinding:
    """Binds an information category to a specific source for an agent."""
    category: str                    # e.g., "counterparty_default_history"
    source: str                      # e.g., "rating_agency", "self_derived", "dealer_risk_assessor"
    channel: Channel                 # The channel with its properties
    priority: int = 0               # If multiple sources, try in priority order
```

An agent's information model then becomes:

```python
# Lender with ratings:
bindings = [
    ChannelBinding("counterparty_default_prob", "rating_agency",
                   InstitutionalChannel(staleness_days=1, coverage=0.8), priority=1),
    ChannelBinding("counterparty_default_prob", "self_derived",
                   SelfDerivedChannel(sample_size=5), priority=2),
]

# Trader:
bindings = [
    ChannelBinding("counterparty_default_prob", "risk_assessor",
                   SelfDerivedChannel(sample_size=20), priority=1),
    ChannelBinding("instrument_fair_value", "market",
                   MarketDerivedChannel(market_thickness=0.5), priority=1),
]
```

The InformationService then resolves estimates by following the agent's declared bindings, not a hard-coded waterfall.

---

## Design: Settlement Engine Purity

The settlement engine (`engines/settlement.py`) is already correct. It must remain a pure Layer 2 mechanism:

**Invariants that must be preserved:**
1. Settlement logic never calls any valuation function
2. Settlement logic never reads `rating_registry` or `risk_assessor`
3. Settlement logic uses only `agent.asset_ids`, `agent.liability_ids`, and `system.state.contracts`
4. Default/settle outcome is determined solely by balance sheet positions at the moment of settlement
5. The `mop_rank` (means of payment hierarchy) is policy, not a heuristic — it defines the order in which payment methods are attempted

**The only feedback from Layer 2 to the belief system** is the `risk_assessor.update_history()` call in `_settle_single_payable()`. This is correct — settlement outcomes are observable facts that agents can use to update their beliefs.

---

## Implementation Phases

### Phase 1: Estimate Dataclass and Provenance

**Goal**: Make valuations first-class objects with provenance, without changing any behavior.

**Changes**:
- Create `Estimate` dataclass in `bilancio/information/estimates.py`
- Modify `RiskAssessor.expected_value()` to return `Estimate` (or add `expected_value_estimate()` alongside for backward compatibility)
- Modify `rating.py: _compute_rating()` to return `Estimate`
- Modify `lending.py` pricer to produce `Estimate` (wrapped in `LinearPricer`)
- Add `estimates` list to agent state or `DayReport` for post-hoc analysis

**Backward compatible**: Existing code continues to use `.value` from estimates. New analysis tools can inspect provenance.

**Files**: `information/estimates.py` (NEW), `dealer/risk_assessment.py`, `engines/rating.py`, `engines/lending.py`, `decision/protocols.py`

### Phase 2: InstrumentValuer Protocol

**Goal**: Unify all "what is this worth?" computations under a single protocol.

**Changes**:
- Create `InstrumentValuer` protocol in `decision/protocols.py`
- Implement `EVHoldValuer` (wraps current RiskAssessor logic)
- Implement `CoverageRatioValuer` (wraps current rating methodology)
- Wire into dealer trading (replace direct `expected_value()` calls)
- Wire into lending (replace direct rate formula)
- Delete or deprecate `engines/valuation.py` stub

**Behavioral change**: None. Same computations, same results, wrapped in protocol.

**Files**: `decision/protocols.py`, `decision/valuers.py` (NEW), `dealer/simulation.py`, `engines/lending.py`, `engines/valuation.py` (deprecate)

### Phase 3: Channel-Declared Information Sources

**Goal**: Replace hard-coded waterfall in `_raw_default_probs()` with agent-declared channel bindings.

**Changes**:
- Create `ChannelBinding` dataclass in `information/channels.py`
- Add `channel_bindings` to `InformationProfile`
- Modify `InformationService._raw_default_probs()` to follow bindings
- Modify `lending._estimate_default_probs()` to follow bindings
- Update presets to declare bindings
- Update config (YAML) to allow channel binding configuration

**Behavioral change**: Same results for default presets. New capability: agents can be configured to use specific information sources.

**Files**: `information/channels.py`, `information/models.py`, `information/service.py`, `information/presets.py`, `engines/lending.py`, `config/models.py`

### Phase 4: VBT Mid as Market Observation

**Goal**: Reframe the dealer's relationship to VBT quotes as information channel observation, not free parameter injection.

The VBT already exists as an entity with its own balance sheet — it holds instruments, has inventory, and quotes bid/ask prices that the dealer transacts on when laying off risk. The change is conceptual and architectural: make the dealer's access to VBT quotes go through a `MarketDerivedChannel` (currently PERFECT, degradable for future simulations) rather than being wired as a direct parameter.

**Changes**:
- Give the dealer a `MarketDerivedChannel` to observe VBT bid/ask (initially PERFECT access)
- Dealer derives observed_mid from channel, uses it to anchor own quotes
- `outside_mid_ratio` reframed as a property of the VBT's own pricing heuristic
- The VBT's pricing heuristic is itself pluggable (simplest: current formula `(1 - system_default_rate) × face`)
- Dealer's bid/ask spread is a separate decision parameter (action, not belief)
- Add dealer-specific InformationProfile and ChannelBindings
- Future simulations can degrade the channel (staleness, noise) to explore what happens when the dealer has imperfect market information

**Behavioral change**: Default VBT model + PERFECT channel reproduces current behavior exactly. Degraded channels available as configuration options for future experiments.

**Files**: `decision/valuers.py`, `dealer/domain/valuation.py` or `engines/dealer_sync.py`, `information/presets.py`, `config/models.py`

### Phase 5: Diagnostic and Analysis Tools

**Goal**: Make it easy to understand why agents valued things the way they did.

**Changes**:
- Add estimate logging to event stream (optional, controlled by config)
- Add estimate visualization to HTML reports
- Add "belief trajectory" analysis: how did agent X's belief about agent Y's creditworthiness evolve over the simulation?
- Add "belief vs reality" comparison: plot estimated default_prob vs actual default outcomes

**Files**: `analysis/beliefs.py` (NEW), `ui/templates/`, `engines/simulation.py`

---

## The Kalecki Ring as Special Case

The basic Kalecki ring is the simplest instantiation of this framework. Most loop stages are trivially empty:

| Loop Stage | Basic Ring | + Dealers | + Lenders | + Rating Agency |
|------------|-----------|-----------|-----------|-----------------|
| **STATE** | Instruments from setup | + dealer/VBT balance sheets | + loan instruments | + agency agent |
| **INFORMATION** | (none needed) | RiskAssessor (self-derived), VBT quotes (market) | + counterparty screening | + institutional ratings |
| **DECISION** | (none — settle if you can) | Trader hold/sell/buy decisions | + lending decisions (4-level protocol) | + rating methodology |
| **ACTION** | (none — no offers) | Dealer quotes, trader orders | + loan offers | + published ratings |
| **AGREEMENT** | (none — obligations pre-exist) | Trade execution | + loan origination | (ratings are published, not transacted) |
| **SETTLEMENT** | Settle or default | Same, + dealer settlement | Same, + loan repayment | Same |

Each additional agent type activates more of the loop without changing the stages already in use. The framework is layered: you can run a basic ring (STATE → SETTLEMENT → NEW STATE) or a full simulation with all stages active, and the same architecture handles both.

---

## Design Principles Summary

1. **Two layers of reality**: Instruments (Layer 1) and settlements (Layer 2). Everything else is belief or action.
2. **Six stages, one causal loop**: State → information → decision → action → agreement → settlement → new state. Each stage has a clear ontological status. Different simulation configurations activate different stages.
3. **Beliefs are estimates**: No computation of "value" or "probability" without an agent, an information source, and a heuristic. Beliefs have provenance (observer, channel, day, confidence).
4. **Actions are decision outputs**: Bids, asks, loan offers are not beliefs — they are commitments produced by running beliefs through decision protocols. They also incorporate risk tolerance, liquidity needs, and spread policy.
5. **Settlement is always mechanical**: The settlement engine reads actual positions and produces outcomes (full settlement, partial + default, or full default). It never consults beliefs and never evaluates actions.
6. **Channels are declared, not hard-coded**: Each agent specifies where its information comes from; the system routes accordingly.
7. **Priors are explicit**: Initial beliefs are scenario parameters that the user confirms. The user chooses what is a prior versus what is channel-derived.
8. **Heuristics are pluggable**: Valuation methodologies are decision protocol implementations, configurable per agent and per experiment.
9. **Complexity remains opt-in**: A basic ring needs no beliefs, no valuations, no channels. Each layer activates only when the simulation demands it. The Kalecki ring is a special case where most stages are trivially empty.
10. **Feedback is factual**: The only information that flows from settlement back to beliefs is observable outcomes (settled, partially settled, defaulted, with recovery amounts). These are facts, not beliefs.

---

## Verification Criteria

### Phase 1
- `Estimate` objects produced by all valuation paths
- Provenance fields populated (observer_id, channel_type, day, confidence)
- Existing test suite passes with no behavioral changes
- New tests verify provenance is correct

### Phase 2
- All valuation goes through `InstrumentValuer` protocol
- `engines/valuation.py` stub deprecated or removed
- Dealer trading decisions use `EVHoldValuer`
- Lending pricing uses protocol-based pricer
- Same simulation outcomes as before

### Phase 3
- No hard-coded waterfall in `_raw_default_probs()`
- Agent presets declare channel bindings
- Different agents can use different information sources for the same quantity
- Default presets reproduce current behavior

### Phase 4
- VBT modeled as price-setting entity with its own heuristic
- Dealer observes VBT quotes through MarketDerived channel
- `outside_mid_ratio` is a VBT property, not a dealer free parameter
- Default VBT model + channel reproduces current behavior
- Dealer's bid/ask (actions) clearly separated from fair value estimate (belief)

### Phase 5
- Belief trajectories visible in HTML reports
- "Belief vs reality" analysis available
- Estimate events in event stream (when enabled)
