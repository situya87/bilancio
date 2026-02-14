# Plan 033: Information & Decision Architecture

**Status**: Design document (conceptual)
**Depends on**: Plan 032 (RiskAssessor), Plan 024 (Dealer Redesign), Non-Bank Lender, Information Access Framework (PR #44)
**Goal**: Define the architectural relationship between information access, heuristic formation, and decision-making across all agent types

---

## Motivation

The simulation currently has two partial implementations of agent cognition:

1. **Dealer/Trader**: The `RiskAssessor` observes defaults and computes probabilities; the `TraderProfile` parameterizes risk behavior. These work but aren't connected to a general framework.

2. **Non-Bank Lender**: The information access framework (PR #44) controls what the lender can observe, with noise and access levels. But it models information as "truth + noise" rather than as something derived from the agent's actual position in the system.

Neither captures the full picture. This document defines the architecture that unifies them.

---

## Core Principle: No Magic Numbers

Every piece of knowledge an agent has must be **traceable to an information source within the system**. Heuristics are not externally injected parameters — they are derived from information flowing through channels. The quality of a heuristic is an emergent property of the channel, not a knob we tune.

**Corollary**: complexity is opt-in. A basic ring simulation needs no information or decision model. Each layer has a trivial default. You engage with richer models only when the simulation demands it.

---

## The Three Aspects of Agent Cognition

### 1. Access — What Can You See?

Some information is structurally invisible to certain agents. A lender cannot see the obligation graph. A trader cannot see balance sheets. A firm knows its own cash but not the system-wide liquidity.

This is the **access control** layer. It answers: "Is this information element available to this agent at all?"

**Implementation**: `AccessLevel` enum (`NONE` / `NOISY` / `PERFECT`) on each information element. Already built in PR #44.

### 2. Channels — Where Does Your Understanding Come From?

Agents don't receive "the truth with noise." They derive understanding through specific channels, each with its own properties:

| Channel | Source | Example | Quality Depends On |
|---------|--------|---------|-------------------|
| **Self-derived** | Own direct observations | "3 of my 20 borrowers defaulted" | Sample size, representativeness |
| **Market-derived** | Prices set by other agents | Dealer bid/ask, VBT mid price | Market thickness, participant informatedness |
| **Network-derived** | Signals from counterparties | Late payments, partial settlements | Network position, relationship depth |
| **Institutional** | Published by system agents | Central bank statistics, rating agency scores | Methodology, staleness, coverage |

Key properties of channels:

- **Endogenous**: Every channel traces back to information within the system. A credit rating comes from a rating agent that itself has imperfect observations.
- **Structured degradation**: Channels don't fail randomly. Self-derived heuristics are poor with few observations. Market prices are poor in thin markets. Network signals are poor at the periphery.
- **Agent-produced**: Some agents are information *producers* (dealers produce prices, rating agencies produce ratings) as well as consumers. This creates information interdependence.

### 3. Estimation Quality — How Well Can You Process What You See?

Given what you can access through your channels, how accurately can you estimate the quantities you need for decisions? This captures:

- **Sample size effects**: 3 observations vs. 300
- **Staleness**: Information from 5 days ago vs. today
- **Aggregation loss**: Seeing a total vs. the breakdown
- **Bilateral limitation**: Only seeing your own interactions vs. everyone's

**Implementation**: The `NoiseConfig` types from PR #44 (`EstimationNoise`, `LagNoise`, `SampleNoise`, `AggregateOnlyNoise`, `BilateralOnlyNoise`). These are not "random error on truth" — they model the structural limitations of the agent's estimation capability given its channels.

The estimation quality is ideally *derived from* channel properties (e.g., `error_fraction` computed from sample size), but can be set directly as a starting point.

---

## The Information Hierarchy

When an agent considers a transaction, its information needs are hierarchical:

```
Level 1: SYSTEM-WIDE
  "What is the general state of the world?"
  ├── Aggregate default rate
  ├── Total system liquidity
  ├── Instrument-type statistics (default rates by type, bucket, issuer kind)
  └── Recovery/settlement rates

Level 2: COUNTERPARTY
  "What do I know about agent B?"
  ├── B's balance sheet (cash, assets, liabilities, net worth)
  ├── B's default/settlement history
  ├── B's track record (recent N days)
  └── B's connectivity / network position

Level 3: INSTRUMENT
  "What do I know about this instrument type in general?"
  ├── Market prices (dealer quotes, VBT anchors)
  ├── Default rates by maturity bucket
  ├── Price trends (spread widening?)
  └── Implied default probability from spreads

Level 4: COUNTERPARTY x INSTRUMENT x ACTION
  "What do I know about B in the context of this specific transaction?"
  ├── B's history with this instrument type
  ├── B's repayment history with ME specifically (bilateral)
  ├── My current exposure to B via this instrument
  └── B's default rate in this maturity bucket
```

Each level narrows the scope and increases specificity. An agent may have perfect Level 1 information but no Level 4 information, or vice versa.

---

## The Decision Hierarchy

Decision-making mirrors the information hierarchy. Each level produces outputs that constrain the next:

### Level 1: Portfolio Strategy

**Question**: "How should I deploy my capital overall?"

**Inputs**: System-wide information (Level 1), own balance sheet (always perfect)

**Outputs**: Target utilization, risk budget, sector/type allocation

**Example — Lender**: "Deploy 80% of capital, target 5% return, keep max single-name exposure at 15%."

**Example — Trader**: "Sell assets only when projected shortfall exceeds buffer B."

**Current implementation**: `LendingConfig` parameters (max_total_exposure, base_rate), `TraderProfile` (risk_aversion, planning_horizon). These are static — they don't adapt to system conditions yet.

### Level 2: Counterparty Screening

**Question**: "Should I transact with agent B at all?"

**Inputs**: Counterparty information (Level 2), portfolio strategy constraints (Level 1 output)

**Outputs**: Eligible counterparty set, per-counterparty risk assessment

**Example — Lender**: "B has defaulted 3 times in the last 10 days — exclude."

**Example — Trader**: "B's claims are too risky given my risk aversion — skip."

**Current implementation**: `max_default_prob` filter in lending, `RiskAssessor.estimate_default_prob()` for traders.

### Level 3: Instrument Selection

**Question**: "Which instrument type should I use for this transaction?"

**Inputs**: Instrument information (Level 3), counterparty assessment (Level 2 output)

**Outputs**: Instrument type, maturity bucket, general terms

**Example — Lender**: "Short-maturity loans for this risk bucket."

**Example — Trader**: "Sell short-maturity claims first (more liquid)."

**Current implementation**: `maturity_days` in LendingConfig (static), bucket ordering in dealer trading.

### Level 4: Transaction Pricing

**Question**: "At what specific terms should I execute?"

**Inputs**: Counterparty x instrument information (Level 4), all upstream constraints

**Outputs**: Price, rate, amount, accept/reject decision

**Example — Lender**: "Lend 500 to B at 8% (base 5% + risk premium 3%)."

**Example — Trader**: "Sell at dealer bid if bid > my expected value of holding."

**Current implementation**: `rate = base_rate + risk_premium_scale * p_default` in lending, `ev_hold` vs. bid comparison in trading.

---

## The Kalecki Ring Progression

The architecture supports progressive complexity. Each simulation mode engages only the levels it needs:

### Basic Ring (no dealers, no lender)

```
Information needed: Level 4 only (own cash, own obligations)
Decision needed:    None (settle if you can, default if you can't)
Channels used:      Self-derived only (own balance sheet)
```

Everything is mechanistic. No information model, no decision model.

### Ring + Dealers

```
Information needed: Levels 1-4
  L1: System default rate (for VBT pricing)
  L2: Counterparty default history (for risk assessment)
  L3: Market prices (dealer quotes)
  L4: Specific claim valuation (ev_hold vs. bid)
Decision needed:    Levels 2-4
  L2: "Is this issuer too risky?" (RiskAssessor)
  L3: "Which bucket to trade in?" (maturity selection)
  L4: "Accept this price?" (ev_hold comparison)
Channels used:      Self-derived (own defaults), Market-derived (dealer prices)
```

The `RiskAssessor` is a self-derived channel. VBT prices are a market-derived channel. Both are endogenous.

### Ring + Dealers + Lender

```
Information needed: All levels
  L1: System conditions (for portfolio strategy)
  L2: Borrower creditworthiness (for screening)
  L3: Instrument characteristics (for term selection)
  L4: Bilateral history, specific exposure (for pricing)
Decision needed:    All levels
  L1: "How much to lend total?"
  L2: "To whom?"
  L3: "What maturity?"
  L4: "At what rate?"
Channels used:      Self-derived, Market-derived (if lender uses dealer quotes),
                    potentially Network-derived and Institutional
```

### Future: Ring + Dealers + Lender + Rating Agency

The rating agency is an information-producing agent:
- **Observes**: Balance sheets, default histories (with its own access limitations)
- **Produces**: Credit ratings (an institutional channel for other agents)
- **Consumed by**: Lender (for counterparty screening), Trader (for risk assessment)

The rating itself is a heuristic derived from the agency's observations — it degrades when the agency has stale data or limited coverage.

---

## Relationship to Current Implementation

### What PR #44 Built (and what it gets right)

The information access framework provides:
- `AccessLevel` enum — correctly models structural visibility constraints
- `NoiseConfig` types — correctly models estimation quality, even if currently set directly rather than derived
- `InformationProfile` — per-agent configuration of what can be seen
- `InformationService` — query mediator with self-access bypass
- Named presets — `OMNISCIENT`, `LENDER_REALISTIC`, `TRADER_BASIC`
- Backward compatibility — `OMNISCIENT` default means existing simulations unchanged

### What needs to evolve

1. **Flat → Hierarchical profile**: The current 28 flat `CategoryAccess` fields should be organized into `SystemAccess`, `CounterpartyAccess`, `InstrumentAccess`, `TransactionAccess` sub-profiles that mirror the information hierarchy.

2. **Service API → Contextual queries**: Instead of `get_counterparty_cash(agent_id)`, the API should support hierarchical queries:
   ```python
   info.system_view()                           # → Level 1
   info.counterparty_view(agent_id)             # → Level 2
   info.instrument_view(kind, bucket)           # → Level 3
   info.transaction_view(agent_id, kind, role)  # → Level 4
   ```

3. **Noise configs → Channel-derived quality**: Over time, `EstimationNoise(error_fraction=0.15)` should become derivable from channel properties (e.g., "this agent has 20 bilateral observations, so its estimation error is approximately 1/sqrt(20)"). But direct setting remains valid as a shortcut.

4. **Decision model formalization**: Each decision level should be a pluggable component with a trivial default:
   - Level 1: `PortfolioStrategy` (default: fixed parameters from config)
   - Level 2: `CounterpartyScreener` (default: max_default_prob threshold)
   - Level 3: `InstrumentSelector` (default: fixed maturity from config)
   - Level 4: `TransactionPricer` (default: base_rate + risk_premium * p_default)

5. **Channel registry**: A mechanism for agents to declare what channels they have (self-derived, market-derived, etc.) and for the system to route information through those channels.

### Migration path

All changes are additive. The current flat profile and direct noise settings continue to work. The hierarchical structure wraps around them:

```
Phase 0 (current):  Flat profile, direct noise, hard-coded decisions    ✓ Done
Phase 1 (next):     Hierarchical profile, contextual queries            Backward compatible
Phase 2 (later):    Channel-derived estimation quality                  Opt-in
Phase 3 (future):   Pluggable decision components                       Opt-in
Phase 4 (future):   Information-producing agents (rating agencies)      Additive
```

Each phase unlocks new experiment dimensions without breaking existing ones.

---

## Design Principles Summary

1. **No magic numbers**: Every heuristic traces to an information source within the system.
2. **Complexity is opt-in**: Trivial defaults at every layer. A basic ring needs zero configuration.
3. **Endogenous channels**: Information flows through traceable paths (self, market, network, institutional), not through external parameter injection.
4. **Structured degradation**: Information quality degrades for traceable reasons (sample size, staleness, network position), not randomly.
5. **Information shapes decisions**: The decision hierarchy mirrors the information hierarchy. What you can decide depends on what you can observe.
6. **Agents produce and consume information**: Dealers produce prices. Rating agencies produce ratings. This creates information interdependence.
7. **Additive evolution**: Each new capability layers on top without replacing what's below.
