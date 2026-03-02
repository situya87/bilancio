# ODD Protocol (Bilancio Core Model)

## Overview

Bilancio is an agent-based model of payment obligations, liquidity management,
and intermediary behavior in a ring economy. The model studies how defaults,
credit intermediation, and trading microstructure interact under changing
liquidity conditions.

Primary outcomes:
- default incidence and write-offs
- lending/trading activity
- system and intermediary losses
- stabilization time and event dynamics

## Design Concepts

- Emergence: systemic defaults and liquidity cascades emerge from local
  settlement constraints and credit decisions.
- Adaptation: lenders, banks, and dealers update behavior based on stress,
  risk estimates, and inventory conditions.
- Objectives:
  - households/firms: settle obligations and avoid default
  - lenders/banks: deploy capital subject to risk and reserve constraints
  - dealers/VBT: intermediate payables while managing inventory and cash
- Stochasticity: ring generation and sweep runs use explicit seeded randomness.

## Entities and State Variables

- Central bank:
  - reserve corridor and backstop behavior
- Banks:
  - reserves, lending parameters, projected settlement outflows
- Households/firms:
  - cash, deposits, payable liabilities and receivables
- Non-bank lenders:
  - cash, exposure limits, risk pricing inputs
- Dealer/VBT subsystem:
  - cash, inventory by issuer/maturity bucket, pricing state
- Contracts:
  - payables, deposits, reserves, loans

## Process Overview and Scheduling

Each simulation day applies configured subsystem phases, then checks invariants:
1. settlement/payments
2. optional lending (bank and/or NBFI)
3. optional dealer trading/intermediation
4. bookkeeping, event emission, optional invariant checks

Runs continue until fixed horizon or quiet-day convergence.

## Initialization

Typical initialization path:
1. create agents and contracts from scenario/ring generator
2. seed cash and reserves
3. initialize optional subsystems (banking, lending, dealer)
4. set run options (max days, invariant mode, default handling, seed)

## Inputs

Core policy and scenario controls include:
- `kappa`, `concentration`, `mu`, `monotonicity`
- `Q_total`, agent count, maturity structure
- lending and banking risk/coverage parameters
- dealer pricing and inventory sensitivity parameters
- random seed and sweep replicate settings

## Outputs

- event stream (`events.jsonl`)
- balances snapshot (`balances.csv`)
- run summary/metrics fields used in experiments
- aggregate analysis artifacts (cells, effects, sweep summaries)

## Submodels

- settlement engine
- non-bank lending engine
- bank lending and reserve projection subsystem
- dealer trading/intermediation subsystem
- experimental runners and statistical post-processing

