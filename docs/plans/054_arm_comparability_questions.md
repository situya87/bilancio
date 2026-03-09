# 054: Arm Comparability Questions

## Status: Open questions (not yet a plan)

## Issue

The balanced comparison arms have an asymmetry in claim topology that may affect cross-arm comparability.

### How it works today

When the compiler sets up a balanced comparison, `vbt_share_per_bucket` and `dealer_share_per_bucket` determine the fraction of each maturity bucket's claims that go to VBT/dealer entities. Traders issue payables to these entities and receive additional cash to fund those obligations.

The `vbt_dealer_cash_scale` then redistributes the *cash* across arms, but the *claim structure* stays the same (except in `bank_idle`/`bank_lend` modes which skip VBT/dealer entirely).

| Arm | Traders owe VBT/Dealer? | VBT/Dealer have cash? | Who provides liquidity? |
|-----|------------------------|-----------------------|------------------------|
| **passive** | Yes | Yes (idle) | Nobody |
| **active (dealer)** | Yes | Yes (active) | Dealer trades claims |
| **nbfi** | Yes | **No** (0 cash) | NBFI lends |
| **banking** | Yes | **No** (0 cash) | Bank lends |
| **bank_idle** | **No** (no VBT/Dealer) | N/A | Bank (idle reserves) |
| **bank_lend** | **No** (no VBT/Dealer) | N/A | Bank lends |

### The problem

1. **Dealer vs NBFI/bank**: In the dealer arm, the claims held by VBT/dealer are functional — they're the tradeable inventory that enables market-making. In NBFI/bank arms, those same claims are dead weight: traders still owe VBT/dealer but get no service in return. This structurally handicaps the NBFI/bank arms, potentially understating their relief effect.

2. **`banking` vs `bank_idle`/`bank_lend`**: Different claim topologies — one has inert VBT/dealer entities draining trader cash, the other doesn't. These aren't directly comparable.

3. **NBFI vs bank**: These *are* comparable to each other (same claim structure, same cash pool, different intermediary type). Within-pair comparisons are clean.

### Questions to resolve

- How large is the VBT/dealer claim burden relative to total system debt? If it's small (e.g. claims are ~37.5% of bucket totals at default pool), does it meaningfully affect results?
- Should non-dealer arms drop VBT/dealer claims entirely (like `bank_idle` does) for cleaner comparison? This would change what "equal capital" means — the dealer's capital is partly in claims, partly in cash.
- Is the right baseline the passive arm (which also has VBT/dealer claims with cash but no trading)? If so, the *differential* between passive and each treatment arm is still valid, since passive shares the same claim burden.
- Could we add a `skip_vbt_dealer` option to NBFI/bank modes to enable both setups and measure the claim burden effect directly?

### Why it might not matter

The passive baseline shares the same claim topology as the NBFI/bank treatment arms (VBT/dealer hold claims + cash but don't trade). So `relief = delta_passive - delta_treatment` is computed against a baseline with the same structural handicap. The handicap cancels out in the difference.

The comparison becomes problematic only if we try to compare relief *across* intermediary types (e.g. "dealer relief > NBFI relief, therefore dealers are better"). The dealer arm's claims are an asset; the NBFI arm's claims are a liability to traders.
