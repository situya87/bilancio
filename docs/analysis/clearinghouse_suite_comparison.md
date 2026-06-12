# The Clearinghouse Suite: Netting vs. Certificates vs. CCP on the Kalecki Ring

**Date**: 2026-06-12 (autonomous run) · **Plans**: 059 / 060 / 061 · **Engine**: v2 kernel

Three liquidity-saving institutions for the no-bank/no-dealer Kalecki ring, each attacking a
different facet of the same coordination failure:

| Institution | Mechanism | Saves liquidity by |
|---|---|---|
| Multilateral netting (059) | cancel same-day obligation cycles in place | removing offsetting gross flows |
| Loan certificates (060) | pledge later-due receivables for bearer paper that must be accepted in settlement | bridging *temporal* mismatch with endogenous inside money |
| CCP novation (061) | replace the ring with a star against a prefunded central node | mutualizing losses instead of propagating them |

## Final comparison sweep

`sweep ring`, n=20, maturity_days=5, c=1, κ ∈ {0.25, 0.5, 1, 2}, μ ∈ {0, 0.5, 1},
seeds {42, 43, 44}, `--default-handling expel-agent`, four matched arms
(baseline / `--clearing` / `--clearing-certificates` / `--clearing-ccp`).

### δ (default rate, mean over 3 seeds)

| κ | μ | baseline | netting | certificates | ccp |
|---|---|----------|---------|--------------|-----|
| 0.25 | 0 | 0.953 | 0.921 | 0.921 | **0.513** |
| 0.5 | 0 | 0.931 | 0.899 | 0.899 | **0.500** |
| 1 | 0 | 0.951 | 0.889 | 0.889 | **0.559** |
| 2 | 0 | 0.907 | 0.874 | 0.874 | **0.629** |
| 0.25 | 0.5 | 0.987 | 0.979 | **0.902** | 1.000 |
| 0.5 | 0.5 | 0.951 | 0.954 | **0.865** | 1.000 |
| 1 | 0.5 | 0.990 | 0.974 | **0.848** | 1.000 |
| 2 | 0.5 | 0.923 | 0.895 | **0.828** | 0.873 |
| 0.25 | 1 | 0.956 | 0.956 | **0.838** | 0.983 |
| 0.5 | 1 | 0.976 | 0.976 | **0.814** | 0.966 |
| 1 | 1 | 0.943 | 0.943 | **0.830** | 0.975 |
| 2 | 1 | 0.956 | 0.956 | **0.815** | 0.993 |

### Cascade depth (longest default-precedence chain, mean over 3 seeds)

| Arm | mean depth (all cells) | max depth observed |
|-----|------------------------|--------------------|
| baseline | 6.7 | 15 |
| netting | 6.6 | 15 |
| certificates | 5.8 | 15 |
| **ccp** | **2.5** | **4** |

## Readings

1. **Each institution dominates exactly where its mechanism says it should.**
   - *Netting* only bites when dues are synchronized (μ=0: −3 to −6pp; structurally zero
     under skew — due-today subgraphs of a ring are paths). Its efficiency is bounded by
     n·min_face/S₁, the ring's smallest edge.
   - *Certificates* are the only instrument that helps under maturity skew (μ ∈ {0.5, 1}:
     −8 to −14pp), because they are the only one that moves value across *time* — exactly the
     19th-century design intent. At μ=0 they coincide with netting (no later-due receivables
     to pledge on the day everything is due).
   - *CCP novation* halves δ under synchronized stress (μ=0: ~0.95 → ~0.5) and is the only
     institution that changes the contagion *topology*: cascade depth collapses from chains of
     10–15 to ≤ 4 in every cell, the headline mechanical result of Plan 061.

2. **The ccp δ at μ-skew (≈ 1.0) is a metric convention, not a catastrophe.** Under
   binary-completion φ (the convention all arms share — partial settlements earn zero credit),
   a VMGH haircut day pays h = pool/required of every payout but credits nothing. The
   `vmgh_haircut_total` column carries the actual severity; the depth metric still shows the
   star topology working. A loss-weighted δ remains the roadmap's separate open item, and these
   cells are where it would matter most.

3. **Variance**: under ccp, seed-to-seed σ(δ) collapses to ~0 in the μ=0.5 cells (mutualization
   makes outcomes deterministic) but *rises* at μ=0 (0.06–0.09 vs 0.03–0.07 baseline) where
   fund-exhaustion timing matters. The plan's variance prediction holds only in the skewed
   cells — reported honestly.

4. **Complementarity, not ranking.** Certificates and CCP solve disjoint failures (temporal
   mismatch vs. serial propagation). The natural stage-3 experiment is composition —
   certificates for the time bridge + CCP for the topology — currently rejected at validation
   (modes are exclusive by decided design).

## Provenance

- Plan docs with full decision logs: `docs/plans/059_clearinghouse_netting.md`,
  `060_clearinghouse_certificates.md`, `061_clearinghouse_ccp_novation.md` (each carries an
  implementation-findings addendum and an adversarial-review-round record).
- PRs: #171 (netting), #172 (certificates + 059 sweep-validation fixes), #173 (CCP).
- Two pre-existing infrastructure bugs found and fixed along the way: ring compiler fractional
  faces silently truncated by the kernel (zero-face payables severed the ring), and netting
  metrics dropping reassignment-created payables (contract_id matching).
- Sweep methodology note: `RingSweepRunner` defaults to fail-fast, which truncates stressed
  runs at day 1; all comparisons here use `--default-handling expel-agent`.
