# Experiment Design Guide

This document describes the recommended entrypoints for Bilancio
experiments and the status of legacy mixed-arm modes.

## Recommended Entrypoints

Each primary mechanism now has a clean two-arm sweep. The aggregate
`summary.json` artifact for each sweep includes an `experiment_design`
object that records the baseline arm, treatment arm, shared controls, and
effect formula.

| Experiment | Command | Baseline | Treatment | Effect |
|-----------|---------|----------|-----------|--------|
| Dealer effect | `bilancio sweep balanced` | `passive` | `active` | `trading_effect = delta_passive - delta_active` |
| Bank lending effect | `bilancio sweep bank` | `bank_idle` | `bank_lend` | `bank_lending_effect = delta_idle - delta_lend` |
| NBFI lending effect | `bilancio sweep nbfi` | `nbfi_idle` | `nbfi_lend` | `lending_effect = delta_idle - delta_lend` |

### Dealer Effect (`sweep balanced`)

The standard dealer comparison keeps the balanced dealer surface focused
on secondary-market trading.

- **Passive**: Big entities hold securities but do not trade.
- **Active**: The dealer provides a secondary market and VBT provides
  reference pricing.

```bash
bilancio sweep balanced --cloud \
  --n-agents 100 --kappas "0.3,0.5,1,2" \
  --concentrations "1" --mus "0"
```

### Bank Lending Effect (`sweep bank`)

The bank comparison isolates bank credit activity.

- **Bank idle**: Banks provide deposits and settlement infrastructure.
- **Bank lend**: Banks actively lend to liquidity-constrained agents.

```bash
bilancio sweep bank --cloud \
  --n-agents 100 --kappas "0.3,0.5,1,2"
```

### NBFI Lending Effect (`sweep nbfi`)

The NBFI comparison isolates non-bank lending.

- **NBFI idle**: The NBFI infrastructure is present but does not lend.
- **NBFI lend**: The NBFI provides short-term loans.

```bash
bilancio sweep nbfi --cloud \
  --n-agents 100 --kappas "0.3,0.5,1,2"
```

## Deprecated: Mixed-Arm Modes

The following optional modes in `sweep balanced` are deprecated
compatibility options:

| Flag | Replacement |
|------|------------|
| `--enable-lender` | `bilancio sweep nbfi` |
| `--enable-dealer-lender` | Split into `bilancio sweep balanced` and `bilancio sweep nbfi` |
| `--enable-bank-passive` | `bilancio sweep bank` |
| `--enable-bank-dealer` | `bilancio sweep bank` |
| `--enable-bank-dealer-nbfi` | `bilancio sweep bank` |

These flags create multi-arm experiments that combine several mechanisms
inside one comparison. The mixed output remains readable for older
workflows, but clean causal interpretation should use the dedicated
commands above.

The CLI prints deprecation warnings for balanced NBFI and banking arms.
The balanced runner also emits a `DeprecationWarning` for legacy banking
arms used through direct Python APIs.
