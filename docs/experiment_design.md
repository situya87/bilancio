# Experiment Design Guide

This document describes the recommended entrypoints for running
bilancio experiments and the status of legacy modes.

## Recommended Entrypoints

Each experiment type has a dedicated sweep command with a clean 2-arm
design:

| Experiment | Command | Arms | What It Tests |
|-----------|---------|------|--------------|
| Dealer effect | `bilancio sweep balanced` | passive, active | Does secondary market trading reduce defaults? |
| Bank lending effect | `bilancio sweep bank` | bank_idle, bank_lend | Does bank lending reduce defaults? |
| NBFI lending effect | `bilancio sweep balanced --enable-lender` | passive, lender | Does non-bank lending reduce defaults? |

### Dealer Effect (`sweep balanced`)

The standard 2-arm comparison:
- **Passive**: Big entities (VBT + dealer) hold securities but don't trade
- **Active**: Dealer provides a secondary market; VBT provides reference pricing

```bash
bilancio sweep balanced --cloud \
  --n-agents 100 --kappas "0.3,0.5,1,2" \
  --concentrations "1" --mus "0"
```

### Bank Lending Effect (`sweep bank`)

Compares bank idle (deposits only) vs bank lending:
- **Bank idle**: Banks hold deposits, provide settlement infrastructure
- **Bank lend**: Banks actively lend to liquidity-constrained agents

```bash
bilancio sweep bank --cloud \
  --n-agents 100 --kappas "0.3,0.5,1,2"
```

### NBFI Lending Effect (`sweep balanced --enable-lender`)

Compares pure settlement vs non-bank lending:
- **Passive**: No intermediary activity
- **Lender**: Non-bank financial intermediary provides short-term loans

```bash
bilancio sweep balanced --cloud --enable-lender \
  --n-agents 100 --kappas "0.3,0.5,1,2"
```

## Deprecated: Mixed-Arm Modes

The following modes in `sweep balanced` are deprecated and will be
removed in a future release:

| Flag | Replacement |
|------|------------|
| `--enable-bank-passive` | `bilancio sweep bank` |
| `--enable-bank-dealer` | `bilancio sweep bank` |
| `--enable-bank-dealer-nbfi` | `bilancio sweep bank` |

These flags added banking arms to the balanced comparison, creating
complex multi-arm experiments that were hard to interpret. The
dedicated `sweep bank` command provides a cleaner experimental design.

Using any of these flags will emit a `DeprecationWarning`.
