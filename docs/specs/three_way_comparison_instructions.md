# Three-Way Liquidity Provision Comparison

## Purpose

Scientific comparison of three liquidity provision mechanisms in the Kalecki ring:
1. **Dealer** — Secondary market intermediation (buy/sell existing claims)
2. **NBFI Lender** — Non-bank credit provision (new loans to stressed agents)
3. **Bank** — Deposit-based lending with CB backstop (money creation)

Each mechanism is tested in isolation against a passive baseline, using identical ring topologies and parameter grids.

## Sweep Arms

| Sweep | Arm A (baseline) | Arm B (treatment) | Active entities | Spec document |
|-------|-------------------|--------------------|-----------------|---------------|
| Dealer | Passive (no dealer) | Active (dealer + VBT) | 100 firms + dealer + VBT | `docs/specs/dealer_passive_vs_active.md` |
| NBFI | Passive (no lender) | Passive + NBFI lender | 100 firms + VBT (no dealer) | `docs/spec/nbfi_decision_profiles.md` |
| Bank | Bank idle | Bank lending | 100 firms + bank + CB (no dealer/VBT) | `docs/analysis/bank_idle_vs_bank_lend_specification.md` |

**Key isolation principle**: Each sweep activates exactly one intermediation mechanism against its own control. No sweep combines mechanisms.

## Unified Parameter Grid

All three sweeps use identical grid parameters:

| Parameter | Symbol | Values | Count |
|-----------|--------|--------|-------|
| Liquidity ratio | κ | 0.25, 0.3, 0.5, 1.0, 1.5, 2.0, 4.0 | 7 |
| Concentration | c | 0.5, 1, 2 | 3 |
| Maturity skew | μ | 0, 0.5 | 2 |
| Outside money ratio | ρ | 0.90 | 1 |
| Replicates | — | 3 (base_seed=42) | 3 |

**Grid size**: 7 × 3 × 2 × 1 × 3 = **126 parameter combos** × 2 arms = **252 runs per sweep**

**Total across all 3 sweeps**: 756 runs

### Fixed Parameters (all sweeps)

| Parameter | Value |
|-----------|-------|
| `n_agents` | 100 |
| `maturity_days` | 10 |
| `face_value` | 20 |
| `Q_total` | 10,000 |
| `default_handling` | expel-agent |
| `rollover` | True |

## CLI Commands

Run each in a separate terminal. All three can run in parallel.

### Terminal 1: Dealer Comparison

```bash
uv run bilancio sweep balanced --cloud \
  --n-agents 100 --maturity-days 10 \
  --kappas "0.25,0.3,0.5,1.0,1.5,2.0,4.0" \
  --concentrations "0.5,1,2" --mus "0,0.5" \
  --outside-mid-ratios "0.90" --base-seed 42 --n-replicates 3 \
  --out-dir out/three_way/dealer
```

### Terminal 2: NBFI Comparison

```bash
uv run bilancio sweep nbfi --cloud \
  --n-agents 100 --maturity-days 10 \
  --kappas "0.25,0.3,0.5,1.0,1.5,2.0,4.0" \
  --concentrations "0.5,1,2" --mus "0,0.5" \
  --outside-mid-ratios "0.90" --base-seed 42 --n-replicates 3 \
  --out-dir out/three_way/nbfi
```

### Terminal 3: Bank Comparison

```bash
uv run bilancio sweep bank --cloud \
  --n-agents 100 --maturity-days 10 \
  --kappas "0.25,0.3,0.5,1.0,1.5,2.0,4.0" \
  --concentrations "0.5,1,2" --mus "0,0.5" \
  --outside-mid-ratios "0.90" --base-seed 42 --n-replicates 3 \
  --out-dir out/three_way/bank
```

**Note**: All sweeps now use spec-aligned defaults. No additional flags are needed for:
- ρ = 0.90 (dealer sweep default updated from 1.0)
- Credit-sensitive bank pricing (`credit_risk_loading=0.5`, `max_borrower_risk=0.4` — bank sweep defaults updated)
- VBT/dealer shares in NBFI (0.20/0.05 — NBFI config defaults updated)

## Cost and Duration Estimates

| Metric | Per sweep | Total (3 sweeps) |
|--------|-----------|-------------------|
| Runs | 252 | 756 |
| Estimated duration (cloud) | ~63 minutes | ~63 min (parallel) |
| Estimated cost | ~$0.08 | ~$0.24 |

Cloud sweeps can run fully in parallel — total wall time is ~63 minutes, not 189.

## Output Structure

Each sweep produces:

```
out/three_way/<sweep>/
├── <job_id>/
│   ├── job_manifest.json
│   ├── passive/ or bank_idle/     # Baseline runs
│   ├── active/ or nbfi_lend/ or bank_lend/  # Treatment runs
│   └── aggregate/
│       ├── comparison.csv         # Side-by-side metrics
│       ├── results.csv            # All run metrics
│       └── dashboard.html         # Visual dashboard
```

## Analytical Framework

### Primary Metrics

All three sweeps report the same core metrics:

| Metric | Symbol | Definition | Direction |
|--------|--------|------------|-----------|
| Default rate | δ | Fraction of total debt defaulted | Lower is better |
| Clearing rate | φ | Fraction of total debt settled | Higher is better |
| System loss | — | total_loss + intermediary_loss | Lower is better |
| Treatment effect | Δ | δ_baseline − δ_treatment | Positive = mechanism helps |

### Loss Decomposition

| Loss channel | Dealer sweep | NBFI sweep | Bank sweep |
|-------------|-------------|------------|------------|
| Payable default loss | ✓ | ✓ | ✓ |
| Deposit loss | — | — | ✓ |
| Intermediary credit loss | Dealer inventory loss | NBFI loan loss | Bank credit loss |
| CB backstop loss | — | — | ✓ |

### Mechanism Comparison

When comparing across sweeps, the key question is:

> **At each κ level, which mechanism reduces δ the most, and at what cost?**

Comparison dimensions:

1. **Effectiveness** (Δδ): Which mechanism reduces defaults most?
   - Dealer: redistributes existing claims (no new money)
   - NBFI: injects new credit (from own capital)
   - Bank: creates deposits (money creation, CB backstopped)

2. **Cost** (intermediary_loss): What does the intermediary absorb?
   - Dealer: inventory losses on defaulted tickets
   - NBFI: loan defaults (principal loss)
   - Bank: loan defaults + CB backstop costs

3. **Regime sensitivity**: How does each mechanism scale with κ?
   - Low κ (< 0.5): severe stress — which mechanism works under extreme scarcity?
   - Medium κ (0.5–1.5): moderate stress — where is each mechanism most effective?
   - High κ (> 2.0): low stress — do mechanisms matter when liquidity is abundant?

4. **Concentration sensitivity**: How does each mechanism handle inequality (c)?
   - c = 0.5: high inequality — few agents owe most
   - c = 2: low inequality — more even distribution

5. **Timing sensitivity**: Does μ matter?
   - μ = 0: front-loaded stress (immediate pressure)
   - μ = 0.5: spread across horizon

## Cross-Sweep Analysis

After all three sweeps complete:

1. **Load comparison CSVs**:
   ```python
   import pandas as pd
   dealer = pd.read_csv("out/three_way/dealer/<job_id>/aggregate/comparison.csv")
   nbfi = pd.read_csv("out/three_way/nbfi/<job_id>/aggregate/comparison.csv")
   bank = pd.read_csv("out/three_way/bank/<job_id>/aggregate/comparison.csv")
   ```

2. **Compute treatment effects**:
   - Dealer: `delta_passive - delta_active`
   - NBFI: `delta_passive - delta_lender` (column names may vary)
   - Bank: `delta_idle - delta_lend`

3. **Plot δ vs κ** for all six arms (3 baselines + 3 treatments) on one chart
   - Baselines should roughly coincide (same ring, same parameters)
   - Treatment lines show each mechanism's marginal impact

4. **Plot treatment effect vs κ** for all three mechanisms
   - Shows where each mechanism is most / least effective

5. **Plot system_loss vs κ** to check whether reducing defaults also reduces total losses
   - A mechanism can reduce δ while increasing system_loss if it concentrates risk

## Reproducibility

All defaults are now spec-aligned. To reproduce:

1. Use the exact CLI commands above (no extra flags needed)
2. `--n-replicates 3` generates seeds 42, 43, 44 (base_seed + 0..n-1) for statistical robustness
3. Each parameter combo produces 3 replicates — report mean ± std

To verify defaults propagated correctly, inspect any output scenario YAML:
```bash
cat out/three_way/dealer/<job_id>/active/run_*/scenario.yaml | head -50
```

---

*Generated for the three-way liquidity provision comparison experiment.*
