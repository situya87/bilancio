# CLI Reference

Bilancio provides a command-line interface for running simulations, analyzing results, and managing cloud experiments.

## Quick Reference

| Command | Description |
|---------|-------------|
| `bilancio run` | Execute a scenario YAML file |
| `bilancio validate` | Validate scenario without running |
| `bilancio new` | Interactive scenario creation wizard |
| `bilancio analyze` | Compute metrics from simulation output |
| `bilancio sweep ring` | Run parameter sweeps across ring scenarios |
| `bilancio sweep balanced` | Compare passive vs active (dealer-enabled) scenarios |
| `bilancio sweep nbfi` | Run clean NBFI idle-vs-lend experiments |
| `bilancio sweep bank` | Run clean bank idle-vs-lend experiments |
| `bilancio sweep comparison` | Run dealer comparison experiments |
| `bilancio sweep strategy-outcomes` | Analyze trading strategy outcomes |
| `bilancio sweep dealer-usage` | Analyze dealer usage patterns |
| `bilancio jobs ls` | List simulation jobs |
| `bilancio jobs get` | Get job details |
| `bilancio jobs runs` | List runs for a job |
| `bilancio jobs metrics` | Show aggregate metrics |
| `bilancio jobs visualize` | Generate interactive visualization |
| `bilancio volume ls` | List experiments in Modal Volume |
| `bilancio volume cleanup` | Clean up old experiments |
| `bilancio volume rm` | Remove a specific experiment |

---

## bilancio run

Execute a scenario YAML file.

```bash
uv run bilancio run <scenario.yaml> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `SCENARIO_FILE` | Path to scenario YAML file (required) |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | `step\|until-stable` | `until-stable` | Simulation run mode |
| `--max-days` | integer | `90` | Maximum days to simulate |
| `--quiet-days` | integer | `2` | Required consecutive quiet days for stable state |
| `--show` | `summary\|detailed\|table` | `detailed` | Event display mode |
| `--agents` | string | none | Comma-separated list of agent IDs to show balances for |
| `--check-invariants` | `setup\|daily\|none` | `setup` | When to check system invariants |
| `--default-handling` | `fail-fast\|expel-agent` | none | Override scenario's default handling mode |
| `--export-balances` | path | none | Path to export balances CSV |
| `--export-events` | path | none | Path to export events JSONL |
| `--html` | path | none | Path to export colored output as HTML |
| `--t-account/--no-t-account` | flag | `--no-t-account` | Use detailed T-account layout for balances |

### Examples

```bash
# Run a scenario until stable
uv run bilancio run examples/scenarios/simple_bank.yaml

# Run with step-by-step mode
uv run bilancio run examples/scenarios/simple_bank.yaml --mode step

# Run with limited days and generate HTML report
uv run bilancio run examples/scenarios/dealer_ring.yaml --max-days 10 --html out/report.html

# Run and export data for analysis
uv run bilancio run examples/scenarios/ring.yaml \
  --export-events out/events.jsonl \
  --export-balances out/balances.csv

# Show only specific agents
uv run bilancio run examples/scenarios/ring.yaml --agents CB,B1,H1

# Override default handling to expel defaulting agents
uv run bilancio run examples/scenarios/ring.yaml --default-handling expel-agent
```

---

## bilancio validate

Validate a scenario YAML file without running the simulation.

```bash
uv run bilancio validate <scenario.yaml>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `SCENARIO_FILE` | Path to scenario YAML file (required) |

### What It Checks

1. Configuration syntax is valid YAML
2. All required fields are present
3. Agent definitions are valid
4. Initial actions can be applied
5. System invariants pass after setup

### Examples

```bash
# Validate a scenario file
uv run bilancio validate examples/scenarios/simple_bank.yaml

# Validate before running
uv run bilancio validate my_scenario.yaml && uv run bilancio run my_scenario.yaml
```

### Output

```
Validating examples/scenarios/simple_bank.yaml...
✓ Configuration syntax is valid
  Name: Simple Bank Example
  Version: 1.0
  Agents: 3
  Initial actions: 5
Checking if configuration can be applied...
✓ Configuration can be applied successfully
✓ System invariants pass

Configuration is valid!

Agents defined:
  • CB (central_bank): Central Bank
  • B1 (bank): Commercial Bank
  • H1 (household): Household
```

---

## bilancio new

Create a new scenario configuration interactively.

```bash
uv run bilancio new [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--from` | string | none | Base template to use |
| `-o, --output` | path | **required** | Output YAML file path |

### Examples

```bash
# Create a new scenario interactively
uv run bilancio new -o my_scenario.yaml

# Create from a template
uv run bilancio new --from ring -o my_ring.yaml
```

---

## bilancio analyze

Compute day-level metrics and optional intraday diagnostics from simulation output.

```bash
uv run bilancio analyze [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--events` | path | **required** | Path to events JSONL exported by a run |
| `--balances` | path | none | Path to balances CSV (optional, improves G_t/M_t) |
| `--days` | string | all | Days to analyze, e.g., "1,2-4" |
| `--out-csv` | path | auto | Output CSV for day-level metrics |
| `--out-json` | path | auto | Output JSON for day-level metrics |
| `--intraday-csv` | path | none | Optional CSV for intraday P_prefix steps |
| `--html` | path | none | Optional HTML analytics report |

### Examples

```bash
# Basic analysis with events only
uv run bilancio analyze --events out/events.jsonl

# Full analysis with balances and HTML report
uv run bilancio analyze \
  --events out/events.jsonl \
  --balances out/balances.csv \
  --html out/report.html

# Analyze specific days
uv run bilancio analyze --events out/events.jsonl --days "1-5,10"

# Export all outputs with custom paths
uv run bilancio analyze \
  --events out/events.jsonl \
  --balances out/balances.csv \
  --out-csv out/metrics.csv \
  --out-json out/metrics.json \
  --intraday-csv out/intraday.csv \
  --html out/analytics.html
```

### Computed Metrics

| Metric | Description |
|--------|-------------|
| `S_t` | Total dues maturing on day t |
| `Mbar_t` | Minimum net liquidity |
| `M_t` | Start-of-day money |
| `G_t` | Liquidity gap |
| `alpha_t` | Netting potential |
| `phi_t` | On-time settlement ratio |
| `delta_t` | Default rate (1 - phi_t) |

---

## bilancio sweep

Run parameter sweep experiments. The `sweep` command has multiple subcommands.

### bilancio sweep ring

Run parameter sweeps across ring topology scenarios.

```bash
uv run bilancio sweep ring [OPTIONS]
```

#### Options

**Output & Execution:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | none | Path to sweep config YAML |
| `--out-dir` | path | auto | Base output directory |
| `--cloud` | flag | false | Run simulations on Modal cloud |
| `--job-id` | string | auto | Job ID (auto-generated if not provided) |

**Grid Sweep:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--grid/--no-grid` | flag | `--grid` | Run coarse grid sweep |
| `--kappas` | string | `"0.25,0.5,1,2,4"` | Comma list for grid kappa values |
| `--concentrations` | string | `"0.2,0.5,1,2,5"` | Comma list for grid Dirichlet concentrations |
| `--mus` | string | `"0,0.25,0.5,0.75,1"` | Comma list for grid mu values |
| `--monotonicities` | string | `"0"` | Comma list for grid monotonicity values |

**Latin Hypercube Sampling:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--lhs` | integer | `0` | Number of Latin Hypercube samples |
| `--kappa-min` | float | `0.2` | LHS min kappa |
| `--kappa-max` | float | `5.0` | LHS max kappa |
| `--c-min` | float | `0.2` | LHS min concentration |
| `--c-max` | float | `5.0` | LHS max concentration |
| `--mu-min` | float | `0.0` | LHS min mu |
| `--mu-max` | float | `1.0` | LHS max mu |
| `--monotonicity-min` | float | `0.0` | LHS min monotonicity |
| `--monotonicity-max` | float | `0.0` | LHS max monotonicity |

**Frontier Search:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--frontier/--no-frontier` | flag | `--no-frontier` | Run frontier search |
| `--frontier-low` | float | `0.1` | Frontier lower bound for kappa |
| `--frontier-high` | float | `4.0` | Initial frontier upper bound for kappa |
| `--frontier-tolerance` | float | `0.02` | Frontier tolerance on delta_total |
| `--frontier-iterations` | integer | `6` | Max bisection iterations per cell |

**Scenario Generation:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--n-agents` | integer | `5` | Ring size (number of agents) |
| `--maturity-days` | integer | `3` | Due day horizon for generator |
| `--q-total` | float | `500.0` | Total dues S1 for generation |
| `--liquidity-mode` | `single_at\|uniform` | `single_at` | Liquidity allocation mode |
| `--liquidity-agent` | string | `H1` | Target for single_at liquidity allocation |
| `--base-seed` | integer | `42` | Base PRNG seed |
| `--name-prefix` | string | `Kalecki Ring Sweep` | Scenario name prefix |
| `--default-handling` | `fail-fast\|expel-agent` | `fail-fast` | Default handling mode for runs |

#### Examples

```bash
# Basic grid sweep
uv run bilancio sweep ring --out-dir out/experiments/my_sweep

# Cloud execution with custom parameters
uv run bilancio sweep ring --cloud \
  --out-dir out/experiments/cloud_sweep \
  --n-agents 50 \
  --kappas "0.3,0.5,1" \
  --concentrations "0.5,1,2"

# Latin Hypercube sampling
uv run bilancio sweep ring --no-grid --lhs 100 \
  --kappa-min 0.1 --kappa-max 3.0 \
  --out-dir out/experiments/lhs_sweep

# Frontier search
uv run bilancio sweep ring --no-grid --frontier \
  --frontier-low 0.1 --frontier-high 5.0 \
  --out-dir out/experiments/frontier_sweep

# With custom job ID
uv run bilancio sweep ring --cloud --job-id my-experiment-001 \
  --out-dir out/experiments/my-experiment-001
```

---

### bilancio sweep balanced

Compare passive holders (C) against active dealers (D) with identical starting balance sheets.
This is the legacy dealer comparison surface. Use `bilancio sweep nbfi` for clean
NBFI lending experiments and `bilancio sweep bank` for clean bank lending
experiments; balanced optional NBFI and banking arms are deprecated
compatibility options.

```bash
uv run bilancio sweep balanced [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--out-dir` | path | **required** | Output directory for results |
| `--n-agents` | integer | `100` | Number of agents in ring |
| `--maturity-days` | integer | `10` | Maturity horizon |
| `--q-total` | decimal | `10000` | Total debt |
| `--base-seed` | integer | `42` | Base random seed |
| `--kappas` | string | `"0.25,0.5,1,2,4"` | Comma-separated kappa values |
| `--concentrations` | string | `"0.2,0.5,1,2,5"` | Comma-separated concentration values |
| `--mus` | string | `"0,0.25,0.5,0.75,1"` | Comma-separated mu values |
| `--face-value` | decimal | `20` | Face value S (cashflow at maturity) |
| `--outside-mid-ratios` | string | `"1.0,0.9,0.8,0.75,0.5"` | Comma-separated M/S ratios to sweep |
| `--big-entity-share` | decimal | `0.25` | Fraction of debt held by big entities |
| `--default-handling` | `fail-fast\|expel-agent` | `fail-fast` | Default handling mode |
| `--detailed-logging/--no-detailed-logging` | flag | `--detailed-logging` | Enable detailed CSV logging |
| `--cloud` | flag | false | Run simulations on Modal cloud |
| `--job-id` | string | auto | Job ID (auto-generated if not provided) |
| `--quiet/--verbose` | flag | `--quiet` | Suppress verbose console output |

#### Output Structure

```
out-dir/
  passive/           # All passive holder runs
  active/            # All active dealer runs
  aggregate/
    comparison.csv   # C vs D metrics
    summary.json     # Aggregate statistics plus experiment_design metadata
```

#### Examples

```bash
# Basic balanced comparison
uv run bilancio sweep balanced --out-dir out/experiments/balanced_test

# Cloud execution with specific parameters
uv run bilancio sweep balanced --cloud \
  --out-dir out/experiments/dealer_study \
  --n-agents 100 \
  --kappas "0.3,0.5,1" \
  --concentrations "0.5,1" \
  --mus "0,0.5,1"

# With custom job ID for tracking
uv run bilancio sweep balanced --cloud \
  --job-id dealer-impact-study \
  --out-dir out/experiments/dealer-impact-study \
  --n-agents 50 \
  --outside-mid-ratios "1.0,0.8,0.6"
```

---

### bilancio sweep nbfi

Run a clean two-arm NBFI lending experiment. Both arms share the same seed,
topology, grid cell, and passive VBT/dealer infrastructure. The baseline arm is
`nbfi_idle`; the treatment arm is `nbfi_lend`; the reported effect is
`lending_effect = delta_idle - delta_lend`.

```bash
uv run bilancio sweep nbfi [OPTIONS]
```

#### Key Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--out-dir` | path | **required** | Output directory for results |
| `--n-agents` | integer | `100` | Number of firms in ring |
| `--n-replicates` | integer | `1` | Replicate seeds per parameter cell |
| `--kappas` | string | `"0.3,0.5,1.0,2.0"` | Comma-separated liquidity ratios |
| `--outside-mid-ratios` | string | `"0.90"` | Comma-separated outside money ratios |
| `--nbfi-share` | decimal | `0.10` | NBFI capital share |
| `--cloud` | flag | false | Run simulations on Modal cloud |

#### Output Structure

```
out-dir/
  nbfi_idle/
  nbfi_lend/
  aggregate/
    comparison.csv
    summary.json     # Includes experiment_design baseline/treatment metadata
```

---

### bilancio sweep bank

Run a clean two-arm bank lending experiment. Both arms share the same seed,
topology, grid cell, and deposit means-of-payment infrastructure. The baseline
arm is `bank_idle`; the treatment arm is `bank_lend`; the reported effect is
`bank_lending_effect = delta_idle - delta_lend`.

```bash
uv run bilancio sweep bank [OPTIONS]
```

#### Key Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--out-dir` | path | **required** | Output directory for results |
| `--n-agents` | integer | `100` | Number of firms in ring |
| `--n-replicates` | integer | `1` | Replicate seeds per parameter cell |
| `--kappas` | string | `"0.3,0.5,1.0,2.0"` | Comma-separated liquidity ratios |
| `--n-banks` | integer | `5` | Number of banks |
| `--reserve-ratio` | decimal | `0.50` | Initial reserves / total deposits |
| `--credit-risk-loading` | decimal | `0.5` | Bank sensitivity to borrower risk |
| `--cloud` | flag | false | Run simulations on Modal cloud |

#### Output Structure

```
out-dir/
  bank_idle/
  bank_lend/
  aggregate/
    comparison.csv
    summary.json     # Includes experiment_design baseline/treatment metadata
```

---

### bilancio sweep comparison

Run dealer comparison experiments (control vs treatment).

```bash
uv run bilancio sweep comparison [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--out-dir` | path | **required** | Output directory for results |
| `--n-agents` | integer | `100` | Ring size |
| `--maturity-days` | integer | `10` | Maturity horizon in days |
| `--q-total` | float | `10000.0` | Total debt amount |
| `--kappas` | string | `"0.25,0.5,1,2,4"` | Comma list for kappa values |
| `--concentrations` | string | `"0.2,0.5,1,2,5"` | Comma list for Dirichlet concentrations |
| `--mus` | string | `"0,0.25,0.5,0.75,1"` | Comma list for mu values |
| `--monotonicities` | string | `"0"` | Comma list for monotonicity values |
| `--base-seed` | integer | `42` | Base PRNG seed |
| `--default-handling` | `fail-fast\|expel-agent` | `fail-fast` | Default handling mode |
| `--dealer-ticket-size` | float | `1.0` | Ticket size for dealer |
| `--dealer-share` | float | `0.25` | Dealer capital as fraction of system cash |
| `--vbt-share` | float | `0.50` | VBT capital as fraction of system cash |
| `--liquidity-mode` | `single_at\|uniform` | `uniform` | Liquidity allocation mode |
| `--liquidity-agent` | string | none | Target agent for single_at mode |
| `--name-prefix` | string | `Dealer Comparison` | Scenario name prefix |

#### Examples

```bash
# Run dealer comparison sweep
uv run bilancio sweep comparison \
  --out-dir out/experiments/dealer_comparison \
  --n-agents 50 \
  --kappas "0.3,0.5,1"
```

---

### bilancio sweep strategy-outcomes

Analyze trading strategy outcomes across experiment runs.

```bash
uv run bilancio sweep strategy-outcomes --experiment <path> [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--experiment` | path | **required** | Path to experiment directory |
| `-v, --verbose` | flag | false | Enable verbose logging |

#### Output

- `aggregate/strategy_outcomes_by_run.csv` - Per-run strategy metrics
- `aggregate/strategy_outcomes_overall.csv` - Aggregated strategy metrics

#### Examples

```bash
uv run bilancio sweep strategy-outcomes --experiment out/experiments/my_sweep
```

---

### bilancio sweep dealer-usage

Analyze dealer usage patterns across experiment runs.

```bash
uv run bilancio sweep dealer-usage --experiment <path> [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--experiment` | path | **required** | Path to experiment directory |
| `-v, --verbose` | flag | false | Enable verbose logging |

#### Output

- `aggregate/dealer_usage_by_run.csv` - Dealer usage summary per run

#### Examples

```bash
uv run bilancio sweep dealer-usage --experiment out/experiments/my_sweep -v
```

---

## bilancio jobs

Query and manage simulation jobs. Requires Supabase configuration for cloud queries.

### bilancio jobs ls

List simulation jobs.

```bash
uv run bilancio jobs ls [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cloud` | flag | false | Query from Supabase cloud storage |
| `--local` | path | none | Local directory containing job manifests |
| `--status` | `pending\|running\|completed\|failed` | none | Filter by job status |
| `--limit` | integer | `20` | Maximum number of jobs to show |

#### Examples

```bash
# List jobs from Supabase
uv run bilancio jobs ls --cloud

# List local jobs
uv run bilancio jobs ls --local out/experiments

# Filter by status
uv run bilancio jobs ls --cloud --status completed --limit 10
```

#### Output

```
JOB ID                               STATUS     CREATED          DURATION   RUNS
--------------------------------------------------------------------------------
castle-river-mountain-forest         completed  2025-01-12 10:30 15.2m      50
bright-ocean-swift-tiger             running    2025-01-12 11:00 running    25
--------------------------------------------------------------------------------
Total: 2 jobs
```

---

### bilancio jobs get

Get detailed information about a specific job.

```bash
uv run bilancio jobs get <job_id> [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `JOB_ID` | Job identifier (required) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cloud` | flag | false | Query from Supabase cloud storage |
| `--local` | path | none | Local directory containing job manifests |

#### Examples

```bash
# Get job details from Supabase
uv run bilancio jobs get castle-river-mountain-forest --cloud

# Get job details from local filesystem
uv run bilancio jobs get castle-river-mountain-forest --local out/experiments
```

---

### bilancio jobs runs

List runs for a specific job.

```bash
uv run bilancio jobs runs <job_id> [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `JOB_ID` | Job identifier (required) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cloud` | flag | false | Query from Supabase cloud storage |
| `--status` | `pending\|running\|completed\|failed` | none | Filter by run status |

#### Examples

```bash
# List runs for a job
uv run bilancio jobs runs castle-river-mountain-forest --cloud

# Filter by status
uv run bilancio jobs runs castle-river-mountain-forest --cloud --status completed
```

---

### bilancio jobs metrics

Show aggregate metrics for a job.

```bash
uv run bilancio jobs metrics <job_id> [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `JOB_ID` | Job identifier (required) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cloud` | flag | false | Query from Supabase cloud storage |

#### Examples

```bash
# Show metrics from Supabase
uv run bilancio jobs metrics castle-river-mountain-forest --cloud
```

#### Output

```
Job: castle-river-mountain-forest
Completed runs: 50 / 50

Delta (default rate):
  Mean:   0.1234
  Min:    0.0000
  Max:    0.4567
Phi (clearing rate):
  Mean:   0.8766
  Min:    0.5433
  Max:    1.0000
```

---

### bilancio jobs visualize

Generate interactive visualization comparing passive vs active runs.

```bash
uv run bilancio jobs visualize <job_id> [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `JOB_ID` | Job identifier (required) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | path | `temp/{job_id}_comparison.html` | Output path for HTML file |
| `-t, --title` | string | auto | Custom title for the visualization |
| `--open-browser` | flag | false | Open the generated HTML in the default browser |

#### Visualization Contents

- Summary statistics
- Passive vs active scatter plot
- Default rate by liquidity (kappa)
- Faceted views by concentration
- Trading effect distributions
- Heatmaps by maturity timing (mu)
- Parallel coordinates overview
- 3D surface plot

#### Examples

```bash
# Generate visualization
uv run bilancio jobs visualize castle-river-mountain-forest

# Custom output path and open in browser
uv run bilancio jobs visualize castle-river-mountain-forest \
  -o results/comparison.html \
  --open-browser

# With custom title
uv run bilancio jobs visualize castle-river-mountain-forest \
  -t "Dealer Impact Analysis"
```

---

## bilancio volume

Manage Modal Volume storage for cloud experiments.

### bilancio volume ls

List experiments in Modal Volume.

```bash
uv run bilancio volume ls [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--volume` | string | `bilancio-results` | Volume name |

#### Examples

```bash
# List all experiments
uv run bilancio volume ls

# Use different volume
uv run bilancio volume ls --volume my-volume
```

---

### bilancio volume cleanup

Clean up old experiments from Modal Volume.

```bash
uv run bilancio volume cleanup [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--volume` | string | `bilancio-results` | Volume name |
| `--older-than` | integer | none | Delete experiments older than N days |
| `--pattern` | string | none | Delete experiments matching pattern (e.g., `test_*`) |
| `--dry-run` | flag | false | Show what would be deleted without deleting |
| `--yes, -y` | flag | false | Skip confirmation prompt |

#### Examples

```bash
# Preview cleanup of old experiments
uv run bilancio volume cleanup --older-than 30 --dry-run

# Delete experiments older than 30 days
uv run bilancio volume cleanup --older-than 30 -y

# Delete by pattern
uv run bilancio volume cleanup --pattern "test_*" --dry-run
uv run bilancio volume cleanup --pattern "test_*" -y

# Combine filters
uv run bilancio volume cleanup --older-than 7 --pattern "debug_*" -y
```

---

### bilancio volume rm

Remove a specific experiment from Modal Volume.

```bash
uv run bilancio volume rm <experiment_id> [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `EXPERIMENT_ID` | Experiment identifier (required) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--volume` | string | `bilancio-results` | Volume name |
| `--yes, -y` | flag | false | Skip confirmation prompt |

#### Examples

```bash
# Remove with confirmation
uv run bilancio volume rm castle-river-mountain-forest

# Remove without confirmation
uv run bilancio volume rm castle-river-mountain-forest -y
```

---

## Environment Variables

For cloud features (Supabase storage), set these environment variables:

```bash
# Supabase configuration
export BILANCIO_SUPABASE_URL=https://xxxx.supabase.co
export BILANCIO_SUPABASE_ANON_KEY=eyJ...

# Or load from .env file
export $(grep -v '^#' .env | xargs)
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (file not found, validation failed, etc.) |

## See Also

- [Getting Started Guide](./getting_started.md)
- [Scenario Configuration](./scenarios.md)
- [Analysis and Metrics](./analysis.md)
