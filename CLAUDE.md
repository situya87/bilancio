- Use implementation subagents (in parallel) to execute code changes whenever possible, but always review what the subagents wrote, find possible issues, and fix them if necessary.

## New Agent Type Design Checklist

When introducing a new agent type, you MUST explicitly define all four aspects before writing implementation code. Do not proceed until each is documented in the plan:

1. **Instruments** — What instruments does it hold (assets) and issue (liabilities)? What are the contract terms (rate, maturity, etc.)? Update `InstrumentKind`, `policy.py`, and create the instrument dataclass.
2. **Means of Payment** — What does it use to settle obligations? (Cash, reserves, deposits?) Define its `mop_rank` in `policy.py`.
3. **Decision-Making Model** — How does it decide what actions to take each day? This must be a behavioral model with tunable parameters (e.g., risk aversion, planning horizon, aggressiveness), not a hard-coded formula. Create a profile dataclass (like `TraderProfile`) so the strategy can be configured per experiment.
4. **Information Model** — What can the agent observe about the system? How does it collect and update its beliefs? Does it have its own risk assessor? Is there observability friction (can it see all defaults, or only some)? Does it learn from its own history (Bayesian updating)?
5. **Capitalization** — How does it get its initial resources? Is it endowed at setup, or does it accumulate over time? What fraction of system resources does it receive (`lender_share`, `dealer_share`, etc.)?
6. **Timing / Phase** — When does it act in the daily simulation cycle? Which subphase? Before or after settlement? Does ordering matter relative to other agents?
7. **Failure Mode** — What happens when this agent defaults or can't meet obligations? Does it cascade? Is it systemically important? Can it be bailed out?
8. **Interactions** — Which other agent types does it transact with? Are there bilateral constraints (e.g., "only lends to firms/households, not to banks")?
9. **State Synchronization** — If the new agent modifies shared state (e.g., agent cash) in its phase, identify ALL other phases that read/write the same state. Ensure synchronization points (like `_sync_trader_cash_from_system`) capture the latest values BEFORE computing deltas. Test the cross-phase interaction explicitly.

Reference: The dealer/trader framework (`bilancio/decision/`) is the gold standard — it has `TraderProfile`, `VBTProfile`, and the `RiskAssessor` with configurable observability.
- Always use `uv run` instead of `python` to run Python commands in this project
- Remove all temporary test_ files when they're no longer needed
- Store temporary test files in a gitignored temp/ folder instead of the project root

## Running Tests
- Run all tests: `uv run pytest tests/ -v`
- Run with coverage: `uv run pytest tests/ --cov=bilancio --cov-report=term-missing`
- Run specific test file: `uv run pytest tests/unit/test_balances.py -v`
- Tests use pytest, installed via: `uv add pytest pytest-cov --dev`

## Jupyter Notebooks
- **ALWAYS TEST NOTEBOOKS BEFORE PRESENTING**: Run code snippets iteratively to catch errors
- Test each cell's code with `uv run python -c "..."` before including in notebook
- Only present notebooks after verifying no errors occur
- To open notebooks in browser: `uv run jupyter notebook <path>` (runs in background)
- Common pitfalls in bilancio:
  - Must use actual agent classes (Bank, Household, Firm) not Agent(kind="bank") - policy checks isinstance()
  - Check function signatures - parameter order matters
  - Verify all imports work before creating notebook

### Balance Sheet Display
- **Use existing display functions** - Always use `bilancio.analysis.visualization.display_agent_balance_table()` and `display_multiple_agent_balances()` instead of creating custom display functions
- **Prefer 'rich' format** - Use `format='rich'` for pretty formatted output (default). This gives nicely formatted tables with colors and borders
- **Use 'simple' format only when needed** - Use `format='simple'` only when balance sheets have many items that would be cramped in the rich format (simple format has more room)
- **Get balance data with agent_balance()** - Use `bilancio.analysis.balances.agent_balance()` to get structured balance sheet data for analysis

### Testing Notebooks - Critical Lessons
- **ALWAYS TEST AFTER ANY CHANGE** - Every time you touch/edit/modify a notebook, you MUST run the complete testing from scratch using `uv run jupyter nbconvert --execute <notebook.ipynb>`. NO EXCEPTIONS.
- **Always test notebooks by executing them directly** - Use `uv run jupyter nbconvert --execute <notebook.ipynb>` to run the actual notebook. Don't extract code to test in separate Python files.
- **Check the ENTIRE output of every cell** - Not just whether it executes without errors, but what each cell actually produces. A notebook can "run" without errors but still produce incorrect results.
- **Read actual outputs, not just success messages** - A notebook can execute "successfully" (no exceptions) but still produce wrong results. Must examine the actual output values.
- **When notebooks don't work as expected** - The issue might not be in the notebook itself but in the underlying code it's calling (check the actual library functions being used).
- **For complex debugging**:
  - First, check if the notebook executes at all
  - Then, examine output of each cell systematically
  - Trace through the logic to find where results diverge from expectations
  - Test individual functions separately only AFTER identifying where the problem occurs
- **When editing notebooks is problematic** - Sometimes it's better to recreate from scratch than to fix complex editing issues with notebook cells

### Creating Notebooks - Essential Rules
- **Ensure correct cell types** - Double-check that code cells are type "code" and markdown cells are type "markdown". Mixed up cell types cause confusing errors.
- **Add sufficient output for debugging** - Every cell should produce enough output to understand what's happening:
  - Print intermediate results and state changes
  - Show balance sheets after each operation
  - Log events and settlements as they occur
  - Display verification checks with actual vs expected values
  - Include descriptive messages explaining what each step does
- **Make notebooks self-documenting** - The output should tell a clear story of what's happening without needing to read the code
- When I tell you to implement a plan from @docs/plans/, always make sure you start from main with clean git status - if not, stop and tell me. Then, create a new branch with the name of the plan and start work.

## UI/Rendering Work
- **ALWAYS TEST HTML OUTPUT**: When making any changes to rendering/UI/display code:
  1. Rebuild the HTML after each change: `uv run bilancio run examples/scenarios/simple_bank.yaml --max-days 3 --html temp/demo.html`
  2. Open it directly in the browser: `open temp/demo.html`
  3. Provide the user with the CLI command to run in their terminal for testing
- **VERIFY COMPLETENESS**: After generating HTML:
  1. Read the source YAML file to understand what should be displayed
  2. Read the generated HTML file to check all information is present
  3. Ensure ALL events are displayed (setup events, payable creation, phase events, etc.)
  4. Verify agent list is shown at the top
  5. Think carefully about what might be missing
- **Verify visual output**: Read the generated HTML file to ensure events, tables, and formatting display correctly
- **Test with real scenarios**: Use actual scenario files to test rendering changes, not just unit tests

## ReadTheDocs Documentation

Documentation is hosted at https://bilancio.readthedocs.io/

### Configuration Files
- `.readthedocs.yaml` - RTD build configuration (Python version, MkDocs settings)
- `mkdocs.yml` - MkDocs site configuration (navigation, theme)
- `docs/` - Documentation source files (Markdown)

### Documentation Structure
```
docs/
├── index.md          # Homepage
├── installation.md   # Installation guide
├── quickstart.md     # Getting started tutorial
├── concepts.md       # Core concepts (agents, instruments, settlement)
├── cli.md            # CLI command reference
├── contributing.md   # Contribution guidelines
└── changelog.md      # Version history
```

### Managing Documentation

**Rebuild docs after changes:**
1. Push changes to `main` branch
2. Go to https://app.readthedocs.org/projects/bilancio/builds/
3. Click on the latest build, then click "Rebuild"
4. Note: No webhook is configured, so rebuilds must be triggered manually

**Test locally before pushing:**
```bash
uv pip install mkdocs
uv run mkdocs serve
# Open http://127.0.0.1:8000 to preview
```

**Add new documentation page:**
1. Create new `.md` file in `docs/`
2. Add to `nav:` section in `mkdocs.yml`
3. Commit, push, and trigger rebuild

### RTD Dashboard
- **Project URL**: https://app.readthedocs.org/projects/bilancio/
- **Builds**: https://app.readthedocs.org/projects/bilancio/builds/
- **Settings**: https://app.readthedocs.org/projects/bilancio/settings/

## Claude Code Web Environment

When running in Claude Code web (claude.ai/code), the VM uses a TLS-inspecting proxy that requires special configuration for external services.

### Connectivity Status

Both Modal and Supabase are configured to work automatically:

- **Supabase**: The `bilancio.storage.supabase_client` module automatically configures httpx with the proxy CA certificate
- **Modal Python SDK**: The `bilancio.cloud.proxy_patch` module patches grpclib for proxy compatibility
- **Modal CLI**: Use the wrapper script `scripts/modal_wrapper.py` instead of `modal` directly

### Using Modal CLI in Claude Code Web

The standard `modal` CLI doesn't work through the proxy. Use the wrapper:

```bash
# Instead of: uv run modal app list
uv run python scripts/modal_wrapper.py app list

# Instead of: uv run modal volume ls bilancio-results
uv run python scripts/modal_wrapper.py volume ls bilancio-results

# Instead of: uv run modal deploy src/bilancio/cloud/modal_app.py
uv run python scripts/modal_wrapper.py deploy src/bilancio/cloud/modal_app.py
```

### Testing Connectivity

Verify services are accessible:

```bash
# Test Supabase
uv run bilancio jobs ls --cloud

# Test Modal (via wrapper)
uv run python scripts/modal_wrapper.py app list
```

### Troubleshooting

If you see SSL/TLS errors like `CERTIFICATE_VERIFY_FAILED`:
1. The proxy CA certificate should be at `/usr/local/share/ca-certificates/swp-ca-production.crt`
2. For Supabase: The fix is built into `bilancio.storage.supabase_client`
3. For Modal CLI: Always use `scripts/modal_wrapper.py`
4. For Modal Python SDK: Import `bilancio.cloud.proxy_patch` before `modal`

---

## Modal Cloud Execution
Cloud simulations run on Modal. Always use `uv run modal` to access the CLI (or the wrapper in Claude Code web).

### Authentication
To authenticate Modal with browser login: `uv run modal token new --profile <workspace-name>`

### Key Commands
- **View logs**: `uv run modal app logs bilancio-simulations` - streams logs from running/recent executions
- **List volume contents**: `uv run modal volume ls bilancio-results [path]` - path is optional subdirectory
  - Example: `uv run modal volume ls bilancio-results` - list experiment folders
  - Example: `uv run modal volume ls bilancio-results my_experiment/runs` - list runs in experiment
- **Download artifacts**: `uv run modal volume get bilancio-results <remote_path> <local_path> --force`
- **Deploy app**: `uv run modal deploy src/bilancio/cloud/modal_app.py` - redeploy after code changes

### Running Cloud Sweeps
- Ring sweep: `uv run bilancio sweep ring --cloud --out-dir out/experiments/my_sweep ...`
- Balanced comparison: `uv run bilancio sweep balanced --cloud --out-dir out/experiments/my_sweep ...`

### Important Notes
- Always redeploy after changing `modal_app.py`: the deployed function runs the version on Modal, not local code
- Artifacts are stored in Modal Volume `bilancio-results` under `<experiment_id>/runs/<run_id>/`
- Use `--cloud` flag with small parameters first to test (saves credits)

### Parallelism and Performance

Cloud sweeps run simulations in parallel using Modal's `.map()` function. Modal automatically scales containers to process inputs concurrently.

**Current parallelism**: ~5-6 concurrent simulations (Modal's default scaling for the account)

**Performance benchmarks** (100 agents, 10 maturity days):
- Single simulation: ~30-60 seconds
- 10 pairs (20 runs): ~4 minutes
- 25 pairs (50 runs): ~13 minutes
- 125 pairs (250 runs): ~45-60 minutes (estimated)

### Estimating Duration and Cost

**IMPORTANT**: Before running any cloud sweep, estimate and inform the user of the expected duration and cost.

**Duration formula**:
```
estimated_minutes = (num_pairs * 2) / 4
```
Where `num_pairs = len(kappas) × len(concentrations) × len(mus) × len(outside_mid_ratios) × len(seeds)`

Example: 5 kappas × 5 concentrations × 5 mus = 125 pairs = 250 runs ≈ 60 minutes

**Cost formula** (Modal pricing as of Jan 2025):
```
cost_per_run ≈ $0.0003 (CPU: $0.0000131/core/sec, ~30 sec/run)
total_cost ≈ num_runs × $0.0003
```

Example: 250 runs × $0.0003 = ~$0.08

**Before running a sweep, always tell the user**:
> This sweep has X pairs (Y runs). Estimated duration: ~Z minutes. Estimated cost: ~$W.

### Monitoring Running Jobs

While a sweep is running:
- Progress is displayed in the terminal: `Progress: 15/50 runs (30%) - ETA: 7.1m`
- View Modal logs: `uv run modal app logs bilancio-simulations`
- Check Modal dashboard: https://modal.com/apps/bilancio/main/deployed/bilancio-simulations

If a job fails or times out:
- Individual run timeout is 30 minutes (configurable in `modal_app.py`)
- Check Modal logs for error details
- Failed runs are marked in the job manifest

---

## Supabase Cloud Storage (Optional)

Jobs and runs can be persisted to Supabase for queryable, durable storage accessible across conversations.

### Configuration

Set environment variables (already in `.env`):
```bash
BILANCIO_SUPABASE_URL=https://xxxx.supabase.co
BILANCIO_SUPABASE_ANON_KEY=eyJ...
```

**Important:** To use Supabase from the CLI, you must load the environment variables first:
```bash
# Load env vars before running commands
export $(grep -v '^#' .env | xargs)

# Now Supabase commands will work
uv run bilancio jobs ls --cloud
```

### Supabase CLI

Use the Supabase CLI for database management instead of the web dashboard:
```bash
# Push migrations to remote database
supabase db push --linked

# List migrations
ls supabase/migrations/

# Create a new migration
# Create file: supabase/migrations/YYYYMMDD_description.sql
# Then push with: supabase db push --linked
```

Migrations are stored in `supabase/migrations/`. The project is linked to the bilancio Supabase project.
### Architecture

- **Cloud-only mode (recommended)**: Jobs stored only in Supabase, no local files
- **Hybrid mode**: Jobs saved to both local filesystem AND Supabase
- **Local-only mode**: Default if Supabase not configured

### Automatic Persistence During Sweeps

When running cloud sweeps (`--cloud` flag), jobs, runs, and metrics are automatically persisted to Supabase:
- **Jobs**: Created at sweep start, updated on completion
- **Runs**: Each simulation run (passive/active) is recorded with parameters
- **Metrics**: delta_total, phi_total, and other metrics are stored per run

No additional configuration needed - just ensure env vars are loaded.

### Using Cloud Storage in Code

```python
from bilancio.jobs import create_job_manager

# Cloud-only (recommended for VMs with limited storage)
manager = create_job_manager(cloud=True, local=False)

# Both local and cloud
manager = create_job_manager(jobs_dir=Path("./jobs"), cloud=True, local=True)

# Creates job (stored per configuration above)
job = manager.create_job(description="My sweep", config=config)
```

### CLI Commands for Querying Jobs

```bash
# List jobs from Supabase
bilancio jobs ls --cloud

# Get job details
bilancio jobs get castle-river-mountain --cloud

# List runs for a job
bilancio jobs runs castle-river-mountain --cloud

# Show aggregate metrics
bilancio jobs metrics castle-river-mountain --cloud
```

### Database Schema

Jobs are stored in Supabase PostgreSQL with these tables:
- `jobs` - Job metadata (job_id, status, config, timestamps)
- `runs` - Individual run records (run_id, job_id, parameters, status)
- `metrics` - Run metrics (delta_total, phi_total, raw_metrics JSONB)
- `job_events` - Lifecycle events (created, started, progress, completed)

### Supabase Dashboard

View data directly at: https://supabase.com/dashboard/project/vghchkriceqqitbpevtz

---

## Modal Volume Management

Manage artifacts stored in Modal Volume:

```bash
# List experiments
bilancio volume ls

# Delete experiments older than 30 days
bilancio volume cleanup --older-than 30 --dry-run
bilancio volume cleanup --older-than 30 -y

# Delete by pattern
bilancio volume cleanup --pattern "test_*" -y

# Delete specific experiment
bilancio volume rm castle-river-mountain -y
```

---

## Claude Code Web Workflow (Autonomous Simulation Jobs)

This project is designed to be run by Claude Code on the web (claude.ai/code), where Claude operates in an Anthropic-managed VM, clones this repo, and runs simulations based on natural language instructions.

### Workflow Overview
1. **User requests a simulation** via Claude Code web interface (natural language)
2. **Claude generates a Job ID** - a memorable 4-word passphrase (e.g., `castle-river-mountain-forest`)
3. **Claude configures and runs** the simulation on Modal cloud
4. **Results are persisted** to Modal Volume and/or external database
5. **User can query results** by Job ID in future conversations

> **⚠️ IMPORTANT FOR CLAUDE**: When running any simulation job, you MUST prominently display to the user:
>
> **Job ID:** `rumbling-appear-flail-squiggle`
> **Modal ID:** `fc-01KESDG0PR7329Q0M7D087XCW6`
>
> The Job ID is how the user references results. The Modal ID helps debug on modal.com if needed.
> Do NOT list individual run IDs (there may be dozens in a sweep).

### Job ID System
Every simulation job gets a unique, memorable identifier:
- Format: `word1-word2-word3-word4` (e.g., `bright-ocean-swift-tiger`)
- Auto-generated when running sweeps, or specify with `--job-id`
- Use this ID when discussing results with colleagues
- Job metadata is stored in `<out_dir>/<job_id>/job_manifest.json`

### Job Lifecycle
1. **Job Created** → `job_manifest.json` written with status=pending
2. **Job Started** → status=running, event logged
3. **Progress** → Each run completion is recorded
4. **Completed/Failed** → Final status, summary metrics saved

### Using Jobs
```bash
# Auto-generate job ID
uv run bilancio sweep balanced --cloud --out-dir out/experiments/my_sweep ...
# Output: Job ID: castle-river-mountain-forest

# Or specify custom job ID
uv run bilancio sweep balanced --cloud --job-id my-experiment-name --out-dir out/experiments/my_sweep ...
```

### Job Manifest Structure
```json
{
  "job_id": "castle-river-mountain-forest",
  "created_at": "2025-01-12T10:30:00",
  "completed_at": "2025-01-12T10:45:00",
  "status": "completed",
  "description": "Balanced comparison sweep (n=50, cloud=true)",
  "config": {
    "sweep_type": "balanced",
    "n_agents": 50,
    "kappas": ["0.3", "0.5"],
    "cloud": true
  },
  "run_ids": ["balanced_passive_abc123", "balanced_active_def456"],
  "events": [...]
}
```

---

## Simulation Configuration Guide

When a user requests a simulation, translate their requirements into these parameters:

### Core Parameters

| Parameter | What it controls | Typical values | When to adjust |
|-----------|------------------|----------------|----------------|
| `n_agents` | Ring size (number of firms) | 10-200 | Larger = more realistic but slower |
| `maturity_days` | Payment horizon | 5-20 | Longer = more complex dynamics |
| `kappa` (κ) | Liquidity ratio (L₀/S₁) | 0.1-5 | Lower = more stressed system |
| `concentration` (c) | Debt distribution inequality | 0.1-10 | Lower = more unequal (some agents owe much more) |
| `mu` (μ) | Maturity timing skew | 0-1 | 0=early due dates, 1=late due dates |
| `outside_mid_ratio` (ρ) | Outside money ratio | 0.5-1.0 | Lower = less external liquidity |

### Parameter Intuition

**Kappa (κ) - Liquidity Stress**
- κ < 0.5: Severely liquidity-constrained (expect many defaults)
- κ = 1: Balanced (system has exactly enough cash for debts)
- κ > 2: Liquidity-abundant (few defaults expected)

**Concentration (c) - Debt Distribution**
- c < 0.5: Very unequal (few agents hold most debt) - more fragile
- c = 1: Moderate inequality
- c > 2: More equal distribution - more stable

**Mu (μ) - Payment Timing**
- μ = 0: All payments due early (front-loaded stress)
- μ = 0.5: Evenly distributed
- μ = 1: All payments due late (back-loaded stress)

### Quick Presets

**"Stressed system test"**: κ=0.3, c=0.5, μ=0, n=50
**"Normal conditions"**: κ=1, c=1, μ=0.5, n=100
**"High liquidity"**: κ=2, c=2, μ=0.5, n=100
**"Explore dealer impact"**: Use `sweep balanced` with multiple κ values

### Sweep Types

| Command | Purpose | When to use |
|---------|---------|-------------|
| `sweep ring` | Basic parameter exploration | Understanding system behavior |
| `sweep balanced` | Compare passive vs active dealers | Measuring dealer/trading impact |

---

## Simulation Outputs

### Per-Job Outputs

Each job produces:

1. **Job Metadata** (`job_manifest.json`)
   - Job ID (passphrase)
   - Timestamp (when triggered)
   - Configuration parameters
   - User notes/description
   - Status (pending/running/completed/failed)
   - Event log (all state changes)

2. **Per-Run Artifacts** (in `runs/<run_id>/`)
   - `scenario.yaml` - Full scenario configuration
   - `out/events.jsonl` - Event log (all simulation events)
   - `out/balances.csv` - Balance sheet snapshots
   - `out/metrics.csv` - Key metrics timeseries
   - `out/metrics.html` - Visual metrics report
   - `run.html` - Full simulation visualization

3. **Aggregate Results** (in `aggregate/`)
   - `results.csv` - Summary metrics for all runs
   - `comparison.csv` - Passive vs active comparison (balanced sweep)
   - `dashboard.html` - Visual dashboard
   - `summary.json` - Aggregate statistics

### Key Metrics to Report

| Metric | Meaning | Good values |
|--------|---------|-------------|
| `delta_total` (δ) | Default rate (fraction of debt defaulted) | Lower is better (0 = no defaults) |
| `phi_total` (φ) | Clearing rate (fraction of debt settled) | Higher is better (1 = full clearing) |
| `time_to_stability` | Days until system stabilizes | Lower is better |
| `trading_effect` | δ_passive - δ_active | Positive = dealers help |

### Retrieving Results

```bash
# Query jobs from Supabase (if configured)
bilancio jobs ls --cloud
bilancio jobs get <job_id> --cloud

# List all jobs in Modal Volume
bilancio volume ls

# Get specific job results from Modal Volume
uv run modal volume get bilancio-results <job_id> ./local_results --force

# View aggregate results
cat ./local_results/aggregate/results.csv
```

---

## Example User Requests → Commands

**"Run a quick test to see if dealers help in a stressed system"**
```bash
uv run bilancio sweep balanced --cloud \
  --out-dir out/experiments/castle-river-mountain-forest \
  --n-agents 50 --kappas "0.3,0.5" --concentrations "1" \
  --mus "0" --outside-mid-ratios "1"
```

**"Do a comprehensive sweep across liquidity levels"**
```bash
uv run bilancio sweep balanced --cloud \
  --out-dir out/experiments/bright-ocean-swift-tiger \
  --n-agents 100 --kappas "0.25,0.5,1,2" \
  --concentrations "0.5,1,2" --mus "0,0.5,1" \
  --outside-mid-ratios "1"
```

**"Just run a single scenario to see what happens"**
```bash
uv run bilancio run examples/scenarios/simple_dealer.yaml \
  --html temp/result.html
```

---

## Sweep Pre-Flight: Interactive Parameter Review

**MANDATORY**: Before running ANY simulation or sweep (however simple), present the full parameter review below to the user. Do NOT run until the user confirms. This ensures every run is deliberate and reproducible.

### Step 1: Confirm Sweep Arms

Ask the user which comparison legs to run:

| Arm | CLI mapping | What it tests |
|-----|-------------|---------------|
| Passive (baseline) | always included | No dealer, no lender — pure settlement |
| Active (dealer) | `--active` (default in balanced) | Dealer provides secondary market |
| Lender | `--lender` | Non-bank lender provides credit |
| Dealer + Lender | `--dealer-lender` | Both dealer and lender active |

Also confirm: **Cloud** (Modal) or **Local** execution.

### Step 2: Present All Parameters by Category

Present the following table to the user. Show the **current value** (from CLI args, previous run, or defaults). The user can accept all defaults or change any value.

#### A. General / Scale

| Parameter | Default | Current | Description |
|-----------|---------|---------|-------------|
| `n_agents` | 100 | ? | Number of firms in ring |
| `maturity_days` | 10 | ? | Payment horizon (days) |
| `face_value` | 20 | ? | Face value per ticket |
| `q_total` | 10000 | ? | Total debt amount |
| `base_seed` | 42 | ? | PRNG seed |
| `default_handling` | expel-agent | ? | fail-fast or expel-agent |
| `rollover` | True | ? | Continuous rollover of matured claims |

#### B. Sweep Grid

| Parameter | Default | Current | Description |
|-----------|---------|---------|-------------|
| `kappas` | 0.25,0.5,1,2,4 | ? | Liquidity stress levels |
| `concentrations` | 1 | ? | Debt inequality (Dirichlet c) |
| `mus` | 0 | ? | Maturity timing skew |
| `outside_mid_ratios` (ρ) | 0.90 | ? | Outside-money discount |
| `seeds` | (single) | ? | Multiple seeds for robustness |

#### C. Trader Behavior

| Parameter | Default | Current | Description | Effect |
|-----------|---------|---------|-------------|--------|
| `risk_aversion` | 0 | ? | 0=risk-neutral, 1=max risk-averse | Higher → pickier buyers (buy_premium = 0.01 + 0.02×RA) |
| `planning_horizon` | 10 | ? | Days to look ahead (1-20) | sell_horizon = PH, buy_horizon = PH/2 |
| `aggressiveness` | 1.0 | ? | 0=conservative buyer, 1=eager | Lower → higher surplus needed to buy |
| `buy_reserve_fraction` | 0.5 | ? | Fraction of upcoming dues reserved | Lower → more buyers eligible, less prudent |
| `default_observability` | 1.0 | ? | 0=ignore defaults, 1=full tracking | Lower → agents slower to react to defaults |
| `trading_motive` | liquidity_then_earning | ? | Trading motivation: liquidity_only, liquidity_then_earning, or unrestricted | Higher → restricts speculative buys |

#### D. Dealer & VBT

| Parameter | Default | Current | Description |
|-----------|---------|---------|-------------|
| `dealer_share_per_bucket` | 0.05 | ? | Dealer capital as fraction of bucket |
| `vbt_share_per_bucket` | 0.20 | ? | VBT capital as fraction of bucket |
| `vbt_mid_sensitivity` | 1.0 | ? | VBT mid-price reaction to defaults (0=fixed, 1=full) |
| `vbt_spread_sensitivity` | 0.0 | ? | VBT spread widening with defaults (0=fixed, 1=widen) |
| Bucket spreads (O) | short=0.20, mid=0.30, long=0.40 | ? | Bid-ask spread per bucket |

#### E. Risk Assessment

| Parameter | Default | Current | Description |
|-----------|---------|---------|-------------|
| `risk_assessment` | True | ? | Enable risk-based trade decisions |
| `risk_premium` | 0.02 | ? | Base risk premium (sellers) |
| `risk_urgency` | 0.10 | ? | Urgency sensitivity (how much stress lowers threshold) |
| `initial_prior` | 0.15 | ? | No-history default probability |
| `alpha_vbt` | 0 | ? | VBT informedness (0=naive, 1=kappa-informed) |
| `alpha_trader` | 0 | ? | Trader informedness (0=naive, 1=kappa-informed) |

#### F. Lending (only if lender arm selected)

| Parameter | Default | Current | Description |
|-----------|---------|---------|-------------|
| `lender_base_rate` | 0.05 | ? | Base interest rate |
| `lender_risk_premium_scale` | 0.20 | ? | Risk premium multiplier |
| `lender_risk_aversion` | 0.3 | ? | 0=aggressive, 1=conservative |
| `max_single_exposure` | 0.15 | ? | Max fraction to single borrower |
| `max_total_exposure` | 0.80 | ? | Max total lending exposure |
| `loan_maturity_days` | 2 | ? | Loan maturity |
| `lender_horizon` | 3 | ? | Look-ahead for obligations |

### Step 3: Scale & Cost Estimate

Compute and show:

1. **Total runs** = `len(kappas) × len(concentrations) × len(mus) × len(outside_mid_ratios) × arms`
   - arms = 2 (passive+active), 3 (+ lender), or 4 (+ dealer-lender)
2. **Estimated duration** = `total_runs / 4` minutes (cloud) or `total_runs × 45s` (local)
3. **Estimated cost** = `total_runs × $0.0003` (cloud only)
4. **Grid explosion guard**: if total_runs > 1000, warn and suggest LHS sampling

### Step 4: Parameter Sanity

| Parameter | Valid range | Warn if |
|-----------|-----------|---------|
| κ (kappa) | (0, ∞) | any value > 10 or < 0.05 |
| c (concentration) | (0, ∞) | any value > 10 or < 0.1 |
| μ (mu) | [0, 1] | outside [0, 1] |
| ρ (outside_mid_ratio) | (0, 1] | < 0.5 |
| n_agents | [3, 1000] | > 500 (slow) or < 10 (noisy) |
| maturity_days | [1, 100] | > 30 (slow) or = 1 (no temporal spread) |
| risk_aversion | [0, 1] | = 0 with multi-issuer buys (buyers accept almost everything) |
| No parameter list may be empty | — | empty list = zero runs |

### Step 5: Economic Viability Checks

For each representative parameter combo (pick the median κ, ρ, etc.), compute:

#### V1. Sell trade viability

```
p = initial_prior                 # default 0.15 (day 0)
EV = (1 - p)                     # expected value per unit face
O_short = 0.04 + 0.6 × p         # short bucket spread
M = ρ × (1 - p)                  # credit-adjusted VBT mid
B = M - O_short/2                # VBT bid (floor for dealer bid)

spread_gap = EV - B = (1 - ρ)(1 - p) + O_short/2
min_urgency = spread_gap / urgency_sensitivity
```

- **min_urgency < 0.5**: Sells clear for moderately stressed agents ✓
- **min_urgency 0.5–1.0**: Sells only for very stressed agents (κ < 0.5) ⚠️
- **min_urgency > 1.0**: Sells IMPOSSIBLE — urgency ratio capped at 1 ✗

#### V2. Buy trade viability

```
A = min(1, M + O_short/2)        # VBT ask (ceiling for dealer ask)
buy_premium = 0.01 + 0.02 × risk_aversion    # buyer's minimum premium
buy_clears = (1-ρ)(1-p) > O_short/2 + buy_premium
```

- ρ = 1.0: buys NEVER clear (LHS = 0) ✗
- ρ < 0.95 with risk_aversion=0: buys very permissive (1% threshold) ⚠️
- ρ < 0.95 with risk_aversion ≥ 0.5: buys selective (2%+ threshold) ✓

#### V3. Both-way trading (dealer inventory turnover)

- **κ < 0.3**: Mostly sells, few buys → one-way flow
- **0.3 ≤ κ ≤ 2**: Sweet spot — stressed sell, surplus buy
- **κ > 2**: Few sells, many potential buyers → one-way flow
- **Flag**: If ALL κ > 2 or ALL < 0.3, trading is one-directional.

#### V4. Dealer capacity

```
dealer_cash ≈ dealer_share × system_value
K_star_initial = floor(dealer_cash / M)
```

- K_star_initial < 3: capacity binds immediately → passthrough ⚠️
- K_star_initial > n_agents/2: dealer never constrained → tight spread
- **Ideal**: between 5 and n_agents/4

#### V5. Lending viability (if lender arm)

```
p_lender = 1 / (1 + κ)
rate ≈ profit_target + risk_premium_scale × p_lender
```

- κ > 3: loans cheap but few borrowers need them ⚠️
- 0.3 < κ < 1.5: meaningful lending ✓
- κ < 0.3: very expensive loans ⚠️

#### V6. Temporal spread

- maturity_days = 1: trading ONLY on day 0 ✗
- maturity_days ≥ 5: multi-day trading ✓
- μ = 0: front-loaded stress ⚠️ / μ = 1: back-loaded burst ⚠️
- μ ∈ [0.3, 0.7]: good spread ✓
- buy_reserve_fraction = 1: buyers disappear after day 0 ✗

#### V7. Trading effect detectable

- n_agents ≥ 10 ✓
- At least one κ < 1 ✓
- At least two κ values ✓
- maturity_days ≥ 5 ✓

### Step 6: Pre-Flight Summary

Present to user before running:

```
SWEEP PRE-FLIGHT
════════════════

Arms:     [passive] [active] [lender?] [dealer+lender?]
Cloud:    [yes/no]

─── A. General ───────────────────────────────────────
  n_agents=100  maturity=10d  face=20  seed=42

─── B. Grid ──────────────────────────────────────────
  κ ∈ {0.3, 0.5, 1.0}   c ∈ {1}   μ ∈ {0}   ρ ∈ {0.90}
  → X pairs, Y runs  ~Z min  ~$W

─── C. Trader Behavior ──────────────────────────────
  risk_aversion=0.5  planning_horizon=10  aggressiveness=1.0
  buy_reserve_fraction=0.5  default_observability=1.0

─── D. Dealer & VBT ─────────────────────────────────
  dealer_share=0.05  vbt_share=0.20
  vbt_mid_sensitivity=1.0  vbt_spread_sensitivity=0.0

─── E. Risk Assessment ──────────────────────────────
  risk_premium=0.02  urgency=0.10  initial_prior=0.15
  alpha_vbt=0  alpha_trader=0

─── F. Lending (if applicable) ──────────────────────
  base_rate=0.05  risk_premium_scale=0.20  risk_aversion=0.3

─── Viability ────────────────────────────────────────
  Sells:    [✓/⚠️/✗] (min_urgency=…)
  Buys:     [✓/⚠️/✗] (buy_premium=…)
  2-way:    [✓/⚠️]
  Capacity: K*=… [✓/⚠️]
  Lending:  [✓/⚠️/n/a]
  Temporal: [✓/⚠️/✗]
  Effect:   [✓/⚠️]

Issues: [list any ✗ or ⚠️ items]
Proceed? [waiting for user confirmation]
```

**IMPORTANT**: If any parameter differs from the default, highlight it explicitly. If the user says "run with defaults", still present the summary — defaults should be a conscious choice, not an accident.
