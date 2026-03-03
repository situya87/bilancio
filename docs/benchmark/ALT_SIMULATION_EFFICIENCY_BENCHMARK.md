# Alternative Simulation Efficiency Benchmark (ASEB)

## Purpose

The ASEB benchmark measures **real simulation performance properties** by running actual simulations at different scales and measuring timing, memory, and scaling behavior. Unlike the original Simulation Efficiency Benchmark (which focuses on Modal cloud deployment safety), ASEB measures:

- **Scaling discipline** — how runtime and memory grow with ring size
- **Cold start cost** — import time for the simulation engine
- **Throughput stability** — per-day timing consistency over a simulation run
- **Memory stability** — no memory leaks across repeated runs
- **Event growth discipline** — events/agent/day stays constant as ring grows
- **Complex path coverage** — dealer and banking subsystems complete within budget

## Running

```bash
uv run python scripts/run_alt_simulation_efficiency_benchmark.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--target-score` | 70 | Minimum score to pass |
| `--out-json` | `temp/alt_simulation_efficiency_benchmark_report.json` | JSON report path |
| `--out-md` | `temp/alt_simulation_efficiency_benchmark_report.md` | Markdown report path |

### Expected runtime

~10-15 seconds locally. The dominant cost is the scaling test (n=200 ring).

## Score Model (100 points)

| # | Category | Points | What it measures |
|---|----------|--------|------------------|
| 1 | Scaling Discipline | 25 | Runtime power-law exponent (12), memory exponent (8), event linearity (5) |
| 2 | Cold Start Cost | 10 | Cold import time via subprocess (10) |
| 3 | Per-Day Throughput Stability | 20 | Day-over-day timing ratio (12), CV of per-day times (8) |
| 4 | Memory Stability | 15 | Peak memory growth across 3 sequential runs (15) |
| 5 | Event Growth Discipline | 10 | Events/agent/day consistency across ring sizes (10) |
| 6 | Complex Path Coverage | 20 | Dealer path within budget (10), bank path within budget (10) |

## Critical Gates

7 critical gates. Any failure caps the grade.

| Gate | Condition | What it catches |
|------|-----------|-----------------|
| `scaling::runtime_exponent` | α_runtime ≤ 2.0 | Super-quadratic scaling |
| `scaling::memory_exponent` | α_memory ≤ 2.0 | Super-quadratic memory growth |
| `cold_start::under_5s` | import time < 5.0s | Import-time regressions |
| `stability::no_blowup` | day ratio < 5.0 | Per-day timing blowup |
| `memory::no_leak` | growth ratio < 1.5 | Memory leaks |
| `complex::dealer_completes` | Dealer run succeeds | Dealer subsystem crashes |
| `complex::bank_completes` | Bank run succeeds | Banking subsystem crashes |

## Category Details

### 1. Scaling Discipline (25 pts)

Runs ring simulations for n ∈ {25, 50, 100, 200} with κ=1.0, maturity=5, max_days=10.

**Power-law exponent fitting:** For consecutive pairs (n₁, n₂), computes α = log(t₂/t₁) / log(n₂/n₁). Takes the worst (maximum) α across all pairs.

**Runtime exponent (12 pts):**

| α_runtime | Score |
|-----------|-------|
| ≤ 1.2 | 12 (full — near-linear) |
| 1.2 - 2.5 | Linear interpolation |
| ≥ 2.5 | 0 (super-quadratic) |

**Memory exponent (8 pts):** Same scoring formula applied to peak tracemalloc memory.

**Event linearity (5 pts):** Ratio of max/min events_per_agent_per_day across ring sizes.

| Ratio | Score |
|-------|-------|
| ≤ 1.3 | 5 (full — events scale linearly with agents) |
| 1.3 - 3.0 | Linear interpolation |
| ≥ 3.0 | 0 |

### 2. Cold Start Cost (10 pts)

Runs `python -c "import bilancio.engines.simulation"` as a subprocess 3 times, takes the median.

| Median import time | Score |
|--------------------|-------|
| ≤ 1.0s | 10 (full) |
| 1.0 - 3.0s | Linear interpolation |
| ≥ 3.0s | 0 |

This detects import-time regressions from heavy module-level imports.

### 3. Per-Day Throughput Stability (20 pts)

n=50, κ=1.0 (balanced), maturity=20, 20 simulation days. Each day timed individually. Uses balanced liquidity so all payables settle (no defaults, no ring shrinkage). Maturity matches the measurement window so every day has real settlement work — avoiding bimodal timing from idle tail days.

**Day ratio (12 pts):** median(last 5 days) / median(first 5 days).

| Day ratio | Score |
|-----------|-------|
| ≤ 1.3 | 12 (full — steady throughput) |
| 1.3 - 3.0 | Linear interpolation |
| ≥ 3.0 | 0 (late days much slower) |

**CV of per-day times (8 pts):** stdev/mean of all 20 day timings.

| CV | Score |
|----|-------|
| ≤ 0.3 | 8 (full — very consistent) |
| 0.3 - 1.0 | Linear interpolation |
| ≥ 1.0 | 0 |

### 4. Memory Stability (15 pts)

Runs the same scenario (n=50, κ=0.5, maturity=5, max_days=10) 3 times sequentially with tracemalloc. Computes growth_ratio = peak₃ / peak₁.

| Growth ratio | Score |
|--------------|-------|
| ≤ 1.05 | 15 (full — no growth) |
| 1.05 - 1.5 | Linear interpolation |
| ≥ 1.5 | 0 (significant leak) |

Each run calls `gc.collect()` before starting to reduce GC noise.

### 5. Event Growth Discipline (10 pts)

Reuses data from Category 1. Computes events_per_agent_per_day for each ring size. The ratio max/min should be close to 1.0 if event generation scales linearly with ring size.

| Ratio | Score |
|-------|-------|
| ≤ 1.3 | 10 (full) |
| 1.3 - 3.0 | Linear interpolation |
| ≥ 3.0 | 0 |

### 6. Complex Path Coverage (20 pts)

Tests that complex simulation paths (dealer, banking) complete within time budgets.

**Dealer path (10 pts):** n=25, κ=0.5, maturity=5. Initializes dealer subsystem with standard `DealerRingConfig`, runs with `enable_dealer=True`. Budget: 30 seconds.

**Bank path (10 pts):** n=25, κ=0.5, maturity=5. Initializes banking subsystem with `BankProfile`, runs with `enable_banking=True, enable_bank_lending=True`. Budget: 15 seconds.

Scoring per path:
- Completed within budget → full points
- Completed within 2× budget → linear from full to 0
- Failed (exception) → 0

## Grading

| Score | Grade |
|-------|-------|
| ≥ 90 | A |
| ≥ 80 | B |
| ≥ 70 | C |
| ≥ 60 | D |
| < 60 | F |

Critical gate failure caps: 1 → C, 2-3 → D, 4+ → F.

## Interpreting Results

### What good scores mean

- **Scaling α ≤ 1.5**: Runtime grows moderately with ring size. OK for cloud sweeps.
- **Scaling α > 2.0**: Super-quadratic. Doubling ring size quadruples runtime. Cloud timeout risk.
- **Cold start < 1s**: Fast function spin-up on Modal.
- **Day ratio < 1.3**: No per-day degradation. Simulations won't slow down mid-run.
- **Memory growth ≤ 1.05**: No memory leak. Safe for long-running sweeps.
- **Event ratio ≤ 1.3**: Event log size scales predictably with ring size.

### Common failure modes

1. **High runtime exponent**: Usually means O(n²) or worse in settlement phase. Check `run_day` inner loops.
2. **Memory growth > 1.5**: Events list, contract dict, or agent state accumulating without cleanup.
3. **High day-time CV**: Some days trigger expensive code paths (e.g., many simultaneous defaults causing cascading expulsions).
4. **Dealer path timeout**: Dealer trading loop has O(n × buckets × arrivals) complexity. Check `N_max` and arrival counting.

## Output

### JSON Report

```json
{
  "benchmark": "ASEB (Alternative Simulation Efficiency Benchmark)",
  "total_score": 86.3,
  "grade": "B",
  "target_met": true,
  "critical_failures": 0,
  "categories": [...],
  "critical_checks": [...]
}
```

### Markdown Report

Human-readable summary with score table, critical gates, and per-category details including raw measurements.
