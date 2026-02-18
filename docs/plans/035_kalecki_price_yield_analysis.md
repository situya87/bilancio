# Plan 035: Kalecki Ring Price & Yield Curve Analysis

**Status**: Draft
**Goal**: Add price analysis (VWAP, bid-ask spreads, discount to face) and yield curve analysis (trade-implied, VBT-implied, dealer-implied) to the Kalecki ring output pipeline — per-run CSV, aggregate comparison, HTML dashboard, and notebook.

---

## Context

The dealer subsystem already captures rich trade microstructure data:
- `TradeRecord` with `unit_price`, `bucket`, `maturity_day`, `face_value`, `is_passthrough`, dealer bid/ask before/after
- `DealerSnapshot` with `bid`, `ask`, `midline`, `vbt_mid`, `vbt_spread` per day/bucket
- Existing `trades.csv` and `inventory_timeseries.csv` per run

**What's missing**: No systematic computation of:
1. Daily aggregate price metrics (VWAP, volume, spread) per bucket
2. Simple yield implied by prices: `y = (1/P - 1) / τ` where τ = remaining maturity
3. Three yield curves (trade-implied, VBT-implied, dealer-implied) per day
4. Yield curve dynamics (level, slope, curvature) over time
5. Cross-run comparison of price/yield metrics (passive vs active, across κ)

---

## Design Decisions

- **Yield formula**: Simple yield `y = (S/P - 1) / τ` (per-day units). Annualize by ×365 for display.
- **Yield curves**: Three sources — trade-implied (from executed trades), VBT-implied (from VBT mid/bid/ask), dealer-implied (from dealer kernel quotes). All at bucket level (3 points: short/mid/long).
- **Price metrics**: Absolute levels AND discount to face (`1 - price`).
- **Spread tracking**: All three — dealer interior spread, VBT outside spread, effective (realized) spread from trades.
- **Granularity**: Daily aggregates per bucket (one row per day×bucket). No individual trade rows in the new CSV.
- **Narrative**: General toolkit supporting both "dealer efficiency" and "stress dynamics" stories.

---

## Part A: New Analysis Module — `bilancio/analysis/price_yield.py`

### A.1 Data Structures

```python
@dataclass
class BucketPriceMetrics:
    """Daily price metrics for a single maturity bucket."""
    day: int
    bucket: str                    # "short", "mid", "long"

    # Price levels (absolute)
    vwap: float | None             # Volume-weighted average trade price
    vwap_buy: float | None         # VWAP of buy trades only
    vwap_sell: float | None        # VWAP of sell trades only

    # Volume
    trade_count: int               # Number of trades
    trade_count_buy: int
    trade_count_sell: int
    face_volume: float             # Total face value traded

    # Discount to face
    discount_vwap: float | None    # 1 - vwap

    # Dealer quotes
    dealer_bid: float | None
    dealer_ask: float | None
    dealer_mid: float | None

    # VBT quotes
    vbt_bid: float | None
    vbt_ask: float | None
    vbt_mid: float | None

    # Spreads (all three)
    dealer_spread: float | None    # dealer ask - dealer bid (interior)
    vbt_spread: float | None       # vbt ask - vbt bid (outside)
    effective_spread: float | None # 2 × |trade_price - midpoint| averaged over trades

    # Yield (simple: y = (1/P - 1) / τ)
    bucket_tau: float              # representative remaining maturity (bucket midpoint in days)
    yield_trade: float | None      # from VWAP
    yield_vbt_mid: float | None    # from VBT mid
    yield_vbt_bid: float | None    # from VBT bid (investor sell yield)
    yield_vbt_ask: float | None    # from VBT ask (investor buy yield)
    yield_dealer_mid: float | None # from dealer midline
    yield_dealer_bid: float | None
    yield_dealer_ask: float | None


@dataclass
class YieldCurveSnapshot:
    """Yield curve at a single point in time (3 points: short/mid/long)."""
    day: int
    source: str                    # "trade", "vbt", "dealer"

    yield_short: float | None
    yield_mid: float | None
    yield_long: float | None

    # Curve shape metrics
    level: float | None            # average of 3 yields
    slope: float | None            # yield_long - yield_short
    curvature: float | None        # 2*yield_mid - yield_short - yield_long


@dataclass
class PriceYieldSummary:
    """Run-level summary of price and yield dynamics."""
    # Average spreads over entire run
    avg_dealer_spread: dict[str, float]   # by bucket
    avg_vbt_spread: dict[str, float]
    avg_effective_spread: dict[str, float]

    # Yield curve summary
    avg_yield_level: dict[str, float]     # by source (trade/vbt/dealer)
    avg_yield_slope: dict[str, float]

    # Yield dynamics
    yield_volatility: dict[str, float]    # std of daily yield level, by source
    max_yield_short: float | None         # peak short-term yield (stress signal)

    # Price impact of defaults
    yield_pre_first_default: dict[str, float | None]    # by source
    yield_post_first_default: dict[str, float | None]
    yield_shift_at_default: dict[str, float | None]     # post - pre
```

### A.2 Core Functions

```python
def compute_bucket_tau(bucket: str, maturity_days: int, day: int) -> float:
    """Compute representative remaining maturity τ for a bucket on a given day.

    Uses bucket midpoints:
    - short: τ = midpoint of [1, 3] still remaining
    - mid: τ = midpoint of [4, 8] still remaining
    - long: τ = midpoint of [9, maturity_days] still remaining

    Clipped to ≥ 0.5 to avoid division by zero.
    """

def simple_yield(price: float, tau: float) -> float | None:
    """Compute simple yield: y = (1/P - 1) / τ.

    Returns None if price ≤ 0 or tau ≤ 0.
    """

def compute_effective_spread(trades: list[TradeRecord], midpoint: float) -> float | None:
    """Effective spread = mean of 2 × |unit_price - midpoint| across trades."""

def compute_daily_price_yield(
    trades: list[TradeRecord],       # from dealer metrics
    snapshots: list[DealerSnapshot], # from dealer metrics
    maturity_days: int,
) -> list[BucketPriceMetrics]:
    """Compute daily price/yield metrics for all days and buckets.

    Returns one BucketPriceMetrics per (day, bucket) combination.
    """

def compute_yield_curves(
    daily_metrics: list[BucketPriceMetrics],
) -> list[YieldCurveSnapshot]:
    """Extract yield curve snapshots from daily metrics.

    Returns 3 curves per day (trade, vbt, dealer).
    """

def compute_yield_curve_dynamics(
    curves: list[YieldCurveSnapshot],
) -> dict[str, list[tuple[int, float]]]:
    """Compute level, slope, curvature time series per source."""

def compute_price_yield_summary(
    daily_metrics: list[BucketPriceMetrics],
    curves: list[YieldCurveSnapshot],
    events: list[dict],              # system events (for default timing)
) -> PriceYieldSummary:
    """Compute run-level summary statistics."""
```

---

## Part B: Per-Run CSV Output — `prices.csv`

### B.1 File: `{run_id}/out/prices.csv`

One row per (day, bucket). Columns:

| Column | Type | Description |
|--------|------|-------------|
| day | int | Simulation day |
| bucket | str | short/mid/long |
| bucket_tau | float | Representative remaining maturity (days) |
| vwap | float | Volume-weighted average trade price |
| vwap_buy | float | VWAP of buy trades |
| vwap_sell | float | VWAP of sell trades |
| discount_vwap | float | 1 - vwap (discount to face) |
| trade_count | int | Number of trades |
| trade_count_buy | int | Buy trades |
| trade_count_sell | int | Sell trades |
| face_volume | float | Total face value traded |
| dealer_bid | float | Dealer bid quote |
| dealer_ask | float | Dealer ask quote |
| dealer_mid | float | Dealer midline |
| vbt_bid | float | VBT bid |
| vbt_ask | float | VBT ask |
| vbt_mid | float | VBT mid |
| dealer_spread | float | Dealer ask - bid |
| vbt_spread | float | VBT ask - bid |
| effective_spread | float | 2 × mean |price - mid| |
| yield_trade | float | Simple yield from VWAP |
| yield_vbt_mid | float | Simple yield from VBT mid |
| yield_vbt_bid | float | Simple yield from VBT bid |
| yield_vbt_ask | float | Simple yield from VBT ask |
| yield_dealer_mid | float | Simple yield from dealer mid |
| yield_dealer_bid | float | Simple yield from dealer bid |
| yield_dealer_ask | float | Simple yield from dealer ask |

### B.2 Integration Point

In `bilancio/analysis/report.py`, after writing `metrics.csv`, also write `prices.csv` when dealer metrics are available.

Add a `write_prices_csv()` function called from the run finalization pipeline.

---

## Part C: Aggregate Comparison — Extend `comparison.csv`

### C.1 New Columns in `BalancedComparisonResult`

Add these fields to the dataclass:

```python
# Price/yield summary — active arm
active_avg_dealer_spread_short: float | None = None
active_avg_dealer_spread_mid: float | None = None
active_avg_dealer_spread_long: float | None = None
active_avg_effective_spread_short: float | None = None
active_avg_effective_spread_mid: float | None = None
active_avg_effective_spread_long: float | None = None
active_avg_yield_level_trade: float | None = None
active_avg_yield_slope_trade: float | None = None
active_avg_yield_level_vbt: float | None = None
active_avg_yield_slope_vbt: float | None = None
active_yield_vol_trade: float | None = None
active_max_yield_short: float | None = None
```

### C.2 Integration Point

In the balanced comparison runner, after extracting dealer metrics for each run, also compute `PriceYieldSummary` and populate the new fields.

---

## Part D: HTML Dashboard — Yield Curve & Price Visualizations

### D.1 New Plotly Figures in `run_comparison.py`

Add these visualization functions:

1. **`create_spread_evolution_plot(df)`** — Line plot of average effective spread vs κ, colored by arm (passive/active). Shows dealer spread compression.

2. **`create_yield_curve_comparison(df)`** — Grouped bar chart: yield at short/mid/long for passive vs active arm, faceted by κ.

3. **`create_yield_slope_vs_kappa(df)`** — Scatter/line plot: yield curve slope vs κ for each arm. Shows whether stress steepens/inverts the curve.

4. **`create_yield_shift_at_default(df)`** — Paired bars: yield before vs after first default event, by arm. Shows default impact on term structure.

5. **`create_price_heatmap(df)`** — Heatmap: VWAP across (κ, bucket) for each arm. Shows price surface.

### D.2 Per-Run `metrics.html` Extension

Add a "Price & Yield" section to the single-run HTML report:
- Yield curve evolution chart (3 curves × N days)
- Bid-ask spread time series
- VWAP vs VBT mid overlay

---

## Part E: Summary Metrics in `results.csv`

### E.1 New Columns

Add to `report.aggregate_runs()` output:

| Column | Description |
|--------|-------------|
| avg_yield_level_trade | Average trade-implied yield level |
| avg_yield_slope_trade | Average trade-implied yield slope |
| avg_yield_level_vbt | Average VBT-implied yield level |
| avg_dealer_spread | Average dealer spread (across buckets) |
| avg_effective_spread | Average effective spread |
| max_yield_short | Peak short-term yield |
| yield_vol_trade | Yield volatility (std of daily level) |

---

## Part F: Analysis Notebook — `notebooks/demo/yield_curve_analysis.ipynb`

### F.1 Structure

Building on the pattern established by `notebooks/demo/price_analysis.ipynb`:

1. **Setup** — Imports, parameters, run simulation (same as price_analysis)
2. **Price Metrics Extraction** — Use `compute_daily_price_yield()` on metrics
3. **Section A: Yield Curves Over Time** — 3 curves (trade/VBT/dealer) at selected days, overlaid on same axes
4. **Section B: Yield Curve Dynamics** — Level, slope, curvature time series with default events overlaid
5. **Section C: Spread Analysis** — All three spreads (dealer/VBT/effective) over time, by bucket
6. **Section D: Cross-Regime Comparison** — Run both passive and active, compare yield curves
7. **Section E: Price Discovery** — VWAP discount to face over time, dealer compression visualization
8. **Summary Statistics** — `PriceYieldSummary` table

---

## Implementation Order

### Phase 1: Core Module (`price_yield.py`)
- [ ] Create `bilancio/analysis/price_yield.py` with all dataclasses and computation functions
- [ ] Add `simple_yield()`, `compute_bucket_tau()`, `compute_effective_spread()`
- [ ] Implement `compute_daily_price_yield()` — main entry point
- [ ] Implement `compute_yield_curves()` and `compute_yield_curve_dynamics()`
- [ ] Implement `compute_price_yield_summary()`
- [ ] Add `write_prices_csv()` for CSV export
- [ ] Unit tests in `tests/unit/test_price_yield.py`

### Phase 2: Per-Run Integration
- [ ] Hook `write_prices_csv()` into `report.py` run finalization
- [ ] Add yield curve section to per-run `metrics.html`
- [ ] Test with a single run: verify `prices.csv` output

### Phase 3: Comparison Integration
- [ ] Extend `BalancedComparisonResult` with price/yield fields
- [ ] Populate fields in comparison runner
- [ ] Extend `comparison.csv` fieldnames
- [ ] Add columns to `results.csv` via `aggregate_runs()`

### Phase 4: Dashboard Visualizations
- [ ] Add Plotly figure functions to `run_comparison.py`
- [ ] Integrate new sections into `generate_comparison_html()`
- [ ] Test with a small balanced sweep

### Phase 5: Notebook
- [ ] Create `notebooks/demo/yield_curve_analysis.ipynb`
- [ ] Test with `uv run jupyter nbconvert --execute`

---

## Bucket Tau Calculation

The representative remaining maturity for each bucket uses the bucket midpoint, adjusted for the current day:

```
Bucket boundaries (default maturity_days=10):
  short: due days 1-3  → τ_short(day=0) = 2    (midpoint of 1,2,3)
  mid:   due days 4-8  → τ_mid(day=0) = 6      (midpoint of 4,5,6,7,8)
  long:  due days 9-10 → τ_long(day=0) = 9.5   (midpoint of 9,10)

On day d, remaining maturity = original_tau - d, clipped to ≥ 0.5
```

In practice, since snapshots already know the day, and bucket boundaries are fixed, we compute `τ` from the bucket definition and current day.

---

## Yield Computation Edge Cases

1. **No trades in a bucket on a day**: `yield_trade` = None (missing, not zero)
2. **Price = 0 or negative**: Should not happen (VBT floor > 0), but guard with None
3. **τ ≤ 0**: Instrument already matured — yield undefined, return None
4. **Price > 1 (premium to face)**: Yield is negative (paying premium for future cashflow). This is valid and should be reported.
5. **Passive arm**: No dealer quotes exist. `yield_dealer_*` = None. VBT quotes still exist. Trade-implied yields exist only if there were trades (there shouldn't be in passive).

---

## Files Modified

| File | Change |
|------|--------|
| `src/bilancio/analysis/price_yield.py` | **NEW** — Core computation module |
| `src/bilancio/analysis/__init__.py` | Export new module |
| `src/bilancio/analysis/report.py` | Call `write_prices_csv()`, add columns to `results.csv` |
| `src/bilancio/experiments/balanced_comparison.py` | Extend `BalancedComparisonResult`, populate new fields |
| `src/bilancio/analysis/visualization/run_comparison.py` | New Plotly figures for yield/price |
| `tests/unit/test_price_yield.py` | **NEW** — Unit tests |
| `notebooks/demo/yield_curve_analysis.ipynb` | **NEW** — Analysis notebook |
