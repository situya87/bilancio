"""Network and temporal extraction engine for cross-sweep analysis.

Processes events.jsonl files from sweep runs to extract per-run network metrics
(contagion, centrality, credit, funding) and per-day timeseries data.
Results are cached to CSV for fast loading by charts, report, and notebook.
"""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from bilancio.analysis.contagion import (
    classify_defaults,
    contagion_by_day,
    default_dependency_graph,
    time_to_contagion,
)
from bilancio.analysis.credit_creation import (
    credit_created_by_type,
    credit_creation_by_day,
    credit_destroyed_by_type,
    credit_destruction_by_day,
    net_credit_impulse,
)
from bilancio.analysis.funding_chains import cash_inflows_by_source
from bilancio.analysis.network_analysis import (
    betweenness_centrality,
    systemic_importance,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gini_coefficient(values: list[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    arr = sorted(values)
    n = len(arr)
    total = sum(arr)
    if total == 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(arr):
        cumulative += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


def _max_cascade_depth(dep_graph: dict[str, list[str]]) -> int:
    """DFS to find max depth of the default dependency graph."""
    if not dep_graph:
        return 0

    memo: dict[str, int] = {}

    def _depth(node: str, visited: set[str]) -> int:
        if node in memo:
            return memo[node]
        if node not in dep_graph or not dep_graph[node]:
            memo[node] = 0
            return 0
        max_d = 0
        for parent in dep_graph[node]:
            if parent in visited:
                continue  # cycle guard
            visited.add(parent)
            max_d = max(max_d, 1 + _depth(parent, visited))
            visited.discard(parent)
        memo[node] = max_d
        return max_d

    return max(
        _depth(node, {node}) for node in dep_graph
    )


def _load_events(path: Path) -> list[dict[str, Any]]:
    """Load events from a JSONL file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _compute_active_obligations_by_day(
    events: list[dict[str, Any]], max_day: int
) -> dict[int, int]:
    """Track active (outstanding) obligations per day.

    An obligation is created by PayableCreated and removed by
    PayableSettled, ObligationDefaulted, or ObligationWrittenOff.
    """
    active_ids: set[str] = set()
    day_counts: dict[int, int] = {}
    current_day = -1

    for e in events:
        day = e.get("day", 0)
        kind = e["kind"]

        # When day changes, snapshot count
        if day != current_day:
            if current_day >= 0:
                day_counts[current_day] = len(active_ids)
            current_day = day

        instr_id = e.get("instr_id", "")
        if kind == "PayableCreated":
            active_ids.add(instr_id)
        elif kind in ("PayableSettled", "ObligationDefaulted", "ObligationWrittenOff"):
            active_ids.discard(instr_id)

    # Final day
    if current_day >= 0:
        day_counts[current_day] = len(active_ids)

    return day_counts


def _system_funding_mix(
    events: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute system-wide funding mix as fractions by source type."""
    inflows = cash_inflows_by_source(events)
    totals: dict[str, Decimal] = {}
    grand_total = Decimal(0)
    for agent_sources in inflows.values():
        for source, amount in agent_sources.items():
            totals[source] = totals.get(source, Decimal(0)) + amount
            grand_total += amount
    if grand_total == 0:
        return {}
    return {k: float(v / grand_total) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------

# Scalar metric columns
SCALAR_COLS = [
    "run_id", "arm", "sweep", "kappa", "concentration", "mu",
    "outside_mid_ratio", "seed",
    # Contagion
    "n_defaults", "n_primary", "n_secondary", "cascade_fraction",
    "time_to_contagion", "max_cascade_depth",
    # Network
    "n_edges", "mean_betweenness", "max_betweenness", "gini_betweenness",
    "mean_systemic", "max_systemic",
    # Credit
    "net_credit_impulse",
    "credit_created_bank_loan", "credit_created_cb_loan",
    "credit_created_nbfi_loan", "credit_created_payable",
    "credit_destroyed_bank_loan", "credit_destroyed_cb_loan",
    "credit_destroyed_payable",
    # Funding mix
    "funding_settlement_received", "funding_ticket_sale",
    "funding_loan_received", "funding_cb_loan", "funding_nbfi_loan",
    "funding_deposit_interest", "funding_deposit",
]

TIMESERIES_COLS = [
    "run_id", "arm", "sweep", "kappa", "day",
    "primary_defaults", "secondary_defaults",
    "cum_primary_defaults", "cum_secondary_defaults",
    "credit_created", "credit_destroyed",
    "active_obligations",
]


def extract_run_metrics(
    events_path: Path,
    run_id: str,
    arm: str,
    sweep: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Extract all scalar metrics from a single run's events."""
    events = _load_events(events_path)

    row: dict[str, Any] = {
        "run_id": run_id,
        "arm": arm,
        "sweep": sweep,
        **params,
    }

    # --- Contagion ---
    defaults = classify_defaults(events)
    n_primary = sum(1 for d in defaults if d.is_primary)
    n_secondary = sum(1 for d in defaults if not d.is_primary)
    n_total = len(defaults)
    row["n_defaults"] = n_total
    row["n_primary"] = n_primary
    row["n_secondary"] = n_secondary
    row["cascade_fraction"] = n_secondary / n_total if n_total > 0 else 0.0
    row["time_to_contagion"] = time_to_contagion(events)
    dep_graph = default_dependency_graph(events)
    row["max_cascade_depth"] = _max_cascade_depth(dep_graph)

    # --- Network ---
    bc = betweenness_centrality(events)
    bc_values = list(bc.values()) if bc else [0.0]
    row["n_edges"] = _count_edges(events)
    row["mean_betweenness"] = float(np.mean(bc_values))
    row["max_betweenness"] = float(np.max(bc_values))
    row["gini_betweenness"] = _gini_coefficient(bc_values)

    si = systemic_importance(events)
    si_scores = [s["score"] for s in si] if si else [0.0]
    row["mean_systemic"] = float(np.mean(si_scores))
    row["max_systemic"] = float(np.max(si_scores))

    # --- Credit ---
    row["net_credit_impulse"] = float(net_credit_impulse(events))
    created = credit_created_by_type(events)
    for key in ("bank_loan", "cb_loan", "nbfi_loan", "payable"):
        row[f"credit_created_{key}"] = float(created.get(key, Decimal(0)))
    destroyed = credit_destroyed_by_type(events)
    for key in ("bank_loan", "cb_loan", "payable"):
        row[f"credit_destroyed_{key}"] = float(destroyed.get(key, Decimal(0)))

    # --- Funding mix ---
    fmix = _system_funding_mix(events)
    for key in (
        "settlement_received", "ticket_sale", "loan_received",
        "cb_loan", "nbfi_loan", "deposit_interest", "deposit",
    ):
        row[f"funding_{key}"] = fmix.get(key, 0.0)

    return row


def _count_edges(events: list[dict[str, Any]]) -> int:
    """Count unique obligation edges (debtor→creditor pairs)."""
    edges: set[tuple[str, str]] = set()
    for e in events:
        if e["kind"] == "PayableCreated":
            debtor = e.get("debtor") or e.get("from", "")
            creditor = e.get("creditor") or e.get("to", "")
            if debtor and creditor:
                edges.add((debtor, creditor))
    return len(edges)


def extract_run_timeseries(
    events_path: Path,
    run_id: str,
    arm: str,
    sweep: str,
    kappa: float,
) -> list[dict[str, Any]]:
    """Extract per-day timeseries from a single run's events."""
    events = _load_events(events_path)

    # Max day
    max_day = max((e.get("day", 0) for e in events), default=0)

    # Contagion by day
    cbd = contagion_by_day(events)

    # Credit flows by day
    cc_day = credit_creation_by_day(events)
    cd_day = credit_destruction_by_day(events)

    # Active obligations
    active_oblig = _compute_active_obligations_by_day(events, max_day)

    rows = []
    cum_primary = 0
    cum_secondary = 0
    for day in range(max_day + 1):
        day_cont = cbd.get(day, {})
        p = day_cont.get("primary", 0)
        s = day_cont.get("secondary", 0)
        cum_primary += p
        cum_secondary += s

        cc = cc_day.get(day, {})
        cd = cd_day.get(day, {})
        created_total = float(sum(cc.values(), Decimal(0)))
        destroyed_total = float(sum(cd.values(), Decimal(0)))

        rows.append({
            "run_id": run_id,
            "arm": arm,
            "sweep": sweep,
            "kappa": kappa,
            "day": day,
            "primary_defaults": p,
            "secondary_defaults": s,
            "cum_primary_defaults": cum_primary,
            "cum_secondary_defaults": cum_secondary,
            "credit_created": created_total,
            "credit_destroyed": destroyed_total,
            "active_obligations": active_oblig.get(day, 0),
        })
    return rows


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------


def _extract_one_run(args: tuple) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Worker function: extract both scalar and timeseries for one run."""
    events_path, run_id, arm, sweep, params = args
    metrics = extract_run_metrics(
        Path(events_path), run_id, arm, sweep, params
    )
    ts = extract_run_timeseries(
        Path(events_path), run_id, arm, sweep, params.get("kappa", 0.0)
    )
    return metrics, ts


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _resolve_run_paths(
    comparison_csv: Path,
    base_dir: Path,
    arm_column_map: dict[str, tuple[str, str]],
    sweep_name: str,
) -> list[tuple[str, str, str, str, dict[str, Any]]]:
    """Build list of (events_path, run_id, arm, sweep, params) from CSV.

    arm_column_map: {arm_name: (run_id_column, subdir_name)}
    """
    tasks = []
    with open(comparison_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            params = {}
            for key in ("kappa", "concentration", "mu", "outside_mid_ratio", "seed"):
                val = row.get(key, "")
                try:
                    params[key] = float(val)
                except (ValueError, TypeError):
                    params[key] = val

            for arm_name, (id_col, subdir) in arm_column_map.items():
                run_id = row.get(id_col, "")
                if not run_id:
                    continue
                status_col = id_col.replace("_run_id", "_status")
                if row.get(status_col, "completed") != "completed":
                    continue
                events_path = base_dir / subdir / "runs" / run_id / "out" / "events.jsonl"
                if events_path.exists():
                    tasks.append(
                        (str(events_path), run_id, arm_name, sweep_name, params)
                    )
    return tasks


def extract_all_network_data(
    nbfi_csv: Path | None,
    bank_csv: Path | None,
    nbfi_dir: Path | None,
    bank_dir: Path | None,
    cache_dir: Path,
    n_workers: int = 4,
    force: bool = False,
) -> tuple[Path, Path]:
    """Extract network metrics and timeseries from all sweep runs.

    Returns (metrics_csv_path, timeseries_csv_path).
    Skips extraction if cache files are newer than comparison CSVs (unless force=True).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cache_dir / "network_metrics.csv"
    timeseries_path = cache_dir / "network_timeseries.csv"

    # Check cache freshness
    if not force and metrics_path.exists() and timeseries_path.exists():
        cache_mtime = min(metrics_path.stat().st_mtime, timeseries_path.stat().st_mtime)
        csv_mtime = 0.0
        if nbfi_csv and nbfi_csv.exists():
            csv_mtime = max(csv_mtime, nbfi_csv.stat().st_mtime)
        if bank_csv and bank_csv.exists():
            csv_mtime = max(csv_mtime, bank_csv.stat().st_mtime)
        if cache_mtime > csv_mtime:
            logger.info("Cache is fresh, skipping extraction")
            return metrics_path, timeseries_path

    # Build task list
    all_tasks: list[tuple] = []

    if nbfi_csv and nbfi_dir and nbfi_csv.exists():
        nbfi_arms = {
            "passive": ("passive_run_id", "passive"),
            "active": ("active_run_id", "active"),
            "nbfi": ("lender_run_id", "nbfi"),
        }
        all_tasks.extend(
            _resolve_run_paths(nbfi_csv, nbfi_dir, nbfi_arms, "nbfi")
        )

    if bank_csv and bank_dir and bank_csv.exists():
        bank_arms = {
            "idle": ("idle_run_id", "bank_idle"),
            "lend": ("lend_run_id", "bank_lend"),
        }
        all_tasks.extend(
            _resolve_run_paths(bank_csv, bank_dir, bank_arms, "bank")
        )

    if not all_tasks:
        logger.warning("No runs found to extract")
        return metrics_path, timeseries_path

    logger.info(f"Extracting network data from {len(all_tasks)} runs with {n_workers} workers")

    all_metrics: list[dict[str, Any]] = []
    all_ts: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_extract_one_run, task): task for task in all_tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            task = futures[future]
            try:
                metrics, ts = future.result()
                all_metrics.append(metrics)
                all_ts.extend(ts)
            except Exception:
                run_id = task[1]
                logger.exception(f"Failed to extract run {run_id}")
            if done % 50 == 0:
                logger.info(f"Progress: {done}/{len(all_tasks)} runs")

    logger.info(f"Extracted {len(all_metrics)} runs, {len(all_ts)} timeseries rows")

    # Write metrics CSV
    if all_metrics:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCALAR_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metrics)

    # Write timeseries CSV
    if all_ts:
        with open(timeseries_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TIMESERIES_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_ts)

    return metrics_path, timeseries_path


# ---------------------------------------------------------------------------
# Data loading (from cached CSVs)
# ---------------------------------------------------------------------------


def load_network_metrics(cache_dir: Path) -> dict[str, np.ndarray]:
    """Load cached network_metrics.csv as column-oriented numpy arrays."""
    path = Path(cache_dir) / "network_metrics.csv"
    if not path.exists():
        return {}
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    # String columns
    str_cols = {"run_id", "arm", "sweep"}
    result: dict[str, np.ndarray] = {}
    for col in rows[0]:
        vals = [r.get(col, "") for r in rows]
        if col in str_cols:
            result[col] = np.array(vals, dtype=object)
        else:
            float_vals = []
            for v in vals:
                try:
                    float_vals.append(float(v) if v != "" and v != "None" else np.nan)
                except (ValueError, TypeError):
                    float_vals.append(np.nan)
            result[col] = np.array(float_vals)
    return result


def load_network_timeseries(cache_dir: Path) -> dict[str, np.ndarray]:
    """Load cached network_timeseries.csv as column-oriented numpy arrays."""
    path = Path(cache_dir) / "network_timeseries.csv"
    if not path.exists():
        return {}
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    str_cols = {"run_id", "arm", "sweep"}
    result: dict[str, np.ndarray] = {}
    for col in rows[0]:
        vals = [r.get(col, "") for r in rows]
        if col in str_cols:
            result[col] = np.array(vals, dtype=object)
        else:
            float_vals = []
            for v in vals:
                try:
                    float_vals.append(float(v) if v != "" and v != "None" else np.nan)
                except (ValueError, TypeError):
                    float_vals.append(np.nan)
            result[col] = np.array(float_vals)
    return result


def aggregate_timeseries_by_arm_kappa(
    ts: dict[str, np.ndarray],
    metric_cols: list[str] | None = None,
) -> dict[str, dict[float, dict[int, dict[str, float]]]]:
    """Aggregate timeseries by arm and kappa, averaging over runs.

    Returns: {arm: {kappa: {day: {metric: mean_value}}}}
    """
    if not ts or "arm" not in ts:
        return {}

    if metric_cols is None:
        metric_cols = [
            "primary_defaults", "secondary_defaults",
            "cum_primary_defaults", "cum_secondary_defaults",
            "credit_created", "credit_destroyed",
            "active_obligations",
        ]

    arms = ts["arm"]
    kappas = ts["kappa"]
    days = ts["day"]

    # Group: (arm, kappa, day) -> list of row indices
    groups: dict[tuple[str, float, int], list[int]] = {}
    for i in range(len(arms)):
        key = (str(arms[i]), float(kappas[i]), int(days[i]))
        groups.setdefault(key, []).append(i)

    result: dict[str, dict[float, dict[int, dict[str, float]]]] = {}
    for (arm, kappa, day), indices in groups.items():
        arm_dict = result.setdefault(arm, {})
        kappa_dict = arm_dict.setdefault(kappa, {})
        day_metrics: dict[str, float] = {}
        for col in metric_cols:
            if col in ts:
                vals = ts[col][indices]
                finite = vals[np.isfinite(vals)]
                day_metrics[col] = float(np.mean(finite)) if len(finite) > 0 else 0.0
        kappa_dict[day] = day_metrics

    return result


# ---------------------------------------------------------------------------
# Section 9: Mechanism internals timeseries
# ---------------------------------------------------------------------------

MECHANISM_COLS = [
    "run_id", "arm", "sweep", "kappa", "day",
    # Dealer
    "n_buys", "n_sells", "n_sell_rejected", "face_bought", "face_sold",
    "mean_buy_price", "mean_sell_price",
    "mean_price_short", "mean_price_mid", "mean_price_long",
    "n_liquidity_driven", "n_earning_driven",
    # Bank
    "n_loans_issued", "loan_volume_issued",
    "n_loan_defaults", "default_volume",
    "n_loan_repaid", "repaid_volume",
    "mean_loan_rate",
    # Interbank
    "interbank_cleared_volume", "n_interbank_cleared",
    "auction_volume", "auction_rate", "n_auction_trades", "n_unfilled",
    # CB
    "n_cb_loans", "cb_loans_created_volume",
    "cb_loans_repaid_volume", "n_cb_backstop",
    # Payments
    "intrabank_volume", "n_intrabank",
    "client_payment_volume", "n_client_payments",
    "reserves_transferred_volume",
]


def extract_mechanism_timeseries(
    events_path: Path,
    run_id: str,
    arm: str,
    sweep: str,
    kappa: float,
) -> list[dict[str, Any]]:
    """Extract per-day mechanism internals from a single run's events.

    Populates dealer fields for active arm, bank/interbank/CB fields for
    bank_lend/bank_idle arms.
    """
    events = _load_events(events_path)
    if not events:
        return []

    max_day = max((e.get("day", 0) for e in events), default=0)

    # Pre-aggregate events by day
    day_data: dict[int, dict[str, Any]] = {
        d: {
            # Dealer
            "buys": [], "sells": [], "sell_rejected": 0,
            "buy_prices": [], "sell_prices": [],
            "prices_short": [], "prices_mid": [], "prices_long": [],
            "liquidity": 0, "earning": 0,
            # Bank
            "loans_issued": [], "loan_defaults": [], "loan_repaid": [],
            "loan_rates": [],
            # Interbank
            "ib_cleared": [], "auction_vol": 0.0, "auction_rate": None,
            "auction_trades": 0, "unfilled": 0,
            # CB
            "cb_created": [], "cb_repaid_vol": 0.0, "cb_backstop": 0,
            # Payments
            "intrabank": [], "client": [], "reserves": [],
        }
        for d in range(max_day + 1)
    }

    for e in events:
        day = e.get("day", 0)
        if day not in day_data:
            continue
        dd = day_data[day]
        kind = e["kind"]

        # --- Dealer events ---
        if kind == "dealer_trade":
            side = e.get("side", "")
            face = float(e.get("face", 0))
            unit_price = float(e.get("unit_price", 0))
            bucket = e.get("bucket", "")
            if side == "buy":
                dd["buys"].append(face)
                dd["buy_prices"].append(unit_price)
            elif side == "sell":
                dd["sells"].append(face)
                dd["sell_prices"].append(unit_price)
            # Bucket prices
            if bucket == "short":
                dd["prices_short"].append(unit_price)
            elif bucket == "mid":
                dd["prices_mid"].append(unit_price)
            elif bucket == "long":
                dd["prices_long"].append(unit_price)
            # Liquidity vs earning
            if e.get("is_liquidity_driven"):
                dd["liquidity"] += 1
            else:
                dd["earning"] += 1

        elif kind == "sell_rejected":
            dd["sell_rejected"] += 1

        # --- Bank events ---
        elif kind == "BankLoanIssued":
            dd["loans_issued"].append(float(e.get("amount", 0)))
            rate_str = e.get("rate", "0")
            try:
                dd["loan_rates"].append(float(rate_str))
            except (ValueError, TypeError):
                pass

        elif kind == "BankLoanDefault":
            dd["loan_defaults"].append(float(e.get("principal", 0)))

        elif kind == "BankLoanRepaid":
            dd["loan_repaid"].append(float(e.get("principal", 0)))

        # --- Interbank ---
        elif kind == "InterbankCleared":
            dd["ib_cleared"].append(float(e.get("amount", 0)))

        elif kind == "InterbankAuction":
            vol = e.get("total_volume", 0)
            dd["auction_vol"] += float(vol) if vol else 0.0
            rate = e.get("clearing_rate")
            if rate is not None:
                try:
                    dd["auction_rate"] = float(rate)
                except (ValueError, TypeError):
                    pass
            dd["auction_trades"] += int(e.get("n_trades", 0))
            dd["unfilled"] += int(e.get("n_unfilled", 0))

        elif kind == "InterbankUnfilled":
            pass  # already counted via InterbankAuction.n_unfilled

        # --- CB ---
        elif kind == "CBLoanCreated":
            dd["cb_created"].append(float(e.get("amount", 0)))

        elif kind == "CBLoanRepaid":
            dd["cb_repaid_vol"] += float(e.get("principal", 0))

        elif kind == "CBBackstopLoan":
            dd["cb_backstop"] += 1

        # --- Payments ---
        elif kind == "IntraBankPayment":
            dd["intrabank"].append(float(e.get("amount", 0)))

        elif kind == "ClientPayment":
            dd["client"].append(float(e.get("amount", 0)))

        elif kind == "ReservesTransferred":
            dd["reserves"].append(float(e.get("amount", 0)))

    # Build rows
    rows = []
    for day in range(max_day + 1):
        dd = day_data[day]
        row: dict[str, Any] = {
            "run_id": run_id,
            "arm": arm,
            "sweep": sweep,
            "kappa": kappa,
            "day": day,
            # Dealer
            "n_buys": len(dd["buys"]),
            "n_sells": len(dd["sells"]),
            "n_sell_rejected": dd["sell_rejected"],
            "face_bought": sum(dd["buys"]),
            "face_sold": sum(dd["sells"]),
            "mean_buy_price": (
                sum(dd["buy_prices"]) / len(dd["buy_prices"])
                if dd["buy_prices"] else float("nan")
            ),
            "mean_sell_price": (
                sum(dd["sell_prices"]) / len(dd["sell_prices"])
                if dd["sell_prices"] else float("nan")
            ),
            "mean_price_short": (
                sum(dd["prices_short"]) / len(dd["prices_short"])
                if dd["prices_short"] else float("nan")
            ),
            "mean_price_mid": (
                sum(dd["prices_mid"]) / len(dd["prices_mid"])
                if dd["prices_mid"] else float("nan")
            ),
            "mean_price_long": (
                sum(dd["prices_long"]) / len(dd["prices_long"])
                if dd["prices_long"] else float("nan")
            ),
            "n_liquidity_driven": dd["liquidity"],
            "n_earning_driven": dd["earning"],
            # Bank
            "n_loans_issued": len(dd["loans_issued"]),
            "loan_volume_issued": sum(dd["loans_issued"]),
            "n_loan_defaults": len(dd["loan_defaults"]),
            "default_volume": sum(dd["loan_defaults"]),
            "n_loan_repaid": len(dd["loan_repaid"]),
            "repaid_volume": sum(dd["loan_repaid"]),
            "mean_loan_rate": (
                sum(dd["loan_rates"]) / len(dd["loan_rates"])
                if dd["loan_rates"] else float("nan")
            ),
            # Interbank
            "interbank_cleared_volume": sum(dd["ib_cleared"]),
            "n_interbank_cleared": len(dd["ib_cleared"]),
            "auction_volume": dd["auction_vol"],
            "auction_rate": dd["auction_rate"] if dd["auction_rate"] is not None else float("nan"),
            "n_auction_trades": dd["auction_trades"],
            "n_unfilled": dd["unfilled"],
            # CB
            "n_cb_loans": len(dd["cb_created"]),
            "cb_loans_created_volume": sum(dd["cb_created"]),
            "cb_loans_repaid_volume": dd["cb_repaid_vol"],
            "n_cb_backstop": dd["cb_backstop"],
            # Payments
            "intrabank_volume": sum(dd["intrabank"]),
            "n_intrabank": len(dd["intrabank"]),
            "client_payment_volume": sum(dd["client"]),
            "n_client_payments": len(dd["client"]),
            "reserves_transferred_volume": sum(dd["reserves"]),
        }
        rows.append(row)
    return rows


def _extract_one_mechanism(args: tuple) -> list[dict[str, Any]]:
    """Worker function: extract mechanism timeseries for one run."""
    events_path, run_id, arm, sweep, params = args
    return extract_mechanism_timeseries(
        Path(events_path), run_id, arm, sweep, params.get("kappa", 0.0)
    )


def extract_all_mechanism_data(
    nbfi_csv: Path | None,
    bank_csv: Path | None,
    nbfi_dir: Path | None,
    bank_dir: Path | None,
    cache_dir: Path,
    n_workers: int = 4,
    force: bool = False,
) -> Path:
    """Extract mechanism internals timeseries from all sweep runs.

    Returns path to mechanism_timeseries.csv.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mech_path = cache_dir / "mechanism_timeseries.csv"

    # Check cache
    if not force and mech_path.exists():
        cache_mtime = mech_path.stat().st_mtime
        csv_mtime = 0.0
        if nbfi_csv and nbfi_csv.exists():
            csv_mtime = max(csv_mtime, nbfi_csv.stat().st_mtime)
        if bank_csv and bank_csv.exists():
            csv_mtime = max(csv_mtime, bank_csv.stat().st_mtime)
        if cache_mtime > csv_mtime:
            logger.info("Mechanism cache is fresh, skipping extraction")
            return mech_path

    # Build task list (reuse _resolve_run_paths)
    all_tasks: list[tuple] = []

    if nbfi_csv and nbfi_dir and nbfi_csv.exists():
        nbfi_arms = {
            "passive": ("passive_run_id", "passive"),
            "active": ("active_run_id", "active"),
            "nbfi": ("lender_run_id", "nbfi"),
        }
        all_tasks.extend(
            _resolve_run_paths(nbfi_csv, nbfi_dir, nbfi_arms, "nbfi")
        )

    if bank_csv and bank_dir and bank_csv.exists():
        bank_arms = {
            "idle": ("idle_run_id", "bank_idle"),
            "lend": ("lend_run_id", "bank_lend"),
        }
        all_tasks.extend(
            _resolve_run_paths(bank_csv, bank_dir, bank_arms, "bank")
        )

    if not all_tasks:
        logger.warning("No runs found for mechanism extraction")
        return mech_path

    logger.info(
        f"Extracting mechanism data from {len(all_tasks)} runs "
        f"with {n_workers} workers"
    )

    all_ts: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_extract_one_mechanism, task): task
            for task in all_tasks
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            task = futures[future]
            try:
                ts = future.result()
                all_ts.extend(ts)
            except Exception:
                run_id = task[1]
                logger.exception(f"Failed mechanism extraction for {run_id}")
            if done % 50 == 0:
                logger.info(f"Mechanism progress: {done}/{len(all_tasks)} runs")

    logger.info(f"Extracted {len(all_ts)} mechanism timeseries rows")

    if all_ts:
        with open(mech_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=MECHANISM_COLS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(all_ts)

    return mech_path


def load_mechanism_timeseries(cache_dir: Path) -> dict[str, np.ndarray]:
    """Load cached mechanism_timeseries.csv as column-oriented numpy arrays."""
    path = Path(cache_dir) / "mechanism_timeseries.csv"
    if not path.exists():
        return {}
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    str_cols = {"run_id", "arm", "sweep"}
    result: dict[str, np.ndarray] = {}
    for col in rows[0]:
        vals = [r.get(col, "") for r in rows]
        if col in str_cols:
            result[col] = np.array(vals, dtype=object)
        else:
            float_vals = []
            for v in vals:
                try:
                    float_vals.append(
                        float(v) if v not in ("", "None", "nan") else np.nan
                    )
                except (ValueError, TypeError):
                    float_vals.append(np.nan)
            result[col] = np.array(float_vals)
    return result


def aggregate_mechanism_timeseries(
    ts: dict[str, np.ndarray],
    metric_cols: list[str] | None = None,
) -> dict[str, dict[float, dict[int, dict[str, float]]]]:
    """Aggregate mechanism timeseries by arm and kappa, averaging over runs.

    Returns: {arm: {kappa: {day: {metric: mean_value}}}}
    """
    if not ts or "arm" not in ts:
        return {}

    if metric_cols is None:
        metric_cols = [c for c in MECHANISM_COLS if c not in (
            "run_id", "arm", "sweep", "kappa", "day",
        )]

    arms = ts["arm"]
    kappas = ts["kappa"]
    days = ts["day"]

    groups: dict[tuple[str, float, int], list[int]] = {}
    for i in range(len(arms)):
        key = (str(arms[i]), float(kappas[i]), int(days[i]))
        groups.setdefault(key, []).append(i)

    result: dict[str, dict[float, dict[int, dict[str, float]]]] = {}
    for (arm, kappa, day), indices in groups.items():
        arm_dict = result.setdefault(arm, {})
        kappa_dict = arm_dict.setdefault(kappa, {})
        day_metrics: dict[str, float] = {}
        for col in metric_cols:
            if col in ts:
                vals = ts[col][indices]
                finite = vals[np.isfinite(vals)]
                day_metrics[col] = (
                    float(np.mean(finite)) if len(finite) > 0 else 0.0
                )
        kappa_dict[day] = day_metrics

    return result


# ---------------------------------------------------------------------------
# Network snapshot reconstruction from events.jsonl
# ---------------------------------------------------------------------------


def reconstruct_network_snapshots(
    events_path: Path,
) -> list[dict[str, Any]]:
    """Reconstruct per-day network snapshots from events.jsonl.

    Tracks active payables, bank loans, CB loans and agent state through the
    event stream.  Returns a list of snapshot dicts suitable for the network
    animation renderer::

        [{"day": 0, "nodes": [...], "edges": [...], "defaulted": set()}, ...]

    Each node dict has ``{id, name, kind, cash}``.
    Each edge dict has ``{source, target, amount, instrument_type, contract_id}``.
    """
    events = _load_events(events_path)
    if not events:
        return []

    # ---- state tracking ----
    active_payables: dict[str, dict] = {}   # payable_id -> edge dict
    active_bank_loans: dict[str, dict] = {}  # loan_id -> edge dict
    active_cb_loans: dict[str, dict] = {}    # loan_id -> edge dict
    agents: dict[str, dict] = {}             # id -> {id, name, kind}
    defaulted_agents: set[str] = set()
    cash: dict[str, int] = {}                # agent_id -> cash amount

    def _ensure_agent(aid: str, kind: str = "firm") -> None:
        if aid not in agents:
            agents[aid] = {"id": aid, "name": aid, "kind": kind}
            cash.setdefault(aid, 0)

    def _agent_kind(aid: str) -> str:
        if aid.startswith("bank_") or aid == "bank":
            return "bank"
        if aid in ("cb", "central_bank", "CB"):
            return "central_bank"
        if aid.startswith("D") and len(aid) <= 3:
            return "dealer"
        if aid.startswith("VBT"):
            return "vbt"
        return "firm"

    # Group events by day
    day_events: dict[int, list[dict]] = {}
    max_day = 0
    for e in events:
        d = int(e.get("day", 0))
        day_events.setdefault(d, []).append(e)
        max_day = max(max_day, d)

    snapshots: list[dict[str, Any]] = []

    for day in range(max_day + 1):
        for e in day_events.get(day, []):
            kind = e.get("kind", "")

            # --- Cash tracking ---
            if kind == "CashMinted":
                aid = e.get("to", "")
                if aid:
                    _ensure_agent(aid, _agent_kind(aid))
                    cash[aid] = cash.get(aid, 0) + int(e.get("amount", 0))

            elif kind == "CashTransferred":
                frm = e.get("frm", "")
                to = e.get("to", "")
                amt = int(e.get("amount", 0))
                if frm:
                    _ensure_agent(frm, _agent_kind(frm))
                    cash[frm] = cash.get(frm, 0) - amt
                if to:
                    _ensure_agent(to, _agent_kind(to))
                    cash[to] = cash.get(to, 0) + amt

            elif kind == "CashRetired":
                frm = e.get("from", e.get("frm", ""))
                if frm:
                    cash[frm] = cash.get(frm, 0) - int(e.get("amount", 0))

            # --- Payables ---
            elif kind == "PayableCreated":
                pid = e.get("payable_id", "")
                debtor = e.get("debtor", "")
                creditor = e.get("creditor", "")
                _ensure_agent(debtor, _agent_kind(debtor))
                _ensure_agent(creditor, _agent_kind(creditor))
                active_payables[pid] = {
                    "source": creditor,
                    "target": debtor,
                    "amount": int(e.get("amount", 0)),
                    "instrument_type": "payable",
                    "contract_id": pid,
                }

            elif kind == "PayableSettled":
                pid = e.get("pid", e.get("contract_id", e.get("payable_id", "")))
                active_payables.pop(pid, None)

            elif kind == "ObligationDefaulted":
                cid = e.get("contract_id", "")
                active_payables.pop(cid, None)

            elif kind == "ObligationWrittenOff":
                cid = e.get("contract_id", "")
                active_payables.pop(cid, None)

            elif kind == "PayableRolledOver":
                pid = e.get("payable_id", "")
                if pid in active_payables:
                    active_payables[pid]["amount"] = int(e.get("amount", 0))

            elif kind == "ClaimTransferredDealer":
                pid = e.get("payable_id", "")
                new_holder = e.get("to_holder", "")
                if pid in active_payables and new_holder:
                    _ensure_agent(new_holder, _agent_kind(new_holder))
                    active_payables[pid]["source"] = new_holder

            # --- Bank loans ---
            elif kind == "BankLoanIssued":
                lid = e.get("loan_id", "")
                bank = e.get("bank", "")
                borrower = e.get("borrower", "")
                _ensure_agent(bank, "bank")
                _ensure_agent(borrower, _agent_kind(borrower))
                active_bank_loans[lid] = {
                    "source": bank,
                    "target": borrower,
                    "amount": int(e.get("amount", 0)),
                    "instrument_type": "bank_loan",
                    "contract_id": lid,
                }

            elif kind in ("BankLoanRepaid", "BankLoanDefault"):
                lid = e.get("loan_id", "")
                active_bank_loans.pop(lid, None)

            # --- CB loans ---
            elif kind in ("CBLoanCreated", "CBBackstopLoan"):
                lid = e.get("loan_id", "")
                bank_id = e.get("bank_id", "")
                if lid and bank_id:
                    _ensure_agent(bank_id, "bank")
                    _ensure_agent("CB", "central_bank")
                    active_cb_loans[lid] = {
                        "source": "CB",
                        "target": bank_id,
                        "amount": int(e.get("amount", 0)),
                        "instrument_type": "cb_loan",
                        "contract_id": lid,
                    }

            elif kind == "CBLoanRepaid":
                lid = e.get("loan_id", "")
                active_cb_loans.pop(lid, None)

            # --- Agent defaults ---
            elif kind == "AgentDefaulted":
                aid = e.get("agent", e.get("frm", ""))
                if aid:
                    defaulted_agents.add(aid)

        # Build snapshot for this day
        nodes = []
        for aid, info in agents.items():
            nodes.append({
                "id": aid,
                "name": info["name"],
                "kind": info["kind"],
                "cash": max(0, cash.get(aid, 0)),
            })

        edges = (
            list(active_payables.values())
            + list(active_bank_loans.values())
            + list(active_cb_loans.values())
        )

        snapshots.append({
            "day": day,
            "nodes": list(nodes),
            "edges": [dict(e) for e in edges],
            "defaulted": set(defaulted_agents),
        })

    return snapshots


def generate_network_animation_html(
    snapshots: list[dict[str, Any]],
    title: str = "Network Animation",
    width: int = 900,
    height: int = 750,
) -> str:
    """Generate a self-contained HTML div with animated network visualization.

    Uses Plotly.js for rendering.  The ring of agents is laid out in a circle
    with edges colored by instrument type, scaled by amount.  Defaulted agents
    are shown in red.  A slider controls the day.

    Returns an HTML string that can be embedded in a report or displayed in
    a Jupyter notebook via ``IPython.display.HTML()``.
    """
    import hashlib
    import json as _json

    uid = hashlib.md5(title.encode()).hexdigest()[:8]

    # Prepare JSON-safe snapshots (convert sets to lists)
    json_snapshots = []
    all_instrument_types: set[str] = set()
    for snap in snapshots:
        for e in snap["edges"]:
            all_instrument_types.add(e["instrument_type"])
        json_snapshots.append({
            "day": snap["day"],
            "nodes": snap["nodes"],
            "edges": snap["edges"],
            "defaulted": list(snap.get("defaulted", set())),
        })

    data_json = _json.dumps({
        "snapshots": json_snapshots,
        "instrument_types": sorted(all_instrument_types),
    })

    return f"""
<div id="net-container-{uid}" style="margin: 20px 0;">
  <h3 style="font-family: sans-serif;">{title}</h3>
  <div id="net-graph-{uid}" style="width: {width}px; height: {height}px;"></div>
  <script type="application/json" id="net-data-{uid}">{data_json}</script>
  <script>
  (function() {{
    const data = JSON.parse(document.getElementById('net-data-{uid}').textContent);
    const snapshots = data.snapshots;
    if (!snapshots.length) return;

    const instrumentColors = {{
      'payable': '#ef4444',
      'bank_loan': '#3b82f6',
      'cb_loan': '#7c3aed',
      'dealer_ticket': '#ec4899',
      'cash': '#10b981',
      'bank_deposit': '#06b6d4',
      'reserve_deposit': '#8b5cf6'
    }};

    const agentColors = {{
      'central_bank': '#7c3aed',
      'bank': '#2563eb',
      'firm': '#f59e0b',
      'dealer': '#ec4899',
      'vbt': '#a855f7',
      'household': '#10b981'
    }};

    // Stable circular layout — sort agents so ring order is consistent
    const allNodeIds = new Set();
    snapshots.forEach(s => s.nodes.forEach(n => allNodeIds.add(n.id)));
    const sortedIds = Array.from(allNodeIds).sort((a, b) => {{
      const na = parseInt(a.replace(/\\D/g, '')) || 0;
      const nb = parseInt(b.replace(/\\D/g, '')) || 0;
      if (na !== nb) return na - nb;
      return a.localeCompare(b);
    }});

    const n = sortedIds.length;
    const radius = 2.0;
    const positions = {{}};
    sortedIds.forEach((id, i) => {{
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      positions[id] = {{ x: radius * Math.cos(angle), y: radius * Math.sin(angle) }};
    }});

    function buildTraces(snapshot) {{
      const defaultedSet = new Set(snapshot.defaulted || []);

      // Aggregate edges between same (source, target, type)
      const edgeMap = {{}};
      snapshot.edges.forEach(e => {{
        const key = e.source + '|' + e.target + '|' + e.instrument_type;
        if (!edgeMap[key]) edgeMap[key] = {{ ...e, amount: 0 }};
        edgeMap[key].amount += e.amount;
      }});
      const aggEdges = Object.values(edgeMap);
      const maxAmt = Math.max(1, ...aggEdges.map(e => e.amount));

      const edgeTraces = aggEdges.map(edge => {{
        const from = positions[edge.source];
        const to = positions[edge.target];
        if (!from || !to) return null;
        const color = instrumentColors[edge.instrument_type] || '#999';
        const w = 1 + 4 * Math.sqrt(edge.amount / maxAmt);

        // Slight curve so bidirectional edges don't overlap
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const cx = (from.x + to.x) / 2 + 0.12 * dy;
        const cy = (from.y + to.y) / 2 - 0.12 * dx;

        return {{
          type: 'scatter', mode: 'lines',
          x: [from.x, cx, to.x], y: [from.y, cy, to.y],
          line: {{ color, width: w, shape: 'spline' }},
          hoverinfo: 'text',
          text: edge.source + ' \\u2192 ' + edge.target + '<br>' +
                edge.instrument_type.replace('_',' ') + ': ' + edge.amount.toLocaleString(),
          showlegend: false
        }};
      }}).filter(Boolean);

      // Node sizing by cash
      const maxCash = Math.max(1, ...snapshot.nodes.map(n => n.cash || 0));
      const nodeX = [], nodeY = [], nColors = [], nSizes = [];
      const nText = [], hText = [], nBorder = [];

      snapshot.nodes.forEach(nd => {{
        const pos = positions[nd.id];
        if (!pos) return;
        nodeX.push(pos.x); nodeY.push(pos.y);
        const dead = defaultedSet.has(nd.id);
        nColors.push(dead ? '#dc2626' : (agentColors[nd.kind] || '#999'));
        nBorder.push(dead ? '#991b1b' : '#fff');
        nSizes.push(8 + 12 * Math.sqrt((nd.cash || 0) / maxCash));
        nText.push(n <= 30 ? nd.name : (nd.kind !== 'firm' ? nd.name : ''));
        hText.push(nd.name + '<br>Type: ' + nd.kind +
                   '<br>Cash: ' + (nd.cash||0).toLocaleString() +
                   (dead ? '<br><b>DEFAULTED</b>' : ''));
      }});

      const nodeTrace = {{
        type: 'scatter', mode: 'markers' + (n <= 30 ? '+text' : ''),
        x: nodeX, y: nodeY,
        marker: {{ size: nSizes, color: nColors, line: {{ color: nBorder, width: 2 }} }},
        text: nText, textposition: 'top center', textfont: {{ size: 9 }},
        hoverinfo: 'text', hovertext: hText, showlegend: false
      }};

      return [...edgeTraces, nodeTrace];
    }}

    // Legend
    const legendTraces = data.instrument_types.map(t => ({{
      type: 'scatter', mode: 'lines', x: [null], y: [null],
      line: {{ color: instrumentColors[t] || '#999', width: 3 }},
      name: t.replace(/_/g, ' '), showlegend: true
    }}));
    const agentKinds = [...new Set(snapshots[0].nodes.map(nd => nd.kind))];
    const agentLegend = agentKinds.map(k => ({{
      type: 'scatter', mode: 'markers', x: [null], y: [null],
      marker: {{ size: 10, color: agentColors[k] || '#999' }},
      name: k.replace(/_/g, ' '), showlegend: true
    }}));

    const layout = {{
      title: {{ text: 'Day ' + snapshots[0].day, font: {{ size: 14 }} }},
      showlegend: true,
      legend: {{ x: 1.02, y: 1, font: {{ size: 10 }} }},
      hovermode: 'closest',
      xaxis: {{ visible: false, range: [-2.8, 2.8] }},
      yaxis: {{ visible: false, range: [-2.8, 2.8], scaleanchor: 'x' }},
      width: {width}, height: {height},
      margin: {{ l: 20, r: 150, t: 50, b: 70 }},
      plot_bgcolor: '#fafafa', paper_bgcolor: '#fff',
      sliders: [{{
        active: 0, pad: {{ t: 30 }},
        steps: snapshots.map((s, i) => ({{
          label: 'Day ' + s.day,
          method: 'animate',
          args: [['frame' + i], {{
            mode: 'immediate',
            transition: {{ duration: 200 }},
            frame: {{ duration: 200, redraw: true }}
          }}]
        }})),
        currentvalue: {{ prefix: 'Day: ', visible: true, xanchor: 'center' }}
      }}],
      updatemenus: [{{
        type: 'buttons', showactive: false, x: 0.1, y: 1.12, xanchor: 'right',
        buttons: [
          {{ label: '\\u25b6 Play', method: 'animate',
            args: [null, {{ fromcurrent: true,
              transition: {{ duration: 200 }},
              frame: {{ duration: 600, redraw: true }} }}] }},
          {{ label: '\\u23f8 Pause', method: 'animate',
            args: [[null], {{ mode: 'immediate',
              transition: {{ duration: 0 }},
              frame: {{ duration: 0, redraw: false }} }}] }}
        ]
      }}]
    }};

    const nLegend = legendTraces.length + agentLegend.length;
    const frames = snapshots.map((s, i) => {{
      const traces = buildTraces(s);
      for (let j = 0; j < nLegend; j++) traces.push({{ type: 'scatter', x: [null], y: [null] }});
      return {{ name: 'frame' + i, data: traces, layout: {{ title: {{ text: 'Day ' + s.day }} }} }};
    }});

    const initTraces = [...buildTraces(snapshots[0]), ...legendTraces, ...agentLegend];
    Plotly.newPlot('net-graph-{uid}', initTraces, layout, {{ responsive: true }})
      .then(() => Plotly.addFrames('net-graph-{uid}', frames));
  }})();
  </script>
</div>
"""
