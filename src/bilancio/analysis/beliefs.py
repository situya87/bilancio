"""Belief trajectory and calibration analysis.

Analyses the estimate log produced during simulation to answer:
- How did beliefs evolve over time?
- How well-calibrated were default probability estimates?
- What information sources were used?
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

from bilancio.information.estimates import Estimate


@dataclass(frozen=True)
class BeliefPoint:
    """A single point in a belief trajectory."""
    day: int
    value: Decimal
    method: str
    estimator_id: str


def belief_trajectory(
    estimates: Sequence[Estimate],
    target_id: str | None = None,
    estimator_id: str | None = None,
    method: str | None = None,
) -> list[BeliefPoint]:
    """Extract a time series of beliefs, optionally filtered.

    Returns BeliefPoints sorted by day, then by position in the
    original sequence (preserving insertion order within a day).

    Args:
        estimates: Sequence of Estimate objects (e.g. system.state.estimate_log)
        target_id: Filter to estimates about this target (e.g. agent ID)
        estimator_id: Filter to estimates from this estimator
        method: Filter to estimates using this method

    Returns:
        List of BeliefPoint objects sorted by day.
    """
    points: list[BeliefPoint] = []
    for est in estimates:
        if target_id is not None and est.target_id != target_id:
            continue
        if estimator_id is not None and est.estimator_id != estimator_id:
            continue
        if method is not None and est.method != method:
            continue
        points.append(BeliefPoint(
            day=est.estimation_day,
            value=est.value,
            method=est.method,
            estimator_id=est.estimator_id,
        ))
    points.sort(key=lambda p: p.day)
    return points


@dataclass
class CalibrationBucket:
    """A bucket for calibration analysis.

    Groups estimates by predicted probability range and compares
    to actual default outcomes.
    """
    predicted_low: Decimal
    predicted_high: Decimal
    count: int = 0
    actual_defaults: int = 0

    @property
    def predicted_mean(self) -> Decimal:
        return (self.predicted_low + self.predicted_high) / 2

    @property
    def actual_rate(self) -> Decimal:
        if self.count == 0:
            return Decimal(0)
        return Decimal(self.actual_defaults) / Decimal(self.count)


def belief_vs_reality(
    estimates: Sequence[Estimate],
    defaulted_agent_ids: set[str],
    n_buckets: int = 5,
) -> list[CalibrationBucket]:
    """Compare predicted default probabilities to actual outcomes.

    Takes the *last* default probability estimate for each target agent
    and groups them into calibration buckets. Compares predicted
    probabilities to whether the agent actually defaulted.

    Only includes estimates with target_type == "agent" and methods
    that produce default probability estimates.

    Args:
        estimates: Sequence of Estimate objects
        defaulted_agent_ids: Set of agent IDs that actually defaulted
        n_buckets: Number of calibration buckets (default 5)

    Returns:
        List of CalibrationBucket objects sorted by predicted probability.
    """
    # Take the latest estimate per target agent
    latest: dict[str, Estimate] = {}
    for est in estimates:
        if est.target_type != "agent":
            continue
        existing = latest.get(est.target_id)
        if existing is None or est.estimation_day >= existing.estimation_day:
            latest[est.target_id] = est

    if not latest:
        return []

    # Build calibration buckets
    bucket_width = Decimal(1) / Decimal(n_buckets)
    buckets = []
    for i in range(n_buckets):
        low = bucket_width * i
        high = bucket_width * (i + 1)
        buckets.append(CalibrationBucket(predicted_low=low, predicted_high=high))

    # Assign each agent to a bucket
    for agent_id, est in latest.items():
        p = min(est.value, Decimal("0.9999"))  # Avoid index overflow
        bucket_idx = min(int(p / bucket_width), n_buckets - 1)
        buckets[bucket_idx].count += 1
        if agent_id in defaulted_agent_ids:
            buckets[bucket_idx].actual_defaults += 1

    return buckets


@dataclass
class EstimateSummary:
    """Summary statistics for a group of estimates."""
    count: int = 0
    methods: dict[str, int] = field(default_factory=dict)
    estimators: dict[str, int] = field(default_factory=dict)
    target_types: dict[str, int] = field(default_factory=dict)
    day_range: tuple[int, int] | None = None
    value_range: tuple[Decimal, Decimal] | None = None


def estimate_summary(estimates: Sequence[Estimate]) -> EstimateSummary:
    """Compute summary statistics over a collection of estimates.

    Args:
        estimates: Sequence of Estimate objects

    Returns:
        EstimateSummary with aggregate counts and ranges.
    """
    if not estimates:
        return EstimateSummary()

    methods: dict[str, int] = defaultdict(int)
    estimators: dict[str, int] = defaultdict(int)
    target_types: dict[str, int] = defaultdict(int)
    min_day = estimates[0].estimation_day
    max_day = estimates[0].estimation_day
    min_val = estimates[0].value
    max_val = estimates[0].value

    for est in estimates:
        methods[est.method] += 1
        estimators[est.estimator_id] += 1
        target_types[est.target_type] += 1
        if est.estimation_day < min_day:
            min_day = est.estimation_day
        if est.estimation_day > max_day:
            max_day = est.estimation_day
        if est.value < min_val:
            min_val = est.value
        if est.value > max_val:
            max_val = est.value

    return EstimateSummary(
        count=len(estimates),
        methods=dict(methods),
        estimators=dict(estimators),
        target_types=dict(target_types),
        day_range=(min_day, max_day),
        value_range=(min_val, max_val),
    )


def export_estimates_jsonl(
    estimates: Sequence[Estimate],
    path: str,
) -> int:
    """Export estimates to a JSONL file.

    Each line is a JSON object with all Estimate fields serialized.
    Decimal values are converted to strings to preserve precision.

    Args:
        estimates: Sequence of Estimate objects
        path: File path for JSONL output

    Returns:
        Number of estimates written.
    """
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with p.open("w") as f:
        for est in estimates:
            record = {
                "value": str(est.value),
                "estimator_id": est.estimator_id,
                "target_id": est.target_id,
                "target_type": est.target_type,
                "estimation_day": est.estimation_day,
                "method": est.method,
                "inputs": _serialize_dict(est.inputs),
                "metadata": _serialize_dict(est.metadata),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    return count


def _serialize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Serialize a dict, converting Decimals to strings."""
    result = {}
    for k, v in d.items():
        if isinstance(v, Decimal):
            result[k] = str(v)
        elif isinstance(v, dict):
            result[k] = _serialize_dict(v)
        else:
            result[k] = v
    return result
