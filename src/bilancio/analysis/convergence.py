"""Multi-dimensional convergence analysis for bilancio simulations.

Evaluates whether a simulation has reached equilibrium by tracking
multiple economic channels simultaneously: clearing rate, default rate,
market prices, agent beliefs, credit creation, and contagion.

Each channel is extracted from the simulation outputs and tested for
convergence independently.  The composite result aggregates across all
active channels, reporting an overall convergence day and a quality
score reflecting how many channels stabilised and how tightly they
converged.

Usage::

    from bilancio.analysis.convergence import evaluate_convergence

    result = evaluate_convergence(day_metrics, events, estimates)
    print(result.converged, result.convergence_day, result.quality)
    for name, ch in result.channels.items():
        print(f"  {name}: converged={ch.converged} day={ch.convergence_day}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenceConfig:
    """Tuning knobs for multi-channel convergence detection.

    Attributes:
        window: Number of consecutive days the absolute delta must stay
            below epsilon for a channel to be declared converged.
        epsilon_clearing: Convergence threshold for the clearing-rate
            (phi_t) channel.
        epsilon_default: Convergence threshold for the default-rate
            (delta_t) channel.
        epsilon_price: Convergence threshold for the average price-ratio
            channel.
        epsilon_belief: Convergence threshold for the mean-belief
            channel.
        epsilon_credit: Convergence threshold for the net credit-impulse
            channel.
        enabled_channels: If set, only these channel names are evaluated.
            ``None`` means auto-detect (include any channel that returns
            a non-empty trajectory).
    """

    window: int = 3
    epsilon_clearing: float = 0.01
    epsilon_default: float = 0.005
    epsilon_price: float = 0.02
    epsilon_belief: float = 0.01
    epsilon_credit: float = 0.05
    enabled_channels: set[str] | None = None


@dataclass(frozen=True)
class ChannelSnapshot:
    """A single observation on one convergence channel.

    Attributes:
        day: Simulation day.
        value: Channel value on that day (e.g. phi_t, average price
            ratio, mean belief).
        delta: Absolute change from the previous day, or ``None`` for
            the first observation.
    """

    day: int
    value: float
    delta: float | None


@dataclass(frozen=True)
class ChannelResult:
    """Convergence verdict for a single channel.

    Attributes:
        name: Human-readable channel identifier (e.g. ``"clearing"``).
        converged: Whether the channel met the sustained-window
            criterion.
        convergence_day: First day of the sustained convergence window,
            or ``None`` if the channel did not converge.
        final_value: Last observed value in the trajectory.
        trajectory: Full list of :class:`ChannelSnapshot` objects.
    """

    name: str
    converged: bool
    convergence_day: int | None
    final_value: float
    trajectory: list[ChannelSnapshot]


@dataclass(frozen=True)
class ConvergenceResult:
    """Composite convergence verdict across all active channels.

    Attributes:
        converged: ``True`` only if every active channel converged.
        convergence_day: Day at which *all* channels have converged
            (i.e. the maximum of individual convergence days), or
            ``None`` if any channel did not converge.
        quality: A score in [0, 1] combining the fraction of channels
            that converged with the tightness of their convergence.
        channels: Per-channel results keyed by channel name.
        active_channels: Number of channels that were evaluated.
        converged_channels: Number of channels that converged.
    """

    converged: bool
    convergence_day: int | None
    quality: float
    channels: dict[str, ChannelResult]
    active_channels: int
    converged_channels: int


# ---------------------------------------------------------------------------
# Channel epsilon mapping
# ---------------------------------------------------------------------------

_CHANNEL_EPSILONS: dict[str, str] = {
    "clearing": "epsilon_clearing",
    "default": "epsilon_default",
    "price": "epsilon_price",
    "belief": "epsilon_belief",
    "credit": "epsilon_credit",
    "contagion": "epsilon_default",  # reuse default threshold
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(val: Any) -> float:
    """Best-effort conversion of Decimal / numeric to plain float."""
    if isinstance(val, float):
        return val
    if isinstance(val, Decimal):
        return float(val)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError, ArithmeticError):
        return 0.0


def _build_trajectory(day_values: list[tuple[int, float]]) -> list[ChannelSnapshot]:
    """Convert ``(day, value)`` pairs into a trajectory with deltas.

    The input must be sorted by day.  The first snapshot gets
    ``delta=None``; subsequent ones get ``abs(value - prev)``.
    """
    trajectory: list[ChannelSnapshot] = []
    prev: float | None = None
    for day, value in day_values:
        delta: float | None = None
        if prev is not None:
            delta = abs(value - prev)
        trajectory.append(ChannelSnapshot(day=day, value=value, delta=delta))
        prev = value
    return trajectory


# ---------------------------------------------------------------------------
# Channel extraction functions
# ---------------------------------------------------------------------------


def _extract_clearing_channel(
    day_metrics: list[dict[str, Any]],
) -> list[ChannelSnapshot]:
    """Extract the clearing-rate (phi_t) trajectory from day metrics.

    Each day-metric dict is expected to contain ``"day"`` (int) and
    ``"phi_t"`` (Decimal or float).  Days where phi_t is ``None`` are
    skipped.
    """
    pairs: list[tuple[int, float]] = []
    for row in day_metrics:
        phi = row.get("phi_t")
        if phi is None:
            continue
        pairs.append((int(row["day"]), _to_float(phi)))
    pairs.sort(key=lambda p: p[0])
    return _build_trajectory(pairs)


def _extract_default_channel(
    day_metrics: list[dict[str, Any]],
) -> list[ChannelSnapshot]:
    """Extract the default-rate (delta_t) trajectory from day metrics.

    Each day-metric dict is expected to contain ``"day"`` (int) and
    ``"delta_t"`` (Decimal or float).  Days where delta_t is ``None``
    are skipped.
    """
    pairs: list[tuple[int, float]] = []
    for row in day_metrics:
        delta = row.get("delta_t")
        if delta is None:
            continue
        pairs.append((int(row["day"]), _to_float(delta)))
    pairs.sort(key=lambda p: p[0])
    return _build_trajectory(pairs)


def _extract_price_channel(
    events: list[dict[str, Any]],
) -> list[ChannelSnapshot]:
    """Extract the average price-ratio trajectory from trade events.

    Uses :func:`bilancio.analysis.pricing_analysis.average_price_ratio_by_day`
    to compute the all-trade average price/face ratio per day.
    """
    from bilancio.analysis.pricing_analysis import average_price_ratio_by_day

    avg_by_day = average_price_ratio_by_day(events)
    pairs: list[tuple[int, float]] = []
    for day in sorted(avg_by_day.keys()):
        all_avg = avg_by_day[day].get("all_avg")
        if all_avg is not None:
            pairs.append((day, _to_float(all_avg)))
    return _build_trajectory(pairs)


def _extract_belief_channel(
    estimates: list[Any],
) -> list[ChannelSnapshot]:
    """Extract the mean-belief trajectory from the estimate log.

    Uses :func:`bilancio.analysis.beliefs.belief_trajectory` to obtain
    :class:`~bilancio.analysis.beliefs.BeliefPoint` objects, then
    averages all belief values that share the same day to produce one
    value per day.
    """
    from bilancio.analysis.beliefs import belief_trajectory

    points = belief_trajectory(estimates)
    if not points:
        return []

    # Group values by day, then take the mean for each day.
    by_day: dict[int, list[float]] = defaultdict(list)
    for bp in points:
        by_day[bp.day].append(_to_float(bp.value))

    pairs: list[tuple[int, float]] = []
    for day in sorted(by_day.keys()):
        vals = by_day[day]
        pairs.append((day, sum(vals) / len(vals)))
    return _build_trajectory(pairs)


def _extract_credit_channel(
    events: list[dict[str, Any]],
) -> list[ChannelSnapshot]:
    """Extract the net credit-impulse trajectory.

    Computes the daily net impulse (creation - destruction) as a
    fraction of total creation.  Days with zero creation are skipped.

    Uses :func:`bilancio.analysis.credit_creation.credit_creation_by_day`
    and :func:`bilancio.analysis.credit_creation.credit_destruction_by_day`.
    """
    from bilancio.analysis.credit_creation import (
        credit_creation_by_day,
        credit_destruction_by_day,
    )

    created_by_day = credit_creation_by_day(events)
    destroyed_by_day = credit_destruction_by_day(events)

    all_days = sorted(set(created_by_day.keys()) | set(destroyed_by_day.keys()))

    # Total creation across all days (for normalisation).
    total_created = sum(
        (sum(v.values(), Decimal(0)) for v in created_by_day.values()),
        Decimal(0),
    )
    if total_created == 0:
        return []

    pairs: list[tuple[int, float]] = []
    for day in all_days:
        day_created = sum(
            created_by_day.get(day, {}).values(), Decimal(0)
        )
        day_destroyed = sum(
            destroyed_by_day.get(day, {}).values(), Decimal(0)
        )
        net = day_created - day_destroyed
        ratio = _to_float(net / total_created)
        pairs.append((day, ratio))

    return _build_trajectory(pairs)


def _extract_contagion_channel(
    events: list[dict[str, Any]],
) -> list[ChannelSnapshot]:
    """Extract secondary-default counts per day.

    Uses :func:`bilancio.analysis.contagion.contagion_by_day` to obtain
    daily primary/secondary default counts and reports the secondary
    count as the channel value.
    """
    from bilancio.analysis.contagion import contagion_by_day

    by_day = contagion_by_day(events)
    if not by_day:
        return []

    pairs: list[tuple[int, float]] = []
    for day in sorted(by_day.keys()):
        secondary = by_day[day].get("secondary", 0)
        pairs.append((day, float(secondary)))
    return _build_trajectory(pairs)


# ---------------------------------------------------------------------------
# Core convergence check
# ---------------------------------------------------------------------------


def _check_channel_convergence(
    trajectory: list[ChannelSnapshot],
    epsilon: float,
    window: int,
) -> tuple[bool, int | None]:
    """Determine whether a channel's trajectory has converged.

    Convergence is declared when ``|delta| < epsilon`` for *window*
    consecutive days **and** that condition is sustained through to the
    end of the trajectory (i.e. no subsequent day violates it).

    Args:
        trajectory: Ordered list of :class:`ChannelSnapshot` objects.
        epsilon: Maximum absolute day-over-day change to count as
            "stable".
        window: Number of consecutive stable days required.

    Returns:
        A ``(converged, convergence_day)`` tuple.  *convergence_day* is
        the day of the first snapshot in the sustained stable window, or
        ``None`` if the channel never converged.
    """
    if len(trajectory) < window + 1:
        # Not enough data points (need at least window snapshots with deltas).
        return False, None

    # Build a boolean mask: True where |delta| < epsilon.
    # The first snapshot has delta=None, so we skip it.
    stable = [
        (snap.delta is not None and abs(snap.delta) < epsilon)
        for snap in trajectory
    ]

    # We need the last `n` entries (from some start point to the end)
    # to ALL be stable, and that run must be at least `window` long.
    # Walk backwards to find the longest stable suffix.
    suffix_start: int | None = None
    for i in range(len(stable) - 1, -1, -1):
        if stable[i]:
            suffix_start = i
        else:
            break

    if suffix_start is None:
        return False, None

    suffix_length = len(stable) - suffix_start
    if suffix_length < window:
        return False, None

    convergence_day = trajectory[suffix_start].day
    return True, convergence_day


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_convergence(
    day_metrics: list[dict[str, Any]],
    events: list[dict[str, Any]],
    estimates: list[Any] | None = None,
    *,
    config: ConvergenceConfig | None = None,
) -> ConvergenceResult:
    """Evaluate multi-channel convergence of a simulation run.

    Extracts each convergence channel from the simulation outputs,
    checks each for sustained stability, and returns a composite
    verdict.

    Args:
        day_metrics: Per-day metric rows as produced by
            :func:`bilancio.analysis.report.compute_day_metrics`.
            Expected keys include ``day``, ``phi_t``, ``delta_t``.
        events: Raw event log (list of dicts), e.g. from
            ``events.jsonl``.
        estimates: Optional estimate log (list of
            :class:`~bilancio.information.estimates.Estimate` objects).
            If ``None``, the belief channel is skipped.
        config: Convergence thresholds and channel selection.  Defaults
            to :class:`ConvergenceConfig` with factory settings.

    Returns:
        A :class:`ConvergenceResult` summarising whether (and when) the
        simulation reached equilibrium across all monitored channels.
    """
    if config is None:
        config = ConvergenceConfig()

    # ------------------------------------------------------------------
    # 1. Extract channel trajectories
    # ------------------------------------------------------------------
    raw_channels: dict[str, list[ChannelSnapshot]] = {
        "clearing": _extract_clearing_channel(day_metrics),
        "default": _extract_default_channel(day_metrics),
        "price": _extract_price_channel(events),
        "credit": _extract_credit_channel(events),
        "contagion": _extract_contagion_channel(events),
    }
    if estimates is not None:
        raw_channels["belief"] = _extract_belief_channel(estimates)

    # ------------------------------------------------------------------
    # 2. Filter to active channels
    # ------------------------------------------------------------------
    active: dict[str, list[ChannelSnapshot]] = {}
    for name, traj in raw_channels.items():
        # Skip channels with empty trajectories (auto-detection).
        if not traj:
            continue
        # Honour explicit channel selection if configured.
        if config.enabled_channels is not None and name not in config.enabled_channels:
            continue
        active[name] = traj

    if not active:
        return ConvergenceResult(
            converged=True,
            convergence_day=None,
            quality=1.0,
            channels={},
            active_channels=0,
            converged_channels=0,
        )

    # ------------------------------------------------------------------
    # 3. Evaluate each channel
    # ------------------------------------------------------------------
    channel_results: dict[str, ChannelResult] = {}
    for name, traj in active.items():
        eps_attr = _CHANNEL_EPSILONS.get(name, "epsilon_default")
        epsilon = getattr(config, eps_attr, config.epsilon_default)
        converged, conv_day = _check_channel_convergence(traj, epsilon, config.window)
        channel_results[name] = ChannelResult(
            name=name,
            converged=converged,
            convergence_day=conv_day,
            final_value=traj[-1].value,
            trajectory=traj,
        )

    # ------------------------------------------------------------------
    # 4. Composite verdict
    # ------------------------------------------------------------------
    n_active = len(channel_results)
    n_converged = sum(1 for cr in channel_results.values() if cr.converged)

    # Composite convergence day = latest channel convergence day.
    if n_converged == n_active:
        conv_days = [
            cr.convergence_day
            for cr in channel_results.values()
            if cr.convergence_day is not None
        ]
        composite_day: int | None = max(conv_days) if conv_days else None
        all_converged = True
    else:
        composite_day = None
        all_converged = False

    # Quality score ------------------------------------------------
    # For converged channels, stability = 1 - mean(|delta|)/epsilon
    # over the final *window* snapshots.  Unconverged channels
    # contribute 0.
    stability_scores: list[float] = []
    for name, cr in channel_results.items():
        if cr.converged:
            eps_attr = _CHANNEL_EPSILONS.get(name, "epsilon_default")
            epsilon = getattr(config, eps_attr, config.epsilon_default)
            # Gather |delta| over the final `window` snapshots.
            tail = cr.trajectory[-config.window:]
            deltas = [abs(s.delta) for s in tail if s.delta is not None]
            if deltas and epsilon > 0:
                mean_delta = sum(deltas) / len(deltas)
                stability_scores.append(max(0.0, 1.0 - mean_delta / epsilon))
            else:
                stability_scores.append(1.0)
        else:
            stability_scores.append(0.0)

    mean_stability = (
        sum(stability_scores) / len(stability_scores)
        if stability_scores
        else 0.0
    )
    quality = (n_converged / n_active) * mean_stability

    return ConvergenceResult(
        converged=all_converged,
        convergence_day=composite_day,
        quality=quality,
        channels=channel_results,
        active_channels=n_active,
        converged_channels=n_converged,
    )
