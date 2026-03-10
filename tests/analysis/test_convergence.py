"""Tests for multi-channel convergence analysis (bilancio.analysis.convergence).

Covers:
1. ChannelSnapshot — basic construction
2. _build_trajectory — helper producing snapshots with deltas
3. _check_channel_convergence — sustained-window convergence logic
4. _extract_clearing_channel — phi_t extraction from day_metrics
5. _extract_default_channel — delta_t extraction from day_metrics
6. evaluate_convergence — integration tests (composite verdict)
7. Quality score — edge cases for the [0, 1] quality metric
"""

from decimal import Decimal

from bilancio.analysis.convergence import (
    ChannelResult,
    ChannelSnapshot,
    ConvergenceConfig,
    ConvergenceResult,
    _build_trajectory,
    _check_channel_convergence,
    _extract_clearing_channel,
    _extract_default_channel,
    evaluate_convergence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_day_metrics(
    phi_values: list[float],
    delta_values: list[float] | None = None,
) -> list[dict]:
    """Build synthetic day-metric dicts with phi_t and optional delta_t."""
    metrics: list[dict] = []
    for i, phi in enumerate(phi_values):
        row: dict = {"day": i + 1, "phi_t": Decimal(str(phi))}
        if delta_values is not None:
            row["delta_t"] = Decimal(str(delta_values[i]))
        metrics.append(row)
    return metrics


def _constant_phi(value: float, n: int) -> list[float]:
    """Return *n* copies of *value* — a flat trajectory."""
    return [value] * n


def _converging_phi(start: float, end: float, n: int) -> list[float]:
    """Linearly interpolate from *start* toward *end* over *n* days."""
    if n <= 1:
        return [start]
    step = (end - start) / (n - 1)
    return [round(start + i * step, 8) for i in range(n)]


def _oscillating(center: float, amplitude: float, n: int) -> list[float]:
    """Alternate above/below *center* by *amplitude*."""
    return [
        round(center + amplitude * (1 if i % 2 == 0 else -1), 8) for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. ChannelSnapshot
# ---------------------------------------------------------------------------


class TestChannelSnapshot:
    def test_basic_construction(self):
        snap = ChannelSnapshot(day=5, value=0.85, delta=0.01)
        assert snap.day == 5
        assert snap.value == 0.85
        assert snap.delta == 0.01

    def test_first_snapshot_delta_none(self):
        snap = ChannelSnapshot(day=1, value=0.50, delta=None)
        assert snap.delta is None

    def test_frozen(self):
        snap = ChannelSnapshot(day=1, value=0.5, delta=None)
        try:
            snap.value = 0.9  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# 2. _build_trajectory
# ---------------------------------------------------------------------------


class TestBuildTrajectory:
    def test_empty_input(self):
        assert _build_trajectory([]) == []

    def test_single_point(self):
        traj = _build_trajectory([(1, 0.5)])
        assert len(traj) == 1
        assert traj[0].day == 1
        assert traj[0].value == 0.5
        assert traj[0].delta is None

    def test_two_points_computes_delta(self):
        traj = _build_trajectory([(1, 0.5), (2, 0.8)])
        assert len(traj) == 2
        assert traj[0].delta is None
        assert traj[1].delta is not None
        assert abs(traj[1].delta - 0.3) < 1e-10

    def test_multiple_points(self):
        traj = _build_trajectory([(1, 1.0), (2, 1.5), (3, 1.2)])
        assert len(traj) == 3
        assert traj[0].delta is None
        assert abs(traj[1].delta - 0.5) < 1e-10  # |1.5 - 1.0|
        assert abs(traj[2].delta - 0.3) < 1e-10  # |1.2 - 1.5|

    def test_constant_values_zero_delta(self):
        traj = _build_trajectory([(1, 0.7), (2, 0.7), (3, 0.7)])
        for snap in traj[1:]:
            assert snap.delta == 0.0

    def test_preserves_day_numbers(self):
        traj = _build_trajectory([(5, 0.1), (10, 0.2), (15, 0.3)])
        assert [s.day for s in traj] == [5, 10, 15]


# ---------------------------------------------------------------------------
# 3. _check_channel_convergence
# ---------------------------------------------------------------------------


class TestCheckChannelConvergence:
    def test_converges_at_known_day(self):
        # 3 noisy points then 4 flat points.  window=3, epsilon=0.01.
        # Deltas: None, 0.1, 0.05, 0.25, 0.0, 0.0, 0.0
        # Stable mask: F, F, F, F, T, T, T
        # Stable suffix starts at index 4 (day 5), length 3 == window.
        values = [0.5, 0.6, 0.55, 0.80, 0.80, 0.80, 0.80]
        traj = _build_trajectory([(i + 1, v) for i, v in enumerate(values)])
        converged, day = _check_channel_convergence(traj, epsilon=0.01, window=3)
        assert converged is True
        assert day == 5

    def test_does_not_converge_when_oscillating(self):
        values = _oscillating(center=0.5, amplitude=0.05, n=20)
        traj = _build_trajectory([(i + 1, v) for i, v in enumerate(values)])
        converged, day = _check_channel_convergence(traj, epsilon=0.01, window=3)
        assert converged is False
        assert day is None

    def test_insufficient_data(self):
        # window=3 requires at least 4 data points (window+1).
        values = [0.5, 0.5, 0.5]  # only 3 points
        traj = _build_trajectory([(i + 1, v) for i, v in enumerate(values)])
        converged, day = _check_channel_convergence(traj, epsilon=0.01, window=3)
        assert converged is False
        assert day is None

    def test_exactly_window_plus_one_converges(self):
        # Minimum data: window+1 = 4 points, all constant.
        values = [0.5, 0.5, 0.5, 0.5]
        traj = _build_trajectory([(i + 1, v) for i, v in enumerate(values)])
        converged, day = _check_channel_convergence(traj, epsilon=0.01, window=3)
        assert converged is True

    def test_converges_on_last_possible_window(self):
        # 8 points: first 3 noisy, last 5 flat.
        # Deltas: None, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0
        # Stable mask: F, F, F, F, T, T, T, T
        # Stable suffix starts at index 4 (day 5), length 4 == window.
        noisy = [0.1, 0.5, 0.9]
        flat = [0.80, 0.80, 0.80, 0.80, 0.80]
        values = noisy + flat
        traj = _build_trajectory([(i + 1, v) for i, v in enumerate(values)])
        converged, day = _check_channel_convergence(traj, epsilon=0.01, window=4)
        assert converged is True
        assert day == 5

    def test_late_violation_breaks_convergence(self):
        # Looks converged early but the last point breaks it.
        values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.9]
        traj = _build_trajectory([(i + 1, v) for i, v in enumerate(values)])
        converged, day = _check_channel_convergence(traj, epsilon=0.01, window=3)
        assert converged is False

    def test_large_epsilon_makes_noisy_data_converge(self):
        values = _oscillating(center=0.5, amplitude=0.05, n=10)
        traj = _build_trajectory([(i + 1, v) for i, v in enumerate(values)])
        # With a large epsilon, even the oscillation is "stable".
        converged, _ = _check_channel_convergence(traj, epsilon=1.0, window=3)
        assert converged is True


# ---------------------------------------------------------------------------
# 4. _extract_clearing_channel
# ---------------------------------------------------------------------------


class TestExtractClearingChannel:
    def test_basic_extraction(self):
        metrics = _make_day_metrics([0.5, 0.6, 0.7])
        traj = _extract_clearing_channel(metrics)
        assert len(traj) == 3
        assert traj[0].value == 0.5
        assert traj[1].value == 0.6
        assert traj[2].value == 0.7

    def test_skips_none_phi(self):
        metrics = [
            {"day": 1, "phi_t": Decimal("0.5")},
            {"day": 2, "phi_t": None},
            {"day": 3, "phi_t": Decimal("0.7")},
        ]
        traj = _extract_clearing_channel(metrics)
        assert len(traj) == 2
        assert traj[0].day == 1
        assert traj[1].day == 3

    def test_empty_input(self):
        assert _extract_clearing_channel([]) == []

    def test_sorts_by_day(self):
        metrics = [
            {"day": 3, "phi_t": Decimal("0.7")},
            {"day": 1, "phi_t": Decimal("0.5")},
            {"day": 2, "phi_t": Decimal("0.6")},
        ]
        traj = _extract_clearing_channel(metrics)
        assert [s.day for s in traj] == [1, 2, 3]

    def test_deltas_computed_correctly(self):
        metrics = _make_day_metrics([0.5, 0.8, 0.75])
        traj = _extract_clearing_channel(metrics)
        assert traj[0].delta is None
        assert abs(traj[1].delta - 0.3) < 1e-10
        assert abs(traj[2].delta - 0.05) < 1e-10


# ---------------------------------------------------------------------------
# 5. _extract_default_channel
# ---------------------------------------------------------------------------


class TestExtractDefaultChannel:
    def test_basic_extraction(self):
        metrics = _make_day_metrics([0.5, 0.6, 0.7], [0.1, 0.2, 0.15])
        traj = _extract_default_channel(metrics)
        assert len(traj) == 3
        assert traj[0].value == 0.1
        assert traj[1].value == 0.2
        assert traj[2].value == 0.15

    def test_skips_none_delta_t(self):
        metrics = [
            {"day": 1, "delta_t": Decimal("0.1")},
            {"day": 2},  # no delta_t key
            {"day": 3, "delta_t": Decimal("0.15")},
        ]
        traj = _extract_default_channel(metrics)
        assert len(traj) == 2
        assert traj[0].day == 1
        assert traj[1].day == 3

    def test_empty_input(self):
        assert _extract_default_channel([]) == []

    def test_no_delta_t_anywhere(self):
        metrics = _make_day_metrics([0.5, 0.6])  # no delta_values
        traj = _extract_default_channel(metrics)
        assert traj == []

    def test_sorts_by_day(self):
        metrics = [
            {"day": 3, "delta_t": Decimal("0.3")},
            {"day": 1, "delta_t": Decimal("0.1")},
        ]
        traj = _extract_default_channel(metrics)
        assert [s.day for s in traj] == [1, 3]


# ---------------------------------------------------------------------------
# 6. evaluate_convergence — integration tests
# ---------------------------------------------------------------------------


class TestEvaluateConvergence:
    def test_all_channels_converge(self):
        # Clearing and default both flat for 10 days.
        n = 10
        phi = _constant_phi(0.90, n)
        delta = _constant_phi(0.05, n)
        metrics = _make_day_metrics(phi, delta)
        config = ConvergenceConfig(
            window=3,
            enabled_channels={"clearing", "default"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert result.converged is True
        assert result.quality > 0.9
        assert result.active_channels == 2
        assert result.converged_channels == 2
        assert result.convergence_day is not None

    def test_passive_run_auto_detects_two_channels(self):
        # No events (no trades, no credit, no contagion) -> only clearing
        # and default channels active.
        n = 10
        phi = _constant_phi(0.85, n)
        delta = _constant_phi(0.10, n)
        metrics = _make_day_metrics(phi, delta)
        # No enabled_channels -> auto-detect.
        result = evaluate_convergence(metrics, events=[])
        assert result.active_channels == 2
        assert "clearing" in result.channels
        assert "default" in result.channels
        assert result.converged is True

    def test_no_data_returns_converged_with_quality_1(self):
        result = evaluate_convergence(day_metrics=[], events=[])
        assert result.converged is True
        assert result.quality == 1.0
        assert result.active_channels == 0
        assert result.converged_channels == 0
        assert result.convergence_day is None

    def test_quality_zero_when_nothing_converges(self):
        # Oscillating clearing and default — neither converges.
        n = 20
        phi = _oscillating(0.5, 0.1, n)
        delta = _oscillating(0.3, 0.1, n)
        metrics = _make_day_metrics(phi, delta)
        config = ConvergenceConfig(
            window=3,
            epsilon_clearing=0.01,
            epsilon_default=0.005,
            enabled_channels={"clearing", "default"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert result.converged is False
        assert result.quality == 0.0
        assert result.converged_channels == 0

    def test_enabled_channels_filter(self):
        n = 10
        phi = _constant_phi(0.90, n)
        delta = _oscillating(0.3, 0.1, n)  # does NOT converge
        metrics = _make_day_metrics(phi, delta)

        # Only request clearing — should converge.
        config = ConvergenceConfig(
            window=3,
            enabled_channels={"clearing"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert result.converged is True
        assert result.active_channels == 1
        assert "clearing" in result.channels
        assert "default" not in result.channels

    def test_convergence_day_is_max_of_channels(self):
        # Clearing converges early (flat from day 1), default converges
        # late (noisy then flat).
        n = 12
        phi = _constant_phi(0.90, n)
        # Default: noisy first 5 days then flat.
        delta = [0.1, 0.3, 0.05, 0.25, 0.08] + _constant_phi(0.10, 7)
        metrics = _make_day_metrics(phi, delta)
        config = ConvergenceConfig(
            window=3,
            epsilon_clearing=0.01,
            epsilon_default=0.01,
            enabled_channels={"clearing", "default"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert result.converged is True
        # Clearing converges at day 2 (first stable delta), default at
        # day 6 or later.  Composite day should be the later one.
        clearing_day = result.channels["clearing"].convergence_day
        default_day = result.channels["default"].convergence_day
        assert clearing_day is not None
        assert default_day is not None
        assert result.convergence_day == max(clearing_day, default_day)
        assert default_day > clearing_day

    def test_partial_convergence(self):
        # Clearing converges, default does not.
        n = 20
        phi = _constant_phi(0.90, n)
        delta = _oscillating(0.3, 0.1, n)
        metrics = _make_day_metrics(phi, delta)
        config = ConvergenceConfig(
            window=3,
            epsilon_clearing=0.01,
            epsilon_default=0.005,
            enabled_channels={"clearing", "default"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert result.converged is False
        assert result.convergence_day is None
        assert result.converged_channels == 1
        assert result.active_channels == 2
        # Quality should be between 0 and 1.
        assert 0.0 < result.quality < 1.0


# ---------------------------------------------------------------------------
# 7. Quality score
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_perfect_quality_when_all_converge_zero_delta(self):
        # All channels flat from day 1 -> delta=0 in final window -> quality=1.
        n = 10
        phi = _constant_phi(0.90, n)
        delta = _constant_phi(0.05, n)
        metrics = _make_day_metrics(phi, delta)
        config = ConvergenceConfig(
            window=3,
            enabled_channels={"clearing", "default"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert result.quality == 1.0

    def test_quality_between_zero_and_one_for_partial(self):
        # One channel converges, the other does not.
        n = 20
        phi = _constant_phi(0.90, n)
        delta = _oscillating(0.3, 0.1, n)
        metrics = _make_day_metrics(phi, delta)
        config = ConvergenceConfig(
            window=3,
            epsilon_clearing=0.01,
            epsilon_default=0.005,
            enabled_channels={"clearing", "default"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert 0.0 < result.quality < 1.0

    def test_quality_zero_when_nothing_converges(self):
        n = 20
        phi = _oscillating(0.5, 0.1, n)
        delta = _oscillating(0.3, 0.1, n)
        metrics = _make_day_metrics(phi, delta)
        config = ConvergenceConfig(
            window=3,
            epsilon_clearing=0.01,
            epsilon_default=0.005,
            enabled_channels={"clearing", "default"},
        )
        result = evaluate_convergence(metrics, events=[], config=config)
        assert result.quality == 0.0

    def test_quality_decreases_with_noisier_convergence(self):
        # Compare two runs: one perfectly flat, one nearly flat.
        n = 10
        config = ConvergenceConfig(
            window=3,
            epsilon_clearing=0.05,
            enabled_channels={"clearing"},
        )

        # Perfect: no variation at all.
        phi_perfect = _constant_phi(0.90, n)
        metrics_perfect = _make_day_metrics(phi_perfect)
        result_perfect = evaluate_convergence(
            metrics_perfect, events=[], config=config
        )

        # Noisy but convergent: small variation within epsilon.
        phi_noisy = _constant_phi(0.90, n)
        # Introduce small deltas within the tail window.
        phi_noisy[-3] = 0.89
        phi_noisy[-2] = 0.91
        phi_noisy[-1] = 0.90
        metrics_noisy = _make_day_metrics(phi_noisy)
        result_noisy = evaluate_convergence(
            metrics_noisy, events=[], config=config
        )

        assert result_perfect.quality >= result_noisy.quality
        assert result_noisy.quality > 0.0
