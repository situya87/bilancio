"""Tests for bilancio.engines.simulation_result — Plan 051.

Verifies:
- SimulationResult construction
- List-like forwarding (len, getitem, iter) for backward compat
- Re-export from simulation.py
"""

from bilancio.engines.simulation import DayReport
from bilancio.engines.simulation_result import SimulationResult
from bilancio.engines.termination import StabilitySnapshot, StopReason


def _make_reports(n: int) -> list[DayReport]:
    return [DayReport(day=i, impacted=0) for i in range(n)]


class TestSimulationResult:
    def test_construction(self):
        reports = _make_reports(3)
        result = SimulationResult(
            reports=reports,
            stop_reason=StopReason.STABILITY_REACHED,
            stop_day=3,
        )
        assert result.reports is reports
        assert result.stop_reason == StopReason.STABILITY_REACHED
        assert result.stop_day == 3
        assert result.stability_snapshots == []
        assert result.winddown_days == 0
        assert result.final_cb_settlement is None

    def test_with_all_fields(self):
        snap = StabilitySnapshot(
            day=2,
            consecutive_quiet=2,
            consecutive_no_defaults=2,
            has_open_obligations=False,
            impacted_count=0,
            default_count=0,
        )
        result = SimulationResult(
            reports=_make_reports(3),
            stop_reason=StopReason.MAX_DAYS_REACHED,
            stop_day=3,
            stability_snapshots=[snap],
            winddown_days=5,
            final_cb_settlement={"loans_attempted": 1, "loans_repaid": 1},
        )
        assert len(result.stability_snapshots) == 1
        assert result.winddown_days == 5
        assert result.final_cb_settlement["loans_repaid"] == 1

    # -- List-like backward compatibility --

    def test_len(self):
        result = SimulationResult(
            reports=_make_reports(5),
            stop_reason=StopReason.STABILITY_REACHED,
            stop_day=5,
        )
        assert len(result) == 5

    def test_getitem(self):
        reports = _make_reports(3)
        result = SimulationResult(
            reports=reports,
            stop_reason=StopReason.STABILITY_REACHED,
            stop_day=3,
        )
        assert result[0].day == 0
        assert result[1].day == 1
        assert result[-1].day == 2

    def test_getitem_slice(self):
        result = SimulationResult(
            reports=_make_reports(5),
            stop_reason=StopReason.STABILITY_REACHED,
            stop_day=5,
        )
        sliced = result[1:3]
        assert len(sliced) == 2
        assert sliced[0].day == 1

    def test_iter(self):
        reports = _make_reports(3)
        result = SimulationResult(
            reports=reports,
            stop_reason=StopReason.STABILITY_REACHED,
            stop_day=3,
        )
        days = [r.day for r in result]
        assert days == [0, 1, 2]


class TestSimulationResultReExport:
    def test_importable_from_simulation(self):
        from bilancio.engines.simulation import SimulationResult as SR
        assert SR is SimulationResult
