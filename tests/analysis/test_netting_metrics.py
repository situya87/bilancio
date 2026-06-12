"""Tests for clearinghouse netting metrics (Plan 059).

Covers phi_delta netting-awareness, the netting_totals helper, and the
gross/net decomposition emitted by the per-run metrics summary.
"""

from decimal import Decimal

from bilancio.analysis.metrics import netted_for_day, netting_totals, phi_delta
from bilancio.analysis.report import (
    aggregate_runs,
    compute_day_metrics,
    summarize_day_metrics,
)


def _ring_dues():
    return [
        {"amount": "100", "pid": "p1", "due_day": 1, "debtor": "A", "creditor": "B"},
        {"amount": "100", "pid": "p2", "due_day": 1, "debtor": "B", "creditor": "C"},
        {"amount": "100", "pid": "p3", "due_day": 1, "debtor": "C", "creditor": "A"},
    ]


def _netted_event(pid: str, netted: str, day: int = 1, **extra):
    event = {
        "kind": "PayableNetted",
        "day": day,
        "pid": pid,
        "netted_amount": netted,
        "original_amount": "100",
        "remaining_amount": "0",
    }
    event.update(extra)
    return event


class TestPhiDeltaNetting:
    """phi_delta counts PayableNetted face as cleared."""

    def test_fully_netted_ring_counts_as_settled(self):
        dues = _ring_dues()
        events = [
            _netted_event("p1", "100"),
            _netted_event("p2", "100"),
            _netted_event("p3", "100"),
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("1")
        assert delta == Decimal("0")

    def test_partial_netting_plus_cash_settlement(self):
        dues = _ring_dues()
        events = [
            _netted_event("p1", "60", remaining_amount="40"),
            {"kind": "PayableSettled", "day": 1, "amount": "40", "pid": "p1"},
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p2"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("200") / Decimal("300")
        assert delta == Decimal("100") / Decimal("300")

    def test_netted_on_other_day_ignored(self):
        dues = _ring_dues()
        events = [_netted_event("p1", "100", day=2)]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("0")
        assert delta == Decimal("1")

    def test_netted_with_unknown_pid_ignored(self):
        dues = _ring_dues()
        events = [_netted_event("p_unknown", "100")]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("0")
        assert delta == Decimal("1")

    def test_netted_matches_by_alias(self):
        dues = [{"amount": "100", "alias": "loan1", "due_day": 1}]
        events = [
            {
                "kind": "PayableNetted",
                "day": 1,
                "alias": "loan1",
                "netted_amount": "100",
            }
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("1")
        assert delta == Decimal("0")

    def test_backward_compatible_without_netted_events(self):
        """Zero PayableNetted events => identical phi/delta to pre-change semantics."""
        dues = _ring_dues()
        events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p1"},
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p2"},
        ]
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("200") / Decimal("300")
        assert delta == Decimal("100") / Decimal("300")

    def test_netting_is_purely_additive(self):
        """Adding a matched PayableNetted event raises phi by exactly its amount."""
        dues = _ring_dues()
        base_events = [
            {"kind": "PayableSettled", "day": 1, "amount": "100", "pid": "p1"},
        ]
        phi_base, _ = phi_delta(base_events, dues, 1)
        phi_netted, _ = phi_delta(base_events + [_netted_event("p2", "100")], dues, 1)
        assert phi_base == Decimal("100") / Decimal("300")
        assert phi_netted == Decimal("200") / Decimal("300")


class TestNettedForDay:
    def test_sums_matched_due_today_reductions(self):
        dues = _ring_dues()
        events = [
            _netted_event("p1", "60"),
            _netted_event("p2", "40"),
            _netted_event("p3", "100", day=2),
            _netted_event("p_unknown", "100"),
        ]
        assert netted_for_day(events, dues, 1) == Decimal("100")

    def test_zero_without_netted_events(self):
        assert netted_for_day([], _ring_dues(), 1) == Decimal("0")


class TestNettingTotals:
    def test_totals_for_netted_ring(self):
        events = [
            {"kind": "PayableCreated", "payable_id": "p1", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "payable_id": "p2", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "payable_id": "p3", "amount": "100", "due_day": 2},
            _netted_event("p1", "100"),
            _netted_event("p2", "30", remaining_amount="70"),
            _netted_event("p3", "100", day=1),  # netted before due day: excluded
        ]
        gross, netted = netting_totals(events)
        assert gross == Decimal("300")
        assert netted == Decimal("130")

    def test_no_netted_events(self):
        events = [
            {"kind": "PayableCreated", "payable_id": "p1", "amount": "100", "due_day": 1},
        ]
        assert netting_totals(events) == (Decimal("100"), Decimal("0"))

    def test_empty_events(self):
        assert netting_totals([]) == (Decimal("0"), Decimal("0"))


class TestNettingEfficiencySummary:
    def test_summary_reports_netting_decomposition(self):
        day_metrics = [
            {"day": 1, "S_t": Decimal("300"), "phi_t": Decimal("1"),
             "delta_t": Decimal("0"), "face_netted_t": Decimal("300")},
            {"day": 2, "S_t": Decimal("100"), "phi_t": Decimal("0"),
             "delta_t": Decimal("1"), "face_netted_t": Decimal("0")},
        ]
        summary = summarize_day_metrics(day_metrics)
        assert summary["gross_face_due"] == 400.0
        assert summary["face_extinguished_by_netting"] == 300.0
        assert summary["netting_efficiency"] == Decimal("300") / Decimal("400")

    def test_efficiency_zero_when_no_face_due(self):
        summary = summarize_day_metrics([])
        assert summary["gross_face_due"] == 0.0
        assert summary["face_extinguished_by_netting"] == 0.0
        assert summary["netting_efficiency"] == Decimal("0")

    def test_efficiency_zero_for_legacy_rows_without_netting_column(self):
        day_metrics = [
            {"day": 1, "S_t": "100", "phi_t": "1", "delta_t": "0"},
        ]
        summary = summarize_day_metrics(day_metrics)
        assert summary["netting_efficiency"] == Decimal("0")
        assert summary["gross_face_due"] == 100.0

    def test_string_rows_from_csv(self):
        day_metrics = [
            {"day": "1", "S_t": "200", "phi_t": "1", "delta_t": "0",
             "face_netted_t": "50"},
        ]
        summary = summarize_day_metrics(day_metrics)
        assert summary["face_extinguished_by_netting"] == 50.0
        assert summary["netting_efficiency"] == Decimal("50") / Decimal("200")


class TestComputeDayMetricsNetting:
    def test_face_netted_t_emitted_per_day(self):
        events = [
            {"kind": "PayableCreated", "payable_id": "p1", "amount": Decimal("100"),
             "due_day": 1, "debtor": "A", "creditor": "B"},
            {"kind": "PayableCreated", "payable_id": "p2", "amount": Decimal("100"),
             "due_day": 1, "debtor": "B", "creditor": "A"},
            {"kind": "PayableNetted", "day": 1, "pid": "p1",
             "netted_amount": Decimal("100"), "remaining_amount": Decimal("0")},
            {"kind": "PayableNetted", "day": 1, "pid": "p2",
             "netted_amount": Decimal("100"), "remaining_amount": Decimal("0")},
        ]
        result = compute_day_metrics(events)
        row = result["day_metrics"][0]
        assert row["day"] == 1
        assert row["face_netted_t"] == Decimal("200")
        assert row["phi_t"] == Decimal("1")
        assert row["delta_t"] == Decimal("0")

        summary = summarize_day_metrics(result["day_metrics"])
        assert summary["netting_efficiency"] == Decimal("1")
        assert summary["gross_face_due"] == 200.0
        assert summary["delta_total"] == Decimal("0")

    def test_no_netting_events_zero_efficiency(self):
        events = [
            {"kind": "PayableCreated", "payable_id": "p1", "amount": Decimal("100"),
             "due_day": 1, "debtor": "A", "creditor": "B"},
            {"kind": "PayableSettled", "day": 1, "pid": "p1",
             "amount": Decimal("100"), "debtor": "A", "creditor": "B"},
        ]
        result = compute_day_metrics(events)
        assert result["day_metrics"][0]["face_netted_t"] == Decimal("0")
        summary = summarize_day_metrics(result["day_metrics"])
        assert summary["netting_efficiency"] == Decimal("0")
        assert summary["phi_total"] == Decimal("1")


def test_aggregate_runs_carries_netting_efficiency(tmp_path):
    registry_dir = tmp_path / "registry"
    out_dir = tmp_path / "runs" / "grid_net1" / "out"
    aggregate_dir = tmp_path / "aggregate"
    registry_dir.mkdir()
    out_dir.mkdir(parents=True)
    aggregate_dir.mkdir()

    metrics_csv = out_dir / "metrics.csv"
    metrics_csv.write_text(
        "day,S_t,phi_t,delta_t,face_netted_t\n"
        "1,100,1,0,100\n"
        "2,100,0.5,0.5,0\n"
    )

    registry_csv = registry_dir / "experiments.csv"
    registry_csv.write_text(
        "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,S1,L0,"
        "metrics_csv,status,time_to_stability,n_defaults,cascade_fraction\n"
        "grid_net1,grid,42,5,0.25,1,0,0,200,50,"
        "../runs/grid_net1/out/metrics.csv,completed,2,0,\n"
    )

    results_csv = aggregate_dir / "results.csv"
    rows = aggregate_runs(registry_csv, results_csv)

    assert rows
    row = rows[0]
    assert row["netting_efficiency"] == Decimal("0.5")
    assert row["gross_face_due"] == 200.0

    header = results_csv.read_text().splitlines()[0]
    assert "netting_efficiency" in header
    assert "gross_face_due" in header
