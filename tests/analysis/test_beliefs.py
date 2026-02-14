"""Tests for belief trajectory and calibration analysis (Plan 034 Phase 5).

Covers:
1. belief_trajectory() — filtering and sorting
2. belief_vs_reality() — calibration buckets
3. estimate_summary() — aggregate statistics
4. export_estimates_jsonl() — JSONL serialization
5. _log_rating_estimates() / _log_dealer_estimates() — simulation integration
6. Config wiring — estimate_logging flag
"""

from decimal import Decimal
import json
from pathlib import Path

import pytest

from bilancio.information.estimates import Estimate
from bilancio.analysis.beliefs import (
    BeliefPoint,
    CalibrationBucket,
    EstimateSummary,
    belief_trajectory,
    belief_vs_reality,
    estimate_summary,
    export_estimates_jsonl,
)


# ── Fixtures ──────────────────────────────────────────────────────


def _make_estimates() -> list[Estimate]:
    """Build a sample sequence of estimates for testing."""
    return [
        Estimate(
            value=Decimal("0.10"),
            estimator_id="dealer_risk_assessor",
            target_id="firm_1",
            target_type="agent",
            estimation_day=1,
            method="bayesian_posterior",
            inputs={"lookback": 5},
        ),
        Estimate(
            value=Decimal("0.15"),
            estimator_id="dealer_risk_assessor",
            target_id="firm_1",
            target_type="agent",
            estimation_day=2,
            method="bayesian_posterior",
            inputs={"lookback": 5},
        ),
        Estimate(
            value=Decimal("0.05"),
            estimator_id="rating_agency",
            target_id="firm_1",
            target_type="agent",
            estimation_day=2,
            method="rating_agency_published",
            inputs={"source": "rating_registry"},
        ),
        Estimate(
            value=Decimal("0.20"),
            estimator_id="dealer_risk_assessor",
            target_id="firm_2",
            target_type="agent",
            estimation_day=1,
            method="bayesian_posterior",
        ),
        Estimate(
            value=Decimal("0.80"),
            estimator_id="system",
            target_id="market",
            target_type="system",
            estimation_day=3,
            method="aggregate_health",
        ),
    ]


# ── 1. belief_trajectory ─────────────────────────────────────────


class TestBeliefTrajectory:
    def test_no_filter_returns_all(self):
        estimates = _make_estimates()
        points = belief_trajectory(estimates)
        assert len(points) == 5

    def test_sorted_by_day(self):
        estimates = _make_estimates()
        points = belief_trajectory(estimates)
        days = [p.day for p in points]
        assert days == sorted(days)

    def test_filter_by_target_id(self):
        estimates = _make_estimates()
        points = belief_trajectory(estimates, target_id="firm_1")
        assert len(points) == 3
        assert all(p.day in (1, 2) for p in points)

    def test_filter_by_estimator_id(self):
        estimates = _make_estimates()
        points = belief_trajectory(estimates, estimator_id="rating_agency")
        assert len(points) == 1
        assert points[0].value == Decimal("0.05")

    def test_filter_by_method(self):
        estimates = _make_estimates()
        points = belief_trajectory(estimates, method="bayesian_posterior")
        assert len(points) == 3

    def test_combined_filters(self):
        estimates = _make_estimates()
        points = belief_trajectory(
            estimates,
            target_id="firm_1",
            estimator_id="dealer_risk_assessor",
        )
        assert len(points) == 2
        assert points[0].value == Decimal("0.10")
        assert points[1].value == Decimal("0.15")

    def test_empty_input(self):
        points = belief_trajectory([])
        assert points == []

    def test_no_match(self):
        estimates = _make_estimates()
        points = belief_trajectory(estimates, target_id="nonexistent")
        assert points == []

    def test_belief_point_fields(self):
        estimates = _make_estimates()
        points = belief_trajectory(estimates, target_id="firm_1", estimator_id="rating_agency")
        assert len(points) == 1
        p = points[0]
        assert p.day == 2
        assert p.value == Decimal("0.05")
        assert p.method == "rating_agency_published"
        assert p.estimator_id == "rating_agency"


# ── 2. belief_vs_reality ─────────────────────────────────────────


class TestBeliefVsReality:
    def test_basic_calibration(self):
        estimates = [
            Estimate(
                value=Decimal("0.10"), estimator_id="e", target_id="a1",
                target_type="agent", estimation_day=1, method="m",
            ),
            Estimate(
                value=Decimal("0.90"), estimator_id="e", target_id="a2",
                target_type="agent", estimation_day=1, method="m",
            ),
        ]
        defaulted = {"a2"}
        buckets = belief_vs_reality(estimates, defaulted, n_buckets=5)
        assert len(buckets) == 5

        # a1 (p=0.10) → bucket 0 (0.0–0.2), no default
        assert buckets[0].count == 1
        assert buckets[0].actual_defaults == 0

        # a2 (p=0.90) → bucket 4 (0.8–1.0), defaulted
        assert buckets[4].count == 1
        assert buckets[4].actual_defaults == 1

    def test_takes_latest_estimate(self):
        estimates = [
            Estimate(
                value=Decimal("0.90"), estimator_id="e", target_id="a1",
                target_type="agent", estimation_day=1, method="m",
            ),
            Estimate(
                value=Decimal("0.10"), estimator_id="e", target_id="a1",
                target_type="agent", estimation_day=5, method="m",
            ),
        ]
        buckets = belief_vs_reality(estimates, set(), n_buckets=5)
        # Latest estimate is 0.10, should be in bucket 0
        assert buckets[0].count == 1
        assert sum(b.count for b in buckets) == 1

    def test_ignores_non_agent_targets(self):
        estimates = [
            Estimate(
                value=Decimal("0.50"), estimator_id="e", target_id="sys",
                target_type="system", estimation_day=1, method="m",
            ),
        ]
        buckets = belief_vs_reality(estimates, set())
        assert all(b.count == 0 for b in buckets)

    def test_empty_estimates(self):
        buckets = belief_vs_reality([], set())
        assert buckets == []

    def test_calibration_bucket_properties(self):
        bucket = CalibrationBucket(
            predicted_low=Decimal("0.0"),
            predicted_high=Decimal("0.2"),
            count=10,
            actual_defaults=3,
        )
        assert bucket.predicted_mean == Decimal("0.1")
        assert bucket.actual_rate == Decimal("0.3")

    def test_calibration_bucket_empty(self):
        bucket = CalibrationBucket(
            predicted_low=Decimal("0.0"),
            predicted_high=Decimal("0.2"),
        )
        assert bucket.actual_rate == Decimal(0)


# ── 3. estimate_summary ──────────────────────────────────────────


class TestEstimateSummary:
    def test_basic_summary(self):
        estimates = _make_estimates()
        summary = estimate_summary(estimates)
        assert summary.count == 5
        assert "bayesian_posterior" in summary.methods
        assert summary.methods["bayesian_posterior"] == 3
        assert "rating_agency_published" in summary.methods
        assert summary.methods["rating_agency_published"] == 1
        assert "aggregate_health" in summary.methods

    def test_estimators(self):
        estimates = _make_estimates()
        summary = estimate_summary(estimates)
        assert summary.estimators["dealer_risk_assessor"] == 3
        assert summary.estimators["rating_agency"] == 1
        assert summary.estimators["system"] == 1

    def test_target_types(self):
        estimates = _make_estimates()
        summary = estimate_summary(estimates)
        assert summary.target_types["agent"] == 4
        assert summary.target_types["system"] == 1

    def test_day_range(self):
        estimates = _make_estimates()
        summary = estimate_summary(estimates)
        assert summary.day_range == (1, 3)

    def test_value_range(self):
        estimates = _make_estimates()
        summary = estimate_summary(estimates)
        assert summary.value_range == (Decimal("0.05"), Decimal("0.80"))

    def test_empty_estimates(self):
        summary = estimate_summary([])
        assert summary.count == 0
        assert summary.methods == {}
        assert summary.day_range is None
        assert summary.value_range is None


# ── 4. export_estimates_jsonl ─────────────────────────────────────


class TestExportEstimatesJsonl:
    def test_writes_jsonl(self, tmp_path: Path):
        estimates = _make_estimates()[:2]
        out = tmp_path / "estimates.jsonl"
        count = export_estimates_jsonl(estimates, str(out))
        assert count == 2
        assert out.exists()

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["value"] == "0.10"
        assert first["estimator_id"] == "dealer_risk_assessor"
        assert first["target_id"] == "firm_1"
        assert first["target_type"] == "agent"
        assert first["estimation_day"] == 1
        assert first["method"] == "bayesian_posterior"

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir" / "estimates.jsonl"
        count = export_estimates_jsonl(_make_estimates()[:1], str(out))
        assert count == 1
        assert out.exists()

    def test_empty_list(self, tmp_path: Path):
        out = tmp_path / "empty.jsonl"
        count = export_estimates_jsonl([], str(out))
        assert count == 0
        assert out.read_text() == ""

    def test_decimal_serialization(self, tmp_path: Path):
        est = Estimate(
            value=Decimal("0.123456789"),
            estimator_id="e",
            target_id="t",
            target_type="agent",
            estimation_day=1,
            method="m",
            inputs={"nested_decimal": Decimal("0.999")},
        )
        out = tmp_path / "decimal.jsonl"
        export_estimates_jsonl([est], str(out))
        record = json.loads(out.read_text().strip())
        assert record["value"] == "0.123456789"
        assert record["inputs"]["nested_decimal"] == "0.999"


# ── 5. Simulation integration: _log_*_estimates ───────────────────


class TestLogRatingEstimates:
    def test_logs_rating_estimates(self):
        from bilancio.engines.simulation import _log_rating_estimates
        from bilancio.engines.system import System

        system = System()
        system.state.estimate_logging_enabled = True
        system.state.rating_registry = {
            "firm_1": Decimal("0.12"),
            "firm_2": Decimal("0.25"),
        }

        _log_rating_estimates(system, current_day=5)

        assert len(system.state.estimate_log) == 2
        est = system.state.estimate_log[0]
        assert est.estimator_id == "rating_agency"
        assert est.target_type == "agent"
        assert est.estimation_day == 5
        assert est.method == "rating_agency_published"

    def test_skips_without_registry(self):
        from bilancio.engines.simulation import _log_rating_estimates
        from bilancio.engines.system import System

        system = System()
        system.state.estimate_logging_enabled = True
        # No rating_registry attribute or empty

        _log_rating_estimates(system, current_day=1)
        assert len(system.state.estimate_log) == 0

    def test_respects_logging_flag(self):
        """Even if called, log_estimate only appends when enabled."""
        from bilancio.engines.simulation import _log_rating_estimates
        from bilancio.engines.system import System

        system = System()
        system.state.estimate_logging_enabled = False
        system.state.rating_registry = {"firm_1": Decimal("0.10")}

        _log_rating_estimates(system, current_day=1)
        assert len(system.state.estimate_log) == 0


class TestLogDealerEstimates:
    def _build_system_with_dealer(self):
        from bilancio.engines.system import System
        from bilancio.engines.dealer_integration import DealerSubsystem
        from bilancio.dealer.risk_assessment import RiskAssessor, RiskAssessmentParams
        from bilancio.domain.agents.firm import Firm

        system = System()
        system.state.estimate_logging_enabled = True

        # Add some agents
        f1 = Firm(id="firm_1", name="Firm 1", kind="firm")
        f2 = Firm(id="firm_2", name="Firm 2", kind="firm")
        system.add_agent(f1)
        system.add_agent(f2)

        # Attach dealer subsystem with risk assessor
        subsystem = DealerSubsystem()
        subsystem.risk_assessor = RiskAssessor(RiskAssessmentParams(
            initial_prior=Decimal("0.15"),
        ))
        system.state.dealer_subsystem = subsystem

        return system

    def test_logs_dealer_estimates(self):
        from bilancio.engines.simulation import _log_dealer_estimates

        system = self._build_system_with_dealer()
        _log_dealer_estimates(system, current_day=3)

        assert len(system.state.estimate_log) == 2
        est = system.state.estimate_log[0]
        assert est.estimator_id == "dealer_risk_assessor"
        assert est.estimation_day == 3

    def test_skips_defaulted_agents(self):
        from bilancio.engines.simulation import _log_dealer_estimates

        system = self._build_system_with_dealer()
        system.state.agents["firm_1"].defaulted = True

        _log_dealer_estimates(system, current_day=3)

        assert len(system.state.estimate_log) == 1
        assert system.state.estimate_log[0].target_id == "firm_2"

    def test_skips_without_dealer(self):
        from bilancio.engines.simulation import _log_dealer_estimates
        from bilancio.engines.system import System

        system = System()
        system.state.estimate_logging_enabled = True

        _log_dealer_estimates(system, current_day=1)
        assert len(system.state.estimate_log) == 0

    def test_skips_without_risk_assessor(self):
        from bilancio.engines.simulation import _log_dealer_estimates
        from bilancio.engines.system import System
        from bilancio.engines.dealer_integration import DealerSubsystem

        system = System()
        system.state.estimate_logging_enabled = True
        subsystem = DealerSubsystem()
        subsystem.risk_assessor = None
        system.state.dealer_subsystem = subsystem

        _log_dealer_estimates(system, current_day=1)
        assert len(system.state.estimate_log) == 0


# ── 6. Config wiring ─────────────────────────────────────────────


class TestEstimateLoggingConfig:
    def test_default_disabled(self):
        from bilancio.config.models import RunConfig
        config = RunConfig()
        assert config.estimate_logging is False

    def test_enabled_via_config(self):
        from bilancio.config.models import RunConfig
        config = RunConfig(estimate_logging=True)
        assert config.estimate_logging is True


# ── 7. Package-level imports ──────────────────────────────────────


class TestPackageImports:
    def test_beliefs_importable_from_analysis(self):
        from bilancio.analysis import (
            BeliefPoint,
            CalibrationBucket,
            EstimateSummary,
            belief_trajectory,
            belief_vs_reality,
            estimate_summary,
            export_estimates_jsonl,
        )
        assert BeliefPoint is not None
        assert callable(belief_trajectory)
        assert callable(belief_vs_reality)
        assert callable(estimate_summary)
        assert callable(export_estimates_jsonl)
