"""Contract tests for benchmark utility functions.

Tests the public API of benchmark_utils.py and benchmark_sim_utils.py
to ensure stable contracts for benchmark scoring, grading, and reporting.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

# Add scripts directory to path so we can import benchmark helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import benchmark_sim_utils  # noqa: E402
import benchmark_utils  # noqa: E402

from bilancio.domain.agent import AgentKind  # noqa: E402
from bilancio.domain.agents.household import Household  # noqa: E402
from bilancio.engines.system import System  # noqa: E402

# ---------------------------------------------------------------------------
# bounded()
# ---------------------------------------------------------------------------

class TestBounded:
    def test_below_range_clamped_to_min(self):
        assert benchmark_utils.bounded(-0.5, 0.0, 1.0) == 0.0

    def test_above_range_clamped_to_max(self):
        assert benchmark_utils.bounded(1.5, 0.0, 1.0) == 1.0

    def test_within_range_unchanged(self):
        assert benchmark_utils.bounded(0.5, 0.0, 1.0) == 0.5


# ---------------------------------------------------------------------------
# grade_for_score()
# ---------------------------------------------------------------------------

class TestGradeForScore:
    def test_score_90_is_A(self):
        assert benchmark_utils.grade_for_score(90) == "A"

    def test_score_95_is_A(self):
        assert benchmark_utils.grade_for_score(95) == "A"

    def test_score_80_is_B(self):
        assert benchmark_utils.grade_for_score(80) == "B"

    def test_score_85_is_B(self):
        assert benchmark_utils.grade_for_score(85) == "B"

    def test_score_70_is_C(self):
        assert benchmark_utils.grade_for_score(70) == "C"

    def test_score_75_is_C(self):
        assert benchmark_utils.grade_for_score(75) == "C"

    def test_score_60_is_D(self):
        assert benchmark_utils.grade_for_score(60) == "D"

    def test_score_65_is_D(self):
        assert benchmark_utils.grade_for_score(65) == "D"

    def test_score_below_60_is_F(self):
        assert benchmark_utils.grade_for_score(59) == "F"

    def test_score_zero_is_F(self):
        assert benchmark_utils.grade_for_score(0) == "F"


# ---------------------------------------------------------------------------
# cap_grade_for_critical_failures()
# ---------------------------------------------------------------------------

class TestCapGradeForCriticalFailures:
    def test_zero_failures_grade_unchanged(self):
        assert benchmark_utils.cap_grade_for_critical_failures("A", 0) == "A"

    def test_one_failure_caps_at_C(self):
        # 1 failure => cap = "C", so "A" becomes "C"
        assert benchmark_utils.cap_grade_for_critical_failures("A", 1) == "C"

    def test_two_failures_caps_at_D(self):
        # 2 failures => cap = "D", so "A" becomes "D"
        assert benchmark_utils.cap_grade_for_critical_failures("A", 2) == "D"

    def test_three_or_more_failures_caps_at_F(self):
        # 3+ failures => cap = "F"
        assert benchmark_utils.cap_grade_for_critical_failures("A", 3) == "F"
        assert benchmark_utils.cap_grade_for_critical_failures("B", 5) == "F"

    def test_already_low_grade_stays_low(self):
        # If base grade is already worse than the cap, base grade wins
        assert benchmark_utils.cap_grade_for_critical_failures("F", 1) == "F"
        assert benchmark_utils.cap_grade_for_critical_failures("D", 1) == "D"


# ---------------------------------------------------------------------------
# report_dict()
# ---------------------------------------------------------------------------

class TestReportDict:
    def test_required_keys_present(self):
        categories = [
            benchmark_utils.CategoryResult(
                name="cat1", max_points=50.0, earned_points=40.0, details={}
            ),
        ]
        checks = [
            benchmark_utils.CriticalCheck(code="G1", passed=True, message="ok"),
        ]
        result = benchmark_utils.report_dict(
            benchmark_name="test-bench",
            target_score=80.0,
            total_score=85.0,
            status="pass",
            meets_target=True,
            base_grade="B",
            grade="B",
            elapsed_seconds=1.23,
            categories=categories,
            critical_checks=checks,
        )

        # Verify all required keys
        assert "total_score" in result
        assert "grade" in result
        assert "benchmark" in result
        assert result["benchmark"] == "test-bench"
        assert "categories" in result
        assert "critical_checks" in result
        assert "generated_at_utc" in result
        assert "status" in result
        assert "meets_target" in result
        assert result["total_score"] == 85.0
        assert result["grade"] == "B"

    def test_critical_failures_extracted(self):
        checks = [
            benchmark_utils.CriticalCheck(code="G1", passed=True, message="ok"),
            benchmark_utils.CriticalCheck(code="G2", passed=False, message="failed"),
        ]
        result = benchmark_utils.report_dict(
            benchmark_name="test",
            target_score=80.0,
            total_score=70.0,
            status="fail",
            meets_target=False,
            base_grade="C",
            grade="C",
            elapsed_seconds=2.0,
            categories=[],
            critical_checks=checks,
        )
        assert len(result["critical_failures"]) == 1
        assert result["critical_failures"][0]["code"] == "G2"


# ---------------------------------------------------------------------------
# build_markdown_report()
# ---------------------------------------------------------------------------

class TestBuildMarkdownReport:
    def test_contains_header_score_and_categories(self):
        categories = [
            benchmark_utils.CategoryResult(
                name="Correctness", max_points=50.0, earned_points=45.0, details={}
            ),
            benchmark_utils.CategoryResult(
                name="Robustness", max_points=50.0, earned_points=40.0, details={}
            ),
        ]
        checks = [
            benchmark_utils.CriticalCheck(code="G1", passed=True, message="ok"),
        ]
        md = benchmark_utils.build_markdown_report(
            title="Test Benchmark Report",
            generated_at="2025-01-01T00:00:00Z",
            target_score=80.0,
            total_score=85.0,
            status="pass",
            grade="B",
            base_grade="B",
            meets_target=True,
            categories=categories,
            critical_checks=checks,
        )

        # Report header
        assert "# Test Benchmark Report" in md
        # Score and grade present
        assert "85.00" in md
        assert "**B**" in md
        # Category breakdown
        assert "Correctness" in md
        assert "Robustness" in md
        assert "Category Scores" in md
        # Critical gates section
        assert "Critical Gates" in md


# ---------------------------------------------------------------------------
# check_operational_budget()
# ---------------------------------------------------------------------------

class TestCheckOperationalBudget:
    def test_within_budget_passes(self):
        result = benchmark_utils.check_operational_budget(
            elapsed_seconds=100.0,
            peak_memory_mb=512.0,
            wall_time_budget_seconds=300.0,
            memory_budget_mb=2048.0,
        )
        assert result["wall_time_ok"] is True
        assert result["memory_ok"] is True
        assert result["all_ok"] is True

    def test_over_budget_fails(self):
        result = benchmark_utils.check_operational_budget(
            elapsed_seconds=500.0,
            peak_memory_mb=4096.0,
            wall_time_budget_seconds=300.0,
            memory_budget_mb=2048.0,
        )
        assert result["wall_time_ok"] is False
        assert result["memory_ok"] is False
        assert result["all_ok"] is False


# ---------------------------------------------------------------------------
# SimulationOutcome dataclass
# ---------------------------------------------------------------------------

class TestSimulationOutcome:
    def test_construction_with_expected_fields(self):
        system = System()
        outcome = benchmark_sim_utils.SimulationOutcome(
            system=system,
            elapsed_seconds=1.5,
            events_count=10,
            defaults_count=2,
            household_count=8,
            default_ratio=0.25,
            total_loss=100.0,
            total_loss_ratio=0.1,
            run_level_metrics={"total_loss": 100.0},
        )
        assert outcome.elapsed_seconds == 1.5
        assert outcome.events_count == 10
        assert outcome.defaults_count == 2
        assert outcome.household_count == 8
        assert outcome.default_ratio == 0.25
        assert outcome.total_loss == 100.0
        assert outcome.total_loss_ratio == 0.1
        assert outcome.run_level_metrics == {"total_loss": 100.0}
        assert outcome.system is system


# ---------------------------------------------------------------------------
# compile_ring_scenario()
# ---------------------------------------------------------------------------

class TestCompileRingScenario:
    def test_returns_valid_scenario_dict(self):
        scenario = benchmark_sim_utils.compile_ring_scenario(
            n_agents=5,
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            seed=42,
            maturity_days=5,
        )
        assert isinstance(scenario, dict)
        # Must contain essential top-level keys for a scenario
        assert "agents" in scenario
        assert "initial_actions" in scenario
        # Agents should be a list of agent dicts
        assert isinstance(scenario["agents"], list)
        assert len(scenario["agents"]) >= 5  # at least n_agents


# ---------------------------------------------------------------------------
# count_households()
# ---------------------------------------------------------------------------

class TestCountHouseholds:
    def test_counts_household_agents(self):
        system = System()
        # Add 3 Household agents
        h1 = Household(id="H1", name="household-1", kind=AgentKind.HOUSEHOLD)
        h2 = Household(id="H2", name="household-2", kind=AgentKind.HOUSEHOLD)
        h3 = Household(id="H3", name="household-3", kind=AgentKind.HOUSEHOLD)
        system.add_agent(h1)
        system.add_agent(h2)
        system.add_agent(h3)

        assert benchmark_sim_utils.count_households(system) == 3

    def test_ignores_non_household_agents(self):
        from bilancio.domain.agents.firm import Firm

        system = System()
        h1 = Household(id="H1", name="household-1", kind=AgentKind.HOUSEHOLD)
        f1 = Firm(id="F1", name="firm-1", kind=AgentKind.FIRM)
        system.add_agent(h1)
        system.add_agent(f1)

        assert benchmark_sim_utils.count_households(system) == 1

    def test_empty_system_returns_zero(self):
        system = System()
        assert benchmark_sim_utils.count_households(system) == 0
