"""Tests for the run_comparison visualization module.

Tests cover:
- RunComparison dataclass construction and computed properties
- comparisons_to_dataframe conversion (single, multiple, empty)
- generate_comparison_html output (with mocked plotly and Supabase)
- Edge cases (zero deltas, correct DataFrame columns)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bilancio.analysis.visualization.run_comparison import (
    RunComparison,
    comparisons_to_dataframe,
    generate_comparison_html,
)

# ============================================================================
# Helpers
# ============================================================================

EXPECTED_DF_COLUMNS = [
    "kappa",
    "concentration",
    "mu",
    "seed",
    "delta_passive",
    "delta_active",
    "phi_passive",
    "phi_active",
    "trading_effect",
    "trading_relief_pct",
]


def _make_comparison(
    kappa: float = 0.5,
    concentration: float = 1.0,
    mu: float = 0.0,
    seed: int = 42,
    delta_passive: float = 0.4,
    delta_active: float = 0.2,
    phi_passive: float = 0.6,
    phi_active: float = 0.8,
) -> RunComparison:
    """Create a RunComparison with sensible defaults."""
    return RunComparison(
        kappa=kappa,
        concentration=concentration,
        mu=mu,
        seed=seed,
        delta_passive=delta_passive,
        delta_active=delta_active,
        phi_passive=phi_passive,
        phi_active=phi_active,
    )


# ============================================================================
# Test 1: RunComparison construction with all required fields
# ============================================================================


class TestRunComparisonConstruction:
    def test_all_fields_populated(self) -> None:
        rc = _make_comparison(
            kappa=1.0,
            concentration=2.0,
            mu=0.5,
            seed=99,
            delta_passive=0.3,
            delta_active=0.1,
            phi_passive=0.7,
            phi_active=0.9,
        )
        assert rc.kappa == 1.0
        assert rc.concentration == 2.0
        assert rc.mu == 0.5
        assert rc.seed == 99
        assert rc.delta_passive == 0.3
        assert rc.delta_active == 0.1
        assert rc.phi_passive == 0.7
        assert rc.phi_active == 0.9


# ============================================================================
# Test 2: RunComparison computed properties
# ============================================================================


class TestRunComparisonProperties:
    def test_trading_effect_positive_when_dealer_helps(self) -> None:
        """trading_effect = delta_passive - delta_active; positive means dealer reduced defaults."""
        rc = _make_comparison(delta_passive=0.4, delta_active=0.2)
        assert rc.trading_effect == pytest.approx(0.2)

    def test_trading_effect_negative_when_dealer_hurts(self) -> None:
        rc = _make_comparison(delta_passive=0.2, delta_active=0.4)
        assert rc.trading_effect == pytest.approx(-0.2)

    def test_trading_relief_pct(self) -> None:
        """50% of passive defaults relieved when active halves them."""
        rc = _make_comparison(delta_passive=0.4, delta_active=0.2)
        assert rc.trading_relief_pct == pytest.approx(50.0)

    def test_trading_relief_pct_when_delta_passive_zero(self) -> None:
        """Returns 0.0 to avoid division by zero."""
        rc = _make_comparison(delta_passive=0.0, delta_active=0.0)
        assert rc.trading_relief_pct == 0.0


# ============================================================================
# Test 3: comparisons_to_dataframe with single comparison
# ============================================================================


class TestComparisonsToDataframeSingle:
    def test_single_comparison_produces_one_row(self) -> None:
        rc = _make_comparison()
        df = comparisons_to_dataframe([rc])
        assert len(df) == 1

    def test_single_comparison_values_match(self) -> None:
        rc = _make_comparison(
            kappa=0.5,
            concentration=1.0,
            mu=0.0,
            seed=42,
            delta_passive=0.4,
            delta_active=0.2,
            phi_passive=0.6,
            phi_active=0.8,
        )
        df = comparisons_to_dataframe([rc])
        row = df.iloc[0]
        assert row["kappa"] == 0.5
        assert row["concentration"] == 1.0
        assert row["mu"] == 0.0
        assert row["seed"] == 42
        assert row["delta_passive"] == 0.4
        assert row["delta_active"] == 0.2
        assert row["phi_passive"] == 0.6
        assert row["phi_active"] == 0.8
        assert row["trading_effect"] == pytest.approx(0.2)
        assert row["trading_relief_pct"] == pytest.approx(50.0)


# ============================================================================
# Test 4: comparisons_to_dataframe with multiple comparisons
# ============================================================================


class TestComparisonsToDataframeMultiple:
    def test_multiple_comparisons_row_count(self) -> None:
        comparisons = [
            _make_comparison(kappa=0.3, delta_passive=0.6, delta_active=0.3),
            _make_comparison(kappa=0.5, delta_passive=0.4, delta_active=0.2),
            _make_comparison(kappa=1.0, delta_passive=0.1, delta_active=0.05),
        ]
        df = comparisons_to_dataframe(comparisons)
        assert len(df) == 3

    def test_multiple_comparisons_kappa_values(self) -> None:
        comparisons = [
            _make_comparison(kappa=0.3),
            _make_comparison(kappa=0.5),
            _make_comparison(kappa=1.0),
        ]
        df = comparisons_to_dataframe(comparisons)
        assert list(df["kappa"]) == [0.3, 0.5, 1.0]


# ============================================================================
# Test 5: comparisons_to_dataframe with empty list
# ============================================================================


class TestComparisonsToDataframeEmpty:
    def test_empty_list_returns_empty_dataframe(self) -> None:
        df = comparisons_to_dataframe([])
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_empty_dataframe_has_no_columns(self) -> None:
        """An empty input list produces a DataFrame with zero columns (no schema)."""
        df = comparisons_to_dataframe([])
        assert len(df.columns) == 0


# ============================================================================
# Test 6: generate_comparison_html creates an HTML file
# ============================================================================


class TestGenerateComparisonHtml:
    @patch("bilancio.analysis.visualization.run_comparison.load_job_comparison_data")
    def test_generates_html_file(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """generate_comparison_html writes a valid HTML file to disk."""
        mock_load.return_value = [
            _make_comparison(kappa=0.3, concentration=1.0, mu=0.0),
            _make_comparison(kappa=0.5, concentration=1.0, mu=0.0),
        ]

        output = tmp_path / "test_report.html"
        result = generate_comparison_html(
            job_id="test-job-abc",
            output_path=output,
            title="Test Report",
        )

        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Test Report" in content
        assert "test-job-abc" not in content or "Test Report" in content

    @patch("bilancio.analysis.visualization.run_comparison.load_job_comparison_data")
    def test_raises_on_empty_comparisons(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Raises ValueError when no comparison data is found."""
        mock_load.return_value = []

        with pytest.raises(ValueError, match="No comparison data found"):
            generate_comparison_html(
                job_id="empty-job",
                output_path=tmp_path / "should_not_exist.html",
            )

    @patch("bilancio.analysis.visualization.run_comparison.load_job_comparison_data")
    def test_default_output_path(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no output_path given, uses temp/<job_id>_comparison.html."""
        mock_load.return_value = [_make_comparison()]
        # Change working directory so the default 'temp/' goes into tmp_path
        monkeypatch.chdir(tmp_path)

        result = generate_comparison_html(job_id="my-test-job")

        assert result == Path("temp") / "my-test-job_comparison.html"
        assert result.exists()


# ============================================================================
# Test 7: RunComparison with edge values (zero deltas)
# ============================================================================


class TestRunComparisonEdgeValues:
    def test_zero_deltas_both_sides(self) -> None:
        rc = _make_comparison(delta_passive=0.0, delta_active=0.0)
        assert rc.trading_effect == 0.0
        assert rc.trading_relief_pct == 0.0

    def test_zero_passive_nonzero_active(self) -> None:
        """Dealer made things worse from a zero baseline."""
        rc = _make_comparison(delta_passive=0.0, delta_active=0.1)
        assert rc.trading_effect == pytest.approx(-0.1)
        # Relief pct returns 0 because delta_passive is zero (avoids div-by-zero)
        assert rc.trading_relief_pct == 0.0

    def test_perfect_clearing(self) -> None:
        """Both regimes clear perfectly."""
        rc = _make_comparison(
            delta_passive=0.0,
            delta_active=0.0,
            phi_passive=1.0,
            phi_active=1.0,
        )
        assert rc.trading_effect == 0.0
        assert rc.phi_passive == 1.0
        assert rc.phi_active == 1.0

    def test_full_default(self) -> None:
        """Both regimes fully default."""
        rc = _make_comparison(
            delta_passive=1.0,
            delta_active=1.0,
            phi_passive=0.0,
            phi_active=0.0,
        )
        assert rc.trading_effect == 0.0
        assert rc.trading_relief_pct == 0.0


# ============================================================================
# Test 8: DataFrame has correct columns
# ============================================================================


class TestDataFrameColumns:
    def test_all_expected_columns_present(self) -> None:
        df = comparisons_to_dataframe([_make_comparison()])
        assert list(df.columns) == EXPECTED_DF_COLUMNS

    def test_column_dtypes_are_numeric(self) -> None:
        df = comparisons_to_dataframe([_make_comparison()])
        for col in EXPECTED_DF_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' should be numeric, got {df[col].dtype}"
            )
