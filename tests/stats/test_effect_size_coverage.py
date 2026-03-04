"""Coverage tests for stats/effect_size.py.

Targets uncovered edge cases: zero pooled_sd, zero sd_d, length mismatch.
"""

from __future__ import annotations

import pytest

from bilancio.stats.effect_size import cohens_d, cohens_d_paired


class TestCohensD:
    """Cover cohens_d edge cases."""

    def test_zero_pooled_sd_returns_zero(self):
        """When both groups are constant (sd=0), return 0.0."""
        control = [5.0, 5.0, 5.0]
        treatment = [5.0, 5.0, 5.0]
        result = cohens_d(control, treatment)
        assert result == 0.0

    def test_insufficient_control_group(self):
        with pytest.raises(ValueError, match="Need >= 2"):
            cohens_d([1.0], [1.0, 2.0])

    def test_insufficient_treatment_group(self):
        with pytest.raises(ValueError, match="Need >= 2"):
            cohens_d([1.0, 2.0], [1.0])


class TestCohensDPaired:
    """Cover cohens_d_paired edge cases."""

    def test_zero_sd_returns_zero(self):
        """When all differences are the same (sd=0), return 0.0."""
        control = [10.0, 20.0, 30.0]
        treatment = [5.0, 15.0, 25.0]
        # All differences are 5.0, sd_d = 0 -> return 0.0
        result = cohens_d_paired(control, treatment)
        assert result == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            cohens_d_paired([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_insufficient_pairs_raises(self):
        with pytest.raises(ValueError, match="Need >= 2"):
            cohens_d_paired([1.0], [2.0])
