"""Tests for experiment design deprecation warnings (WI-5)."""

from __future__ import annotations

import warnings
from pathlib import Path

from bilancio.experiments.balanced_comparison import (
    BalancedComparisonConfig,
    BalancedComparisonRunner,
)


class TestDeprecationWarnings:
    """Verify banking arms emit DeprecationWarning."""

    def test_banking_arms_emit_deprecation_bank_passive(self, tmp_path: Path) -> None:
        config = BalancedComparisonConfig(enable_bank_passive=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BalancedComparisonRunner(config=config, out_dir=tmp_path)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()
            assert "sweep bank" in str(dep_warnings[0].message).lower()

    def test_banking_arms_emit_deprecation_bank_dealer(self, tmp_path: Path) -> None:
        config = BalancedComparisonConfig(enable_bank_dealer=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BalancedComparisonRunner(config=config, out_dir=tmp_path)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

    def test_banking_arms_emit_deprecation_bank_dealer_nbfi(self, tmp_path: Path) -> None:
        config = BalancedComparisonConfig(enable_bank_dealer_nbfi=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BalancedComparisonRunner(config=config, out_dir=tmp_path)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

    def test_clean_2arm_no_warning(self, tmp_path: Path) -> None:
        config = BalancedComparisonConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BalancedComparisonRunner(config=config, out_dir=tmp_path)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
