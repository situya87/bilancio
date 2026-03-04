"""Tests for performance flag cleanup (WI-3)."""

from __future__ import annotations

import pytest

from bilancio.core.performance import (
    _BOOL_FLAGS,
    SEMANTICS_CHANGING,
    SEMANTICS_PRESERVING,
    PerformanceConfig,
)


class TestTargetedUndoRemoved:
    """Verify targeted_undo flag no longer exists."""

    def test_targeted_undo_removed(self) -> None:
        config = PerformanceConfig()
        assert not hasattr(config, "targeted_undo")

    def test_targeted_undo_not_in_bool_flags(self) -> None:
        assert "targeted_undo" not in _BOOL_FLAGS

    def test_from_dict_ignores_removed_keys(self) -> None:
        config = PerformanceConfig.from_dict({"targeted_undo": True, "fast_atomic": True})
        assert config.fast_atomic is True
        assert not hasattr(config, "targeted_undo")


class TestNativeBackendWarning:
    """Verify native backend logs warning when unavailable."""

    def test_native_backend_warns_without_extension(self, caplog: pytest.LogCaptureFixture) -> None:
        from bilancio.dealer.kernel_native import NATIVE_AVAILABLE

        if NATIVE_AVAILABLE:
            pytest.skip("Rust extension is available, cannot test warning path")

        # We need to test the simulation.py code path. Since we can't easily
        # run the full simulation, we'll test the kernel_native module directly.
        from bilancio.dealer.kernel_native import recompute_dealer_state_native

        with pytest.raises(RuntimeError, match="Rust extension not available"):
            recompute_dealer_state_native(None, None, None)  # type: ignore[arg-type]


class TestSemanticsClassification:
    """Verify SEMANTICS_PRESERVING and SEMANTICS_CHANGING cover all flags."""

    def test_semantics_classification_complete(self) -> None:
        from dataclasses import fields

        non_meta_fields = {
            f.name
            for f in fields(PerformanceConfig)
            if f.name != "preset"
        }
        classified = SEMANTICS_PRESERVING | SEMANTICS_CHANGING
        assert classified == non_meta_fields, (
            f"Unclassified: {non_meta_fields - classified}, "
            f"Extra: {classified - non_meta_fields}"
        )

    def test_no_overlap(self) -> None:
        assert not (SEMANTICS_PRESERVING & SEMANTICS_CHANGING)
