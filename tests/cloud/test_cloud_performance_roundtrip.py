"""Tests for performance config round-trip through cloud serialization."""

from __future__ import annotations

from bilancio.core.performance import PerformanceConfig
from bilancio.runners.cloud_executor import CloudExecutor
from bilancio.runners.models import RunOptions


class TestPerformanceConfigSerialized:
    """Verify RunOptions -> _options_to_dict includes performance keys."""

    def test_performance_config_serialized(self) -> None:
        perf = PerformanceConfig.create("fast")
        options = RunOptions(performance=perf)
        executor = CloudExecutor(experiment_id="test")
        d = executor._options_to_dict(options)
        assert "performance" in d
        assert d["performance"]["fast_atomic"] is True

    def test_performance_config_roundtrip(self) -> None:
        original = PerformanceConfig.create("aggressive")
        options = RunOptions(performance=original)
        executor = CloudExecutor(experiment_id="test")
        d = executor._options_to_dict(options)
        restored = PerformanceConfig.from_dict(d["performance"])
        assert restored.fast_atomic == original.fast_atomic
        assert restored.prune_ineligible == original.prune_ineligible
        assert restored.cache_dealer_quotes == original.cache_dealer_quotes
        assert restored.preview_buy == original.preview_buy
        assert restored.dirty_bucket_recompute == original.dirty_bucket_recompute
        assert restored.incremental_intentions == original.incremental_intentions
        assert restored.matching_order == original.matching_order
        assert restored.dealer_backend == original.dealer_backend

    def test_performance_none_excluded(self) -> None:
        options = RunOptions()
        executor = CloudExecutor(experiment_id="test")
        d = executor._options_to_dict(options)
        assert "performance" not in d

    def test_show_balances_serialized(self) -> None:
        options = RunOptions(show_balances=["agent_0", "agent_1"])
        executor = CloudExecutor(experiment_id="test")
        d = executor._options_to_dict(options)
        assert d["show_balances"] == ["agent_0", "agent_1"]

    def test_show_balances_none_excluded(self) -> None:
        options = RunOptions()
        executor = CloudExecutor(experiment_id="test")
        d = executor._options_to_dict(options)
        assert "show_balances" not in d
