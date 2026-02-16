"""Tests for file-based storage implementations."""

import json
from pathlib import Path

import pytest

from bilancio.storage.file_store import FileRegistryStore, FileResultStore
from bilancio.storage.models import (
    RegistryEntry,
    RunArtifacts,
    RunResult,
    RunStatus,
)


class TestFileResultStore:
    """Tests for FileResultStore."""

    def test_init_creates_base_dir(self, tmp_path):
        """Test that initialization creates the base directory."""
        base_dir = tmp_path / "results"
        assert not base_dir.exists()
        FileResultStore(base_dir)
        assert base_dir.exists()

    def test_init_accepts_string_path(self, tmp_path):
        """Test that initialization accepts string path."""
        base_dir = str(tmp_path / "results")
        FileResultStore(base_dir)
        assert Path(base_dir).exists()

    def test_save_artifact_creates_correct_structure(self, tmp_path):
        """Test that save_artifact creates correct directory structure."""
        store = FileResultStore(tmp_path)
        content = b"test content"

        store.save_artifact(
            experiment_id="exp_001",
            run_id="run_001",
            name="events.jsonl",
            content=content,
        )

        # Check that the file exists in the correct location
        expected_path = tmp_path / "exp_001" / "runs" / "run_001" / "out" / "events.jsonl"
        assert expected_path.exists()
        assert expected_path.read_bytes() == content

    def test_save_artifact_scenario_yaml_at_run_level(self, tmp_path):
        """Test that scenario.yaml is saved at run level, not in out/."""
        store = FileResultStore(tmp_path)
        content = b"scenario: test"

        store.save_artifact(
            experiment_id="exp_001",
            run_id="run_001",
            name="scenario.yaml",
            content=content,
        )

        # scenario.yaml should be at run level, not in out/
        expected_path = tmp_path / "exp_001" / "runs" / "run_001" / "scenario.yaml"
        assert expected_path.exists()
        out_path = tmp_path / "exp_001" / "runs" / "run_001" / "out" / "scenario.yaml"
        assert not out_path.exists()

    def test_save_artifact_returns_relative_path(self, tmp_path):
        """Test that save_artifact returns correct relative path."""
        store = FileResultStore(tmp_path)

        rel_path = store.save_artifact(
            experiment_id="exp_001",
            run_id="run_001",
            name="events.jsonl",
            content=b"test",
        )

        # Normalize path separators for cross-platform compatibility
        assert rel_path.replace("\\", "/") == "exp_001/runs/run_001/out/events.jsonl"

    def test_save_artifact_scenario_returns_relative_path(self, tmp_path):
        """Test that save_artifact returns correct relative path for scenario."""
        store = FileResultStore(tmp_path)

        rel_path = store.save_artifact(
            experiment_id="exp_001",
            run_id="run_001",
            name="scenario.yaml",
            content=b"test",
        )

        # Normalize path separators for cross-platform compatibility
        assert rel_path.replace("\\", "/") == "exp_001/runs/run_001/scenario.yaml"

    def test_save_run_creates_result_json(self, tmp_path):
        """Test that save_run creates result.json with correct format."""
        store = FileResultStore(tmp_path)
        result = RunResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 42, "kappa": 0.1},
            metrics={"phi_total": 100.0},
            artifacts=RunArtifacts(scenario_yaml="exp/runs/run_001/scenario.yaml"),
            execution_time_ms=5000,
        )

        store.save_run("exp_001", result)

        result_path = tmp_path / "exp_001" / "runs" / "run_001" / "result.json"
        assert result_path.exists()

        data = json.loads(result_path.read_text())
        assert data["run_id"] == "run_001"
        assert data["status"] == "completed"
        assert data["parameters"] == {"seed": 42, "kappa": 0.1}
        assert data["metrics"] == {"phi_total": 100.0}
        assert data["artifacts"]["scenario_yaml"] == "exp/runs/run_001/scenario.yaml"
        assert data["execution_time_ms"] == 5000
        assert data["error"] is None

    def test_save_run_with_error(self, tmp_path):
        """Test that save_run correctly saves error field."""
        store = FileResultStore(tmp_path)
        result = RunResult(
            run_id="run_fail",
            status=RunStatus.FAILED,
            error="Simulation diverged",
        )

        store.save_run("exp_001", result)

        result_path = tmp_path / "exp_001" / "runs" / "run_fail" / "result.json"
        data = json.loads(result_path.read_text())
        assert data["status"] == "failed"
        assert data["error"] == "Simulation diverged"

    def test_load_run_returns_none_for_nonexistent(self, tmp_path):
        """Test that load_run returns None for non-existent run."""
        store = FileResultStore(tmp_path)
        result = store.load_run("exp_001", "nonexistent_run")
        assert result is None

    def test_load_run_returns_none_for_nonexistent_experiment(self, tmp_path):
        """Test that load_run returns None for non-existent experiment."""
        store = FileResultStore(tmp_path)
        result = store.load_run("nonexistent_exp", "run_001")
        assert result is None

    def test_load_run_correctly_deserializes(self, tmp_path):
        """Test that load_run correctly deserializes saved run."""
        store = FileResultStore(tmp_path)
        original = RunResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 42, "kappa": 0.1},
            metrics={"phi_total": 100.0, "delta_total": 50.0},
            artifacts=RunArtifacts(
                scenario_yaml="exp/runs/run_001/scenario.yaml",
                events_jsonl="exp/runs/run_001/out/events.jsonl",
            ),
            execution_time_ms=5000,
        )

        store.save_run("exp_001", original)
        loaded = store.load_run("exp_001", "run_001")

        assert loaded is not None
        assert loaded.run_id == original.run_id
        assert loaded.status == original.status
        assert loaded.parameters == original.parameters
        assert loaded.metrics == original.metrics
        assert loaded.artifacts.scenario_yaml == original.artifacts.scenario_yaml
        assert loaded.artifacts.events_jsonl == original.artifacts.events_jsonl
        assert loaded.execution_time_ms == original.execution_time_ms

    def test_load_artifact_returns_correct_content(self, tmp_path):
        """Test that load_artifact returns correct content."""
        store = FileResultStore(tmp_path)
        content = b"test artifact content"

        rel_path = store.save_artifact(
            experiment_id="exp_001",
            run_id="run_001",
            name="events.jsonl",
            content=content,
        )

        loaded_content = store.load_artifact(rel_path)
        assert loaded_content == content

    def test_roundtrip_save_then_load(self, tmp_path):
        """Test complete round-trip: save then load preserves all data."""
        store = FileResultStore(tmp_path)

        # Save artifacts
        scenario_content = b"agents:\n  - name: Bank1"
        events_content = b'{"day": 1, "event": "payment"}\n'

        scenario_path = store.save_artifact("exp_001", "run_001", "scenario.yaml", scenario_content)
        events_path = store.save_artifact("exp_001", "run_001", "events.jsonl", events_content)

        # Save run result
        original = RunResult(
            run_id="run_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 42, "kappa": 0.15, "n_agents": 10},
            metrics={"phi_total": 150.0, "delta_total": 75.0, "time_to_stability": 5},
            artifacts=RunArtifacts(
                scenario_yaml=scenario_path,
                events_jsonl=events_path,
            ),
            execution_time_ms=10000,
        )
        store.save_run("exp_001", original)

        # Load everything back
        loaded = store.load_run("exp_001", "run_001")
        loaded_scenario = store.load_artifact(loaded.artifacts.scenario_yaml)
        loaded_events = store.load_artifact(loaded.artifacts.events_jsonl)

        # Verify everything matches
        assert loaded.run_id == original.run_id
        assert loaded.status == original.status
        assert loaded.parameters == original.parameters
        assert loaded.metrics == original.metrics
        assert loaded.execution_time_ms == original.execution_time_ms
        assert loaded_scenario == scenario_content
        assert loaded_events == events_content


class TestFileRegistryStore:
    """Tests for FileRegistryStore."""

    def test_upsert_creates_registry_file_and_directory(self, tmp_path):
        """Test that upsert creates registry file and directory structure."""
        store = FileRegistryStore(tmp_path)
        entry = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
        )

        store.upsert(entry)

        registry_path = tmp_path / "exp_001" / "registry" / "experiments.csv"
        assert registry_path.exists()

    def test_upsert_updates_existing_entry(self, tmp_path):
        """Test that upsert updates existing entry (not duplicates)."""
        store = FileRegistryStore(tmp_path)

        # First upsert
        entry1 = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.RUNNING,
            parameters={"seed": 42},
        )
        store.upsert(entry1)

        # Second upsert with same run_id
        entry2 = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 42},
            metrics={"phi_total": 100.0},
        )
        store.upsert(entry2)

        # Should only have one entry
        run_ids = store.list_runs("exp_001")
        assert len(run_ids) == 1
        assert run_ids[0] == "run_001"

        # Entry should be updated
        loaded = store.get("exp_001", "run_001")
        assert loaded.status == RunStatus.COMPLETED

    def test_get_returns_none_for_nonexistent_entry(self, tmp_path):
        """Test that get returns None for non-existent entry."""
        store = FileRegistryStore(tmp_path)
        result = store.get("exp_001", "nonexistent_run")
        assert result is None

    def test_get_returns_none_for_nonexistent_experiment(self, tmp_path):
        """Test that get returns None for non-existent experiment."""
        store = FileRegistryStore(tmp_path)
        result = store.get("nonexistent_exp", "run_001")
        assert result is None

    def test_get_returns_correct_entry(self, tmp_path):
        """Test that get returns correct entry."""
        store = FileRegistryStore(tmp_path)
        entry = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 42, "kappa": 0.1},
            metrics={"phi_total": 100.0},
            artifact_paths={"scenario_yaml": "exp_001/runs/run_001/scenario.yaml"},
        )
        store.upsert(entry)

        loaded = store.get("exp_001", "run_001")

        assert loaded is not None
        assert loaded.run_id == "run_001"
        assert loaded.experiment_id == "exp_001"
        assert loaded.status == RunStatus.COMPLETED
        assert loaded.parameters["seed"] == 42

    def test_list_runs_returns_all_run_ids(self, tmp_path):
        """Test that list_runs returns all run IDs."""
        store = FileRegistryStore(tmp_path)

        # Add multiple entries
        for i in range(5):
            entry = RegistryEntry(
                run_id=f"run_{i:03d}",
                experiment_id="exp_001",
                status=RunStatus.COMPLETED,
            )
            store.upsert(entry)

        run_ids = store.list_runs("exp_001")

        assert len(run_ids) == 5
        assert set(run_ids) == {"run_000", "run_001", "run_002", "run_003", "run_004"}

    def test_list_runs_returns_empty_for_nonexistent_experiment(self, tmp_path):
        """Test that list_runs returns empty list for non-existent experiment."""
        store = FileRegistryStore(tmp_path)
        run_ids = store.list_runs("nonexistent_exp")
        assert run_ids == []

    def test_get_completed_keys_returns_correct_keys(self, tmp_path):
        """Test that get_completed_keys returns correct keys."""
        store = FileRegistryStore(tmp_path)

        # Add some completed entries
        for seed in [1, 2, 3]:
            for kappa in [0.1, 0.2]:
                entry = RegistryEntry(
                    run_id=f"run_s{seed}_k{kappa}",
                    experiment_id="exp_001",
                    status=RunStatus.COMPLETED,
                    parameters={"seed": seed, "kappa": kappa, "concentration": 0.5},
                )
                store.upsert(entry)

        completed = store.get_completed_keys("exp_001")

        # Default key_fields is ["seed", "kappa", "concentration"]
        assert len(completed) == 6
        assert (1, 0.1, 0.5) in completed
        assert (2, 0.2, 0.5) in completed

    def test_get_completed_keys_excludes_non_completed(self, tmp_path):
        """Test that get_completed_keys excludes non-completed runs."""
        store = FileRegistryStore(tmp_path)

        # Add completed entry
        entry1 = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 1, "kappa": 0.1, "concentration": 0.5},
        )
        store.upsert(entry1)

        # Add failed entry
        entry2 = RegistryEntry(
            run_id="run_002",
            experiment_id="exp_001",
            status=RunStatus.FAILED,
            parameters={"seed": 2, "kappa": 0.1, "concentration": 0.5},
        )
        store.upsert(entry2)

        completed = store.get_completed_keys("exp_001")

        assert len(completed) == 1
        assert (1, 0.1, 0.5) in completed
        assert (2, 0.1, 0.5) not in completed

    def test_get_completed_keys_with_custom_key_fields(self, tmp_path):
        """Test that get_completed_keys works with custom key_fields."""
        store = FileRegistryStore(tmp_path)

        entry = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 42, "n_agents": 10, "mu": 0.5},
        )
        store.upsert(entry)

        completed = store.get_completed_keys("exp_001", key_fields=["seed", "n_agents"])

        assert len(completed) == 1
        assert (42, 10) in completed

    def test_get_completed_keys_returns_empty_for_nonexistent(self, tmp_path):
        """Test that get_completed_keys returns empty set for non-existent experiment."""
        store = FileRegistryStore(tmp_path)
        completed = store.get_completed_keys("nonexistent_exp")
        assert completed == set()

    def test_query_with_no_filters_returns_all(self, tmp_path):
        """Test that query with no filters returns all entries."""
        store = FileRegistryStore(tmp_path)

        # Add multiple entries with different statuses
        for i, status in enumerate([RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.COMPLETED]):
            entry = RegistryEntry(
                run_id=f"run_{i:03d}",
                experiment_id="exp_001",
                status=status,
            )
            store.upsert(entry)

        results = store.query("exp_001")

        assert len(results) == 3
        run_ids = {r.run_id for r in results}
        assert run_ids == {"run_000", "run_001", "run_002"}

    def test_query_with_filters_returns_matching(self, tmp_path):
        """Test that query with filters returns matching entries only."""
        store = FileRegistryStore(tmp_path)

        # Add entries with different parameters
        for seed in [1, 2, 3]:
            for kappa in [0.1, 0.2]:
                entry = RegistryEntry(
                    run_id=f"run_s{seed}_k{kappa}",
                    experiment_id="exp_001",
                    status=RunStatus.COMPLETED,
                    parameters={"seed": seed, "kappa": kappa},
                )
                store.upsert(entry)

        # Filter by kappa
        results = store.query("exp_001", filters={"kappa": 0.1})

        assert len(results) == 3
        for r in results:
            assert r.parameters["kappa"] == 0.1

    def test_query_with_status_filter(self, tmp_path):
        """Test that query can filter by status."""
        store = FileRegistryStore(tmp_path)

        # Add entries with different statuses
        entry1 = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
        )
        entry2 = RegistryEntry(
            run_id="run_002",
            experiment_id="exp_001",
            status=RunStatus.FAILED,
        )
        entry3 = RegistryEntry(
            run_id="run_003",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
        )
        store.upsert(entry1)
        store.upsert(entry2)
        store.upsert(entry3)

        results = store.query("exp_001", filters={"status": "completed"})

        assert len(results) == 2
        assert all(r.status == RunStatus.COMPLETED for r in results)

    def test_query_returns_empty_for_nonexistent_experiment(self, tmp_path):
        """Test that query returns empty list for non-existent experiment."""
        store = FileRegistryStore(tmp_path)
        results = store.query("nonexistent_exp")
        assert results == []

    def test_query_with_multiple_filters(self, tmp_path):
        """Test that query with multiple filters works correctly."""
        store = FileRegistryStore(tmp_path)

        # Add various entries
        for seed in [1, 2]:
            for kappa in [0.1, 0.2]:
                for status in [RunStatus.COMPLETED, RunStatus.FAILED]:
                    entry = RegistryEntry(
                        run_id=f"run_s{seed}_k{kappa}_{status.value}",
                        experiment_id="exp_001",
                        status=status,
                        parameters={"seed": seed, "kappa": kappa},
                    )
                    store.upsert(entry)

        # Filter by multiple criteria
        results = store.query("exp_001", filters={"seed": 1, "status": "completed"})

        assert len(results) == 2
        for r in results:
            assert r.parameters["seed"] == 1
            assert r.status == RunStatus.COMPLETED

    def test_upsert_preserves_existing_entries(self, tmp_path):
        """Test that upsert preserves other entries when adding new ones."""
        store = FileRegistryStore(tmp_path)

        # Add first entry
        entry1 = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 1},
        )
        store.upsert(entry1)

        # Add second entry
        entry2 = RegistryEntry(
            run_id="run_002",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 2},
        )
        store.upsert(entry2)

        # Both entries should exist
        loaded1 = store.get("exp_001", "run_001")
        loaded2 = store.get("exp_001", "run_002")

        assert loaded1 is not None
        assert loaded1.parameters["seed"] == 1
        assert loaded2 is not None
        assert loaded2.parameters["seed"] == 2

    def test_dynamic_fields_are_preserved(self, tmp_path):
        """Test that dynamically added fields are preserved in registry."""
        store = FileRegistryStore(tmp_path)

        # Add entry with non-standard parameters
        entry = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"custom_param": "custom_value", "another_param": 123},
        )
        store.upsert(entry)

        # Load it back
        loaded = store.get("exp_001", "run_001")

        assert loaded is not None
        assert "custom_param" in loaded.parameters
        assert loaded.parameters["custom_param"] == "custom_value"

    def test_metrics_roundtrip_through_csv(self, tmp_path):
        """Test that metrics are correctly stored and retrieved from CSV.

        This test verifies the fix for the bug where metrics were incorrectly
        assigned to parameters instead of metrics dict in _row_to_entry.
        """
        store = FileRegistryStore(tmp_path)

        # Create entry with all standard metrics
        entry = RegistryEntry(
            run_id="run_001",
            experiment_id="exp_001",
            status=RunStatus.COMPLETED,
            parameters={"seed": 42, "kappa": 0.1},
            metrics={
                "phi_total": 150.5,
                "delta_total": 75.25,
                "time_to_stability": 12.0,
            },
        )
        store.upsert(entry)

        # Load it back
        loaded = store.get("exp_001", "run_001")

        assert loaded is not None
        # Verify metrics are in metrics dict (not parameters)
        assert "phi_total" in loaded.metrics
        assert "delta_total" in loaded.metrics
        assert "time_to_stability" in loaded.metrics
        assert loaded.metrics["phi_total"] == 150.5
        assert loaded.metrics["delta_total"] == 75.25
        assert loaded.metrics["time_to_stability"] == 12.0

        # Verify metrics are NOT in parameters
        assert "phi_total" not in loaded.parameters
        assert "delta_total" not in loaded.parameters
        assert "time_to_stability" not in loaded.parameters

        # Verify parameters are still in parameters dict
        assert loaded.parameters["seed"] == 42
        assert loaded.parameters["kappa"] == 0.1


class TestFileStoreSecurityValidation:
    """Tests for security validations in file stores."""

    def test_load_artifact_rejects_path_traversal(self, tmp_path):
        """Test that load_artifact rejects path traversal attempts."""
        store = FileResultStore(tmp_path)

        # Create a legitimate file
        (tmp_path / "test.txt").write_bytes(b"test")

        # Try to traverse outside base_dir
        with pytest.raises(ValueError, match="path traversal detected"):
            store.load_artifact("../../../etc/passwd")

    def test_load_artifact_rejects_absolute_path_traversal(self, tmp_path):
        """Test that load_artifact rejects absolute path traversal."""
        store = FileResultStore(tmp_path)

        with pytest.raises(ValueError, match="path traversal detected"):
            store.load_artifact("foo/../../../etc/passwd")

    def test_save_artifact_rejects_invalid_experiment_id(self, tmp_path):
        """Test that save_artifact rejects invalid experiment_id."""
        store = FileResultStore(tmp_path)

        with pytest.raises(ValueError, match="Invalid experiment_id"):
            store.save_artifact(
                experiment_id="../malicious",
                run_id="run_001",
                name="test.txt",
                content=b"test",
            )

    def test_save_artifact_rejects_invalid_run_id(self, tmp_path):
        """Test that save_artifact rejects invalid run_id."""
        store = FileResultStore(tmp_path)

        with pytest.raises(ValueError, match="Invalid run_id"):
            store.save_artifact(
                experiment_id="exp_001",
                run_id="../../../etc",
                name="test.txt",
                content=b"test",
            )

    def test_registry_rejects_invalid_experiment_id(self, tmp_path):
        """Test that registry operations reject invalid experiment_id."""
        store = FileRegistryStore(tmp_path)

        with pytest.raises(ValueError, match="Invalid experiment_id"):
            store.get("../malicious", "run_001")

    def test_valid_ids_with_special_chars_allowed(self, tmp_path):
        """Test that valid IDs with dashes, underscores, dots are allowed."""
        store = FileResultStore(tmp_path)

        # These should all work
        ref = store.save_artifact(
            experiment_id="exp-001_test.v2",
            run_id="run.001-abc_123",
            name="test.txt",
            content=b"test",
        )
        assert "exp-001_test.v2" in ref
        assert "run.001-abc_123" in ref
