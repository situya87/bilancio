"""File-based result storage implementation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Union

from .models import RunResult, RegistryEntry, RunArtifacts, RunStatus


# Pattern for valid IDs (alphanumeric, dash, underscore, dot)
_VALID_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')


def _validate_id(value: str, field_name: str) -> None:
    """Validate an ID to prevent path traversal attacks.

    Args:
        value: The ID value to validate.
        field_name: Name of the field for error messages.

    Raises:
        ValueError: If the ID contains invalid characters.
    """
    if not value:
        return  # Empty is allowed for some cases (e.g., empty experiment_id)
    if not _VALID_ID_PATTERN.match(value):
        raise ValueError(
            f"Invalid {field_name}: '{value}'. "
            f"Must contain only alphanumeric characters, dashes, underscores, and dots."
        )


class FileResultStore:
    """Store results as files on local filesystem."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, experiment_id: str, run_id: str) -> Path:
        _validate_id(experiment_id, "experiment_id")
        _validate_id(run_id, "run_id")
        return self.base_dir / experiment_id / "runs" / run_id

    def save_artifact(
        self,
        experiment_id: str,
        run_id: str,
        name: str,
        content: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Save artifact to file, return relative path."""
        run_dir = self._run_dir(experiment_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Determine subdirectory based on artifact type
        if name in ("scenario.yaml",):
            path = run_dir / name
        else:
            out_dir = run_dir / "out"
            out_dir.mkdir(exist_ok=True)
            path = out_dir / name

        path.write_bytes(content)
        return str(path.relative_to(self.base_dir))

    def save_run(self, experiment_id: str, result: RunResult) -> None:
        """Save run result metadata as JSON."""
        run_dir = self._run_dir(experiment_id, result.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        meta_path = run_dir / "result.json"
        meta = {
            "run_id": result.run_id,
            "status": result.status.value,
            "parameters": result.parameters,
            "metrics": result.metrics,
            "artifacts": {
                k: v for k, v in vars(result.artifacts).items() if v is not None
            },
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    def load_run(self, experiment_id: str, run_id: str) -> Optional[RunResult]:
        """Load run result from file."""
        meta_path = self._run_dir(experiment_id, run_id) / "result.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text())
        return RunResult(
            run_id=meta["run_id"],
            status=RunStatus(meta["status"]),
            parameters=meta.get("parameters", {}),
            metrics=meta.get("metrics", {}),
            artifacts=RunArtifacts(**meta.get("artifacts", {})),
            error=meta.get("error"),
            execution_time_ms=meta.get("execution_time_ms"),
        )

    def load_artifact(self, reference: str) -> bytes:
        """Load artifact content by path.

        Raises:
            ValueError: If reference attempts path traversal outside base_dir.
        """
        path = self.base_dir / reference
        # Validate path doesn't escape base_dir (security: prevent path traversal)
        try:
            path.resolve().relative_to(self.base_dir.resolve())
        except ValueError:
            raise ValueError(f"Invalid artifact reference: path traversal detected in '{reference}'")
        return path.read_bytes()


class FileRegistryStore:
    """Store registry as CSV file."""

    # Default fields - will be extended dynamically
    DEFAULT_FIELDS = [
        "run_id", "experiment_id", "phase", "status", "error",
        # Common parameters
        "seed", "n_agents", "kappa", "concentration", "mu", "monotonicity",
        "maturity_days", "Q_total", "S1", "L0", "default_handling", "dealer_enabled",
        # Common metrics
        "phi_total", "delta_total", "time_to_stability",
        # Common artifact paths
        "scenario_yaml", "events_jsonl", "balances_csv", "metrics_csv",
        "metrics_html", "run_html",
    ]

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    def _registry_path(self, experiment_id: str) -> Path:
        _validate_id(experiment_id, "experiment_id")
        if experiment_id:
            return self.base_dir / experiment_id / "registry" / "experiments.csv"
        # Empty experiment_id means use base_dir directly
        return self.base_dir / "registry" / "experiments.csv"

    def upsert(self, entry: RegistryEntry) -> None:
        """Insert or update registry entry."""
        registry_path = self._registry_path(entry.experiment_id)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing entries
        entries: Dict[str, Dict[str, str]] = {}
        fieldnames = list(self.DEFAULT_FIELDS)

        if registry_path.exists():
            with open(registry_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    fieldnames = list(reader.fieldnames)
                for row in reader:
                    entries[row["run_id"]] = dict(row)

        # Build row from entry
        new_row: Dict[str, str] = {
            "run_id": entry.run_id,
            "experiment_id": entry.experiment_id,
            "status": entry.status.value,
            "error": entry.error or "",
        }

        # Add parameters, metrics, artifact paths
        for k, v in entry.parameters.items():
            new_row[k] = str(v) if v is not None else ""
            if k not in fieldnames:
                fieldnames.append(k)

        for k, v in entry.metrics.items():
            new_row[k] = str(v) if v is not None else ""
            if k not in fieldnames:
                fieldnames.append(k)

        for k, v in entry.artifact_paths.items():
            new_row[k] = str(v) if v is not None else ""
            if k not in fieldnames:
                fieldnames.append(k)

        entries[entry.run_id] = new_row

        # Write back
        with open(registry_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in entries.values():
                writer.writerow(r)

    def get(self, experiment_id: str, run_id: str) -> Optional[RegistryEntry]:
        """Get registry entry by ID."""
        registry_path = self._registry_path(experiment_id)
        if not registry_path.exists():
            return None

        with open(registry_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["run_id"] == run_id:
                    return self._row_to_entry(row)
        return None

    def list_runs(self, experiment_id: str) -> List[str]:
        """List all run IDs."""
        registry_path = self._registry_path(experiment_id)
        if not registry_path.exists():
            return []

        with open(registry_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return [row["run_id"] for row in reader]

    def get_completed_keys(
        self,
        experiment_id: str,
        key_fields: Optional[List[str]] = None
    ) -> Set[Any]:
        """Get completed parameter keys for resumption."""
        if key_fields is None:
            key_fields = ["seed", "kappa", "concentration"]

        registry_path = self._registry_path(experiment_id)
        if not registry_path.exists():
            return set()

        completed = set()
        with open(registry_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "completed":
                    key_values: List[Union[int, float, str]] = []
                    for field in key_fields:
                        val = row.get(field, "")
                        # Try to parse as number for consistent hashing
                        try:
                            if "." in val:
                                key_values.append(float(val))
                            else:
                                key_values.append(int(val))
                        except (ValueError, TypeError):
                            key_values.append(val)
                    completed.add(tuple(key_values))
        return completed

    def query(
        self,
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RegistryEntry]:
        """Query registry with filters."""
        registry_path = self._registry_path(experiment_id)
        if not registry_path.exists():
            return []

        results = []
        with open(registry_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if filters:
                    match = all(
                        str(row.get(k, "")) == str(v)
                        for k, v in filters.items()
                    )
                    if not match:
                        continue
                results.append(self._row_to_entry(row))
        return results

    def _row_to_entry(self, row: Dict[str, str]) -> RegistryEntry:
        """Convert CSV row to RegistryEntry."""
        # Known parameter, metric, and artifact keys
        param_keys = {"phase", "seed", "n_agents", "kappa", "concentration", "mu",
                      "monotonicity", "maturity_days", "Q_total", "S1", "L0",
                      "default_handling", "dealer_enabled"}
        metric_keys = {"phi_total", "delta_total", "time_to_stability"}
        artifact_keys = {"scenario_yaml", "events_jsonl", "balances_csv",
                        "metrics_csv", "metrics_json", "metrics_html", "run_html",
                        "dealer_metrics_json", "trades_csv", "repayment_events_csv"}
        meta_keys = {"run_id", "experiment_id", "status", "error"}

        parameters: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}
        artifact_paths: Dict[str, str] = {}

        for k, v in row.items():
            if not v or k in meta_keys:
                continue
            if k in param_keys:
                # Try to parse as number
                try:
                    if "." in v:
                        parameters[k] = float(v)
                    else:
                        parameters[k] = int(v)
                except ValueError:
                    parameters[k] = v
            elif k in metric_keys:
                try:
                    metrics[k] = float(v)
                except ValueError:
                    metrics[k] = v
            elif k in artifact_keys:
                artifact_paths[k] = v
            else:
                # Unknown field - add to parameters
                parameters[k] = v

        return RegistryEntry(
            run_id=row["run_id"],
            experiment_id=row.get("experiment_id", ""),
            status=RunStatus(row.get("status", "completed")),
            parameters=parameters,
            metrics=metrics,
            artifact_paths=artifact_paths,
            error=row.get("error") or None,
        )
