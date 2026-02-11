"""Supabase-based registry store implementation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any, Set, TYPE_CHECKING, cast

from .models import RegistryEntry, RunStatus

if TYPE_CHECKING:
    from supabase import Client

logger = logging.getLogger(__name__)


class SupabaseRegistryStore:
    """Store registry entries in Supabase PostgreSQL database.

    This implementation stores run metadata in the `runs` table and
    associated metrics in the `metrics` table. It implements the
    RegistryStore protocol for compatibility with the storage abstraction.

    The schema expects:
    - runs: run_id, job_id, status, kappa, concentration, mu, seed, etc.
    - metrics: run_id, job_id, delta_total, phi_total, raw_metrics, etc.
    """

    # Parameter fields that map directly to runs table columns
    RUNS_PARAM_COLUMNS = {
        "kappa", "concentration", "mu", "outside_mid_ratio", "seed", "regime"
    }

    # Metric fields that map directly to metrics table columns
    METRICS_COLUMNS = {
        "delta_total", "phi_total", "n_defaults", "n_clears",
        "time_to_stability", "trading_effect", "total_trades", "total_trade_volume"
    }

    def __init__(self, client: Optional["Client"] = None):
        """Initialize Supabase registry store.

        Args:
            client: Optional Supabase client. If not provided, will be
                    created lazily using get_supabase_client().
        """
        self._client = client
        self._initialized = client is not None

    @property
    def client(self) -> "Client":
        """Get or create the Supabase client.

        Returns:
            Supabase client instance.

        Raises:
            RuntimeError: If Supabase is not configured.
        """
        if not self._initialized:
            from bilancio.storage.supabase_client import (
                get_supabase_client,
                is_supabase_configured,
            )

            if not is_supabase_configured():
                raise RuntimeError(
                    "Supabase is not configured. Set SUPABASE_URL and "
                    "SUPABASE_KEY environment variables."
                )
            self._client = get_supabase_client()
            self._initialized = True

        return self._client  # type: ignore[return-value]

    def upsert(self, entry: RegistryEntry) -> None:
        """Insert or update a registry entry.

        This method performs two operations:
        1. Upserts the run record in the `runs` table
        2. Upserts the metrics record in the `metrics` table

        Args:
            entry: The registry entry to upsert.
        """
        try:
            # Build runs table row
            runs_row = self._build_runs_row(entry)

            # Upsert into runs table
            self.client.table("runs").upsert(
                runs_row,
                on_conflict="run_id"
            ).execute()

            # Build and upsert metrics if we have any
            if entry.metrics:
                metrics_row = self._build_metrics_row(entry)

                # Check if metrics row already exists for this run
                existing = self.client.table("metrics").select("id").eq(
                    "run_id", entry.run_id
                ).execute()

                if existing.data:
                    # Update existing metrics row
                    self.client.table("metrics").update(metrics_row).eq(
                        "run_id", entry.run_id
                    ).execute()
                else:
                    # Insert new metrics row
                    self.client.table("metrics").insert(metrics_row).execute()

            logger.debug(f"Upserted registry entry for run {entry.run_id}")

        except Exception as e:
            logger.warning(f"Failed to upsert registry entry {entry.run_id}: {e}")

    def get(self, experiment_id: str, run_id: str) -> Optional[RegistryEntry]:
        """Get a specific registry entry.

        Args:
            experiment_id: The job/experiment ID.
            run_id: The run ID to retrieve.

        Returns:
            RegistryEntry if found, None otherwise.
        """
        try:
            # Query runs table with metrics join
            result = self.client.table("runs").select(
                "*, metrics(*)"
            ).eq("run_id", run_id).eq("job_id", experiment_id).execute()

            if not result.data:
                return None

            row = cast(Dict[str, Any], result.data[0])
            return self._row_to_entry(row)

        except Exception as e:
            logger.warning(f"Failed to get registry entry {run_id}: {e}")
            return None

    def list_runs(self, experiment_id: str) -> List[str]:
        """List all run IDs for an experiment.

        Args:
            experiment_id: The job/experiment ID.

        Returns:
            List of run IDs.
        """
        try:
            result = self.client.table("runs").select("run_id").eq(
                "job_id", experiment_id
            ).execute()

            return [cast(Dict[str, Any], row)["run_id"] for row in result.data]

        except Exception as e:
            logger.warning(f"Failed to list runs for {experiment_id}: {e}")
            return []

    def get_completed_keys(
        self,
        experiment_id: str,
        key_fields: Optional[List[str]] = None
    ) -> Set[Any]:
        """Get set of completed parameter combinations.

        Used for sweep resumption to identify which parameter combinations
        have already been completed.

        Args:
            experiment_id: The job/experiment ID.
            key_fields: List of parameter field names to use as keys.
                       Defaults to ["seed", "kappa", "concentration"].

        Returns:
            Set of tuples containing completed parameter combinations.
        """
        if key_fields is None:
            key_fields = ["seed", "kappa", "concentration"]

        try:
            # Build select clause for requested fields
            select_fields = ",".join(key_fields)

            result = self.client.table("runs").select(select_fields).eq(
                "job_id", experiment_id
            ).eq("status", "completed").execute()

            completed: Set[Any] = set()
            for raw_row in result.data:
                row_dict = cast(Dict[str, Any], raw_row)
                key_values: List[Any] = []
                for field in key_fields:
                    val = row_dict.get(field)
                    if val is not None:
                        # Convert Decimal to float for consistent hashing
                        if isinstance(val, (Decimal, str)):
                            try:
                                val = float(val)
                            except (ValueError, TypeError):
                                pass
                        key_values.append(val)
                    else:
                        key_values.append(None)
                completed.add(tuple(key_values))

            return completed

        except Exception as e:
            logger.warning(
                f"Failed to get completed keys for {experiment_id}: {e}"
            )
            return set()

    def query(
        self,
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RegistryEntry]:
        """Query registry entries with optional filters.

        Args:
            experiment_id: The job/experiment ID.
            filters: Optional dict of field=value filters to apply.

        Returns:
            List of matching RegistryEntry objects.
        """
        try:
            # Start with base query including metrics
            query = self.client.table("runs").select("*, metrics(*)").eq(
                "job_id", experiment_id
            )

            # Apply filters
            if filters:
                for field, value in filters.items():
                    query = query.eq(field, value)

            result = query.execute()

            return [self._row_to_entry(cast(Dict[str, Any], row)) for row in result.data]

        except Exception as e:
            logger.warning(
                f"Failed to query registry for {experiment_id}: {e}"
            )
            return []

    def _build_runs_row(self, entry: RegistryEntry) -> Dict[str, Any]:
        """Build a row dict for the runs table.

        Args:
            entry: The registry entry to convert.

        Returns:
            Dict suitable for inserting into runs table.
        """
        row: Dict[str, Any] = {
            "run_id": entry.run_id,
            "job_id": entry.experiment_id,
            "status": entry.status.value,
        }

        # Add error if present
        if entry.error:
            row["error"] = entry.error

        # Add timestamps based on status
        now = datetime.now(timezone.utc).isoformat()
        if entry.status == RunStatus.COMPLETED:
            row["completed_at"] = now
        elif entry.status == RunStatus.FAILED:
            row["completed_at"] = now

        # Map parameters to runs table columns
        for param, value in entry.parameters.items():
            if param in self.RUNS_PARAM_COLUMNS:
                row[param] = self._convert_value(value)

        # Store Modal volume path if present in artifact_paths
        if entry.artifact_paths:
            # Derive volume path from any artifact path
            for artifact_path in entry.artifact_paths.values():
                if artifact_path:
                    # Extract base path (e.g., "experiment/runs/run_id")
                    parts = artifact_path.split("/")
                    if "runs" in parts:
                        idx = parts.index("runs")
                        if idx + 2 <= len(parts):
                            row["modal_volume_path"] = "/".join(parts[:idx + 2])
                            break

        return row

    def _build_metrics_row(self, entry: RegistryEntry) -> Dict[str, Any]:
        """Build a row dict for the metrics table.

        Args:
            entry: The registry entry to convert.

        Returns:
            Dict suitable for inserting into metrics table.
        """
        row: Dict[str, Any] = {
            "run_id": entry.run_id,
            "job_id": entry.experiment_id,
            "raw_metrics": entry.metrics,  # Store full metrics as JSONB
        }

        # Map known metrics to dedicated columns
        for metric, value in entry.metrics.items():
            if metric in self.METRICS_COLUMNS:
                row[metric] = self._convert_value(value)

        return row

    def _row_to_entry(self, row: Dict[str, Any]) -> RegistryEntry:
        """Convert a database row to RegistryEntry.

        Args:
            row: Dict from Supabase query result.

        Returns:
            RegistryEntry object.
        """
        # Extract parameters from runs table columns
        parameters: Dict[str, Any] = {}
        for param in self.RUNS_PARAM_COLUMNS:
            if param in row and row[param] is not None:
                parameters[param] = self._parse_value(row[param])

        # Extract metrics from nested metrics relation or raw_metrics
        metrics: Dict[str, Any] = {}
        metrics_data = row.get("metrics")

        if metrics_data:
            # metrics is a list from the join, take first item
            if isinstance(metrics_data, list) and metrics_data:
                metrics_row = metrics_data[0]
            else:
                metrics_row = metrics_data

            # Prefer raw_metrics if available (has all metrics)
            raw_metrics = metrics_row.get("raw_metrics")
            if raw_metrics and isinstance(raw_metrics, dict):
                metrics = raw_metrics
            else:
                # Fall back to individual columns
                for metric in self.METRICS_COLUMNS:
                    if metric in metrics_row and metrics_row[metric] is not None:
                        metrics[metric] = self._parse_value(metrics_row[metric])

        # Build artifact paths from modal_volume_path
        artifact_paths: Dict[str, str] = {}
        volume_path = row.get("modal_volume_path")
        if volume_path:
            # Reconstruct standard artifact paths
            artifact_paths["scenario_yaml"] = f"{volume_path}/scenario.yaml"
            artifact_paths["events_jsonl"] = f"{volume_path}/out/events.jsonl"
            artifact_paths["balances_csv"] = f"{volume_path}/out/balances.csv"
            artifact_paths["metrics_csv"] = f"{volume_path}/out/metrics.csv"
            artifact_paths["run_html"] = f"{volume_path}/run.html"

        return RegistryEntry(
            run_id=row["run_id"],
            experiment_id=row.get("job_id", ""),
            status=RunStatus(row.get("status", "completed")),
            parameters=parameters,
            metrics=metrics,
            artifact_paths=artifact_paths,
            error=row.get("error"),
        )

    @staticmethod
    def _convert_value(value: Any) -> Any:
        """Convert Python value for Supabase storage.

        Args:
            value: Value to convert.

        Returns:
            Converted value suitable for database storage.
        """
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, bool):
            return value
        return value

    @staticmethod
    def _parse_value(value: Any) -> Any:
        """Parse database value to Python type.

        Args:
            value: Value from database.

        Returns:
            Parsed Python value.
        """
        if isinstance(value, str):
            # Try to parse as number
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
        if isinstance(value, Decimal):
            return float(value)
        return value
