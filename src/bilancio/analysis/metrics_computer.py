"""Metrics computation extracted from LocalExecutor for reuse.

This module provides a MetricsComputer class that can compute metrics from
simulation artifacts (events.jsonl, balances.csv) regardless of where the
simulation ran (local or remote).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from bilancio.storage.artifact_loaders import ArtifactLoader
from bilancio.analysis.loaders import read_events_jsonl, read_balances_csv
from bilancio.analysis.report import (
    compute_day_metrics,
    compute_run_level_metrics,
    summarize_day_metrics,
    write_day_metrics_csv,
    write_day_metrics_json,
    write_debtor_shares_csv,
    write_intraday_csv,
    write_metrics_html,
)


@dataclass
class MetricsBundle:
    """Bundle of computed metrics from a simulation run.

    Contains all computed metrics in structured form for further processing
    or writing to output files.
    """

    day_metrics: List[Dict[str, Any]]
    """Per-day metrics rows (S_t, Mbar_t, phi_t, etc.)."""

    debtor_shares: List[Dict[str, Any]]
    """Debtor shortfall shares (day, agent, DS_t)."""

    intraday: List[Dict[str, Any]]
    """Intraday settlement steps (day, step, P_prefix, etc.)."""

    summary: Dict[str, Any]
    """Aggregate summary from summarize_day_metrics (phi_total, delta_total, etc.)."""

    run_level_metrics: Dict[str, Any] = field(default_factory=dict)
    """Run-level metrics (cascade, contagion, etc.)."""


class MetricsComputer:
    """Computes metrics from simulation artifacts.

    Uses an ArtifactLoader to read events and balances, then applies
    the standard metrics computation pipeline from bilancio.analysis.report.

    Example:
        loader = LocalArtifactLoader(base_path=Path("/some/experiment/dir"))
        computer = MetricsComputer(loader)

        artifacts = {
            "events_jsonl": "runs/run_001/out/events.jsonl",
            "balances_csv": "runs/run_001/out/balances.csv",
        }
        bundle = computer.compute(artifacts)

        output_paths = computer.write_outputs(bundle, Path("/output/dir"))
    """

    def __init__(self, loader: ArtifactLoader) -> None:
        """Initialize with an artifact loader.

        Args:
            loader: An ArtifactLoader implementation for reading artifacts.
        """
        self.loader = loader

    def compute(
        self,
        artifacts: Dict[str, str],
        day_list: Optional[List[int]] = None,
    ) -> MetricsBundle:
        """Compute metrics from simulation artifacts.

        Args:
            artifacts: Dict mapping artifact names to references.
                Required: 'events_jsonl'
                Optional: 'balances_csv' (for M_t, G_t metrics)
            day_list: Optional list of days to compute metrics for.
                If None, days are inferred from events.

        Returns:
            MetricsBundle containing all computed metrics.

        Raises:
            KeyError: If required artifact 'events_jsonl' is missing.
            FileNotFoundError: If referenced artifact files don't exist.
        """
        # Load events (required)
        events_ref = artifacts.get("events_jsonl")
        if not events_ref:
            raise KeyError("Missing required artifact: 'events_jsonl'")

        events_text = self.loader.load_text(events_ref)
        # Parse JSONL from text - we need to handle this since read_events_jsonl takes a path
        events = self._parse_events_from_text(events_text)

        # Load balances (optional, enables M_t and G_t computation)
        balances_rows: Optional[List[Dict[str, Any]]] = None
        balances_ref = artifacts.get("balances_csv")
        if balances_ref and self.loader.exists(balances_ref):
            balances_text = self.loader.load_text(balances_ref)
            balances_rows = self._parse_balances_from_text(balances_text)

        # Compute day metrics
        result = compute_day_metrics(
            events=events,
            balances_rows=balances_rows,
            day_list=day_list,
        )

        # Compute summary
        summary = summarize_day_metrics(result["day_metrics"])

        # Compute run-level metrics (cascade, contagion)
        run_level = compute_run_level_metrics(events)
        # Merge into summary for downstream consumers
        summary["n_defaults"] = run_level["n_defaults"]
        summary["cascade_fraction"] = run_level["cascade_fraction"]

        return MetricsBundle(
            day_metrics=result["day_metrics"],
            debtor_shares=result["debtor_shares"],
            intraday=result["intraday"],
            summary=summary,
            run_level_metrics=run_level,
        )

    def write_outputs(
        self,
        bundle: MetricsBundle,
        output_dir: Path,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Write metrics bundle to output files.

        Args:
            bundle: The computed MetricsBundle to write.
            output_dir: Directory to write output files to.
            title: Optional title for HTML report.
            subtitle: Optional subtitle for HTML report.

        Returns:
            Dict mapping output types to their file paths:
                - 'metrics_csv': Path to metrics.csv
                - 'metrics_json': Path to metrics.json
                - 'debtor_shares_csv': Path to debtor_shares.csv
                - 'intraday_csv': Path to intraday.csv
                - 'metrics_html': Path to metrics.html
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}

        # Write metrics CSV
        metrics_csv_path = output_dir / "metrics.csv"
        write_day_metrics_csv(metrics_csv_path, bundle.day_metrics)
        paths["metrics_csv"] = metrics_csv_path

        # Write metrics JSON
        metrics_json_path = output_dir / "metrics.json"
        write_day_metrics_json(metrics_json_path, bundle.day_metrics)
        paths["metrics_json"] = metrics_json_path

        # Write debtor shares CSV
        debtor_shares_path = output_dir / "debtor_shares.csv"
        write_debtor_shares_csv(debtor_shares_path, bundle.debtor_shares)
        paths["debtor_shares_csv"] = debtor_shares_path

        # Write intraday CSV
        intraday_path = output_dir / "intraday.csv"
        write_intraday_csv(intraday_path, bundle.intraday)
        paths["intraday_csv"] = intraday_path

        # Write metrics HTML
        metrics_html_path = output_dir / "metrics.html"
        write_metrics_html(
            metrics_html_path,
            day_metrics=bundle.day_metrics,
            debtor_shares=bundle.debtor_shares,
            intraday=bundle.intraday,
            title=title,
            subtitle=subtitle,
            run_level_metrics=bundle.run_level_metrics,
        )
        paths["metrics_html"] = metrics_html_path

        return paths

    def _parse_events_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse events from JSONL text content.

        This mirrors read_events_jsonl but works from text instead of file path.
        """
        import json
        from decimal import Decimal

        events: List[Dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            evt = json.loads(line)
            # Normalize common fields (same as read_events_jsonl)
            if "amount" in evt:
                evt["amount"] = self._to_decimal(evt["amount"])
            if "day" in evt and evt["day"] is not None:
                try:
                    evt["day"] = int(evt["day"])
                except (ValueError, TypeError):
                    pass
            if "due_day" in evt and evt["due_day"] is not None:
                try:
                    evt["due_day"] = int(evt["due_day"])
                except (ValueError, TypeError):
                    pass
            events.append(evt)
        return events

    def _parse_balances_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse balances from CSV text content.

        This mirrors read_balances_csv but works from text instead of file path.
        """
        import csv
        from io import StringIO

        rows: List[Dict[str, Any]] = []
        reader = csv.DictReader(StringIO(text))
        for row in reader:
            rows.append(dict(row))
        return rows

    @staticmethod
    def _to_decimal(val: Any) -> Any:
        """Best-effort Decimal conversion for numeric values."""
        from decimal import Decimal

        if val is None:
            return Decimal("0")
        if isinstance(val, Decimal):
            return val
        if isinstance(val, bool):
            return Decimal(int(val))
        try:
            return Decimal(str(val))
        except (ValueError, ArithmeticError, TypeError):
            return Decimal("0")
