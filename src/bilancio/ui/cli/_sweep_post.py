"""Shared post-sweep analysis helpers for CLI sweep commands."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import click

from ._common import optional_extra_message

VALID_POST_ANALYSES = (
    # Data analyses (CSV/JSON)
    "frontier",
    "strategy_outcomes",
    "dealer_usage",
    "mechanism_activity",
    "contagion",
    "credit_creation",
    "network",
    "pricing",
    "beliefs",
    "funding",
    # Visualizations (HTML)
    "drilldowns",
    "deltas",
    "dynamics",
    "narrative",
    "treynor",
    "comparison",
    "report",
    "notebook",
)


def _arm_runs_dir(out_dir: Path, sweep_type: str, arm_label: str) -> Path:
    """Resolve the runs directory for a given sweep type and arm.

    The actual run directories live under arm-specific subdirectories,
    not a flat ``out_dir/runs`` path.
    """
    _DIR_MAP: dict[tuple[str, str], str] = {
        ("dealer", "active"): "active",
        ("dealer", "passive"): "passive",
        ("bank", "lend"): "bank_lend",
        ("bank", "idle"): "bank_idle",
        ("nbfi", "lend"): "nbfi_lend",
        ("nbfi", "idle"): "nbfi_idle",
    }
    subdir = _DIR_MAP.get((sweep_type, arm_label), arm_label)
    return out_dir / subdir / "runs"


def _run_per_run_analysis(
    out_dir: Path,
    analysis_name: str,
    sweep_type: str,
) -> Path | None:
    """Run a per-run analysis across all runs and aggregate the results."""
    csv_path = out_dir / "aggregate" / "comparison.csv"
    if not csv_path.is_file():
        return None

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    if sweep_type == "dealer":
        arm_cols = [("passive", "passive_run_id"), ("active", "active_run_id")]
    elif sweep_type in ("bank", "nbfi"):
        arm_cols = [("idle", "idle_run_id"), ("lend", "lend_run_id")]
    else:
        arm_cols = [("passive", "passive_run_id"), ("active", "active_run_id")]

    all_results: list[dict[str, Any]] = []

    for row in rows:
        kappa = row.get("kappa", "")
        concentration = row.get("concentration", "")

        for arm_label, col_name in arm_cols:
            run_id = row.get(col_name, "")
            if not run_id:
                continue

            runs_dir = _arm_runs_dir(out_dir, sweep_type, arm_label)
            events_path = None
            for candidate in [
                runs_dir / run_id / "out" / "events.jsonl",
                runs_dir / run_id / "events.jsonl",
            ]:
                if candidate.is_file():
                    events_path = candidate
                    break

            if events_path is None:
                continue

            events: list[dict[str, Any]] = []
            with events_path.open() as ef:
                for line in ef:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if not events:
                continue

            try:
                result_row = _extract_per_run(
                    analysis_name,
                    events,
                    run_id,
                    kappa,
                    arm_label,
                    concentration,
                )
                if result_row:
                    all_results.append(result_row)
            except Exception as exc:
                click.echo(f"  {analysis_name} ({run_id}): {exc}")

    if not all_results:
        return None

    analysis_dir = out_dir / "aggregate" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / f"{analysis_name}.csv"

    fieldnames = list(all_results[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    return out_path


def _extract_per_run(
    analysis_name: str,
    events: list[dict[str, Any]],
    run_id: str,
    kappa: str,
    arm: str,
    concentration: str,
) -> dict[str, Any] | None:
    """Extract per-run metrics for a given analysis type."""
    base: dict[str, Any] = {
        "run_id": run_id,
        "kappa": kappa,
        "arm": arm,
        "concentration": concentration,
    }

    if analysis_name == "contagion":
        from bilancio.analysis.contagion import default_counts_by_type, time_to_contagion

        counts = default_counts_by_type(events)
        ttc = time_to_contagion(events)
        return {
            **base,
            "n_primary": counts.get("primary", 0),
            "n_secondary": counts.get("secondary", 0),
            "n_total": counts.get("total", 0),
            "time_to_contagion": ttc if ttc is not None else "",
        }

    if analysis_name == "credit_creation":
        from bilancio.analysis.credit_creation import (
            credit_created_by_type,
            credit_destroyed_by_type,
            net_credit_impulse,
        )

        created = credit_created_by_type(events)
        destroyed = credit_destroyed_by_type(events)
        impulse = net_credit_impulse(events)
        row = dict(base)
        for key, value in created.items():
            row[f"created_{key}"] = str(value)
        for key, value in destroyed.items():
            row[f"destroyed_{key}"] = str(value)
        row["net_impulse"] = str(impulse)
        return row

    if analysis_name == "network":
        from bilancio.analysis.network_analysis import systemic_importance

        rankings = systemic_importance(events)
        if rankings:
            top = rankings[0]
            return {
                **base,
                "top_agent": top.get("agent_id", ""),
                "top_score": str(top.get("score", 0)),
                "n_agents": len(rankings),
            }
        return {**base, "top_agent": "", "top_score": "", "n_agents": 0}

    if analysis_name == "pricing":
        from bilancio.analysis.pricing_analysis import fire_sale_indicator, trade_prices_by_day

        prices = trade_prices_by_day(events)
        fire_sales = fire_sale_indicator(events)
        all_ratios: list[float] = []
        for day_trades in prices.values():
            for trade in day_trades:
                price_ratio = trade.get("price_ratio")
                if price_ratio is not None:
                    all_ratios.append(float(price_ratio))
        avg_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else None
        return {
            **base,
            "avg_price_ratio": str(avg_ratio) if avg_ratio is not None else "",
            "n_fire_sales": len(fire_sales),
            "n_trade_days": len(prices),
        }

    if analysis_name == "beliefs":
        return {
            **base,
            "note": "requires estimate_log (not available from events.jsonl)",
        }

    if analysis_name == "funding":
        from bilancio.analysis.funding_chains import liquidity_providers

        providers = liquidity_providers(events)
        if providers:
            top = providers[0]
            return {
                **base,
                "top_provider": top.get("agent_id", ""),
                "top_net_provision": str(top.get("net_provision", 0)),
                "n_providers": len(providers),
            }
        return {**base, "top_provider": "", "top_net_provision": "", "n_providers": 0}

    return None


def _offer_post_sweep_analysis(
    out_dir: Path,
    sweep_type: str,
    post_analysis: str | None,
    cloud: bool = False,
) -> None:
    """Offer interactive post-sweep analysis menu or run specified analyses."""
    from bilancio.analysis.post_sweep import run_post_sweep_analysis
    from bilancio.ui.sweep_setup import (
        PostSweepAnalysisResult,
        _available_data_analyses,
        _available_visualizations,
        run_post_sweep_questionnaire,
    )

    csv_path = out_dir / "aggregate" / "comparison.csv"
    if not csv_path.is_file():
        if cloud:
            return
        click.echo("  (No comparison.csv found — skipping post-sweep analysis)")
        return

    if post_analysis == "none":
        return

    avail_data = _available_data_analyses(sweep_type)
    avail_viz = _available_visualizations(sweep_type)

    if post_analysis == "all":
        result = PostSweepAnalysisResult(
            data_analyses=list(avail_data.keys()),
            visualizations=list(avail_viz.keys()),
            treynor_kappas=["auto"] if "treynor" in avail_viz else None,
            kappas=None,
        )
    elif post_analysis is not None:
        requested = [item.strip() for item in post_analysis.split(",") if item.strip()]
        invalid = [item for item in requested if item not in VALID_POST_ANALYSES]
        if invalid:
            valid = ", ".join(VALID_POST_ANALYSES)
            click.echo(f"  Warning: unknown analysis {invalid} (valid: {valid})")
        sel_data = [item for item in requested if item in avail_data]
        sel_viz = [item for item in requested if item in avail_viz]
        result = PostSweepAnalysisResult(
            data_analyses=sel_data,
            visualizations=sel_viz,
            treynor_kappas=["auto"] if "treynor" in sel_viz else None,
            kappas=None,
        )
    else:
        result = run_post_sweep_questionnaire(sweep_type)

    if not result.data_analyses and not result.visualizations:
        return

    core_viz = [name for name in result.visualizations if name not in ("treynor", "comparison", "report", "notebook")]
    if core_viz:
        click.echo(f"\nRunning core visualizations: {', '.join(core_viz)}...")
        try:
            core_results = run_post_sweep_analysis(
                experiment_root=out_dir,
                sweep_type=sweep_type,
                analyses=core_viz,
                kappas=result.kappas,
            )
            for name, path in (core_results or {}).items():
                click.echo(f"  {name}: {path}")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            click.echo(f"  Core visualization failed: {exc}")

    for data_name in result.data_analyses:
        click.echo(f"\nRunning {data_name}...")
        try:
            if data_name == "frontier":
                from bilancio.analysis.intermediary_frontier import (
                    build_frontier_artifact,
                    write_frontier_artifact,
                )

                artifact = build_frontier_artifact(out_dir)
                analysis_dir = out_dir / "aggregate" / "analysis"
                analysis_dir.mkdir(parents=True, exist_ok=True)
                paths = write_frontier_artifact(artifact, analysis_dir)
                for name, path in paths.items():
                    click.echo(f"  {name}: {path}")
            elif data_name == "strategy_outcomes":
                from bilancio.analysis.strategy_outcomes import run_strategy_analysis

                by_run_path, overall_path = run_strategy_analysis(out_dir)
                for path in (by_run_path, overall_path):
                    click.echo(f"  {data_name}: {path}")
            elif data_name == "dealer_usage":
                from bilancio.analysis.dealer_usage_summary import run_dealer_usage_analysis

                path = run_dealer_usage_analysis(out_dir)
                click.echo(f"  {data_name}: {path}")
            elif data_name == "mechanism_activity":
                from bilancio.analysis.mechanism_activity import run_mechanism_activity_analysis

                paths = run_mechanism_activity_analysis(out_dir, sweep_type)
                for name, path in paths.items():
                    click.echo(f"  {name}: {path}")
            elif data_name in (
                "contagion",
                "credit_creation",
                "network",
                "pricing",
                "beliefs",
                "funding",
            ):
                analysis_path = _run_per_run_analysis(out_dir, data_name, sweep_type)
                if analysis_path is None:
                    click.echo(f"  {data_name}: no data found")
                else:
                    click.echo(f"  {data_name}: {analysis_path}")
        except ImportError as exc:
            click.echo(f"  {data_name} failed: {optional_extra_message(data_name, 'analysis')} ({exc})")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            click.echo(f"  {data_name} failed: {exc}")

    if result.has_treynor:
        click.echo("\nGenerating Treynor pricing dashboards...")
        try:
            from bilancio.analysis.post_sweep import (
                _auto_detect_kappas,
                _find_run_in_csv,
                _read_csv,
                _run_dir_path,
            )
            from bilancio.analysis.treynor_viz import build_treynor_dashboard

            rows = _read_csv(csv_path)
            if result.treynor_kappas == ["auto"]:
                kappa_values = _auto_detect_kappas(csv_path)
            else:
                kappa_values = [float(kappa) for kappa in (result.treynor_kappas or [])]

            if not kappa_values:
                click.echo("  No kappas found for Treynor generation.")
            else:
                run_id_col = {
                    "dealer": "active_run_id",
                    "bank": "lend_run_id",
                    "nbfi": "lend_run_id",
                }.get(sweep_type, "active_run_id")
                arm_label = {
                    "dealer": "active",
                    "bank": "lend",
                    "nbfi": "lend",
                }.get(sweep_type, "active")

                analysis_dir = out_dir / "aggregate" / "analysis"
                analysis_dir.mkdir(parents=True, exist_ok=True)

                for kappa in kappa_values:
                    run_id = _find_run_in_csv(rows, kappa, run_id_col)
                    if not run_id:
                        click.echo(f"  Treynor (kappa={kappa}): no matching run found")
                        continue
                    runs_dir = _arm_runs_dir(out_dir, sweep_type, arm_label)
                    run_dir = _run_dir_path(runs_dir, run_id)
                    if run_dir is None:
                        click.echo(f"  Treynor (kappa={kappa}): run dir not found for {run_id}")
                        continue

                    treynor_html = build_treynor_dashboard(run_dir)
                    out_path = analysis_dir / f"treynor_k{kappa}_{arm_label}.html"
                    out_path.write_text(treynor_html)
                    click.echo(f"  treynor (kappa={kappa}): {out_path}")
        except ImportError as exc:
            click.echo(f"  Treynor generation failed: {optional_extra_message('Treynor dashboards', 'analysis')} ({exc})")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            click.echo(f"  Treynor generation failed: {exc}")

    if "comparison" in result.visualizations:
        click.echo("\nGenerating comparison report...")
        try:
            from bilancio.analysis.visualization.run_comparison import generate_comparison_html

            analysis_dir = out_dir / "aggregate" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            out_path = analysis_dir / "comparison_report.html"
            generated_path = generate_comparison_html(str(out_dir), output_path=out_path)
            click.echo(f"  comparison: {generated_path}")
        except ImportError as exc:
            click.echo(f"  Comparison report failed: {optional_extra_message('comparison reports', 'analysis')} ({exc})")
        except Exception as exc:
            click.echo(f"  Comparison report failed: {exc}")

    if "report" in result.visualizations:
        click.echo("\nGenerating comprehensive report...")
        try:
            from bilancio.analysis.comprehensive_report import build_comprehensive_report

            analysis_dir = out_dir / "aggregate" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            report_html = build_comprehensive_report(out_dir, sweep_type)
            out_path = analysis_dir / "comprehensive_report.html"
            out_path.write_text(report_html)
            click.echo(f"  report: {out_path}")
        except ImportError as exc:
            click.echo(f"  Report generation failed: {optional_extra_message('comprehensive report', 'analysis')} ({exc})")
        except Exception as exc:
            click.echo(f"  Report generation failed: {exc}")

    if "notebook" in result.visualizations:
        click.echo("\nGenerating presentation notebook...")
        try:
            from bilancio.analysis.notebook_generator import generate_sweep_notebook

            analysis_dir = out_dir / "aggregate" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            nb_path = generate_sweep_notebook(out_dir, sweep_type, analysis_dir)
            click.echo(f"  notebook: {nb_path}")
        except ImportError as exc:
            click.echo(f"  Notebook generation failed: {optional_extra_message('presentation notebook', 'analysis')} ({exc})")
        except Exception as exc:
            click.echo(f"  Notebook generation failed: {exc}")

    analysis_dir = out_dir / "aggregate" / "analysis"
    if analysis_dir.is_dir():
        htmls = list(analysis_dir.glob("*.html"))
        if htmls:
            click.echo(f'\n  Open with: open "{htmls[0]}"')
