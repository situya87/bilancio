"""Run-level contract tests for bilancio.analysis.post_sweep.

Test intent:
- Keep run aggregation outputs stable for downstream dashboards.
- Verify analysis helper failures degrade to empty structures instead of
  interrupting report generation.
- Pin treatment-delta report sections that compare defaults, credit, funding,
  network structure, and loss metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from bilancio.analysis.post_sweep import (
    _analyse_run,
    _resolve_sweep_paths,
    _run_treatment_deltas,
)


class TestRunAnalysisContract:
    def test_analyse_run_aggregates_core_and_loss_metrics(self, tmp_path: Path):
        """_analyse_run should normalize analysis helper outputs into dashboard-ready data."""
        run_dir = tmp_path / "run_a"
        out_dir = run_dir / "out"
        out_dir.mkdir(parents=True)
        (out_dir / "dealer_metrics.json").write_text(json.dumps({"dealer_pnl": -7}))
        (run_dir / "scenario.yaml").write_text("version: 1\nname: fixture\n")
        events = [{"kind": "DealerTrade", "day": 1}, {"kind": "PayableDefaulted", "day": 2}]

        with (
            patch("bilancio.analysis.default_counts_by_type", return_value={"primary": 1, "secondary": 1, "total": 2}),
            patch("bilancio.analysis.contagion_by_day", return_value={2: 1}),
            patch("bilancio.analysis.credit_created_by_type", return_value={"bank": 120}),
            patch("bilancio.analysis.credit_destroyed_by_type", return_value={"payable": 30}),
            patch("bilancio.analysis.net_credit_impulse", return_value=90),
            patch(
                "bilancio.analysis.cash_inflows_by_source",
                return_value={"A": {"cash": 5}, "B": {"cash": 2, "bank": 3}},
            ),
            patch("bilancio.analysis.trade_prices_by_day", return_value={1: [{"side": "buy", "price_ratio": 0.91}]}),
            patch("bilancio.analysis.trade_volume_by_day", return_value={1: 4}),
            patch("bilancio.analysis.bid_ask_spread_by_day", return_value={1: 0.04}),
            patch("bilancio.analysis.node_degree", return_value={"A": {"out_degree": 2}}),
            patch(
                "bilancio.analysis.systemic_importance",
                return_value=[{"agent_id": "A", "total_obligations": 10, "betweenness": 0.5, "score": 0.7}],
            ),
            patch(
                "bilancio.analysis.report.compute_run_level_metrics",
                return_value={
                    "payable_default_loss": 11,
                    "deposit_loss_gross": 4,
                    "nbfi_loan_loss": 2,
                    "bank_credit_loss": 3,
                    "cb_backstop_loss": 5,
                },
            ),
            patch(
                "bilancio.analysis.report.compute_intermediary_losses",
                return_value={"dealer_vbt_loss": 7, "intermediary_loss_total": 17},
            ),
            patch("bilancio.analysis.report.extract_initial_capitals", return_value={"dealer": 100}),
        ):
            result = _analyse_run(events, "dealer", is_treatment=True, run_dir=run_dir)

        assert result["n_events"] == 2
        assert result["default_counts"] == {"primary": 1, "secondary": 1, "total": 2}
        assert result["credit_created"] == {"bank": 120.0}
        assert result["credit_destroyed"] == {"payable": 30.0}
        assert result["net_credit_impulse"] == 90.0
        assert result["funding_mix"] == {"cash": 7.0, "bank": 3.0}
        assert result["trade_prices_by_day"] == {1: [{"side": "buy", "price_ratio": 0.91}]}
        assert result["trade_volume_by_day"] == {1: 4}
        assert result["bid_ask_spread_by_day"] == {1: 0.04}
        assert result["node_degrees"] == {"A": {"out_degree": 2}}
        assert result["systemic_importance"] == [{"agent_id": "A", "total_obligations": 10.0, "betweenness": 0.5, "score": 0.7}]
        assert result["loss_metrics"] == {
            "payable_default_loss": 11,
            "deposit_loss_gross": 4,
            "total_loss": 15,
            "nbfi_loan_loss": 2,
            "bank_credit_loss": 3,
            "cb_backstop_loss": 5,
            "dealer_vbt_loss": 7,
            "intermediary_loss_total": 17,
            "system_loss": 32,
        }
        assert result["intermediary_losses"] == {"dealer_vbt_loss": 7, "intermediary_loss_total": 17}
        assert result["initial_capitals"] == {"dealer": 100}

    def test_analyse_run_degrades_to_empty_structures_on_helper_errors(self):
        events = [{"kind": "bad"}]
        with (
            patch("bilancio.analysis.default_counts_by_type", side_effect=KeyError("boom")),
            patch("bilancio.analysis.contagion_by_day", side_effect=ValueError("boom")),
            patch("bilancio.analysis.credit_created_by_type", side_effect=TypeError("boom")),
            patch("bilancio.analysis.credit_destroyed_by_type", side_effect=TypeError("boom")),
            patch("bilancio.analysis.net_credit_impulse", side_effect=ValueError("boom")),
            patch("bilancio.analysis.cash_inflows_by_source", side_effect=KeyError("boom")),
            patch("bilancio.analysis.node_degree", side_effect=ValueError("boom")),
            patch("bilancio.analysis.systemic_importance", side_effect=TypeError("boom")),
        ):
            result = _analyse_run(events, "nbfi", is_treatment=False)

        assert result["default_counts"] == {"primary": 0, "secondary": 0, "total": 0}
        assert result["contagion_by_day"] == {}
        assert result["credit_created"] == {}
        assert result["credit_destroyed"] == {}
        assert result["net_credit_impulse"] == 0.0
        assert result["funding_mix"] == {}
        assert result["node_degrees"] == {}
        assert result["systemic_importance"] == []
        assert result["trade_prices_by_day"] == {}
        assert result["loss_metrics"] == {}


class TestTreatmentDeltaGoldenOutput:
    def test_treatment_delta_dashboard_includes_loss_sections(self, tmp_path: Path):
        agg = tmp_path / "aggregate"
        agg.mkdir()
        (agg / "comparison.csv").write_text(
            "kappa,concentration,mu,outside_mid_ratio,seed,"
            "delta_passive,delta_active,passive_run_id,active_run_id,"
            "system_loss_pct_passive,system_loss_pct_active,"
            "total_loss_pct_passive,total_loss_pct_active,"
            "intermediary_loss_pct_passive,intermediary_loss_pct_active,"
            "loss_capital_ratio_passive,loss_capital_ratio_active,"
            "system_loss_trading_effect\n"
            "0.5,1,0.5,0.9,42,0.30,0.10,passive_a,active_a,"
            "0.40,0.20,0.35,0.12,0.05,0.08,0.20,0.32,0.20\n"
        )
        for arm, run_id in (("active", "active_a"), ("passive", "passive_a")):
            out_dir = tmp_path / arm / "runs" / run_id / "out"
            out_dir.mkdir(parents=True)
            (out_dir / "events.jsonl").write_text('{"kind": "PhaseA", "day": 0}\n')

        out_dir = tmp_path / "analysis"
        out_dir.mkdir()
        treatment_result = {
            "default_counts": {"primary": 1, "secondary": 0, "total": 1},
            "net_credit_impulse": 120,
            "funding_mix": {"bank": 30},
            "node_degrees": {"A": {"out_degree": 2}},
        }
        baseline_result = {
            "default_counts": {"primary": 2, "secondary": 1, "total": 3},
            "net_credit_impulse": 20,
            "funding_mix": {"bank": 10, "cash": 5},
            "node_degrees": {"A": {"out_degree": 1}},
        }

        with patch(
            "bilancio.analysis.post_sweep._analyse_run",
            side_effect=[treatment_result, baseline_result],
        ):
            output = _run_treatment_deltas(_resolve_sweep_paths(tmp_path, "dealer"), [0.5], out_dir)

        html = output.read_text()
        assert "Dealer Treatment Deltas" in html
        assert "System Loss Comparison" in html
        assert "Loss Attribution" in html
        assert "Delta-Based vs Loss-Based Treatment Effect" in html
        assert "loss_capital" in html
        assert "Loss/capital ratio shows" in html
        assert "Treatment minus baseline comparison across" in html
        assert "<td>0.5</td><td>-2</td><td>-1</td><td>-1</td>" in html
