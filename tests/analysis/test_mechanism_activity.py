"""Tests for bilancio.analysis.mechanism_activity."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from bilancio.analysis.mechanism_activity import (
    _activity_metric_cols,
    _build_activity_summary,
    _generate_findings,
    _parse_bank_events,
    _parse_nbfi_events,
    _safe_float,
    _safe_mean,
    run_mechanism_activity_analysis,
)


# ---------------------------------------------------------------------------
# 1. _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float(3.14) == 3.14

    def test_valid_int(self):
        assert _safe_float(7) == 7.0

    def test_string_number(self):
        assert _safe_float("2.5") == 2.5

    def test_none_returns_zero(self):
        assert _safe_float(None) == 0.0

    def test_invalid_string_returns_zero(self):
        assert _safe_float("not-a-number") == 0.0

    def test_empty_string_returns_zero(self):
        assert _safe_float("") == 0.0


# ---------------------------------------------------------------------------
# 2. _safe_mean
# ---------------------------------------------------------------------------


class TestSafeMean:
    def test_valid_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = _safe_mean(df, "x")
        assert result == pytest.approx(2.0)

    def test_missing_column_returns_none(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        assert _safe_mean(df, "y") is None

    def test_all_nan_returns_none(self):
        df = pd.DataFrame({"x": [float("nan"), float("nan")]})
        assert _safe_mean(df, "x") is None

    def test_mixed_nan_and_values(self):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        result = _safe_mean(df, "x")
        assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3. _activity_metric_cols
# ---------------------------------------------------------------------------


class TestActivityMetricCols:
    def test_dealer_cols(self):
        cols = _activity_metric_cols("dealer")
        assert "dealer_trade_count" in cols
        assert "total_face_traded" in cols
        assert "total_cash_volume" in cols
        assert len(cols) == 11

    def test_bank_cols(self):
        cols = _activity_metric_cols("bank")
        assert "n_loans" in cols
        assert "total_lent" in cols
        assert "loan_default_rate" in cols
        assert "lgd" in cols
        assert len(cols) == 11

    def test_nbfi_cols(self):
        cols = _activity_metric_cols("nbfi")
        assert "n_loans" in cols
        assert "approval_rate" in cols
        assert "n_rejected" in cols
        assert "frac_rejected_defaulted" in cols
        assert len(cols) == 13

    def test_unknown_returns_empty(self):
        assert _activity_metric_cols("unknown") == []
        assert _activity_metric_cols("") == []


# ---------------------------------------------------------------------------
# 4-5. _parse_bank_events
# ---------------------------------------------------------------------------


class TestParseBankEvents:
    def test_empty_file(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("")
        metrics = _parse_bank_events(events_path)

        assert metrics["n_loans"] == 0
        assert metrics["total_lent"] == 0
        assert metrics["n_repaid"] == 0
        assert metrics["n_defaulted"] == 0
        assert metrics["loan_default_rate"] == 0
        assert metrics["first_loan_day"] is None
        assert metrics["last_loan_day"] is None
        assert metrics["cb_freeze_day"] is None
        assert metrics["n_payables_settled"] == 0
        assert metrics["n_payables_rolled"] == 0

    def test_bank_loan_events(self, tmp_path):
        events = [
            {
                "kind": "BankLoanIssued",
                "day": 1,
                "amount": 100.0,
                "rate": 0.05,
                "borrower": "firm_0",
                "bank": "bank_0",
            },
            {
                "kind": "BankLoanIssued",
                "day": 3,
                "amount": 200.0,
                "rate": 0.08,
                "borrower": "firm_1",
                "bank": "bank_0",
            },
            {
                "kind": "BankLoanRepaid",
                "day": 4,
                "borrower": "firm_0",
            },
            {
                "kind": "BankLoanDefault",
                "day": 5,
                "borrower": "firm_1",
                "principal": 200.0,
                "recovered": 50.0,
            },
            {
                "kind": "CBLendingFreezeActivated",
                "day": 6,
                "cutoff_day": 6,
            },
            {"kind": "PayableSettled", "day": 2},
            {"kind": "PayableSettled", "day": 3},
            {"kind": "PayableRolledOver", "day": 4},
        ]
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("\n".join(json.dumps(e) for e in events))

        metrics = _parse_bank_events(events_path)

        assert metrics["n_loans"] == 2
        assert metrics["total_lent"] == pytest.approx(300.0)
        assert metrics["unique_borrowers"] == 2
        assert metrics["unique_banks"] == 1
        assert metrics["avg_loan_size"] == pytest.approx(150.0)
        assert metrics["avg_rate"] == pytest.approx(0.065)
        assert metrics["min_rate"] == pytest.approx(0.05)
        assert metrics["max_rate"] == pytest.approx(0.08)
        assert metrics["first_loan_day"] == 1
        assert metrics["last_loan_day"] == 3
        assert metrics["n_repaid"] == 1
        assert metrics["n_defaulted"] == 1
        assert metrics["loan_default_rate"] == pytest.approx(0.5)
        # lgd = (200 - 50) / 200 = 0.75
        assert metrics["lgd"] == pytest.approx(0.75)
        # 1 borrower defaulted out of 2
        assert metrics["frac_borrowers_defaulted"] == pytest.approx(0.5)
        assert metrics["cb_freeze_day"] == 6
        assert metrics["n_payables_settled"] == 2
        assert metrics["n_payables_rolled"] == 1

    def test_malformed_json_lines_skipped(self, tmp_path):
        """Malformed JSON lines should be silently skipped."""
        events_path = tmp_path / "events.jsonl"
        content = (
            '{"kind": "BankLoanIssued", "day": 1, "amount": 50, "rate": 0.03, "borrower": "f0", "bank": "b0"}\n'
            "this is not valid json\n"
            '{"kind": "BankLoanRepaid", "day": 2}\n'
        )
        events_path.write_text(content)
        metrics = _parse_bank_events(events_path)
        assert metrics["n_loans"] == 1
        assert metrics["n_repaid"] == 1


# ---------------------------------------------------------------------------
# 6-7. _parse_nbfi_events
# ---------------------------------------------------------------------------


class TestParseNbfiEvents:
    def test_empty_file(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("")
        metrics = _parse_nbfi_events(events_path)

        assert metrics["n_loans"] == 0
        assert metrics["n_rejected"] == 0
        assert metrics["approval_rate"] == 0
        assert metrics["total_lent"] == 0
        assert metrics["unique_borrowers"] == 0
        assert metrics["first_loan_day"] is None
        assert metrics["last_loan_day"] is None
        assert metrics["avg_rejected_coverage"] is None
        assert metrics["n_loan_defaults"] == 0
        assert metrics["n_agent_defaults"] == 0
        assert metrics["n_payables_settled"] == 0
        assert metrics["n_payables_rolled"] == 0

    def test_nbfi_loan_events(self, tmp_path):
        events = [
            {
                "kind": "NonBankLoanCreated",
                "day": 1,
                "amount": 80.0,
                "rate": 0.07,
                "borrower_id": "firm_0",
            },
            {
                "kind": "NonBankLoanCreatedPreventive",
                "day": 2,
                "amount": 60.0,
                "rate": 0.09,
                "borrower_id": "firm_1",
            },
            {
                "kind": "NonBankLoanRepaid",
                "day": 4,
                "borrower_id": "firm_0",
            },
            {
                "kind": "NonBankLoanDefaulted",
                "day": 5,
                "borrower_id": "firm_1",
            },
            {
                "kind": "NonBankLoanRejectedCoverage",
                "day": 2,
                "borrower_id": "firm_2",
                "coverage": 1.5,
            },
            {
                "kind": "NonBankLoanRejectedCoverage",
                "day": 3,
                "borrower_id": "firm_3",
                "coverage": 2.0,
            },
            {
                "kind": "AgentDefaulted",
                "day": 6,
                "agent": "firm_2",
            },
            {"kind": "PayableSettled", "day": 2},
            {"kind": "PayableRolledOver", "day": 3},
            {"kind": "PayableRolledOver", "day": 4},
        ]
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("\n".join(json.dumps(e) for e in events))

        metrics = _parse_nbfi_events(events_path)

        assert metrics["n_loans"] == 2
        assert metrics["n_rejected"] == 2
        # approval = 2 / (2 + 2) = 0.5
        assert metrics["approval_rate"] == pytest.approx(0.5)
        assert metrics["total_lent"] == pytest.approx(140.0)
        assert metrics["unique_borrowers"] == 2
        assert metrics["avg_loan_size"] == pytest.approx(70.0)
        assert metrics["avg_rate"] == pytest.approx(0.08)
        assert metrics["first_loan_day"] == 1
        assert metrics["last_loan_day"] == 2
        assert metrics["avg_rejected_coverage"] == pytest.approx(1.75)
        assert metrics["n_repaid"] == 1
        assert metrics["n_loan_defaults"] == 1
        assert metrics["loan_default_rate"] == pytest.approx(0.5)
        # firm_1 defaulted on loan, 1 of 2 borrowers
        assert metrics["frac_borrowers_defaulted"] == pytest.approx(0.5)
        # firm_2 was rejected and then defaulted as agent; firm_3 rejected but didn't default
        # 1 out of 2 rejected borrowers defaulted
        assert metrics["frac_rejected_defaulted"] == pytest.approx(0.5)
        assert metrics["n_agent_defaults"] == 1
        assert metrics["n_payables_settled"] == 1
        assert metrics["n_payables_rolled"] == 2


# ---------------------------------------------------------------------------
# 8-9. _build_activity_summary
# ---------------------------------------------------------------------------


class TestBuildActivitySummary:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        summary = _build_activity_summary(df, "dealer", "trading_effect")
        assert summary["sweep_type"] == "dealer"
        assert summary["n_runs"] == 0
        # No kappa breakdown or correlations for empty DF
        assert "by_kappa" not in summary

    def test_sample_data(self):
        df = pd.DataFrame(
            {
                "kappa": [0.5, 0.5, 1.0, 1.0, 2.0, 2.0],
                "mu": [0, 0, 0, 0, 0, 0],
                "bank_lending_effect": [0.05, 0.03, -0.01, 0.0, 0.02, 0.01],
                "n_loans": [10, 8, 5, 3, 7, 6],
                "total_lent": [500, 400, 200, 100, 300, 250],
                "loan_default_rate": [0.1, 0.2, 0.0, 0.1, 0.05, 0.0],
            },
        )
        summary = _build_activity_summary(df, "bank", "bank_lending_effect")

        assert summary["n_runs"] == 6
        assert summary["sweep_type"] == "bank"

        # by_kappa breakdown
        assert "by_kappa" in summary
        assert "0.5" in summary["by_kappa"]
        assert summary["by_kappa"]["0.5"]["n_runs"] == 2
        assert summary["by_kappa"]["0.5"]["mean_effect"] == pytest.approx(0.04)
        assert "mean_n_loans" in summary["by_kappa"]["0.5"]

        # by_kappa_mu breakdown
        assert "by_kappa_mu" in summary
        assert "k=0.5_mu=0" in summary["by_kappa_mu"]

        # correlations
        assert "correlations_with_effect" in summary
        # With 6 data points we should get correlations
        assert "n_loans" in summary["correlations_with_effect"]

        # effectiveness
        assert "effectiveness" in summary
        # 0.05, 0.03, 0.02, 0.01 > 0.001 -> 4 helps; -0.01 < -0.001 -> 1 hurts; 0.0 neutral -> 1
        assert summary["effectiveness"]["helps"] == 4
        assert summary["effectiveness"]["hurts"] == 1
        assert summary["effectiveness"]["neutral"] == 1

        # key_findings
        assert "key_findings" in summary
        assert len(summary["key_findings"]) > 0


# ---------------------------------------------------------------------------
# 10-12. _generate_findings
# ---------------------------------------------------------------------------


class TestGenerateFindings:
    def test_dealer_findings(self):
        df = pd.DataFrame(
            {
                "trading_effect": [0.05, 0.03, 0.01],
                "dealer_trade_count": [10, 0, 5],
                "dealer_active_fraction": [0.8, 0.0, 0.5],
            },
        )
        findings = _generate_findings(df, "dealer", "trading_effect")
        assert any("Mean trading_effect" in f for f in findings)
        assert any("Mean dealer trades per run" in f for f in findings)
        # 1 run with zero trades
        assert any("1/3 runs had zero dealer trades" in f for f in findings)
        assert any("Mean dealer active fraction" in f for f in findings)

    def test_bank_findings(self):
        df = pd.DataFrame(
            {
                "bank_lending_effect": [0.02, 0.04, 0.01],
                "n_loans": [5, 0, 8],
                "loan_default_rate": [0.1, 0.0, 0.2],
                "total_lent": [200, 0, 400],
            },
        )
        findings = _generate_findings(df, "bank", "bank_lending_effect")
        assert any("Mean bank_lending_effect" in f for f in findings)
        assert any("Mean loans per run" in f for f in findings)
        # 1 run with zero bank loans
        assert any("1/3 runs had zero bank loans" in f for f in findings)
        assert any("Mean loan default rate" in f for f in findings)
        assert any("Mean total lent per run" in f for f in findings)

    def test_nbfi_findings(self):
        df = pd.DataFrame(
            {
                "lending_effect": [0.01, 0.02, 0.03],
                "approval_rate": [0.6, 0.7, 0.8],
                "n_rejected": [4, 3, 2],
                "frac_rejected_defaulted": [0.5, 0.33, 0.25],
                "n_loans": [6, 7, 8],
            },
        )
        findings = _generate_findings(df, "nbfi", "lending_effect")
        assert any("Mean lending_effect" in f for f in findings)
        assert any("Mean approval rate" in f for f in findings)
        assert any("Mean rejections per run" in f for f in findings)
        assert any("rejected borrowers that defaulted" in f for f in findings)
        assert any("Mean loans per run" in f for f in findings)

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame()
        findings = _generate_findings(df, "dealer", "trading_effect")
        assert findings == []

    def test_effect_direction_increases(self):
        """Negative mean effect should report 'increases' defaults."""
        df = pd.DataFrame({"lending_effect": [-0.05, -0.03, -0.01]})
        findings = _generate_findings(df, "nbfi", "lending_effect")
        assert any("increases" in f for f in findings)


# ---------------------------------------------------------------------------
# 13-14. run_mechanism_activity_analysis
# ---------------------------------------------------------------------------


class TestRunMechanismActivityAnalysis:
    def test_invalid_sweep_type_raises(self, tmp_path):
        # Create a fake comparison.csv so we get past the file-existence check
        agg_dir = tmp_path / "aggregate"
        agg_dir.mkdir()
        (agg_dir / "comparison.csv").write_text("kappa,delta\n0.5,0.1\n")

        with pytest.raises(ValueError, match="Unknown sweep_type"):
            run_mechanism_activity_analysis(tmp_path, "invalid_type")

    def test_missing_comparison_csv_returns_empty(self, tmp_path):
        # experiment_root exists but has no aggregate/comparison.csv
        result = run_mechanism_activity_analysis(tmp_path, "bank")
        assert result == {}


# ---------------------------------------------------------------------------
# 15. _analyze_bank_activity with mock comparison.csv and events
# ---------------------------------------------------------------------------


class TestAnalyzeBankActivityIntegration:
    def test_with_mock_data(self, tmp_path):
        """End-to-end test: comparison.csv + events.jsonl -> activity DataFrame."""
        from bilancio.analysis.mechanism_activity import _analyze_bank_activity

        # Set up directory structure: experiment_root / bank_lend / runs / run_001 / out / events.jsonl
        run_id = "run_001"
        run_dir = tmp_path / "bank_lend" / "runs" / run_id / "out"
        run_dir.mkdir(parents=True)

        events = [
            {
                "kind": "BankLoanIssued",
                "day": 1,
                "amount": 150.0,
                "rate": 0.06,
                "borrower": "firm_0",
                "bank": "bank_0",
            },
            {
                "kind": "BankLoanIssued",
                "day": 2,
                "amount": 250.0,
                "rate": 0.04,
                "borrower": "firm_1",
                "bank": "bank_0",
            },
            {
                "kind": "BankLoanRepaid",
                "day": 3,
                "borrower": "firm_0",
            },
            {"kind": "PayableSettled", "day": 2},
        ]
        (run_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events)
        )

        # Build a comparison DataFrame matching what the bank analyzer expects
        comp_df = pd.DataFrame(
            {
                "lend_run_id": [run_id],
                "kappa": [0.5],
                "concentration": [1.0],
                "mu": [0.0],
                "outside_mid_ratio": [0.9],
                "seed": [42],
                "delta_idle": [0.2],
                "delta_lend": [0.1],
                "bank_lending_effect": [0.1],
                "phi_idle": [0.8],
                "phi_lend": [0.9],
                "n_defaults_idle": [5],
                "n_defaults_lend": [2],
            }
        )

        result_df = _analyze_bank_activity(tmp_path, comp_df)

        assert len(result_df) == 1
        row = result_df.iloc[0]
        assert row["run_id"] == run_id
        assert row["kappa"] == 0.5
        assert row["bank_lending_effect"] == pytest.approx(0.1)
        assert row["n_loans"] == 2
        assert row["total_lent"] == pytest.approx(400.0)
        assert row["unique_borrowers"] == 2
        assert row["unique_banks"] == 1
        assert row["avg_rate"] == pytest.approx(0.05)
        assert row["n_repaid"] == 1
        assert row["n_defaulted"] == 0
        assert row["n_payables_settled"] == 1

    def test_missing_events_skips_run(self, tmp_path):
        """Runs with missing events.jsonl are silently skipped."""
        from bilancio.analysis.mechanism_activity import _analyze_bank_activity

        comp_df = pd.DataFrame(
            {
                "lend_run_id": ["nonexistent_run"],
                "kappa": [0.5],
                "bank_lending_effect": [0.1],
            }
        )
        result_df = _analyze_bank_activity(tmp_path, comp_df)
        assert result_df.empty

    def test_no_lend_run_id_skips(self, tmp_path):
        """Rows without lend_run_id are skipped."""
        from bilancio.analysis.mechanism_activity import _analyze_bank_activity

        comp_df = pd.DataFrame(
            {
                "lend_run_id": [""],
                "kappa": [0.5],
            }
        )
        result_df = _analyze_bank_activity(tmp_path, comp_df)
        assert result_df.empty
