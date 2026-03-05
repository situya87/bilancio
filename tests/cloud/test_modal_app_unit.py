"""Unit tests for pure functions in modal_app.py.

Tests _compute_dealer_vbt_loss, compute_metrics_from_events,
save_run_to_supabase, compute_aggregate_metrics, and health_check
without requiring the Modal SDK installed.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the `modal` module so we can import modal_app without the SDK.
#
# The @app.function() decorator is a MagicMock, so decorated functions
# (compute_aggregate_metrics, health_check, run_simulation) become
# MagicMocks themselves.  We capture the raw functions passed to the
# mock decorator and expose them for direct testing.
# ---------------------------------------------------------------------------

_mock_modal = MagicMock()

# Capture raw functions passed to @app.function()(fn)
_captured_decorated_fns: dict[str, object] = {}


def _capture_decorator(*args, **kwargs):
    """Return a fake decorator that records the wrapped function."""

    def _inner(fn):
        _captured_decorated_fns[fn.__name__] = fn
        return fn  # Return the real function, not a MagicMock

    return _inner


_mock_modal.App.return_value.function = _capture_decorator
_mock_modal.App.return_value.local_entrypoint = _capture_decorator
_mock_modal.Volume.from_name.return_value = MagicMock()

# Make the Image builder chain work (each method returns self)
_mock_image = MagicMock()
_mock_image.debian_slim.return_value = _mock_image
_mock_image.pip_install.return_value = _mock_image
_mock_image.add_local_python_source.return_value = _mock_image
_mock_modal.Image = _mock_image
_mock_modal.Secret = MagicMock()

# Patch before importing modal_app
sys.modules.setdefault("modal", _mock_modal)

from bilancio.cloud.modal_app import (  # noqa: E402
    SupabaseCredentialsError,
    _compute_dealer_vbt_loss,
    compute_metrics_from_events,
    save_run_to_supabase,
)

# Retrieve the raw decorated functions
_raw_compute_aggregate_metrics = _captured_decorated_fns["compute_aggregate_metrics"]
_raw_health_check = _captured_decorated_fns["health_check"]


# ---------------------------------------------------------------------------
# 1. _compute_dealer_vbt_loss  (6 tests)
# ---------------------------------------------------------------------------


class TestComputeDealerVbtLoss:
    """Tests for _compute_dealer_vbt_loss."""

    def test_none_path_returns_zero(self) -> None:
        assert _compute_dealer_vbt_loss(None) == 0.0

    def test_nonexistent_file_returns_zero(self, tmp_path) -> None:
        missing = str(tmp_path / "no_such_file.json")
        assert _compute_dealer_vbt_loss(missing) == 0.0

    def test_negative_pnl_returns_positive_loss(self, tmp_path) -> None:
        p = tmp_path / "dealer_metrics.json"
        p.write_text(json.dumps({"dealer_total_pnl": -100.5}))
        assert _compute_dealer_vbt_loss(str(p)) == pytest.approx(100.5)

    def test_positive_pnl_returns_zero(self, tmp_path) -> None:
        p = tmp_path / "dealer_metrics.json"
        p.write_text(json.dumps({"dealer_total_pnl": 50.0}))
        assert _compute_dealer_vbt_loss(str(p)) == 0.0

    def test_malformed_json_returns_zero(self, tmp_path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("NOT VALID JSON {{{}}")
        assert _compute_dealer_vbt_loss(str(p)) == 0.0

    def test_missing_key_returns_zero(self, tmp_path) -> None:
        p = tmp_path / "dealer_metrics.json"
        p.write_text(json.dumps({"other_field": 42}))
        assert _compute_dealer_vbt_loss(str(p)) == 0.0


# ---------------------------------------------------------------------------
# 2. compute_metrics_from_events  (6 tests)
# ---------------------------------------------------------------------------


def _write_events(path, events: list[dict]) -> None:
    """Write a list of event dicts as JSONL."""
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _minimal_settlement_events() -> list[dict]:
    """Minimal event set: one payable A->B due day 1, fully settled."""
    return [
        {"kind": "CashMinted", "day": 0, "phase": "setup", "to": "A", "amount": 100, "instr_id": "C1"},
        {"kind": "CashMinted", "day": 0, "phase": "setup", "to": "B", "amount": 100, "instr_id": "C2"},
        {
            "kind": "PayableCreated",
            "day": 0,
            "phase": "setup",
            "debtor": "A",
            "creditor": "B",
            "amount": 50,
            "due_day": 1,
            "maturity_distance": 1,
            "payable_id": "PAY_001",
            "alias": "P_A_B",
        },
        {"kind": "PhaseA", "day": 1, "phase": "simulation"},
        {"kind": "PhaseB", "day": 1, "phase": "simulation"},
        {"kind": "SubphaseB1", "day": 1, "phase": "simulation"},
        {"kind": "SubphaseB2", "day": 1, "phase": "simulation"},
        {
            "kind": "CashTransferred",
            "day": 1,
            "phase": "simulation",
            "frm": "A",
            "to": "B",
            "amount": 50,
            "instr_id": "C3",
        },
        {
            "kind": "PayableSettled",
            "day": 1,
            "phase": "simulation",
            "debtor": "A",
            "creditor": "B",
            "amount": 50,
            "pid": "PAY_001",
            "alias": "P_A_B",
        },
        {"kind": "PhaseC", "day": 1, "phase": "simulation"},
    ]


class TestComputeMetricsFromEvents:
    """Tests for compute_metrics_from_events."""

    def test_empty_file_returns_none_values(self, tmp_path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text("")
        result = compute_metrics_from_events(str(p))
        assert result["delta_total"] is None
        assert result["phi_total"] is None
        assert result["time_to_stability"] is None
        assert result["max_G_t"] is None
        assert result["raw_metrics"] == {}

    def test_settlement_events_produce_metrics(self, tmp_path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, _minimal_settlement_events())
        result = compute_metrics_from_events(str(p))
        assert result["delta_total"] is not None
        assert result["phi_total"] is not None
        assert result["phi_total"] == pytest.approx(1.0)
        assert result["delta_total"] == pytest.approx(0.0)

    def test_dealer_metrics_path_included(self, tmp_path) -> None:
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _minimal_settlement_events())
        dealer_path = tmp_path / "dealer_metrics.json"
        dealer_path.write_text(json.dumps({"dealer_total_pnl": -25.0}))

        result = compute_metrics_from_events(str(events_path), dealer_metrics_path=str(dealer_path))
        assert result["dealer_vbt_loss"] == pytest.approx(25.0)
        assert result["intermediary_loss_total"] >= 25.0

    def test_result_contains_expected_keys(self, tmp_path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, _minimal_settlement_events())
        result = compute_metrics_from_events(str(p))
        expected_keys = {
            "delta_total", "phi_total", "time_to_stability", "max_G_t",
            "alpha_1", "Mpeak_1", "v_1", "HHIplus_1",
            "n_defaults", "cascade_fraction",
            "cb_loans_created_count", "cb_interest_total_paid",
            "cb_loans_outstanding_pre_final", "bank_defaults_final",
            "cb_reserves_initial", "cb_reserves_final", "cb_reserve_destruction_pct",
            "delta_bank", "deposit_loss_gross", "deposit_loss_pct",
            "total_deposits_created", "bank_obligations_created", "bank_writeoffs",
            "payable_default_loss", "total_loss", "total_loss_pct", "S_total",
            "nbfi_loan_loss", "bank_credit_loss", "cb_backstop_loss",
            "dealer_vbt_loss", "intermediary_loss_total", "raw_metrics",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_decimal_values_converted_to_float(self, tmp_path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, _minimal_settlement_events())
        result = compute_metrics_from_events(str(p))

        for key in ("delta_total", "phi_total"):
            val = result[key]
            if val is not None:
                assert isinstance(val, float), f"{key} should be float, got {type(val)}"

        for key, val in result["raw_metrics"].items():
            assert not isinstance(val, Decimal), f"raw_metrics[{key}] is still Decimal"

    def test_missing_dealer_metrics_returns_zero_loss(self, tmp_path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, _minimal_settlement_events())
        result = compute_metrics_from_events(str(p), dealer_metrics_path=None)
        assert result["dealer_vbt_loss"] == 0.0


# ---------------------------------------------------------------------------
# 3. save_run_to_supabase  (6 tests)
# ---------------------------------------------------------------------------


def _make_supabase_client_mock() -> MagicMock:
    """Build a mock Supabase client with proper method chains."""
    client = MagicMock()
    tbl = MagicMock()
    tbl.upsert.return_value.execute.return_value = MagicMock()
    tbl.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    tbl.insert.return_value.execute.return_value = MagicMock()
    tbl.update.return_value.eq.return_value.execute.return_value = MagicMock()
    client.table.return_value = tbl
    return client


class TestSaveRunToSupabase:
    """Tests for save_run_to_supabase (Supabase mocked)."""

    @pytest.fixture()
    def _supabase_env(self, monkeypatch):
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://fake.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "fake-key")

    @pytest.mark.usefixtures("_supabase_env")
    def test_successful_save(self) -> None:
        mock_client = _make_supabase_client_mock()
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            result = save_run_to_supabase(
                run_id="test_001",
                job_id="job_001",
                status="completed",
                metrics={"delta_total": 0.1, "phi_total": 0.9, "raw_metrics": {}},
                params={"kappa": 0.5, "concentration": 1.0},
                execution_time_ms=5000,
                modal_call_id="mc-123",
                modal_volume_path="exp/runs/test_001",
            )

        assert result is True
        # Both runs upsert and metrics insert should have been called
        assert mock_client.table.call_count >= 2

    def test_missing_credentials_returns_false(self, monkeypatch) -> None:
        monkeypatch.delenv("BILANCIO_SUPABASE_URL", raising=False)
        monkeypatch.delenv("BILANCIO_SUPABASE_ANON_KEY", raising=False)

        mock_sb = MagicMock()
        with patch.dict(sys.modules, {"supabase": mock_sb}):
            result = save_run_to_supabase(
                run_id="test_002",
                job_id="job_001",
                status="completed",
                metrics={"delta_total": 0.1, "phi_total": 0.9},
                params={},
                execution_time_ms=1000,
                modal_call_id=None,
                modal_volume_path="path",
            )
        assert result is False

    @pytest.mark.usefixtures("_supabase_env")
    def test_parameter_mapping(self) -> None:
        mock_client = _make_supabase_client_mock()
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            save_run_to_supabase(
                run_id="test_003",
                job_id="job_001",
                status="completed",
                metrics={"delta_total": None, "phi_total": None},
                params={"kappa": 0.5, "concentration": 1.0, "mu": 0.0, "seed": 42, "regime": "passive"},
                execution_time_ms=2000,
                modal_call_id="mc-456",
                modal_volume_path="exp/runs/test_003",
            )

        upsert = mock_client.table.return_value.upsert
        assert upsert.called
        row = upsert.call_args[0][0]
        assert row["run_id"] == "test_003"
        assert row["job_id"] == "job_001"
        assert row["status"] == "completed"
        assert row["kappa"] == 0.5
        assert row["concentration"] == 1.0
        assert row["mu"] == 0.0
        assert row["seed"] == 42
        assert row["regime"] == "passive"

    @pytest.mark.usefixtures("_supabase_env")
    def test_metrics_mapping(self) -> None:
        mock_client = _make_supabase_client_mock()
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            save_run_to_supabase(
                run_id="test_004",
                job_id="job_001",
                status="completed",
                metrics={
                    "delta_total": 0.2,
                    "phi_total": 0.8,
                    "time_to_stability": 5,
                    "max_G_t": 100.0,
                    "alpha_1": 0.5,
                    "Mpeak_1": 200.0,
                    "v_1": 1.5,
                    "HHIplus_1": 0.3,
                    "raw_metrics": {},
                },
                params={},
                execution_time_ms=3000,
                modal_call_id=None,
                modal_volume_path="path",
            )

        insert = mock_client.table.return_value.insert
        assert insert.called
        metrics_row = insert.call_args[0][0]
        assert metrics_row["run_id"] == "test_004"
        assert metrics_row["job_id"] == "job_001"
        assert metrics_row["delta_total"] == 0.2
        assert metrics_row["phi_total"] == 0.8
        assert metrics_row["time_to_stability"] == 5

    @pytest.mark.usefixtures("_supabase_env")
    def test_none_metrics_skips_metrics_table(self) -> None:
        mock_client = _make_supabase_client_mock()
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            mock_client.reset_mock()
            save_run_to_supabase(
                run_id="test_005",
                job_id="job_001",
                status="failed",
                metrics={"delta_total": None, "phi_total": None},
                params={},
                execution_time_ms=100,
                modal_call_id=None,
                modal_volume_path="path",
            )

        table_names = [c[0][0] for c in mock_client.table.call_args_list]
        assert "runs" in table_names
        assert "metrics" not in table_names

    @pytest.mark.usefixtures("_supabase_env")
    def test_connection_error_returns_false(self) -> None:
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(side_effect=ConnectionError("Connection refused"))

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            result = save_run_to_supabase(
                run_id="test_006",
                job_id="job_001",
                status="completed",
                metrics={"delta_total": 0.1, "phi_total": 0.9},
                params={},
                execution_time_ms=1000,
                modal_call_id=None,
                modal_volume_path="path",
            )
        assert result is False


# ---------------------------------------------------------------------------
# 4. compute_aggregate_metrics  (5 tests)
#
# The raw function was captured from the mock decorator into
# _raw_compute_aggregate_metrics.
# ---------------------------------------------------------------------------


def _make_aggregate_client(data_rows):
    """Build a mock Supabase client pre-loaded with *data_rows*."""
    client = MagicMock()
    # Primary data fetch: client.table("runs").select(...).eq(...).in_(...).execute()
    primary = MagicMock()
    primary.data = data_rows
    client.table.return_value.select.return_value.eq.return_value.in_.return_value.execute.return_value = primary
    # job_metrics select (no existing row)
    client.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    client.table.return_value.insert.return_value.execute.return_value = MagicMock()
    client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    return client


class TestComputeAggregateMetrics:
    """Tests for compute_aggregate_metrics (raw function, Supabase mocked)."""

    def test_raises_without_credentials(self, monkeypatch) -> None:
        monkeypatch.delenv("BILANCIO_SUPABASE_URL", raising=False)
        monkeypatch.delenv("BILANCIO_SUPABASE_ANON_KEY", raising=False)

        mock_sb = MagicMock()
        with patch.dict(sys.modules, {"supabase": mock_sb}):
            with pytest.raises(SupabaseCredentialsError):
                _raw_compute_aggregate_metrics(job_id="test-job", run_ids=["r1"])

    def test_groups_by_parameter_tuple(self, monkeypatch) -> None:
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://fake.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "fake-key")

        rows = [
            {
                "run_id": "rp", "kappa": 0.5, "concentration": 1.0,
                "mu": 0.0, "outside_mid_ratio": 0.9, "seed": 42,
                "regime": "passive",
                "metrics": [{"delta_total": 0.3, "phi_total": 0.7}],
            },
            {
                "run_id": "ra", "kappa": 0.5, "concentration": 1.0,
                "mu": 0.0, "outside_mid_ratio": 0.9, "seed": 42,
                "regime": "active",
                "metrics": [{"delta_total": 0.1, "phi_total": 0.9}],
            },
        ]
        client = _make_aggregate_client(rows)
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            result = _raw_compute_aggregate_metrics(job_id="test-job", run_ids=["rp", "ra"])

        assert result["status"] == "completed"
        assert len(result["comparisons"]) == 1

    def test_trading_effect_computation(self, monkeypatch) -> None:
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://fake.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "fake-key")

        rows = [
            {
                "run_id": "p1", "kappa": 1.0, "concentration": 1.0,
                "mu": 0.0, "outside_mid_ratio": 0.9, "seed": 42,
                "regime": "passive",
                "metrics": [{"delta_total": 0.4, "phi_total": 0.6}],
            },
            {
                "run_id": "a1", "kappa": 1.0, "concentration": 1.0,
                "mu": 0.0, "outside_mid_ratio": 0.9, "seed": 42,
                "regime": "active",
                "metrics": [{"delta_total": 0.15, "phi_total": 0.85}],
            },
        ]
        client = _make_aggregate_client(rows)
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            result = _raw_compute_aggregate_metrics(job_id="job-te", run_ids=["p1", "a1"])

        comp = result["comparisons"][0]
        assert comp["trading_effect"] == pytest.approx(0.25)
        assert comp["delta_passive"] == 0.4
        assert comp["delta_active"] == 0.15

    def test_empty_runs_no_crash(self, monkeypatch) -> None:
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://fake.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "fake-key")

        client = _make_aggregate_client([])
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            result = _raw_compute_aggregate_metrics(job_id="empty-job", run_ids=[])

        assert result["status"] == "error"
        assert "No runs found" in result["error"]

    def test_upserts_to_job_metrics_table(self, monkeypatch) -> None:
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://fake.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "fake-key")

        rows = [
            {
                "run_id": "p2", "kappa": 0.5, "concentration": 1.0,
                "mu": 0.0, "outside_mid_ratio": 0.9, "seed": 7,
                "regime": "passive",
                "metrics": [{"delta_total": 0.2, "phi_total": 0.8}],
            },
            {
                "run_id": "a2", "kappa": 0.5, "concentration": 1.0,
                "mu": 0.0, "outside_mid_ratio": 0.9, "seed": 7,
                "regime": "active",
                "metrics": [{"delta_total": 0.05, "phi_total": 0.95}],
            },
        ]
        client = _make_aggregate_client(rows)
        mock_sb = MagicMock()
        mock_sb.create_client = MagicMock(return_value=client)

        with patch.dict(sys.modules, {"supabase": mock_sb}):
            _raw_compute_aggregate_metrics(job_id="jm-job", run_ids=["p2", "a2"])

        table_names = [c[0][0] for c in client.table.call_args_list]
        assert "job_metrics" in table_names


# ---------------------------------------------------------------------------
# 5. health_check  (2 tests)
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health_check (raw function captured from mock decorator)."""

    def test_successful_import(self) -> None:
        result = _raw_health_check()
        assert result["status"] == "ok"
        assert result["bilancio_available"] is True
        assert "version" in result

    def test_import_failure_returns_error(self) -> None:
        # Temporarily break the bilancio import so the function's
        # except ImportError path is exercised.
        saved = sys.modules.get("bilancio")
        sys.modules["bilancio"] = None  # type: ignore[assignment]
        try:
            result = _raw_health_check()
        finally:
            if saved is not None:
                sys.modules["bilancio"] = saved
            else:
                del sys.modules["bilancio"]

        assert result["status"] == "error"
        assert result["bilancio_available"] is False
        assert "error" in result
