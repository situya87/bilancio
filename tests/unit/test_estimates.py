"""Tests for the Estimate dataclass."""

import pytest
from decimal import Decimal

from bilancio.information.estimates import Estimate


class TestEstimateCreation:
    """Test basic Estimate creation and field access."""

    def test_minimal_creation(self):
        est = Estimate(
            value=Decimal("0.15"),
            estimator_id="dealer_1",
            target_id="firm_5",
            target_type="agent",
            estimation_day=10,
            method="bayesian_default_prob",
        )
        assert est.value == Decimal("0.15")
        assert est.estimator_id == "dealer_1"
        assert est.target_id == "firm_5"
        assert est.target_type == "agent"
        assert est.estimation_day == 10
        assert est.method == "bayesian_default_prob"
        assert est.inputs == {}
        assert est.metadata == {}

    def test_full_provenance(self):
        est = Estimate(
            value=Decimal("0.85"),
            estimator_id="system",
            target_id="ticket_42",
            target_type="instrument",
            estimation_day=5,
            method="ev_hold",
            inputs={"face_value": "100", "p_default": "0.15"},
            metadata={"issuer_id": "firm_3", "maturity_day": 8},
        )
        assert est.value == Decimal("0.85")
        assert est.inputs["face_value"] == "100"
        assert est.metadata["issuer_id"] == "firm_3"

    def test_system_target_type(self):
        est = Estimate(
            value=Decimal("0.05"),
            estimator_id="observer",
            target_id="global",
            target_type="system",
            estimation_day=1,
            method="aggregate_default_rate",
        )
        assert est.target_type == "system"


class TestEstimateImmutability:
    """Test that Estimate is frozen (immutable)."""

    def test_frozen_value(self):
        est = Estimate(
            value=Decimal("0.10"),
            estimator_id="a",
            target_id="b",
            target_type="agent",
            estimation_day=1,
            method="test",
        )
        with pytest.raises(AttributeError):
            est.value = Decimal("0.99")

    def test_frozen_method(self):
        est = Estimate(
            value=Decimal("0.10"),
            estimator_id="a",
            target_id="b",
            target_type="agent",
            estimation_day=1,
            method="test",
        )
        with pytest.raises(AttributeError):
            est.method = "other"


class TestEstimateValidation:
    """Test target_type validation."""

    def test_invalid_target_type(self):
        with pytest.raises(ValueError, match="target_type must be one of"):
            Estimate(
                value=Decimal("0.10"),
                estimator_id="a",
                target_id="b",
                target_type="invalid",
                estimation_day=1,
                method="test",
            )

    @pytest.mark.parametrize("target_type", ["agent", "instrument", "system"])
    def test_all_valid_target_types(self, target_type):
        est = Estimate(
            value=Decimal("0.10"),
            estimator_id="a",
            target_id="b",
            target_type=target_type,
            estimation_day=1,
            method="test",
        )
        assert est.target_type == target_type
