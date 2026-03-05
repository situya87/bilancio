"""Smoke tests to verify basic installation and imports."""

from decimal import Decimal

import pytest


def test_core_imports():
    """Test that core modules can be imported."""
    from bilancio.core.atomic import Money, Quantity, Rate
    from bilancio.core.time import TimeCoordinate, TimeInterval, now

    # Basic instantiation tests
    t = TimeCoordinate(0.0)
    assert t.t == 0.0

    interval = TimeInterval(TimeCoordinate(0.0), TimeCoordinate(1.0))
    assert interval.start.t == 0.0
    assert interval.end.t == 1.0

    current_time = now()
    assert current_time.t == 0.0

    # Test Money
    money = Money(Decimal("100.50"), "USD")
    assert money.amount == Decimal("100.50")
    assert money.currency == "USD"
    assert money.value == Decimal("100.50")  # AtomicValue protocol

    # Test Quantity
    qty = Quantity(10.5, "kg")
    assert qty.value == 10.5
    assert qty.unit == "kg"

    # Test Rate
    rate = Rate(Decimal("0.05"), "annual")
    assert rate.value == Decimal("0.05")
    assert rate.basis == "annual"


def test_domain_imports():
    """Test that domain modules can be imported."""
    from bilancio.domain.agent import Agent
    from bilancio.domain.instruments.contract import BaseContract
    from bilancio.domain.instruments.policy import BasePolicy

    # Test that base classes are properly defined
    assert hasattr(Agent, "__init__")
    assert hasattr(BaseContract, "id") and hasattr(BaseContract, "parties")
    assert hasattr(BasePolicy, "evaluate")


def test_ops_imports():
    """Test that ops modules can be imported."""
    from bilancio.ops.cashflows import CashFlowStream

    # Simple test - we can't create CashFlow without Agent instances
    # but we can test the CashFlowStream
    stream = CashFlowStream()
    assert len(stream) == 0
    assert stream.get_all_flows() == []


def test_engines_imports():
    """Test that engine modules can be imported."""
    from bilancio.engines.simulation import MonteCarloEngine

    # Test basic instantiation
    monte_carlo = MonteCarloEngine(n_simulations=1000)
    assert monte_carlo.n_simulations == 1000


def test_analysis_imports():
    """Test that analysis modules can be imported."""
    from bilancio.analysis.metrics import calculate_irr, calculate_npv

    # These are placeholders so they should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        calculate_npv([], 0.05)

    with pytest.raises(NotImplementedError):
        calculate_irr([])


def test_package_metadata():
    """Test that package metadata is accessible."""
    import bilancio

    assert bilancio.__version__ == "0.1.0"

    # Test that basic imports from __init__ work
    from bilancio import BilancioError, TimeCoordinate

    assert TimeCoordinate is not None
    assert BilancioError is not None
