"""Smoke tests to verify basic installation and imports."""

import pytest
from decimal import Decimal


def test_core_imports():
    """Test that core modules can be imported."""
    from bilancio.core.time import TimeCoordinate, TimeInterval, now
    from bilancio.core.errors import (
        BilancioError,
        ValidationError,
        DefaultError,
    )
    from bilancio.core.atomic import AtomicValue, Money, Quantity, Rate
    
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
    from bilancio.domain.instruments.contract import Contract, BaseContract
    from bilancio.domain.instruments.policy import Policy, BasePolicy
    
    # Test that base classes exist
    assert Agent is not None
    assert Contract is not None
    assert BaseContract is not None
    assert Policy is not None
    assert BasePolicy is not None


def test_ops_imports():
    """Test that ops modules can be imported."""
    from bilancio.ops.cashflows import CashFlow, CashFlowStream
    from bilancio.core.atomic import Money
    from bilancio.core.time import TimeCoordinate
    
    # Simple test - we can't create CashFlow without Agent instances
    # but we can test the CashFlowStream
    stream = CashFlowStream()
    assert len(stream) == 0
    assert stream.get_all_flows() == []


def test_engines_imports():
    """Test that engine modules can be imported."""
    from bilancio.engines.simulation import SimulationEngine, MonteCarloEngine
    from bilancio.core.atomic import Money

    # Test basic instantiation
    monte_carlo = MonteCarloEngine(n_simulations=1000)
    assert monte_carlo.n_simulations == 1000


def test_analysis_imports():
    """Test that analysis modules can be imported."""
    from bilancio.analysis.metrics import calculate_npv, calculate_irr
    
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
    from bilancio import TimeCoordinate, TimeInterval, now
    from bilancio import BilancioError, ValidationError
    
    assert TimeCoordinate is not None
    assert BilancioError is not None