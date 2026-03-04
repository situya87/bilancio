"""Re-export for backwards compatibility.

Trade viability checks have moved to ``bilancio.scenarios.viability``.
This shim keeps every existing import path working.
"""

from bilancio.scenarios.viability import (
    InterbankViabilityReport,
    SimulationViabilityReport,
    SweepViabilityReport,
    ViabilityReport,
    check_interbank_viability,
    check_simulation_viability,
    check_sweep_viability,
    check_trade_viability,
)

__all__ = [
    "InterbankViabilityReport",
    "SimulationViabilityReport",
    "SweepViabilityReport",
    "ViabilityReport",
    "check_interbank_viability",
    "check_simulation_viability",
    "check_sweep_viability",
    "check_trade_viability",
]
