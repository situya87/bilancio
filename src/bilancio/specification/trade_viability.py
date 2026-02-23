"""Re-export for backwards compatibility.

Trade viability checks have moved to ``bilancio.scenarios.viability``.
This shim keeps every existing import path working.
"""

from bilancio.scenarios.viability import ViabilityReport, check_trade_viability

__all__ = ["ViabilityReport", "check_trade_viability"]
