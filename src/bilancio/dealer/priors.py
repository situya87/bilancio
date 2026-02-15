"""Shared default-probability priors for VBT pricing and trader risk assessment.

The kappa-informed prior uses the system liquidity ratio to estimate
the probability of default before any payment history is observed.
Both VBT and traders share this prior to prevent adverse selection.
"""

from __future__ import annotations

from decimal import Decimal


def kappa_informed_prior(kappa: Decimal) -> Decimal:
    """Compute a default-probability prior from the system liquidity ratio.

    Formula:
        P_prior = 0.05 + 0.15 * max(0, 1 - kappa) / (1 + kappa)

    At kappa=0 (no cash): P = 0.05 + 0.15 = 0.20
    At kappa=0.5:         P = 0.05 + 0.15 * 0.5/1.5 = 0.10
    At kappa=1 (balanced): P = 0.05
    At kappa>=1:           P = 0.05 (floor)

    Args:
        kappa: System liquidity ratio L0/S1 (must be > 0).

    Returns:
        Estimated default probability in [0.05, 0.20].
    """
    stress = max(Decimal(0), Decimal(1) - kappa) / (Decimal(1) + kappa)
    return Decimal("0.05") + Decimal("0.15") * stress


__all__ = ["kappa_informed_prior"]
