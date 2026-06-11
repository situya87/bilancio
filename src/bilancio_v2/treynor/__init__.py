"""Treynor dealer framework: faithful formalizations of the three chapters.

- :mod:`kernel` — the shared corridor mathematics (Chapter 1's model,
  generalized to the asymmetric two-sided corridors and split boundary
  tolls of Chapters 2 and 3).
- :mod:`spot` — the Chapter 1 spot dealer.
- :mod:`repo` — the Chapter 2 repo dealer (three books).
- :mod:`bank` — the Chapter 3 bank-as-dealer (stacked Treynors).

Design stance: the 50/50 zero-information prior is the *dealer's internal
model*, baked into pricing via the stationary-distribution arithmetic. The
ring's actual flow is whatever the simulation generates; the framework
instruments the gap (realized direction frequencies, layoff rates vs λ,
realized P&L vs the zero-profit prediction) rather than forcing flow to
match the prior.
"""

from bilancio_v2.treynor.kernel import (
    Orientation,
    TreynorCorridor,
    WalkStats,
    simulate_walk,
)

__all__ = ["Orientation", "TreynorCorridor", "WalkStats", "simulate_walk"]
